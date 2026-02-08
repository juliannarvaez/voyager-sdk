#!/usr/bin/env python
#Copyright (c) 2011 Sam Gambrell, 2016-2017 Michael Droogleever
#Licensed under the Simplified BSD License.

# NOTE: This code has not been thoroughly tested and may not function as advertised.
# Please report and findings to the author so that they may be addressed in a stable release.

# pylint: disable=C0103,R0912,R0913

"""
pyrow.py
Interface to concept2 indoor rower
"""

import datetime
import time
import sys
import threading

import usb.core
import usb.util
from usb import USBError

try:
    from .csafe import csafe_cmd  # Relative import for package use
except (ImportError, ValueError):
    try:
        from csafe import csafe_cmd  # Direct import for script use
    except ImportError:
        # Fallback for when csafe module isn't needed
        csafe_cmd = None

C2_VENDOR_ID = 0x17a4
MIN_FRAME_GAP = .100 # 100ms for PM5 firmware 459+ (spec is 50ms minimum)
INTERFACE = 0

ERG_MAPPING = {
    # List of stroke states
    'strokestate': [
        'Wait for min speed',
        'Wait for acceleration',
        'Drive',
        'Dwelling',
        'Recovery',
    ],
    # List of workout types
    'workouttype': [
        'Just Row / no splits',
        'Just Row / splits',
        'Fixed Distance / splits',
        'Fixed Distance / no splits',
        'Fixed Time / no splits',
        'Fixed Time Interval',
        'Fixed Distance Interval',
        'Variable Interval',
    ],
    # List of workout state
    'workoutstate': [
        'Waiting begin',
        'Workout row',
        'Countdown pause',
        'Interval rest',
        'Work time inverval',
        'Work distance interval',
        'Rest end time',
        'Rest end distance',
        'Time end rest',
        'Distance end rest',
        'Workout end',
        'Workout terminate',
        'Workout logged',
        'Workout rearm'
    ],
    # List of workout types
    'inttype': [
        'Time',
        'Distance',
        'Rest',
    ],
    # List of display types
    'displaytype': [
        'Standard',
        'Force/Velocity',
        'Paceboat',
        'Per Stroke',
        'Simple',
        'Target',
    ],
    # List of display units types
    'displayunitstype': [
        'Time/Meters',
        'Pace',
        'Watts',
        'Calories',
    ],
    # List of machine states
    'status': [
        'Error',
        'Ready',
        'Idle',
        'Have ID',
        'N/A',
        'In Use',
        'Pause',
        'Finished',
        'Manual',
        'Offline'
    ]
}

def checkvalue(value, label, minimum, maximum):
    """
    Checks that value is an integer and within the specified range
    """
    if not isinstance(value, int):
        raise TypeError(label)
    if  not minimum <= value <= maximum:
        raise ValueError(label + " outside of range")
    return True

def get_pretty(data_dict, pretty):
    """
    Makes data_dict values pretty
    """
    if pretty:
        for key in data_dict.keys():
            if key in ERG_MAPPING:
                try:
                    data_dict[key] = ERG_MAPPING[key][data_dict[key]]
                except IndexError:
                    # TODO, find exceptions and patch into ERG_MAPPING,found:
                    # inttype 255
                    pass
                    # print("IndexError")
    return data_dict

def find():
    """
    Returns list of pyusb Devices which are ergs.
    """
    try:
        ergs = usb.core.find(find_all=True, idVendor=C2_VENDOR_ID)
    except USBError as e:
        # Errno 16: Resource busy - device exists but is in use
        # Errno 19: No such device - device disconnected
        # Don't raise on errno 16, just return empty (device will reconnect)
        if e.errno == 16:
            return []  # Return empty list, device busy
        raise ConnectionRefusedError(f"USB error (errno {e.errno}): {e}")
    if ergs is None:
        raise ValueError('Ergs not found')
    return ergs

def find_all():
    """
    Scans all usb devices and lists everything found to stdout
    :return: nothing
    """
    dev = usb.core.find(find_all=True)
    for cfg in dev:
        sys.stdout.write(
            'VendorID = 0x{:04X}'.format(cfg.idVendor) + ' :: ProductID = 0x{:04X}'.format(cfg.idProduct) + '\n')

class PyErg(object):
    """
    Manages low-level erg communication
    """
    def __init__(self, erg):
        """
        Configures usb connection and sets erg value
        """
        from warnings import warn

        if sys.platform != 'win32':
            try:
                if erg.is_kernel_driver_active(INTERFACE):
                    erg.detach_kernel_driver(INTERFACE)
            except Exception:
                pass

        # Release interface if already claimed (fixes "Resource busy" errors)
        for attempt in range(5):
            try:
                usb.util.release_interface(erg, INTERFACE)
                time.sleep(0.2 * (attempt + 1))
            except:
                pass
        time.sleep(0.5)
        
        # Claim interface
        try:
            usb.util.claim_interface(erg, INTERFACE)
        except USBError as e:
            if e.errno == 16:  # Resource busy - retry after detaching driver
                try:
                    if sys.platform != 'win32' and erg.is_kernel_driver_active(INTERFACE):
                        erg.detach_kernel_driver(INTERFACE)
                except:
                    pass
                time.sleep(1.0)
                usb.util.claim_interface(erg, INTERFACE)
            else:
                raise

        # Set configuration only if needed
        try:
            current_config = erg.get_active_configuration()
            if current_config is None or current_config.bConfigurationValue != 1:
                erg.set_configuration()
        except USBError as e:
            if e.errno != 16:  # Ignore "Resource busy" - already configured
                from warnings import warn
                warn(f"USB error setting configuration: {e}")

        self.erg = erg

        configuration = erg[0]
        iface = configuration[(0, 0)]
        self.inEndpoint = iface[0].bEndpointAddress
        self.outEndpoint = iface[1].bEndpointAddress

        self.__lastsend = datetime.datetime.now()
        self.__send_lock = threading.Lock()  # Thread-safe USB access for PM5 firmware 459+

    def close(self):
        """Release USB interface properly to avoid 'Resource busy' on reconnect"""
        try:
            if hasattr(self, 'erg'):
                usb.util.release_interface(self.erg, INTERFACE)
        except Exception:
            pass  # Already released or device gone

    def __del__(self):
        """Cleanup on deletion"""
        self.close()

    @staticmethod
    def _checkvalue(*args, **kwargs):
        return checkvalue(*args, **kwargs)

    def send(self, message):
        """
        Send CSAFE message to PM5 and return response.
        Thread-safe with MIN_FRAME_GAP timing per CSAFE spec.
        """
        with self.__send_lock:
            # Enforce MIN_FRAME_GAP between consecutive sends
            now = datetime.datetime.now()
            delta = (now - self.__lastsend).total_seconds()
            if delta < MIN_FRAME_GAP:
                time.sleep(MIN_FRAME_GAP - delta)

            # Send message
            csafe = csafe_cmd.write(message)
            try:
                self.erg.write(self.outEndpoint, csafe, timeout=2000)
            except USBError as e:
                if e.errno in (19, 110):  # No device / timeout
                    raise ConnectionError(f"PM5 USB error ({e.errno}): disconnected or cable issue")
                elif e.errno == 16:  # Resource busy
                    raise ConnectionError(f"PM5 USB error ({e.errno}): device busy")
                else:
                    raise ConnectionError(f"PM5 USB error ({e.errno}): {str(e)}")

            # Receive response
            response = []
            while not response:
                try:
                    transmission = self.erg.read(self.inEndpoint, 64, timeout=2000)
                    response = csafe_cmd.read(transmission)
                except USBError as e:
                    if e.errno in (19, 110, 16):
                        raise ConnectionError(f"PM5 USB error ({e.errno}): {str(e)}")
                    elif e.errno == 75:
                        raise ConnectionError(f"PM5 USB error ({e.errno}): buffer overflow")
                    else:
                        raise ConnectionError(f"PM5 USB error ({e.errno}): {str(e)}")

            self.__lastsend = datetime.datetime.now()
            return response
