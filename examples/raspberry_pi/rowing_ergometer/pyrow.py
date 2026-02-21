#!/usr/bin/env python
#Copyright (c) 2011 Sam Gambrell, 2016-2017 Michael Droogleever
#Licensed under the Simplified BSD License.

# NOTE: This code has not been thoroughly tested and may not function as advertised.
# Please report and findings to the author so that they may be addressed in a stable release.

# pylint: disable=C0103,R0912,R0913

"""
pyrow.py
Interface to concept2 indoor rower

Uses the Linux kernel HID driver via /dev/hidraw for proper multi-packet
USB HID report handling.  This enables report ID #2 (121-byte frames,
120 bytes CSAFE payload) instead of report ID #1 (21-byte frames, 20 bytes
payload).  The kernel HID layer automatically splits/reassembles reports
across multiple 64-byte USB transactions.

Previous versions used pyusb which detaches the kernel driver and accesses
raw USB endpoints, limiting each transfer to wMaxPacketSize (64 bytes) and
breaking multi-packet HID reports.
"""

import datetime
import os
import fcntl
import time
import sys
import threading

try:
    from .csafe import csafe_cmd  # Relative import for package use
except (ImportError, ValueError):
    try:
        from csafe import csafe_cmd  # Direct import for script use
    except ImportError:
        csafe_cmd = None

C2_VENDOR_ID = 0x17a4
C2_PRODUCT_ID = 0x000a
MIN_FRAME_GAP = .050  # 50ms — CSAFE spec minimum; kernel HID driver is reliable
INTERFACE = 0

# HID report sizes (from PM5 HID descriptor):
#   Report ID 1:   20 data bytes + 1 ID byte =  21 bytes
#   Report ID 2:  120 data bytes + 1 ID byte = 121 bytes  <-- default
#   Report ID 4:  500 data bytes + 1 ID byte = 501 bytes  (firmware doesn't respond)
REPORT_ID = 2
REPORT_DATA_SIZE = 120  # CSAFE payload bytes for report ID #2
REPORT_TOTAL_SIZE = REPORT_DATA_SIZE + 1  # including report ID byte

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
                    pass
    return data_dict


def _find_hidraw_for_c2():
    """
    Scan /sys/class/hidraw/ to find the hidraw device node for a Concept2
    ergometer.  Returns a list of /dev/hidrawN paths.
    """
    results = []
    sysfs_base = '/sys/class/hidraw'
    if not os.path.isdir(sysfs_base):
        return results
    for entry in sorted(os.listdir(sysfs_base)):
        uevent_path = os.path.join(sysfs_base, entry, 'device', 'uevent')
        if not os.path.exists(uevent_path):
            continue
        with open(uevent_path) as fp:
            content = fp.read()
        # HID_ID line looks like: HID_ID=0003:000017A4:0000000A
        for line in content.splitlines():
            if line.startswith('HID_ID='):
                parts = line.split('=', 1)[1].split(':')
                if len(parts) >= 3:
                    vid = int(parts[1], 16)
                    pid = int(parts[2], 16)
                    if vid == C2_VENDOR_ID:
                        devpath = f'/dev/{entry}'
                        # Create the device node if it doesn't exist (Docker)
                        if not os.path.exists(devpath):
                            dev_file = os.path.join(sysfs_base, entry, 'dev')
                            if os.path.exists(dev_file):
                                with open(dev_file) as df:
                                    major, minor = df.read().strip().split(':')
                                try:
                                    os.mknod(devpath, 0o666 | 0o020000,
                                             os.makedev(int(major), int(minor)))
                                except (OSError, PermissionError):
                                    continue
                        results.append(devpath)
    return results


def _reattach_kernel_driver():
    """
    If the kernel HID driver was previously detached (by pyusb), reattach it
    so that /dev/hidrawN becomes available.  Requires pyusb to be installed.
    """
    try:
        import usb.core
        import usb.util
        dev = usb.core.find(idVendor=C2_VENDOR_ID)
        if dev is None:
            return
        try:
            usb.util.release_interface(dev, INTERFACE)
        except Exception:
            pass
        try:
            if not dev.is_kernel_driver_active(INTERFACE):
                dev.attach_kernel_driver(INTERFACE)
        except Exception:
            pass
        # Give the kernel time to create the hidraw node
        time.sleep(0.5)
    except ImportError:
        pass  # pyusb not installed — kernel driver should already be attached


def find():
    """
    Returns list of hidraw device paths for connected Concept2 ergometers.
    """
    paths = _find_hidraw_for_c2()
    if not paths:
        # Kernel driver may have been detached by a previous pyusb session
        _reattach_kernel_driver()
        paths = _find_hidraw_for_c2()
    if not paths:
        raise ValueError('Ergs not found — no /dev/hidraw device for Concept2. '
                         'Is the PM5 connected via USB?')
    return paths


class PyErg(object):
    """
    Manages low-level erg communication via Linux hidraw.

    Uses the kernel HID driver which properly handles multi-packet USB
    transfers, enabling report ID #2 (120 bytes CSAFE payload per frame)
    instead of the 20-byte report ID #1.
    """
    def __init__(self, hidraw_path):
        """
        Opens the hidraw device for read/write.

        Parameters
        ----------
        hidraw_path : str
            Path to /dev/hidrawN device, as returned by find().
        """
        self._path = hidraw_path
        self._fd = os.open(hidraw_path, os.O_RDWR)

        # Flush stale data from a previous session
        # Set non-blocking temporarily
        flags = fcntl.fcntl(self._fd, fcntl.F_GETFL)
        fcntl.fcntl(self._fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        for _ in range(32):
            try:
                os.read(self._fd, 512)
            except BlockingIOError:
                break
        # Restore blocking mode
        fcntl.fcntl(self._fd, fcntl.F_SETFL, flags)

        self.__lastsend = datetime.datetime.now()
        self.__send_lock = threading.Lock()

    def close(self):
        """Close the hidraw file descriptor."""
        if hasattr(self, '_fd') and self._fd >= 0:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = -1

    def __del__(self):
        self.close()

    @staticmethod
    def _checkvalue(*args, **kwargs):
        return checkvalue(*args, **kwargs)

    def _write_report(self, csafe_frame):
        """
        Wrap a CSAFE frame in a HID report and write to the device.
        The frame must NOT include the report ID — this method adds it.
        """
        # Pad to full report size
        data = csafe_frame + [0] * (REPORT_DATA_SIZE - len(csafe_frame))
        buf = bytes([REPORT_ID]) + bytes(data)
        os.write(self._fd, buf)

    def _read_report(self, timeout_ms=2000):
        """
        Read one HID report from the device.

        Returns the full report bytes including the report ID as byte[0].
        On Linux hidraw with multi-report-ID devices, the kernel includes
        the report ID in reads.
        """
        import select
        # Use select for timeout (os.read on hidraw blocks forever)
        r, _, _ = select.select([self._fd], [], [], timeout_ms / 1000.0)
        if not r:
            raise TimeoutError(f"PM5 read timeout ({timeout_ms}ms)")
        data = os.read(self._fd, 512)
        if not data:
            raise ConnectionError("PM5 hidraw: empty read")
        return data

    def send(self, message):
        """
        Send CSAFE message to PM5 and return parsed response dict.
        Thread-safe with MIN_FRAME_GAP timing per CSAFE spec.
        """
        with self.__send_lock:
            now = datetime.datetime.now()
            delta = (now - self.__lastsend).total_seconds()
            if delta < MIN_FRAME_GAP:
                time.sleep(MIN_FRAME_GAP - delta)

            csafe_frame = csafe_cmd.write(message)
            try:
                self._write_report(csafe_frame)
            except OSError as e:
                raise ConnectionError(f"PM5 write error: {e}")

            response = []
            while not response:
                try:
                    raw = self._read_report(timeout_ms=2000)
                    # raw[0] is the report ID; CSAFE data starts at raw[1:]
                    transmission = list(raw)
                    response = csafe_cmd.read(transmission)
                except TimeoutError as e:
                    raise ConnectionError(str(e))
                except OSError as e:
                    raise ConnectionError(f"PM5 read error: {e}")

            self.__lastsend = datetime.datetime.now()
            return response

    def send_raw(self, message):
        """Like send() but returns (parsed_response, raw_bytes) tuple."""
        with self.__send_lock:
            now = datetime.datetime.now()
            delta = (now - self.__lastsend).total_seconds()
            if delta < MIN_FRAME_GAP:
                time.sleep(MIN_FRAME_GAP - delta)

            csafe_frame = csafe_cmd.write(message)
            try:
                self._write_report(csafe_frame)
            except OSError as e:
                raise ConnectionError(f"PM5 write error: {e}")

            try:
                raw = self._read_report(timeout_ms=2000)
                transmission = list(raw)
                import warnings
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")
                    response = csafe_cmd.read(transmission)
            except TimeoutError as e:
                raise ConnectionError(str(e))
            except OSError as e:
                raise ConnectionError(f"PM5 read error: {e}")

            self.__lastsend = datetime.datetime.now()
            return response, bytes(raw)