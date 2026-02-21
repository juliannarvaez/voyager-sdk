#!/usr/bin/env python3
"""
probe_report_ids.py
Probe a Concept2 PM5 to discover which HID report IDs are actually
functional, their declared sizes (from the HID report descriptor),
and the endpoint's wMaxPacketSize.

This answers the question: "Which report IDs can I actually use?"
independently of what the spec claims.

Usage:
    python probe_report_ids.py
"""

import sys
import time
import struct

import usb.core
import usb.util

C2_VENDOR_ID = 0x17a4
INTERFACE = 0

# HID class-specific requests
HID_GET_REPORT_DESC = 0x22  # descriptor type for HID Report Descriptor
USB_REQ_GET_DESCRIPTOR = 0x06


def find_pm5():
    dev = usb.core.find(idVendor=C2_VENDOR_ID)
    if dev is None:
        print("ERROR: No Concept2 PM found.")
        sys.exit(1)
    print(f"Found: {dev.manufacturer} {dev.product}")
    print(f"  VID:PID = {dev.idVendor:#06x}:{dev.idProduct:#06x}")
    print(f"  bcdUSB = {dev.bcdUSB:#06x}  bcdDevice = {dev.bcdDevice:#06x}")
    return dev


def setup_device(dev):
    """Detach kernel driver, claim interface, return (ep_in, ep_out)."""
    if sys.platform != 'win32':
        try:
            if dev.is_kernel_driver_active(INTERFACE):
                dev.detach_kernel_driver(INTERFACE)
                print("  Detached kernel driver.")
        except Exception:
            pass

    try:
        usb.util.release_interface(dev, INTERFACE)
    except Exception:
        pass
    time.sleep(0.3)
    usb.util.claim_interface(dev, INTERFACE)

    cfg = dev[0]
    iface = cfg[(0, 0)]

    ep_in = usb.util.find_descriptor(
        iface,
        custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_IN
    )
    ep_out = usb.util.find_descriptor(
        iface,
        custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_OUT
    )
    return ep_in, ep_out


def dump_endpoint_info(ep_in, ep_out):
    """Print physical endpoint properties."""
    print("\n═══ USB Endpoint Descriptors ═══")
    for label, ep in [("IN", ep_in), ("OUT", ep_out)]:
        xfer = {0: "CONTROL", 1: "ISOCHRONOUS", 2: "BULK", 3: "INTERRUPT"}
        print(f"  EP {ep.bEndpointAddress:#04x} ({label}):")
        print(f"    bmAttributes:   {ep.bmAttributes:#04x} "
              f"({xfer.get(usb.util.endpoint_type(ep.bmAttributes), '?')})")
        print(f"    wMaxPacketSize: {ep.wMaxPacketSize} bytes")
        print(f"    bInterval:      {ep.bInterval} ms")
    print(f"\n  → Physical limit: {ep_in.wMaxPacketSize} bytes per IN transfer")
    print(f"  → Physical limit: {ep_out.wMaxPacketSize} bytes per OUT transfer")


def read_hid_report_descriptor(dev):
    """Read and parse the HID Report Descriptor via control transfer.

    This tells us what report IDs and sizes the PM5 *declares* it supports
    (which may differ from what its firmware actually implements).
    """
    print("\n═══ HID Report Descriptor (declared report IDs) ═══")

    # GET_DESCRIPTOR request: wValue = (descriptor_type << 8) | descriptor_index
    # For HID Report Descriptor: type=0x22, index=0
    # wIndex = interface number
    try:
        desc = dev.ctrl_transfer(
            bmRequestType=0x81,  # Device-to-host, Standard, Interface
            bRequest=USB_REQ_GET_DESCRIPTOR,
            wValue=(HID_GET_REPORT_DESC << 8) | 0,
            wIndex=INTERFACE,
            data_or_wLength=4096,  # request up to 4K
            timeout=2000
        )
    except Exception as e:
        print(f"  Failed to read HID report descriptor: {e}")
        print("  (This is expected if the kernel HID driver was not bound)")
        return {}

    raw = bytes(desc)
    print(f"  Raw descriptor: {len(raw)} bytes")
    print(f"  Hex: {raw.hex(' ')}")

    # Parse HID report descriptor items to find Report ID + Report Count + Report Size
    report_ids = {}
    current_report_id = None
    current_report_size = None
    current_report_count = None
    i = 0
    while i < len(raw):
        prefix = raw[i]
        # Short items: size encoded in bits 0-1 of prefix
        bSize = prefix & 0x03
        if bSize == 3:
            bSize = 4  # size=3 means 4 bytes
        bType = (prefix >> 2) & 0x03   # 0=Main, 1=Global, 2=Local
        bTag = (prefix >> 4) & 0x0F

        if i + 1 + bSize > len(raw):
            break
        data = raw[i+1:i+1+bSize]

        value = 0
        if bSize == 1:
            value = data[0]
        elif bSize == 2:
            value = struct.unpack('<H', data)[0]
        elif bSize == 4:
            value = struct.unpack('<I', data)[0]

        item_name = ""
        # Global items
        if bType == 1:
            if bTag == 8:   # Report ID
                current_report_id = value
                item_name = f"Report ID = {value} ({value:#04x})"
            elif bTag == 7:  # Report Size (bits per field)
                current_report_size = value
                item_name = f"Report Size = {value} bits"
            elif bTag == 9:  # Report Count (number of fields)
                current_report_count = value
                item_name = f"Report Count = {value}"
        # Main items
        elif bType == 0:
            if bTag == 8:  # Input
                item_name = "Input"
                if current_report_id is not None and current_report_size is not None and current_report_count is not None:
                    total_bytes = (current_report_size * current_report_count) // 8
                    key = f"ID #{current_report_id} IN"
                    report_ids[key] = report_ids.get(key, 0) + total_bytes
            elif bTag == 9:  # Output
                item_name = "Output"
                if current_report_id is not None and current_report_size is not None and current_report_count is not None:
                    total_bytes = (current_report_size * current_report_count) // 8
                    key = f"ID #{current_report_id} OUT"
                    report_ids[key] = report_ids.get(key, 0) + total_bytes

        i += 1 + bSize

    if report_ids:
        print("\n  Declared HID reports:")
        for key, size in sorted(report_ids.items()):
            print(f"    {key:15s}: {size} data bytes + 1 report ID byte = {size+1} total")
    else:
        print("  (Could not parse report IDs from descriptor)")

    return report_ids


def build_getstatus_frame(report_id, pad_to):
    """Build a minimal CSAFE GETSTATUS command with the specified report ID.

    Frame: [report_id] [F1 start] [80 GETSTATUS] [80 checksum] [F2 stop] [padding...]
    """
    frame = [report_id, 0xF1, 0x80, 0x80, 0xF2]
    frame += [0x00] * (pad_to - len(frame))
    return bytes(frame)


def probe_report_id(dev, ep_in, ep_out, report_id, frame_size, read_size, timeout_ms=2000):
    """Send GETSTATUS using the given report ID and frame size.

    frame_size: exact size to write (report ID + data bytes as declared by HID descriptor)
    read_size:  how many bytes to request from the IN endpoint for the response

    Returns (success: bool, detail: str, raw_response: bytes|None).
    """
    frame = build_getstatus_frame(report_id, frame_size)

    # Flush any stale data
    for _ in range(8):
        try:
            dev.read(ep_in.bEndpointAddress, ep_in.wMaxPacketSize, timeout=50)
        except Exception:
            break

    try:
        dev.write(ep_out.bEndpointAddress, frame, timeout=timeout_ms)
    except Exception as e:
        return False, f"write failed: {e}", None

    try:
        resp = dev.read(ep_in.bEndpointAddress, read_size, timeout=timeout_ms)
        raw = bytes(resp)
        # Check if it's a valid CSAFE response
        if len(raw) > 1 and raw[1] == 0xF1:
            status = raw[2] if len(raw) > 2 else 0
            return True, f"OK, status={status:#04x}, {len(raw)} bytes", raw
        else:
            return False, f"response not CSAFE: {raw[:8].hex(' ')}", raw
    except Exception as e:
        return False, f"read failed: {e}", None


def _post_failure_recovery(dev, ep_in):
    """After a probe failure, flush and check if device is still alive.
    Returns True if device is alive, False if disconnected."""
    time.sleep(1.0)
    for _ in range(16):
        try:
            dev.read(ep_in.bEndpointAddress, ep_in.wMaxPacketSize, timeout=100)
        except Exception:
            break
    # Check alive
    try:
        dev.read(ep_in.bEndpointAddress, ep_in.wMaxPacketSize, timeout=200)
    except Exception as e:
        if hasattr(e, 'errno') and e.errno == 19:
            return False
    return True


def live_probe(dev, ep_in, ep_out, hid_reports):
    """Test each report ID with the actual PM5 to see which ones work.

    Uses the declared sizes from the HID report descriptor (hid_reports)
    so we send EXACTLY the right frame size.
    """
    print("\n═══ Live Report ID Probe ═══")
    print("  Sending CSAFE_GETSTATUS_CMD with each report ID …")
    print("  Using sizes from HID report descriptor (not from spec PDF).\n")

    # Build test cases from the parsed HID descriptor
    # Each report ID may have IN and OUT sizes — we write using OUT size, read using IN size
    test_ids = []
    for rid in sorted(set(int(k.split('#')[1].split()[0]) for k in hid_reports)):
        out_key = f"ID #{rid} OUT"
        in_key = f"ID #{rid} IN"
        out_size = hid_reports.get(out_key)
        in_size = hid_reports.get(in_key)
        if out_size is not None and in_size is not None:
            test_ids.append((rid, out_size + 1, in_size + 1))  # +1 for report ID byte

    if not test_ids:
        # Fallback to spec values if descriptor parsing failed
        test_ids = [(1, 21, 21), (2, 121, 121), (4, 63, 63)]

    for report_id, write_size, read_size in test_ids:
        desc = (f"ID #{report_id} — write {write_size} bytes "
                f"({write_size-1} data + 1 ID), "
                f"read up to {read_size} bytes")
        ok, detail, raw = probe_report_id(
            dev, ep_in, ep_out, report_id, write_size, read_size)
        status = "✓ WORKS" if ok else "✗ FAILED"
        print(f"  {desc}")
        print(f"    → {status}: {detail}")
        if raw and ok:
            resp_id = raw[0]
            print(f"    Response report ID: {resp_id:#04x}, "
                  f"response size: {len(raw)} bytes")
            show = min(32, len(raw))
            print(f"    Response hex ({show}B): {raw[:show].hex(' ')}")
        print()

        if not ok:
            alive = _post_failure_recovery(dev, ep_in)
            if not alive:
                print("    ⚠ Device disconnected after this report ID!")
                print("    Remaining report IDs cannot be tested.")
                return

    # Bonus tests: try wrong sizes to understand PM5's strictness
    print("  ── Bonus tests (size sensitivity) ──\n")

    bonus_tests = [
        (0x01, 63, 21,  "ID #1 oversized (63 instead of 21)"),
        (0x01, 501, 21, "ID #1 massively oversized (501 instead of 21)"),
    ]
    # Only add ID #4 with ID #1 size test if ID #4 exists
    if any(rid == 4 for rid, _, _ in test_ids):
        bonus_tests.append(
            (0x04, 21, 501, "ID #4 undersized (21 instead of 501)"))

    for report_id, write_size, read_size, desc in bonus_tests:
        ok, detail, raw = probe_report_id(
            dev, ep_in, ep_out, report_id, write_size, read_size)
        status = "✓ WORKS" if ok else "✗ FAILED"
        print(f"  {desc}")
        print(f"    → {status}: {detail}")
        if raw and ok:
            show = min(24, len(raw))
            print(f"    Response hex: {raw[:show].hex(' ')}")
        print()
        if not ok:
            alive = _post_failure_recovery(dev, ep_in)
            if not alive:
                print("    ⚠ Device disconnected!")
                return


def main():
    dev = find_pm5()
    ep_in, ep_out = setup_device(dev)

    dump_endpoint_info(ep_in, ep_out)
    hid_reports = read_hid_report_descriptor(dev)
    live_probe(dev, ep_in, ep_out, hid_reports)

    # Cleanup
    try:
        usb.util.release_interface(dev, INTERFACE)
    except Exception:
        pass

    print("\nDone.")


if __name__ == "__main__":
    main()
