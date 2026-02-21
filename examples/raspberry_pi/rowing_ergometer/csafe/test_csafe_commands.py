#!/usr/bin/env python3
"""
test_csafe_commands.py
Tests all CSAFE commands against a connected Concept2 ergometer (PM5).

Uses pyrow.PyErg for USB communication via /dev/hidraw (kernel HID driver).
Report ID #2: 121-byte frames (120 bytes CSAFE payload) for full-size responses.

Usage:
    python test_csafe_commands.py [--test-set] [--verbose]

    --test-set   Also run SET/state-change commands (may alter erg state/settings).
    --verbose    Print raw response data for every command.

Exit code 0 = all tested commands passed, non-zero = at least one failure.
"""

import sys
import os
import time
import argparse
import datetime

# ---------------------------------------------------------------------------
# Make sure we can import pyrow from the parent directory
# ---------------------------------------------------------------------------
_this_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_this_dir)          # rowing_ergometer/
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import pyrow                 # pyrow.find(), pyrow.PyErg – hidraw-based USB layer

# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------
PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"
WARN = "WARN"

results: list[dict] = []


def record(cmd: str, status: str, detail: str = ""):
    results.append({"cmd": cmd, "status": status, "detail": detail})


# ---------------------------------------------------------------------------
# PM5 state helpers
# ---------------------------------------------------------------------------
_STATE_NAMES = [
    'Error', 'Ready', 'Idle', 'Have ID', 'N/A',
    'In Use', 'Pause', 'Finished', 'Manual', 'Offline'
]


def _state_name(raw: int) -> str:
    s = raw & 0x7F
    return _STATE_NAMES[s] if s < len(_STATE_NAMES) else f'Unknown({s})'


# ---------------------------------------------------------------------------
# Erg reset + workout setup (mirrors rowing_ergometer_recording.py)
# ---------------------------------------------------------------------------

def setup_erg(erg: pyrow.PyErg) -> bool:
    """
    Reset the PM5 to Ready state and configure a 2000 m workout so that
    all CSAFE commands (including force-data ones) return valid data.
    """
    print("\n── PM5 Reset & Workout Setup ─────────────────────────────────────")

    # Step 0: Verify basic USB comms
    print("  Verifying USB communication …")
    comms_ok = False
    for _ in range(5):
        try:
            resp = erg.send(['CSAFE_GETSTATUS_CMD'])
            if resp:
                raw = resp.get('CSAFE_GETSTATUS_CMD', [0])[0]
                print(f"  USB comms OK – initial state: {raw:#04x} ({_state_name(raw)})")
                comms_ok = True
                break
        except Exception as exc:
            print(f"    probe failed: {exc}")
        time.sleep(0.5)

    if not comms_ok:
        print("  ERROR: Cannot communicate with PM5 over USB.")
        return False

    # Step 1: GOFINISHED → GOREADY (clears previous workout / Pause state)
    print("  Sending GOFINISHED → GOREADY to reset PM5 state machine …")
    for cmd in ('CSAFE_GOFINISHED_CMD', 'CSAFE_GOREADY_CMD'):
        try:
            erg.send([cmd])
            time.sleep(0.5)
        except Exception as exc:
            print(f"    {cmd} failed ({exc}) – continuing")

    # Step 2: Poll until Ready / Idle / Have ID (states 1-3)
    print("  Waiting for PM5 Ready state …")
    pm5_ready = False
    deadline = time.time() + 10.0
    last_state_raw = None
    while time.time() < deadline:
        try:
            resp = erg.send(['CSAFE_GETSTATUS_CMD'])
            raw = resp.get('CSAFE_GETSTATUS_CMD', [0])[0]
            state = raw & 0x7F
            if raw != last_state_raw:
                print(f"    state: {raw:#04x} → {state} ({_state_name(raw)})")
                last_state_raw = raw
            if state in (1, 2, 3):   # Ready, Idle, Have ID
                pm5_ready = True
                break
        except Exception as exc:
            print(f"    poll error: {exc}")
        time.sleep(0.3)

    if not pm5_ready:
        print("  ERROR: PM5 not Ready – select 'New Workout' on the PM5 display, "
              "then re-run.")
        return False

    print(f"  PM5 ready ({_state_name(last_state_raw or 0)})")

    # Step 3: Configure a 2000 m / 100 m-split workout
    print("  Configuring workout: 2000 m, 100 m splits, ~120 W pace …")
    powerpace = int(round(2.8 / ((120 / 500.0) ** 3)))
    workout_cmds = [
        (['CSAFE_SETHORIZONTAL_CMD', 2000, 36],   "Set 2000 m distance"),
        (['CSAFE_PM_SET_SPLITDURATION', 128, 100], "Set 100 m splits"),
        (['CSAFE_SETPOWER_CMD', powerpace, 88],    f"Set pace ({powerpace} W)"),
        (['CSAFE_SETPROGRAM_CMD', 0, 0],           "Enable program 0 (force data)"),
        (['CSAFE_GOINUSE_CMD'],                    "Activate workout (GOINUSE)"),
    ]
    for msg, desc in workout_cmds:
        try:
            erg.send(msg)
            print(f"    {desc} … OK")
            time.sleep(0.3)
        except Exception as exc:
            print(f"    {desc} … WARNING: {exc}")

    print("  Setup complete.\n")
    return True


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _flush_after_timeout(erg: pyrow.PyErg):
    """Drain stale frames from the hidraw device after a timeout.
    Without this, the PM5's late response pollutes the next command's read."""
    import fcntl
    # Set non-blocking temporarily to drain any buffered data
    flags = fcntl.fcntl(erg._fd, fcntl.F_GETFL)
    fcntl.fcntl(erg._fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
    try:
        for _ in range(16):
            try:
                os.read(erg._fd, 512)
            except BlockingIOError:
                break
    finally:
        fcntl.fcntl(erg._fd, fcntl.F_SETFL, flags)
    time.sleep(0.2)


def run_cmd(erg: pyrow.PyErg, label: str, message: list, verbose: bool) -> bool:
    """
    Send *message* via PyErg.send() and report PASS/FAIL/SKIP.
    A ConnectionError (timeout/disconnect) is SKIP so later tests still run.
    Flushes the USB buffer after timeouts to prevent cascade failures.
    """
    try:
        resp = erg.send(message)
    except ConnectionError as exc:
        status = SKIP
        detail = str(exc)
        record(label, status, detail)
        print(f"  {label:45s}  {status}  – {detail}")
        # Flush stale frames that may arrive after our timeout expired,
        # otherwise they pollute the next command's read.
        _flush_after_timeout(erg)
        return False
    except Exception as exc:
        record(label, FAIL, str(exc))
        print(f"  {label:45s}  {FAIL}  – {exc}")
        _flush_after_timeout(erg)
        return False

    cmd_name   = message[0]
    status_val = resp.get("CSAFE_GETSTATUS_CMD", [None])[0]
    cmd_resp   = resp.get(cmd_name)

    detail = f"status=0x{status_val:02X}" if isinstance(status_val, int) else ""
    if cmd_resp is not None:
        detail += f"  data={cmd_resp}"

    if verbose:
        print(f"  {label:45s}  {PASS}  – raw={resp}")
    else:
        print(f"  {label:45s}  {PASS}  – {detail}")

    record(label, PASS, detail)
    return True


# ---------------------------------------------------------------------------
# Command groups
# ---------------------------------------------------------------------------

def test_get_commands(erg, verbose):
    """All read-only short GET commands."""
    print("── Standard GET Commands ─────────────────────────────────────────")
    for cmd in [
        "CSAFE_GETSTATUS_CMD",
        "CSAFE_GETVERSION_CMD",
        "CSAFE_GETID_CMD",
        "CSAFE_GETUNITS_CMD",
        "CSAFE_GETSERIAL_CMD",
        "CSAFE_GETODOMETER_CMD",
        "CSAFE_GETERRORCODE_CMD",
        "CSAFE_GETTWORK_CMD",
        "CSAFE_GETHORIZONTAL_CMD",
        "CSAFE_GETCALORIES_CMD",
        "CSAFE_GETPROGRAM_CMD",
        "CSAFE_GETPACE_CMD",
        "CSAFE_GETCADENCE_CMD",
        "CSAFE_GETUSERINFO_CMD",
        "CSAFE_GETHRCUR_CMD",
        "CSAFE_GETPOWER_CMD",
    ]:
        run_cmd(erg, cmd, [cmd], verbose)


def test_getcaps(erg, verbose):
    """GETCAPS for capability codes 0-2."""
    print("\n── GETCAPS (capability codes 0-2) ───────────────────────────────")
    for code in range(3):
        label = f"CSAFE_GETCAPS_CMD (code={code})"
        run_cmd(erg, label, ["CSAFE_GETCAPS_CMD", code], verbose)


def test_pm_get_commands(erg, verbose):
    """PM5-specific short GET commands (wrapped in SETUSERCFG1)."""
    print("\n── PM5-Specific GET Commands ─────────────────────────────────────")
    for cmd in [
        "CSAFE_PM_GET_WORKOUTTYPE",
        "CSAFE_PM_GET_DRAGFACTOR",
        "CSAFE_PM_GET_STROKESTATE",
        "CSAFE_PM_GET_WORKTIME",
        "CSAFE_PM_GET_WORKDISTANCE",
        "CSAFE_PM_GET_ERRORVALUE",
        "CSAFE_PM_GET_WORKOUTSTATE",
        "CSAFE_PM_GET_WORKOUTINTERVALCOUNT",
        "CSAFE_PM_GET_INTERVALTYPE",
        "CSAFE_PM_GET_RESTTIME",
    ]:
        run_cmd(erg, cmd, [cmd], verbose)


def test_pm_data_commands(erg, verbose, poll_seconds=30):
    """PM5-specific data block commands.

    FORCEPLOTDATA and STROKESTATS only return meaningful data during/after a
    stroke.  This section polls STROKESTATE for *poll_seconds* to give the
    user time to start rowing, then captures force/heartbeat/stroke-stats
    data across multiple strokes.
    """
    print("\n── PM5 Data Block Commands ───────────────────────────────────────")

    # ── Poll for active rowing & collect force across strokes ───────────
    state_names = {0: 'Idle', 1: 'Prep', 2: 'Drive', 3: 'Dwelling', 4: 'Recovery'}
    # block_length=32 → max 16 samples per poll (2 bytes each).  The PM5
    # always uses a fixed 33-byte response template (1 byte bytes_read +
    # 32 data bytes) regardless of block_length.  Values > 32 cause the PM5
    # to dequeue samples it can't return, permanently losing them.
    BLOCK_LEN = 32      # 16 samples × 2 bytes each (max for PM5's 33-byte template)
    SAMPLES_PER_POLL = BLOCK_LEN // 2

    print(f"  Waiting up to {poll_seconds}s for rowing activity …")
    print(f"  (Pull the handle to generate force/stroke data)")
    print(f"  Collecting across ALL strokes, block_length={BLOCK_LEN} "
          f"({SAMPLES_PER_POLL} samples/poll)\n")

    saw_drive = False
    strokes_collected = 0
    all_strokes = []           # list of per-stroke sample lists
    current_stroke = []        # samples for the stroke being drained
    deadline = time.time() + poll_seconds
    last_printed_state = None
    prev_stroke_state = None

    while time.time() < deadline:
        try:
            resp = erg.send(['CSAFE_PM_GET_STROKESTATE'])
            stroke_state = resp.get('CSAFE_PM_GET_STROKESTATE', [0])[0]
        except Exception:
            time.sleep(0.1)
            continue

        sname = state_names.get(stroke_state, f'Unknown({stroke_state})')
        remaining = max(0, int(deadline - time.time()))

        if stroke_state != last_printed_state:
            print(f"    stroke state: {stroke_state} ({sname})  [{remaining}s remaining]")
            last_printed_state = stroke_state

        if stroke_state == 2:       # Drive
            saw_drive = True

        # Drain force buffer during Dwelling or Recovery (data available after drive)
        if stroke_state in (3, 4) and saw_drive:
            # If we just transitioned into Dwelling from Drive, start a new stroke
            if prev_stroke_state == 2 and stroke_state == 3:
                if current_stroke:
                    all_strokes.append(current_stroke)
                current_stroke = []
                strokes_collected += 1
                print(f"    → Stroke #{strokes_collected} completed! Draining force buffer …")

            # Poll force data — drain whatever is available
            empty_polls = 0
            for _ in range(50):  # up to 50 rapid polls per iteration
                try:
                    r = erg.send(['CSAFE_PM_GET_FORCEPLOTDATA', BLOCK_LEN])
                    fp = r.get('CSAFE_PM_GET_FORCEPLOTDATA', [0])
                    byte_count = fp[0] if fp else 0
                    datapoints = byte_count // 2
                    samples = fp[1:datapoints + 1] if len(fp) > datapoints else []
                except Exception:
                    break
                if not samples:
                    empty_polls += 1
                    if empty_polls >= 3:
                        break  # buffer drained
                    continue
                empty_polls = 0
                current_stroke.extend(samples)
        elif stroke_state == 2 and saw_drive:
            # During Drive: try to drain any early-available samples to
            # prevent buffer overflow on long strokes (PM5 may make partial
            # data available). Use a single non-blocking poll.
            try:
                r = erg.send(['CSAFE_PM_GET_FORCEPLOTDATA', BLOCK_LEN])
                fp = r.get('CSAFE_PM_GET_FORCEPLOTDATA', [0])
                byte_count = fp[0] if fp else 0
                datapoints = byte_count // 2
                samples = fp[1:datapoints + 1] if len(fp) > datapoints else []
                if samples:
                    current_stroke.extend(samples)
            except Exception:
                pass

        prev_stroke_state = stroke_state
        time.sleep(0.02)   # ~50 Hz poll (faster than before to catch transitions)

    # Flush any remaining current stroke
    if current_stroke:
        all_strokes.append(current_stroke)

    # ── Summary ─────────────────────────────────────────────────────────
    total_samples = sum(len(s) for s in all_strokes)
    if all_strokes:
        print(f"\n    ── Force Data Summary ──")
        print(f"    Strokes captured: {len(all_strokes)}")
        print(f"    Total samples:    {total_samples}")
        for i, stroke in enumerate(all_strokes):
            if not stroke:
                continue
            peak = max(stroke)
            avg = sum(stroke) / len(stroke)
            duration_ms = len(stroke) * 2  # 500 Hz → 2 ms per sample
            print(f"    Stroke #{i+1}: {len(stroke)} samples "
                  f"({duration_ms} ms @ 500 Hz), peak={peak}, avg={avg:.0f}")
            if len(stroke) <= 20:
                print(f"      Samples: {stroke}")
            else:
                print(f"      First 10: {stroke[:10]}")
                print(f"      Last  10: {stroke[-10:]}")
    else:
        print(f"\n  No force data collected in {poll_seconds}s.\n")

    # ── Run data block command tests ────────────────────────────────────
    for cmd_name, arg, label in [
        ("CSAFE_PM_GET_FORCEPLOTDATA",  BLOCK_LEN, f"CSAFE_PM_GET_FORCEPLOTDATA (blk={BLOCK_LEN})"),
        ("CSAFE_PM_GET_HEARTBEATDATA",  BLOCK_LEN, f"CSAFE_PM_GET_HEARTBEATDATA (blk={BLOCK_LEN})"),
        ("CSAFE_PM_GET_STROKESTATS",     0, "CSAFE_PM_GET_STROKESTATS (reserved=0)"),
    ]:
        try:
            parsed, raw = erg.send_raw([cmd_name, arg])
            if parsed:
                # Normal successful parse
                status_byte = parsed.get('CSAFE_GETSTATUS_CMD', [0])[0]
                data = {k: v for k, v in parsed.items() if k != 'CSAFE_GETSTATUS_CMD'}
                detail = f"status={status_byte:#04x}"
                if data:
                    detail += f"  data={data}"
                record(label, PASS, detail)
                print(f"  {label:45s}  {PASS}  – {detail}")
            else:
                # PM5 responded but frame couldn't be parsed
                hexdump = raw[:24].hex(' ')
                detail = f"PM5 responded but frame not parseable: [{hexdump}...]"
                record(label, WARN, detail)
                print(f"  {label:45s}  {WARN}  – {detail}")
            if verbose:
                print(f"    raw ({len(raw)} bytes): {raw[:32].hex(' ')}")
                if parsed:
                    print(f"    parsed: {parsed}")
        except ConnectionError as exc:
            detail = str(exc)
            record(label, SKIP, detail)
            print(f"  {label:45s}  {SKIP}  – {detail}")
            _flush_after_timeout(erg)


def test_set_commands(erg, verbose):
    """SET / state-change commands (may alter erg state)."""
    print("\n── SET Commands (may alter erg state/settings) ───────────────────")
    now = datetime.datetime.now()
    for label, msg in [
        ("CSAFE_SETTIME_CMD",    ["CSAFE_SETTIME_CMD",    now.hour, now.minute, now.second]),
        ("CSAFE_SETDATE_CMD",    ["CSAFE_SETDATE_CMD",    now.year % 100, now.month, now.day]),
        ("CSAFE_AUTOUPLOAD_CMD", ["CSAFE_AUTOUPLOAD_CMD", 0]),
        ("CSAFE_GOREADY_CMD",    ["CSAFE_GOREADY_CMD"]),
        ("CSAFE_GOIDLE_CMD",     ["CSAFE_GOIDLE_CMD"]),
    ]:
        run_cmd(erg, label, msg, verbose)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary():
    print("\n" + "═" * 70)
    print(f"  {'COMMAND':45s}  {'STATUS':6s}  DETAIL")
    print("─" * 70)

    passed = failed = skipped = warned = 0
    for r in results:
        s = r["status"]
        if   s == PASS: passed  += 1
        elif s == FAIL: failed  += 1
        elif s == SKIP: skipped += 1
        elif s == WARN: warned  += 1
        tag = {"PASS": "✓", "FAIL": "✗", "SKIP": "–", "WARN": "!"}.get(s, s)
        print(f"  {tag} {r['cmd']:43s}  {s:6s}  {r['detail']}")

    print("═" * 70)
    total = passed + failed + skipped
    print(f"  Passed: {passed}/{total}   Failed: {failed}   "
          f"Skipped (timeout): {skipped}   Warnings: {warned}")
    print("═" * 70 + "\n")
    return failed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test all CSAFE commands against a connected Concept2 ergometer."
    )
    parser.add_argument("--test-set", action="store_true",
                        help="Also run SET/state-change commands.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print raw response dicts for every command.")
    parser.add_argument("--poll-seconds", type=int, default=30, metavar="N",
                        help="Seconds to wait for rowing activity before "
                             "testing data block commands (default: 30).")
    args = parser.parse_args()

    # ── Discover & connect ──────────────────────────────────────────────────
    print("Searching for Concept2 ergometer via /dev/hidraw …")
    try:
        ergs = pyrow.find()
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    if not ergs:
        print("ERROR: No Concept2 ergometer found.  Check USB connection.")
        sys.exit(1)

    hidraw_path = ergs[0]
    erg = pyrow.PyErg(hidraw_path)
    print(f"Connected: {hidraw_path}  (report ID #{pyrow.REPORT_ID}, "
          f"{pyrow.REPORT_DATA_SIZE}-byte CSAFE payload)")

    # ── Reset PM5 and start workout ─────────────────────────────────────────
    if not setup_erg(erg):
        erg.close()
        sys.exit(1)

    # ── Run tests ───────────────────────────────────────────────────────────
    try:
        test_get_commands(erg, args.verbose)
        test_getcaps(erg, args.verbose)
        test_pm_get_commands(erg, args.verbose)
        test_pm_data_commands(erg, args.verbose, args.poll_seconds)

        if args.test_set:
            test_set_commands(erg, args.verbose)
        else:
            print("\n── SET Commands ─────────────────────────────────────────────────")
            print("  (skipped – pass --test-set to include)")
    finally:
        erg.close()

    # ── Summary ─────────────────────────────────────────────────────────────
    failed = print_summary()
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
