#!/usr/bin/env python
# Copyright Axelera AI, 2025
# Rowing Ergometer Phase Controller

"""
Phase controller for rowing ergometer stroke detection.
Provides manual and automatic phase transitions for event-based keypoint capture.
"""

import json
import multiprocessing
import sys
import time
import traceback
from enum import IntEnum
from typing import Optional, Callable

from axelera.app import logging_utils

LOG = logging_utils.getLogger(__name__)


class StrokePhase(IntEnum):
    """Rowing stroke phases"""
    IDLE = 0           # No rowing activity
    WAIT_ACCEL = 1     # Waiting for acceleration (catch position)
    DRIVE = 2          # Power/drive phase (pulling)
    DWELLING = 3       # Transition after drive
    RECOVERY = 4       # Recovery phase (return to catch)


def _poll_ergometer_loop_process(callback, cached_phase, cached_stroke_json,
                                 stroke_data_ready, stop_event,
                                 poll_interval_idle, poll_interval_drive):
    """
    Ergometer USB polling loop - runs in separate process to avoid GIL contention.

    Force data is drained incrementally inside the callback (get_phase()).
    When force drain completes, stroke data is written to shared memory and
    stroke_data_ready Event is set so the save worker can read it.

    Args:
        callback: Ergometer phase callback function
        cached_phase: multiprocessing.Value for shared phase state
        cached_stroke_json: multiprocessing.Array for stroke data (JSON bytes)
        stroke_data_ready: multiprocessing.Event — set when stroke data is in shared memory
        stop_event: multiprocessing.Event for shutdown signal
        poll_interval_idle: Polling interval during idle/recovery phases
        poll_interval_drive: Polling interval during drive phase
    """
    # ARM optimization: Pin USB polling to cores 0-1
    try:
        import os
        os.sched_setaffinity(0, {0, 1})
    except Exception:
        pass

    while not stop_event.is_set():
        try:
            if callback:
                result = callback()

                if isinstance(result, dict):
                    phase = result.get('phase', StrokePhase.IDLE)
                    force_data = result.get('force', [])
                    drag_factor = result.get('drag_factor')
                    stroke_stats = result.get('stroke_stats')
                else:
                    phase = result
                    force_data = []
                    drag_factor = None
                    stroke_stats = None

                with cached_phase.get_lock():
                    cached_phase.value = phase

                # Write stroke data to shared memory when force drain completes
                if force_data:
                    stroke_data = {'force': force_data}
                    if drag_factor is not None:
                        stroke_data['drag_factor'] = drag_factor
                    if stroke_stats is not None:
                        stroke_data['stroke_stats'] = stroke_stats

                    encoded = json.dumps(stroke_data, separators=(',', ':')).encode('utf-8')
                    with cached_stroke_json.get_lock():
                        length = len(encoded)
                        cached_stroke_json[:4] = length.to_bytes(4, 'little')
                        cached_stroke_json[4:4+length] = encoded

                    # Signal the save worker that stroke data is ready
                    stroke_data_ready.set()
                    print(f"[ErgPoller] Stroke data written to shared memory: "
                          f"{len(force_data)} force samples, {length} JSON bytes",
                          file=sys.stderr, flush=True)

                poll_interval = poll_interval_drive if phase == StrokePhase.DRIVE else poll_interval_idle
            else:
                poll_interval = poll_interval_idle

        except Exception as e:
            # DO NOT silently swallow — log the error
            print(f"[ErgPoller] ERROR in polling loop: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            poll_interval = poll_interval_idle

        time.sleep(poll_interval)


class PhaseController:
    """
    Controls stroke phase transitions for event-based keypoint capture.
    Uses Concept2 PM5 ergometer with background USB polling in separate process.
    Uses hidraw-based pyrow for reliable multi-packet HID report handling.
    """

    def __init__(self, recorder):
        self.recorder = recorder
        self.current_phase = StrokePhase.IDLE

        # Let recorder access our force data
        recorder.phase_controller = self

        self.external_phase_callback: Optional[Callable[[], int]] = None

        # Shared memory for phase state
        self._cached_phase = multiprocessing.Value('i', StrokePhase.IDLE, lock=True)

        # Shared byte array for stroke data (JSON: force + drag_factor + stroke_stats)
        self._cached_stroke_json = multiprocessing.Array('B', 16384, lock=True)
        self._cached_stroke_json[:4] = (0).to_bytes(4, 'little')

        # Event: set by polling process when stroke data is written to shared memory.
        # Save worker waits on this instead of polling — no race, no missed data.
        self._stroke_data_ready = multiprocessing.Event()

        self._polling_process = None
        self._stop_polling = multiprocessing.Event()
        self._poll_interval_idle = 0.050
        self._poll_interval_drive = 0.050

        LOG.info("PhaseController initialized (multiprocessing mode, event-signaled)")

    def set_external_phase_callback(self, callback: Callable[[], int]):
        """Set callback and start background USB polling process."""
        self.external_phase_callback = callback

        if self._polling_process is None:
            self._stop_polling.clear()
            self._polling_process = multiprocessing.Process(
                target=_poll_ergometer_loop_process,
                args=(callback, self._cached_phase, self._cached_stroke_json,
                      self._stroke_data_ready, self._stop_polling,
                      self._poll_interval_idle, self._poll_interval_drive),
                daemon=True,
                name="ErgometerPoller"
            )
            self._polling_process.start()
            LOG.info(f"USB polling started in separate process "
                     f"(idle={self._poll_interval_idle*1000:.0f}ms, "
                     f"drive={self._poll_interval_drive*1000:.0f}ms)")

    def update_phase_from_external(self):
        """Update phase from cached value (no USB I/O)."""
        if self.external_phase_callback:
            new_phase = StrokePhase(self._cached_phase.value)
            if new_phase != self.current_phase:
                self._set_phase(new_phase)

    def _set_phase(self, phase: int):
        if phase != self.current_phase:
            LOG.info(f"Phase transition: {StrokePhase(self.current_phase).name} -> {StrokePhase(phase).name}")
            self.current_phase = phase
            self.recorder.set_phase(phase)

    def _read_stroke_data(self) -> dict:
        """Read stroke data from shared memory. Returns empty dict if no data."""
        with self._cached_stroke_json.get_lock():
            length = int.from_bytes(bytes(self._cached_stroke_json[:4]), 'little')
            if length == 0:
                return {}
            raw = bytes(self._cached_stroke_json[4:4+length])
        try:
            return json.loads(raw.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            LOG.error(f"Failed to decode stroke data ({length} bytes): {e}")
            return {}

    def wait_for_stroke_data(self, timeout: float = 5.0) -> dict:
        """Wait for stroke data to become available in shared memory.

        The polling process sets _stroke_data_ready when force drain completes.
        This method blocks until the event is set or timeout expires.

        Returns:
            dict with 'force', 'drag_factor', 'stroke_stats' keys, or {} on timeout.
        """
        if self._stroke_data_ready.wait(timeout=timeout):
            data = self._read_stroke_data()
            if data:
                LOG.info(f"Stroke data received: {len(data.get('force', []))} force samples, "
                         f"drag={data.get('drag_factor')}")
            else:
                LOG.warning("stroke_data_ready was set but shared memory is empty")
            return data
        else:
            LOG.warning(f"No stroke data after {timeout}s timeout")
            return {}

    def clear_force_data(self):
        """Clear shared memory and reset the event after save worker has read."""
        with self._cached_stroke_json.get_lock():
            self._cached_stroke_json[:4] = (0).to_bytes(4, 'little')
        self._stroke_data_ready.clear()
        LOG.debug("Stroke data cleared + event reset")

    def get_all_stroke_data(self) -> dict:
        """Get all stroke data in a single shared-memory read."""
        return self._read_stroke_data()

    def get_force_data(self) -> list:
        return self._read_stroke_data().get('force', [])

    def get_drag_factor(self):
        return self._read_stroke_data().get('drag_factor')

    def get_stroke_stats(self) -> dict:
        return self._read_stroke_data().get('stroke_stats', {})

    def stop(self):
        """Stop background USB polling process."""
        self._stop_polling.set()
        if self._polling_process and self._polling_process.is_alive():
            self._polling_process.join(timeout=1.0)
            if self._polling_process.is_alive():
                self._polling_process.terminate()
            LOG.info("Ergometer USB polling process stopped")

    def get_phase_name(self) -> str:
        return StrokePhase(self.current_phase).name


__all__ = ['PhaseController', 'StrokePhase']
