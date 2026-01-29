#!/usr/bin/env python
# Copyright Axelera AI, 2025
# Rowing Ergometer Phase Controller

"""
Phase controller for rowing ergometer stroke detection.
Provides manual and automatic phase transitions for event-based keypoint capture.
"""

import multiprocessing
import time
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


def _poll_ergometer_loop_process(callback, cached_phase, cached_force_data, stop_event, poll_interval_idle, poll_interval_drive):
    """
    Ergometer USB polling loop - runs in separate process to avoid GIL contention.
    
    Args:
        callback: Ergometer phase callback function
        cached_phase: multiprocessing.Value for shared phase state
        cached_force_data: multiprocessing.Manager().list() for shared force data
        stop_event: multiprocessing.Event for shutdown signal
        poll_interval_idle: Polling interval during idle/recovery phases
        poll_interval_drive: Polling interval during drive phase
    """
    # Note: This runs in a separate process, so it has its own Python interpreter and no GIL contention
    last_phase = StrokePhase.IDLE
    
    while not stop_event.is_set():
        try:
            if callback:
                # USB request-response (blocks for MIN_FRAME_GAP per CSAFE spec)
                result = callback()
                
                # Handle both old (int) and new (dict) callback formats
                if isinstance(result, dict):
                    phase = result.get('phase', StrokePhase.IDLE)
                    force_data = result.get('force', [])
                else:
                    phase = result
                    force_data = []
                
                # Update shared phase state (atomic write to multiprocessing.Value)
                cached_phase.value = phase
                
                # Update shared force data if available
                if force_data:
                    cached_force_data[:] = force_data  # Replace list contents atomically
                elif phase != StrokePhase.DRIVE and last_phase == StrokePhase.DRIVE:
                    # Clear force data when exiting Drive phase
                    cached_force_data[:] = []
                
                # Adaptive polling: faster during Drive phase to catch all force samples
                poll_interval = poll_interval_drive if phase == StrokePhase.DRIVE else poll_interval_idle
                last_phase = phase
            else:
                poll_interval = poll_interval_idle
                
        except Exception:
            # Suppress errors in separate process (would be lost anyway)
            poll_interval = poll_interval_idle
        
        # Sleep until next poll
        time.sleep(poll_interval)


class PhaseController:
    """
    Controls stroke phase transitions for event-based keypoint capture.
    Uses Concept2 PM5 ergometer with background USB polling in separate process (no GIL contention).
    Note: PM5 CSAFE protocol does not support unsolicited updates - it's request-response only.
    """
    
    def __init__(self, recorder):
        """
        Initialize phase controller.
        
        Args:
            recorder: KeypointRecorder instance to update
        """
        self.recorder = recorder
        self.current_phase = StrokePhase.IDLE
        
        # Let recorder access our force data
        recorder.phase_controller = self
        
        # External phase callback (from ergometer USB)
        self.external_phase_callback: Optional[Callable[[], int]] = None
        
        # Shared memory for phase state (accessed from main process and USB polling process)
        self._cached_phase = multiprocessing.Value('i', StrokePhase.IDLE)  # Shared integer
        self._manager = multiprocessing.Manager()
        self._cached_force_data = self._manager.list()  # Shared list for force data
        self._stroke_force_data = {}  # Force data keyed by stroke timestamp (main process only)
        
        # USB polling process (separate process = no GIL contention)
        self._polling_process = None
        self._stop_polling = multiprocessing.Event()
        self._poll_interval_idle = 0.3  # 300ms polling during idle/recovery
        self._poll_interval_drive = 0.05  # 50ms polling during Drive - catch PM5 force buffer before it clears
        self._last_phase = StrokePhase.IDLE  # Track phase transitions
        
        LOG.info("PhaseController initialized for ergometer phase detection (multiprocessing mode)")
    
    def set_external_phase_callback(self, callback: Callable[[], int]):
        """
        Set callback function to get phase from external source (e.g., PM5 ergometer).
        Starts background USB polling process (PM5 doesn't support event-driven updates).
        Runs in separate process to avoid GIL contention.
        
        Args:
            callback: Function that returns current phase as int (0-4)
        """
        self.external_phase_callback = callback
        
        # Start background polling process (separate process = no GIL blocking)
        if self._polling_process is None:
            self._stop_polling.clear()
            self._polling_process = multiprocessing.Process(
                target=_poll_ergometer_loop_process,
                args=(callback, self._cached_phase, self._cached_force_data, 
                      self._stop_polling, self._poll_interval_idle, self._poll_interval_drive),
                daemon=True,  # Exit with main program
                name="ErgometerPoller"
            )
            self._polling_process.start()
            LOG.info(f"External phase callback registered, background USB polling started in separate process (idle={self._poll_interval_idle*1000:.0f}ms, drive={self._poll_interval_drive*1000:.0f}ms)")
    
    def update_phase_from_external(self):
        """Update phase from cached value (set by background polling process - no USB I/O here)"""
        if self.external_phase_callback:
            # Read from shared memory (no USB I/O, no GIL contention)
            new_phase = StrokePhase(self._cached_phase.value)
            if new_phase != self.current_phase:
                self._set_phase(new_phase)
    
    def _set_phase(self, phase: int):
        """
        Internal method to set stroke phase (called by ergometer callback).
        
        Args:
            phase: Phase number (0: idle, 1: prep, 2: drive, 3: dwelling, 4: recovery)
        """
        if phase != self.current_phase:
            LOG.info(f"Phase transition: {StrokePhase(self.current_phase).name} -> {StrokePhase(phase).name}")
            self.current_phase = phase
            self.recorder.set_phase(phase)
    
    def get_force_data(self) -> list:
        """Get force curve data for current stroke from shared memory"""
        # Read from shared list (thread-safe)
        return list(self._cached_force_data)
    
    def clear_force_data(self):
        """Clear cached force data after stroke has been saved"""
        self._cached_force_data[:] = []
    
    def stop(self):
        """Stop background USB polling process"""
        self._stop_polling.set()
        if self._polling_process and self._polling_process.is_alive():
            self._polling_process.join(timeout=1.0)
            if self._polling_process.is_alive():
                self._polling_process.terminate()  # Force kill if not responding
            LOG.info("Ergometer USB polling process stopped")
    
    def get_phase_name(self) -> str:
        """Get current phase name"""
        return StrokePhase(self.current_phase).name


# Exports
__all__ = ['PhaseController', 'StrokePhase']
