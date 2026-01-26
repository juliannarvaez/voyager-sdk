#!/usr/bin/env python
# Copyright Axelera AI, 2025
# Rowing Ergometer Phase Controller

"""
Phase controller for rowing ergometer stroke detection.
Provides manual and automatic phase transitions for event-based keypoint capture.
"""

import threading
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


class PhaseController:
    """
    Controls stroke phase transitions for event-based keypoint capture.
    Uses Concept2 PM5 ergometer with background USB polling thread.
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
        
        # Background polling thread (PM5 CSAFE is request-response, requires polling)
        self._cached_phase = StrokePhase.IDLE  # Cached from background thread
        self._cached_force_data = []  # Force curve for current stroke
        self._stroke_force_data = {}  # Force data keyed by stroke timestamp
        self._force_data_lock = threading.Lock()  # Protect force data access
        self._polling_thread = None
        self._stop_polling = threading.Event()
        self._poll_interval = 0.05  # 50ms polling rate for higher sample rate
        self._last_phase = StrokePhase.IDLE  # Track phase transitions
        
        LOG.info("PhaseController initialized for ergometer phase detection")
    
    def set_external_phase_callback(self, callback: Callable[[], int]):
        """
        Set callback function to get phase from external source (e.g., PM5 ergometer).
        Starts background USB polling thread (PM5 doesn't support event-driven updates).
        
        Args:
            callback: Function that returns current phase as int (0-4)
        """
        self.external_phase_callback = callback
        
        # Start background polling thread
        if self._polling_thread is None:
            self._stop_polling.clear()
            self._polling_thread = threading.Thread(
                target=self._poll_ergometer_loop,
                daemon=True,  # Exit with main program
                name="ErgometerPoller"
            )
            self._polling_thread.start()
            LOG.info(f"External phase callback registered, background USB polling started ({self._poll_interval*1000:.0f}ms interval)")
    
    def _poll_ergometer_loop(self):
        """Background thread that continuously polls ergometer via USB (CSAFE request-response)"""
        LOG.info("Ergometer USB polling thread started")
        poll_count = 0
        last_log_time = time.time()
        
        while not self._stop_polling.is_set():
            try:
                if self.external_phase_callback:
                    # USB request-response (blocks for MIN_FRAME_GAP per CSAFE spec)
                    result = self.external_phase_callback()
                    
                    # Handle both old (int) and new (dict) callback formats
                    if isinstance(result, dict):
                        phase = result.get('phase', StrokePhase.IDLE)
                        force_data = result.get('force', [])
                    else:
                        phase = result
                        force_data = []
                    
                    # Detect drive→dwelling/recovery transition (when force data becomes available)
                    if self._last_phase == 2 and phase in (3, 4) and len(force_data) > 0:
                        # Capture force data for this stroke
                        with self._force_data_lock:
                            self._cached_force_data = list(force_data)
                        LOG.info(f"Captured force curve: {len(force_data)} samples at phase transition 2→{phase}")
                    
                    # Update cached values
                    old_phase = self._cached_phase
                    self._cached_phase = phase
                    self._last_phase = phase
                    
                    # Log phase changes and periodic heartbeat
                    poll_count += 1
                    if phase != old_phase:
                        LOG.info(f"Ergometer phase changed: {old_phase} -> {phase}")
                    elif time.time() - last_log_time > 5.0:  # Heartbeat every 5 seconds
                        force_len = len(self._cached_force_data)
                        LOG.info(f"Ergometer polling active (poll #{poll_count}, phase={phase}, force_cached={force_len})")
                        last_log_time = time.time()
                        
            except Exception as e:
                LOG.error(f"Error in ergometer polling thread: {e}")
                import traceback
                LOG.error(traceback.format_exc())
            
            # Sleep for poll interval
            time.sleep(self._poll_interval)
        LOG.info("Ergometer USB polling thread stopped")
    
    def get_force_data(self):
        """
        Get cached force curve data from last completed stroke.
        Returns: List of force values (empty if not available)
        """
        with self._force_data_lock:
            return list(self._cached_force_data)
    
    def clear_force_data(self):
        """
        Clear cached force data after stroke has been saved.
        """
        with self._force_data_lock:
            self._cached_force_data = []
    
    def update_phase_from_external(self):
        """
        Update phase from cached ergometer value.
        Fast operation - just reads cached int (no USB I/O, no blocking).
        """
        # Read cached phase value (atomic read in Python)
        new_phase = self._cached_phase
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
    
    def stop(self):
        """Stop background USB polling thread"""
        if self._polling_thread is not None:
            LOG.info("Stopping ergometer USB polling thread")
            self._stop_polling.set()
            self._polling_thread.join(timeout=2.0)
            self._polling_thread = None
    
    def get_phase_name(self) -> str:
        """Get current phase name"""
        return StrokePhase(self.current_phase).name
