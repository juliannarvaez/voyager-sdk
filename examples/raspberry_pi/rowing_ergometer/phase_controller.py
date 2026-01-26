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
    Uses Concept2 PM5 ergometer for automatic phase detection.
    """
    
    def __init__(self, recorder):
        """
        Initialize phase controller.
        
        Args:
            recorder: KeypointRecorder instance to update
        """
        self.recorder = recorder
        self.current_phase = StrokePhase.IDLE
        
        # External phase callback (from ergometer USB)
        self.external_phase_callback: Optional[Callable[[], int]] = None
        
        LOG.info("PhaseController initialized for ergometer phase detection")
    
    def set_external_phase_callback(self, callback: Callable[[], int]):
        """
        Set callback function to get phase from external source (e.g., PM5 ergometer).
        
        Args:
            callback: Function that returns current phase as int (0-4)
        """
        self.external_phase_callback = callback
        LOG.info("External phase callback registered")
    
    def update_phase_from_external(self):
        """Update phase from ergometer callback (Concept2 PM5)"""
        if self.external_phase_callback:
            try:
                new_phase = self.external_phase_callback()
                if new_phase != self.current_phase:
                    self._set_phase(new_phase)
            except Exception as e:
                LOG.error(f"Error getting ergometer phase: {e}")
    
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
    

    
    def get_phase_name(self) -> str:
        """Get current phase name"""
        return StrokePhase(self.current_phase).name
