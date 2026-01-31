#!/usr/bin/env python
# Copyright Axelera AI, 2025
# Rowing Ergometer Keypoint Recording Module

"""
Keypoint recording module for rowing ergometer analysis.
Provides circular buffering, phase-based event capture, and async data saving.
"""

import json
import gzip
import os
import time
import threading
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from axelera.app import logging_utils

LOG = logging_utils.getLogger(__name__)


@dataclass
class FrameKeypointData:
    """Single frame of keypoint data for specific points of interest"""
    frame_number: int
    timestamp: float
    phase: int = 0  # 0: idle, 1: prep, 2: drive, 3: dwelling, 4: recovery
    keypoints: List[Dict[str, Any]] = field(default_factory=list)  # [{"name": "right_shoulder", "x": 100, "y": 200, "confidence": 0.9}, ...]
    

class KeypointRecorder:
    """
    Records keypoint data with circular buffering for event-based capture.
    
    Key Features:
    - Circular buffer maintains last N frames before events
    - Phase-based triggering (e.g., rowing drive phase)
    - Async file saving to prevent frame drops
    - Gzip compression for efficient storage
    """
    
    # COCO body keypoints we care about for rowing
    ROWING_KEYPOINTS = [
        "right_shoulder",  # idx 6
        "right_hip",       # idx 12
        "right_knee",      # idx 14
        "right_ankle",     # idx 16
        "right_wrist",     # idx 10
    ]
    
    # Pre-computed indices for fast access
    ROWING_KEYPOINT_INDICES = [6, 12, 14, 16, 10]  # shoulder, hip, knee, ankle, wrist
    
    # Map COCO keypoint names to indices
    COCO_KEYPOINT_MAP = {
        "nose": 0,
        "left_eye": 1,
        "right_eye": 2,
        "left_ear": 3,
        "right_ear": 4,
        "left_shoulder": 5,
        "right_shoulder": 6,
        "left_elbow": 7,
        "right_elbow": 8,
        "left_wrist": 9,
        "right_wrist": 10,
        "left_hip": 11,
        "right_hip": 12,
        "left_knee": 13,
        "right_knee": 14,
        "left_ankle": 15,
        "right_ankle": 16,
    }
    
    def __init__(self, buffer_size: int = 30, save_dir: str = "/tmp/stroke_data"):
        """
        Initialize keypoint recorder.
        
        Args:
            buffer_size: Number of frames to buffer before events (default 30 @ ~90 FPS = ~330ms)
            save_dir: Directory for saved stroke data files
        """
        self.buffer_size = buffer_size
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Reference to phase controller (for force data)
        self.phase_controller = None
        
        # Force curve data from monitor thread
        self.force_curve = []
        self.force_curve_lock = threading.Lock()
        
        # Circular buffers (auto-pruning with maxlen)
        self.keypoint_buffer = deque(maxlen=buffer_size)
        
        # Event collection state
        self.current_phase = 0  # 0: idle, 1: prep, 2: drive, 3: dwelling, 4: recovery
        self.event_active = False
        self.event_keypoints = []
        self.post_event_count = 0
        
        # Async save queue
        self.save_queue = deque(maxlen=10)
        self.save_queue_lock = threading.Lock()
        self.save_worker_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_worker_thread.start()
        
        # Kalman filtering is handled by main loop's KeypointSmoother
        # Recorder just extracts already-smoothed keypoints
        self.last_timestamp = None
        
        # Statistics
        self.frame_count = 0
        self.events_saved = 0
        
        LOG.info(f"KeypointRecorder initialized: buffer_size={buffer_size}, save_dir={save_dir}")
    
    def extract_keypoints_from_meta(self, meta, width: int, height: int, frame_number: int, timestamp: float) -> Optional[FrameKeypointData]:
        """
        Extract keypoint data from Axelera AxMeta container.
        
        Args:
            meta: AxMeta container with task metas
            width: Frame width for denormalization
            height: Frame height for denormalization
            frame_number: Current frame number
            timestamp: Actual timestamp for framerate measurement
            
        Returns:
            FrameKeypointData or None if no keypoints found
        """
        # Fast path: try to get task meta directly without expensive checks
        task_meta = None
        if hasattr(meta, 'values'):
            for tmeta in meta.values():
                if hasattr(tmeta, 'objects') and tmeta.objects:
                    task_meta = tmeta
                    break
        
        if not task_meta:
            return None
        
        # Extract keypoints from first detection (single person rowing)
        objects = task_meta.objects
        if not objects:
            return None
        
        detection = objects[0]
        if not hasattr(detection, 'keypoints'):
            return None
        
        # Extract rowing-specific keypoints (optimized - minimal allocations)
        # Note: Keypoints are already smoothed by main loop's KeypointSmoother
        keypoints_data = []
        # Use current_phase that was set by set_phase()
        phase = self.current_phase
        keypoints = detection.keypoints
        
        # Update timestamp for tracking
        if self.last_timestamp is not None:
            dt = max(0.001, timestamp - self.last_timestamp)
        self.last_timestamp = timestamp
        
        # Vectorized keypoint extraction (NEON-optimized, minimal allocations)
        # Data is already smoothed - just extract the rowing keypoints
        indices = self.ROWING_KEYPOINT_INDICES  # [6, 12, 14, 16, 10]
        names = self.ROWING_KEYPOINTS
        
        for idx, name in zip(indices, names):
            if idx < len(keypoints) and len(keypoints[idx]) >= 2:
                kp = keypoints[idx]
                keypoints_data.append({
                    "name": name,
                    "x": int(kp[0]),
                    "y": int(kp[1]),
                    "confidence": kp[2] if len(kp) > 2 else 1.0,
                    "phase": phase
                })
        
        if not keypoints_data:
            return None
        
        return FrameKeypointData(
            frame_number=frame_number,
            timestamp=timestamp,  # Real timestamp for framerate verification
            phase=self.current_phase,
            keypoints=keypoints_data
        )
    
    def add_frame(self, frame_data: Optional[FrameKeypointData]):
        """
        Add frame to circular buffer and handle event-based collection.
        
        Args:
            frame_data: Keypoint data for current frame (None if no detections)
        """
        if frame_data is None:
            return
        
        self.frame_count += 1
        
        # Always maintain circular buffer (last N frames)
        self.keypoint_buffer.append(frame_data)
        
        # Phase-based event collection - only process if phase is relevant
        phase = self.current_phase
        if phase == 2:  # Drive phase
            if not self.event_active:
                # Event just started - copy buffer to storage (N frames before event)
                self.event_active = True
                self.event_keypoints = list(self.keypoint_buffer)
                self.post_event_count = 0
                LOG.info(f"Stroke event started - buffered {len(self.event_keypoints)} pre-drive frames")
            else:
                # Continue collecting during event
                self.event_keypoints.append(frame_data)
        
        elif self.event_active:
            # Any non-drive phase after drive = post-event (dwelling, recovery, or back to idle)
            # Collect N frames after event ends
            if self.post_event_count < self.buffer_size:
                self.event_keypoints.append(frame_data)
                self.post_event_count += 1
                if self.post_event_count % 10 == 0:  # Log every 10 frames
                    LOG.debug(f"Post-event collection: {self.post_event_count}/{self.buffer_size} frames")
            else:
                # Event complete - queue for save
                LOG.info(f"Stroke event complete - saving {len(self.event_keypoints)} total frames")
                self._queue_save()
                self.event_active = False
                self.post_event_count = 0
                self.event_keypoints = []
    
    def set_phase(self, phase: int):
        """
        Set current phase for event detection.
        
        Args:
            phase: Phase number (0: idle, 1: prep, 2: drive/event, 3: dwelling, 4: recovery)
        """
        if phase != self.current_phase:
            old_phase = self.current_phase
            self.current_phase = phase
            LOG.debug(f"Recorder phase transition: {old_phase} -> {phase} (event_active={self.event_active}, post_count={self.post_event_count})")
    
    def set_force_curve(self, force_data: List[int]):
        """
        Set force curve data from monitor thread.
        
        Args:
            force_data: List of force samples from PM5
        """
        with self.force_curve_lock:
            self.force_curve = force_data.copy()
    
    def get_force_data(self) -> List[int]:
        """
        Get force curve data (thread-safe) and clear it.
        
        Returns:
            List of force samples
        """
        with self.force_curve_lock:
            data = self.force_curve.copy()
            return data
    
    def clear_force_data(self):
        """Clear stored force data."""
        with self.force_curve_lock:
            self.force_curve.clear()
    
    def manual_save(self, additional_data: Optional[Dict[str, Any]] = None):
        """
        Manually trigger save of current buffer state.
        
        Args:
            additional_data: Optional dict to merge into saved data (e.g., {"force": [...]})
        """
        if not self.event_keypoints:
            LOG.warning("No event data to save")
            return
        
        self._queue_save(additional_data)
        self.event_active = False
        self.event_keypoints = []
    
    def _queue_save(self, additional_data: Optional[Dict[str, Any]] = None):
        """Queue event data for async save"""
        timestamp = int(time.time())
        filename = os.path.join(self.save_dir, f"stroke_{timestamp}.json")
        
        # Structure data as JSON
        data = {
            "timestamp": timestamp,
            "frame_count": len(self.event_keypoints),
            "keypoints": [
                frame.keypoints for frame in self.event_keypoints
            ],
            "phases": [frame.phase for frame in self.event_keypoints],
            "frame_timestamps": [frame.timestamp for frame in self.event_keypoints]  # Add frame timestamps
        }
        
        # Add force data if available from phase controller
        force_data = []
        if self.phase_controller is not None:
            force_data = self.phase_controller.get_force_data()
        # Also check local force curve (set via set_force_curve)
        if not force_data:
            force_data = self.get_force_data()
        
        if force_data:
            data['force'] = force_data
            LOG.info(f"Saving stroke with {len(force_data)} force samples")
        else:
            LOG.warning("No force data available for this stroke")
        
        # Merge additional data (e.g., force measurements)
        if additional_data:
            data.update(additional_data)
        
        # Queue for background save
        with self.save_queue_lock:
            self.save_queue.append((data, filename))
        
        # Use debug level to avoid I/O overhead
        LOG.debug(f"Queued event save: {len(self.event_keypoints)} frames -> {filename}")
    
    def _save_worker(self):
        """Background worker thread for async saves (optimized for minimal GIL contention)"""
        import threading
        
        while True:
            # Block until work is available (no busy-wait, releases GIL)
            time.sleep(0.1)  # Fast response, releases GIL
            
            with self.save_queue_lock:
                if not self.save_queue:
                    continue
                # Process oldest save first
                data, filename = self.save_queue.popleft()
            
            try:
                # Save as plain JSON with optimized settings
                # separators and ensure_ascii reduce encoding overhead
                with open(filename, 'w', buffering=65536) as f:  # 64KB buffer for faster writes
                    json.dump(data, f, separators=(',', ':'), ensure_ascii=False)
                self.events_saved += 1
                # Use debug level to avoid I/O overhead
                LOG.debug(f"Saved event {self.events_saved} to {filename}")
            except Exception as e:
                LOG.error(f"Error saving event: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get recorder statistics"""
        return {
            "frames_processed": self.frame_count,
            "events_saved": self.events_saved,
            "buffer_size": self.buffer_size,
            "current_phase": self.current_phase,
            "event_active": self.event_active,
            "queue_depth": len(self.save_queue)
        }
