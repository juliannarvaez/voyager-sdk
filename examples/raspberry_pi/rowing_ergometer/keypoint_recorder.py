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
    velocities: List[Dict[str, Any]] = field(default_factory=list)  # [{"name": "right_knee", "vx": 1.2, "vy": -0.3, "speed": 1.25}, ...]
    

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
            timestamp: Frame timestamp
            
        Returns:
            FrameKeypointData or None if no keypoints found
        """
        from axelera.app.meta.keypoint import KeypointObjectWithBbox, CocoBodyKeypointsMeta
        
        # Find keypoint task meta in container
        task_meta = None
        for key, tmeta in meta.items():
            if isinstance(tmeta, CocoBodyKeypointsMeta) or 'keypoint' in key.lower() or 'pose' in key.lower():
                task_meta = tmeta
                break
        
        if not task_meta or not hasattr(task_meta, 'objects'):
            return None
        
        # Extract keypoints from first detection (single person rowing)
        objects = task_meta.objects
        if not objects or len(objects) == 0:
            return None
        
        detection = objects[0]
        if not isinstance(detection, KeypointObjectWithBbox):
            return None
        
        # Extract rowing-specific keypoints
        keypoints_data = []
        velocities_data = []
        
        for kp_name in self.ROWING_KEYPOINTS:
            kp_idx = self.COCO_KEYPOINT_MAP.get(kp_name)
            if kp_idx is None or kp_idx >= len(detection.keypoints):
                continue
            
            kp = detection.keypoints[kp_idx]
            if len(kp) < 2:
                continue
            
            # Keypoint coordinates (already denormalized by SDK)
            x, y = kp[0], kp[1]
            confidence = kp[2] if len(kp) > 2 else 1.0
            
            keypoints_data.append({
                "name": kp_name,
                "x": int(x),
                "y": int(y),
                "confidence": float(confidence),
                "phase": self.current_phase
            })
            
            # Velocity data (if available from Kalman filtering)
            # Note: Axelera SDK doesn't expose velocity directly in Python yet
            # This would require C++ integration or velocity calculation in Python
            # velocities_data.append({
            #     "name": kp_name,
            #     "vx": 0.0,  # TODO: Extract from C++ Kalman filter
            #     "vy": 0.0,
            #     "speed": 0.0
            # })
        
        if not keypoints_data:
            return None
        
        return FrameKeypointData(
            frame_number=frame_number,
            timestamp=timestamp,
            phase=self.current_phase,
            keypoints=keypoints_data,
            velocities=velocities_data
        )
    
    def add_frame(self, frame_data: Optional[FrameKeypointData]):
        """
        Add frame to circular buffer and handle event-based collection.
        
        Args:
            frame_data: Keypoint data for current frame (None if no detections)
        """
        self.frame_count += 1
        
        if frame_data is None:
            return
        
        # Always maintain circular buffer (last N frames)
        self.keypoint_buffer.append(frame_data)
        
        # Phase-based event collection (similar to rowing stroke detection)
        # Phase 2 = "drive" phase in rowing ergometer
        if self.current_phase == 2:  # Drive phase
            if not self.event_active:
                # Event just started - copy buffer to storage (N frames before event)
                self.event_active = True
                self.event_keypoints = list(self.keypoint_buffer)
                self.post_event_count = 0
                LOG.info(f"Event started at frame {frame_data.frame_number} (phase {self.current_phase})")
            else:
                # Continue collecting during event
                self.event_keypoints.append(frame_data)
        
        elif self.event_active and self.current_phase in (3, 4):  # Post-event phases (dwelling, recovery)
            # Collect N frames after event ends
            if self.post_event_count < self.buffer_size:
                self.event_keypoints.append(frame_data)
                self.post_event_count += 1
            else:
                # Event complete - queue for save
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
            LOG.debug(f"Phase transition: {self.current_phase} -> {phase}")
            self.current_phase = phase
    
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
        filename = os.path.join(self.save_dir, f"stroke_{timestamp}.json.gz")
        
        # Structure data as JSON
        data = {
            "timestamp": timestamp,
            "frame_count": len(self.event_keypoints),
            "keypoints": [
                [kp for kp in frame.keypoints] for frame in self.event_keypoints
            ],
            "velocities": [
                [v for v in frame.velocities] for frame in self.event_keypoints
            ],
            "phases": [frame.phase for frame in self.event_keypoints]
        }
        
        # Merge additional data (e.g., force measurements)
        if additional_data:
            data.update(additional_data)
        
        # Queue for background save
        with self.save_queue_lock:
            self.save_queue.append((data, filename))
        
        LOG.info(f"Queued event save: {len(self.event_keypoints)} frames -> {filename}")
    
    def _save_worker(self):
        """Background worker thread for async saves"""
        while True:
            time.sleep(0.1)  # Check queue every 100ms
            
            with self.save_queue_lock:
                if not self.save_queue:
                    continue
                # Process oldest save first
                data, filename = self.save_queue.popleft()
            
            try:
                # GIL is released during gzip compression and I/O
                with gzip.open(filename, 'wt', compresslevel=3) as f:
                    json.dump(data, f, separators=(',', ':'))
                self.events_saved += 1
                LOG.info(f"Saved event {self.events_saved} to {filename}")
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
