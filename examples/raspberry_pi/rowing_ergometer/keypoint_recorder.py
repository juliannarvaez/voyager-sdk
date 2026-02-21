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
        
        # For raw velocity computation when no filter is used
        self.prev_keypoints = None  # Previous frame keypoints for finite difference
        self.prev_timestamp = None  # Previous frame timestamp
        
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
                # Use TaskMeta.keypoints (same array that Kalman filter smooths)
                # NOT Detection.keypoints which may be a different array
                if hasattr(tmeta, 'keypoints') and tmeta.keypoints is not None:
                    task_meta = tmeta
                    break
        
        if not task_meta:
            return None
        
        # Use TaskMeta.keypoints directly (already smoothed by Kalman filter)
        # This is the same array that smoother.smooth_inplace() modifies
        all_keypoints = task_meta.keypoints
        if all_keypoints is None or len(all_keypoints) == 0:
            return None
        
        # Get first person's keypoints (already smoothed)
        keypoints = all_keypoints[0]
        if len(keypoints) == 0:
            return None
        
        # Extract rowing-specific keypoints (optimized - minimal allocations)
        # Note: Keypoints are already smoothed by main loop's KeypointSmoother
        keypoints_data = []
        # Use current_phase that was set by set_phase()
        phase = self.current_phase
        
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
                # Store as FLOAT to preserve sub-pixel precision from Kalman filter
                # Integer truncation causes ±1px jitter → ±1 m/s velocity noise at 100fps
                keypoints_data.append({
                    "name": name,
                    "x": round(float(kp[0]), 2),  # 2 decimal places = 0.01px precision
                    "y": round(float(kp[1]), 2),
                    "confidence": float(kp[2]) if len(kp) > 2 else 1.0,
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
    
    def extract_keypoints_with_velocity(self, meta, width: int, height: int, 
                                         frame_number: int, timestamp: float,
                                         smoother) -> Optional[FrameKeypointData]:
        """
        Extract keypoint data including Kalman filter velocities.
        
        This is the preferred method - velocities from Kalman state are much 
        more accurate than computing velocity from position differences.
        
        Args:
            meta: AxMeta container with task metas
            width: Frame width
            height: Frame height
            frame_number: Current frame number
            timestamp: Frame timestamp
            smoother: KeypointSmoother instance to get velocities from
            
        Returns:
            FrameKeypointData with velocity fields, or None if no keypoints
        """
        # Fast path: try to get task meta directly
        task_meta = None
        if hasattr(meta, 'values'):
            for tmeta in meta.values():
                if hasattr(tmeta, 'keypoints') and tmeta.keypoints is not None:
                    task_meta = tmeta
                    break
        
        if not task_meta:
            return None
        
        all_keypoints = task_meta.keypoints
        if all_keypoints is None or len(all_keypoints) == 0:
            return None
        
        keypoints = all_keypoints[0]
        if len(keypoints) == 0:
            return None
        
        # Get velocities from filter (if enabled), otherwise compute raw finite difference
        if smoother is not None:
            vx_all, vy_all = smoother.get_velocities(len(keypoints))
        else:
            # No filter - compute raw velocities using finite differences
            vx_all = []
            vy_all = []
            
            if self.prev_keypoints is not None and self.prev_timestamp is not None:
                dt = timestamp - self.prev_timestamp
                if dt > 0:
                    for idx in range(len(keypoints)):
                        if idx < len(self.prev_keypoints) and len(keypoints[idx]) >= 2 and len(self.prev_keypoints[idx]) >= 2:
                            # Raw velocity from position difference
                            vx = (keypoints[idx][0] - self.prev_keypoints[idx][0]) / dt
                            vy = (keypoints[idx][1] - self.prev_keypoints[idx][1]) / dt
                            vx_all.append(vx)
                            vy_all.append(vy)
                        else:
                            vx_all.append(0.0)
                            vy_all.append(0.0)
                else:
                    vx_all = [0.0] * len(keypoints)
                    vy_all = [0.0] * len(keypoints)
            else:
                # First frame - no previous data
                vx_all = [0.0] * len(keypoints)
                vy_all = [0.0] * len(keypoints)
            
            # Store current keypoints for next frame
            self.prev_keypoints = [kp.copy() if len(kp) >= 2 else kp for kp in keypoints]
            self.prev_timestamp = timestamp
        
        keypoints_data = []
        phase = self.current_phase
        self.last_timestamp = timestamp
        
        # Capture ALL 17 COCO keypoints, not just rowing-specific ones
        coco_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        
        for idx, name in enumerate(coco_names):
            if idx < len(keypoints) and len(keypoints[idx]) >= 2:
                kp = keypoints[idx]
                # Include Kalman velocity estimates (much smoother than differentiation)
                keypoints_data.append({
                    "name": name,
                    "x": round(float(kp[0]), 2),
                    "y": round(float(kp[1]), 2),
                    "vx": round(float(vx_all[idx]), 2) if idx < len(vx_all) else 0.0,  # px/s
                    "vy": round(float(vy_all[idx]), 2) if idx < len(vy_all) else 0.0,  # px/s
                    "confidence": float(kp[2]) if len(kp) > 2 else 1.0,
                    "phase": phase
                })
        
        if not keypoints_data:
            return None
        
        return FrameKeypointData(
            frame_number=frame_number,
            timestamp=timestamp,
            phase=self.current_phase,
            keypoints=keypoints_data
        )
    
    def add_frame(self, frame_data: Optional[FrameKeypointData]):
        """
        Add frame to circular buffer and handle event-based collection.
        
        Collects: buffer_size pre-drive frames + all drive frames + buffer_size
        post-drive frames.  Then queues a lightweight reference to the save
        worker — all data assembly, force data polling, JSON serialization,
        and file I/O happen on the background save worker thread (cores 0-1).
        
        Args:
            frame_data: Keypoint data for current frame (None if no detections)
        """
        if frame_data is None:
            return
        
        self.frame_count += 1
        
        # Always maintain circular buffer (last N frames)
        self.keypoint_buffer.append(frame_data)
        
        phase = self.current_phase
        
        if phase == 2:  # Drive phase
            if not self.event_active:
                # Start new stroke collection
                self.event_active = True
                self.event_keypoints = list(self.keypoint_buffer)
                self.post_event_count = 0
                LOG.info(f"Stroke event started - buffered {len(self.event_keypoints)} pre-drive frames")
            else:
                # Continue collecting during drive
                self.event_keypoints.append(frame_data)
        
        elif self.event_active:
            # Drive just ended — collect post-drive frames
            self.event_keypoints.append(frame_data)
            self.post_event_count += 1
            if self.post_event_count >= self.buffer_size:
                # 30 post-drive frames collected — hand off to save worker
                LOG.info(f"Stroke complete: {len(self.event_keypoints)} frames, queuing save")
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
        """Queue a lightweight reference for the save worker — near-zero inference impact.
        
        Only swaps the list reference and appends to the deque (~microseconds).
        All heavy work (data assembly, force-data polling, JSON serialization,
        file I/O) runs on the background save worker thread (cores 0-1).
        """
        timestamp = int(time.time())
        filename = os.path.join(self.save_dir, f"stroke_{timestamp}.json")
        
        # Swap the list reference (near-instant)
        event_snapshot = self.event_keypoints
        pc = self.phase_controller
        
        with self.save_queue_lock:
            self.save_queue.append((event_snapshot, filename, timestamp, pc, additional_data))
        
        LOG.debug(f"Queued save: {len(event_snapshot)} frames -> {filename}")
    
    def _save_worker(self):
        """Background save worker — assembles data, waits for force, serializes, writes.

        Runs on cores 0-1 to keep inference cores 2-3 free.  Uses
        multiprocessing.Event (wait_for_stroke_data) to block until the
        polling process signals that force data is in shared memory.
        No polling, no race, no missed data.
        """
        try:
            import os as _os
            _os.sched_setaffinity(0, {0, 1})
            LOG.debug("Save worker thread pinned to cores 0-1")
        except Exception:
            pass
        while True:
            time.sleep(0.05)  # 50ms poll for queue items

            with self.save_queue_lock:
                if not self.save_queue:
                    continue
                event_snapshot, filename, timestamp, pc, additional_data = self.save_queue.popleft()

            try:
                # Assemble keypoint data
                data = {
                    "timestamp": timestamp,
                    "frame_count": len(event_snapshot),
                    "keypoints": [f.keypoints for f in event_snapshot],
                    "phases": [f.phase for f in event_snapshot],
                    "frame_timestamps": [f.timestamp for f in event_snapshot],
                }

                # Wait for force data via multiprocessing.Event (set by polling process)
                if pc is not None:
                    stroke_data = pc.wait_for_stroke_data(timeout=5.0)

                    force_data = stroke_data.get('force', [])
                    if force_data:
                        data['force'] = force_data

                    drag_factor = stroke_data.get('drag_factor')
                    if drag_factor is not None:
                        data['drag_factor'] = drag_factor

                    stroke_stats = stroke_data.get('stroke_stats', {})
                    if stroke_stats:
                        data['stroke_stats'] = stroke_stats

                    pc.clear_force_data()
                else:
                    force_data = self.get_force_data()
                    if force_data:
                        data['force'] = force_data

                if additional_data:
                    data.update(additional_data)

                # Serialize + write (write syscall releases GIL)
                json_bytes = json.dumps(data, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
                with open(filename, 'wb', buffering=65536) as f:
                    f.write(json_bytes)

                self.events_saved += 1
                force_count = len(data.get('force', []))
                LOG.info(f"Saved stroke {self.events_saved}: {len(event_snapshot)} frames, "
                         f"{force_count} force samples -> {filename}")
            except Exception as e:
                LOG.error(f"Error saving stroke: {e}")
                import traceback
                traceback.print_exc()
    
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
