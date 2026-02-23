#!/usr/bin/env python3
# Copyright Axelera AI, 2025
# One Euro Filter for keypoint smoothing - adaptive low-pass filter

"""
One Euro Filter implementation for pose keypoint smoothing.

The One Euro Filter is an adaptive low-pass filter that adjusts its cutoff
frequency based on the speed of movement:
- When stationary: low cutoff = heavy smoothing = no jitter
- When moving fast: high cutoff = light smoothing = responsive tracking

This is the industry standard for real-time pose smoothing, used by:
- Google MediaPipe
- Meta Quest hand tracking
- OpenPose

Reference: "1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems"
           Géry Casiez, Nicolas Roussel, Daniel Vogel (CHI 2012)
           https://gery.casiez.net/1euro/
"""

import numpy as np
from typing import Optional, Tuple
import math

# Constants
MAX_KEYPOINTS = 17
CONFIDENCE_THRESHOLD = 0.3


def smoothing_factor(t_e: float, cutoff: float) -> float:
    """Compute exponential smoothing factor alpha from time constant and cutoff frequency."""
    r = 2.0 * math.pi * cutoff * t_e
    return r / (r + 1.0)


class LowPassFilter:
    """Simple first-order low-pass filter with time-varying alpha."""
    __slots__ = ('x_prev', 'initialized')
    
    def __init__(self):
        self.x_prev = 0.0
        self.initialized = False
    
    def filter(self, x: float, alpha: float) -> float:
        if not self.initialized:
            self.initialized = True
            self.x_prev = x
            return x
        
        # Exponential smoothing: x_filtered = alpha * x + (1 - alpha) * x_prev
        result = alpha * x + (1.0 - alpha) * self.x_prev
        self.x_prev = result
        return result
    
    def reset(self):
        self.initialized = False
        self.x_prev = 0.0


class OneEuroFilter1D:
    """
    One Euro Filter for a single dimension (x or y).
    
    The key insight is that the cutoff frequency adapts to speed:
    - fc = min_cutoff + beta * |dx/dt|
    
    Parameters:
        min_cutoff: Minimum cutoff frequency (Hz). Lower = smoother when stationary.
                   Default: 1.0 Hz
        beta: Speed coefficient. Higher = more responsive to fast movements.
              Default: 0.007
        d_cutoff: Cutoff frequency for derivative computation.
                  Default: 1.0 Hz
    """
    __slots__ = ('min_cutoff', 'beta', 'd_cutoff', 'x_filter', 'dx_filter', 
                 'last_time', 'initialized')
    
    def __init__(self, min_cutoff: float = 0.5, beta: float = 0.007, d_cutoff: float = 0.5):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.last_time = 0.0
        self.initialized = False
    
    def filter(self, x: float, t: float) -> Tuple[float, float]:
        """
        Filter a value and return (filtered_value, velocity).
        
        Args:
            x: Raw measurement
            t: Timestamp in seconds
            
        Returns:
            Tuple of (filtered_x, estimated_velocity)
        """
        if not self.initialized:
            self.initialized = True
            self.last_time = t
            self.x_filter.filter(x, 1.0)  # Initialize with alpha=1 (no smoothing)
            self.dx_filter.filter(0.0, 1.0)
            return x, 0.0
        
        # Compute time delta
        t_e = t - self.last_time
        if t_e <= 0:
            t_e = 1.0 / 60.0  # Default to 60fps
        self.last_time = t
        
        # Estimate derivative (velocity)
        dx = (x - self.x_filter.x_prev) / t_e
        
        # Filter the derivative
        alpha_d = smoothing_factor(t_e, self.d_cutoff)
        dx_filtered = self.dx_filter.filter(dx, alpha_d)
        
        # Compute adaptive cutoff frequency based on speed
        # Faster movement -> higher cutoff -> less smoothing
        cutoff = self.min_cutoff + self.beta * abs(dx_filtered)
        
        # Filter the position with adaptive alpha
        alpha = smoothing_factor(t_e, cutoff)
        x_filtered = self.x_filter.filter(x, alpha)
        
        return x_filtered, dx_filtered
    
    def reset(self):
        self.x_filter.reset()
        self.dx_filter.reset()
        self.initialized = False


class OneEuroFilterOptimized:
    """
    Vectorized One Euro Filter for all 17 COCO keypoints.
    
    Optimized for real-time performance with pre-allocated arrays.
    Uses uniform parameters for all keypoints (pitchpipe-tuned).
    """
    __slots__ = (
        'min_cutoff', 'beta', 'd_cutoff',
        'x', 'y', 'x_prev', 'y_prev',
        'dx', 'dy', 'dx_prev', 'dy_prev',
        'vx', 'vy',  # Velocity estimates
        'initialized', 'last_time',
        '_work_valid',
    )
    
    def __init__(self, min_cutoff: float = 3.5, beta: float = 0.003, d_cutoff: float = 0.5):
        """
        Initialize One Euro Filter with pitchpipe-tuned defaults.
        
        Args:
            min_cutoff: Minimum cutoff frequency (Hz). 
                       Higher = less smoothing, more responsive.
                       Default: 3.5 Hz (lowered from 7.0 for stronger high-freq filtering)
            beta: Speed coefficient. 
                  Higher = more responsive to fast movements.
                  Default: 0.003 (pitchpipe-tuned for rowing lag reduction)
            d_cutoff: Cutoff for derivative estimation.
                      Default: 0.5 Hz (lowered from 1.0 for smoother derivative)
        """
        self.min_cutoff = np.full(MAX_KEYPOINTS, min_cutoff, dtype=np.float32)
        self.beta = np.full(MAX_KEYPOINTS, beta, dtype=np.float32)
        self.d_cutoff = np.full(MAX_KEYPOINTS, d_cutoff, dtype=np.float32)
        
        # Position state
        self.x = np.zeros(MAX_KEYPOINTS, dtype=np.float32)
        self.y = np.zeros(MAX_KEYPOINTS, dtype=np.float32)
        self.x_prev = np.zeros(MAX_KEYPOINTS, dtype=np.float32)
        self.y_prev = np.zeros(MAX_KEYPOINTS, dtype=np.float32)
        
        # Derivative (velocity) state
        self.dx = np.zeros(MAX_KEYPOINTS, dtype=np.float32)
        self.dy = np.zeros(MAX_KEYPOINTS, dtype=np.float32)
        self.dx_prev = np.zeros(MAX_KEYPOINTS, dtype=np.float32)
        self.dy_prev = np.zeros(MAX_KEYPOINTS, dtype=np.float32)
        
        # Velocity output (for JSON recording)
        self.vx = np.zeros(MAX_KEYPOINTS, dtype=np.float32)
        self.vy = np.zeros(MAX_KEYPOINTS, dtype=np.float32)
        
        self.initialized = np.zeros(MAX_KEYPOINTS, dtype=np.bool_)
        self.last_time = np.float64(0.0)
        
        self._work_valid = np.zeros(MAX_KEYPOINTS, dtype=np.bool_)
    
    def update_inplace(self, keypoints: np.ndarray, timestamp: float) -> None:
        """
        Update filter and modify keypoints array IN-PLACE.
        
        Args:
            keypoints: Shape (N, 3) array [x, y, confidence], modified in-place
            timestamp: Frame timestamp in seconds
        """
        n = min(len(keypoints), MAX_KEYPOINTS)
        if n == 0:
            return
        
        # Extract measurements
        mx = keypoints[:n, 0]
        my = keypoints[:n, 1]
        conf = keypoints[:n, 2] if keypoints.shape[1] >= 3 else None
        
        # Determine valid keypoints
        if conf is not None:
            np.greater(conf, CONFIDENCE_THRESHOLD, out=self._work_valid[:n])
        else:
            self._work_valid[:n] = True
        
        # Compute time delta
        if self.last_time > 0:
            t_e = max(0.001, timestamp - self.last_time)
        else:
            t_e = 1.0 / 60.0  # Default 60fps
        self.last_time = timestamp
        
        # Pre-compute smoothing factors
        two_pi = 2.0 * np.pi
        
        for i in range(n):
            if not self._work_valid[i]:
                continue
            
            if not self.initialized[i]:
                # First observation - initialize
                self.x[i] = mx[i]
                self.y[i] = my[i]
                self.x_prev[i] = mx[i]
                self.y_prev[i] = my[i]
                self.dx[i] = 0.0
                self.dy[i] = 0.0
                self.dx_prev[i] = 0.0
                self.dy_prev[i] = 0.0
                self.vx[i] = 0.0
                self.vy[i] = 0.0
                self.initialized[i] = True
                continue
            
            # Compute raw derivative
            raw_dx = (mx[i] - self.x_prev[i]) / t_e
            raw_dy = (my[i] - self.y_prev[i]) / t_e
            
            # Smooth the derivative (using joint-specific d_cutoff)
            r_d = two_pi * self.d_cutoff[i] * t_e
            alpha_d = r_d / (r_d + 1.0)
            self.dx[i] = alpha_d * raw_dx + (1.0 - alpha_d) * self.dx_prev[i]
            self.dy[i] = alpha_d * raw_dy + (1.0 - alpha_d) * self.dy_prev[i]
            
            # Store velocity (in pixels/second)
            self.vx[i] = self.dx[i]
            self.vy[i] = self.dy[i]
            
            # Compute adaptive cutoff based on speed (using joint-specific parameters)
            speed = np.sqrt(self.dx[i]**2 + self.dy[i]**2)
            cutoff = self.min_cutoff[i] + self.beta[i] * speed
            
            # Compute position smoothing factor
            r = two_pi * cutoff * t_e
            alpha = r / (r + 1.0)
            
            # Filter position
            self.x[i] = alpha * mx[i] + (1.0 - alpha) * self.x_prev[i]
            self.y[i] = alpha * my[i] + (1.0 - alpha) * self.y_prev[i]
            
            # Update previous values
            self.x_prev[i] = self.x[i]
            self.y_prev[i] = self.y[i]
            self.dx_prev[i] = self.dx[i]
            self.dy_prev[i] = self.dy[i]
        
        # Write filtered positions back
        keypoints[:n, 0] = np.where(self.initialized[:n], self.x[:n], mx)
        keypoints[:n, 1] = np.where(self.initialized[:n], self.y[:n], my)
    
    def reset(self):
        """Reset all filter state."""
        self.x.fill(0)
        self.y.fill(0)
        self.x_prev.fill(0)
        self.y_prev.fill(0)
        self.dx.fill(0)
        self.dy.fill(0)
        self.dx_prev.fill(0)
        self.dy_prev.fill(0)
        self.vx.fill(0)
        self.vy.fill(0)
        self.initialized.fill(False)
        self.last_time = 0.0


class OneEuroSmoother:
    """
    High-level interface for One Euro Filter smoothing.
    Drop-in replacement for KeypointSmoother.
    """
    __slots__ = ('_filter', '_frame_count')
    
    def __init__(self,
                 min_cutoff: float = 3.5,
                 beta: float = 0.003,
                 d_cutoff: float = 0.5):
        """
        Initialize One Euro Filter smoother with pitchpipe-tuned defaults.
        
        Args:
            min_cutoff: Minimum cutoff frequency (Hz).
                       Default: 3.5 Hz (lowered from 7.0 for stronger high-freq filtering)
            beta: Speed coefficient.
                  Default: 0.003 (pitchpipe-tuned for rowing)
            d_cutoff: Derivative cutoff frequency (Hz).
                     Default: 0.5 Hz (lowered from 1.0 for smoother derivative)
        """
        self._filter = OneEuroFilterOptimized(
            min_cutoff=min_cutoff,
            beta=beta,
            d_cutoff=d_cutoff,
        )
        self._frame_count = 0
    
    def smooth(self, keypoints: np.ndarray, timestamp: float) -> np.ndarray:
        """Smooth keypoints for single person (returns copy)."""
        self._frame_count += 1
        result = keypoints.copy()
        self._filter.update_inplace(result, timestamp)
        return result
    
    def smooth_inplace(self, keypoints: np.ndarray, timestamp: float) -> None:
        """Smooth keypoints in-place (fastest, modifies input)."""
        self._frame_count += 1
        self._filter.update_inplace(keypoints, timestamp)
    
    def reset(self):
        """Reset filter state."""
        self._filter.reset()
        self._frame_count = 0
    
    @property
    def frame_count(self) -> int:
        return self._frame_count
    
    @property
    def filter(self):
        """Access underlying filter for parameter inspection."""
        return self._filter
    
    def get_velocities(self, n_keypoints: int = 17) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current velocity estimates.
        
        Returns:
            Tuple of (vx_array, vy_array) in pixels/second
        """
        n = min(n_keypoints, MAX_KEYPOINTS)
        return (
            self._filter.vx[:n].copy(),
            self._filter.vy[:n].copy()
        )
    
    def get_velocity(self, keypoint_idx: int) -> Tuple[float, float]:
        """
        Get velocity for a specific keypoint.
        
        Returns:
            Tuple of (vx, vy) in pixels/second
        """
        if keypoint_idx >= MAX_KEYPOINTS or not self._filter.initialized[keypoint_idx]:
            return (0.0, 0.0)
        return (
            float(self._filter.vx[keypoint_idx]),
            float(self._filter.vy[keypoint_idx])
        )


# ============================================================================
# Quick test
# ============================================================================
if __name__ == "__main__":
    import time
    
    print("=" * 60)
    print("One Euro Filter Test")
    print("=" * 60)
    
    # Create test data with jitter
    np.random.seed(42)
    
    smoother = OneEuroSmoother(min_cutoff=1.0, beta=0.007)
    
    # Simulate stationary keypoints with noise
    print("\n[Stationary with noise]")
    base_pos = np.array([[320.0, 240.0, 0.9]] * 17, dtype=np.float32)
    
    for i in range(20):
        # Add ±3 pixel noise
        noisy = base_pos.copy()
        noisy[:, :2] += np.random.randn(17, 2).astype(np.float32) * 3
        
        smoother.smooth_inplace(noisy, i / 60.0)
        
        if i % 5 == 0:
            print(f"  Frame {i}: raw=(320±3), filtered=({noisy[0,0]:.1f}, {noisy[0,1]:.1f})")
    
    # Simulate moving keypoints
    print("\n[Moving keypoints]")
    smoother.reset()
    
    for i in range(30):
        # Move right at 10 pixels/frame
        pos = np.array([[100.0 + i * 10, 240.0, 0.9]] * 17, dtype=np.float32)
        # Add noise
        pos[:, :2] += np.random.randn(17, 2).astype(np.float32) * 3
        
        smoother.smooth_inplace(pos, i / 60.0)
        vx, vy = smoother.get_velocities(1)
        
        if i % 5 == 0:
            expected_x = 100.0 + i * 10
            print(f"  Frame {i}: expected_x={expected_x:.0f}, filtered=({pos[0,0]:.1f}), vx={vx[0]:.1f} px/s")
    
    # Performance benchmark
    print("\n[Performance Benchmark]")
    smoother.reset()
    keypoints = np.random.rand(17, 3).astype(np.float32) * 640
    keypoints[:, 2] = 0.9
    
    n_frames = 10000
    start = time.perf_counter()
    for i in range(n_frames):
        kpts = keypoints + np.random.randn(17, 3).astype(np.float32) * 5
        kpts[:, 2] = 0.9
        smoother.smooth_inplace(kpts, i / 60.0)
    elapsed = time.perf_counter() - start
    
    print(f"  {n_frames} frames in {elapsed*1000:.1f}ms")
    print(f"  Per-frame: {elapsed/n_frames*1e6:.1f}µs")
    print(f"  Max FPS: {n_frames/elapsed:.0f}")
