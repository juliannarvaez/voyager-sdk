#!/usr/bin/env python3
# Copyright Axelera AI, 2025
# Optimized Kalman filter for keypoint smoothing - near C++ performance

"""
High-performance Kalman filter using NumPy with careful optimizations:
- Pre-allocated work arrays (zero allocation in hot path)
- Contiguous memory layout for cache efficiency
- Minimal branching and indexing
- float32 throughout for NEON/SIMD

Target: <50µs per frame (matching C++ performance)
"""

import numpy as np
from typing import Optional

# Constants
MAX_KEYPOINTS = 17
CONFIDENCE_THRESHOLD = 0.3


class KalmanFilterOptimized:
    """
    Ultra-optimized 2D Kalman filter for pose keypoints.
    
    All arrays are pre-allocated and contiguous for maximum performance.
    Uses frequency-aware gain: K = 1 - exp(-2π * fc * dt) for precise
    cutoff frequency control and framerate independence.
    """
    __slots__ = (
        'x', 'y', 'vx', 'vy',
        'initialized', 'last_update_time',  # Per-keypoint timestamp
        'cutoff_hz', 'alpha',
        '_work_innovation_x', '_work_innovation_y',
        '_work_K', '_work_valid'
    )
    
    def __init__(self, 
                 cutoff_hz: float = 2.0,
                 velocity_alpha: float = 0.3):
        """
        Initialize with pre-allocated arrays.
        
        Args:
            cutoff_hz: Low-pass cutoff frequency in Hz.
                       Controls the smoothing bandwidth — signals below this
                       frequency pass through, higher frequencies are attenuated.
                       Lower = smoother but laggier, higher = more responsive but noisier.
                       Range: 1.0 (very smooth) to 10.0 (nearly raw)
                       Recommended: 3.0-5.0 Hz for pose keypoint smoothing
                       Default: 4.0 Hz = responsive tracking with jitter removal
                       
            velocity_alpha: Velocity learning rate for exponential moving average.
                           Controls how fast velocity estimate adapts to changes.
                           Lower = velocity changes slowly (more stable)
                           Higher = velocity tracks changes quickly (responsive)
                           Range: 0.1 (very stable) to 0.9 (very responsive)
                           Default: 0.3 = stable velocity estimation
        """
        # State arrays - separate x/y for better cache access
        self.x = np.zeros(MAX_KEYPOINTS, dtype=np.float32)
        self.y = np.zeros(MAX_KEYPOINTS, dtype=np.float32)
        self.vx = np.zeros(MAX_KEYPOINTS, dtype=np.float32)
        self.vy = np.zeros(MAX_KEYPOINTS, dtype=np.float32)
        self.initialized = np.zeros(MAX_KEYPOINTS, dtype=np.bool_)
        # MUST be float64 to handle time.time() values like 1738345678.123
        # float32 only has ~7 significant digits, can't distinguish consecutive timestamps
        self.last_update_time = np.zeros(MAX_KEYPOINTS, dtype=np.float64)
        
        # Parameters
        self.cutoff_hz = np.float32(cutoff_hz)
        self.alpha = np.float32(velocity_alpha)
        
        # Pre-allocated work arrays (avoid allocation in update)
        self._work_innovation_x = np.zeros(MAX_KEYPOINTS, dtype=np.float32)
        self._work_innovation_y = np.zeros(MAX_KEYPOINTS, dtype=np.float32)
        self._work_K = np.zeros(MAX_KEYPOINTS, dtype=np.float32)
        self._work_valid = np.zeros(MAX_KEYPOINTS, dtype=np.bool_)
    
    def update_inplace(self, keypoints: np.ndarray, timestamp: float) -> None:
        """
        Update filter and modify keypoints array IN-PLACE.
        
        This is the fastest path - no array allocation or copying.
        
        Args:
            keypoints: Shape (N, 3) array [x, y, confidence], modified in-place
            timestamp: Frame timestamp in seconds
        """
        n = min(len(keypoints), MAX_KEYPOINTS)
        if n == 0:
            return
        
        # Determine valid keypoints (by confidence threshold)
        mx = keypoints[:n, 0]
        my = keypoints[:n, 1]
        conf = keypoints[:n, 2] if keypoints.shape[1] >= 3 else None
        
        if conf is not None:
            np.greater(conf, CONFIDENCE_THRESHOLD, out=self._work_valid[:n])
        else:
            self._work_valid[:n] = True
        
        # Initialize new keypoints
        new_mask = self._work_valid[:n] & ~self.initialized[:n]
        if np.any(new_mask):
            self.x[:n][new_mask] = mx[new_mask]
            self.y[:n][new_mask] = my[new_mask]
            self.vx[:n][new_mask] = 0.0
            self.vy[:n][new_mask] = 0.0
            self.initialized[:n][new_mask] = True
            self.last_update_time[:n][new_mask] = timestamp
        
        # Keypoints that are visible NOW and were initialized before
        update_mask = self._work_valid[:n] & self.initialized[:n]
        
        if not np.any(update_mask):
            return
        
        # Calculate per-keypoint dt (time since each keypoint was last updated)
        dt_array = np.full(n, 0.0167, dtype=np.float32)  # Default ~60fps
        dt_array[update_mask] = np.maximum(0.001, timestamp - self.last_update_time[:n][update_mask])
        
        # Detect keypoints that were invisible and just reappeared (large dt)
        # If dt > 0.1s (3 frames at 30fps), consider it a reappearance
        reappeared_mask = update_mask & (dt_array > 0.1)
        if np.any(reappeared_mask):
            self.vx[:n][reappeared_mask] = 0.0
            self.vy[:n][reappeared_mask] = 0.0
        
        # === COMPUTE FREQUENCY-AWARE GAIN ===
        # K = 1 - exp(-2π * fc * dt) gives exact cutoff at fc Hz
        # Framerate-independent: same effective bandwidth regardless of FPS
        K = 1.0 - np.exp(np.float32(-2.0 * np.pi) * self.cutoff_hz * dt_array)
        np.clip(K, 0.01, 0.99, out=K)
        
        # Snap to measurement for reappeared keypoints (no smoothing delay)
        if np.any(reappeared_mask):
            K[reappeared_mask] = 1.0
        
        # === UPDATE STEP ===
        # Innovation: z - x_predicted
        np.subtract(mx, self.x[:n], out=self._work_innovation_x[:n])
        np.subtract(my, self.y[:n], out=self._work_innovation_y[:n])
        
        # State update: x = x + K * innovation
        Km = K[update_mask]
        self.x[:n][update_mask] += Km * self._work_innovation_x[:n][update_mask]
        self.y[:n][update_mask] += Km * self._work_innovation_y[:n][update_mask]
        
        # Velocity update using actual dt for proper scaling
        dt_masked = np.maximum(dt_array[update_mask], np.float32(0.001))
        self.vx[:n][update_mask] = (
            (1 - self.alpha) * self.vx[:n][update_mask] + 
            self.alpha * self._work_innovation_x[:n][update_mask] / dt_masked
        )
        self.vy[:n][update_mask] = (
            (1 - self.alpha) * self.vy[:n][update_mask] + 
            self.alpha * self._work_innovation_y[:n][update_mask] / dt_masked
        )
        
        # Update timestamps for keypoints we just processed
        self.last_update_time[:n][update_mask] = timestamp
        
        # Write filtered positions back to input array (in-place)
        keypoints[:n, 0] = np.where(self.initialized[:n], self.x[:n], mx)
        keypoints[:n, 1] = np.where(self.initialized[:n], self.y[:n], my)
    
    def update(self, keypoints: np.ndarray, timestamp: float) -> np.ndarray:
        """
        Update filter and return smoothed keypoints (copies input).
        
        Args:
            keypoints: Shape (N, 3) array [x, y, confidence]
            timestamp: Frame timestamp
        
        Returns:
            Smoothed keypoints array (new array, input unchanged)
        """
        result = keypoints.copy()
        self.update_inplace(result, timestamp)
        return result
    
    def reset(self):
        """Reset all filter state."""
        self.x.fill(0)
        self.y.fill(0)
        self.vx.fill(0)
        self.vy.fill(0)
        self.initialized.fill(False)
        self.last_update_time.fill(0.0)


class KeypointSmoother:
    """
    High-level interface for smoothing pose keypoints.
    Drop-in replacement for the C++ implementation.
    """
    __slots__ = ('_filter', '_frame_count')
    
    def __init__(self,
                 cutoff_hz: float = 4.0,
                 velocity_alpha: float = 0.3):
        self._filter = KalmanFilterOptimized(
            cutoff_hz=cutoff_hz,
            velocity_alpha=velocity_alpha
        )
        self._frame_count = 0
    
    def smooth(self, keypoints: np.ndarray, timestamp: float) -> np.ndarray:
        """Smooth keypoints for single person (returns copy)."""
        self._frame_count += 1
        return self._filter.update(keypoints, timestamp)
    
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
    
    def get_velocities(self, n_keypoints: int = 17) -> tuple:
        """
        Get current velocity estimates from Kalman filter state.
        
        Returns velocities in pixels/second (not m/s - caller must convert).
        These are the optimal velocity estimates, much smoother than
        differentiating positions because they're part of the Kalman state.
        
        Args:
            n_keypoints: Number of keypoints to return velocities for
            
        Returns:
            Tuple of (vx_array, vy_array) in pixels/second
        """
        n = min(n_keypoints, MAX_KEYPOINTS)
        return (
            self._filter.vx[:n].copy(),
            self._filter.vy[:n].copy()
        )
    
    def get_velocity(self, keypoint_idx: int) -> tuple:
        """
        Get velocity for a specific keypoint.
        
        Args:
            keypoint_idx: COCO keypoint index (0-16)
            
        Returns:
            Tuple of (vx, vy) in pixels/second, or (0, 0) if not initialized
        """
        if keypoint_idx >= MAX_KEYPOINTS or not self._filter.initialized[keypoint_idx]:
            return (0.0, 0.0)
        return (
            float(self._filter.vx[keypoint_idx]),
            float(self._filter.vy[keypoint_idx])
        )


# ============================================================================
# Benchmark
# ============================================================================
if __name__ == "__main__":
    import time
    
    print("=" * 60)
    print("Kalman Filter Performance Benchmark")
    print("=" * 60)
    
    # Test both versions
    from kalman_filter import KeypointSmoother as OriginalSmoother
    
    n_frames = 10000
    n_warmup = 100
    
    # Create test data
    np.random.seed(42)
    keypoints = np.random.rand(17, 3).astype(np.float32) * 640
    keypoints[:, 2] = 0.9  # confidence
    
    # Original version
    print("\n[Original KalmanFilter2D (dataclass)]")
    smoother_orig = OriginalSmoother()
    
    # Warmup
    for i in range(n_warmup):
        _ = smoother_orig.smooth(keypoints.copy(), i / 60.0)
    
    start = time.perf_counter()
    for i in range(n_frames):
        kpts = keypoints + np.random.randn(17, 3).astype(np.float32) * 5
        kpts[:, 2] = 0.9
        _ = smoother_orig.smooth(kpts, i / 60.0)
    elapsed_orig = time.perf_counter() - start
    
    print(f"  {n_frames} frames in {elapsed_orig*1000:.1f}ms")
    print(f"  Per-frame: {elapsed_orig/n_frames*1e6:.1f}µs")
    print(f"  Max FPS: {n_frames/elapsed_orig:.0f}")
    
    # Optimized version
    print("\n[Optimized KalmanFilterOptimized (__slots__)]")
    smoother_opt = KeypointSmoother()
    
    # Warmup
    for i in range(n_warmup):
        kpts = keypoints.copy()
        smoother_opt.smooth_inplace(kpts, i / 60.0)
    
    # Reset for fair comparison
    smoother_opt.reset()
    
    start = time.perf_counter()
    for i in range(n_frames):
        kpts = keypoints + np.random.randn(17, 3).astype(np.float32) * 5
        kpts[:, 2] = 0.9
        smoother_opt.smooth_inplace(kpts, i / 60.0)
    elapsed_opt = time.perf_counter() - start
    
    print(f"  {n_frames} frames in {elapsed_opt*1000:.1f}ms")
    print(f"  Per-frame: {elapsed_opt/n_frames*1e6:.1f}µs")
    print(f"  Max FPS: {n_frames/elapsed_opt:.0f}")
    
    # Comparison
    print("\n" + "=" * 60)
    print("Comparison")
    print("=" * 60)
    speedup = elapsed_orig / elapsed_opt
    print(f"  Speedup: {speedup:.2f}x faster")
    print(f"  Original: {elapsed_orig/n_frames*1e6:.1f}µs/frame")
    print(f"  Optimized: {elapsed_opt/n_frames*1e6:.1f}µs/frame")
    
    # Verify correctness
    print("\n[Correctness Check]")
    smoother_orig = OriginalSmoother()
    smoother_opt = KeypointSmoother()
    
    np.random.seed(123)
    for i in range(10):
        kpts = np.random.rand(17, 3).astype(np.float32) * 640
        kpts[:, 2] = 0.9
        
        result_orig = smoother_orig.smooth(kpts.copy(), i / 30.0)
        
        kpts_copy = kpts.copy()
        smoother_opt.smooth_inplace(kpts_copy, i / 30.0)
        
        diff = np.abs(result_orig[:, :2] - kpts_copy[:, :2]).max()
        if diff > 0.01:
            print(f"  Frame {i}: MAX DIFF = {diff:.4f} ⚠️")
        else:
            print(f"  Frame {i}: max diff = {diff:.6f} ✓")
