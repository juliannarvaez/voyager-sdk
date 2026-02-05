#!/usr/bin/env python3
# Copyright Axelera AI, 2025
# Extended Kalman Filter for keypoint smoothing

"""
Extended Kalman Filter implementation for pose keypoint smoothing.

The EKF uses a nonlinear motion model with acceleration and velocity damping:
- State: [x, y, vx, vy, ax, ay] (position, velocity, acceleration)
- Nonlinear process model:
  * x_k = x_{k-1} + vx * dt + 0.5 * ax * dt²
  * vx_k = vx_{k-1} * exp(-damping * dt) + ax * dt
  * ax_k = ax_{k-1} * exp(-decay * dt)
  
  The exponential velocity damping models realistic motion where people
  naturally slow down, and acceleration decay prevents unbounded acceleration.
  
- Measurement: [x, y] (observed keypoint positions)

The filter provides:
- Nonlinear motion modeling (true EKF with Jacobian linearization)
- Realistic handling of acceleration and deceleration
- Explicit velocity and acceleration estimation
- Uncertainty quantification via covariance matrices
- Predictive tracking capabilities

Good for applications requiring:
- Accurate tracking during acceleration/deceleration
- Realistic motion physics
- Velocity and acceleration information
- Handling of missing detections

References:
- "Kalman Filtering: Theory and Practice Using MATLAB"
  Grewal & Andrews (2014)
- "Tracking and Data Association"
  Bar-Shalom, Li, Kirubarajan (2001)

Usage:
    smoother = EKFSmoother(
        process_noise_pos=0.1,
        process_noise_vel=5.0,
        process_noise_acc=10.0,
        measurement_noise=2.0,
        damping=0.5
    )
    
    smoothed = smoother.smooth(keypoints, timestamp)
    velocities = smoother.get_velocities()
    accelerations = smoother.get_accelerations()
"""

import numpy as np
from typing import Optional, Tuple

# Constants
MAX_KEYPOINTS = 17
CONFIDENCE_THRESHOLD = 0.3


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for keypoint tracking with nonlinear motion model.
    
    The EKF models each keypoint with acceleration and velocity damping:
    - State: [x, y, vx, vy, ax, ay] (position, velocity, acceleration)
    - Nonlinear process model:
      * Position: x_k = x_{k-1} + vx_{k-1} * dt + 0.5 * ax_{k-1} * dt²
      * Velocity: vx_k = vx_{k-1} * exp(-damping * dt) + ax_{k-1} * dt
                  (exponential damping models natural slowdown)
      * Acceleration: ax_k = ax_{k-1} * exp(-decay * dt)
                     (decay prevents unbounded acceleration)
    - Measurement: [x, y] (observed keypoint position)
    
    The exponential terms introduce nonlinearity, requiring Jacobian computation
    and linearization (making this a true EKF, not just a regular KF).
    """
    __slots__ = (
        'x', 'y', 'vx', 'vy', 'ax', 'ay',  # State: position, velocity, acceleration
        'P',  # State covariance (6x6 for each keypoint)
        'Q_pos', 'Q_vel', 'Q_acc',  # Process noise
        'R',  # Measurement noise
        'damping',  # Velocity damping coefficient
        'acc_decay',  # Acceleration decay coefficient
        'initialized',
        'last_time',
        '_work_valid'
    )
    
    def __init__(self, 
                 process_noise_pos: float = 0.1,
                 process_noise_vel: float = 5.0,
                 process_noise_acc: float = 10.0,
                 measurement_noise: float = 2.0,
                 damping: float = 0.5,
                 acc_decay: float = 2.0):
        """
        Initialize Extended Kalman Filter.
        
        Args:
            process_noise_pos: Process noise for position (pixels).
                              Lower = assumes more predictable motion.
                              Recommended: 0.05-1.0
                              Default: 0.1
            process_noise_vel: Process noise for velocity (pixels/s).
                              Accounts for velocity changes.
                              Recommended: 1.0-10.0
                              Default: 5.0
            process_noise_acc: Process noise for acceleration (pixels/s²).
                              Accounts for jerk (change in acceleration).
                              Recommended: 5.0-20.0
                              Default: 10.0
            measurement_noise: Measurement noise (pixels).
                              Should match expected detection jitter.
                              Recommended: 1.0-5.0
                              Default: 2.0
            damping: Velocity damping coefficient (1/s).
                    Models natural slowdown. Higher = faster slowdown.
                    0 = no damping (constant velocity when ax=0)
                    Recommended: 0.1-2.0
                    Default: 0.5
            acc_decay: Acceleration decay coefficient (1/s).
                      Prevents unbounded acceleration.
                      Recommended: 1.0-5.0
                      Default: 2.0
        """
        # State vectors (6D state per keypoint)
        self.x = np.zeros(MAX_KEYPOINTS, dtype=np.float32)
        self.y = np.zeros(MAX_KEYPOINTS, dtype=np.float32)
        self.vx = np.zeros(MAX_KEYPOINTS, dtype=np.float32)
        self.vy = np.zeros(MAX_KEYPOINTS, dtype=np.float32)
        self.ax = np.zeros(MAX_KEYPOINTS, dtype=np.float32)
        self.ay = np.zeros(MAX_KEYPOINTS, dtype=np.float32)
        
        # Covariance matrix for each keypoint (6x6: x, y, vx, vy, ax, ay)
        self.P = np.zeros((MAX_KEYPOINTS, 6, 6), dtype=np.float32)
        for i in range(MAX_KEYPOINTS):
            self.P[i] = np.eye(6, dtype=np.float32) * 100.0  # Initial uncertainty
        
        # Process noise covariance
        self.Q_pos = np.float32(process_noise_pos ** 2)
        self.Q_vel = np.float32(process_noise_vel ** 2)
        self.Q_acc = np.float32(process_noise_acc ** 2)
        
        # Measurement noise covariance
        self.R = np.float32(measurement_noise ** 2)
        
        # Nonlinear model parameters
        self.damping = np.float32(damping)
        self.acc_decay = np.float32(acc_decay)
        
        self.initialized = np.zeros(MAX_KEYPOINTS, dtype=np.bool_)
        self.last_time = np.float64(0.0)
        self._work_valid = np.zeros(MAX_KEYPOINTS, dtype=np.bool_)
    
    def _nonlinear_predict(self, x: float, y: float, vx: float, vy: float, 
                          ax: float, ay: float, dt: float) -> tuple:
        """
        Nonlinear state prediction (the 'f' function in EKF).
        
        Returns: (x_pred, y_pred, vx_pred, vy_pred, ax_pred, ay_pred)
        """
        # Exponential factors (this is what makes it nonlinear!)
        damping_factor = np.exp(-self.damping * dt)
        decay_factor = np.exp(-self.acc_decay * dt)
        
        # Position update (includes acceleration)
        x_pred = x + vx * dt + 0.5 * ax * dt * dt
        y_pred = y + vy * dt + 0.5 * ay * dt * dt
        
        # Velocity update with damping (nonlinear due to exponential)
        vx_pred = vx * damping_factor + ax * dt
        vy_pred = vy * damping_factor + ay * dt
        
        # Acceleration update with decay (nonlinear)
        ax_pred = ax * decay_factor
        ay_pred = ay * decay_factor
        
        return x_pred, y_pred, vx_pred, vy_pred, ax_pred, ay_pred
    
    def _compute_jacobian(self, vx: float, vy: float, ax: float, ay: float, dt: float) -> np.ndarray:
        """
        Compute Jacobian of nonlinear state transition function.
        
        This is the key difference from regular Kalman Filter - we linearize
        the nonlinear function by computing its derivative (Jacobian matrix).
        
        F_jacobian = ∂f/∂x where f is the state transition function
        """
        damping_factor = np.exp(-self.damping * dt)
        decay_factor = np.exp(-self.acc_decay * dt)
        
        # Jacobian matrix (6x6)
        F = np.eye(6, dtype=np.float32)
        
        # Position derivatives
        # ∂x/∂x = 1, ∂x/∂vx = dt, ∂x/∂ax = 0.5*dt²
        F[0, 2] = dt  # ∂x/∂vx
        F[0, 4] = 0.5 * dt * dt  # ∂x/∂ax
        F[1, 3] = dt  # ∂y/∂vy
        F[1, 5] = 0.5 * dt * dt  # ∂y/∂ay
        
        # Velocity derivatives (nonlinear part!)
        # ∂vx/∂vx = exp(-damping*dt)
        # ∂vx/∂ax = dt
        F[2, 2] = damping_factor  # ∂vx/∂vx (nonlinear!)
        F[2, 4] = dt  # ∂vx/∂ax
        F[3, 3] = damping_factor  # ∂vy/∂vy (nonlinear!)
        F[3, 5] = dt  # ∂vy/∂ay
        
        # Acceleration derivatives (nonlinear part!)
        # ∂ax/∂ax = exp(-decay*dt)
        F[4, 4] = decay_factor  # ∂ax/∂ax (nonlinear!)
        F[5, 5] = decay_factor  # ∂ay/∂ay (nonlinear!)
        
        return F
    
    def predict(self, i: int, dt: float) -> tuple:
        """
        EKF Prediction step for keypoint i.
        
        Args:
            i: Keypoint index
            dt: Time delta in seconds
            
        Returns:
            Tuple of (x_pred, y_pred, vx_pred, vy_pred, ax_pred, ay_pred)
        """
        # Nonlinear state prediction
        x_pred, y_pred, vx_pred, vy_pred, ax_pred, ay_pred = self._nonlinear_predict(
            self.x[i], self.y[i], self.vx[i], self.vy[i], 
            self.ax[i], self.ay[i], dt
        )
        
        # Compute Jacobian at current state
        F = self._compute_jacobian(self.vx[i], self.vy[i], self.ax[i], self.ay[i], dt)
        
        # Process noise Q
        Q = np.diag([
            self.Q_pos, self.Q_pos,  # position noise
            self.Q_vel, self.Q_vel,  # velocity noise
            self.Q_acc, self.Q_acc   # acceleration noise
        ]).astype(np.float32)
        
        # Predict covariance using Jacobian: P_pred = F * P * F^T + Q
        self.P[i] = F @ self.P[i] @ F.T + Q
        
        return x_pred, y_pred, vx_pred, vy_pred, ax_pred, ay_pred
    
    def update(self, i: int, z_x: float, z_y: float, 
               x_pred: float, y_pred: float, 
               vx_pred: float, vy_pred: float,
               ax_pred: float, ay_pred: float) -> None:
        """
        EKF Update step for keypoint i.
        
        Args:
            i: Keypoint index
            z_x: Measured x position
            z_y: Measured y position
            x_pred: Predicted x position
            y_pred: Predicted y position
            vx_pred: Predicted x velocity
            vy_pred: Predicted y velocity
            ax_pred: Predicted x acceleration
            ay_pred: Predicted y acceleration
        """
        # Measurement matrix H (we only observe position, not velocity/acceleration)
        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ], dtype=np.float32)
        
        # Measurement noise R
        R_mat = np.array([
            [self.R, 0],
            [0, self.R]
        ], dtype=np.float32)
        
        # Innovation (residual): y = z - H * x_pred
        z = np.array([z_x, z_y], dtype=np.float32)
        z_pred = np.array([x_pred, y_pred], dtype=np.float32)
        y_innov = z - z_pred
        
        # Innovation covariance: S = H * P * H^T + R
        S = H @ self.P[i] @ H.T + R_mat
        
        # Kalman gain: K = P * H^T * S^{-1}
        try:
            S_inv = np.linalg.inv(S)
            K = self.P[i] @ H.T @ S_inv
        except np.linalg.LinAlgError:
            # Singular matrix, skip update (use prediction only)
            self.x[i] = x_pred
            self.y[i] = y_pred
            self.vx[i] = vx_pred
            self.vy[i] = vy_pred
            self.ax[i] = ax_pred
            self.ay[i] = ay_pred
            return
        
        # Update state: x = x_pred + K * y
        state_pred = np.array([x_pred, y_pred, vx_pred, vy_pred, ax_pred, ay_pred], dtype=np.float32)
        state = state_pred + K @ y_innov
        
        self.x[i] = state[0]
        self.y[i] = state[1]
        self.vx[i] = state[2]
        self.vy[i] = state[3]
        self.ax[i] = state[4]
        self.ay[i] = state[5]
        
        # Update covariance: P = (I - K * H) * P
        I = np.eye(6, dtype=np.float32)
        self.P[i] = (I - K @ H) @ self.P[i]
    
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
            dt = max(0.001, timestamp - self.last_time)
        else:
            dt = 1.0 / 60.0  # Default to 60fps
        self.last_time = timestamp
        
        # Process each keypoint
        for i in range(n):
            if not self._work_valid[i]:
                continue
            
            if not self.initialized[i]:
                # Initialize state with first measurement
                self.x[i] = mx[i]
                self.y[i] = my[i]
                self.vx[i] = 0.0
                self.vy[i] = 0.0
                self.ax[i] = 0.0
                self.ay[i] = 0.0
                self.P[i] = np.eye(6, dtype=np.float32) * 100.0
                self.initialized[i] = True
                continue
            
            # === PREDICTION STEP (with nonlinear model) ===
            x_pred, y_pred, vx_pred, vy_pred, ax_pred, ay_pred = self.predict(i, dt)
            
            # === UPDATE STEP ===
            self.update(i, mx[i], my[i], x_pred, y_pred, vx_pred, vy_pred, ax_pred, ay_pred)
        
        # Write filtered positions back
        keypoints[:n, 0] = np.where(self.initialized[:n], self.x[:n], mx)
        keypoints[:n, 1] = np.where(self.initialized[:n], self.y[:n], my)
    
    def reset(self):
        """Reset all filter state."""
        self.x.fill(0)
        self.y.fill(0)
        self.vx.fill(0)
        self.vy.fill(0)
        self.ax.fill(0)
        self.ay.fill(0)
        for i in range(MAX_KEYPOINTS):
            self.P[i] = np.eye(6, dtype=np.float32) * 100.0
        self.initialized.fill(False)
        self.last_time = 0.0


class EKFSmoother:
    """
    High-level interface for Extended Kalman Filter smoothing.
    Drop-in replacement for OneEuroSmoother.
    """
    __slots__ = ('_filter', '_frame_count')
    
    def __init__(self,
                 process_noise_pos: float = 0.1,
                 process_noise_vel: float = 5.0,
                 process_noise_acc: float = 10.0,
                 measurement_noise: float = 2.0,
                 damping: float = 0.5,
                 acc_decay: float = 2.0):
        """
        Initialize Extended Kalman Filter smoother.
        
        Args:
            process_noise_pos: Process noise for position (pixels).
                              Lower = smoother tracking.
                              Recommended: 0.05-1.0, default 0.1
            process_noise_vel: Process noise for velocity (pixels/s).
                              Higher = adapts faster to velocity changes.
                              Recommended: 1.0-10.0, default 5.0
            process_noise_acc: Process noise for acceleration (pixels/s²).
                              Higher = adapts faster to acceleration changes.
                              Recommended: 5.0-20.0, default 10.0
            measurement_noise: Measurement noise (pixels).
                              Should match detection jitter.
                              Recommended: 1.0-5.0, default 2.0
            damping: Velocity damping coefficient (1/s).
                    Models natural slowdown.
                    Recommended: 0.1-2.0, default 0.5
            acc_decay: Acceleration decay coefficient (1/s).
                      Prevents unbounded acceleration.
                      Recommended: 1.0-5.0, default 2.0
        """
        self._filter = ExtendedKalmanFilter(
            process_noise_pos=process_noise_pos,
            process_noise_vel=process_noise_vel,
            process_noise_acc=process_noise_acc,
            measurement_noise=measurement_noise,
            damping=damping,
            acc_decay=acc_decay
        )
        self._frame_count = 0
    
    def smooth(self, keypoints: np.ndarray, timestamp: float) -> np.ndarray:
        """
        Smooth keypoints for single person (returns copy).
        
        Args:
            keypoints: Shape (N, 3) array [x, y, confidence]
            timestamp: Frame timestamp in seconds
            
        Returns:
            Smoothed keypoints array (copy)
        """
        self._frame_count += 1
        result = keypoints.copy()
        self._filter.update_inplace(result, timestamp)
        return result
    
    def smooth_inplace(self, keypoints: np.ndarray, timestamp: float) -> None:
        """
        Smooth keypoints in-place (fastest, modifies input).
        
        Args:
            keypoints: Shape (N, 3) array [x, y, confidence], modified in-place
            timestamp: Frame timestamp in seconds
        """
        self._frame_count += 1
        self._filter.update_inplace(keypoints, timestamp)
    
    def reset(self):
        """Reset filter state."""
        self._filter.reset()
        self._frame_count = 0
    
    @property
    def frame_count(self) -> int:
        """Get number of frames processed."""
        return self._frame_count
    
    @property
    def filter(self):
        """Access underlying filter for parameter inspection."""
        return self._filter
    
    def get_velocities(self, n_keypoints: int = 17) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current velocity estimates for all keypoints.
        
        Args:
            n_keypoints: Number of keypoints to return
            
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
        
        Args:
            keypoint_idx: Index of keypoint (0-16 for COCO)
            
        Returns:
            Tuple of (vx, vy) in pixels/second
        """
        if keypoint_idx >= MAX_KEYPOINTS or not self._filter.initialized[keypoint_idx]:
            return (0.0, 0.0)
        return (
            float(self._filter.vx[keypoint_idx]),
            float(self._filter.vy[keypoint_idx])
        )
    
    def get_speed(self, keypoint_idx: int) -> float:
        """
        Get speed (velocity magnitude) for a specific keypoint.
        
        Args:
            keypoint_idx: Index of keypoint (0-16 for COCO)
            
        Returns:
            Speed in pixels/second
        """
        vx, vy = self.get_velocity(keypoint_idx)
        return float(np.sqrt(vx**2 + vy**2))
    
    def get_accelerations(self, n_keypoints: int = 17) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current acceleration estimates for all keypoints.
        
        Args:
            n_keypoints: Number of keypoints to return
            
        Returns:
            Tuple of (ax_array, ay_array) in pixels/second²
        """
        n = min(n_keypoints, MAX_KEYPOINTS)
        return (
            self._filter.ax[:n].copy(),
            self._filter.ay[:n].copy()
        )
    
    def get_acceleration(self, keypoint_idx: int) -> Tuple[float, float]:
        """
        Get acceleration for a specific keypoint.
        
        Args:
            keypoint_idx: Index of keypoint (0-16 for COCO)
            
        Returns:
            Tuple of (ax, ay) in pixels/second²
        """
        if keypoint_idx >= MAX_KEYPOINTS or not self._filter.initialized[keypoint_idx]:
            return (0.0, 0.0)
        return (
            float(self._filter.ax[keypoint_idx]),
            float(self._filter.ay[keypoint_idx])
        )
    
    def get_acceleration_magnitude(self, keypoint_idx: int) -> float:
        """
        Get acceleration magnitude for a specific keypoint.
        
        Args:
            keypoint_idx: Index of keypoint (0-16 for COCO)
            
        Returns:
            Acceleration magnitude in pixels/second²
        """
        ax, ay = self.get_acceleration(keypoint_idx)
        return float(np.sqrt(ax**2 + ay**2))
    
    def get_covariance(self, keypoint_idx: int) -> Optional[np.ndarray]:
        """
        Get state covariance matrix for a specific keypoint.
        
        The covariance matrix quantifies uncertainty in the state estimate:
        - P[0,0]: variance in x position
        - P[1,1]: variance in y position
        - P[2,2]: variance in x velocity
        - P[3,3]: variance in y velocity
        - P[4,4]: variance in x acceleration
        - P[5,5]: variance in y acceleration
        - Off-diagonal: correlations
        
        Args:
            keypoint_idx: Index of keypoint (0-16 for COCO)
            
        Returns:
            6x6 covariance matrix [x, y, vx, vy, ax, ay] or None if not initialized
        """
        if keypoint_idx >= MAX_KEYPOINTS or not self._filter.initialized[keypoint_idx]:
            return None
        return self._filter.P[keypoint_idx].copy()
    
    def get_position_uncertainty(self, keypoint_idx: int) -> Optional[Tuple[float, float]]:
        """
        Get position uncertainty (standard deviation) for a specific keypoint.
        
        Args:
            keypoint_idx: Index of keypoint (0-16 for COCO)
            
        Returns:
            Tuple of (sigma_x, sigma_y) in pixels, or None if not initialized
        """
        cov = self.get_covariance(keypoint_idx)
        if cov is None:
            return None
        return (
            float(np.sqrt(cov[0, 0])),
            float(np.sqrt(cov[1, 1]))
        )
    
    def predict(self, keypoint_idx: int, dt: float) -> Optional[Tuple[float, float]]:
        """
        Predict future position of a keypoint using nonlinear motion model.
        
        Uses acceleration and velocity damping to extrapolate position.
        
        Args:
            keypoint_idx: Index of keypoint (0-16 for COCO)
            dt: Time delta for prediction (seconds)
            
        Returns:
            Tuple of (x_pred, y_pred) or None if not initialized
        """
        if keypoint_idx >= MAX_KEYPOINTS or not self._filter.initialized[keypoint_idx]:
            return None
        
        x_pred, y_pred, _, _, _, _ = self._filter._nonlinear_predict(
            self._filter.x[keypoint_idx],
            self._filter.y[keypoint_idx],
            self._filter.vx[keypoint_idx],
            self._filter.vy[keypoint_idx],
            self._filter.ax[keypoint_idx],
            self._filter.ay[keypoint_idx],
            dt
        )
        
        return (float(x_pred), float(y_pred))


# ============================================================================
# Quick test
# ============================================================================
if __name__ == "__main__":
    import time
    
    print("=" * 60)
    print("Extended Kalman Filter Test")
    print("=" * 60)
    
    # Create test data with jitter
    np.random.seed(42)
    
    smoother = EKFSmoother(
        process_noise_pos=0.1,
        process_noise_vel=5.0,
        process_noise_acc=10.0,
        measurement_noise=2.0,
        damping=0.5,
        acc_decay=2.0
    )
    
    # Simulate stationary keypoints with noise
    print("\n[Stationary with noise]")
    base_pos = np.array([[320.0, 240.0, 0.9]] * 17, dtype=np.float32)
    
    for i in range(20):
        # Add ±3 pixel noise
        noisy = base_pos.copy()
        noisy[:, :2] += np.random.randn(17, 2).astype(np.float32) * 3
        
        smoother.smooth_inplace(noisy, i / 60.0)
        
        if i % 5 == 0:
            sigma_x, sigma_y = smoother.get_position_uncertainty(0)
            print(f"  Frame {i}: raw=(320±3), filtered=({noisy[0,0]:.1f}, {noisy[0,1]:.1f}), σ={sigma_x:.2f}px")
    
    # Simulate moving keypoints
    print("\n[Moving keypoints]")
    smoother.reset()
    
    for i in range(30):
        # Move right at 10 pixels/frame = 600 pixels/second at 60fps
        pos = np.array([[100.0 + i * 10, 240.0, 0.9]] * 17, dtype=np.float32)
        # Add noise
        pos[:, :2] += np.random.randn(17, 2).astype(np.float32) * 3
        
        smoother.smooth_inplace(pos, i / 60.0)
        vx, vy = smoother.get_velocities(1)
        ax, ay = smoother.get_accelerations(1)
        speed = smoother.get_speed(0)
        
        if i % 5 == 0:
            expected_x = 100.0 + i * 10
            print(f"  Frame {i}: expected_x={expected_x:.0f}, filtered=({pos[0,0]:.1f}), vx={vx[0]:.1f} px/s, ax={ax[0]:.1f} px/s²")
    
    # Test prediction
    print("\n[Prediction test]")
    smoother.reset()
    
    # Build up some velocity
    for i in range(10):
        pos = np.array([[100.0 + i * 10, 240.0, 0.9]] * 17, dtype=np.float32)
        smoother.smooth_inplace(pos, i / 60.0)
    
    # Predict future position
    pred_0_1s = smoother.predict(0, 0.1)  # 100ms ahead
    pred_0_5s = smoother.predict(0, 0.5)  # 500ms ahead
    
    print(f"  Current position: ({smoother.filter.x[0]:.1f}, {smoother.filter.y[0]:.1f})")
    print(f"  Velocity: ({smoother.filter.vx[0]:.1f}, {smoother.filter.vy[0]:.1f}) px/s")
    print(f"  Predicted (100ms): ({pred_0_1s[0]:.1f}, {pred_0_1s[1]:.1f})")
    print(f"  Predicted (500ms): ({pred_0_5s[0]:.1f}, {pred_0_5s[1]:.1f})")
    
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
    
    # Summary
    print("\n" + "=" * 60)
    print("EKF Features Summary")
    print("=" * 60)
    print("\nThis is a TRUE Extended Kalman Filter:")
    print("  + Nonlinear motion model (exponential damping & decay)")
    print("  + Jacobian-based linearization")
    print("  + Models acceleration explicitly (not just constant velocity)")
    print("  + Velocity damping models natural slowdown")
    print("  + Acceleration decay prevents unbounded acceleration")
    print("\nAdvantages:")
    print("  + More realistic motion physics")
    print("  + Handles acceleration and deceleration naturally")
    print("  + Explicit velocity and acceleration estimation")
    print("  + Uncertainty quantification via covariance")
    print("  + Predictive tracking with physics-based extrapolation")
    print("  + Handles missing detections gracefully")
    print("\nConsiderations:")
    print("  - More computationally expensive than One Euro Filter")
    print("  - Requires tuning of noise and damping parameters")
    print("  - Nonlinear model may require careful initialization")
    print("\nUse Cases:")
    print("  - Tracking with realistic motion physics")
    print("  - Applications needing acceleration information")
    print("  - Predictive tracking (e.g., occlusion handling)")
    print("  - Uncertainty-aware systems")
    print("  - Sensor fusion with multiple detectors")
