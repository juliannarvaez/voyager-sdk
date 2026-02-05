#!/usr/bin/env python3
# Copyright Axelera AI, 2025
# Rowing Stroke Data Visualizer

"""
Analyze and visualize recorded stroke data with interactive plots.

Usage:
    python3 analyze_strokes.py              # Interactive visualization
    python3 analyze_strokes.py --text-only  # Text summary only
"""

import os
import sys
import json
import gzip
import glob
import argparse
import numpy as np
from typing import List

# Try to import matplotlib - if not available, run in text-only mode
try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed - running in text-only mode")
    print("Install with: pip3 install matplotlib --user")

# Moving average window size
MOVING_AVG_WINDOW = 3

# Data folder
DATA_FOLDER = "/tmp/stroke_data"
FILE_PATTERN = "stroke_*.json"

current_index = [0]


def compute_frequency_response(timestamps, velocity):
    """
    Compute frequency response using FFT.
    
    Args:
        timestamps: Array of timestamps (seconds)
        velocity: Array of velocity values (px/s)
        
    Returns:
        frequencies (Hz), power spectral density (dB)
    """
    # Compute sampling rate
    dt = np.mean(np.diff(timestamps))
    fs = 1.0 / dt  # Sampling frequency
    
    # Compute FFT
    n = len(velocity)
    fft_values = np.fft.rfft(velocity)
    fft_freq = np.fft.rfftfreq(n, dt)
    
    # Power spectral density in dB
    psd = np.abs(fft_values) ** 2
    psd_db = 10 * np.log10(psd + 1e-12)  # Add epsilon to avoid log(0)
    
    return fft_freq, psd_db


def compute_angle(a, b, c):
    """Compute angle at point b formed by a-b-c.
    
    Args:
        a, b, c: Tuples of (x, y) coordinates or dicts with 'x' and 'y' keys
    """
    # Handle both tuple and dict formats
    if isinstance(a, tuple):
        ax, ay = a
        bx, by = b
        cx, cy = c
    else:
        ax, ay = a['x'], a['y']
        bx, by = b['x'], b['y']
        cx, cy = c['x'], c['y']
    
    v1 = np.array([ax - bx, ay - by])
    v2 = np.array([cx - bx, cy - by])
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return np.nan
    cos_theta = np.dot(v1, v2) / (norm1 * norm2)
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return float(np.degrees(angle_rad))


def moving_average(data, window_size=3):
    """Apply moving window average to smooth data."""
    if len(data) == 0:
        return np.array([])
    
    data_array = np.array(data)
    if window_size <= 1:
        return data_array
    
    result = np.copy(data_array).astype(float)
    
    for i in range(len(data_array)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(data_array), i + window_size // 2 + 1)
        window_data = data_array[start_idx:end_idx]
        
        if np.all(np.isnan(window_data)):
            result[i] = np.nan
        else:
            result[i] = np.nanmean(window_data)
    
    return result


def compute_velocity_savgol(positions, timestamps=None, fps=60, pixels_per_meter=100.0, window=15, polyorder=2):
    """Compute velocity using Savitzky-Golay-style polynomial derivative.
    
    This method fits a polynomial to a window of points and computes the analytical
    derivative, resulting in much smoother velocity estimates than finite differences.
    Uses numpy polyfit (no scipy dependency).
    
    Args:
        positions: Array of position values (x or y coordinates)
        timestamps: Optional array of actual timestamps (uses these if provided)
        fps: Frames per second (used if timestamps not provided)
        pixels_per_meter: Calibration factor (default 100 px/m)
        window: Window size for polynomial fitting (must be odd, default 15 = ~150ms at 100fps)
        polyorder: Polynomial order (default 2 for quadratic fit)
    
    Returns:
        Array of velocities in meters/second
    """
    positions = np.array(positions, dtype=float)
    n = len(positions)
    
    if n < 3:
        return compute_velocity_central(positions, timestamps, fps, pixels_per_meter)
    
    # Ensure window is odd and not larger than data
    window = min(window, n)
    if window % 2 == 0:
        window -= 1
    window = max(3, window)
    
    # Handle NaN values by interpolation
    valid_mask = ~np.isnan(positions)
    if not np.any(valid_mask):
        return np.full(n, np.nan)
    
    # Interpolate NaN values for fitting
    positions_interp = np.interp(
        np.arange(n),
        np.where(valid_mask)[0],
        positions[valid_mask]
    )
    
    # Build time array
    if timestamps is not None and len(timestamps) == n:
        t = np.array(timestamps)
    else:
        t = np.arange(n) / fps
    
    # Compute velocity at each point using local polynomial fit
    velocities = np.zeros(n)
    half_window = window // 2
    
    for i in range(n):
        # Define window bounds
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)
        
        # Extract window data
        t_window = t[start:end]
        p_window = positions_interp[start:end]
        
        if len(t_window) < 3:
            # Not enough points, use simple difference
            if i == 0:
                dt = t[1] - t[0] if n > 1 else 1/fps
                velocities[i] = (positions_interp[1] - positions_interp[0]) / dt if dt > 0 else 0
            else:
                dt = t[i] - t[i-1]
                velocities[i] = (positions_interp[i] - positions_interp[i-1]) / dt if dt > 0 else 0
            continue
        
        # Center time for numerical stability
        t_center = t[i]
        t_local = t_window - t_center
        
        # Fit polynomial (quadratic by default)
        try:
            # polyfit returns coefficients highest power first
            # For quadratic: [a, b, c] where p(t) = a*t^2 + b*t + c
            # Derivative: p'(t) = 2*a*t + b
            # At t=0 (center): p'(0) = b
            order = min(polyorder, len(t_window) - 1)
            coeffs = np.polyfit(t_local, p_window, order)
            
            # Velocity is the coefficient of the linear term (second from last)
            # For quadratic [a,b,c]: derivative at t=0 is b
            if order >= 1:
                velocities[i] = coeffs[-2]  # Linear coefficient
            else:
                velocities[i] = 0
        except (np.linalg.LinAlgError, ValueError):
            # Fallback to central difference
            if i > 0 and i < n - 1:
                dt = t[i+1] - t[i-1]
                velocities[i] = (positions_interp[i+1] - positions_interp[i-1]) / dt if dt > 0 else 0
            elif i == 0 and n > 1:
                dt = t[1] - t[0]
                velocities[i] = (positions_interp[1] - positions_interp[0]) / dt if dt > 0 else 0
            else:
                velocities[i] = 0
    
    # Convert to m/s
    velocities = velocities / pixels_per_meter
    
    # Restore NaN where original was NaN
    velocities[~valid_mask] = np.nan
    
    return velocities


def compute_velocity_central(positions, timestamps=None, fps=60, pixels_per_meter=100.0):
    """Compute velocity using central differences (more accurate than forward diff).
    
    Central differences: v[i] = (x[i+1] - x[i-1]) / (2*dt)
    This is second-order accurate vs first-order for forward differences.
    
    Args:
        positions: Array of position values (x or y coordinates)
        timestamps: Optional array of actual timestamps
        fps: Frames per second (used if timestamps not provided)
        pixels_per_meter: Calibration factor (default 100 px/m)
    
    Returns:
        Array of velocities in meters/second
    """
    positions = np.array(positions, dtype=float)
    n = len(positions)
    if n < 2:
        return np.array([])
    
    velocities = np.zeros(n)
    
    if timestamps is not None and len(timestamps) == n:
        # Use actual timestamps for each interval
        timestamps = np.array(timestamps)
        
        # Central differences for interior points
        for i in range(1, n - 1):
            dt = timestamps[i + 1] - timestamps[i - 1]
            if dt > 0:
                velocities[i] = (positions[i + 1] - positions[i - 1]) / dt
            else:
                velocities[i] = np.nan
        
        # Forward difference for first point
        dt0 = timestamps[1] - timestamps[0]
        if dt0 > 0:
            velocities[0] = (positions[1] - positions[0]) / dt0
        else:
            velocities[0] = velocities[1] if n > 1 else 0
        
        # Backward difference for last point
        dt_last = timestamps[-1] - timestamps[-2]
        if dt_last > 0:
            velocities[-1] = (positions[-1] - positions[-2]) / dt_last
        else:
            velocities[-1] = velocities[-2] if n > 1 else 0
    else:
        # Use constant fps
        dt = 1.0 / fps
        
        # Central differences for interior points
        velocities[1:-1] = (positions[2:] - positions[:-2]) / (2 * dt)
        
        # Forward/backward for endpoints
        velocities[0] = (positions[1] - positions[0]) / dt if n > 1 else 0
        velocities[-1] = (positions[-1] - positions[-2]) / dt if n > 1 else 0
    
    # Convert to m/s
    return velocities / pixels_per_meter


def compute_velocity(positions, fps=60, pixels_per_meter=100.0):
    """Compute velocity from position data (legacy forward differences).
    
    Args:
        positions: Array of position values (x or y coordinates)
        fps: Frames per second (default 60, will be overridden by detected FPS)
        pixels_per_meter: Calibration factor (default 100 px/m = 1m per 100 pixels)
    
    Returns:
        Array of velocities in meters/second
    """
    if len(positions) < 2:
        return np.array([])
    
    # Compute finite differences in pixels/second, then convert to m/s
    velocities = np.diff(positions) * fps / pixels_per_meter
    # Pad to match original length (duplicate last value)
    velocities = np.concatenate([velocities, [velocities[-1] if len(velocities) > 0 else 0]])
    return velocities


def load_stroke_file(filepath: str) -> dict:
    """Load a stroke JSON file (gzipped or plain)."""
    try:
        with gzip.open(filepath, 'rt') as f:
            return json.load(f)
    except gzip.BadGzipFile:
        with open(filepath, 'r') as f:
            return json.load(f)


def visualize_keypoints(filepath: str, frame_indices: List[int] = None):
    """Visualize keypoint skeleton for selected frames."""
    if not HAS_MATPLOTLIB:
        print("ERROR: matplotlib not available. Install with: pip3 install matplotlib")
        return None
    
    data = load_stroke_file(filepath)
    
    # Handle both old Python format and new C++ format
    frames_array = data.get('frames', [])
    if frames_array:
        # New C++ format
        keypoints_list = [f.get('keypoints', []) for f in frames_array]
        phases = [f.get('phase', 0) for f in frames_array]
    else:
        # Old Python format
        keypoints_list = data.get('keypoints', [])
        phases = data.get('phases', [])
    
    if not keypoints_list:
        print("No keypoints found in file")
        return None
    
    # Define skeleton connections
    connections = [
        ('left_shoulder', 'left_hip'),
        ('left_hip', 'left_knee'),
        ('left_knee', 'left_ankle'),
        ('right_shoulder', 'right_hip'),
        ('right_hip', 'right_knee'),
        ('right_knee', 'right_ankle'),
        ('left_shoulder', 'right_shoulder'),
        ('left_hip', 'right_hip'),
    ]
    
    phase_names = {0: "IDLE", 1: "WAIT_ACCEL", 2: "DRIVE", 3: "DWELLING", 4: "RECOVERY"}
    phase_colors = {0: 'gray', 1: 'yellow', 2: 'red', 3: 'blue', 4: 'green'}
    
    # If no specific frames, select key frames from each phase
    if frame_indices is None:
        frame_indices = []
        seen_phases = set()
        for i, phase in enumerate(phases):
            if phase not in seen_phases and phase in [2, 4]:  # Drive and Recovery
                frame_indices.append(i)
                seen_phases.add(phase)
        
        # Add some mid-points
        if len(keypoints_list) > 0:
            frame_indices.extend([0, len(keypoints_list) // 4, len(keypoints_list) // 2, 
                                 3 * len(keypoints_list) // 4, len(keypoints_list) - 1])
        frame_indices = sorted(list(set(frame_indices)))[:6]  # Max 6 frames
    
    # Create subplot grid
    n_frames = len(frame_indices)
    cols = min(3, n_frames)
    rows = (n_frames + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    if n_frames == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if rows > 1 else list(axes)
    
    fig.suptitle(f'Keypoint Skeleton: {os.path.basename(filepath)}', fontsize=14)
    
    for idx, frame_idx in enumerate(frame_indices):
        if frame_idx >= len(keypoints_list):
            continue
            
        ax = axes[idx]
        frame_kpts = keypoints_list[frame_idx]
        phase = phases[frame_idx] if frame_idx < len(phases) else 0
        
        # Build keypoint dict
        kp_dict = {}
        for kp in frame_kpts:
            kp_dict[kp['name']] = (kp['x'], kp['y'])
        
        # Draw connections
        for conn_start, conn_end in connections:
            if conn_start in kp_dict and conn_end in kp_dict:
                x1, y1 = kp_dict[conn_start]
                x2, y2 = kp_dict[conn_end]
                ax.plot([x1, x2], [y1, y2], 'b-', linewidth=2, alpha=0.6)
        
        # Draw keypoints
        for name, (x, y) in kp_dict.items():
            ax.plot(x, y, 'ro', markersize=8)
            ax.text(x, y-15, name.replace('left_', 'L_').replace('right_', 'R_'), 
                   fontsize=8, ha='center', va='bottom')
        
        # Configure plot
        ax.set_xlim(0, 640)
        ax.set_ylim(480, 0)  # Inverted Y axis
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Frame {frame_idx} - {phase_names.get(phase, f"Phase{phase}")}', 
                    fontsize=10, color=phase_colors.get(phase, 'black'))
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
    
    # Hide empty subplots
    for idx in range(len(frame_indices), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig


def parse_stroke_file(filepath: str):
    """Parse stroke file and extract angles and positions."""
    data = load_stroke_file(filepath)
    
    # Handle both old Python format and new C++ format
    frames_array = data.get('frames', [])
    if frames_array:
        # New C++ format: frames array with timestamp, keypoints, phase per frame
        frame_timestamps = [f.get('timestamp', 0) for f in frames_array]
        keypoints_list = [f.get('keypoints', []) for f in frames_array]
        phases = [f.get('phase', 0) for f in frames_array]
    else:
        # Old Python format: top-level keypoints, phases, frame_timestamps arrays
        frame_timestamps = data.get('frame_timestamps', [])
        keypoints_list = data.get('keypoints', [])
        phases = data.get('phases', [])
    
    # Detect actual FPS from timestamps
    detected_fps = 60.0  # Default fallback
    if len(frame_timestamps) >= 10:
        intervals = [frame_timestamps[i+1] - frame_timestamps[i] for i in range(min(100, len(frame_timestamps)-1))]
        valid_intervals = [x for x in intervals if x > 0]
        if valid_intervals:
            avg_interval = sum(valid_intervals) / len(valid_intervals)
            detected_fps = 1.0 / avg_interval if avg_interval > 0 else 60.0
    
    # Initialize data lists
    knee_angles = []
    hip_angles = []
    shoulder_y = []
    hip_y = []
    knee_y = []
    ankle_y = []
    shoulder_x = []
    hip_x = []
    knee_x = []
    ankle_x = []
    # Wrist data
    left_wrist_x = []
    left_wrist_y = []
    right_wrist_x = []
    right_wrist_y = []
    left_elbow_x = []
    left_elbow_y = []
    right_elbow_x = []
    right_elbow_y = []
    left_wrist_vx_kalman = []
    left_wrist_vy_kalman = []
    right_wrist_vx_kalman = []
    right_wrist_vy_kalman = []
    left_elbow_vx_kalman = []
    left_elbow_vy_kalman = []
    right_elbow_vx_kalman = []
    right_elbow_vy_kalman = []
    # Kalman filter velocities (if available in JSON)
    shoulder_vx_kalman = []
    shoulder_vy_kalman = []
    hip_vx_kalman = []
    hip_vy_kalman = []
    knee_vx_kalman = []
    knee_vy_kalman = []
    ankle_vx_kalman = []
    ankle_vy_kalman = []
    has_kalman_velocity = False
    timestamps = []  # Real timestamps for FPS analysis
    
    # Perspective correction parameters
    # Camera is pointed at center of erg (hips at drive start)
    # Need to find the camera focal point (hip position at drive start)
    focal_point_x = None
    
    # First pass: find drive start to get focal point
    for i, (frame_kpts, phase) in enumerate(zip(keypoints_list, phases)):
        if phase == 2:  # DRIVE phase start
            kp_dict_temp = {}
            for kp in frame_kpts:
                kp_dict_temp[kp['name']] = (kp['x'], kp['y'])
            hip_coords_temp = kp_dict_temp.get('left_hip') or kp_dict_temp.get('right_hip')
            if hip_coords_temp:
                focal_point_x = hip_coords_temp[0]
                break
    
    # Fallback to center of first frame if no drive detected
    if focal_point_x is None and keypoints_list:
        kp_dict_temp = {}
        for kp in keypoints_list[0]:
            kp_dict_temp[kp['name']] = (kp['x'], kp['y'])
        hip_coords_temp = kp_dict_temp.get('left_hip') or kp_dict_temp.get('right_hip')
        if hip_coords_temp:
            focal_point_x = hip_coords_temp[0]
        else:
            focal_point_x = 320  # Default camera center for 640px width
    
    def apply_perspective_correction(x, y, focal_x, correction_strength=0.0003):
        """
        Apply perspective correction to account for lens focal effect.
        Points closer to camera (lower x in rowing) appear larger.
        Points further from camera (higher x) appear smaller.
        Distortion increases non-linearly with distance from focal point.
        
        Args:
            x, y: Keypoint coordinates
            focal_x: X coordinate of camera focal point (hips at drive start)
            correction_strength: Strength of perspective effect (tune based on camera)
        
        Returns:
            Corrected x, y coordinates
        """
        # Distance from focal point
        dx = x - focal_x
        
        # Non-linear perspective scale factor
        # Distortion increases quadratically with distance
        # This models the pinhole camera effect more accurately
        distortion = dx * correction_strength
        scale_factor = 1.0 + distortion + (distortion * abs(distortion) * 0.5)
        
        # Apply correction to horizontal displacement from focal point
        x_corrected = focal_x + dx * scale_factor
        
        return x_corrected, y
    
    # Process each frame
    for frame_kpts in keypoints_list:
        # Build keypoint dict for this frame (include velocity if present)
        kp_dict = {}
        kp_vel_dict = {}
        for kp in frame_kpts:
            # Apply perspective correction to keypoint positions
            x_corrected, y_corrected = apply_perspective_correction(kp['x'], kp['y'], focal_point_x)
            kp_dict[kp['name']] = (x_corrected, y_corrected)
            # Check for Kalman velocities (vx, vy in px/s)
            # Velocities also need perspective correction (distortion increases with distance)
            if 'vx' in kp and 'vy' in kp:
                # Velocity correction: scale velocities by same non-linear perspective factor
                dx = kp['x'] - focal_point_x
                distortion = dx * 0.0003
                scale_factor = 1.0 + distortion + (distortion * abs(distortion) * 0.5)
                kp_vel_dict[kp['name']] = (kp['vx'] * scale_factor, kp['vy'] * scale_factor)
                has_kalman_velocity = True
        
        # Extract coordinates
        shoulder_coords = kp_dict.get('left_shoulder') or kp_dict.get('right_shoulder')
        hip_coords = kp_dict.get('left_hip') or kp_dict.get('right_hip')
        knee_coords = kp_dict.get('left_knee') or kp_dict.get('right_knee')
        ankle_coords = kp_dict.get('left_ankle') or kp_dict.get('right_ankle')
        
        # Extract wrist and elbow coordinates (both left and right)
        left_wrist_coords = kp_dict.get('left_wrist')
        right_wrist_coords = kp_dict.get('right_wrist')
        left_elbow_coords = kp_dict.get('left_elbow')
        right_elbow_coords = kp_dict.get('right_elbow')
        
        # Extract Kalman velocities if available
        shoulder_vel = kp_vel_dict.get('left_shoulder') or kp_vel_dict.get('right_shoulder')
        hip_vel = kp_vel_dict.get('left_hip') or kp_vel_dict.get('right_hip')
        knee_vel = kp_vel_dict.get('left_knee') or kp_vel_dict.get('right_knee')
        ankle_vel = kp_vel_dict.get('left_ankle') or kp_vel_dict.get('right_ankle')
        left_wrist_vel = kp_vel_dict.get('left_wrist')
        right_wrist_vel = kp_vel_dict.get('right_wrist')
        left_elbow_vel = kp_vel_dict.get('left_elbow')
        right_elbow_vel = kp_vel_dict.get('right_elbow')
        
        # Store wrist positions
        left_wrist_x.append(left_wrist_coords[0] if left_wrist_coords else np.nan)
        left_wrist_y.append(left_wrist_coords[1] if left_wrist_coords else np.nan)
        right_wrist_x.append(right_wrist_coords[0] if right_wrist_coords else np.nan)
        right_wrist_y.append(right_wrist_coords[1] if right_wrist_coords else np.nan)
        
        # Store elbow positions
        left_elbow_x.append(left_elbow_coords[0] if left_elbow_coords else np.nan)
        left_elbow_y.append(left_elbow_coords[1] if left_elbow_coords else np.nan)
        right_elbow_x.append(right_elbow_coords[0] if right_elbow_coords else np.nan)
        right_elbow_y.append(right_elbow_coords[1] if right_elbow_coords else np.nan)
        
        # Store wrist velocities
        left_wrist_vx_kalman.append(left_wrist_vel[0] if left_wrist_vel else np.nan)
        left_wrist_vy_kalman.append(left_wrist_vel[1] if left_wrist_vel else np.nan)
        right_wrist_vx_kalman.append(right_wrist_vel[0] if right_wrist_vel else np.nan)
        right_wrist_vy_kalman.append(right_wrist_vel[1] if right_wrist_vel else np.nan)
        
        # Store elbow velocities
        left_elbow_vx_kalman.append(left_elbow_vel[0] if left_elbow_vel else np.nan)
        left_elbow_vy_kalman.append(left_elbow_vel[1] if left_elbow_vel else np.nan)
        right_elbow_vx_kalman.append(right_elbow_vel[0] if right_elbow_vel else np.nan)
        right_elbow_vy_kalman.append(right_elbow_vel[1] if right_elbow_vel else np.nan)
        
        # Store Kalman velocities (px/s)
        shoulder_vx_kalman.append(shoulder_vel[0] if shoulder_vel else np.nan)
        shoulder_vy_kalman.append(shoulder_vel[1] if shoulder_vel else np.nan)
        hip_vx_kalman.append(hip_vel[0] if hip_vel else np.nan)
        hip_vy_kalman.append(hip_vel[1] if hip_vel else np.nan)
        knee_vx_kalman.append(knee_vel[0] if knee_vel else np.nan)
        knee_vy_kalman.append(knee_vel[1] if knee_vel else np.nan)
        ankle_vx_kalman.append(ankle_vel[0] if ankle_vel else np.nan)
        ankle_vy_kalman.append(ankle_vel[1] if ankle_vel else np.nan)
        
        # Compute knee angle
        if hip_coords and knee_coords and ankle_coords:
            knee_angle = compute_angle(hip_coords, knee_coords, ankle_coords)
            knee_angles.append(knee_angle)
        else:
            knee_angles.append(np.nan)
        
        # Compute hip angle
        if shoulder_coords and hip_coords and knee_coords:
            hip_angle = compute_angle(shoulder_coords, hip_coords, knee_coords)
            hip_angles.append(hip_angle)
        else:
            hip_angles.append(np.nan)
        
        # Y positions
        shoulder_y.append(shoulder_coords[1] if shoulder_coords else np.nan)
        hip_y.append(hip_coords[1] if hip_coords else np.nan)
        knee_y.append(knee_coords[1] if knee_coords else np.nan)
        ankle_y.append(ankle_coords[1] if ankle_coords else np.nan)
        
        # X positions
        shoulder_x.append(shoulder_coords[0] if shoulder_coords else np.nan)
        hip_x.append(hip_coords[0] if hip_coords else np.nan)
        knee_x.append(knee_coords[0] if knee_coords else np.nan)
        ankle_x.append(ankle_coords[0] if ankle_coords else np.nan)
    
    # Get timestamps from data for accurate velocity calculation
    timestamps = data.get('frame_timestamps', [])
    ts_array = np.array(timestamps) if timestamps else None
    
    # Kalman filter velocities should always be available
    
    if not has_kalman_velocity:
        raise ValueError("Kalman filter velocity data missing from JSON file. Ensure recording uses --filter kalman or --filter oneeuro")
    
    # Use Kalman velocities directly (in px/s)
    # Negate vy because image Y increases downward but we want up=positive
    shoulder_vx = np.array(shoulder_vx_kalman)
    shoulder_vy = -np.array(shoulder_vy_kalman)
    hip_vx = np.array(hip_vx_kalman)
    hip_vy = -np.array(hip_vy_kalman)
    knee_vx = np.array(knee_vx_kalman)
    knee_vy = -np.array(knee_vy_kalman)
    ankle_vx = np.array(ankle_vx_kalman)
    ankle_vy = -np.array(ankle_vy_kalman)
    print("Using Kalman filter velocities (px/s)")
    
    # Apply additional smoothing to velocity components (on top of Kalman smoothing)
    # Larger window for more visible smoothing effect
    window = 11  # ~183ms at 60fps (was 5 = 83ms)
    shoulder_vx = moving_average(shoulder_vx, window)
    shoulder_vy = moving_average(shoulder_vy, window)
    hip_vx = moving_average(hip_vx, window)
    hip_vy = moving_average(hip_vy, window)
    knee_vx = moving_average(knee_vx, window)
    knee_vy = moving_average(knee_vy, window)
    ankle_vx = moving_average(ankle_vx, window)
    ankle_vy = moving_average(ankle_vy, window)
    
    # Compute speed (magnitude of velocity)
    shoulder_speed = np.sqrt(shoulder_vx**2 + shoulder_vy**2)
    hip_speed = np.sqrt(hip_vx**2 + hip_vy**2)
    knee_speed = np.sqrt(knee_vx**2 + knee_vy**2)
    ankle_speed = np.sqrt(ankle_vx**2 + ankle_vy**2)
    
    # Right wrist speed with additional smoothing
    right_wrist_vx_arr = np.array(right_wrist_vx_kalman)
    right_wrist_vy_arr = -np.array(right_wrist_vy_kalman)
    right_wrist_vx_arr = moving_average(right_wrist_vx_arr, window)
    right_wrist_vy_arr = moving_average(right_wrist_vy_arr, window)
    right_wrist_speed = np.sqrt(right_wrist_vx_arr**2 + right_wrist_vy_arr**2)
    
    # Calculate frame intervals and instantaneous FPS from timestamps
    frame_intervals = []
    instantaneous_fps = []
    
    if len(timestamps) >= 2:
        for i in range(1, len(timestamps)):
            interval = timestamps[i] - timestamps[i-1]
            frame_intervals.append(interval * 1000)  # Convert to ms
            if interval > 0:
                instantaneous_fps.append(1.0 / interval)
            else:
                instantaneous_fps.append(0)
        # Pad first frame
        if frame_intervals:
            frame_intervals.insert(0, frame_intervals[0])
            instantaneous_fps.insert(0, instantaneous_fps[0])
    
    return {
        # Position data
        'shoulder_x': np.array(shoulder_x),
        'shoulder_y': np.array(shoulder_y),
        'hip_x': np.array(hip_x),
        'hip_y': np.array(hip_y),
        'knee_x': np.array(knee_x),
        'knee_y': np.array(knee_y),
        'ankle_x': np.array(ankle_x),
        'ankle_y': np.array(ankle_y),
        'left_wrist_x': np.array(left_wrist_x),
        'left_wrist_y': np.array(left_wrist_y),
        'right_wrist_x': np.array(right_wrist_x),
        'right_wrist_y': np.array(right_wrist_y),
        'left_elbow_x': np.array(left_elbow_x),
        'left_elbow_y': np.array(left_elbow_y),
        'right_elbow_x': np.array(right_elbow_x),
        'right_elbow_y': np.array(right_elbow_y),
        # Angle data
        'knee_angles': np.array(knee_angles),
        'hip_angles': np.array(hip_angles),
        # Velocity data
        'shoulder_vx': shoulder_vx,
        'shoulder_vy': shoulder_vy,
        'hip_vx': hip_vx,
        'hip_vy': hip_vy,
        'knee_vx': knee_vx,
        'knee_vy': knee_vy,
        'ankle_vx': ankle_vx,
        'ankle_vy': ankle_vy,
        'left_wrist_vx': np.array(left_wrist_vx_kalman),
        'left_wrist_vy': -np.array(left_wrist_vy_kalman),
        'right_wrist_vx': np.array(right_wrist_vx_kalman),
        'right_wrist_vy': -np.array(right_wrist_vy_kalman),
        'left_elbow_vx': np.array(left_elbow_vx_kalman),
        'left_elbow_vy': -np.array(left_elbow_vy_kalman),
        'right_elbow_vx': np.array(right_elbow_vx_kalman),
        'right_elbow_vy': -np.array(right_elbow_vy_kalman),
        # Speed data
        'shoulder_speed': shoulder_speed,
        'hip_speed': hip_speed,
        'knee_speed': knee_speed,
        'ankle_speed': ankle_speed,
        'right_wrist_speed': right_wrist_speed,
        # Phase and timing data
        'phases': phases,
        'timestamps': np.array(timestamps),
        'frame_intervals': np.array(frame_intervals),
        'instantaneous_fps': np.array(instantaneous_fps),
        'detected_fps': detected_fps,
        'force_curve': data.get('force', [])
    }


def show_plot(filepath: str, current_file_idx: int, total_files: int, fig=None, axes=None):
    """Show interactive plot for a single stroke.
    
    Args:
        filepath: Path to stroke JSON file
        current_file_idx: Current stroke index
        total_files: Total number of stroke files
        fig: Optional existing figure to update (for smooth transitions)
        axes: Optional existing axes to update (for smooth transitions)
    """
    if not HAS_MATPLOTLIB:
        print("ERROR: matplotlib not available. Install with: pip3 install matplotlib")
        return None
    
    data = parse_stroke_file(filepath)
    
    # Position data
    shoulder_x = data['shoulder_x']
    shoulder_y = data['shoulder_y']
    hip_x = data['hip_x']
    hip_y = data['hip_y']
    knee_x = data['knee_x']
    knee_y = data['knee_y']
    ankle_x = data['ankle_x']
    ankle_y = data['ankle_y']
    left_wrist_x = data['left_wrist_x']
    left_wrist_y = data['left_wrist_y']
    right_wrist_x = data['right_wrist_x']
    right_wrist_y = data['right_wrist_y']
    left_elbow_x = data['left_elbow_x']
    left_elbow_y = data['left_elbow_y']
    right_elbow_x = data['right_elbow_x']
    right_elbow_y = data['right_elbow_y']
    # Velocity data
    shoulder_vx = data['shoulder_vx']
    shoulder_vy = data['shoulder_vy']
    hip_vx = data['hip_vx']
    hip_vy = data['hip_vy']
    knee_vx = data['knee_vx']
    knee_vy = data['knee_vy']
    ankle_vx = data['ankle_vx']
    ankle_vy = data['ankle_vy']
    left_wrist_vx = data['left_wrist_vx']
    left_wrist_vy = data['left_wrist_vy']
    right_wrist_vx = data['right_wrist_vx']
    right_wrist_vy = data['right_wrist_vy']
    # Speed data
    shoulder_speed = data['shoulder_speed']
    hip_speed = data['hip_speed']
    knee_speed = data['knee_speed']
    ankle_speed = data['ankle_speed']
    right_wrist_speed = data['right_wrist_speed']
    phases = data['phases']
    
    # No smoothing - use raw data directly
    shoulder_y_smooth = shoulder_y
    hip_y_smooth = hip_y
    knee_y_smooth = knee_y
    ankle_y_smooth = ankle_y
    shoulder_x_smooth = shoulder_x
    hip_x_smooth = hip_x
    knee_x_smooth = knee_x
    ankle_x_smooth = ankle_x
    left_wrist_x_smooth = left_wrist_x
    left_wrist_y_smooth = left_wrist_y
    right_wrist_x_smooth = right_wrist_x
    right_wrist_y_smooth = right_wrist_y
    left_elbow_x_smooth = left_elbow_x
    left_elbow_y_smooth = left_elbow_y
    right_elbow_x_smooth = right_elbow_x
    right_elbow_y_smooth = right_elbow_y
    
    # Compute elbow speeds from velocities
    right_elbow_vx = data['right_elbow_vx']
    right_elbow_vy = data['right_elbow_vy']
    right_elbow_speed = np.sqrt(right_elbow_vx**2 + right_elbow_vy**2)
    
    # Use velocities and speeds directly without additional smoothing
    shoulder_speed_smooth = shoulder_speed
    hip_speed_smooth = hip_speed
    knee_speed_smooth = knee_speed
    ankle_speed_smooth = ankle_speed
    right_wrist_speed_smooth = right_wrist_speed
    right_elbow_speed_smooth = right_elbow_speed
    shoulder_vx_smooth = shoulder_vx
    hip_vx_smooth = hip_vx
    knee_vx_smooth = knee_vx
    ankle_vx_smooth = ankle_vx
    shoulder_vy_smooth = shoulder_vy
    hip_vy_smooth = hip_vy
    knee_vy_smooth = knee_vy
    ankle_vy_smooth = ankle_vy
    
    frames = np.arange(len(shoulder_x))
    
    # Create or reuse figure with 4 subplots (skeleton, speeds, force, frequency)
    if fig is None or axes is None:
        fig = plt.figure(figsize=(16, 10))
        ax1 = fig.add_subplot(2, 3, 1)  # Skeleton
        ax2 = fig.add_subplot(2, 3, 2)  # Speeds
        ax3 = fig.add_subplot(2, 3, 3)  # Frequency response
        ax4 = fig.add_subplot(2, 1, 2)  # Force curve (full width bottom)
        axes = (ax1, ax2, ax3, ax4)
    else:
        ax1, ax2, ax3, ax4 = axes
        # Stop and clear any existing animations
        if hasattr(fig, '_animations'):
            for anim in fig._animations:
                anim.event_source.stop()
            fig._animations = []
        # Clear existing content
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()
        # Clear any twin axes that may have been created
        for ax in [ax2, ax3, ax4]:
            if hasattr(ax, '_twinned_axes') and ax._twinned_axes:
                for twin in ax._twinned_axes.get_siblings(ax):
                    if twin is not ax:
                        twin.remove()
    
    fig.suptitle(f'Stroke Analysis [{current_file_idx + 1}/{total_files}]: {os.path.basename(filepath)}', fontsize=14)
    
    # Phase colors
    phase_colors = {
        0: ('lightgray', 'IDLE'),
        1: ('yellow', 'WAIT_ACCEL'),
        2: ('lightcoral', 'DRIVE'),
        3: ('lightskyblue', 'DWELLING'),
        4: ('lightgreen', 'RECOVERY')
    }
    
    # Plot 1: Animated Skeleton
    # Set up the skeleton plot
    x_min = min(min(shoulder_x_smooth), min(hip_x_smooth), min(knee_x_smooth), min(ankle_x_smooth), min(right_wrist_x_smooth), min(right_elbow_x_smooth))
    x_max = max(max(shoulder_x_smooth), max(hip_x_smooth), max(knee_x_smooth), max(ankle_x_smooth), max(right_wrist_x_smooth), max(right_elbow_x_smooth))
    y_min = min(min(shoulder_y_smooth), min(hip_y_smooth), min(knee_y_smooth), min(ankle_y_smooth), min(right_wrist_y_smooth), min(right_elbow_y_smooth))
    y_max = max(max(shoulder_y_smooth), max(hip_y_smooth), max(knee_y_smooth), max(ankle_y_smooth), max(right_wrist_y_smooth), max(right_elbow_y_smooth))
    
    ax1.set_xlim(x_min - 20, x_max + 20)
    ax1.set_ylim(y_max + 20, y_min - 20)  # Inverted Y
    ax1.set_xlabel('X Position (px)', fontsize=10)
    ax1.set_ylabel('Y Position (px)', fontsize=10)
    ax1.set_title('Skeleton Animation (Frame: 0)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Draw skeleton connections (will be updated in animation)
    skeleton_lines = []
    # Torso: shoulder -> hip
    line1, = ax1.plot([], [], 'b-', linewidth=3, label='Torso')
    skeleton_lines.append(line1)
    # Thigh: hip -> knee
    line2, = ax1.plot([], [], 'g-', linewidth=3, label='Thigh')
    skeleton_lines.append(line2)
    # Shin: knee -> ankle
    line3, = ax1.plot([], [], 'orange', linewidth=3, label='Shin')
    skeleton_lines.append(line3)
    # Upper arm: shoulder -> elbow
    line4, = ax1.plot([], [], 'r-', linewidth=3, label='Upper Arm')
    skeleton_lines.append(line4)
    # Forearm: elbow -> wrist
    line5, = ax1.plot([], [], 'darkred', linewidth=3, label='Forearm')
    skeleton_lines.append(line5)
    
    # Draw joints as scatter points
    joints_scatter = ax1.scatter([], [], s=100, c='red', zorder=5)
    
    # Trail showing recent positions
    trail_length = 10
    trail_lines = []
    for _ in range(6):  # One trail per joint (shoulder, hip, knee, ankle, elbow, wrist)
        trail, = ax1.plot([], [], 'gray', alpha=0.3, linewidth=1)
        trail_lines.append(trail)
    
    ax1.legend(loc='upper left', fontsize=8)
    
    # Animation function
    def update_skeleton(frame_idx):
        if frame_idx >= len(shoulder_x_smooth):
            frame_idx = len(shoulder_x_smooth) - 1
        
        # Update skeleton lines
        skeleton_lines[0].set_data([shoulder_x_smooth[frame_idx], hip_x_smooth[frame_idx]], 
                                   [shoulder_y_smooth[frame_idx], hip_y_smooth[frame_idx]])
        skeleton_lines[1].set_data([hip_x_smooth[frame_idx], knee_x_smooth[frame_idx]], 
                                   [hip_y_smooth[frame_idx], knee_y_smooth[frame_idx]])
        skeleton_lines[2].set_data([knee_x_smooth[frame_idx], ankle_x_smooth[frame_idx]], 
                                   [knee_y_smooth[frame_idx], ankle_y_smooth[frame_idx]])
        skeleton_lines[3].set_data([shoulder_x_smooth[frame_idx], right_elbow_x_smooth[frame_idx]], 
                                   [shoulder_y_smooth[frame_idx], right_elbow_y_smooth[frame_idx]])
        skeleton_lines[4].set_data([right_elbow_x_smooth[frame_idx], right_wrist_x_smooth[frame_idx]], 
                                   [right_elbow_y_smooth[frame_idx], right_wrist_y_smooth[frame_idx]])
        
        # Update joint positions
        joint_x = [shoulder_x_smooth[frame_idx], hip_x_smooth[frame_idx], knee_x_smooth[frame_idx], 
                   ankle_x_smooth[frame_idx], right_elbow_x_smooth[frame_idx], right_wrist_x_smooth[frame_idx]]
        joint_y = [shoulder_y_smooth[frame_idx], hip_y_smooth[frame_idx], knee_y_smooth[frame_idx], 
                   ankle_y_smooth[frame_idx], right_elbow_y_smooth[frame_idx], right_wrist_y_smooth[frame_idx]]
        joints_scatter.set_offsets(np.c_[joint_x, joint_y])
        
        # Update trails
        start_idx = max(0, frame_idx - trail_length)
        trail_lines[0].set_data(shoulder_x_smooth[start_idx:frame_idx+1], shoulder_y_smooth[start_idx:frame_idx+1])
        trail_lines[1].set_data(hip_x_smooth[start_idx:frame_idx+1], hip_y_smooth[start_idx:frame_idx+1])
        trail_lines[2].set_data(knee_x_smooth[start_idx:frame_idx+1], knee_y_smooth[start_idx:frame_idx+1])
        trail_lines[3].set_data(ankle_x_smooth[start_idx:frame_idx+1], ankle_y_smooth[start_idx:frame_idx+1])
        trail_lines[4].set_data(right_elbow_x_smooth[start_idx:frame_idx+1], right_elbow_y_smooth[start_idx:frame_idx+1])
        trail_lines[5].set_data(right_wrist_x_smooth[start_idx:frame_idx+1], right_wrist_y_smooth[start_idx:frame_idx+1])
        
        # Update title with current frame and phase
        phase_name = phase_colors.get(phases[frame_idx], ('white', f'Phase{phases[frame_idx]}'))[1]
        ax1.set_title(f'Skeleton Animation (Frame: {frame_idx}/{len(frames)-1}, Phase: {phase_name})', fontsize=12)
        
        return skeleton_lines + [joints_scatter] + trail_lines
    
    # Create animation
    anim = FuncAnimation(fig, update_skeleton, frames=len(frames), interval=33, blit=True, repeat=True)
    
    # Store animation reference to prevent garbage collection
    if not hasattr(fig, '_animations'):
        fig._animations = []
    fig._animations.append(anim)
    
    # Draw phase backgrounds for 2D plots
    for ax in [ax2, ax4]:
        current_phase = None
        phase_start = 0
        for i, phase in enumerate(phases + [-1]):  # Add sentinel
            if phase != current_phase:
                if current_phase is not None:
                    color, label = phase_colors.get(current_phase, ('white', f'Phase{current_phase}'))
                    ax.axvspan(phase_start, i - 1, alpha=0.2, color=color, label=label)
                current_phase = phase
                phase_start = i
    
    # Plot 2: Joint Speeds (magnitude of velocity = sqrt(vx^2 + vy^2))
    ax2.plot(frames, shoulder_speed_smooth, 'purple', label='Shoulder', linewidth=2)
    ax2.plot(frames, hip_speed_smooth, 'blue', label='Hip', linewidth=2)
    ax2.plot(frames, knee_speed_smooth, 'green', label='Knee', linewidth=2)
    ax2.plot(frames, ankle_speed_smooth, 'orange', label='Ankle', linewidth=2)
    ax2.plot(frames, right_wrist_speed_smooth, 'red', label='Right Wrist', linewidth=2)
    ax2.set_ylabel('Speed (px/s)', fontsize=10)
    ax2.set_xlabel('Frame', fontsize=10)
    ax2.set_title('Joint Speed (√(vx² + vy²))', fontsize=12)
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Frequency Response of All Right-Side Joints
    filter_type = data.get('filter_type', 'unknown')
    timestamps = data['timestamps']
    
    # Compute frequency response for all right-side joints
    try:
        dt = np.mean(np.diff(timestamps))
        fs = 1.0 / dt  # Sampling frequency
        
        # Compute frequency response for each joint
        freq_shoulder, psd_shoulder = compute_frequency_response(timestamps, shoulder_speed_smooth)
        freq_hip, psd_hip = compute_frequency_response(timestamps, hip_speed_smooth)
        freq_knee, psd_knee = compute_frequency_response(timestamps, knee_speed_smooth)
        freq_ankle, psd_ankle = compute_frequency_response(timestamps, ankle_speed_smooth)
        freq_wrist, psd_wrist = compute_frequency_response(timestamps, right_wrist_speed_smooth)
        freq_elbow, psd_elbow = compute_frequency_response(timestamps, right_elbow_speed_smooth)
        
        # Sum all frequency responses (convert from dB back to linear, sum, then back to dB)
        psd_linear_sum = (10**(psd_shoulder/10) + 10**(psd_hip/10) + 10**(psd_knee/10) + 
                          10**(psd_ankle/10) + 10**(psd_wrist/10) + 10**(psd_elbow/10))
        psd_sum = 10 * np.log10(psd_linear_sum)
        
        # === FREQUENCY ANALYSIS LOGGING ===
        print(f"\n{'='*60}")
        print(f"FREQUENCY RESPONSE ANALYSIS")
        print(f"{'='*60}")
        print(f"Sampling rate: {fs:.1f} Hz")
        print(f"Filter type: {filter_type}")
        print(f"Stroke file: {os.path.basename(filepath)}")
        
        # Find dominant frequency in summed response (skip DC component at index 0)
        freq_range_mask = (freq_shoulder[1:] > 0.1) & (freq_shoulder[1:] < 2.0)  # Focus on stroke rate range
        if np.any(freq_range_mask):
            dominant_idx = np.argmax(psd_sum[1:][freq_range_mask])
            dominant_freq = freq_shoulder[1:][freq_range_mask][dominant_idx]
            dominant_power = psd_sum[1:][freq_range_mask][dominant_idx]
            stroke_rate_spm = dominant_freq * 60  # Convert Hz to strokes per minute
            print(f"\nDominant Frequency: {dominant_freq:.3f} Hz ({stroke_rate_spm:.1f} SPM)")
            print(f"  Power at dominant: {dominant_power:.1f} dB")
        
        # Analyze power in different frequency bands
        bands = [
            ("Stroke Rate (0.2-1 Hz)", 0.2, 1.0),
            ("Low Motion (1-3 Hz)", 1.0, 3.0),
            ("Mid Motion (3-5 Hz)", 3.0, 5.0),
            ("High Freq Noise (5-10 Hz)", 5.0, 10.0),
            ("Very High Noise (10-20 Hz)", 10.0, 20.0)
        ]
        
        print(f"\nPower Distribution by Frequency Band:")
        for band_name, f_low, f_high in bands:
            band_mask = (freq_shoulder[1:] >= f_low) & (freq_shoulder[1:] < f_high)
            if np.any(band_mask):
                avg_power = np.mean(psd_sum[1:][band_mask])
                max_power = np.max(psd_sum[1:][band_mask])
                print(f"  {band_name:25s}: avg={avg_power:6.1f} dB, max={max_power:6.1f} dB")
        
        # Per-joint dominant frequencies
        print(f"\nPer-Joint Dominant Frequencies (0.1-2 Hz range):")
        joint_data = [
            ("Shoulder", psd_shoulder),
            ("Hip", psd_hip),
            ("Knee", psd_knee),
            ("Ankle", psd_ankle),
            ("Right Wrist", psd_wrist),
            ("Right Elbow", psd_elbow)
        ]
        for joint_name, psd in joint_data:
            if np.any(freq_range_mask):
                idx = np.argmax(psd[1:][freq_range_mask])
                freq = freq_shoulder[1:][freq_range_mask][idx]
                power = psd[1:][freq_range_mask][idx]
                print(f"  {joint_name:12s}: {freq:.3f} Hz ({freq*60:.1f} SPM), power={power:.1f} dB")
        
        print(f"{'='*60}\n")
        
        # Plot summed frequency response (thicker, prominent)
        ax3.plot(freq_shoulder[1:], psd_sum[1:], 'black', linewidth=3, label='Sum (All Joints)', alpha=0.9)
        
        # Plot all joints overlayed (thinner, more transparent)
        ax3.plot(freq_shoulder[1:], psd_shoulder[1:], 'purple', linewidth=1.5, label='Shoulder', alpha=0.5)
        ax3.plot(freq_hip[1:], psd_hip[1:], 'blue', linewidth=1.5, label='Hip', alpha=0.5)
        ax3.plot(freq_knee[1:], psd_knee[1:], 'green', linewidth=1.5, label='Knee', alpha=0.5)
        ax3.plot(freq_ankle[1:], psd_ankle[1:], 'orange', linewidth=1.5, label='Ankle', alpha=0.5)
        ax3.plot(freq_wrist[1:], psd_wrist[1:], 'red', linewidth=1.5, label='Right Wrist', alpha=0.5)
        ax3.plot(freq_elbow[1:], psd_elbow[1:], 'brown', linewidth=1.5, label='Right Elbow', alpha=0.5)
        
        ax3.set_xlabel('Frequency (Hz)', fontsize=10)
        ax3.set_ylabel('Power (dB)', fontsize=10)
        ax3.set_title(f'Joint Velocity Frequency Response ({filter_type} filter)', fontsize=12)
        ax3.set_xlim([0, min(fs/2, 30)])  # Show up to Nyquist or 30 Hz
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='best', fontsize=8)
        
        # Add text with filter info
        info_text = f"Fs: {fs:.1f} Hz\nFilter: {filter_type}"
        ax3.text(0.98, 0.98, info_text, transform=ax3.transAxes,
                fontsize=8, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    except Exception as e:
        ax3.text(0.5, 0.5, f'Error computing frequency response:\n{str(e)}', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=10)
        ax3.set_title('Frequency Response (Error)', fontsize=12)
    
    # Plot 4: Force Curve (only during drive phase)
    force_curve = data.get('force_curve', [])  # parse_stroke_file returns 'force_curve'
    if force_curve:
        # Find drive phase frames
        drive_frames = [i for i, p in enumerate(phases) if p == 2]
        if drive_frames:
            # Map force samples to drive phase frames
            drive_start = drive_frames[0]
            drive_end = drive_frames[-1]
            
            # Interpolate force curve to match drive frames
            force_x = np.linspace(drive_start, drive_end, len(force_curve))
            
            ax4.plot(force_x, force_curve, 'red', label='Force Curve', linewidth=2.5)
            ax4.fill_between(force_x, 0, force_curve, alpha=0.3, color='red')
            ax4.set_ylabel('Force (PM5 units)', fontsize=10)
            ax4.set_xlabel('Frame', fontsize=10)
            ax4.set_title('Force Curve (Drive Phase)', fontsize=12)
            ax4.legend(loc='best', fontsize=8)
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No drive phase detected', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Force Curve (No Data)', fontsize=12)
    else:
        ax4.text(0.5, 0.5, 'No force data available', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Force Curve (No Data)', fontsize=12)
    
    # Highlight buffer zones (first 30 and last 30 frames) on 2D plots
    buffer_size = 30
    for ax in [ax2, ax4]:
        if len(frames) > buffer_size:
            ax.axvspan(0, buffer_size, alpha=0.1, color='gray', linestyle='--')
            ax.axvspan(len(frames) - buffer_size, len(frames), alpha=0.1, color='gray', linestyle='--')
    
    plt.tight_layout()
    return fig, (ax1, ax2, ax3, ax4)


def show_averages(files: List[str]):
    """Show average stroke data across all strokes including joint velocities."""
    if not HAS_MATPLOTLIB:
        print("ERROR: matplotlib not available. Install with: pip3 install matplotlib")
        return None
    
    all_knee_angles = []
    all_hip_angles = []
    all_shoulder_vx = []
    all_shoulder_vy = []
    all_hip_vx = []
    all_hip_vy = []
    all_knee_vx = []
    all_knee_vy = []
    all_ankle_vx = []
    all_ankle_vy = []
    all_right_wrist_vx = []
    all_right_wrist_vy = []
    all_right_elbow_vx = []
    all_right_elbow_vy = []
    max_len = 0
    
    for filepath in files:
        data = parse_stroke_file(filepath)
        all_knee_angles.append(data['knee_angles'])
        all_hip_angles.append(data['hip_angles'])
        all_shoulder_vx.append(data['shoulder_vx'])
        all_shoulder_vy.append(data['shoulder_vy'])
        all_hip_vx.append(data['hip_vx'])
        all_hip_vy.append(data['hip_vy'])
        all_knee_vx.append(data['knee_vx'])
        all_knee_vy.append(data['knee_vy'])
        all_ankle_vx.append(data['ankle_vx'])
        all_ankle_vy.append(data['ankle_vy'])
        all_right_wrist_vx.append(data['right_wrist_vx'])
        all_right_wrist_vy.append(data['right_wrist_vy'])
        all_right_elbow_vx.append(data['right_elbow_vx'])
        all_right_elbow_vy.append(data['right_elbow_vy'])
        
        max_len = max(max_len, len(data['knee_angles']))
    
    # Pad arrays to same length
    for i in range(len(all_knee_angles)):
        if len(all_knee_angles[i]) < max_len:
            padding = np.full(max_len - len(all_knee_angles[i]), np.nan)
            all_knee_angles[i] = np.concatenate([all_knee_angles[i], padding])
            all_hip_angles[i] = np.concatenate([all_hip_angles[i], padding])
            all_shoulder_vx[i] = np.concatenate([all_shoulder_vx[i], padding])
            all_shoulder_vy[i] = np.concatenate([all_shoulder_vy[i], padding])
            all_hip_vx[i] = np.concatenate([all_hip_vx[i], padding])
            all_hip_vy[i] = np.concatenate([all_hip_vy[i], padding])
            all_knee_vx[i] = np.concatenate([all_knee_vx[i], padding])
            all_knee_vy[i] = np.concatenate([all_knee_vy[i], padding])
            all_ankle_vx[i] = np.concatenate([all_ankle_vx[i], padding])
            all_ankle_vy[i] = np.concatenate([all_ankle_vy[i], padding])
            all_right_wrist_vx[i] = np.concatenate([all_right_wrist_vx[i], padding])
            all_right_wrist_vy[i] = np.concatenate([all_right_wrist_vy[i], padding])
            all_right_elbow_vx[i] = np.concatenate([all_right_elbow_vx[i], padding])
            all_right_elbow_vy[i] = np.concatenate([all_right_elbow_vy[i], padding])
    
    # Compute averages
    avg_knee = np.nanmean(all_knee_angles, axis=0)
    avg_hip = np.nanmean(all_hip_angles, axis=0)
    avg_shoulder_vx = np.nanmean(all_shoulder_vx, axis=0)
    avg_shoulder_vy = np.nanmean(all_shoulder_vy, axis=0)
    avg_hip_vx = np.nanmean(all_hip_vx, axis=0)
    avg_hip_vy = np.nanmean(all_hip_vy, axis=0)
    avg_knee_vx = np.nanmean(all_knee_vx, axis=0)
    avg_knee_vy = np.nanmean(all_knee_vy, axis=0)
    avg_ankle_vx = np.nanmean(all_ankle_vx, axis=0)
    avg_ankle_vy = np.nanmean(all_ankle_vy, axis=0)
    avg_right_wrist_vx = np.nanmean(all_right_wrist_vx, axis=0)
    avg_right_wrist_vy = np.nanmean(all_right_wrist_vy, axis=0)
    avg_right_elbow_vx = np.nanmean(all_right_elbow_vx, axis=0)
    avg_right_elbow_vy = np.nanmean(all_right_elbow_vy, axis=0)
    
    # Apply smoothing
    avg_knee_smooth = moving_average(avg_knee, 3)
    avg_hip_smooth = moving_average(avg_hip, 3)
    avg_shoulder_vx_smooth = moving_average(avg_shoulder_vx, 3)
    avg_shoulder_vy_smooth = moving_average(avg_shoulder_vy, 3)
    avg_hip_vx_smooth = moving_average(avg_hip_vx, 3)
    avg_hip_vy_smooth = moving_average(avg_hip_vy, 3)
    avg_knee_vx_smooth = moving_average(avg_knee_vx, 3)
    avg_knee_vy_smooth = moving_average(avg_knee_vy, 3)
    avg_ankle_vx_smooth = moving_average(avg_ankle_vx, 3)
    avg_ankle_vy_smooth = moving_average(avg_ankle_vy, 3)
    avg_right_wrist_vx_smooth = moving_average(avg_right_wrist_vx, 3)
    avg_right_wrist_vy_smooth = moving_average(avg_right_wrist_vy, 3)
    avg_right_elbow_vx_smooth = moving_average(avg_right_elbow_vx, 3)
    avg_right_elbow_vy_smooth = moving_average(avg_right_elbow_vy, 3)
    
    frames = np.arange(len(avg_knee))
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
    fig.suptitle(f'Average Across {len(files)} Strokes', fontsize=14)
    
    # Plot 1: Joint Angles
    ax1.plot(frames, avg_knee_smooth, 'b-', label='Avg Knee Angle', linewidth=2)
    ax1.plot(frames, avg_hip_smooth, 'r-', label='Avg Hip Angle', linewidth=2)
    ax1.set_xlabel('Frame', fontsize=12)
    ax1.set_ylabel('Angle (degrees)', fontsize=12)
    ax1.set_title('Average Joint Angles', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Horizontal Velocities (Vx)
    ax2.plot(frames, avg_shoulder_vx_smooth, 'purple', label='Shoulder', linewidth=2)
    ax2.plot(frames, avg_hip_vx_smooth, 'blue', label='Hip', linewidth=2)
    ax2.plot(frames, avg_knee_vx_smooth, 'green', label='Knee', linewidth=2)
    ax2.plot(frames, avg_ankle_vx_smooth, 'orange', label='Ankle', linewidth=2)
    ax2.plot(frames, avg_right_wrist_vx_smooth, 'red', label='Right Wrist', linewidth=2)
    ax2.plot(frames, avg_right_elbow_vx_smooth, 'brown', label='Right Elbow', linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Frame', fontsize=12)
    ax2.set_ylabel('Horizontal Velocity (px/s)', fontsize=12)
    ax2.set_title('Average Horizontal Velocities (Vx)', fontsize=12)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Vertical Velocities (Vy)
    ax3.plot(frames, avg_shoulder_vy_smooth, 'purple', label='Shoulder', linewidth=2)
    ax3.plot(frames, avg_hip_vy_smooth, 'blue', label='Hip', linewidth=2)
    ax3.plot(frames, avg_knee_vy_smooth, 'green', label='Knee', linewidth=2)
    ax3.plot(frames, avg_ankle_vy_smooth, 'orange', label='Ankle', linewidth=2)
    ax3.plot(frames, avg_right_wrist_vy_smooth, 'red', label='Right Wrist', linewidth=2)
    ax3.plot(frames, avg_right_elbow_vy_smooth, 'brown', label='Right Elbow', linewidth=2)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax3.set_xlabel('Frame', fontsize=12)
    ax3.set_ylabel('Vertical Velocity (px/s)', fontsize=12)
    ax3.set_title('Average Vertical Velocities (Vy)', fontsize=12)
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Print statistics to console
    print(f"\n{'='*60}")
    print(f"AVERAGE VELOCITY STATISTICS ({len(files)} strokes)")
    print(f"{'='*60}")
    print(f"\nAverage Peak Horizontal Velocities (Vx, px/s):")
    print(f"  Shoulder    : {np.nanmax(np.abs(avg_shoulder_vx_smooth)):.1f} px/s")
    print(f"  Hip         : {np.nanmax(np.abs(avg_hip_vx_smooth)):.1f} px/s")
    print(f"  Knee        : {np.nanmax(np.abs(avg_knee_vx_smooth)):.1f} px/s")
    print(f"  Ankle       : {np.nanmax(np.abs(avg_ankle_vx_smooth)):.1f} px/s")
    print(f"  Right Wrist : {np.nanmax(np.abs(avg_right_wrist_vx_smooth)):.1f} px/s")
    print(f"  Right Elbow : {np.nanmax(np.abs(avg_right_elbow_vx_smooth)):.1f} px/s")
    print(f"\nAverage Peak Vertical Velocities (Vy, px/s):")
    print(f"  Shoulder    : {np.nanmax(np.abs(avg_shoulder_vy_smooth)):.1f} px/s")
    print(f"  Hip         : {np.nanmax(np.abs(avg_hip_vy_smooth)):.1f} px/s")
    print(f"  Knee        : {np.nanmax(np.abs(avg_knee_vy_smooth)):.1f} px/s")
    print(f"  Ankle       : {np.nanmax(np.abs(avg_ankle_vy_smooth)):.1f} px/s")
    print(f"  Right Wrist : {np.nanmax(np.abs(avg_right_wrist_vy_smooth)):.1f} px/s")
    print(f"  Right Elbow : {np.nanmax(np.abs(avg_right_elbow_vy_smooth)):.1f} px/s")
    print(f"{'='*60}\n")
    
    return fig


def on_key(event, files: List[str], current_idx: list, fig_state: dict):
    """Handle keyboard events for navigation.
    
    Args:
        event: Keyboard event
        files: List of stroke files
        current_idx: List containing current index (mutable)
        fig_state: Dict containing 'fig' and 'axes' for reuse
    """
    if event.key == 'right' and current_idx[0] < len(files) - 1:
        current_idx[0] += 1
        fig, axes = show_plot(files[current_idx[0]], current_idx[0], len(files), 
                             fig_state.get('fig'), fig_state.get('axes'))
        if fig:
            fig_state['fig'] = fig
            fig_state['axes'] = axes
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
    elif event.key == 'left' and current_idx[0] > 0:
        current_idx[0] -= 1
        fig, axes = show_plot(files[current_idx[0]], current_idx[0], len(files),
                             fig_state.get('fig'), fig_state.get('axes'))
        if fig:
            fig_state['fig'] = fig
            fig_state['axes'] = axes
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
    elif event.key == 'a':
        plt.close('all')
        fig = show_averages(files)
        if fig:
            fig_state['fig'] = None  # Reset for averages view
            fig_state['axes'] = None
            fig.canvas.mpl_connect('key_press_event', lambda e: on_key(e, files, current_idx, fig_state))
            plt.show()


def print_text_summary(files: List[str], stroke_num: int = None):
    """Print text-only summary of stroke data."""
    if stroke_num is not None:
        if stroke_num < 1 or stroke_num > len(files):
            print(f"Error: Stroke number must be between 1 and {len(files)}")
            return
        
        filepath = files[stroke_num - 1]
        data = load_stroke_file(filepath)
        
        print(f"\n{'='*70}")
        print(f"Stroke {stroke_num}: {os.path.basename(filepath)}")
        print(f"{'='*70}")
        
        # Handle both old and new formats
        frames_array = data.get('frames', [])
        if frames_array:
            # New C++ format
            phases = [f.get('phase', 0) for f in frames_array]
            timestamps = [f.get('timestamp', 0) for f in frames_array]
            keypoints = [f.get('keypoints', []) for f in frames_array]
        else:
            # Old Python format
            phases = data.get('phases', [])
            timestamps = data.get('frame_timestamps', [])
            keypoints = data.get('keypoints', [])
        
        frame_count = data.get('frame_count', len(phases))
        
        print(f"\nTotal frames: {frame_count}")
        
        # Force curve analysis (new feature)
        force_curve = data.get('force', [])  # Changed from 'force_curve' to 'force'
        if force_curve:
            print(f"\nForce Curve:")
            print(f"  Samples: {len(force_curve)}")
            print(f"  Range: {min(force_curve)} - {max(force_curve)}")
            peak_idx = force_curve.index(max(force_curve))
            print(f"  Peak force: {max(force_curve)} at sample {peak_idx} ({peak_idx/len(force_curve)*100:.0f}%)")
            print(f"  First 10 samples: {force_curve[:10]}")
            print(f"  Last 10 samples: {force_curve[-10:]}")
        
        # Phase distribution
        if phases:
            phase_counts = {}
            for p in phases:
                phase_counts[p] = phase_counts.get(p, 0) + 1
            
            phase_names = {0: "IDLE", 1: "WAIT_ACCEL", 2: "DRIVE", 3: "DWELLING", 4: "RECOVERY"}
            print("\nPhase distribution:")
            for phase_id, count in sorted(phase_counts.items()):
                pct = 100 * count / len(phases)
                print(f"  {phase_names.get(phase_id, f'Phase{phase_id}'):12s}: {count:3d} frames ({pct:5.1f}%)")
        
        # Drive phase analysis
        if phases:
            drive_frames = [i for i, p in enumerate(phases) if p == 2]
            if drive_frames:
                drive_start = drive_frames[0]
                drive_end = drive_frames[-1]
                drive_duration = drive_end - drive_start + 1
                
                # Calculate FPS from timestamps if available
                detected_fps = 60.0
                if len(timestamps) >= 10:
                    intervals = [timestamps[i+1] - timestamps[i] for i in range(min(100, len(timestamps)-1))]
                    valid = [x for x in intervals if x > 0]
                    if valid:
                        detected_fps = 1.0 / (sum(valid) / len(valid))
                
                print(f"\nDrive phase:")
                print(f"  Start frame: {drive_start}")
                print(f"  End frame: {drive_end}")
                print(f"  Duration: {drive_duration} frames (~{drive_duration * 1000 / detected_fps:.0f}ms @ {detected_fps:.1f} FPS)")
        
        # Timestamp analysis
        
        if len(timestamps) >= 2:
            intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            avg_interval = sum(intervals) / len(intervals)
            min_interval = min(intervals)
            max_interval = max(intervals)
            jitter = np.std(intervals)
            avg_fps = 1.0 / avg_interval if avg_interval > 0 else 0
            
            print(f"\nFrame Timing:")
            print(f"  Total frames with timestamps: {len(timestamps)}")
            print(f"  Average interval: {avg_interval*1000:.2f}ms")
            print(f"  Min interval: {min_interval*1000:.2f}ms ({1.0/min_interval:.1f} FPS)")
            print(f"  Max interval: {max_interval*1000:.2f}ms ({1.0/max_interval:.1f} FPS)")
            print(f"  Average FPS: {avg_fps:.1f}")
            print(f"  Jitter (std dev): {jitter*1000:.2f}ms")
            print(f"  Total duration: {timestamps[-1] - timestamps[0]:.2f}s")
        else:
            print(f"\nFrame Timing:")
            print(f"  No timestamp data available (recorded before timestamp feature)")
            print(f"  Estimated duration: {len(phases) / 60.0:.2f}s @ 60 FPS (assumed)")
    else:
        # Summary of all strokes
        print(f"\n{'='*70}")
        print(f"SUMMARY OF ALL {len(files)} STROKES")
        print(f"{'='*70}")
        
        all_total_frames = []
        all_drive_durations = []
        all_force_samples = []
        all_peak_forces = []
        
        for filepath in files:
            data = load_stroke_file(filepath)
            
            # Handle both old and new formats
            frames_array = data.get('frames', [])
            if frames_array:
                # New C++ format
                phases = [f.get('phase', 0) for f in frames_array]
                frame_count = data.get('frame_count', len(phases))
            else:
                # Old Python format
                phases = data.get('phases', [])
                frame_count = len(phases)
            
            all_total_frames.append(frame_count)
            
            drive_frames = [i for i, p in enumerate(phases) if p == 2]
            if drive_frames:
                drive_duration = drive_frames[-1] - drive_frames[0] + 1
                all_drive_durations.append(drive_duration)
            
            # Collect force curve stats
            force_curve = data.get('force', [])  # Changed from 'force_curve' to 'force'
            if force_curve:
                all_force_samples.append(len(force_curve))
                all_peak_forces.append(max(force_curve))
        
        if all_total_frames:
            print(f"\nTotal frames per stroke:")
            print(f"  Min: {min(all_total_frames)}")
            print(f"  Max: {max(all_total_frames)}")
            print(f"  Avg: {np.mean(all_total_frames):.1f}")
        
        if all_force_samples:
            print(f"\nForce curve data ({len(all_force_samples)}/{len(files)} strokes):")
            print(f"  Samples per stroke: {min(all_force_samples)}-{max(all_force_samples)} (avg: {np.mean(all_force_samples):.1f})")
            print(f"  Peak forces: {min(all_peak_forces)}-{max(all_peak_forces)} (avg: {np.mean(all_peak_forces):.1f})")
        
        if all_drive_durations:
            print(f"\nDrive phase duration:")
            print(f"  Min: {min(all_drive_durations)} frames (~{min(all_drive_durations) * 1000 / 90:.0f}ms)")
            print(f"  Max: {max(all_drive_durations)} frames (~{max(all_drive_durations) * 1000 / 90:.0f}ms)")
            print(f"  Avg: {np.mean(all_drive_durations):.1f} frames (~{np.mean(all_drive_durations) * 1000 / 90:.0f}ms)")




def main():
    parser = argparse.ArgumentParser(
        description='Analyze recorded rowing stroke data from /tmp/stroke_data/'
    )
    parser.add_argument(
        '--stroke', 
        type=int, 
        default=None, 
        help='Analyze specific stroke number (1-indexed)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/tmp/stroke_data',
        help='Directory containing stroke files (default: /tmp/stroke_data)'
    )
    parser.add_argument(
        '--text-only',
        action='store_true',
        help='Show text summary instead of visualization (useful if matplotlib not available)'
    )
    parser.add_argument(
        '--average',
        action='store_true',
        help='Show average across all strokes (visualization mode only)'
    )
    parser.add_argument(
        '--keypoints',
        action='store_true',
        help='Show keypoint skeleton visualization instead of angles/positions'
    )
    args = parser.parse_args()
    
    # Find all stroke files (plain JSON, not compressed)
    pattern = os.path.join(args.data_dir, "stroke_*.json")
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"No stroke files found in {args.data_dir}")
        print(f"Looking for pattern: {pattern}")
        print(f"\nMake sure rowing_ergometer_recording.py has been run and captured strokes.")
        return
    
    # Exclude first and last stroke (often incomplete/warmup/cooldown)
    if len(files) > 2:
        files = files[1:-1]
        print(f"Found {len(files)} stroke file(s) in {args.data_dir} (excluding first and last)")
    else:
        print(f"Found {len(files)} stroke file(s) in {args.data_dir} (need >2 strokes to exclude first/last)")
    
    if not files:
        print("No strokes to analyze after excluding first and last")
        return
    
    # Text-only mode
    if args.text_only or not HAS_MATPLOTLIB:
        if not HAS_MATPLOTLIB:
            print("Note: matplotlib not available, using text-only mode")
            print("Install matplotlib with: pip3 install matplotlib\n")
        
        print_text_summary(files, args.stroke)
        return
    
    # Keypoint skeleton visualization
    if args.keypoints:
        start_idx = (args.stroke - 1) if args.stroke else 0
        if start_idx < 0 or start_idx >= len(files):
            print(f"Error: Stroke number must be between 1 and {len(files)}")
            return
        
        print(f"\nShowing keypoint skeleton for stroke {start_idx + 1}")
        fig = visualize_keypoints(files[start_idx])
        if fig:
            plt.show()
        return
    
    # Visualization mode
    if args.average:
        fig = show_averages(files)
        if fig:
            current_idx = [0]
            fig_state = {'fig': None, 'axes': None}
            fig.canvas.mpl_connect('key_press_event', lambda e: on_key(e, files, current_idx, fig_state))
            print("\nKeyboard controls:")
            print("  Left/Right arrows: Navigate between strokes")
            print("  'a': Show average across all strokes")
            print("  'q': Quit")
            plt.show()
    else:
        start_idx = (args.stroke - 1) if args.stroke else 0
        if start_idx < 0 or start_idx >= len(files):
            print(f"Error: Stroke number must be between 1 and {len(files)}")
            return
        
        current_idx = [start_idx]
        fig_state = {}  # Shared state for figure reuse
        fig, axes = show_plot(files[current_idx[0]], current_idx[0], len(files))
        if fig:
            fig_state['fig'] = fig
            fig_state['axes'] = axes
            fig.canvas.mpl_connect('key_press_event', lambda e: on_key(e, files, current_idx, fig_state))
            print("\nKeyboard controls:")
            print("  Left/Right arrows: Navigate between strokes")
            print("  'a': Show average across all strokes")
            print("  'q': Quit")
            plt.show()


if __name__ == "__main__":
    main()
