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
    left_wrist_vx_kalman = []
    left_wrist_vy_kalman = []
    right_wrist_vx_kalman = []
    right_wrist_vy_kalman = []
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
    
    # Process each frame
    for frame_kpts in keypoints_list:
        # Build keypoint dict for this frame (include velocity if present)
        kp_dict = {}
        kp_vel_dict = {}
        for kp in frame_kpts:
            kp_dict[kp['name']] = (kp['x'], kp['y'])
            # Check for Kalman velocities (vx, vy in px/s)
            if 'vx' in kp and 'vy' in kp:
                kp_vel_dict[kp['name']] = (kp['vx'], kp['vy'])
                has_kalman_velocity = True
        
        # Extract coordinates
        shoulder_coords = kp_dict.get('left_shoulder') or kp_dict.get('right_shoulder')
        hip_coords = kp_dict.get('left_hip') or kp_dict.get('right_hip')
        knee_coords = kp_dict.get('left_knee') or kp_dict.get('right_knee')
        ankle_coords = kp_dict.get('left_ankle') or kp_dict.get('right_ankle')
        
        # Extract wrist coordinates (both left and right)
        left_wrist_coords = kp_dict.get('left_wrist')
        right_wrist_coords = kp_dict.get('right_wrist')
        
        # Extract Kalman velocities if available
        shoulder_vel = kp_vel_dict.get('left_shoulder') or kp_vel_dict.get('right_shoulder')
        hip_vel = kp_vel_dict.get('left_hip') or kp_vel_dict.get('right_hip')
        knee_vel = kp_vel_dict.get('left_knee') or kp_vel_dict.get('right_knee')
        ankle_vel = kp_vel_dict.get('left_ankle') or kp_vel_dict.get('right_ankle')
        left_wrist_vel = kp_vel_dict.get('left_wrist')
        right_wrist_vel = kp_vel_dict.get('right_wrist')
        
        # Store wrist positions
        left_wrist_x.append(left_wrist_coords[0] if left_wrist_coords else np.nan)
        left_wrist_y.append(left_wrist_coords[1] if left_wrist_coords else np.nan)
        right_wrist_x.append(right_wrist_coords[0] if right_wrist_coords else np.nan)
        right_wrist_y.append(right_wrist_coords[1] if right_wrist_coords else np.nan)
        
        # Store wrist velocities
        left_wrist_vx_kalman.append(left_wrist_vel[0] if left_wrist_vel else np.nan)
        left_wrist_vy_kalman.append(left_wrist_vel[1] if left_wrist_vel else np.nan)
        right_wrist_vx_kalman.append(right_wrist_vel[0] if right_wrist_vel else np.nan)
        right_wrist_vy_kalman.append(right_wrist_vel[1] if right_wrist_vel else np.nan)
        
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
        'knee_angles': np.array(knee_angles),
        'hip_angles': np.array(hip_angles),
        'shoulder_y': np.array(shoulder_y),
        'hip_y': np.array(hip_y),
        'knee_y': np.array(knee_y),
        'ankle_y': np.array(ankle_y),
        'shoulder_x': np.array(shoulder_x),
        'hip_x': np.array(hip_x),
        'knee_x': np.array(knee_x),
        'ankle_x': np.array(ankle_x),
        'left_wrist_x': np.array(left_wrist_x),
        'left_wrist_y': np.array(left_wrist_y),
        'right_wrist_x': np.array(right_wrist_x),
        'right_wrist_y': np.array(right_wrist_y),
        'left_wrist_vx': np.array(left_wrist_vx_kalman),
        'left_wrist_vy': -np.array(left_wrist_vy_kalman),
        'right_wrist_vx': np.array(right_wrist_vx_kalman),
        'right_wrist_vy': -np.array(right_wrist_vy_kalman),
        'shoulder_vx': shoulder_vx,
        'shoulder_vy': shoulder_vy,
        'hip_vx': hip_vx,
        'hip_vy': hip_vy,
        'knee_vx': knee_vx,
        'knee_vy': knee_vy,
        'ankle_vx': ankle_vx,
        'ankle_vy': ankle_vy,
        'shoulder_speed': shoulder_speed,
        'hip_speed': hip_speed,
        'knee_speed': knee_speed,
        'ankle_speed': ankle_speed,
        'right_wrist_speed': right_wrist_speed,
        'phases': phases,
        'timestamps': np.array(timestamps),
        'frame_intervals': np.array(frame_intervals),
        'instantaneous_fps': np.array(instantaneous_fps),
        'detected_fps': detected_fps,
        'force_curve': data.get('force', [])  # Changed from 'force_curve' to 'force'
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
    
    knee_angles = data['knee_angles']
    hip_angles = data['hip_angles']
    shoulder_y = data['shoulder_y']
    hip_y = data['hip_y']
    knee_y = data['knee_y']
    ankle_y = data['ankle_y']
    shoulder_x = data['shoulder_x']
    hip_x = data['hip_x']
    knee_x = data['knee_x']
    ankle_x = data['ankle_x']
    left_wrist_x = data['left_wrist_x']
    left_wrist_y = data['left_wrist_y']
    right_wrist_x = data['right_wrist_x']
    right_wrist_y = data['right_wrist_y']
    left_wrist_vx = data['left_wrist_vx']
    left_wrist_vy = data['left_wrist_vy']
    right_wrist_vx = data['right_wrist_vx']
    right_wrist_vy = data['right_wrist_vy']
    shoulder_speed = data['shoulder_speed']
    hip_speed = data['hip_speed']
    knee_speed = data['knee_speed']
    ankle_speed = data['ankle_speed']
    right_wrist_speed = data['right_wrist_speed']
    shoulder_vx = data['shoulder_vx']
    hip_vx = data['hip_vx']
    knee_vx = data['knee_vx']
    ankle_vx = data['ankle_vx']
    shoulder_vy = data['shoulder_vy']
    hip_vy = data['hip_vy']
    knee_vy = data['knee_vy']
    ankle_vy = data['ankle_vy']
    phases = data['phases']
    
    # Apply smoothing
    window = 5
    knee_smooth = moving_average(knee_angles, window)
    hip_smooth = moving_average(hip_angles, window)
    shoulder_y_smooth = moving_average(shoulder_y, window)
    hip_y_smooth = moving_average(hip_y, window)
    knee_y_smooth = moving_average(knee_y, window)
    ankle_y_smooth = moving_average(ankle_y, window)
    shoulder_x_smooth = moving_average(shoulder_x, window)
    hip_x_smooth = moving_average(hip_x, window)
    knee_x_smooth = moving_average(knee_x, window)
    ankle_x_smooth = moving_average(ankle_x, window)
    left_wrist_x_smooth = moving_average(left_wrist_x, window)
    left_wrist_y_smooth = moving_average(left_wrist_y, window)
    right_wrist_x_smooth = moving_average(right_wrist_x, window)
    right_wrist_y_smooth = moving_average(right_wrist_y, window)
    
    # Velocities and speeds already smoothed in parse_stroke_file, use directly
    shoulder_speed_smooth = shoulder_speed
    hip_speed_smooth = hip_speed
    knee_speed_smooth = knee_speed
    ankle_speed_smooth = ankle_speed
    right_wrist_speed_smooth = right_wrist_speed
    shoulder_vx_smooth = moving_average(shoulder_vx, window)
    hip_vx_smooth = moving_average(hip_vx, window)
    knee_vx_smooth = moving_average(knee_vx, window)
    ankle_vx_smooth = moving_average(ankle_vx, window)
    shoulder_vy_smooth = moving_average(shoulder_vy, window)
    hip_vy_smooth = moving_average(hip_vy, window)
    knee_vy_smooth = moving_average(knee_vy, window)
    ankle_vy_smooth = moving_average(ankle_vy, window)
    
    frames = np.arange(len(knee_angles))
    
    # Create or reuse figure with 3 subplots (angles, speeds, force)
    if fig is None or axes is None:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))
        axes = (ax1, ax2, ax3)
    else:
        ax1, ax2, ax3 = axes
        # Clear existing content including any twin axes
        ax1.clear()
        ax2.clear()
        ax3.clear()
        # Clear any twin axes that may have been created
        for ax in [ax1, ax2, ax3]:
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
    
    # Draw phase backgrounds
    for ax in [ax1, ax2, ax3]:
        current_phase = None
        phase_start = 0
        for i, phase in enumerate(phases + [-1]):  # Add sentinel
            if phase != current_phase:
                if current_phase is not None:
                    color, label = phase_colors.get(current_phase, ('white', f'Phase{current_phase}'))
                    ax.axvspan(phase_start, i - 1, alpha=0.2, color=color, label=label)
                current_phase = phase
                phase_start = i
    
    # Plot 1: Joint angles
    ax1.plot(frames, knee_smooth, 'b-', label='Knee Angle', linewidth=2)
    ax1.plot(frames, hip_smooth, 'r-', label='Hip Angle', linewidth=2)
    ax1.set_ylabel('Angle (degrees)', fontsize=12)
    ax1.set_title('Joint Angles Over Time', fontsize=12)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Joint Speeds (magnitude of velocity = sqrt(vx^2 + vy^2))
    ax2.plot(frames, shoulder_speed_smooth, 'purple', label='Shoulder', linewidth=2)
    ax2.plot(frames, hip_speed_smooth, 'blue', label='Hip', linewidth=2)
    ax2.plot(frames, knee_speed_smooth, 'green', label='Knee', linewidth=2)
    ax2.plot(frames, ankle_speed_smooth, 'orange', label='Ankle', linewidth=2)
    ax2.plot(frames, right_wrist_speed_smooth, 'red', label='Right Wrist', linewidth=2)
    ax2.set_ylabel('Speed (px/s)', fontsize=12)
    ax2.set_title('Joint Speed (√(vx² + vy²)) - Right +, Up +', fontsize=12)
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Joint Accelerations
    ax3.plot(frames, shoulder_accel, 'purple', label='Shoulder', linewidth=2)
    ax3.plot(frames, hip_accel, 'blue', label='Hip', linewidth=2)
    ax3.plot(frames, knee_accel, 'green', label='Knee', linewidth=2)
    ax3.plot(frames, ankle_accel, 'orange', label='Ankle', linewidth=2)
    ax3.plot(frames, right_wrist_accel, 'red', label='Right Wrist', linewidth=2)
    ax3.set_ylabel('Acceleration (px/s²)', fontsize=12)
    ax3.set_title('Joint Acceleration (√(ax² + ay²))', fontsize=12)
    ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax3.grid(True, alpha=0.3)
    
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
            
            # Add velocity overlay for correlation analysis
            # Only use velocities from actual drive phase frames (not interpolated)
            drive_frames_range = range(drive_start, drive_end + 1)
            drive_frame_indices = [i for i in drive_frames_range if i < len(hip_speed_smooth)]
            
            if drive_frame_indices:
                # Plot normalized velocities on secondary y-axis for comparison
                ax4_twin = ax4.twinx()
                max_force = max(force_curve) if force_curve else 1
                
                # Extract only drive phase velocities
                hip_drive = hip_speed_smooth[drive_frame_indices]
                knee_drive = knee_speed_smooth[drive_frame_indices]
                wrist_drive = right_wrist_speed_smooth[drive_frame_indices]
                
                # Plot each joint velocity normalized to force scale
                if len(hip_drive) > 0 and np.max(hip_drive) > 0:
                    normalized_hip = hip_drive / np.max(hip_drive) * max_force
                    ax4_twin.plot(drive_frame_indices, normalized_hip, 
                                 alpha=0.5, linestyle='--', linewidth=1.5, color='blue', label='Hip Speed (norm)')
                
                if len(knee_drive) > 0 and np.max(knee_drive) > 0:
                    normalized_knee = knee_drive / np.max(knee_drive) * max_force
                    ax4_twin.plot(drive_frame_indices, normalized_knee, 
                                 alpha=0.5, linestyle='--', linewidth=1.5, color='green', label='Knee Speed (norm)')
                
                if len(wrist_drive) > 0 and np.max(wrist_drive) > 0:
                    normalized_wrist = wrist_drive / np.max(wrist_drive) * max_force
                    ax4_twin.plot(drive_frame_indices, normalized_wrist, 
                                 alpha=0.5, linestyle='--', linewidth=1.5, color='purple', label='Wrist Speed (norm)')
                
                ax4.set_ylabel('Force (PM5 units)', fontsize=12)
                ax4_twin.set_ylabel('Speed (normalized)', fontsize=10)
                ax4.set_title('Force Curve vs Joint Speeds (Drive Phase)', fontsize=12)
                ax4.legend(loc='upper left')
                ax4_twin.legend(loc='upper right', fontsize=9)
            else:
                ax4.set_ylabel('Force (PM5 units)', fontsize=12)
                ax4.set_title('Force Curve (Drive Phase Only)', fontsize=12)
                ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No drive phase detected', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Force Curve (No Data)', fontsize=12)
    else:
        ax3.text(0.5, 0.5, 'No force data available', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Force Curve (No Data)', fontsize=12)
    
    ax3.set_xlabel('Frame', fontsize=12)
    
    # Highlight buffer zones (first 30 and last 30 frames)
    buffer_size = 30
    for ax in [ax1, ax2, ax3]:
        if len(frames) > buffer_size:
            ax.axvspan(0, buffer_size, alpha=0.1, color='gray', linestyle='--')
            ax.axvspan(len(frames) - buffer_size, len(frames), alpha=0.1, color='gray', linestyle='--')
    
    plt.tight_layout()
    return fig, (ax1, ax2, ax3)


def show_averages(files: List[str]):
    """Show average stroke data across all strokes."""
    if not HAS_MATPLOTLIB:
        print("ERROR: matplotlib not available. Install with: pip3 install matplotlib")
        return None
    
    all_knee_angles = []
    all_hip_angles = []
    """Show average stroke data across all strokes."""
    if not HAS_MATPLOTLIB:
        print("ERROR: matplotlib not available. Install with: pip3 install matplotlib")
        return None
    
    all_knee_angles = []
    all_hip_angles = []
    max_len = 0
    
    for filepath in files:
        data = parse_stroke_file(filepath)
        all_knee_angles.append(data['knee_angles'])
        all_hip_angles.append(data['hip_angles'])
        max_len = max(max_len, len(data['knee_angles']))
    
    # Pad arrays to same length
    for i in range(len(all_knee_angles)):
        if len(all_knee_angles[i]) < max_len:
            padding = np.full(max_len - len(all_knee_angles[i]), np.nan)
            all_knee_angles[i] = np.concatenate([all_knee_angles[i], padding])
            all_hip_angles[i] = np.concatenate([all_hip_angles[i], padding])
    
    # Compute averages
    avg_knee = np.nanmean(all_knee_angles, axis=0)
    avg_hip = np.nanmean(all_hip_angles, axis=0)
    
    # Apply smoothing
    avg_knee_smooth = moving_average(avg_knee, 3)
    avg_hip_smooth = moving_average(avg_hip, 3)
    
    frames = np.arange(len(avg_knee))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(f'Average Joint Angles Across {len(files)} Strokes', fontsize=14)
    
    ax.plot(frames, avg_knee_smooth, 'b-', label='Avg Knee Angle', linewidth=2)
    ax.plot(frames, avg_hip_smooth, 'r-', label='Avg Hip Angle', linewidth=2)
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Angle (degrees)', fontsize=12)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
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
    
    print(f"Found {len(files)} stroke file(s) in {args.data_dir}")
    
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
