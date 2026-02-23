#!/usr/bin/env python3
"""
Wrist Velocity to Stroke Force Correlation Script

This script analyzes the correlation between right wrist velocity and stroke force
from recorded rowing data, accounting for flywheel inertia and drag factor.

Physics Model:  F = A·(dω/dt) + B·ω²
- Force is only applied during positive (leftward) wrist motion
- A = inertial coefficient (≈ I/r_eff)
- B = drag coefficient (≈ c/r_eff), modulated by drag factor and stroke length
- ω = v / r_eff (angular velocity from handle velocity)
- Data loaded from stroke JSON files (same format as analyze_strokes.py)
"""

import os
import sys
import json
import gzip
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from scipy.signal import savgol_filter


# Data folder and file pattern (matches analyze_strokes.py)
DATA_FOLDER = "/tmp/stroke_data"
FILE_PATTERN = "stroke_*.json"


def load_stroke_file(filepath: str) -> dict:
    """Load a stroke JSON file (gzipped or plain)."""
    try:
        with gzip.open(filepath, 'rt') as f:
            return json.load(f)
    except gzip.BadGzipFile:
        with open(filepath, 'r') as f:
            return json.load(f)


def apply_perspective_correction(x: float, y: float, focal_x: float, 
                                 correction_strength: float = 0.0003) -> Tuple[float, float]:
    """
    Apply perspective correction to account for lens focal effect.
    
    Args:
        x, y: Keypoint coordinates
        focal_x: X coordinate of camera focal point (hips at drive start)
        correction_strength: Strength of perspective effect
    
    Returns:
        Corrected x, y coordinates
    """
    dx = x - focal_x
    distortion = dx * correction_strength
    scale_factor = 1.0 + distortion + (distortion * abs(distortion) * 0.5)
    x_corrected = focal_x + dx * scale_factor
    return x_corrected, y


def moving_average(data, window_size=11):
    """Apply moving window average to smooth data (same as analyze_strokes.py)."""
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


def extract_wrist_data_from_file(filepath: str, apply_perspective: bool = True,
                                  handle_offset_m: float = 0.0,
                                  pixels_per_meter: float = 100.0) -> Dict:
    """
    Extract right wrist velocity and force data from stroke JSON file.
    
    Args:
        filepath: Path to stroke JSON file
        apply_perspective: Whether to apply perspective correction (default: True)
        handle_offset_m: Offset from wrist to handle along forearm direction (metres).
                        When > 0, a synthetic 'handle' keypoint is computed by
                        projecting from the wrist along the elbow→wrist direction.
                        Handle velocity is derived from the resulting positions.
                        Typical value for rowing: 0.15 m (15 cm).
        pixels_per_meter: Camera scale for converting Kalman pixel velocities to m/s.
                         100.0 is a rough default; auto-calibrate from PM5 stroke length
                         for accurate physics-based force modelling.
    
    Returns:
        Dictionary containing timestamps, wrist_vx, wrist_vy, force, phases, and fps
    """
    data = load_stroke_file(filepath)
    
    # Handle both old Python format and new C++ format
    frames_array = data.get('frames', [])
    if frames_array:
        # New C++ format
        frame_timestamps = [f.get('timestamp', 0) for f in frames_array]
        keypoints_list = [f.get('keypoints', []) for f in frames_array]
        phases = [f.get('phase', 0) for f in frames_array]
    else:
        # Old Python format
        frame_timestamps = data.get('frame_timestamps', [])
        keypoints_list = data.get('keypoints', [])
        phases = data.get('phases', [])
    
    # Get force curve
    force_curve = data.get('force', [])
    
    # Detect actual FPS from timestamps
    detected_fps = 60.0  # Default fallback
    if len(frame_timestamps) >= 10:
        intervals = [frame_timestamps[i+1] - frame_timestamps[i] 
                    for i in range(min(100, len(frame_timestamps)-1))]
        valid_intervals = [x for x in intervals if x > 0]
        if valid_intervals:
            avg_interval = sum(valid_intervals) / len(valid_intervals)
            detected_fps = 1.0 / avg_interval if avg_interval > 0 else 60.0
    
    # Find focal point (hip position at drive start)
    focal_point_x = None
    for i, (frame_kpts, phase) in enumerate(zip(keypoints_list, phases)):
        if phase == 2:  # DRIVE phase start
            kp_dict_temp = {}
            for kp in frame_kpts:
                kp_dict_temp[kp['name']] = (kp['x'], kp['y'])
            hip_coords_temp = kp_dict_temp.get('left_hip') or kp_dict_temp.get('right_hip')
            if hip_coords_temp:
                focal_point_x = hip_coords_temp[0]
                break
    
    if focal_point_x is None and keypoints_list:
        focal_point_x = 320  # Default camera center
    
    # Extract ALL keypoint velocities (not just wrist)
    # Keys: keypoint name -> lists of (vx, vy) per frame
    kp_vx_lists: Dict[str, list] = {}  # keypoint_name -> [vx_frame0, vx_frame1, ...]
    kp_vy_lists: Dict[str, list] = {}
    kp_x_lists: Dict[str, list] = {}   # positions for reference
    kp_y_lists: Dict[str, list] = {}   # y positions (needed for handle offset)
    
    for frame_kpts in keypoints_list:
        kp_dict = {}
        kp_vel_dict = {}
        
        for kp in frame_kpts:
            # Apply perspective correction
            x_corrected, y_corrected = apply_perspective_correction(
                kp['x'], kp['y'], focal_point_x
            )
            kp_dict[kp['name']] = (x_corrected, y_corrected)
            
            # Get Kalman velocities with perspective correction
            if 'vx' in kp and 'vy' in kp:
                if apply_perspective:
                    dx = kp['x'] - focal_point_x
                    distortion = dx * 0.0003
                    scale_factor = 1.0 + distortion + (distortion * abs(distortion) * 0.5)
                    kp_vel_dict[kp['name']] = (kp['vx'] * scale_factor, kp['vy'] * scale_factor)
                else:
                    # Use raw Kalman velocities without perspective correction
                    kp_vel_dict[kp['name']] = (kp['vx'], kp['vy'])
        
        # Collect data for ALL keypoints seen in this frame
        all_kp_names = set(kp_dict.keys()) | set(kp_vel_dict.keys())
        for name in all_kp_names:
            if name not in kp_vx_lists:
                # First time seeing this keypoint — backfill with NaN
                kp_vx_lists[name] = [np.nan] * len(kp_vx_lists.get('right_wrist', []))
                kp_vy_lists[name] = [np.nan] * len(kp_vy_lists.get('right_wrist', []))
                kp_x_lists[name] = [np.nan] * len(kp_x_lists.get('right_wrist', []))
                kp_y_lists[name] = [np.nan] * len(kp_y_lists.get('right_wrist', []))
        
        for name in kp_vx_lists:
            coords = kp_dict.get(name)
            vel = kp_vel_dict.get(name)
            kp_x_lists[name].append(coords[0] if coords else np.nan)
            kp_y_lists[name].append(coords[1] if coords else np.nan)
            kp_vx_lists[name].append(vel[0] if vel else np.nan)
            kp_vy_lists[name].append(vel[1] if vel else np.nan)
    
    # Build per-keypoint velocity arrays (m/s)
    keypoint_velocities = {}  # name -> {'vx': array, 'speed': array}
    for name in kp_vx_lists:
        vx = np.array(kp_vx_lists[name])
        vy = -np.array(kp_vy_lists[name])  # Flip Y (same as analyze_strokes)
        speed = np.sqrt(vx**2 + vy**2)
        keypoint_velocities[name] = {
            'vx': vx / pixels_per_meter,       # m/s
            'speed': speed / pixels_per_meter,  # m/s
        }

    # Build per-keypoint position arrays (pixels, perspective-corrected)
    keypoint_positions = {}  # name -> {'x': array, 'y': array} in pixel coords
    for name in kp_x_lists:
        keypoint_positions[name] = {
            'x': np.array(kp_x_lists[name]),
            'y': np.array(kp_y_lists[name]),
        }

    # --- Synthetic 'handle' keypoint from wrist + forearm offset ---
    if handle_offset_m > 0:
        wrist_x = np.array(kp_x_lists.get('right_wrist', []), dtype=np.float64)
        wrist_y = np.array(kp_y_lists.get('right_wrist', []), dtype=np.float64)
        elbow_x = np.array(kp_x_lists.get('right_elbow', []), dtype=np.float64)
        elbow_y = np.array(kp_y_lists.get('right_elbow', []), dtype=np.float64)

        if len(wrist_x) == len(elbow_x) and len(wrist_x) > 0:
            # Purely horizontal offset: wrist bends to keep handle level
            dx = wrist_x - elbow_x
            sign_x = np.sign(dx)
            sign_x[sign_x == 0] = 1.0  # default rightward if aligned

            # Handle position = wrist + horizontal offset (pixels)
            offset_px = handle_offset_m * pixels_per_meter
            handle_x = wrist_x + offset_px * sign_x
            handle_y = wrist_y  # same Y as wrist (horizontal)

            # Handle velocity = extra-smoothed wrist velocity (no delay)
            from scipy.ndimage import uniform_filter1d
            wrist_vel = keypoint_velocities.get('right_wrist', {})
            handle_vx_ms = uniform_filter1d(wrist_vel.get('vx', np.zeros(len(wrist_x))).copy(), 21)
            handle_speed_ms = uniform_filter1d(wrist_vel.get('speed', np.zeros(len(wrist_x))).copy(), 21)

            keypoint_velocities['handle'] = {
                'vx': handle_vx_ms,
                'speed': handle_speed_ms,
            }

    # For backward compat, also expose right_wrist directly
    wrist_data = keypoint_velocities.get('right_wrist', {})
    right_wrist_vx_ms = wrist_data.get('vx', np.zeros(len(frame_timestamps)))
    right_wrist_speed_ms = wrist_data.get('speed', np.zeros(len(frame_timestamps)))
    
    # Extract PM5 ergometer metadata
    drag_factor_pm5 = data.get('drag_factor', 0)  # Concept2 units (typ. 80-220)
    stroke_stats = data.get('stroke_stats', {})
    stroke_length_cm = stroke_stats.get('stroke_length', 0)  # centimetres
    stroke_length_m = stroke_length_cm / 100.0 if stroke_length_cm else 0.0
    drive_time = stroke_stats.get('drive_time', 0)
    recovery_time = stroke_stats.get('recovery_time', 0)
    pm5_peak_force = stroke_stats.get('peak_force', 0)
    pm5_avg_force = stroke_stats.get('avg_force', 0)
    work_per_stroke = stroke_stats.get('work_per_stroke', 0)

    # Estimate flywheel RPM from wrist velocity and effective gear ratio
    # ω = |v_handle| / r_eff, RPM = ω × 60 / (2π)
    r_eff = 0.0175 * 2.4  # sprocket_radius × gear_ratio (Concept2 default)
    flywheel_omega = np.abs(right_wrist_vx_ms) / r_eff  # rad/s
    flywheel_rpm = flywheel_omega * 60.0 / (2.0 * np.pi)

    return {
        'timestamps': np.array(frame_timestamps),
        'wrist_vx': right_wrist_vx_ms,  # Horizontal velocity in m/s (backward compat)
        'wrist_speed': right_wrist_speed_ms,  # Total speed in m/s (backward compat)
        'keypoint_velocities': keypoint_velocities,  # ALL keypoints: name -> {vx, speed}
        'keypoint_positions': keypoint_positions,     # ALL keypoints: name -> {x, y} in pixels
        'force': np.array(force_curve),
        'phases': np.array(phases),
        'fps': detected_fps,
        'filename': os.path.basename(filepath),
        # PM5 ergometer metadata
        'drag_factor': drag_factor_pm5,
        'stroke_length': stroke_length_m,  # metres
        'stroke_stats': stroke_stats,
        'flywheel_rpm': flywheel_rpm,  # estimated RPM per frame
    }


@dataclass
class FlywheelParameters:
    """Physical parameters for the flywheel model"""
    inertia: float = 0.1001  # kg·m² (Concept2 standard)
    drag_factor: float = 150.0  # Drag coefficient (Concept2 units, ×10⁻⁶ N·m·s²)
    radius: float = 0.15  # meters (flywheel radius)
    force_coefficient: float = 1.5  # Force scaling: F = coefficient * v² (FIXED)
    initial_angular_velocity: float = 0.0  # rad/s
    sprocket_radius: float = 0.0175  # meters (Concept2 chain sprocket radius)
    gear_ratio: float = 2.4  # Effective gear ratio (handle → flywheel)
    stroke_length: float = 0.0  # meters (PM5-reported stroke length, 0 = unknown)
    drag_factor_pm5: int = 0  # PM5-reported drag factor (0 = unknown)

    @property
    def effective_radius(self) -> float:
        """Effective radius: r = sprocket_radius × gear_ratio"""
        return self.sprocket_radius * self.gear_ratio

    def physics_coefficients(self) -> Tuple[float, float]:
        """Compute A, B from physical parameters.

        The force model is F = A·(dω/dt) + B·ω², derived from:
            T = I·(dω/dt) + k·ω²      (flywheel torque equation)
            F = T / r_eff               (chain force)

        So:  A = I / r_eff    B = k / r_eff

        where r_eff = effective_radius, k = drag_factor_pm5 × 1e-6 (SI).
        If drag_factor_pm5 is unknown (0), B = 0.
        """
        r = self.effective_radius
        A_phys = self.inertia / r
        if self.drag_factor_pm5 > 0:
            c_si = self.drag_factor_pm5 * 1e-6  # Convert Concept2 units → N·m·s²
            B_phys = c_si / r
        else:
            B_phys = 0.0
        return A_phys, B_phys


class RowingDynamicsSimulator:
    """
    Simulates rowing dynamics relating wrist velocity to stroke force
    
    Uses simplified model: F = k * v²
    where v is handle velocity magnitude during drive phase
    """
    
    def __init__(self, flywheel_params: FlywheelParameters, dt: float = 0.01):
        """
        Initialize the simulator
        
        Args:
            flywheel_params: Flywheel physical parameters
            dt: Time step for simulation (seconds)
        """
        self.params = flywheel_params
        self.dt = dt
        self.angular_velocity = flywheel_params.initial_angular_velocity
        
    def calculate_drag_torque(self, angular_velocity: float) -> float:
        """
        Calculate drag torque on flywheel (not used in simplified model)
        
        Torque = drag_factor * omega^2 (sign preserved)
        
        Args:
            angular_velocity: Current angular velocity (rad/s)
            
        Returns:
            Drag torque (N·m)
        """
        if angular_velocity == 0:
            return 0.0
        sign = np.sign(angular_velocity)
        return -sign * self.params.drag_factor * angular_velocity**2
    
    def calculate_drive_torque(self, handle_velocity: float) -> float:
        """
        Calculate driving torque from handle velocity
        
        Only applies when handle is moving LEFT (positive velocity) - the drive/pull phase
        Torque from handle driving the flywheel
        
        Args:
            handle_velocity: Linear velocity of handle/wrist (m/s)
            
        Returns:
            Drive torque (N·m)
        """
        # Force only applied when moving left (toward rower) = positive velocity
        if handle_velocity <= 0:
            return 0.0
        
        # Calculate expected angular velocity from handle velocity
        # Positive handle velocity (leftward) creates positive angular velocity (forward spin)
        target_angular_velocity = handle_velocity / self.params.radius
        
        # Drive torque proportional to handle velocity
        # Using a coupling coefficient to model chain/handle connection
        coupling_coefficient = 50.0  # N·m·s/rad
        drive_torque = coupling_coefficient * target_angular_velocity
        
        return drive_torque
    
    def update_flywheel(self, handle_velocity: float) -> Tuple[float, float]:
        """
        Update flywheel state based on handle velocity
        
        Args:
            handle_velocity: Linear velocity of handle/wrist (m/s)
            
        Returns:
            Tuple of (angular_acceleration, total_torque)
        """
        # Calculate torques
        drive_torque = self.calculate_drive_torque(handle_velocity)
        drag_torque = self.calculate_drag_torque(self.angular_velocity)
        
        total_torque = drive_torque + drag_torque
        
        # Calculate angular acceleration: τ = I·α
        angular_acceleration = total_torque / self.params.inertia
        
        # Update angular velocity
        self.angular_velocity += angular_acceleration * self.dt
        
        # Prevent negative angular velocity (flywheel only spins one direction)
        if self.angular_velocity < 0:
            self.angular_velocity = 0.0
        
        return angular_acceleration, total_torque
    
    def calculate_stroke_force(self, handle_velocity: float) -> float:
        """
        Calculate stroke force from handle velocity using simplified model
        
        Physics basis: F ∝ v² during drive phase
        - During drive (positive velocity): rower pulls left, force applied
        - During recovery (negative velocity): no force
        
        Args:
            handle_velocity: Linear velocity of handle/wrist (m/s)
            
        Returns:
            Stroke force (N)
        """
        # Force only applied during drive phase (positive velocity = pulling leftward)
        if handle_velocity <= 0:
            return 0.0
        
        # Force proportional to velocity squared: F = k * v²
        force = self.params.force_coefficient * handle_velocity ** 2
        
        return force
    
    def simulate_stroke(self, 
                       time_array: np.ndarray, 
                       velocity_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate a complete rowing stroke using simplified force model
        
        Args:
            time_array: Time points (seconds)
            velocity_array: Handle/wrist velocity at each time point (m/s)
            
        Returns:
            Tuple of (force_array, angular_velocity_array, power_array)
            Note: angular_velocity_array calculated from v = ω*r for display purposes only
        """
        force_array = np.zeros_like(velocity_array)
        angular_velocity_array = np.zeros_like(velocity_array)
        power_array = np.zeros_like(velocity_array)
        
        for i, velocity in enumerate(velocity_array):
            force = self.calculate_stroke_force(velocity)
            force_array[i] = force
            # For display: approximate angular velocity from linear velocity
            # Assuming v = ω * r, so ω = v / r
            angular_velocity_array[i] = abs(velocity) / 0.15  # Using typical radius
            # Power is force × speed
            power_array[i] = force * velocity
        
        return force_array, angular_velocity_array, power_array


def generate_sample_wrist_velocity(duration: float = 2.0, 
                                   dt: float = 0.01,
                                   stroke_rate: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sample wrist velocity data for a rowing stroke
    
    Args:
        duration: Total duration (seconds)
        dt: Time step (seconds)
        stroke_rate: Strokes per minute
        
    Returns:
        Tuple of (time_array, velocity_array)
    """
    time_array = np.arange(0, duration, dt)
    velocity_array = np.zeros_like(time_array)
    
    # Stroke period
    stroke_period = 60.0 / stroke_rate
    drive_ratio = 0.4  # Drive phase is 40% of stroke cycle
    drive_duration = stroke_period * drive_ratio
    
    for i, t in enumerate(time_array):
        # Determine phase within stroke cycle
        cycle_time = t % stroke_period
        
        if cycle_time < drive_duration:
            # Drive phase: positive velocity (pulling leftward toward rower)
            phase = cycle_time / drive_duration
            # Bell curve shape for velocity
            velocity_array[i] = 1.5 * np.sin(np.pi * phase)
        else:
            # Recovery phase: negative velocity (return rightward)
            recovery_time = cycle_time - drive_duration
            recovery_duration = stroke_period - drive_duration
            phase = recovery_time / recovery_duration
            velocity_array[i] = -0.8 * np.sin(np.pi * phase)
    
    return time_array, velocity_array


def plot_results(time: np.ndarray, 
                velocity: np.ndarray, 
                force: np.ndarray,
                angular_velocity: np.ndarray,
                power: np.ndarray):
    """
    Plot simulation results
    
    Args:
        time: Time array
        velocity: Handle velocity array
        force: Stroke force array
        angular_velocity: Flywheel angular velocity array
        power: Power output array
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    
    # Velocity plot
    axes[0].plot(time, velocity, 'b-', linewidth=2)
    axes[0].set_ylabel('Handle Velocity (m/s)')
    axes[0].set_title('Wrist/Handle Velocity')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Force plot
    axes[1].plot(time, force, 'r-', linewidth=2)
    axes[1].set_ylabel('Stroke Force (N)')
    axes[1].set_title('Calculated Stroke Force')
    axes[1].grid(True, alpha=0.3)
    axes[1].fill_between(time, 0, force, alpha=0.3, color='red')
    
    # Angular velocity plot
    axes[2].plot(time, angular_velocity, 'g-', linewidth=2)
    axes[2].set_ylabel('Angular Velocity (rad/s)')
    axes[2].set_title('Flywheel Angular Velocity')
    axes[2].grid(True, alpha=0.3)
    
    # Power plot
    axes[3].plot(time, power, 'm-', linewidth=2)
    axes[3].set_ylabel('Power (W)')
    axes[3].set_xlabel('Time (s)')
    axes[3].set_title('Power Output')
    axes[3].grid(True, alpha=0.3)
    axes[3].fill_between(time, 0, power, alpha=0.3, color='magenta')
    
    plt.tight_layout()
    print("Displaying synthetic data plots...")
    plt.show(block=True)


def calculate_stroke_metrics(force: np.ndarray, 
                             power: np.ndarray, 
                             velocity: np.ndarray,
                             dt: float,
                             vx_threshold: float = 0.0) -> dict:
    """
    Calculate stroke performance metrics
    
    Args:
        force: Force array
        power: Power array
        velocity: Velocity array
        dt: Time step
        vx_threshold: Velocity threshold for drive detection (m/s)
        
    Returns:
        Dictionary of metrics
    """
    # Find drive phases (positive velocity = pulling leftward, above threshold)
    drive_mask = velocity > vx_threshold
    
    metrics = {
        'peak_force': np.max(force),
        'average_force': np.mean(force[drive_mask]) if np.any(drive_mask) else 0,
        'peak_power': np.max(power),
        'average_power': np.mean(power[drive_mask]) if np.any(drive_mask) else 0,
        'total_work': np.sum(power) * dt,  # Joules
        'drive_time': np.sum(drive_mask) * dt,
    }
    
    return metrics


def compute_com_acceleration(keypoint_velocities: Dict, timestamps: np.ndarray,
                             smooth_window: int = 21) -> Optional[np.ndarray]:
    """Compute whole-body CoM horizontal acceleration from Kalman-filtered keypoint velocities.

    Uses Dumas et al. (2007) anthropometric regression for males, simplified
    for the 5 available rowing keypoints (right_shoulder, right_hip,
    right_knee, right_ankle, right_wrist).

    Each segment's CoM velocity is the weighted interpolation of its
    proximal and distal joint Kalman velocities (already in m/s).  The
    whole-body CoM velocity is the mass-weighted sum across segments.
    A single differentiation then gives acceleration.

    Velocities are expected to be pre-smoothed (same filter as handle velocity)
    so that acceleration derived here is consistent with handle acceleration.

    Segment model (mass fractions from Dumas et al. Table 2, males):
        head+neck  6.7%  — at shoulder (proxy, no head marker)
        torso     33.3%  — 42% from shoulder toward hip
        pelvis    14.2%  — at hip
        arms       9.4%  — 50% from shoulder toward wrist (both arms combined,
                           upper arm 2.4%×2 + forearm 1.7%×2 + hand 0.6%×2)
        thighs    24.6%  — 43% from hip toward knee (both legs, 12.3%×2)
        shanks     9.6%  — 41% from knee toward ankle (both legs, 4.8%×2)
        feet       2.8%  — at ankle (proxy, 1.4%×2 from de Leva 1996)

    Args:
        keypoint_velocities: dict of name → {'vx': array, 'speed': array} in m/s
                             (pre-smoothed with same filter as handle velocity)
        timestamps:          time array (s) matching velocity arrays
        smooth_window:       Savitzky-Golay window for smoothing acceleration

    Returns:
        CoM horizontal acceleration (m/s²), positive = leftward (drive direction),
        same length as timestamps.  None if required keypoints are missing.
    """
    # Dumas et al. (2007) male segment model for 5 keypoints
    # (name, proximal_kp, distal_kp, mass_fraction, com_position_along_segment)
    SEGMENTS = [
        ('head+neck', 'right_shoulder', None,           0.067, 0.0),
        ('torso',     'right_shoulder', 'right_hip',    0.333, 0.42),
        ('pelvis',    'right_hip',      None,           0.142, 0.0),
        ('arms',      'right_shoulder', 'right_wrist',  0.094, 0.50),
        ('thighs',    'right_hip',      'right_knee',   0.246, 0.429),
        ('shanks',    'right_knee',     'right_ankle',  0.096, 0.41),
        ('feet',      'right_ankle',    None,           0.028, 0.0),
    ]

    total_mass_frac = sum(s[3] for s in SEGMENTS)  # ≈1.006, normalise

    required = {'right_shoulder', 'right_hip', 'right_knee', 'right_ankle', 'right_wrist'}
    available = set(keypoint_velocities.keys())
    if not required.issubset(available):
        missing = required - available
        print(f"  WARNING: Missing keypoints for CoM: {missing}")
        return None

    # Horizontal velocity arrays (m/s, pre-smoothed)
    vel_x = {name: keypoint_velocities[name]['vx'] for name in required}

    n = len(timestamps)
    for name in required:
        if len(vel_x[name]) != n:
            print(f"  WARNING: Keypoint {name} has {len(vel_x[name])} frames, expected {n}")
            return None

    # Weighted CoM horizontal velocity (m/s)
    # Each segment CoM velocity = (1-f)*v_proximal + f*v_distal
    com_vx = np.zeros(n)
    for (_name, prox, dist, mass_frac, com_frac) in SEGMENTS:
        w = mass_frac / total_mass_frac
        if dist is None:
            com_vx += w * vel_x[prox]
        else:
            com_vx += w * ((1.0 - com_frac) * vel_x[prox] + com_frac * vel_x[dist])

    # Interpolate NaN gaps (missing keypoint detections)
    nan_mask = np.isnan(com_vx)
    if np.any(nan_mask) and not np.all(nan_mask):
        good = ~nan_mask
        com_vx[nan_mask] = np.interp(
            np.where(nan_mask)[0], np.where(good)[0], com_vx[good])

    # Single differentiation for acceleration (velocities are pre-smoothed)
    com_ax = np.gradient(com_vx, timestamps)

    # Smooth the acceleration
    win = min(smooth_window, len(com_ax))
    if win % 2 == 0:
        win -= 1
    if win >= 5:
        com_ax = savgol_filter(com_ax, win, 3)

    # Negate: vx from Kalman is rightward-positive, our convention is positive = leftward (drive)
    com_ax = -com_ax

    return com_ax


def compute_force_model(wrist_vx: np.ndarray,
                        accel: np.ndarray,
                        A: float,
                        B: float,
                        drag_factor_pm5: int = 0,
                        stroke_length: float = 0.0,
                        flywheel_params: Optional[FlywheelParameters] = None,
                        dt: Optional[float] = None,
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Flywheel force model: F_handle = A·(dω/dt) + B·ω²

    Computes the handle/chain force from wrist velocity and acceleration.
    This is purely the flywheel-side equation — the force the handle
    transmits to the flywheel through the chain.

    The rower's body dynamics are handled separately via the Colloud
    equation: F_handle + F_stretcher = m·a_CoM  (Newton's 2nd law for
    the rower's body, applied when --rower-mass > 0).

        ω = v / r_eff           (handle angular velocity)
        α = (dv/dt) / r_eff     (handle angular acceleration)
        F = A·α + B·ω²          (clamped to ≥ 0)

    Args:
        wrist_vx:        Smoothed wrist velocity (m/s), positive = drive (leftward)
        accel:           Smoothed acceleration (m/s²), i.e. dv/dt
        A:               Inertial coefficient (physically ≈ I / r_eff)
        B:               Drag coefficient (physically ≈ k / r_eff)
        drag_factor_pm5: PM5-reported drag factor (Concept2 units, 0 = ignore)
        stroke_length:   PM5-reported stroke length in metres (0 = ignore)
        flywheel_params: FlywheelParameters for r_eff
        dt:              Time step between frames in seconds (unused, kept for API compat)

    Returns:
        (force_model, force_inertial, force_drag, clutch_state)  all same shape as wrist_vx
    """
    n = len(wrist_vx)
    force_model = np.zeros(n)
    force_inertial = np.zeros(n)
    force_drag = np.zeros(n)
    clutch_state = np.ones(n, dtype=bool)  # always "engaged"

    if flywheel_params is not None:
        r_eff = flywheel_params.effective_radius
    else:
        r_eff = 0.0175 * 2.4  # default

    # Stroke-length mechanical advantage correction
    ref_stroke_length = 1.40
    stroke_scale = 1.0
    if stroke_length > 0:
        stroke_scale = (stroke_length / ref_stroke_length) ** 0.3

    for i in range(n):
        vel = wrist_vx[i]
        if vel > 0 and r_eff > 0:
            omega = vel / r_eff
            alpha = accel[i] / r_eff
            inertial = A * alpha
            drag = B * omega**2 * stroke_scale
            f_total = inertial + drag
            force_inertial[i] = inertial
            force_drag[i] = drag
            force_model[i] = max(0.0, f_total)

    return force_model, force_inertial, force_drag, clutch_state


def optimize_model_parameters(wrist_vx: np.ndarray,
                              accel: np.ndarray,
                              force_target: np.ndarray,
                              timestamps: np.ndarray,
                              initial_A: float = 1.0,
                              initial_B: float = 1.5,
                              drag_factor_pm5: int = 0,
                              stroke_length: float = 0.0,
                              flywheel_params: Optional[FlywheelParameters] = None,
                              dt: Optional[float] = None,
                              vx_threshold: float = 0.0,
                              ) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Optimize A and B to best fit the force model to measured PM5 force.

        F = A·(dω/dt) + B·ω²

    Uses scipy.optimize.least_squares (Trust Region Reflective) with
    non-negative bounds on A and B.  Runs multiple starts from a grid
    of initial guesses spanning several orders of magnitude to avoid
    local minima.

    Args:
        wrist_vx:          Smoothed wrist velocity on analysis timestamps
        accel:             Smoothed acceleration (dv/dt) on analysis timestamps
        force_target:      PM5 force interpolated onto analysis timestamps
        timestamps:        Analysis timestamp grid
        initial_A:         Starting guess for inertial coefficient (≈ I/r_eff)
        initial_B:         Starting guess for drag coefficient (≈ k/r_eff)
        dt:                Time step between frames (None = auto-detect)

    Returns:
        (opt_params dict, force_model, force_inertial, force_drag, clutch_state)
    """
    from scipy.optimize import least_squares

    if dt is None:
        if len(timestamps) >= 2:
            dt = float(np.median(np.diff(timestamps)))
        else:
            dt = 1.0 / 60.0

    n = len(force_target)

    def residuals(params):
        """Compute residual vector for least-squares."""
        A_c = params[0]
        B_c = params[1]
        fm, _, _, _ = compute_force_model(
            wrist_vx, accel, A_c, B_c,
            drag_factor_pm5=drag_factor_pm5,
            stroke_length=stroke_length,
            flywheel_params=flywheel_params,
            dt=dt)
        return fm - force_target

    def cost(params):
        return float(np.sum(residuals(params)**2))

    # Bounds: A >= 0, B >= 0
    lb = [0.0, 0.0]
    ub = [np.inf, np.inf]

    # Multi-start grid: span several orders of magnitude around initial guess
    A_starts = [initial_A * f for f in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]]
    B_starts = [initial_B * f for f in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]]

    print(f"  Optimizing A, B (least-squares, multi-start)...")
    print(f"    Initial guess: A={initial_A:.6f}, B={initial_B:.6f}")
    print(f"    Grid: {len(A_starts)} × {len(B_starts)} = "
          f"{len(A_starts)*len(B_starts)} starts")

    best_cost = np.inf
    best_result = None
    best_x0 = None
    n_starts = 0

    for A0 in A_starts:
        for B0 in B_starts:
            x0 = [max(1e-12, A0), max(1e-12, B0)]
            try:
                result = least_squares(
                    residuals, x0,
                    bounds=(lb, ub),
                    method='trf',
                    ftol=1e-10,
                    xtol=1e-10,
                    gtol=1e-10,
                    max_nfev=5000,
                )
                c = result.cost  # 0.5 * sum(residuals²)
                n_starts += 1
                if c < best_cost:
                    best_cost = c
                    best_result = result
                    best_x0 = x0
            except Exception:
                pass

    if best_result is None:
        # Fallback: single run from initial guess
        x0 = [max(1e-12, initial_A), max(1e-12, initial_B)]
        best_result = least_squares(residuals, x0, bounds=(lb, ub), method='trf',
                                    ftol=1e-8, xtol=1e-8, max_nfev=5000)
        best_x0 = x0

    A_opt = best_result.x[0]
    B_opt = best_result.x[1]

    opt_params = {
        'inertia_coefficient': A_opt,
        'force_coefficient': B_opt,
        'delay': 0.0,
    }

    force_model_opt, force_inertial_opt, force_drag_opt, clutch_state_opt = compute_force_model(
        wrist_vx, accel, A_opt, B_opt,
        drag_factor_pm5=drag_factor_pm5,
        stroke_length=stroke_length,
        flywheel_params=flywheel_params,
        dt=dt,
    )

    # Fit quality metrics
    rmse = np.sqrt(np.mean((force_target - force_model_opt)**2))
    ss_res = np.sum((force_target - force_model_opt)**2)
    ss_tot = np.sum((force_target - np.mean(force_target))**2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    peak_actual = np.max(force_target) if len(force_target) > 0 else 1.0
    peak_model = np.max(force_model_opt) if len(force_model_opt) > 0 else 0.0
    peak_ratio = peak_model / peak_actual if peak_actual > 0 else 0.0

    # Shape correlation (how well do the curves track each other?)
    drive_mask = wrist_vx > vx_threshold
    if np.sum(drive_mask) >= 3:
        shape_corr = np.corrcoef(force_target[drive_mask], force_model_opt[drive_mask])[0, 1]
    else:
        shape_corr = 0.0

    # Physics comparison
    r_eff = flywheel_params.effective_radius if flywheel_params else 0.042
    A_phys = flywheel_params.inertia / r_eff if flywheel_params else 0.0
    c_drag = (drag_factor_pm5 * 1e-6) if drag_factor_pm5 > 0 else 0.0
    B_phys = c_drag / r_eff if r_eff > 0 else 0.0

    print(f"  Optimization complete ({n_starts} converged starts):")
    print(f"    A (inertia)    = {A_opt:.6f}  (physics: {A_phys:.6f}, ratio: {A_opt/A_phys:.3f}×)" if A_phys > 0
          else f"    A (inertia)    = {A_opt:.6f}")
    print(f"    B (drag)       = {B_opt:.6f}  (physics: {B_phys:.6f}, ratio: {B_opt/B_phys:.3f}×)" if B_phys > 0
          else f"    B (drag)       = {B_opt:.6f}")
    print(f"    RMSE           = {rmse:.2f} N")
    print(f"    R²             = {r_squared:.4f}")
    print(f"    Shape corr     = {shape_corr:.4f}")
    print(f"    Peak ratio     = {peak_ratio:.3f}  (model {peak_model:.1f} / actual {peak_actual:.1f} N)")
    if flywheel_params:
        print(f"    Flywheel I={flywheel_params.inertia}, r_eff={r_eff:.4f} m")

    return opt_params, force_model_opt, force_inertial_opt, force_drag_opt, clutch_state_opt


def optimize_physics_constrained(wrist_vx: np.ndarray,
                                 accel: np.ndarray,
                                 force_target: np.ndarray,
                                 timestamps: np.ndarray,
                                 flywheel_params: FlywheelParameters,
                                 drag_factor_pm5: int = 0,
                                 stroke_length: float = 0.0,
                                 dt: Optional[float] = None,
                                 vx_threshold: float = 0.0,
                                 ) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Physics-constrained optimisation: find velocity scale s.

    The physics determines the A/B relationship exactly:
        A_phys = I / r_eff
        B_phys = c_drag / r_eff

    A velocity scaling error s transforms:
        v_true = s · v_cam
        a_true = s · a_cam

    Substituting into F = A_phys · α + B_phys · ω²:
        F = A_phys · (s·a/r) + B_phys · (s·v/r)²
          = s · A_phys · α  +  s² · B_phys · ω²

    So A_eff = s · A_phys, B_eff = s² · B_phys.
    This preserves the physics structure with a single scale degree of freedom.

    Parameters optimised: s (velocity scale).

    Returns:
        (opt_params dict, force_model, force_inertial, force_drag, clutch_state)
    """
    from scipy.optimize import least_squares

    if dt is None:
        if len(timestamps) >= 2:
            dt = float(np.median(np.diff(timestamps)))
        else:
            dt = 1.0 / 60.0

    r_eff = flywheel_params.effective_radius
    A_phys = flywheel_params.inertia / r_eff if r_eff > 0 else 1.0
    c_drag = (drag_factor_pm5 * 1e-6) if drag_factor_pm5 > 0 else 0.0
    B_phys = c_drag / r_eff if r_eff > 0 else 0.0

    def residuals(params):
        s = params[0]
        A_eff = s * A_phys
        B_eff = s * s * B_phys
        fm, _, _, _ = compute_force_model(
            wrist_vx, accel, A_eff, B_eff,
            drag_factor_pm5=drag_factor_pm5,
            stroke_length=stroke_length,
            flywheel_params=flywheel_params,
            dt=dt)
        return fm - force_target

    # Bounds: s > 0
    lb = [1e-6]
    ub = [100.0]

    # Multi-start grid
    s_starts = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

    print(f"  Physics-constrained optimisation (velocity scale s)...")
    print(f"    A_phys = I/r_eff = {A_phys:.6f}")
    print(f"    B_phys = c/r_eff = {B_phys:.6f}")
    print(f"    Grid: {len(s_starts)} starts")

    best_cost = np.inf
    best_result = None
    n_starts = 0

    for s0 in s_starts:
        x0 = [s0]
        try:
            result = least_squares(
                residuals, x0,
                bounds=(lb, ub),
                method='trf',
                ftol=1e-10, xtol=1e-10, gtol=1e-10,
                max_nfev=5000,
            )
            n_starts += 1
            if result.cost < best_cost:
                best_cost = result.cost
                best_result = result
        except Exception:
            pass

    if best_result is None:
        best_result = least_squares(residuals, [1.0],
                                    bounds=(lb, ub), method='trf', max_nfev=5000)

    s_opt = best_result.x[0]
    A_opt = s_opt * A_phys
    B_opt = s_opt * s_opt * B_phys

    opt_params = {
        'inertia_coefficient': A_opt,
        'force_coefficient': B_opt,
        'delay': 0.0,
        'velocity_scale': s_opt,
    }

    force_model_opt, force_inertial_opt, force_drag_opt, clutch_state_opt = compute_force_model(
        wrist_vx, accel, A_opt, B_opt,
        drag_factor_pm5=drag_factor_pm5,
        stroke_length=stroke_length,
        flywheel_params=flywheel_params,
        dt=dt,
    )

    rmse = np.sqrt(np.mean((force_target - force_model_opt)**2))
    ss_res = np.sum((force_target - force_model_opt)**2)
    ss_tot = np.sum((force_target - np.mean(force_target))**2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    peak_actual = np.max(force_target) if len(force_target) > 0 else 1.0
    peak_model = np.max(force_model_opt) if len(force_model_opt) > 0 else 0.0
    peak_ratio = peak_model / peak_actual if peak_actual > 0 else 0.0

    drive_mask = wrist_vx > vx_threshold
    shape_corr = (np.corrcoef(force_target[drive_mask], force_model_opt[drive_mask])[0, 1]
                  if np.sum(drive_mask) >= 3 else 0.0)

    print(f"  Constrained optimisation complete ({n_starts} converged starts):")
    print(f"    velocity scale = {s_opt:.6f}")
    print(f"    A_eff = s·A_phys = {A_opt:.6f}  (A_phys={A_phys:.6f})")
    print(f"    B_eff = s²·B_phys = {B_opt:.6f}  (B_phys={B_phys:.6f})")
    print(f"    A/B ratio        = {A_opt/B_opt:.1f}  (physics: {A_phys/B_phys:.1f})" if B_opt > 0
          else f"    A/B ratio        = inf  (B=0)")
    print(f"    RMSE             = {rmse:.2f} N")
    print(f"    R²               = {r_squared:.4f}")
    print(f"    Shape corr       = {shape_corr:.4f}")
    print(f"    Peak ratio       = {peak_ratio:.3f}  (model {peak_model:.1f} / actual {peak_actual:.1f} N)")

    print(f"    Shape corr       = {shape_corr:.4f}")
    print(f"    Peak ratio       = {peak_ratio:.3f}  (model {peak_model:.1f} / actual {peak_actual:.1f} N)")

    return opt_params, force_model_opt, force_inertial_opt, force_drag_opt, clutch_state_opt


# ---------------------------------------------------------------------------
# Global (multi-stroke) optimisation
# ---------------------------------------------------------------------------

def optimize_global(stroke_data_list: list,
                    initial_A: float = 1.0,
                    initial_B: float = 1.5,
                    vx_threshold: float = 0.0,
                    ) -> Dict:
    """Fit a single (A, B) across ALL strokes simultaneously.

    Each stroke contributes its own residual vector; the optimizer minimises
    the concatenated residual over every stroke at once.

    Args:
        stroke_data_list:  List of dicts from analyze_single_stroke (data-only pass).
                           Each must have: wrist_vx_analysis, accel,
                           force_actual_interp, timestamps_analysis,
                           flywheel_params, drag_factor_pm5, stroke_length, model_dt.
        initial_A:         Starting guess for A.
        initial_B:         Starting guess for B.
        vx_threshold:      Velocity threshold for shape-correlation metric.

    Returns:
        dict with 'A', 'B', and quality metrics.
    """
    from scipy.optimize import least_squares

    # Collect per-stroke data
    strokes = []
    total_frames = 0
    for sd in stroke_data_list:
        vx = sd['wrist_vx_analysis']
        acc = sd['accel']
        ft = sd['force_actual_interp']
        fw = sd['flywheel_params']
        df = fw.drag_factor_pm5 if fw else 0
        sl = fw.stroke_length if fw else 0.0
        dt = sd.get('model_dt', None)
        if dt is None:
            ts = sd['timestamps_analysis']
            dt = float(np.median(np.diff(ts))) if len(ts) >= 2 else 1.0/60.0
        strokes.append((vx, acc, ft, fw, df, sl, dt))
        total_frames += len(vx)

    print(f"\n{'='*60}")
    print(f"GLOBAL OPTIMISATION: {len(strokes)} strokes, {total_frames} total frames")
    print(f"{'='*60}")

    def global_residuals(params):
        A_c, B_c = params
        all_res = []
        for (vx, acc, ft, fw, df, sl, dt) in strokes:
            fm, _, _, _ = compute_force_model(
                vx, acc, A_c, B_c,
                drag_factor_pm5=df, stroke_length=sl,
                flywheel_params=fw, dt=dt)
            all_res.append(fm - ft)
        return np.concatenate(all_res)

    lb = [0.0, 0.0]
    ub = [np.inf, np.inf]

    A_starts = [initial_A * f for f in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]]
    B_starts = [initial_B * f for f in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]]

    print(f"  Optimizing A, B globally (least-squares, multi-start)...")
    print(f"    Initial guess: A={initial_A:.6f}, B={initial_B:.6f}")
    print(f"    Grid: {len(A_starts)} × {len(B_starts)} = {len(A_starts)*len(B_starts)} starts")

    best_cost = np.inf
    best_result = None
    n_starts = 0

    for A0 in A_starts:
        for B0 in B_starts:
            x0 = [max(1e-12, A0), max(1e-12, B0)]
            try:
                result = least_squares(
                    global_residuals, x0,
                    bounds=(lb, ub), method='trf',
                    ftol=1e-10, xtol=1e-10, gtol=1e-10,
                    max_nfev=10000,
                )
                n_starts += 1
                if result.cost < best_cost:
                    best_cost = result.cost
                    best_result = result
            except Exception:
                pass

    if best_result is None:
        x0 = [max(1e-12, initial_A), max(1e-12, initial_B)]
        best_result = least_squares(global_residuals, x0, bounds=(lb, ub),
                                    method='trf', max_nfev=10000)

    A_opt, B_opt = best_result.x

    # Quality metrics over all strokes
    all_residuals = global_residuals([A_opt, B_opt])
    rmse = np.sqrt(np.mean(all_residuals**2))

    # Per-stroke metrics
    per_stroke_rmse = []
    offset = 0
    for (vx, acc, ft, fw, df, sl, dt) in strokes:
        n = len(vx)
        res = all_residuals[offset:offset+n]
        per_stroke_rmse.append(np.sqrt(np.mean(res**2)))
        offset += n

    # Physics comparison from first stroke's flywheel params
    fw0 = strokes[0][3]
    r_eff = fw0.effective_radius if fw0 else 0.042
    A_phys = fw0.inertia / r_eff if fw0 else 0.0
    df0 = strokes[0][4]
    c_drag = (df0 * 1e-6) if df0 > 0 else 0.0
    B_phys = c_drag / r_eff if r_eff > 0 else 0.0

    print(f"  Global optimisation complete ({n_starts} converged starts):")
    print(f"    A (inertia)    = {A_opt:.6f}" +
          (f"  (physics: {A_phys:.6f}, ratio: {A_opt/A_phys:.3f}×)" if A_phys > 0 else ""))
    print(f"    B (drag)       = {B_opt:.6f}" +
          (f"  (physics: {B_phys:.6f}, ratio: {B_opt/B_phys:.3f}×)" if B_phys > 0 else ""))
    print(f"    Global RMSE    = {rmse:.2f} N  ({len(strokes)} strokes, {total_frames} frames)")
    print(f"    Per-stroke RMSE: min={min(per_stroke_rmse):.2f}, "
          f"max={max(per_stroke_rmse):.2f}, "
          f"mean={np.mean(per_stroke_rmse):.2f} N")
    if fw0:
        print(f"    Flywheel I={fw0.inertia}, r_eff={r_eff:.4f} m")

    return {
        'A': A_opt,
        'B': B_opt,
        'rmse': rmse,
        'per_stroke_rmse': per_stroke_rmse,
        'n_strokes': len(strokes),
        'total_frames': total_frames,
    }


def optimize_global_constrained(stroke_data_list: list,
                                flywheel_params: FlywheelParameters,
                                drag_factor_pm5: int = 0,
                                vx_threshold: float = 0.0,
                                ) -> Dict:
    """Physics-constrained global optimisation: fit velocity scale s across all strokes.

    A_eff = s · A_phys,  B_eff = s² · B_phys.

    Returns dict with 'A', 'B', 's', and metrics.
    """
    from scipy.optimize import least_squares

    r_eff = flywheel_params.effective_radius
    A_phys = flywheel_params.inertia / r_eff if r_eff > 0 else 1.0
    c_drag = (drag_factor_pm5 * 1e-6) if drag_factor_pm5 > 0 else 0.0
    B_phys = c_drag / r_eff if r_eff > 0 else 0.0

    strokes = []
    total_frames = 0
    for sd in stroke_data_list:
        vx = sd['wrist_vx_analysis']
        acc = sd['accel']
        ft = sd['force_actual_interp']
        fw = sd['flywheel_params']
        df = fw.drag_factor_pm5 if fw else drag_factor_pm5
        sl = fw.stroke_length if fw else 0.0
        dt = sd.get('model_dt', None)
        if dt is None:
            ts = sd['timestamps_analysis']
            dt = float(np.median(np.diff(ts))) if len(ts) >= 2 else 1.0/60.0
        strokes.append((vx, acc, ft, fw, df, sl, dt))
        total_frames += len(vx)

    print(f"\n{'='*60}")
    print(f"GLOBAL CONSTRAINED OPTIMISATION: {len(strokes)} strokes, {total_frames} frames")
    print(f"{'='*60}")
    print(f"  A_phys = I/r_eff = {A_phys:.6f}")
    print(f"  B_phys = c/r_eff = {B_phys:.6f}")

    def global_residuals(params):
        s = params[0]
        A_eff = s * A_phys
        B_eff = s * s * B_phys
        all_res = []
        for (vx, acc, ft, fw, df, sl, dt) in strokes:
            fm, _, _, _ = compute_force_model(
                vx, acc, A_eff, B_eff,
                drag_factor_pm5=df, stroke_length=sl,
                flywheel_params=fw, dt=dt)
            all_res.append(fm - ft)
        return np.concatenate(all_res)

    s_starts = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    print(f"  Grid: {len(s_starts)} starts")

    best_cost = np.inf
    best_result = None
    n_starts = 0

    for s0 in s_starts:
        try:
            result = least_squares(
                global_residuals, [s0],
                bounds=([1e-6], [100.0]), method='trf',
                ftol=1e-10, xtol=1e-10, gtol=1e-10,
                max_nfev=10000,
            )
            n_starts += 1
            if result.cost < best_cost:
                best_cost = result.cost
                best_result = result
        except Exception:
            pass

    if best_result is None:
        best_result = least_squares(global_residuals, [1.0],
                                    bounds=([1e-6], [100.0]), method='trf', max_nfev=10000)

    s_opt = best_result.x[0]
    A_opt = s_opt * A_phys
    B_opt = s_opt * s_opt * B_phys

    all_residuals = global_residuals([s_opt])
    rmse = np.sqrt(np.mean(all_residuals**2))

    per_stroke_rmse = []
    offset = 0
    for (vx, acc, ft, fw, df, sl, dt) in strokes:
        n = len(vx)
        res = all_residuals[offset:offset+n]
        per_stroke_rmse.append(np.sqrt(np.mean(res**2)))
        offset += n

    print(f"  Global constrained optimisation complete ({n_starts} starts):")
    print(f"    velocity scale = {s_opt:.6f}")
    print(f"    A_eff = s·A_phys = {A_opt:.6f}  (A_phys={A_phys:.6f})")
    print(f"    B_eff = s²·B_phys = {B_opt:.6f}  (B_phys={B_phys:.6f})")
    print(f"    A/B ratio        = {A_opt/B_opt:.1f}  (physics: {A_phys/B_phys:.1f})" if B_opt > 0
          else f"    A/B ratio        = inf  (B=0)")
    print(f"    Global RMSE      = {rmse:.2f} N  ({len(strokes)} strokes, {total_frames} frames)")
    print(f"    Per-stroke RMSE: min={min(per_stroke_rmse):.2f}, "
          f"max={max(per_stroke_rmse):.2f}, "
          f"mean={np.mean(per_stroke_rmse):.2f} N")

    return {
        'A': A_opt,
        'B': B_opt,
        's': s_opt,
        'rmse': rmse,
        'per_stroke_rmse': per_stroke_rmse,
        'n_strokes': len(strokes),
        'total_frames': total_frames,
    }


def plot_single_stroke_analysis(time: np.ndarray,
                                wrist_vx: np.ndarray,
                                wrist_speed: np.ndarray,
                                force_actual: np.ndarray,
                                force_model: np.ndarray,
                                filename: str,
                                opt_params: Dict = None,
                                drive_start_frame: int = 0,
                                keypoint_name: str = 'right_shoulder',
                                clutch_state: np.ndarray = None,  # unused, kept for compat
                                keypoint_velocities: Dict = None,
                phases: np.ndarray = None,
                existing_fig=None,
                stroke_index: int = 0,
                total_strokes: int = 1,
                f_stretcher: np.ndarray = None,
                f_body_inertia: np.ndarray = None):
    """
    Plot analysis for a single stroke comparing actual and model force with velocity overlay,
    plus multi-joint velocity and acceleration.

    Returns the matplotlib figure (does NOT call plt.show).
    """
    has_kp_data = keypoint_velocities is not None and len(keypoint_velocities) > 0
    if existing_fig is not None:
        fig = existing_fig
        fig.clear()
    else:
        if has_kp_data:
            fig = plt.figure(figsize=(16, 14))
        else:
            fig = plt.figure(figsize=(14, 10))
    if has_kp_data:
        ax1 = fig.add_subplot(2, 2, 1)
        ax3 = fig.add_subplot(2, 2, 3)
        ax_vel = fig.add_subplot(2, 2, 2)
        ax_acc = fig.add_subplot(2, 2, 4)
    else:
        ax1 = fig.add_subplot(2, 1, 1)
        ax3 = fig.add_subplot(2, 1, 2)
    # Use actual frame numbers from the stroke, not starting at 0
    frame_indices = np.arange(drive_start_frame, drive_start_frame + len(time))
    
    # Top-left: Force and velocity comparison
    ax1.plot(frame_indices, force_actual, 'r-', linewidth=2.5, label='F_handle (PM5)', alpha=0.9)
    ax1.set_ylabel('Force (N)', fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3)
    ax1.fill_between(frame_indices, 0, force_actual, alpha=0.15, color='red')
    
    # Create secondary y-axis for velocity
    ax2 = ax1.twinx()
    kp_label = keypoint_name.replace('_', ' ').title()
    ax2.plot(frame_indices, wrist_vx, 'b-', linewidth=2, label=f'{kp_label} Velocity', alpha=0.7)
    ax2.set_ylabel(f'{kp_label} Velocity (m/s)  [\u2190 +, \u2192 \u2212]', fontsize=12, color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.axhline(y=0, color='b', linestyle='--', alpha=0.3, linewidth=1)
    
    clutch_patches = []
    
    # Title with navigation info
    nav_str = f" [{stroke_index+1}/{total_strokes}]" if total_strokes > 1 else ""
    title_str = f'Force and Velocity - {filename}{nav_str}'
    ax1.set_title(title_str, fontsize=11)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + clutch_patches,
               labels1 + labels2 + [p.get_label() for p in clutch_patches],
               loc='upper left', fontsize=8)
    
    # Bottom-left: Colloud (2006) force balance — F_stretcher + F_handle = m·a_CoM
    # F_handle on the rower is −F_PM5 (chain pulls toward flywheel)
    has_colloud = f_stretcher is not None and f_body_inertia is not None
    if has_colloud:
        f_handle_signed = -force_actual  # signed: negative during drive (toward flywheel)
        ax3.plot(frame_indices, f_stretcher, 'darkorange', linewidth=2,
                 label='F_stretcher (feet)', alpha=0.9)
        ax3.plot(frame_indices, f_handle_signed, 'r-', linewidth=2,
                 label='F_handle (chain, −PM5)', alpha=0.9)
        ax3.plot(frame_indices, f_body_inertia, 'b-', linewidth=2.5,
                 label='m·a_CoM', alpha=0.9)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
        ax3.set_title('Colloud (2006):  F_stretcher + F_handle = m·a_CoM', fontsize=11)

        # Ratio F_stretcher / F_handle (|PM5|) on secondary y-axis
        ax3_ratio = ax3.twinx()
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(force_actual > 1.0, f_stretcher / force_actual, np.nan)
        ax3_ratio.plot(frame_indices, ratio, color='green', linewidth=1.5,
                       linestyle=':', label='F_stretcher / F_handle', alpha=0.8)
        ax3_ratio.set_ylabel('Ratio', fontsize=10, color='green')
        ax3_ratio.tick_params(axis='y', labelcolor='green')
        ax3_ratio.legend(loc='upper right', fontsize=8)
    else:
        # Fallback: residuals
        residuals = force_actual - force_model
        ax3.plot(frame_indices, residuals, 'purple', linewidth=2, label='Residuals (Actual - Model)', alpha=0.8)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
        ax3.fill_between(frame_indices, 0, residuals, alpha=0.3, color='purple')
        rmse = np.sqrt(np.mean(residuals**2))
        mean_residual = np.mean(residuals)
        residual_text = f'RMSE: {rmse:.2f} N | Mean: {mean_residual:.2f} N'
        ax3.text(0.98, 0.95, residual_text, transform=ax3.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax3.set_ylabel('Force (N)', fontsize=12)
    ax3.set_xlabel('Frame Number', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left', fontsize=8)
    
    # --- Multi-joint velocity and acceleration plots (right column) ---
    if has_kp_data:
        # Joint display order and colours (matching analyze_strokes.py)
        joint_styles = [
            ('right_shoulder', 'left_shoulder', 'Shoulder', 'purple', '-'),
            ('right_hip',      'left_hip',      'Hip',      'blue',   '-'),
            ('right_knee',     'left_knee',     'Knee',     'green',  '-'),
            ('right_ankle',    'left_ankle',    'Ankle',    'orange', '-'),
            ('right_wrist',    None,            'Wrist',    'red',    '-'),
            ('handle',         None,            'Handle',   'magenta','--'),
        ]
        
        n_full = max(len(v['vx']) for v in keypoint_velocities.values())

        # Align x-axis so frame 0 = wrist velocity zero-crossing (catch/drive onset)
        # Wrist vx is stored with rightward=positive; negated for display so positive=leftward.
        # The catch is where -wrist_vx crosses zero upward (wrist_vx goes negative).
        wrist_kp = keypoint_velocities.get('right_wrist')
        zero_crossing_frame = 0
        if wrist_kp is not None and len(wrist_kp['vx']) == n_full:
            wrist_vx_disp = -wrist_kp['vx']  # positive = leftward = drive
            for i in range(1, len(wrist_vx_disp)):
                if wrist_vx_disp[i - 1] <= 0 < wrist_vx_disp[i]:
                    zero_crossing_frame = i
                    break
        full_frames = np.arange(n_full) - zero_crossing_frame

        # Phase background shading
        if phases is not None and len(phases) == n_full:
            phase_colors_map = {
                0: ('lightgray',),  1: ('lightyellow',),
                2: ('lightgreen',), 3: ('lightsalmon',),
            }
            for ax_bg in [ax_vel, ax_acc]:
                cur = None
                start = 0
                for fi, ph in enumerate(list(phases) + [-1]):
                    if ph != cur:
                        if cur is not None:
                            c = phase_colors_map.get(cur, ('white',))[0]
                            ax_bg.axvspan(start - zero_crossing_frame,
                                          fi - 1 - zero_crossing_frame,
                                          alpha=0.15, color=c)
                        cur = ph
                        start = fi
        
        # Top-right: Horizontal velocity (vx) for all joints
        for prim, alt, label, color, ls in joint_styles:
            kp = keypoint_velocities.get(prim) or (keypoint_velocities.get(alt) if alt else None)
            if kp is None or len(kp['vx']) != n_full:
                continue
            ax_vel.plot(full_frames, -kp['vx'], color=color, linewidth=2, label=label, linestyle=ls)  # negate so positive=leftward

        # Centre of mass velocity (Dumas et al. anthropometric model)
        _COM_SEGMENTS = [
            ('right_shoulder', None,           0.067, 0.0),
            ('right_shoulder', 'right_hip',    0.333, 0.42),
            ('right_hip',      None,           0.142, 0.0),
            ('right_shoulder', 'right_wrist',  0.094, 0.50),
            ('right_hip',      'right_knee',   0.246, 0.429),
            ('right_knee',     'right_ankle',  0.096, 0.41),
            ('right_ankle',    None,           0.028, 0.0),
        ]
        _com_total = sum(s[2] for s in _COM_SEGMENTS)
        _com_required = {'right_shoulder', 'right_hip', 'right_knee', 'right_ankle', 'right_wrist'}
        if _com_required.issubset(set(keypoint_velocities.keys())):
            _vx = {name: keypoint_velocities[name]['vx'] for name in _com_required}
            if all(len(_vx[n]) == n_full for n in _com_required):
                com_vx_plot = np.zeros(n_full)
                for (prox, dist, mf, cf) in _COM_SEGMENTS:
                    w = mf / _com_total
                    if dist is None:
                        com_vx_plot += w * _vx[prox]
                    else:
                        com_vx_plot += w * ((1.0 - cf) * _vx[prox] + cf * _vx[dist])
                ax_vel.plot(full_frames, -com_vx_plot, color='black', linewidth=2.5,
                            label='CoM', linestyle='-', alpha=0.9)
        
        ax_vel.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
        ax_vel.axvline(x=0, color='k', linestyle=':', alpha=0.4, linewidth=1, label='Zero crossing')
        ax_vel.axvspan(drive_start_frame - zero_crossing_frame,
                       drive_start_frame + len(time) - 1 - zero_crossing_frame,
                       alpha=0.08, color='blue', zorder=0)
        ax_vel.set_ylabel('Horizontal Velocity (m/s)  [\u2190 +, \u2192 \u2212]', fontsize=10)
        ax_vel.set_xlabel('Frame (0 = wrist zero-crossing)', fontsize=10)
        ax_vel.set_title('Multi-Joint Velocity (vx)', fontsize=12)
        ax_vel.legend(loc='best', fontsize=8)
        ax_vel.grid(True, alpha=0.3)
        
        # Bottom-right: Acceleration for all joints (horizontal vx, not speed magnitude)
        for prim, alt, label, color, ls in joint_styles:
            kp = keypoint_velocities.get(prim) or (keypoint_velocities.get(alt) if alt else None)
            if kp is None or len(kp['vx']) != n_full:
                continue
            acc = np.gradient(-kp['vx'])  # d/dt of horizontal velocity (positive=leftward)
            win = min(11, len(acc))
            if win % 2 == 0:
                win -= 1
            if win >= 5:
                acc = savgol_filter(acc, win, 3)
            ax_acc.plot(full_frames, acc, color=color, linewidth=2, label=label, linestyle=ls)

        # CoM acceleration
        if _com_required.issubset(set(keypoint_velocities.keys())):
            _vx2 = {name: keypoint_velocities[name]['vx'] for name in _com_required}
            if all(len(_vx2[n]) == n_full for n in _com_required):
                com_vx_acc = np.zeros(n_full)
                for (prox, dist, mf, cf) in _COM_SEGMENTS:
                    w = mf / _com_total
                    if dist is None:
                        com_vx_acc += w * _vx2[prox]
                    else:
                        com_vx_acc += w * ((1.0 - cf) * _vx2[prox] + cf * _vx2[dist])
                com_acc_plot = np.gradient(-com_vx_acc)
                win = min(11, len(com_acc_plot))
                if win % 2 == 0:
                    win -= 1
                if win >= 5:
                    com_acc_plot = savgol_filter(com_acc_plot, win, 3)
                ax_acc.plot(full_frames, com_acc_plot, color='black', linewidth=2.5,
                            label='CoM', linestyle='-', alpha=0.9)
        
        ax_acc.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
        ax_acc.axvline(x=0, color='k', linestyle=':', alpha=0.4, linewidth=1)
        ax_acc.axvspan(drive_start_frame - zero_crossing_frame,
                       drive_start_frame + len(time) - 1 - zero_crossing_frame,
                       alpha=0.08, color='blue', zorder=0)
        ax_acc.set_ylabel('Acceleration (m/s\u00b2)', fontsize=10)
        ax_acc.set_xlabel('Frame (0 = wrist zero-crossing)', fontsize=10)
        ax_acc.set_title('Multi-Joint Acceleration (horizontal)', fontsize=12)
        ax_acc.legend(loc='best', fontsize=8)
        ax_acc.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def analyze_single_stroke(filepath: str, args, global_A: float = None, global_B: float = None) -> Dict:
    """Analyze a single stroke file and return all data needed for plotting.

    If global_A and global_B are provided, use those coefficients directly
    (global optimisation mode). Otherwise optimise per-stroke or use
    initial values depending on args.optimize / args.constrain_ratio.

    Returns a dict with plot-ready arrays, or None on failure.
    """
    try:
        # Determine pixels_per_meter for this stroke
        ppm = args.pixels_per_meter if args.pixels_per_meter > 0 else 100.0  # initial guess
        stroke_data = extract_wrist_data_from_file(
            filepath,
            apply_perspective=not args.no_perspective_correction,
            handle_offset_m=args.handle_offset,
            pixels_per_meter=ppm,
        )

        timestamps = stroke_data['timestamps']

        # Select keypoint velocity
        kp_name = args.keypoint
        kp_vels = stroke_data.get('keypoint_velocities', {})
        if kp_name in kp_vels:
            wrist_vx = kp_vels[kp_name]['vx']
            wrist_speed = kp_vels[kp_name]['speed']
        else:
            available = list(kp_vels.keys())
            print(f"  WARNING: keypoint '{kp_name}' not found, available: {available}")
            print(f"  Falling back to right_wrist")
            wrist_vx = stroke_data['wrist_vx']
            wrist_speed = stroke_data['wrist_speed']
            kp_name = 'right_wrist'

        # Convention: positive velocity = moving left (drive direction)
        wrist_vx = -wrist_vx

        # --- Auto-calibrate pixels_per_meter from PM5 stroke length ---
        pm5_stroke_m = args.override_stroke_length or stroke_data.get('stroke_length', 0.0)
        if args.pixels_per_meter <= 0 and pm5_stroke_m > 0:
            # Use pixel positions to measure handle/wrist displacement during drive
            kp_pos = stroke_data.get('keypoint_positions', {})
            cal_kp = kp_pos.get(kp_name) or kp_pos.get('right_wrist')
            if cal_kp is not None:
                x_px = cal_kp['x']
                # Drive = where wrist moves leftward (positive after negation)
                drive_mask_cal = wrist_vx > args.vx_threshold
                drive_idx_cal = np.where(drive_mask_cal)[0]
                if len(drive_idx_cal) > 1:
                    x_drive = x_px[drive_idx_cal]
                    pixel_distance = abs(float(x_drive[-1] - x_drive[0]))
                    if pixel_distance > 1.0:  # sanity check
                        ppm = pixel_distance / pm5_stroke_m
                        print(f"  Pixel calibration: {kp_name} moved {pixel_distance:.1f} px "
                              f"over PM5 stroke={pm5_stroke_m:.3f} m")
                        print(f"    → pixels_per_meter = {ppm:.1f}")
                        # Re-extract with calibrated ppm so ALL velocities are correct
                        stroke_data = extract_wrist_data_from_file(
                            filepath,
                            apply_perspective=not args.no_perspective_correction,
                            handle_offset_m=args.handle_offset,
                            pixels_per_meter=ppm,
                        )
                        timestamps = stroke_data['timestamps']
                        kp_vels = stroke_data.get('keypoint_velocities', {})
                        if kp_name in kp_vels:
                            wrist_vx = -kp_vels[kp_name]['vx']
                            wrist_speed = kp_vels[kp_name]['speed']
                        else:
                            wrist_vx = -stroke_data['wrist_vx']
                            wrist_speed = stroke_data['wrist_speed']

        force = stroke_data['force']
        phases = stroke_data['phases']
        fps = stroke_data['fps']

        # PM5 ergometer metadata
        drag_factor_pm5 = args.override_drag_factor or stroke_data.get('drag_factor', 0)
        stroke_length = args.override_stroke_length or stroke_data.get('stroke_length', 0.0)

        fw_params = FlywheelParameters(
            inertia=args.flywheel_inertia,
            sprocket_radius=args.sprocket_radius,
            gear_ratio=args.gear_ratio,
            drag_factor_pm5=drag_factor_pm5,
            stroke_length=stroke_length,
        )

        A_init = args.inertia_coefficient
        B_init = args.force_coefficient
        if args.use_physics:
            A_phys, B_phys = fw_params.physics_coefficients()
            # Only override with physics values when user hasn't explicitly set them
            a_explicit = any(s.startswith('--inertia-coeff') for s in sys.argv)
            b_explicit = any(s.startswith('--force-coeff') for s in sys.argv)
            if A_phys > 0 and not a_explicit:
                A_init = A_phys
            if B_phys > 0 and not b_explicit:
                B_init = B_phys
            if a_explicit or b_explicit:
                print(f"  --use-physics: keeping explicit CLI values"
                      f"{' A=' + str(A_init) if a_explicit else ''}"
                      f"{' B=' + str(B_init) if b_explicit else ''}"
                      f" (physics: A={A_phys:.4f}, B={B_phys:.6f})")

        # Positive-velocity region detection (where wrist moves leftward above threshold)
        vx_thr = args.vx_threshold
        positive_vx_mask = wrist_vx > vx_thr  # positive = leftward after negation
        positive_vx_indices = np.where(positive_vx_mask)[0]
        if len(positive_vx_indices) == 0:
            print(f"WARNING: No wrist velocity > {vx_thr:.2f} m/s in {os.path.basename(filepath)}")
            return None

        drive_start_frame = positive_vx_indices[0]
        drive_end_frame = positive_vx_indices[-1]

        original_drive_start_time = timestamps[drive_start_frame]
        original_drive_end_time = timestamps[drive_end_frame]

        force_start_time = original_drive_start_time + args.force_offset
        force_end_time = original_drive_end_time + args.force_offset

        # Time scaling
        if args.keypoint_time_scale != 1.0:
            start_time = timestamps[0]
            timestamps = start_time + (timestamps - start_time) * args.keypoint_time_scale
            fps = fps / args.keypoint_time_scale
            wrist_vx = wrist_vx / args.keypoint_time_scale
            wrist_speed = wrist_speed / args.keypoint_time_scale

        # --- Smooth ALL keypoint velocities (full-length) before any slicing ---
        # This ensures handle velocity, multi-joint plots, and CoM acceleration
        # all use consistently smoothed velocity, and acceleration is derived
        # from the smoothed signal (matching how handle accel works).
        if not args.no_smooth:
            vel_win_full = args.smooth_window
            if vel_win_full % 2 == 0:
                vel_win_full += 1
            kp_vels = stroke_data.get('keypoint_velocities', {})
            for kp_name_s, kp_data in kp_vels.items():
                for key in ('vx', 'speed'):
                    arr = kp_data[key]
                    w = min(vel_win_full, len(arr))
                    if w % 2 == 0:
                        w -= 1
                    if w >= 5:
                        kp_data[key] = savgol_filter(arr, window_length=w, polyorder=3)
                    else:
                        kp_data[key] = moving_average(arr, window_size=args.smooth_window)
            # Re-read selected keypoint from smoothed data
            if kp_name in kp_vels:
                wrist_vx = -kp_vels[kp_name]['vx']
                wrist_speed = kp_vels[kp_name]['speed']
            elif kp_name == 'right_wrist' or kp_name not in kp_vels:
                wrist_vx = -kp_vels.get('right_wrist', {}).get('vx', wrist_vx)
                wrist_speed = kp_vels.get('right_wrist', {}).get('speed', wrist_speed)

        # Drive-phase slices (velocities already smoothed)
        wrist_vx_analysis = wrist_vx[drive_start_frame:drive_end_frame + 1]
        wrist_speed_analysis = wrist_speed[drive_start_frame:drive_end_frame + 1]
        timestamps_analysis = timestamps[drive_start_frame:drive_end_frame + 1]

        # Force interpolation
        if len(force) > 0:
            force_timestamps = np.linspace(force_start_time, force_end_time, len(force))
            force_actual_interp = np.interp(timestamps_analysis, force_timestamps, force, left=0.0, right=0.0)
        else:
            force_actual_interp = np.zeros_like(wrist_vx_analysis)

        # Smoothing (force only — velocities already smoothed above on full array)
        if not args.no_smooth:
            force_actual_interp = moving_average(force_actual_interp, window_size=args.smooth_force_window)

        # Acceleration
        accel = np.gradient(wrist_vx_analysis, timestamps_analysis)
        if not args.no_smooth:
            win = args.smooth_accel_window
            if win % 2 == 0:
                win += 1
            win = min(win, len(accel))
            if win % 2 == 0:
                win -= 1
            if args.smooth_accel_method == 'savgol' and win >= 5:
                accel = savgol_filter(accel, window_length=win, polyorder=3)
            else:
                accel = moving_average(accel, window_size=args.smooth_accel_window)

        # --- Body centre-of-mass acceleration (Dumas et al. anthropometric model) ---
        rower_mass = getattr(args, 'rower_mass', 0.0)
        com_accel_analysis = None
        if rower_mass > 0:
            kp_vels = stroke_data.get('keypoint_velocities', {})
            if kp_vels:
                com_accel_full = compute_com_acceleration(
                    kp_vels, timestamps,
                    smooth_window=args.smooth_accel_window,
                )
                if com_accel_full is not None:
                    # Slice to the same drive-phase window as wrist_vx_analysis
                    com_accel_analysis = com_accel_full[drive_start_frame:drive_end_frame + 1]
                    print(f"  Body inertia: mass={rower_mass:.1f} kg, "
                          f"a_COM range=[{np.min(com_accel_analysis):.2f}, "
                          f"{np.max(com_accel_analysis):.2f}] m/s², "
                          f"peak |F_body|={rower_mass * np.max(np.abs(com_accel_analysis)):.1f} N")
                else:
                    print(f"  WARNING: Could not compute CoM acceleration (missing keypoints)")
            else:
                print(f"  WARNING: No keypoint velocities available for CoM calculation")

        # Flywheel model
        if len(timestamps_analysis) >= 2:
            model_dt = float(np.median(np.diff(timestamps_analysis)))
        else:
            model_dt = 1.0 / fps

        if global_A is not None and global_B is not None:
            # ---- Global optimisation: use pre-fitted A, B ----
            opt_params = {
                'inertia_coefficient': global_A,
                'force_coefficient': global_B,
                'delay': 0.0,
            }
            force_model, force_model_inertial, force_model_drag, clutch_state = compute_force_model(
                wrist_vx_analysis, accel, global_A, global_B,
                drag_factor_pm5=drag_factor_pm5, stroke_length=stroke_length,
                flywheel_params=fw_params, dt=model_dt,
            )
        elif args.constrain_ratio:
            (opt_params, force_model, force_model_inertial, force_model_drag,
             clutch_state) = optimize_physics_constrained(
                wrist_vx_analysis, accel, force_actual_interp,
                timestamps_analysis,
                flywheel_params=fw_params,
                drag_factor_pm5=drag_factor_pm5, stroke_length=stroke_length,
                dt=model_dt,
                vx_threshold=args.vx_threshold,
            )
        elif args.optimize:
            (opt_params, force_model, force_model_inertial, force_model_drag,
             clutch_state) = optimize_model_parameters(
                wrist_vx_analysis, accel, force_actual_interp,
                timestamps_analysis,
                initial_A=A_init, initial_B=B_init,
                drag_factor_pm5=drag_factor_pm5, stroke_length=stroke_length,
                flywheel_params=fw_params, dt=model_dt,
                vx_threshold=args.vx_threshold,
            )
        else:
            opt_params = {
                'force_coefficient': B_init,
                'inertia_coefficient': A_init,
                'delay': 0.0,
            }
            force_model, force_model_inertial, force_model_drag, clutch_state = compute_force_model(
                wrist_vx_analysis, accel, A_init, B_init,
                drag_factor_pm5=drag_factor_pm5, stroke_length=stroke_length,
                flywheel_params=fw_params, dt=model_dt,
            )

        # Colloud (2006): Newton's 2nd law on the rower's body
        # Positive direction = away from flywheel (drive direction)
        # F_stretcher (feet push away, +) + F_handle_on_rower (chain pulls toward flywheel, −) = m·a_CoM
        # F_handle_on_rower = −F_PM5 (PM5 measures magnitude of chain tension)
        # => F_stretcher = m·a_CoM + F_PM5
        f_stretcher = None
        f_body_inertia = None
        if rower_mass > 0 and com_accel_analysis is not None:
            f_body_inertia = rower_mass * com_accel_analysis
            f_stretcher = f_body_inertia + force_actual_interp
            print(f"  Colloud equation: F_stretcher + F_handle = m·a_CoM")
            print(f"    F_stretcher range = [{np.min(f_stretcher):.1f}, {np.max(f_stretcher):.1f}] N")
            print(f"    m·a_CoM range     = [{np.min(f_body_inertia):.1f}, {np.max(f_body_inertia):.1f}] N")
            print(f"    F_handle range    = [{np.min(force_actual_interp):.1f}, {np.max(force_actual_interp):.1f}] N")

        # Metrics
        rmse = np.sqrt(np.mean((force_actual_interp - force_model) ** 2))
        rmse_norm = rmse / np.max(force_actual_interp) if np.max(force_actual_interp) > 0 else 0

        drive_vx_mask = wrist_vx_analysis > args.vx_threshold  # positive = leftward = drive
        if np.sum(drive_vx_mask) < 3:
            print(f"WARNING: Insufficient drive-phase frames in {os.path.basename(filepath)}")
            return None

        wrist_vx_drive = wrist_vx_analysis[drive_vx_mask]
        force_actual_drive = force_actual_interp[drive_vx_mask]
        force_model_drive = force_model[drive_vx_mask]

        if len(wrist_vx_drive) >= 3 and len(force_actual_drive) >= 3:
            correlation = np.corrcoef(wrist_vx_drive, force_actual_drive)[0, 1]
        else:
            correlation = 0.0

        return {
            'filepath': filepath,
            'filename': stroke_data['filename'],
            'timestamps_analysis': timestamps_analysis,
            'wrist_vx_analysis': wrist_vx_analysis,
            'wrist_speed_analysis': wrist_speed_analysis,
            'force_actual_interp': force_actual_interp,
            'force_model': force_model,
            'force_model_inertial': force_model_inertial,
            'force_model_drag': force_model_drag,
            'accel': accel,
            'com_accel': com_accel_analysis,
            'f_stretcher': f_stretcher,
            'f_body_inertia': f_body_inertia,
            'flywheel_params': fw_params,
            'model_dt': model_dt,
            'opt_params': opt_params,
            'drive_start_frame': drive_start_frame,
            'keypoint_name': kp_name,
            'clutch_state': clutch_state,
            'keypoint_velocities': stroke_data.get('keypoint_velocities'),
            'phases': stroke_data.get('phases'),
            'correlation': correlation,
            'rmse': rmse,
            'rmse_norm': rmse_norm,
        }

    except Exception as e:
        print(f"ERROR analyzing {filepath}: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_residual_diagnostics(result: Dict, fw_params=None):
    """Print detailed frame-by-frame residual breakdown to help identify missing force terms.

    Shows per-frame columns:
        t, vel, accel, omega, F_inertial, F_drag, F_model, F_actual, residual

    Then prints correlation of residual vs candidate physics terms:
        velocity, acceleration, jerk, omega, omega², dω/dt, v·a, etc.
    """
    if result is None:
        return

    t = result['timestamps_analysis']
    vel = result['wrist_vx_analysis']
    accel = result['accel']
    f_actual = result['force_actual_interp']
    f_model = result['force_model']
    f_inertial = result['force_model_inertial']
    f_drag = result['force_model_drag']
    params = result['opt_params']

    residual = f_actual - f_model

    r_eff = fw_params.effective_radius if fw_params else 0.042
    omega = np.where(vel > 0, vel / r_eff, 0.0)
    d_omega_dt = np.where(vel > 0, accel / r_eff, 0.0)

    # Jerk = d(accel)/dt
    jerk = np.gradient(accel, t)

    # Header
    print(f"\n{'─'*120}")
    print(f"RESIDUAL DIAGNOSTICS: {result['filename']}")
    print(f"  A (inertia) = {params.get('inertia_coefficient', '?')}")
    print(f"  B (drag)    = {params.get('force_coefficient', '?')}")
    print(f"  r_eff       = {r_eff:.4f} m")
    print(f"{'─'*120}")

    # Per-frame table (drive phase only — where vel > 0)
    drive_mask = vel > 0
    n_drive = np.sum(drive_mask)
    idx_drive = np.where(drive_mask)[0]

    print(f"\n  Drive phase: {n_drive} frames")
    print(f"  {'frm':>4s}  {'t(s)':>6s}  {'vel':>8s}  {'accel':>8s}  {'omega':>8s}  {'dw/dt':>8s}"
          f"  {'F_iner':>8s}  {'F_drag':>8s}  {'F_mod':>8s}  {'F_act':>8s}  {'resid':>8s}")
    print(f"  {'─'*4}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}"
          f"  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")

    # Print every frame (or subsample if too many)
    step = max(1, n_drive // 40)  # cap at ~40 rows
    for k, i in enumerate(idx_drive):
        if k % step != 0 and k != n_drive - 1:
            continue
        print(f"  {i:4d}  {t[i]:6.3f}  {vel[i]:8.1f}  {accel[i]:8.1f}  {omega[i]:8.1f}  {d_omega_dt[i]:8.1f}"
              f"  {f_inertial[i]:8.2f}  {f_drag[i]:8.2f}  {f_model[i]:8.2f}  {f_actual[i]:8.2f}"
              f"  {residual[i]:+8.2f}")

    # Summary statistics of residuals during drive
    res_drive = residual[drive_mask]
    print(f"\n  Residual stats (drive):")
    print(f"    mean   = {np.mean(res_drive):+.2f} N")
    print(f"    std    = {np.std(res_drive):.2f} N")
    print(f"    max    = {np.max(res_drive):+.2f} N")
    print(f"    min    = {np.min(res_drive):+.2f} N")
    print(f"    RMSE   = {np.sqrt(np.mean(res_drive**2)):.2f} N")

    # Where does residual peak?
    peak_res_idx = idx_drive[np.argmax(np.abs(res_drive))]
    print(f"    peak |residual| at frame {peak_res_idx}, t={t[peak_res_idx]:.3f}s, "
          f"residual={residual[peak_res_idx]:+.2f} N")

    # Candidate terms for correlation with residual
    print(f"\n  Correlation of residual with candidate terms (drive phase):")
    print(f"  {'candidate':>24s}  {'corr':>8s}  {'slope':>12s}  {'interpretation'}")
    print(f"  {'─'*24}  {'─'*8}  {'─'*12}  {'─'*40}")

    vel_d   = vel[drive_mask]
    acc_d   = accel[drive_mask]
    jrk_d   = jerk[drive_mask]
    omg_d   = omega[drive_mask]
    domg_d  = d_omega_dt[drive_mask]
    f_act_d = f_actual[drive_mask]
    t_d     = t[drive_mask] - t[drive_mask][0]  # relative time within drive

    candidates = [
        ('velocity (v)',          vel_d,                 'residual ~ v → missing v term?'),
        ('acceleration (dv/dt)',  acc_d,                 'residual ~ a → need more inertia (A)?'),
        ('jerk (d²v/dt²)',       jrk_d,                 'residual ~ jerk → rate of accel term?'),
        ('omega (ω)',            omg_d,                 'residual ~ ω → linear drag term?'),
        ('omega² (ω²)',          omg_d**2,              'residual ~ ω² → need more drag (B)?'),
        ('dω/dt',                domg_d,                'residual ~ dω/dt → need more inertia?'),
        ('v × a (power-like)',   vel_d * acc_d,         'residual ~ v·a → energy coupling?'),
        ('v²',                   vel_d**2,              'residual ~ v² → quadratic vel term?'),
        ('1/v (catch surge)',    np.where(vel_d > 1, 1.0/vel_d, 0.0),
                                                        'residual ~ 1/v → catch/compression?'),
        ('time (linear trend)',  t_d,                   'residual ~ t → time-dependent drift?'),
        ('F_actual',             f_act_d,               'residual ~ F → model scale error?'),
        ('√|accel|·sign(a)',     np.sign(acc_d)*np.sqrt(np.abs(acc_d)),
                                                        'residual ~ √|a| → sub-linear inertia?'),
    ]

    for name, term, interp in candidates:
        if np.std(term) < 1e-12 or np.std(res_drive) < 1e-12:
            print(f"  {name:>24s}  {'n/a':>8s}  {'n/a':>12s}  {interp}")
            continue
        corr = np.corrcoef(res_drive, term)[0, 1]
        # Linear regression slope
        slope = np.polyfit(term, res_drive, 1)[0]
        marker = ' ***' if abs(corr) > 0.7 else (' **' if abs(corr) > 0.5 else (' *' if abs(corr) > 0.3 else ''))
        print(f"  {name:>24s}  {corr:+8.4f}  {slope:+12.4f}  {interp}{marker}")

    # Phase analysis: split drive into thirds
    n3 = max(1, n_drive // 3)
    early  = res_drive[:n3]
    mid    = res_drive[n3:2*n3]
    late   = res_drive[2*n3:]
    print(f"\n  Residual by drive phase:")
    print(f"    early  (frames 0-{n3-1}):     mean={np.mean(early):+.2f} N, std={np.std(early):.2f} N")
    print(f"    middle (frames {n3}-{2*n3-1}):   mean={np.mean(mid):+.2f} N, std={np.std(mid):.2f} N")
    print(f"    late   (frames {2*n3}-{len(res_drive)-1}):   mean={np.mean(late):+.2f} N, std={np.std(late):.2f} N")

    # Ratio analysis: what fraction of actual force is explained?
    f_mod_d = f_model[drive_mask]
    f_iner_d = f_inertial[drive_mask]
    f_drag_d = f_drag[drive_mask]
    total_actual = np.sum(np.abs(f_act_d))
    total_model = np.sum(np.abs(f_mod_d))
    total_inertial = np.sum(np.abs(f_iner_d))
    total_drag = np.sum(np.abs(f_drag_d))

    print(f"\n  Force component breakdown (sum of |F| over drive):")
    print(f"    |F_actual|   = {total_actual:.1f} N·frames")
    print(f"    |F_model|    = {total_model:.1f} N·frames  ({100*total_model/total_actual:.1f}% of actual)")
    print(f"    |F_inertial| = {total_inertial:.1f} N·frames  ({100*total_inertial/total_actual:.1f}% of actual)")
    print(f"    |F_drag|     = {total_drag:.1f} N·frames  ({100*total_drag/total_actual:.1f}% of actual)")
    print(f"    |residual|   = {np.sum(np.abs(res_drive)):.1f} N·frames  ({100*np.sum(np.abs(res_drive))/total_actual:.1f}% of actual)")

    print(f"{'─'*120}\n")


def show_stroke(result: Dict, existing_fig=None, stroke_index: int = 0, total_strokes: int = 1):
    """Plot a pre-analyzed stroke result. Returns the figure."""
    if result is None:
        return None
    return plot_single_stroke_analysis(
        result['timestamps_analysis'],
        result['wrist_vx_analysis'],
        result['wrist_speed_analysis'],
        result['force_actual_interp'],
        result['force_model'],
        result['filename'],
        result['opt_params'],
        drive_start_frame=result['drive_start_frame'],
        keypoint_name=result['keypoint_name'],
        clutch_state=result['clutch_state'],
        keypoint_velocities=result.get('keypoint_velocities'),
        phases=result.get('phases'),
        existing_fig=existing_fig,
        stroke_index=stroke_index,
        total_strokes=total_strokes,
        f_stretcher=result.get('f_stretcher'),
        f_body_inertia=result.get('f_body_inertia'),
    )


def on_key(event, results: list, current_idx: list, fig_state: dict):
    """Handle keyboard events for stroke navigation.

    Args:
        event: matplotlib key event
        results: list of pre-analyzed stroke dicts (from analyze_single_stroke)
        current_idx: mutable list [index] for current stroke
        fig_state: dict with 'fig' key for figure reuse
    """
    if event.key == 'right' and current_idx[0] < len(results) - 1:
        current_idx[0] += 1
        fig = show_stroke(results[current_idx[0]],
                          existing_fig=fig_state.get('fig'),
                          stroke_index=current_idx[0],
                          total_strokes=len(results))
        if fig:
            fig_state['fig'] = fig
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
    elif event.key == 'left' and current_idx[0] > 0:
        current_idx[0] -= 1
        fig = show_stroke(results[current_idx[0]],
                          existing_fig=fig_state.get('fig'),
                          stroke_index=current_idx[0],
                          total_strokes=len(results))
        if fig:
            fig_state['fig'] = fig
            fig.canvas.draw_idle()
            fig.canvas.flush_events()


def main():
    """Main execution function"""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Correlate keypoint velocity with stroke force from recorded data'
    )
    parser.add_argument('--data-folder', type=str, default=DATA_FOLDER,
                       help=f'Folder containing stroke JSON files (default: {DATA_FOLDER})')
    parser.add_argument('--file', type=str, default=None,
                       help='Specific stroke file to analyze (overrides data-folder)')
    parser.add_argument('--synthetic', action='store_true',
                       help='Use synthetic data instead of real recordings')
    parser.add_argument('--debug-csv', action='store_true',
                       help='Output full data in CSV format to stdout for debugging')
    parser.add_argument('--no-smooth', action='store_true',
                       help='Disable smoothing filters on velocity and force data')
    parser.add_argument('--smooth-window', type=int, default=21,
                       help='Smoothing window size for velocity (default: 21, Savitzky-Golay)')
    parser.add_argument('--smooth-force-window', type=int, default=2,
                       help='Smoothing window size for force (default: 2)')
    parser.add_argument('--smooth-accel-window', type=int, default=21,
                       help='Smoothing window size for computed acceleration dv/dt (default: 21, wider than velocity to reduce derivative noise)')
    parser.add_argument('--smooth-accel-method', type=str, default='savgol',
                       choices=['savgol', 'moving_average'],
                       help='Smoothing method for acceleration: savgol (Savitzky-Golay, preserves peaks, default) or moving_average')
    parser.add_argument('--force-coefficient', type=float, default=1.5,
                       help='Drag coefficient B in F = A·(dω/dt) + B·ω² model (default: 1.5)')
    parser.add_argument('--inertia-coefficient', type=float, default=1.0,
                       help='Inertial coefficient A in F = A·(dω/dt) + B·ω² model (default: 1.0)')
    parser.add_argument('--optimize', action='store_true',
                       help='Optimize A, B and ω₀ to best fit model to measured force (unconstrained)')
    parser.add_argument('--constrain-ratio', action='store_true',
                       help='Physics-constrained optimisation: find velocity scale s and ω₀ such that '
                            'A=s·A_phys, B=s²·B_phys.  Preserves the physics A/B relationship. '
                            'Implies --use-physics --optimize.')
    parser.add_argument('--vx-threshold', type=float, default=0.1,
                       help='Wrist velocity threshold (m/s) for defining the force-fitting region (default: 0.1). '
                            'Force curve is fitted to frames where wrist vx > threshold.')
    parser.add_argument('--force-offset', type=float, default=0.0,
                       help='Time offset for force curve in seconds (positive = shift force later, negative = earlier)')
    parser.add_argument('--keypoint-time-scale', type=float, default=1.0,
                       help='Time scaling factor for keypoints (>1.0 = stretch/slower motion, <1.0 = compress/faster motion). Scales both timestamps and velocities to match force duration.')
    parser.add_argument('--no-perspective-correction', action='store_true',
                       help='Use raw Kalman velocities without perspective correction (may be smoother)')
    parser.add_argument('--use-physics', action='store_true',
                       help='Derive initial A, B from physical parameters (flywheel inertia, drag factor, gear ratio) instead of manual values')
    parser.add_argument('--flywheel-inertia', type=float, default=0.1001,
                       help='Flywheel moment of inertia in kg·m² (Concept2 default: 0.1001)')
    parser.add_argument('--sprocket-radius', type=float, default=0.0175,
                       help='Chain sprocket radius in metres (Concept2 default: 0.0175)')
    parser.add_argument('--gear-ratio', type=float, default=2.4,
                       help='Effective gear ratio handle→flywheel (Concept2 default: 2.4)')
    parser.add_argument('--override-drag-factor', type=int, default=0,
                       help='Override PM5 drag factor (Concept2 units, 0 = use value from JSON)')
    parser.add_argument('--override-stroke-length', type=float, default=0.0,
                       help='Override stroke length in metres (0 = use value from JSON)')
    parser.add_argument('--rower-mass', type=float, default=0.0,
                       help='Rower body mass in kg for Colloud (2006) stretcher force. '
                            'F_stretcher = m·a_CoM − F_handle.  '
                            'Uses whole-body centre-of-mass acceleration from keypoint '
                            'velocities (Dumas et al. anthropometric model). '
                            '0 = disabled (default: 0).')
    parser.add_argument('--pixels-per-meter', type=float, default=0.0,
                       help='Camera scale: pixels per metre for Kalman velocity conversion. '
                            '0 = auto-calibrate from PM5 stroke length and pixel displacement '
                            '(default: 0). Set manually if PM5 data is unavailable.')
    parser.add_argument('--handle-offset', type=float, default=0.15,
                       help='Offset from wrist to handle along forearm direction in metres '
                            '(default: 0.15 m). '
                            'Creates a synthetic "handle" keypoint projected horizontally from the wrist. '
                            'Set to 0 to disable handle computation.')
    parser.add_argument('--keypoint', type=str, default='handle',
                       help='Keypoint to correlate with force (default: handle). '
                            'Available: right_shoulder, right_wrist, right_hip, right_knee, right_ankle, '
                            'handle (uses --handle-offset for smoothed handle velocity)')
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"Keypoint Velocity to Stroke Force Correlation Analyzer")
    print(f"  Keypoint: {args.keypoint}")
    if args.handle_offset > 0:
        print(f"  Handle offset: {args.handle_offset:.3f} m ({args.handle_offset*100:.1f} cm from wrist)")
    if args.pixels_per_meter > 0:
        print(f"  Pixels/metre: {args.pixels_per_meter:.1f} (manual)")
    else:
        print(f"  Pixels/metre: auto (calibrate from PM5 stroke length)")
    if args.rower_mass > 0:
        print(f"  Rower mass: {args.rower_mass:.1f} kg (body inertia term enabled)")
    print("=" * 60)
    print()
    
    if args.synthetic:
        # Original synthetic data mode
        print("Mode: Synthetic Data Simulation")
        print()
        
        # Set up force model parameters
        flywheel = FlywheelParameters(
            force_coefficient=args.force_coefficient  # F = k * v² during drive phase
        )
        
        print(f"Force Model Parameters:")
        print(f"  Force Coefficient: {flywheel.force_coefficient} (F = k * v²)")
        print()
        
        # Create simulator
        dt = 0.01
        simulator = RowingDynamicsSimulator(flywheel, dt=dt)
        
        # Generate sample wrist velocity data
        print("Generating sample wrist velocity data...")
        time, velocity = generate_sample_wrist_velocity(duration=4.0, dt=dt, stroke_rate=24)
        
        # Simulate stroke dynamics
        print("Simulating rowing dynamics...")
        force, angular_velocity, power = simulator.simulate_stroke(time, velocity)
        
        # Calculate metrics
        metrics = calculate_stroke_metrics(force, power, velocity, dt, vx_threshold=args.vx_threshold)
        
        print()
        print("Stroke Metrics:")
        print(f"  Peak Force: {metrics['peak_force']:.2f} N")
        print(f"  Average Force (drive): {metrics['average_force']:.2f} N")
        print(f"  Peak Power: {metrics['peak_power']:.2f} W")
        print(f"  Average Power (drive): {metrics['average_power']:.2f} W")
        print(f"  Total Work: {metrics['total_work']:.2f} J")
        print(f"  Drive Time: {metrics['drive_time']:.2f} s")
        print()
        
        # Correlation analysis
        drive_mask = velocity > args.vx_threshold
        if np.any(drive_mask):
            # Velocity is positive during drive (leftward), force is positive → direct correlation
            correlation = np.corrcoef(velocity[drive_mask], force[drive_mask])[0, 1]
            print(f"Velocity-Force Correlation (drive phase): {correlation:.4f}")
            print()
        
        # Plot results
        print("Generating plots...")
        plot_results(time, velocity, force, angular_velocity, power)
        
        print()
        print("Key Physics Features Implemented:")
        print("  ✓ Force only applied when wrist moving left (positive velocity = drive)")
        print("  ✓ Flywheel inertia (rotational inertia modeled)")
        print("  ✓ Handle velocity = wrist velocity")
        print("  ✓ Drag factor = 150")
        print("  ✓ F = A·(dω/dt) + B·ω²")
        
    else:
        # Real data mode
        print("Mode: Real Data Analysis")
        print("(Use Left/Right arrow keys to navigate between strokes)")
        print()
        
        # Find stroke files
        if args.file:
            if not os.path.exists(args.file):
                print(f"ERROR: File not found: {args.file}")
                return
            stroke_files = [args.file]
        else:
            if not os.path.exists(args.data_folder):
                print(f"ERROR: Data folder not found: {args.data_folder}")
                print(f"Please specify a valid folder with --data-folder")
                print(f"Or use --synthetic for synthetic data mode")
                return
            
            pattern = os.path.join(args.data_folder, FILE_PATTERN)
            stroke_files = sorted(glob.glob(pattern))
        
        if not stroke_files:
            print(f"No stroke files found in {args.data_folder}")
            print(f"Looking for pattern: {FILE_PATTERN}")
            print(f"\nUse --synthetic for synthetic data mode")
            return
        
        print(f"Found {len(stroke_files)} stroke file(s)")
        print(f"Data folder: {args.data_folder if not args.file else os.path.dirname(args.file)}")
        
        # Show configuration options
        if args.no_perspective_correction:
            print(f"\nVelocity Processing:")
            print(f"  Perspective correction: DISABLED (using raw Kalman velocities)")
        
        # Show timing adjustments if any are applied
        if args.force_offset != 0.0 or args.keypoint_time_scale != 1.0:
            print(f"\nTiming Adjustments:")
            if args.keypoint_time_scale != 1.0:
                print(f"  Keypoint Time Scaling: {args.keypoint_time_scale:.3f}x ({'stretch/slower' if args.keypoint_time_scale > 1.0 else 'compress/faster'})")
            if args.force_offset != 0.0:
                print(f"  Force Curve Offset: {args.force_offset:+.3f} s ({'later' if args.force_offset > 0 else 'earlier'})")
        
        print()
        
        # Pre-analyze all stroke files
        # When optimising, use a two-pass approach:
        #   Pass 1: prepare data with initial A, B (no per-stroke optimisation)
        #   Global fit: find single (A, B) or s across all strokes
        #   Pass 2: recompute each stroke with global A, B
        need_global_opt = (args.optimize or args.constrain_ratio) and len(stroke_files) > 1

        if need_global_opt:
            print("Pass 1: Preparing stroke data (initial coefficients)...")
        else:
            print("Analyzing all strokes...")
        all_results = []
        
        for i, filepath in enumerate(stroke_files):
            print(f"\n{'='*60}")
            print(f"{'Preparing' if need_global_opt else 'Analyzing'} stroke {i+1}/{len(stroke_files)}: {os.path.basename(filepath)}")
            print('='*60)
            
            if need_global_opt:
                # Pass 1: use initial A, B without per-stroke optimisation
                # Temporarily disable optimize/constrain_ratio for data preparation
                save_opt, save_cr = args.optimize, args.constrain_ratio
                args.optimize, args.constrain_ratio = False, False
                result = analyze_single_stroke(filepath, args)
                args.optimize, args.constrain_ratio = save_opt, save_cr
            else:
                result = analyze_single_stroke(filepath, args)

            if result is not None:
                all_results.append(result)
                if not need_global_opt:
                    print(f"  Correlation: {result['correlation']:.4f}, RMSE: {result['rmse']:.2f} N ({result['rmse_norm']*100:.1f}%)")
                    print_residual_diagnostics(result, fw_params=result.get('flywheel_params'))
        
        if not all_results:
            print("\nNo strokes could be analyzed successfully.")
            return

        # ---- Global optimisation (single A, B across all strokes) ----
        if need_global_opt:
            fw0 = all_results[0]['flywheel_params']
            A_init = args.inertia_coefficient
            B_init = args.force_coefficient
            if args.use_physics and fw0:
                A_phys, B_phys = fw0.physics_coefficients()
                a_explicit = any(s.startswith('--inertia-coeff') for s in sys.argv)
                b_explicit = any(s.startswith('--force-coeff') for s in sys.argv)
                if A_phys > 0 and not a_explicit:
                    A_init = A_phys
                if B_phys > 0 and not b_explicit:
                    B_init = B_phys

            if args.constrain_ratio:
                df0 = fw0.drag_factor_pm5 if fw0 else 0
                global_result = optimize_global_constrained(
                    all_results, flywheel_params=fw0,
                    drag_factor_pm5=df0,
                    vx_threshold=args.vx_threshold,
                )
            else:
                global_result = optimize_global(
                    all_results,
                    initial_A=A_init, initial_B=B_init,
                    vx_threshold=args.vx_threshold,
                )

            global_A = global_result['A']
            global_B = global_result['B']

            # Pass 2: recompute each stroke with global A, B
            print(f"\nPass 2: Recomputing {len(all_results)} strokes with global A={global_A:.6f}, B={global_B:.6f}...")
            recomputed = []
            for i, prev in enumerate(all_results):
                filepath = prev['filepath']
                print(f"  Stroke {i+1}/{len(all_results)}: {prev['filename']}")
                result = analyze_single_stroke(filepath, args, global_A=global_A, global_B=global_B)
                if result is not None:
                    recomputed.append(result)
                    print(f"    RMSE: {result['rmse']:.2f} N ({result['rmse_norm']*100:.1f}%)")
            all_results = recomputed

            if not all_results:
                print("\nNo strokes survived global recomputation.")
                return

            # Print diagnostics for each stroke with global params
            for result in all_results:
                print_residual_diagnostics(result, fw_params=result.get('flywheel_params'))
        
        # Summary
        all_correlations = [r['correlation'] for r in all_results]
        all_rmse = [r['rmse'] for r in all_results]
        if all_correlations:
            print(f"\n{'='*60}")
            print("SUMMARY")
            print('='*60)
            print(f"Analyzed {len(all_correlations)} strokes successfully")
            if need_global_opt:
                print(f"Global A = {global_A:.6f}, B = {global_B:.6f}")
                if 's' in global_result:
                    print(f"Velocity scale s = {global_result['s']:.6f}")
            print(f"Average Velocity-Force Correlation: {np.mean(all_correlations):.4f}")
            print(f"  Std Dev: {np.std(all_correlations):.4f}")
            print(f"  Min: {np.min(all_correlations):.4f}, Max: {np.max(all_correlations):.4f}")
            print(f"Average RMSE: {np.mean(all_rmse):.2f} N")
            print(f"  Std Dev: {np.std(all_rmse):.2f} N")
            print(f"  Min: {np.min(all_rmse):.2f}, Max: {np.max(all_rmse):.2f} N")
            print()
        
        # Interactive display with arrow-key navigation
        current_idx = [0]
        fig_state = {}
        
        fig = show_stroke(all_results[0], stroke_index=0, total_strokes=len(all_results))
        if fig:
            fig_state['fig'] = fig
            fig.canvas.mpl_connect(
                'key_press_event',
                lambda e: on_key(e, all_results, current_idx, fig_state),
            )
            print("\nKeyboard controls:")
            print("  Left/Right arrows: Navigate between strokes")
            print("  'q': Quit")
            plt.show()
        
        print()
        print("Analysis Features:")
        print("  ✓ Real wrist velocity from Kalman filter (all keypoints)")
        print("  ✓ Real force data from PM5 (mapped to all keypoints)")
        print("  ✓ PM5 drag factor → RPM-modulated drag term")
        print("  ✓ PM5 stroke length → mechanical advantage correction")
        print("  ✓ Estimated flywheel RPM from handle velocity")
        print("  ✓ Physics-based coefficients (--use-physics) from I, c, r")
        print("  ✓ Perspective correction applied")
        print("  ✓ Force generated during positive velocity (leftward drive/pull phase)")
        print("  ✓ F = A·(dω/dt) + B·ω² force model")
        print("  ✓ Visual comparison: Actual vs Initial vs Optimized")
        print("  ✓ Arrow-key navigation between strokes")


if __name__ == "__main__":
    main()
