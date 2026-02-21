#!/usr/bin/env python3
"""
Wrist Velocity to Stroke Force Correlation Script

This script analyzes the correlation between right wrist velocity and stroke force
from recorded rowing data, accounting for flywheel inertia, coasting, and drag factor.

Physics Model:
- Force is only applied during positive (right) wrist motion
- Flywheel has rotational inertia
- Flywheel coasts when not driven
- Drag factor affects resistance
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
from scipy.optimize import minimize


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


def extract_wrist_data_from_file(filepath: str, apply_perspective: bool = True) -> Dict:
    """
    Extract right wrist velocity and force data from stroke JSON file.
    
    Args:
        filepath: Path to stroke JSON file
        apply_perspective: Whether to apply perspective correction (default: True)
    
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
    
    # Extract right wrist data
    right_wrist_x = []
    right_wrist_vx_kalman = []
    right_wrist_vy_kalman = []
    
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
        
        # Extract right wrist data
        right_wrist_coords = kp_dict.get('right_wrist')
        right_wrist_vel = kp_vel_dict.get('right_wrist')
        
        right_wrist_x.append(right_wrist_coords[0] if right_wrist_coords else np.nan)
        right_wrist_vx_kalman.append(right_wrist_vel[0] if right_wrist_vel else np.nan)
        right_wrist_vy_kalman.append(right_wrist_vel[1] if right_wrist_vel else np.nan)
    
    # Convert to numpy arrays
    right_wrist_vx = np.array(right_wrist_vx_kalman)
    right_wrist_vy = -np.array(right_wrist_vy_kalman)  # Flip Y (same as analyze_strokes)
    
    # Calculate speed magnitude (in pixels/second from Kalman filter)
    right_wrist_speed = np.sqrt(right_wrist_vx**2 + right_wrist_vy**2)
    
    # Convert to m/s (assuming 100 pixels per meter, same as analyze_strokes.py)
    pixels_per_meter = 100.0
    right_wrist_vx_ms = right_wrist_vx / pixels_per_meter
    right_wrist_speed_ms = right_wrist_speed / pixels_per_meter
    
    return {
        'timestamps': np.array(frame_timestamps),
        'wrist_vx': right_wrist_vx_ms,  # Horizontal velocity in m/s
        'wrist_speed': right_wrist_speed_ms,  # Total speed in m/s
        'force': np.array(force_curve),
        'phases': np.array(phases),
        'fps': detected_fps,
        'filename': os.path.basename(filepath)
    }


@dataclass
class FlywheelParameters:
    """Physical parameters for the flywheel model"""
    inertia: float = 0.1001  # kg·m² (Concept2 standard)
    drag_factor: float = 150.0  # Drag coefficient (Concept2 units)
    radius: float = 0.15  # meters (flywheel radius)
    force_coefficient: float = 1.5  # Force scaling: F = coefficient * v² (FIXED)
    initial_angular_velocity: float = 0.0  # rad/s


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
        
        Only applies when handle is moving LEFT (negative velocity) - the drive/pull phase
        Torque is reduced when flywheel is already spinning (coasting effect)
        
        Args:
            handle_velocity: Linear velocity of handle/wrist (m/s)
            
        Returns:
            Drive torque (N·m)
        """
        # Force only applied when moving left (toward rower) = negative velocity
        if handle_velocity >= 0:
            return 0.0
        
        # Calculate expected angular velocity from handle velocity
        # Negative handle velocity (leftward) creates positive angular velocity (forward spin)
        target_angular_velocity = -handle_velocity / self.params.radius
        
        # Coasting effect: reduce drive torque based on how fast flywheel is already spinning
        velocity_difference = target_angular_velocity - self.angular_velocity
        
        # If flywheel is already spinning faster than handle would drive it, no drive torque
        if velocity_difference <= 0:
            return 0.0
        
        # Drive torque proportional to velocity difference (coasting reduces effectiveness)
        # Using a coupling coefficient to model chain/handle connection
        coupling_coefficient = 50.0  # N·m·s/rad
        drive_torque = coupling_coefficient * velocity_difference
        
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
        - During drive (negative velocity): rower pulls left, force applied
        - During recovery (positive velocity): no force
        
        Args:
            handle_velocity: Linear velocity of handle/wrist (m/s)
            
        Returns:
            Stroke force (N)
        """
        # Force only applied during drive phase (negative velocity = pulling leftward)
        if handle_velocity >= 0:
            return 0.0
        
        # Force proportional to velocity squared: F = k * v²
        # Use absolute value since velocity is negative during drive
        velocity_mag = abs(handle_velocity)
        force = self.params.force_coefficient * velocity_mag ** 2
        
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
            # Power is force × speed (use absolute value since velocity is negative during drive)
            power_array[i] = force * abs(velocity)
        
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
            # Drive phase: negative velocity (pulling leftward toward rower)
            phase = cycle_time / drive_duration
            # Bell curve shape for velocity
            velocity_array[i] = -1.5 * np.sin(np.pi * phase)
        else:
            # Recovery phase: positive velocity (return rightward)
            recovery_time = cycle_time - drive_duration
            recovery_duration = stroke_period - drive_duration
            phase = recovery_time / recovery_duration
            velocity_array[i] = 0.8 * np.sin(np.pi * phase)
    
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
    axes[2].set_title('Flywheel Angular Velocity (showing inertia & coasting)')
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
                             dt: float) -> dict:
    """
    Calculate stroke performance metrics
    
    Args:
        force: Force array
        power: Power array
        velocity: Velocity array
        dt: Time step
        
    Returns:
        Dictionary of metrics
    """
    # Find drive phases (negative velocity = pulling leftward)
    drive_mask = velocity < 0
    
    metrics = {
        'peak_force': np.max(force),
        'average_force': np.mean(force[drive_mask]) if np.any(drive_mask) else 0,
        'peak_power': np.max(power),
        'average_power': np.mean(power[drive_mask]) if np.any(drive_mask) else 0,
        'total_work': np.sum(power) * dt,  # Joules
        'drive_time': np.sum(drive_mask) * dt,
    }
    
    return metrics


def compute_force_model(wrist_vx: np.ndarray,
                        accel: np.ndarray,
                        A: float,
                        B: float,
                        force_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute two-term flywheel force model: F = A*(-dv/dt) + B*v²

    Active only when BOTH conditions are true:
      - v < 0  (wrist moving toward rower = drive direction)
      - force_mask[i] is True (PM5 reports non-zero force = chain is loaded)

    The second condition removes the catch-slack period (wrist crossing zero
    before the chain loads) and the finish coast (wrist still moving backward
    after PM5 force drops to zero) which would otherwise inject large spurious
    inertial-term forces into the model.

    Args:
        force_mask: Boolean array, True where PM5 force >= threshold.
                    If None, only the v < 0 condition is used (old behaviour).

    Returns:
        force_model, force_inertial, force_drag  (all same shape as wrist_vx)
    """
    force_model = np.zeros_like(wrist_vx)
    force_inertial = np.zeros_like(wrist_vx)
    force_drag = np.zeros_like(wrist_vx)
    for i, vel in enumerate(wrist_vx):
        if vel < 0 and (force_mask is None or force_mask[i]):
            inertial = A * (-accel[i])
            drag = B * vel**2
            force_inertial[i] = inertial
            force_drag[i] = drag
            force_model[i] = max(0.0, inertial + drag)
    return force_model, force_inertial, force_drag


def optimize_model_parameters(wrist_vx: np.ndarray,
                              accel: np.ndarray,
                              force_raw: np.ndarray,
                              force_timestamps: np.ndarray,
                              timestamps: np.ndarray,
                              initial_A: float = 1.0,
                              initial_B: float = 1.5,
                              smooth_accel_window: int = 21,
                              force_threshold: float = 1.0) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Optimize A, B and timing delay to best fit F = A*(-dv/dt) + B*v² to measured force.

    The delay shifts the force curve in time (positive = force appears later).
    A and B are constrained to be non-negative.

    Args:
        wrist_vx:          Smoothed wrist velocity on analysis timestamps
        accel:             Smoothed acceleration on analysis timestamps
        force_raw:         Raw PM5 force samples
        force_timestamps:  Timestamps for force_raw samples (already offset-adjusted)
        timestamps:        Analysis timestamp grid
        initial_A:         Starting guess for inertial coefficient
        initial_B:         Starting guess for drag coefficient
        smooth_accel_window: Window used for accel smoothing (kept for reference)

    Returns:
        (opt_params dict, force_model, force_inertial, force_drag)
    """

    def interpolate_force(delay: float) -> np.ndarray:
        """Re-interpolate PM5 force onto analysis grid with a timing delay."""
        return np.interp(
            timestamps,
            force_timestamps + delay,  # shift force curve by delay seconds
            force_raw,
            left=0.0,
            right=0.0
        )

    def objective(params):
        A_c = max(0.0, params[0])
        B_c = max(0.0, params[1])
        delay = params[2]
        force_target = interpolate_force(delay)
        # Gate model on PM5 activity at this delay — same threshold as final output
        mask = force_target >= force_threshold
        fm, _, _ = compute_force_model(wrist_vx, accel, A_c, B_c, force_mask=mask)
        return np.sqrt(np.mean((force_target - fm)**2))

    x0 = [initial_A, initial_B, 0.0]

    print("  Optimizing A, B, delay (Nelder-Mead)...")
    result = minimize(
        objective,
        x0,
        method='Nelder-Mead',
        options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True}
    )

    A_opt = max(0.0, result.x[0])
    B_opt = max(0.0, result.x[1])
    delay_opt = result.x[2]

    opt_params = {
        'inertia_coefficient': A_opt,
        'force_coefficient': B_opt,
        'delay': delay_opt,
    }

    force_target_opt = interpolate_force(delay_opt)
    mask_opt = force_target_opt >= force_threshold
    force_model_opt, force_inertial_opt, force_drag_opt = compute_force_model(
        wrist_vx, accel, A_opt, B_opt, force_mask=mask_opt
    )

    rmse = np.sqrt(np.mean((force_target_opt - force_model_opt)**2))
    print(f"  Optimization complete:")
    print(f"    A (inertial) = {A_opt:.4f}  (initial: {initial_A})")
    print(f"    B (drag)     = {B_opt:.4f}  (initial: {initial_B})")
    print(f"    delay        = {delay_opt:+.4f} s (positive = force shifted later)")
    print(f"    RMSE         = {rmse:.2f} N")

    return opt_params, force_model_opt, force_inertial_opt, force_drag_opt, force_target_opt


def plot_single_stroke_analysis(time: np.ndarray,
                                wrist_vx: np.ndarray,
                                wrist_speed: np.ndarray,
                                force_actual: np.ndarray,
                                force_model: np.ndarray,
                                filename: str,
                                opt_params: Dict = None,
                                drive_start_frame: int = 0):
    """
    Plot analysis for a single stroke comparing actual and model force with velocity overlay.
    
    Args:
        time: Time array for analysis window (all keypoints)
        wrist_vx: Horizontal wrist velocity
        wrist_speed: Wrist speed magnitude
        force_actual: Actual force curve from PM5
        force_model: Model force (F = k * v²)
        filename: Name of the file
        opt_params: Model parameters dictionary
        drive_start_frame: Frame number where analysis window starts (for alignment)
    """
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1])
    
    # Use actual frame numbers from the stroke, not starting at 0
    frame_indices = np.arange(drive_start_frame, drive_start_frame + len(time))
    
    # Top plot: Force and velocity comparison
    # Plot force on primary y-axis
    ax1.plot(frame_indices, force_actual, 'r-', linewidth=2.5, label='Actual Force (PM5)', alpha=0.9)
    if opt_params and 'inertia_coefficient' in opt_params:
        force_coeff_label = f"A={opt_params['inertia_coefficient']:.2f}, B={opt_params['force_coefficient']:.2f}"
    elif opt_params:
        force_coeff_label = f"k={opt_params['force_coefficient']:.2f}"
    else:
        force_coeff_label = "Model"
    ax1.plot(frame_indices, force_model, 'g-', linewidth=2, label=f'Model Force ({force_coeff_label})', alpha=0.8)
    ax1.set_ylabel('Force (N)', fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3)
    ax1.fill_between(frame_indices, 0, force_actual, alpha=0.15, color='red')
    
    # Create secondary y-axis for velocity (clamped to show only pulling)
    ax2 = ax1.twinx()
    pulling_velocity = np.maximum(0, -wrist_vx)  # Clamp to 0, show only rightward (pulling) velocity
    ax2.plot(frame_indices, pulling_velocity, 'b-', linewidth=2, label='Pulling Velocity (clamped to 0+)', alpha=0.7)
    ax2.set_ylabel('Pulling Velocity (m/s)', fontsize=12, color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.axhline(y=0, color='b', linestyle='--', alpha=0.3, linewidth=1)
    
    # Title
    title_str = f'Force and Velocity Comparison - {filename}'
    if opt_params and 'inertia_coefficient' in opt_params:
        title_str += f"\nModel: F = {opt_params['inertia_coefficient']:.2f}·(-dv/dt) + {opt_params['force_coefficient']:.2f}·v²"
    elif opt_params:
        title_str += f"\nModel: F = {opt_params['force_coefficient']:.2f} * v²"
    ax1.set_title(title_str, fontsize=11)
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Bottom plot: Residuals (Actual - Model)
    residuals = force_actual - force_model
    ax3.plot(frame_indices, residuals, 'purple', linewidth=2, label='Residuals (Actual - Model)', alpha=0.8)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
    ax3.fill_between(frame_indices, 0, residuals, alpha=0.3, color='purple')
    ax3.set_ylabel('Residual Force (N)', fontsize=12)
    ax3.set_xlabel('Frame Number', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left')
    
    # Calculate and display residual statistics
    rmse = np.sqrt(np.mean(residuals**2))
    mean_residual = np.mean(residuals)
    residual_text = f'RMSE: {rmse:.2f} N | Mean: {mean_residual:.2f} N'
    ax3.text(0.98, 0.95, residual_text, transform=ax3.transAxes, 
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    print(f"Displaying plot for {filename}...")
    plt.show(block=True)


def main():
    """Main execution function"""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Correlate wrist velocity with stroke force from recorded data'
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
    parser.add_argument('--smooth-window', type=int, default=11,
                       help='Smoothing window size for velocity (default: 11, same as analyze_strokes)')
    parser.add_argument('--smooth-force-window', type=int, default=11,
                       help='Smoothing window size for force (default: 11)')
    parser.add_argument('--smooth-accel-window', type=int, default=21,
                       help='Smoothing window size for computed acceleration dv/dt (default: 21, wider than velocity to reduce derivative noise)')
    parser.add_argument('--force-coefficient', type=float, default=1.5,
                       help='Drag coefficient B in F = A*(-dv/dt) + B*v² model (default: 1.5)')
    parser.add_argument('--inertia-coefficient', type=float, default=1.0,
                       help='Inertial coefficient A in F = A*(-dv/dt) + B*v² model (default: 1.0)')
    parser.add_argument('--optimize', action='store_true',
                       help='Optimize A, B and timing delay to best fit model to measured force')
    parser.add_argument('--force-threshold', type=float, default=1.0,
                       help='Minimum PM5 force (N) to consider the chain loaded; model outputs zero below this threshold, eliminating catch-slack and finish-coast artefacts (default: 1.0 N)')
    parser.add_argument('--force-offset', type=float, default=0.0,
                       help='Time offset for force curve in seconds (positive = shift force later, negative = earlier)')
    parser.add_argument('--keypoint-time-scale', type=float, default=1.0,
                       help='Time scaling factor for keypoints (>1.0 = stretch/slower motion, <1.0 = compress/faster motion). Scales both timestamps and velocities to match force duration.')
    parser.add_argument('--no-perspective-correction', action='store_true',
                       help='Use raw Kalman velocities without perspective correction (may be smoother)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Wrist Velocity to Stroke Force Correlation Analyzer")
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
        metrics = calculate_stroke_metrics(force, power, velocity, dt)
        
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
        drive_mask = velocity < 0
        if np.any(drive_mask):
            # Use absolute velocity for correlation (since both should increase together)
            correlation = np.corrcoef(np.abs(velocity[drive_mask]), force[drive_mask])[0, 1]
            print(f"Velocity-Force Correlation (drive phase): {correlation:.4f}")
            print()
        
        # Plot results
        print("Generating plots...")
        plot_results(time, velocity, force, angular_velocity, power)
        
        print()
        print("Key Physics Features Implemented:")
        print("  ✓ Force only applied when wrist moving right (positive velocity)")
        print("  ✓ Flywheel inertia (rotational inertia modeled)")
        print("  ✓ Flywheel coasting (continues spinning after drive)")
        print("  ✓ Handle velocity = wrist velocity")
        print("  ✓ Drag factor = 150")
        print("  ✓ Coasting reduces handle velocity effect on force")
        
    else:
        # Real data mode
        print("Mode: Real Data Analysis")
        print("(Close plot windows to continue to next stroke)")
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
        
        # Analyze each stroke file
        all_correlations = []
        
        for i, filepath in enumerate(stroke_files):
            print(f"\n{'='*60}")
            print(f"Analyzing stroke {i+1}/{len(stroke_files)}: {os.path.basename(filepath)}")
            print('='*60)
            
            try:
                # Extract data from file
                stroke_data = extract_wrist_data_from_file(filepath, apply_perspective=not args.no_perspective_correction)
                
                timestamps = stroke_data['timestamps']
                wrist_vx = stroke_data['wrist_vx']
                wrist_speed = stroke_data['wrist_speed']
                force = stroke_data['force']
                phases = stroke_data['phases']
                fps = stroke_data['fps']
                
                # Find drive phase BEFORE time scaling to get original force curve timing
                drive_mask = phases == 2  # Phase 2 = DRIVE
                drive_indices = np.where(drive_mask)[0]
                
                if len(drive_indices) == 0:
                    print("WARNING: No drive phase detected in this stroke")
                    continue
                
                drive_start_frame = drive_indices[0]
                drive_end_frame = drive_indices[-1]
                
                # Save ORIGINAL drive phase timing for force curve (before any time scaling)
                original_drive_start_time = timestamps[drive_start_frame]
                original_drive_end_time = timestamps[drive_end_frame]
                original_drive_duration = original_drive_end_time - original_drive_start_time
                
                # Apply force offset FIRST (before keypoint time scaling)
                # This shifts the force curve timing relative to keypoints
                force_start_time = original_drive_start_time + args.force_offset
                force_end_time = original_drive_end_time + args.force_offset
                if args.force_offset != 0.0:
                    print(f"\nApplied force offset: {args.force_offset:+.3f}s")
                    print(f"  Force time range: {force_start_time:.3f} to {force_end_time:.3f} s")
                
                # Apply time scaling to keypoint timestamps if requested
                # NOTE: Force curve timing is NOT affected by keypoint scaling
                if args.keypoint_time_scale != 1.0:
                    # Scale timestamps relative to start time
                    start_time = timestamps[0]
                    timestamps = start_time + (timestamps - start_time) * args.keypoint_time_scale
                    # Adjust FPS accordingly (scaled time = different effective frame rate)
                    fps = fps / args.keypoint_time_scale
                    # Scale velocities inversely (same motion over longer time = slower velocity)
                    wrist_vx = wrist_vx / args.keypoint_time_scale
                    wrist_speed = wrist_speed / args.keypoint_time_scale
                    print(f"\nApplied time scaling to keypoints: {args.keypoint_time_scale:.3f}x")
                    print(f"  Original duration: {(stroke_data['timestamps'][-1] - stroke_data['timestamps'][0]):.3f} s")
                    print(f"  Scaled duration: {(timestamps[-1] - timestamps[0]):.3f} s")
                    print(f"  Effective FPS: {fps:.1f} Hz")
                    print(f"  Velocities scaled by: {1.0/args.keypoint_time_scale:.3f}x (inverse of time scale)")
                    print(f"  Force curve duration: UNCHANGED ({original_drive_duration:.3f}s)")
                
                print(f"\nFPS: {fps:.1f} Hz")
                print(f"Frames: {len(timestamps)}")
                print(f"Duration: {timestamps[-1] - timestamps[0]:.2f} seconds")
                print(f"Force samples: {len(force)}")
                print()
                
                print(f"Drive phase (detected): frames {drive_start_frame} to {drive_end_frame}")
                
                # Use full stroke (all keypoints)
                wrist_vx_analysis = wrist_vx
                wrist_speed_analysis = wrist_speed
                timestamps_analysis = timestamps
                
                print(f"\nAnalysis window: all {len(wrist_vx_analysis)} keypoints in stroke")
                
                # Map force curve to analysis window
                if len(force) > 0:
                    # Force samples span the DETECTED drive phase (with offset already applied)
                    # The PM5 force curve corresponds to the drive phase only
                    
                    # Create timestamps for force samples (using offset applied earlier)
                    force_timestamps = np.linspace(
                        force_start_time,
                        force_end_time,
                        len(force)
                    )
                    
                    # Interpolate force to match analysis timestamps (all keypoints)
                    # Force will be zero outside the detected drive phase
                    force_actual_interp = np.interp(
                        timestamps_analysis,   # All keypoint timestamps in analysis window
                        force_timestamps,      # Force sample timestamps (drive phase only)
                        force,                 # Force values
                        left=0.0,             # Zero force before drive starts
                        right=0.0             # Zero force after drive ends
                    )
                    
                    print(f"\nForce interpolation:")
                    print(f"  Force samples: {len(force)} (from PM5, drive phase only)")
                    print(f"  Force time range: {force_timestamps[0]:.3f} to {force_timestamps[-1]:.3f} s")
                    print(f"  Analysis samples: {len(wrist_vx_analysis)} (all keypoints)")
                    print(f"  Analysis time range: {timestamps_analysis[0]:.3f} to {timestamps_analysis[-1]:.3f} s")
                else:
                    print("WARNING: No force data available")
                    force_actual_interp = np.zeros_like(wrist_vx_analysis)
                
                # Apply smoothing to reduce noise (unless disabled)
                if not args.no_smooth:
                    print("\nApplying smoothing filters...")
                    wrist_vx_analysis_raw = wrist_vx_analysis.copy()
                    force_actual_raw = force_actual_interp.copy()
                    
                    # Smooth velocity (same as analyze_strokes.py)
                    wrist_vx_analysis = moving_average(wrist_vx_analysis, window_size=args.smooth_window)
                    wrist_speed_analysis = moving_average(wrist_speed_analysis, window_size=args.smooth_window)
                    
                    # Smooth force
                    force_actual_interp = moving_average(force_actual_interp, window_size=args.smooth_force_window)
                    
                    print(f"  Velocity smoothed: window={args.smooth_window}, moving average")
                    print(f"  Force smoothed: window={args.smooth_force_window}, moving average")
                    print(f"  Acceleration smoothed: window={args.smooth_accel_window}, moving average (applied after gradient)")
                else:
                    print("\nSmoothing disabled (--no-smooth)")
                
                # Compute acceleration via central differences on smoothed velocity
                accel = np.gradient(wrist_vx_analysis, timestamps_analysis)

                # Smooth acceleration separately — numerical differentiation amplifies
                # high-frequency noise, so a wider window is applied independently
                if not args.no_smooth:
                    accel = moving_average(accel, window_size=args.smooth_accel_window)

                # ----------------------------------------------------------------
                # Two-term flywheel model: F = A*(-dv/dt) + B*v²
                # ----------------------------------------------------------------
                # PM5 activity mask: True where chain is loaded (force >= threshold)
                force_active_mask = force_actual_interp >= args.force_threshold
                n_masked = np.sum(~force_active_mask & (wrist_vx_analysis < 0))
                if n_masked > 0:
                    print(f"\nForce threshold: {args.force_threshold} N — masking {n_masked} frame(s) "
                          f"where v<0 but PM5 force < threshold (catch slack / finish coast)")

                if args.optimize:
                    print("\nOptimizing model parameters...")
                    (
                        opt_params,
                        force_model,
                        force_model_inertial,
                        force_model_drag,
                        force_actual_interp,   # re-interpolated with optimized delay
                    ) = optimize_model_parameters(
                        wrist_vx_analysis,
                        accel,
                        force,
                        force_timestamps,
                        timestamps_analysis,
                        initial_A=args.inertia_coefficient,
                        initial_B=args.force_coefficient,
                        smooth_accel_window=args.smooth_accel_window,
                        force_threshold=args.force_threshold,
                    )
                    print(f"\nUsing optimized model: F = A*(-dv/dt) + B*v²")
                    print(f"  A (inertial) = {opt_params['inertia_coefficient']:.4f}")
                    print(f"  B (drag)     = {opt_params['force_coefficient']:.4f}")
                    print(f"  delay        = {opt_params['delay']:+.4f} s")
                else:
                    print("\nUsing model: F = A*(-dv/dt) + B*v²")
                    print(f"  A (inertial) = {args.inertia_coefficient}")
                    print(f"  B (drag)     = {args.force_coefficient}")
                    print(f"  Force gate threshold: {args.force_threshold} N")
                    opt_params = {
                        'force_coefficient': args.force_coefficient,
                        'inertia_coefficient': args.inertia_coefficient,
                        'delay': 0.0,
                    }
                    force_model, force_model_inertial, force_model_drag = compute_force_model(
                        wrist_vx_analysis, accel,
                        args.inertia_coefficient, args.force_coefficient,
                        force_mask=force_active_mask
                    )
                
                # Calculate RMSE for reporting
                rmse = np.sqrt(np.mean((force_actual_interp - force_model)**2))
                rmse_norm = rmse / np.max(force_actual_interp) if np.max(force_actual_interp) > 0 else 0
                
                A_label = opt_params['inertia_coefficient']
                B_label = opt_params['force_coefficient']
                D_label = opt_params.get('delay', 0.0)
                print(f"\nModel Output (F = {A_label:.4f}*(-dv/dt) + {B_label:.4f}*v², delay={D_label:+.4f}s, all {len(wrist_vx_analysis)} frames):")
                print(f"  {'Frame':<7} {'Vel(m/s)':<11} {'Accel(m/s²)':<13} {'F_inertial':<12} {'F_drag':<10} {'F_model':<10} {'F_actual':<10} {'Diff':<10}")
                print(f"  {'-'*7} {'-'*11} {'-'*13} {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

                for idx in range(len(wrist_vx_analysis)):
                    diff = force_model[idx] - force_actual_interp[idx]
                    print(f"  {idx:<7} {wrist_vx_analysis[idx]:<11.4f} {accel[idx]:<13.4f} {force_model_inertial[idx]:<12.2f} {force_model_drag[idx]:<10.2f} {force_model[idx]:<10.2f} {force_actual_interp[idx]:<10.2f} {diff:<10.2f}")
                
                print(f"\nRMSE: {rmse:.2f} N ({rmse_norm*100:.1f}% of peak force)")
                
                # Filter negative wrist velocity (moving left toward rower - the DRIVE/PULL phase)
                negative_vx_mask = wrist_vx_analysis < 0
                
                if np.sum(negative_vx_mask) < 3:
                    print("WARNING: Insufficient negative wrist velocity samples (drive phase)")
                    continue
                
                wrist_vx_drive_phase = wrist_vx_analysis[negative_vx_mask]
                force_actual_drive = force_actual_interp[negative_vx_mask]
                force_model_drive = force_model[negative_vx_mask]
                
                # Calculate residuals
                residuals = force_actual_interp - force_model
                residuals_drive = force_actual_drive - force_model_drive
                
                # Calculate correlations
                if len(wrist_vx_drive_phase) >= 3 and len(force_actual_drive) >= 3:
                    correlation_actual = np.corrcoef(np.abs(wrist_vx_drive_phase), force_actual_drive)[0, 1]
                    correlation_model = np.corrcoef(np.abs(wrist_vx_drive_phase), force_model_drive)[0, 1]
                    all_correlations.append(correlation_actual)
                    
                    print()
                    print("="*70)
                    print("Summary Statistics")
                    print("="*70)
                    print(f"\nVelocity (m/s):")
                    print(f"  Range: {np.min(wrist_vx_analysis):.3f} to {np.max(wrist_vx_analysis):.3f}")
                    print(f"  Drive phase frames: {np.sum(wrist_vx_analysis < 0)}/{len(wrist_vx_analysis)}")
                    print(f"  Max pulling velocity: {np.min(wrist_vx_analysis):.3f} m/s")
                    
                    print(f"\nActual Force (N):")
                    print(f"  Peak: {np.max(force_actual_interp):.2f}")
                    print(f"  Mean (drive): {np.mean(force_actual_drive):.2f}")
                    
                    print(f"\nModel Force (F = {A_label:.4f}*(-dv/dt) + {B_label:.4f}*v², delay={D_label:+.4f}s):")
                    print(f"  Peak: {np.max(force_model):.2f} N")
                    print(f"  Mean (drive): {np.mean(force_model_drive):.2f} N")
                    print(f"  Peak inertial term: {np.max(force_model_inertial):.2f} N")
                    print(f"  Peak drag term: {np.max(force_model_drag):.2f} N")
                    print(f"  Correlation: {correlation_model:.4f}")
                    print(f"  RMSE: {rmse:.2f} N ({rmse_norm*100:.1f}% of peak)")
                    
                    print(f"\nResiduals (Actual - Model):")
                    print(f"  Mean: {np.mean(residuals):.2f} N")
                    print(f"  Std Dev: {np.std(residuals):.2f} N")
                    print(f"  Max over-prediction: {np.min(residuals):.2f} N")
                    print(f"  Max under-prediction: {np.max(residuals):.2f} N")
                    print(f"  Mean (drive only): {np.mean(residuals_drive):.2f} N")
                    print(f"  Std Dev (drive only): {np.std(residuals_drive):.2f} N")
                    print("="*70)
                    
                    # CSV debug output if requested
                    if args.debug_csv:
                        print("\n" + "="*70)
                        print("CSV DEBUG OUTPUT")
                        print("="*70)
                        print("frame,timestamp,velocity_m_s,accel_m_s2,f_inertial,f_drag,model_force,actual_force")
                        for idx in range(len(wrist_vx_analysis)):
                            print(f"{idx},{timestamps_analysis[idx]:.6f},{wrist_vx_analysis[idx]:.6f},"
                                  f"{accel[idx]:.6f},{force_model_inertial[idx]:.4f},"
                                  f"{force_model_drag[idx]:.4f},{force_model[idx]:.4f},{force_actual_interp[idx]:.4f}")
                        print("="*70 + "\n")
                    
                    # Plot this stroke
                    print(f"\nGenerating plots for {stroke_data['filename']}...")
                    plot_single_stroke_analysis(
                        timestamps_analysis,
                        wrist_vx_analysis,
                        wrist_speed_analysis,
                        force_actual_interp,
                        force_model,
                        stroke_data['filename'],
                        opt_params,
                        drive_start_frame=0
                    )
                
            except Exception as e:
                print(f"ERROR analyzing {filepath}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Summary
        if all_correlations:
            print(f"\n{'='*60}")
            print("SUMMARY")
            print('='*60)
            print(f"Analyzed {len(all_correlations)} strokes successfully")
            print(f"Average Velocity-Force Correlation: {np.mean(all_correlations):.4f}")
            print(f"Std Dev: {np.std(all_correlations):.4f}")
            print(f"Min: {np.min(all_correlations):.4f}")
            print(f"Max: {np.max(all_correlations):.4f}")
            print()
        
        print()
        print("Analysis Features:")
        print("  ✓ Real wrist velocity from Kalman filter (all keypoints)")
        print("  ✓ Real force data from PM5 (mapped to all keypoints)")
        print("  ✓ Initial model with default physics parameters")
        print("  ✓ Fixed force coefficient model (F = k*v²)")
        print("  ✓ Perspective correction applied")
        print("  ✓ Force generated during negative velocity (drive/pull phase)")
        print("  ✓ Analysis includes both drive and recovery phases")
        print("  ✓ Adjustable timing window with fraction-based control")
        print("  ✓ Flywheel coasting and inertia effects in model")
        print("  ✓ Visual comparison: Actual vs Initial vs Optimized")


if __name__ == "__main__":
    main()
