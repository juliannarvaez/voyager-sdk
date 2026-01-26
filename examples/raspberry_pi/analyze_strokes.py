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
    """Compute angle at point b formed by a-b-c."""
    v1 = np.array([a['x'] - b['x'], a['y'] - b['y']])
    v2 = np.array([c['x'] - b['x'], c['y'] - b['y']])
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
    
    keypoints_list = data.get('keypoints', [])
    phases = data.get('phases', [])
    
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
    
    # Process each frame
    for frame_kpts in keypoints_list:
        # Build keypoint dict for this frame
        kp_dict = {}
        for kp in frame_kpts:
            kp_dict[kp['name']] = (kp['x'], kp['y'])
        
        # Extract coordinates
        shoulder_coords = kp_dict.get('left_shoulder') or kp_dict.get('right_shoulder')
        hip_coords = kp_dict.get('left_hip') or kp_dict.get('right_hip')
        knee_coords = kp_dict.get('left_knee') or kp_dict.get('right_knee')
        ankle_coords = kp_dict.get('left_ankle') or kp_dict.get('right_ankle')
        
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
        'phases': phases
    }


def show_plot(filepath: str, current_file_idx: int, total_files: int):
    """Show interactive plot for a single stroke."""
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
    phases = data['phases']
    
    # Apply smoothing
    window = 3
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
    
    frames = np.arange(len(knee_angles))
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))
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
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Y positions
    ax2.plot(frames, shoulder_y_smooth, 'purple', label='Shoulder Y', linewidth=2)
    ax2.plot(frames, hip_y_smooth, 'blue', label='Hip Y', linewidth=2)
    ax2.plot(frames, knee_y_smooth, 'green', label='Knee Y', linewidth=2)
    ax2.plot(frames, ankle_y_smooth, 'orange', label='Ankle Y', linewidth=2)
    ax2.set_ylabel('Y Position (pixels)', fontsize=12)
    ax2.set_title('Vertical Positions Over Time', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()  # Invert since y=0 is at top
    
    # Plot 3: X positions
    ax3.plot(frames, shoulder_x_smooth, 'purple', label='Shoulder X', linewidth=2)
    ax3.plot(frames, hip_x_smooth, 'blue', label='Hip X', linewidth=2)
    ax3.plot(frames, knee_x_smooth, 'green', label='Knee X', linewidth=2)
    ax3.plot(frames, ankle_x_smooth, 'orange', label='Ankle X', linewidth=2)
    ax3.set_xlabel('Frame', fontsize=12)
    ax3.set_ylabel('X Position (pixels)', fontsize=12)
    ax3.set_title('Horizontal Positions Over Time', fontsize=12)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # Highlight buffer zones (first 30 and last 30 frames)
    buffer_size = 30
    for ax in [ax1, ax2, ax3]:
        if len(frames) > buffer_size:
            ax.axvspan(0, buffer_size, alpha=0.1, color='gray', linestyle='--')
            ax.axvspan(len(frames) - buffer_size, len(frames), alpha=0.1, color='gray', linestyle='--')
    
    plt.tight_layout()
    return fig


def show_averages(files: List[str]):
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
    
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle(f'Average Joint Angles Across {len(files)} Strokes', fontsize=14)
    
    ax.plot(frames, avg_knee_smooth, 'b-', label='Avg Knee Angle', linewidth=2)
    ax.plot(frames, avg_hip_smooth, 'r-', label='Avg Hip Angle', linewidth=2)
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Angle (degrees)', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def on_key(event, files: List[str], current_idx: list):
    """Handle keyboard events for navigation."""
    if event.key == 'right' and current_idx[0] < len(files) - 1:
        current_idx[0] += 1
        plt.close('all')
        fig = show_plot(files[current_idx[0]], current_idx[0], len(files))
        if fig:
            fig.canvas.mpl_connect('key_press_event', lambda e: on_key(e, files, current_idx))
            plt.show()
    elif event.key == 'left' and current_idx[0] > 0:
        current_idx[0] -= 1
        plt.close('all')
        fig = show_plot(files[current_idx[0]], current_idx[0], len(files))
        if fig:
            fig.canvas.mpl_connect('key_press_event', lambda e: on_key(e, files, current_idx))
            plt.show()
    elif event.key == 'a':
        plt.close('all')
        fig = show_averages(files)
        if fig:
            fig.canvas.mpl_connect('key_press_event', lambda e: on_key(e, files, current_idx))
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
        
        phases = data.get('phases', [])
        keypoints = data.get('keypoints', [])
        
        print(f"\nTotal frames: {len(phases)}")
        
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
                
                print(f"\nDrive phase:")
                print(f"  Start frame: {drive_start}")
                print(f"  End frame: {drive_end}")
                print(f"  Duration: {drive_duration} frames (~{drive_duration * 1000 / 90:.0f}ms @ 90 FPS)")
    else:
        # Summary of all strokes
        print(f"\n{'='*70}")
        print(f"SUMMARY OF ALL {len(files)} STROKES")
        print(f"{'='*70}")
        
        all_total_frames = []
        all_drive_durations = []
        
        for filepath in files:
            data = load_stroke_file(filepath)
            phases = data.get('phases', [])
            
            all_total_frames.append(len(phases))
            
            drive_frames = [i for i, p in enumerate(phases) if p == 2]
            if drive_frames:
                drive_duration = drive_frames[-1] - drive_frames[0] + 1
                all_drive_durations.append(drive_duration)
        
        if all_total_frames:
            print(f"\nTotal frames per stroke:")
            print(f"  Min: {min(all_total_frames)}")
            print(f"  Max: {max(all_total_frames)}")
            print(f"  Avg: {np.mean(all_total_frames):.1f}")
        
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
            fig.canvas.mpl_connect('key_press_event', lambda e: on_key(e, files, current_idx))
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
        fig = show_plot(files[current_idx[0]], current_idx[0], len(files))
        if fig:
            fig.canvas.mpl_connect('key_press_event', lambda e: on_key(e, files, current_idx))
            print("\nKeyboard controls:")
            print("  Left/Right arrows: Navigate between strokes")
            print("  'a': Show average across all strokes")
            print("  'q': Quit")
            plt.show()


if __name__ == "__main__":
    main()
