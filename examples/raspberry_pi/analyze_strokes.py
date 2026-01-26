#!/usr/bin/env python
# Copyright Axelera AI, 2025
# Quick analysis tool for recorded keypoint data

"""
Analyze recorded stroke data and display basic statistics.
Compatible with cameraerg plotter.py data format.

Automatically searches /tmp/stroke_data/ for stroke files.

Usage:
    python analyze_strokes.py                # Analyze first and last stroke
    python analyze_strokes.py --stroke 5     # Analyze specific stroke
    python analyze_strokes.py --compare      # Compare all strokes
"""

import os
import sys
import json
import gzip
import glob
import argparse
from typing import List, Dict, Any
import numpy as np


def load_stroke_file(filepath: str) -> Dict[str, Any]:
    """Load gzipped JSON stroke file"""
    with gzip.open(filepath, 'rt') as f:
        return json.load(f)


def analyze_stroke(data: Dict[str, Any], filepath: str):
    """Analyze single stroke and print summary"""
    print("\n" + "="*70)
    print(f"File: {os.path.basename(filepath)}")
    print("="*70)
    
    # Basic info
    timestamp = data.get('timestamp', 0)
    frame_count = data.get('frame_count', 0)
    phases = data.get('phases', [])
    keypoints = data.get('keypoints', [])
    velocities = data.get('velocities', [])
    
    print(f"Timestamp: {timestamp} ({time.ctime(timestamp) if timestamp else 'N/A'})")
    print(f"Total frames: {frame_count}")
    print(f"Keypoint frames: {len(keypoints)}")
    print(f"Velocity frames: {len(velocities)}")
    
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
    
    # Keypoint statistics
    if keypoints:
        print("\nKeypoint detection:")
        
        # Count detections per keypoint name
        kp_counts = {}
        for frame_kpts in keypoints:
            for kp in frame_kpts:
                name = kp.get('name', 'unknown')
                kp_counts[name] = kp_counts.get(name, 0) + 1
        
        print("  Keypoint detections:")
        for name, count in sorted(kp_counts.items()):
            pct = 100 * count / len(keypoints)
            print(f"    {name:20s}: {count:3d}/{len(keypoints)} frames ({pct:5.1f}%)")
        
        # Sample first frame keypoints
        if keypoints[0]:
            print("\n  First frame sample:")
            for kp in keypoints[0]:
                print(f"    {kp['name']:20s}: x={kp['x']:4d}, y={kp['y']:4d}, conf={kp.get('confidence', 0):.2f}")
    
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
            print(f"  Duration: {drive_duration} frames")
            
            # Assuming ~90 FPS
            drive_time_ms = drive_duration * (1000 / 90)
            print(f"  Duration: ~{drive_time_ms:.0f}ms @ 90 FPS")
    
    # Force data (if present from ergometer)
    if 'force' in data:
        force = data['force']
        if force:
            print(f"\nForce data:")
            print(f"  Samples: {len(force)}")
            print(f"  Max force: {max(force):.1f} N")
            print(f"  Avg force: {np.mean(force):.1f} N")
            print(f"  Peak: {np.max(force):.1f} N")


def compare_strokes(files: List[str]):
    """Compare multiple strokes and show averages"""
    print("\n" + "="*70)
    print("STROKE COMPARISON")
    print("="*70)
    
    all_drive_durations = []
    all_total_frames = []
    
    for filepath in files:
        data = load_stroke_file(filepath)
        phases = data.get('phases', [])
        
        total_frames = len(phases)
        all_total_frames.append(total_frames)
        
        drive_frames = [i for i, p in enumerate(phases) if p == 2]
        if drive_frames:
            drive_duration = drive_frames[-1] - drive_frames[0] + 1
            all_drive_durations.append(drive_duration)
    
    print(f"\nTotal strokes: {len(files)}")
    
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
    import time
    
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
        '--compare', 
        action='store_true', 
        help='Compare all strokes'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/tmp/stroke_data',
        help='Directory containing stroke files (default: /tmp/stroke_data)'
    )
    args = parser.parse_args()
    
    # Find all stroke files
    pattern = os.path.join(args.data_dir, "stroke_*.json.gz")
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"No stroke files found in {args.data_dir}")
        print(f"Looking for pattern: {pattern}")
        print(f"\nMake sure rowing_ergometer_recording.py has been run and captured strokes.")
        return
    
    print(f"Found {len(files)} stroke file(s) in {args.data_dir}")
    print(f"Data directory: {args.data_dir}")
    
    # Analyze specific stroke
    if args.stroke is not None:
        if args.stroke < 1 or args.stroke > len(files):
            print(f"Error: Stroke number must be between 1 and {len(files)}")
            return
        
        filepath = files[args.stroke - 1]
        data = load_stroke_file(filepath)
        analyze_stroke(data, filepath)
    
    # Compare all strokes
    elif args.compare:
        compare_strokes(files)
        
        # Also analyze each stroke briefly
        for i, filepath in enumerate(files, 1):
            print(f"\n{'='*70}")
            print(f"Stroke {i}/{len(files)}: {os.path.basename(filepath)}")
            print(f"{'='*70}")
            data = load_stroke_file(filepath)
            phases = data.get('phases', [])
            drive_frames = [j for j, p in enumerate(phases) if p == 2]
            drive_duration = drive_frames[-1] - drive_frames[0] + 1 if drive_frames else 0
            print(f"Total frames: {len(phases)}, Drive duration: {drive_duration} frames")
    
    # Default: analyze first and last
    else:
        print("\nAnalyzing first stroke...")
        data = load_stroke_file(files[0])
        analyze_stroke(data, files[0])
        
        if len(files) > 1:
            print("\n\nAnalyzing last stroke...")
            data = load_stroke_file(files[-1])
            analyze_stroke(data, files[-1])
        
        if len(files) > 2:
            print(f"\n\nUse --compare to see comparison of all {len(files)} strokes")
            print(f"Use --stroke N to analyze specific stroke (1-{len(files)})")


if __name__ == "__main__":
    main()
