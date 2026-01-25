#!/usr/bin/env python
# Copyright Axelera AI, 2025

"""
Record frames with keypoint detections to a folder for timing analysis.
This script runs keypoint inference and saves frames with detected keypoints and timing data.

Usage:
    python record_keypoints.py --network yolov8npose-coco --source usb:10/yuyv --output ./recorded_frames
"""

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path

if not os.environ.get('AXELERA_FRAMEWORK'):
    sys.exit("Please activate the Axelera environment with source venv/bin/activate and run again")

from tqdm import tqdm

from axelera.app import (
    config,
    create_inference_stream,
    logging_utils,
    yaml_parser,
)

try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst
except ImportError:
    pass

import cv2
import numpy as np

LOG = logging_utils.getLogger(__name__)
PBAR = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"


def create_output_dir(base_path):
    """Create timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_path) / f"keypoints_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create CSV file for timing data
    timing_file = output_dir / "frame_timing.csv"
    with open(timing_file, 'w') as f:
        f.write("frame_number,timestamp,relative_time_ms,fps,num_keypoints\n")
    
    # Create directory for keypoint data
    keypoints_dir = output_dir / "keypoints_data"
    keypoints_dir.mkdir(exist_ok=True)
    
    return output_dir, timing_file, keypoints_dir


def save_keypoint_data(keypoints_dir, frame_number, meta):
    """Save keypoint metadata to JSON file."""
    if meta is None:
        return 0
    
    keypoint_data = {
        'frame_number': frame_number,
        'detections': []
    }
    
    num_keypoints = 0
    
    # meta is an AxMeta container, need to find the keypoint task meta inside it
    # Iterate through the container to find keypoint detection meta
    task_meta = None
    if hasattr(meta, 'items'):
        # AxMeta acts like a dictionary
        for key, value in meta.items():
            # Look for keypoint-related meta (has keypoints attribute)
            if hasattr(value, 'keypoints') or 'keypoint' in key.lower() or 'pose' in key.lower():
                task_meta = value
                break
    
    if task_meta is None:
        # Fallback: try to access meta directly if it's already a task meta
        if hasattr(meta, 'objects'):
            task_meta = meta
    
    # Extract keypoint information using the objects property
    # For keypoint detection, task_meta.objects returns KeypointObject or KeypointObjectWithBbox instances
    if task_meta and hasattr(task_meta, 'objects'):
        for detection in task_meta.objects:
            det_info = {}
            
            # Get bounding box if available (KeypointObjectWithBbox)
            if hasattr(detection, 'box'):
                det_info['bbox'] = detection.box.tolist() if hasattr(detection.box, 'tolist') else list(detection.box)
            
            # Get keypoints
            if hasattr(detection, 'keypoints'):
                kpts = detection.keypoints
                det_info['keypoints'] = kpts.tolist() if hasattr(kpts, 'tolist') else list(kpts)
                # Count keypoints: shape is (K, 2) or (K, 3)
                num_keypoints += len(kpts) if isinstance(kpts, (list, np.ndarray)) else 0
            
            # Get detection score
            if hasattr(detection, 'score'):
                det_info['confidence'] = float(detection.score)
            
            # Get class_id if available
            if hasattr(detection, 'class_id'):
                det_info['class_id'] = int(detection.class_id)
            
            keypoint_data['detections'].append(det_info)
    
    # Save to JSON
    json_path = keypoints_dir / f"frame_{frame_number:05d}.json"
    with open(json_path, 'w') as f:
        json.dump(keypoint_data, f, indent=2)
    
    return num_keypoints


def record_keypoints_inference(args, stream, output_dir, max_frames):
    """Record frames with keypoint inference."""
    output_dir, timing_file, keypoints_dir = create_output_dir(output_dir)
    
    print(f"Recording keypoint detections to: {output_dir}")
    print(f"Press Ctrl+C to stop recording (max {max_frames} frames)")
    
    frame_count = 0
    start_time = time.time()
    last_time = start_time
    
    try:
        for event in tqdm(
            stream.with_events(),
            desc=f"Recording keypoints... {' ':>20}",
            unit='frames',
            leave=False,
            bar_format=PBAR,
            disable=None,
            total=max_frames,
        ):
            if frame_count >= max_frames:
                break
                
            if not event.result:
                LOG.warning(f"Unknown event received: {event!r}")
                continue
            
            frame_result = event.result
            image, meta = frame_result.image, frame_result.meta
            
            if image is None:
                continue
            
            current_time = time.time()
            relative_time = (current_time - start_time) * 1000  # milliseconds
            frame_fps = 1.0 / (current_time - last_time) if current_time > last_time else 0
            
            # Save keypoint metadata
            num_keypoints = save_keypoint_data(keypoints_dir, frame_count, meta)
            
            # Log timing
            with open(timing_file, 'a') as f:
                f.write(f"{frame_count},{current_time:.6f},{relative_time:.3f},{frame_fps:.2f},{num_keypoints}\n")
            
            frame_count += 1
            last_time = current_time
            
            if frame_count % 30 == 0:
                avg_fps = frame_count / (current_time - start_time)
                LOG.info(f"Recorded {frame_count} frames (avg FPS: {avg_fps:.1f})")
    
    except KeyboardInterrupt:
        print("\nStopping recording...")
    
    finally:
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"Recording complete!")
        print(f"Total frames: {frame_count}")
        print(f"Duration: {total_time:.2f} seconds")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Output directory: {output_dir}")
        print(f"Timing data: {timing_file}")
        print(f"Keypoint data: {keypoints_dir}")
        print(f"{'='*60}")


def main():
    network_yaml_info = yaml_parser.get_network_yaml_info()
    parser = config.create_inference_argparser(
        network_yaml_info, description='Record frames with keypoint detections'
    )
    parser.add_argument(
        '--record-dir',
        type=str,
        default='/tmp/keypoint_data',
        help='Output directory for recorded frames and keypoint data (use /tmp for host access)'
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=300,
        help='Maximum number of frames to record'
    )
    
    args = parser.parse_args()
    
    # Check if network supports keypoints
    if 'pose' not in args.network.lower():
        LOG.warning(f"Network '{args.network}' may not support keypoint detection. Consider using a pose model like 'yolov8npose-coco'")
    
    try:
        stream = create_inference_stream(
            config.SystemConfig.from_parsed_args(args),
            config.InferenceStreamConfig.from_parsed_args(args),
            config.PipelineConfig.from_parsed_args(args),
            config.LoggingConfig.from_parsed_args(args),
            config.DeployConfig.from_parsed_args(args),
        )
        
        record_keypoints_inference(args, stream, args.record_dir, args.max_frames)
        
    except KeyboardInterrupt:
        LOG.exit_with_error_log()
    except logging_utils.UserError as e:
        LOG.exit_with_error_log(e.format())
    except Exception as e:
        LOG.exit_with_error_log(e)
    finally:
        if 'stream' in locals():
            stream.stop()
    
    if Gst.is_initialized():
        Gst.deinit()


if __name__ == "__main__":
    main()
