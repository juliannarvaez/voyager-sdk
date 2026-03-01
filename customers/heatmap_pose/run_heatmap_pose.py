#!/usr/bin/env python
# Copyright Axelera AI, 2025
"""
Run cascade heatmap pose pipeline: YOLOv8n detector → SimpleBaseline heatmap pose.

GStreamer runs both models on AIPU. Stage 2 outputs raw heatmap tensors via
libdecode_to_raw_tensor.so. This script decodes heatmaps → keypoints in Python
and prints wrist confidence per frame.

Usage:
    python run_heatmap_pose.py yolov8n-simplebaseline-pose usb:10/yuyv --no-display
    python run_heatmap_pose.py yolov8n-simplebaseline-pose usb:10/yuyv
"""
import os
import sys
import time

if not os.environ.get('AXELERA_FRAMEWORK'):
    sys.exit("Please activate the Axelera environment: source venv/bin/activate")

import numpy as np

from axelera.app import config, create_inference_stream, display, logging_utils, yaml_parser
from axelera.app.meta.keypoint import CocoBodyKeypointsMeta
from axelera.app.meta.tensor import TensorMeta
from axelera.app.meta.base import AxTaskMeta

try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst
except ImportError:
    pass

LOG = logging_utils.getLogger(__name__)

# COCO keypoint indices
LEFT_WRIST = 9
RIGHT_WRIST = 10
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Skeleton connections for drawing
SKELETON = [
    (15, 13), (13, 11), (11, 5), (5, 6), (6, 12), (12, 14), (14, 16),  # legs + shoulders
    (5, 7), (7, 9),   # left arm
    (6, 8), (8, 10),  # right arm
]

def heatmaps_to_keypoints(heatmaps, box):
    """Decode heatmap tensor [K, H, W] to keypoints in image coordinates.

    Resolution-agnostic: maps heatmap grid directly to the detection box.

    Args:
        heatmaps: np.ndarray shape (K, H_hm, W_hm)
        box: [x1, y1, x2, y2] bounding box in image coordinates

    Returns:
        keypoints: np.ndarray shape (K, 3) — [x, y, visibility] in image coords
    """
    K, H, W = heatmaps.shape
    keypoints = np.zeros((K, 3), dtype=np.float32)
    x1, y1, x2, y2 = box
    box_w = x2 - x1
    box_h = y2 - y1

    for k in range(K):
        hm = heatmaps[k]
        idx = np.argmax(hm)
        y_hm, x_hm = np.unravel_index(idx, (H, W))

        # Sub-pixel refinement
        dx = 0.0
        dy = 0.0
        if 0 < x_hm < W - 1:
            dx = (float(hm[y_hm, x_hm + 1]) - float(hm[y_hm, x_hm - 1])) * 0.25
        if 0 < y_hm < H - 1:
            dy = (float(hm[y_hm + 1, x_hm]) - float(hm[y_hm - 1, x_hm])) * 0.25

        # Heatmap coords → box coords (direct mapping, no model_w/h intermediate)
        keypoints[k, 0] = (x_hm + dx + 0.5) / W * box_w + x1
        keypoints[k, 1] = (y_hm + dy + 0.5) / H * box_h + y1
        keypoints[k, 2] = float(hm[y_hm, x_hm])  # visibility = peak value

    return keypoints


def process_frame_meta(meta, diag=False):
    """Extract detection boxes and decode heatmap tensors into keypoints.

    Returns list of (keypoints_17x3, box_4, person_score) tuples.
    """
    results = []

    # Find detection meta (Stage 1)
    det_meta = None
    tensor_meta = None
    tensor_key = None
    for key, val in meta.items():
        vtype = type(val).__name__
        if 'ObjectDetection' in vtype or 'DetectionMeta' in vtype:
            det_meta = val
        if isinstance(val, TensorMeta):
            tensor_meta = val
            tensor_key = key
        if isinstance(val, CocoBodyKeypointsMeta):
            # Already decoded (torch-aipu path)
            return [(val.keypoints[i], val.boxes[i], val.scores[i]) for i in range(len(val))]

    if det_meta is None or len(det_meta) == 0:
        return results

    if tensor_meta is None:
        # No heatmap tensors — Stage 2 didn't run or no ROIs
        return results

    boxes = det_meta.xyxy() if hasattr(det_meta, 'xyxy') else det_meta.boxes
    n_dets = len(boxes)

    # tensor_meta.tensors contains one tensor per ROI crop
    for i, tensor in enumerate(tensor_meta.tensors):
        if i >= n_dets:
            break
        box = boxes[i]
        score = float(det_meta.scores[i]) if hasattr(det_meta, 'scores') else 1.0

        if diag:
            print(f"  tensor[{i}] shape={tensor.shape} dtype={tensor.dtype} "
                  f"min={tensor.min():.4f} max={tensor.max():.4f} "
                  f"mean={tensor.mean():.4f}")

        # tensor shape: [1, 17, H, W] or [17, H, W] or flattened
        if tensor.ndim == 4:
            heatmaps = tensor[0]
        elif tensor.ndim == 3:
            heatmaps = tensor
        elif tensor.ndim in (1, 2):
            total = tensor.size
            if total % 17 == 0:
                spatial = total // 17
                # Infer H, W assuming 4:3 aspect ratio (standard for pose)
                h_cand = int(np.sqrt(spatial * 4 / 3))
                w_cand = spatial // h_cand if h_cand > 0 else 0
                if h_cand * w_cand == spatial:
                    heatmaps = tensor.reshape(17, h_cand, w_cand)
                else:
                    LOG.warning(f"Cannot infer heatmap dims from {total} elements")
                    continue
            else:
                LOG.warning(f"Cannot reshape tensor of size {total} (not divisible by 17)")
                continue
        else:
            LOG.warning(f"Unexpected tensor shape: {tensor.shape}")
            continue

        # Normalize heatmaps if they seem to be in INT8 range
        hm_max = heatmaps.max()
        if hm_max > 1.0:
            # Likely dequantized INT8 values, apply sigmoid to get [0, 1]
            heatmaps = 1.0 / (1.0 + np.exp(-heatmaps.astype(np.float32)))
        elif hm_max <= 0.0:
            # All zeros or negative — try sigmoid anyway
            heatmaps = 1.0 / (1.0 + np.exp(-heatmaps.astype(np.float32)))

        kpts = heatmaps_to_keypoints(heatmaps, box)
        results.append((kpts, box, score))

    return results


def inference_loop(stream, window):
    """Main inference loop with optional display."""
    frame_num = 0
    last_time = time.time()
    diag_frames = 5  # print tensor diagnostics for first N frames with detections
    diag_count = 0

    for frame_result in stream:
        now = time.time()
        dt = now - last_time
        fps = 1.0 / dt if dt > 0 else 0
        last_time = now

        meta = frame_result.meta
        show_diag = diag_count < diag_frames
        persons = process_frame_meta(meta, diag=show_diag)

        if not persons:
            print(f"[{frame_num:5d}] {fps:5.1f} fps | no detections")
        else:
            if show_diag:
                diag_count += 1
            for i, (kpts, box, score) in enumerate(persons):
                lw = kpts[LEFT_WRIST]
                rw = kpts[RIGHT_WRIST]
                print(
                    f"[{frame_num:5d}] {fps:5.1f} fps | "
                    f"person {i} ({score:.2f}) | "
                    f"L wrist: vis={lw[2]:.3f} ({lw[0]:.0f},{lw[1]:.0f}) | "
                    f"R wrist: vis={rw[2]:.3f} ({rw[0]:.0f},{rw[1]:.0f})"
                )

        # Always remove TensorMeta (crashes renderer) and replace with keypoints if available
        if meta is not None:
            tensor_key = None
            for key, val in meta.items():
                if isinstance(val, TensorMeta):
                    tensor_key = key
                    break
            if tensor_key is not None:
                meta.delete_instance(tensor_key)
                if persons:
                    # Build CocoBodyKeypointsMeta from decoded keypoints
                    all_kpts = np.stack([p[0] for p in persons], axis=0)  # (N, 17, 3)
                    all_boxes = np.stack([np.array(p[1], dtype=np.float32) for p in persons], axis=0)  # (N, 4)
                    all_scores = np.array([p[2] for p in persons], dtype=np.float32)  # (N,)
                    pose_meta = CocoBodyKeypointsMeta(
                        keypoints=all_kpts,
                        boxes=all_boxes,
                        scores=all_scores,
                    )
                    meta.add_instance(tensor_key, pose_meta)

        # Show with renderer (now draws keypoints instead of raw tensors)
        if window is not None:
            window.show(frame_result.image, frame_result.meta, frame_result.stream_id)
            if window.is_closed:
                break

        frame_num += 1


def main():
    network_yaml_info = yaml_parser.get_network_yaml_info()
    parser = config.create_inference_argparser(
        network_yaml_info, description='Cascade heatmap pose estimation'
    )
    args = parser.parse_args()

    stream = create_inference_stream(
        config.SystemConfig.from_parsed_args(args),
        config.InferenceStreamConfig.from_parsed_args(args),
        config.PipelineConfig.from_parsed_args(args),
        config.LoggingConfig.from_parsed_args(args),
        config.DeployConfig.from_parsed_args(args),
    )

    try:
        use_display = not getattr(args, 'no_display', False)
        if use_display:
            with display.App(
                renderer=args.display,
                opengl=stream.hardware_caps.opengl,
                buffering=not stream.is_single_image(),
            ) as app:
                wnd = app.create_window('Heatmap Pose', size=getattr(args, 'window_size', None))
                app.start_thread(inference_loop, (stream, wnd), name='InferenceThread')
                app.run(interval=1 / 30)
        else:
            inference_loop(stream, None)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()

    if Gst.is_initialized():
        Gst.deinit()


if __name__ == "__main__":
    main()
