#!/usr/bin/env python
# Copyright Axelera AI, 2025
"""
Run cascade RTMPose pipeline: YOLOv8n detector → RTMPose-M pose estimation.

GStreamer runs both models on AIPU. Stage 2 outputs raw backbone tensors [1,17,8,6]
via libdecode_to_raw_tensor.so. This script runs the SimCC head on CPU
(~0.65ms) to decode keypoint coordinates.

Usage:
    python run_rtmpose.py yolov8n-rtmpose-m usb:10/yuyv --no-display
    python run_rtmpose.py yolov8n-rtmpose-m usb:10/yuyv
"""
import os
import sys
import time
from pathlib import Path

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
    (15, 13), (13, 11), (11, 5), (5, 6), (6, 12), (12, 14), (14, 16),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
]

# SimCC parameters
SIMCC_SPLIT_RATIO = 2.0
MODEL_W = 192
MODEL_H = 256

# Lazy-loaded ORT session for SimCC head
_simcc_session = None


def _get_simcc_session():
    """Get or create the ONNX Runtime session for the SimCC head."""
    global _simcc_session
    if _simcc_session is not None:
        return _simcc_session

    import onnxruntime as ort
    head_path = str(Path(__file__).parent / 'rtmpose-m-simcc-head.onnx')
    if not Path(head_path).exists():
        raise FileNotFoundError(
            f'SimCC head ONNX not found at {head_path}. '
            f'Run the extraction script first.'
        )
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 2
    _simcc_session = ort.InferenceSession(
        head_path, opts, providers=['CPUExecutionProvider']
    )
    LOG.info(f'Loaded SimCC head: {head_path}')
    return _simcc_session


def decode_simcc(features, box):
    """Decode backbone features [1,17,8,6] → keypoints in image coordinates.

    Args:
        features: np.ndarray [1,17,8,6] or [17,8,6] backbone output
        box: [x1, y1, x2, y2] detection bounding box in image coords

    Returns:
        keypoints: np.ndarray [17, 3] — (x, y, confidence) in image coords
    """
    if features.ndim == 3:
        features = features[np.newaxis]
    features = features.astype(np.float32)

    # Run SimCC head on CPU
    sess = _get_simcc_session()
    input_name = sess.get_inputs()[0].name
    simcc_x, simcc_y = sess.run(None, {input_name: features})

    # simcc_x: [1, 17, 384], simcc_y: [1, 17, 512]
    # Argmax → model pixel coords
    x_locs = np.argmax(simcc_x[0], axis=-1).astype(np.float32) / SIMCC_SPLIT_RATIO  # [17]
    y_locs = np.argmax(simcc_y[0], axis=-1).astype(np.float32) / SIMCC_SPLIT_RATIO  # [17]

    # Confidence via softmax peak
    def _softmax_max(logits):
        e = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        p = e / np.sum(e, axis=-1, keepdims=True)
        return np.max(p, axis=-1)

    x_conf = _softmax_max(simcc_x[0])  # [17]
    y_conf = _softmax_max(simcc_y[0])  # [17]
    conf = np.minimum(x_conf, y_conf)   # [17]

    # Map from model coords → image coords via bounding box
    x1, y1, x2, y2 = box
    box_w = x2 - x1
    box_h = y2 - y1

    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[:, 0] = x_locs * box_w / MODEL_W + x1
    kpts[:, 1] = y_locs * box_h / MODEL_H + y1
    kpts[:, 2] = conf

    return kpts


def process_frame_meta(meta, diag=False):
    """Extract detection boxes and decode SimCC tensors into keypoints.

    Returns list of (keypoints_17x3, box_4, person_score) tuples.
    """
    results = []

    # Find detection meta (Stage 1) and tensor meta (Stage 2)
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
        return results

    boxes = det_meta.xyxy() if hasattr(det_meta, 'xyxy') else det_meta.boxes
    n_dets = len(boxes)

    for i, tensor in enumerate(tensor_meta.tensors):
        if i >= n_dets:
            break
        box = boxes[i]
        score = float(det_meta.scores[i]) if hasattr(det_meta, 'scores') else 1.0

        if diag:
            print(f"  tensor[{i}] shape={tensor.shape} dtype={tensor.dtype} "
                  f"min={tensor.min():.4f} max={tensor.max():.4f}")

        # Reshape backbone output: could be [1,17,8,6] or flattened
        if tensor.ndim == 4 and tensor.shape[1:] == (17, 8, 6):
            features = tensor
        elif tensor.ndim == 3 and tensor.shape == (17, 8, 6):
            features = tensor[np.newaxis]
        elif tensor.ndim in (1, 2):
            total = tensor.size
            if total == 17 * 8 * 6 or total % (17 * 8 * 6) == 0:
                features = tensor.reshape(1, 17, 8, 6)
            else:
                LOG.warning(f"Cannot reshape tensor of {total} elems to [1,17,8,6]")
                continue
        else:
            LOG.warning(f"Unexpected tensor shape: {tensor.shape}")
            continue

        kpts = decode_simcc(features, box)
        results.append((kpts, box, score))

    return results


def inference_loop(stream, window):
    """Main inference loop with optional display."""
    frame_num = 0
    last_time = time.time()
    diag_frames = 5
    diag_count = 0
    simcc_times = []

    for frame_result in stream:
        now = time.time()
        dt = now - last_time
        fps = 1.0 / dt if dt > 0 else 0
        last_time = now

        meta = frame_result.meta
        show_diag = diag_count < diag_frames

        t0 = time.perf_counter()
        persons = process_frame_meta(meta, diag=show_diag)
        simcc_ms = (time.perf_counter() - t0) * 1000
        if persons:
            simcc_times.append(simcc_ms)

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
                    f"person {i} ({score:.2f}) simcc={simcc_ms:.1f}ms | "
                    f"L wrist: vis={lw[2]:.3f} ({lw[0]:.0f},{lw[1]:.0f}) | "
                    f"R wrist: vis={rw[2]:.3f} ({rw[0]:.0f},{rw[1]:.0f})"
                )

        # Replace TensorMeta with CocoBodyKeypointsMeta for renderer
        if meta is not None:
            tensor_key = None
            for key, val in meta.items():
                if isinstance(val, TensorMeta):
                    tensor_key = key
                    break
            if tensor_key is not None:
                meta.delete_instance(tensor_key)
                if persons:
                    all_kpts = np.stack([p[0] for p in persons], axis=0)
                    all_boxes = np.stack(
                        [np.array(p[1], dtype=np.float32) for p in persons], axis=0
                    )
                    all_scores = np.array([p[2] for p in persons], dtype=np.float32)
                    pose_meta = CocoBodyKeypointsMeta(
                        keypoints=all_kpts,
                        boxes=all_boxes,
                        scores=all_scores,
                    )
                    meta.add_instance(tensor_key, pose_meta)

        if window is not None:
            window.show(frame_result.image, frame_result.meta, frame_result.stream_id)
            if window.is_closed:
                break

        frame_num += 1

        # Print summary stats periodically
        if frame_num % 100 == 0 and simcc_times:
            avg = np.mean(simcc_times[-100:])
            p95 = np.percentile(simcc_times[-100:], 95)
            print(f"  --- SimCC decode: avg={avg:.2f}ms, p95={p95:.2f}ms ---")


def main():
    network_yaml_info = yaml_parser.get_network_yaml_info()
    parser = config.create_inference_argparser(
        network_yaml_info, description='Cascade RTMPose SimCC pose estimation'
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
                wnd = app.create_window('RTMPose', size=getattr(args, 'window_size', None))
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
