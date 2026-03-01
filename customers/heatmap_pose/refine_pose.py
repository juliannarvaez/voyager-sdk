#!/usr/bin/env python
# Copyright Axelera AI, 2025
"""
Hybrid AIPU+CPU pose estimation:
  - YOLOv8L-pose on AIPU → person detection + initial keypoints
  - MobileNetV2 heatmap (64×64 input) on CPU → refine keypoints

The AIPU provides fast detection and good initial keypoints. The tiny
64×64 heatmap model refines keypoint positions using the cropped person
ROI, running on CPU in ~15ms.

Usage (inside Docker container):
    source venv/bin/activate
    python customers/heatmap_pose/refine_pose.py yolov8lpose-896x512-coco usb:0
    python customers/heatmap_pose/refine_pose.py yolov8lpose-896x512-coco usb:0 --no-display
    python customers/heatmap_pose/refine_pose.py yolov8lpose-896x512-coco image.jpg --no-display
"""
import os
import sys
import time

if not os.environ.get('AXELERA_FRAMEWORK'):
    sys.exit("Please activate the Axelera environment: source venv/bin/activate")

import numpy as np
import onnxruntime as ort

from axelera.app import config, create_inference_stream, display, logging_utils, yaml_parser
from axelera.app.meta.keypoint import CocoBodyKeypointsMeta

LOG = logging_utils.getLogger(__name__)

# COCO keypoint names (17 total)
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Heatmap refinement model config
HEATMAP_INPUT_SIZE = 64  # 64×64 input to heatmap model
HEATMAP_ONNX = os.path.join(
    os.environ.get('AXELERA_FRAMEWORK', '/voyager-sdk'),
    'customers/heatmap_pose/mobilenetv2_pose_dynamic.onnx'
)

# ImageNet normalization (matches heatmap model training)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

# Blending: how much to trust heatmap vs YOLO keypoints
# 0.0 = pure YOLO, 1.0 = pure heatmap
HEATMAP_BLEND_WEIGHT = 0.6
# Only use heatmap keypoint if its confidence exceeds this
HEATMAP_MIN_CONFIDENCE = 0.1

# Only refine these keypoints (wrists)
REFINE_INDICES = [9, 10]  # left_wrist, right_wrist


def create_heatmap_session():
    """Initialize ONNX Runtime session for heatmap model."""
    if not os.path.exists(HEATMAP_ONNX):
        LOG.error(f"Heatmap ONNX not found: {HEATMAP_ONNX}")
        return None
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 2
    opts.intra_op_num_threads = 2
    sess = ort.InferenceSession(HEATMAP_ONNX, opts, providers=['CPUExecutionProvider'])
    LOG.info(f"Loaded heatmap model: {HEATMAP_ONNX}")
    # Warm up
    dummy = np.zeros((1, 3, HEATMAP_INPUT_SIZE, HEATMAP_INPUT_SIZE), dtype=np.float32)
    sess.run(None, {sess.get_inputs()[0].name: dummy})
    return sess


import cv2

# Pre-compute normalization constants for blobFromImage
# blobFromImage does: (pixel * scalefactor - mean) / std  ... but actually:
# blobFromImage does: (pixel - mean) * scalefactor
# So we need to handle std separately, OR do a single fused scale+offset.
# Formula: output = (pixel/255 - mean) / std = pixel * (1/(255*std)) - mean/std
_SCALE = 1.0 / 255.0
# blobFromImage swapRB handles BGR→RGB
# blobFromImage mean parameter subtracts AFTER scaling
# So: blob = (pixel * scale - mean_param)
# We want: (pixel/255 - MEAN) / STD = pixel/(255*STD) - MEAN/STD
_BLOB_SCALE = _SCALE  # we'll divide by STD after
_BLOB_MEAN = (0.485 * 255, 0.456 * 255, 0.406 * 255)  # pre-scaled mean for blobFromImage
_INV_STD = np.array([1.0/0.229, 1.0/0.224, 1.0/0.225], dtype=np.float32).reshape(3, 1, 1)


def crop_and_preprocess(image, box, is_bgr=True):
    """Crop person ROI, resize to 64×64, normalize. Uses cv2 SIMD where possible.

    Args:
        image: numpy array (H, W, 3 or 4) uint8
        box:   [x1, y1, x2, y2] in pixel coordinates
        is_bgr: True if image is BGR(A) from GStreamer

    Returns:
        tensor: (1, 3, 64, 64) float32 normalized, or None
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = image[y1:y2, x1:x2]

    # Drop alpha channel if present (BGRA → BGR)
    if crop.ndim == 3 and crop.shape[2] == 4:
        crop = crop[:, :, :3]

    # blobFromImage: resize + BGR→RGB + scale + transpose in one fused SIMD call
    # Output: (1, 3, 64, 64) float32, values = pixel * (1/255)
    blob = cv2.dnn.blobFromImage(
        crop,
        scalefactor=_SCALE,
        size=(HEATMAP_INPUT_SIZE, HEATMAP_INPUT_SIZE),
        mean=(0, 0, 0),
        swapRB=is_bgr,
        crop=False,
    )
    # blob shape: (1, 3, 64, 64), values in [0, 1]
    # Apply ImageNet normalization: (x - mean) / std
    blob[0] -= MEAN
    blob[0] *= _INV_STD
    return blob


def decode_heatmaps(heatmaps, box, indices=None):
    """Decode only selected heatmap channels to image coordinates.

    Args:
        heatmaps: (17, H_hm, W_hm) float32
        box: [x1, y1, x2, y2] person bounding box in image coords
        indices: list of keypoint indices to decode (default: REFINE_INDICES)

    Returns:
        dict: {keypoint_index: (x, y, confidence)} in image coordinates
    """
    if indices is None:
        indices = REFINE_INDICES
    _, H, W = heatmaps.shape
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    box_w = x2 - x1
    box_h = y2 - y1

    # Only extract the channels we care about
    sel = np.array(indices, dtype=np.intp)
    K = len(sel)
    hm = heatmaps[sel]                  # (K, H, W)
    flat = hm.reshape(K, -1)
    idx = flat.argmax(axis=1)           # (K,)
    y_hm = idx // W
    x_hm = idx % W
    conf = flat[np.arange(K), idx]

    # Sub-pixel refinement
    dx = np.zeros(K, dtype=np.float32)
    dy = np.zeros(K, dtype=np.float32)
    mask_x = (x_hm > 0) & (x_hm < W - 1)
    mask_y = (y_hm > 0) & (y_hm < H - 1)
    ki = np.arange(K)
    if mask_x.any():
        dx[mask_x] = (hm[ki[mask_x], y_hm[mask_x], x_hm[mask_x] + 1].astype(np.float32)
                     - hm[ki[mask_x], y_hm[mask_x], x_hm[mask_x] - 1].astype(np.float32)) * 0.25
    if mask_y.any():
        dy[mask_y] = (hm[ki[mask_y], y_hm[mask_y] + 1, x_hm[mask_y]].astype(np.float32)
                     - hm[ki[mask_y], y_hm[mask_y] - 1, x_hm[mask_y]].astype(np.float32)) * 0.25

    cx = (x_hm.astype(np.float32) + dx + 0.5) / W
    cy = (y_hm.astype(np.float32) + dy + 0.5) / H

    result = {}
    for j in range(K):
        result[indices[j]] = (cx[j] * box_w + x1, cy[j] * box_h + y1, float(conf[j]))
    return result


def refine_keypoints(yolo_kpts, hm_dict, blend=HEATMAP_BLEND_WEIGHT,
                     min_conf=HEATMAP_MIN_CONFIDENCE):
    """Blend YOLO keypoints with heatmap-refined wrists only.

    Args:
        yolo_kpts: (17, 3) from YOLOv8L-pose [x, y, conf]
        hm_dict:   {keypoint_index: (x, y, confidence)} from decode_heatmaps
        blend:     weight for heatmap (0=pure YOLO, 1=pure heatmap)
        min_conf:  minimum heatmap confidence to apply refinement

    Returns:
        refined: (17, 3) blended keypoints
    """
    refined = yolo_kpts.copy()
    for k, (hx, hy, hc) in hm_dict.items():
        yolo_conf = yolo_kpts[k, 2]
        if hc >= min_conf and yolo_conf > 0:
            w = blend
            refined[k, 0] = (1 - w) * yolo_kpts[k, 0] + w * hx
            refined[k, 1] = (1 - w) * yolo_kpts[k, 1] + w * hy
            refined[k, 2] = max(yolo_conf, hc)
    return refined


def get_yolo_keypoints_and_boxes(meta):
    """Extract keypoints and boxes from YOLOv8-pose meta.

    Returns list of (keypoints_17x3, box_4, score) tuples.
    """
    results = []
    for key, val in meta.items():
        if isinstance(val, CocoBodyKeypointsMeta):
            for i in range(len(val.scores)):
                kpts = val.keypoints[i]  # (17, 3)
                box = val.boxes[i]       # (4,) xyxy
                score = float(val.scores[i])
                results.append((kpts, box, score, key))
            return results
    return results


def inference_loop(stream, heatmap_sess, window, blend_weight=HEATMAP_BLEND_WEIGHT):
    """Main inference loop: AIPU detection + CPU heatmap refinement."""
    frame_num = 0
    last_time = time.time()
    input_name = heatmap_sess.get_inputs()[0].name if heatmap_sess else None

    refine_times = []

    for frame_result in stream:
        try:
            now = time.time()
            dt = now - last_time
            fps = 1.0 / dt if dt > 0 else 0
            last_time = now

            image = frame_result.image
            meta = frame_result.meta

            # Get image as numpy array (H, W, 3) uint8
            if isinstance(image, np.ndarray) and image.ndim == 3:
                img_np = image
            elif hasattr(image, 'asarray'):
                # axelera.types.img.Image
                img_np = image.asarray()
            elif hasattr(image, 'convert'):
                # PIL Image
                img_np = np.asarray(image.convert('RGB'))
            elif hasattr(image, 'numpy'):
                img_np = image.numpy()
            else:
                img_np = np.array(image)
            
            if img_np.ndim < 2:
                frame_num += 1
                continue

            persons = get_yolo_keypoints_and_boxes(meta)

            if not persons:
                print(f"[{frame_num:5d}] {fps:5.1f} fps | no detections")
            else:
                for i, (yolo_kpts, box, score, meta_key) in enumerate(persons):
                    if heatmap_sess is not None:
                        t0 = time.perf_counter()

                        # Crop and preprocess person ROI
                        tensor = crop_and_preprocess(img_np, box)

                        if tensor is not None:
                            # Run heatmap model on CPU
                            heatmap_out = heatmap_sess.run(None, {input_name: tensor})
                            heatmaps = heatmap_out[0][0]  # (17, H, W)

                            # Decode heatmap keypoints
                            hm_kpts = decode_heatmaps(heatmaps, box)

                            # Blend with YOLO keypoints
                            refined = refine_keypoints(yolo_kpts, hm_kpts, blend=blend_weight)

                            t_refine = (time.perf_counter() - t0) * 1000
                            refine_times.append(t_refine)

                            # Update meta with refined keypoints
                            meta_obj = meta[meta_key]
                            meta_obj.keypoints[i] = refined

                            # Print wrist positions
                            lw = refined[9]   # left_wrist
                            rw = refined[10]  # right_wrist
                            avg_refine = np.mean(refine_times[-100:])
                            print(
                                f"[{frame_num:5d}] {fps:5.1f} fps | "
                                f"refine {t_refine:.1f}ms (avg {avg_refine:.1f}ms) | "
                                f"L wrist ({lw[0]:.0f},{lw[1]:.0f}) conf={lw[2]:.2f} | "
                                f"R wrist ({rw[0]:.0f},{rw[1]:.0f}) conf={rw[2]:.2f}"
                            )
                        else:
                            print(f"[{frame_num:5d}] {fps:5.1f} fps | crop failed, using YOLO kpts")
                    else:
                        lw = yolo_kpts[9]
                        rw = yolo_kpts[10]
                        print(
                            f"[{frame_num:5d}] {fps:5.1f} fps | YOLO only | "
                            f"L wrist ({lw[0]:.0f},{lw[1]:.0f}) | "
                            f"R wrist ({rw[0]:.0f},{rw[1]:.0f})"
                        )

            # Display
            if window is not None:
                window.show(frame_result.image, frame_result.meta, frame_result.stream_id)
                if window.is_closed:
                    break

            frame_num += 1
        except Exception as e:
            import traceback
            traceback.print_exc()
            frame_num += 1


def main():
    network_yaml_info = yaml_parser.get_network_yaml_info()
    parser = config.create_inference_argparser(
        network_yaml_info, description='Hybrid AIPU+CPU pose refinement'
    )
    parser.add_argument('--no-refine', action='store_true',
                        help='Disable heatmap refinement (YOLO keypoints only)')
    parser.add_argument('--blend', type=float, default=HEATMAP_BLEND_WEIGHT,
                        help=f'Heatmap blend weight (default: {HEATMAP_BLEND_WEIGHT})')
    args = parser.parse_args()

    # Initialize heatmap model on CPU
    heatmap_sess = None
    if not args.no_refine:
        heatmap_sess = create_heatmap_session()
        if heatmap_sess is None:
            LOG.warning("Heatmap model not available, running YOLO-only")

    # Create AIPU inference stream
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
                wnd = app.create_window('Pose Refinement', size=getattr(args, 'window_size', None))
                app.start_thread(inference_loop, (stream, heatmap_sess, wnd, args.blend), name='InferenceThread')
                app.run(interval=1 / 30)
        else:
            inference_loop(stream, heatmap_sess, None, args.blend)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()


if __name__ == "__main__":
    main()
