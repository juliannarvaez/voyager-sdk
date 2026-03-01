#!/usr/bin/env python
# Copyright Axelera AI, 2025
"""
Run standalone heatmap pose on AIPU via GStreamer pipeline.

The AIPU model outputs raw heatmap tensors via libdecode_to_raw_tensor.so.
This script decodes heatmaps → CocoBodyKeypointsMeta in Python so the
renderer draws keypoints/skeleton correctly.

Usage:
    python run_standalone_heatmap.py mobilenetv2-pose-svd usb:10/yuyv
    python run_standalone_heatmap.py mobilenetv2-pose-svd usb:10/yuyv --no-display
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

LOG = logging_utils.getLogger(__name__)


def decode_heatmap_tensor(tensor, img_w, img_h):
    """Decode raw heatmap tensor to keypoints in image coordinates.

    Resolution-agnostic: works with any (17, H, W) heatmap output.
    Coordinate mapping is heatmap_coord / heatmap_size * image_size (STRETCH).

    Args:
        tensor: np.ndarray, raw model output — [1,17,H,W] or [17,H,W]
        img_w, img_h: original image dimensions

    Returns:
        keypoints (17,3), box (4,), score float
    """
    # Handle shape variants
    if tensor.ndim == 4:
        heatmaps = tensor[0]
    elif tensor.ndim == 3:
        heatmaps = tensor
    elif tensor.ndim in (1, 2):
        total = tensor.size
        # Try to infer 17-keypoint heatmap shape from total element count
        if total % 17 == 0:
            spatial = total // 17
            # Common heatmap sizes: 64x48, 96x72, 128x96, etc. (4:3 ratio)
            h_cand = int(np.sqrt(spatial * 4 / 3))
            w_cand = spatial // h_cand if h_cand > 0 else 0
            if h_cand * w_cand == spatial:
                heatmaps = tensor.reshape(17, h_cand, w_cand)
            else:
                LOG.warning(f"Cannot infer heatmap dims from {total} elements")
                return None, None, None
        else:
            LOG.warning(f"Cannot reshape tensor of size {total} (not divisible by 17)")
            return None, None, None
    else:
        LOG.warning(f"Unexpected tensor shape: {tensor.shape}")
        return None, None, None

    heatmaps = heatmaps.astype(np.float32)

    # Normalize: if values exceed [0,1], apply sigmoid
    hm_max = heatmaps.max()
    if hm_max > 1.0 or hm_max <= 0.0:
        heatmaps = 1.0 / (1.0 + np.exp(-heatmaps))

    K, H, W = heatmaps.shape

    # Vectorized argmax
    flat = heatmaps.reshape(K, -1)
    idx = flat.argmax(axis=1)
    y_hm = idx // W
    x_hm = idx % W
    conf = flat[np.arange(K), idx]

    # Sub-pixel refinement
    dx = np.zeros(K, dtype=np.float32)
    dy = np.zeros(K, dtype=np.float32)
    ki = np.arange(K)
    mx = (x_hm > 0) & (x_hm < W - 1)
    my = (y_hm > 0) & (y_hm < H - 1)
    if mx.any():
        dx[mx] = (heatmaps[ki[mx], y_hm[mx], x_hm[mx]+1] -
                   heatmaps[ki[mx], y_hm[mx], x_hm[mx]-1]) * 0.25
    if my.any():
        dy[my] = (heatmaps[ki[my], y_hm[my]+1, x_hm[my]] -
                   heatmaps[ki[my], y_hm[my]-1, x_hm[my]]) * 0.25

    # Model coords → image coords (STRETCH: simple ratio)
    kx = (x_hm.astype(np.float32) + dx + 0.5) / W * img_w
    ky = (y_hm.astype(np.float32) + dy + 0.5) / H * img_h

    keypoints = np.stack([kx, ky, conf], axis=1)  # (17, 3)
    box = np.array([0, 0, img_w, img_h], dtype=np.float32)
    score = float(conf.max())
    return keypoints, box, score


def inference_loop(stream, window):
    """Main inference loop — decode raw tensors into keypoints for renderer."""
    frame_num = 0
    last_time = time.time()

    for frame_result in stream:
        now = time.time()
        dt = now - last_time
        fps = 1.0 / dt if dt > 0 else 0
        last_time = now

        image = frame_result.image
        meta = frame_result.meta

        # Get image dimensions
        if hasattr(image, 'size'):
            img_w, img_h = image.size[0], image.size[1]
        elif hasattr(image, 'width'):
            img_w, img_h = image.width, image.height
        elif hasattr(image, 'asarray'):
            arr = image.asarray()
            img_h, img_w = arr.shape[:2]
        else:
            img_w, img_h = 896, 504  # fallback

        # Find TensorMeta and decode it
        tensor_key = None
        tensor_meta = None
        for key, val in meta.items():
            if isinstance(val, TensorMeta):
                tensor_key = key
                tensor_meta = val
                break

        if tensor_meta is not None and tensor_meta.tensors:
            tensor = tensor_meta.tensors[0]
            kpts, box, score = decode_heatmap_tensor(tensor, img_w, img_h)

            # Remove TensorMeta (can't render) and replace with keypoints
            meta.delete_instance(tensor_key)

            if kpts is not None:
                pose_meta = CocoBodyKeypointsMeta(
                    keypoints=kpts.reshape(1, 17, 3),
                    boxes=np.array([box], dtype=np.float32),
                    scores=np.array([score], dtype=np.float32),
                )
                meta.add_instance(tensor_key, pose_meta)

                lw = kpts[9]   # left wrist
                rw = kpts[10]  # right wrist
                print(
                    f"[{frame_num:5d}] {fps:5.1f} fps | "
                    f"L wrist ({lw[0]:.0f},{lw[1]:.0f}) conf={lw[2]:.2f} | "
                    f"R wrist ({rw[0]:.0f},{rw[1]:.0f}) conf={rw[2]:.2f}"
                )
            else:
                print(f"[{frame_num:5d}] {fps:5.1f} fps | decode failed")
        else:
            print(f"[{frame_num:5d}] {fps:5.1f} fps | no tensor")

        if window is not None:
            window.show(frame_result.image, frame_result.meta, frame_result.stream_id)
            if window.is_closed:
                break

        frame_num += 1


def main():
    network_yaml_info = yaml_parser.get_network_yaml_info()
    parser = config.create_inference_argparser(
        network_yaml_info, description='Standalone heatmap pose (GStreamer + AIPU)'
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


if __name__ == "__main__":
    main()
