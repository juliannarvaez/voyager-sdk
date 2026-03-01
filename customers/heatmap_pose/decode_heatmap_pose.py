# Copyright Axelera AI, 2025
# Decoder for heatmap-based top-down pose estimation models.
# Converts [B, 17, H, W] heatmap tensors to CocoBodyKeypointsMeta.
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from axelera import types
from axelera.app import gst_builder, logging_utils
from axelera.app.meta import CocoBodyKeypointsMeta
from axelera.app.meta.bbox_state import BBoxState
from axelera.app.operators import AxOperator, PipelineContext
from axelera.app.torch_utils import torch

LOG = logging_utils.getLogger(__name__)


def heatmaps_to_keypoints(heatmaps: np.ndarray, model_w: int, model_h: int):
    """Convert heatmap tensor to keypoint coordinates.

    Args:
        heatmaps: shape (K, H_hm, W_hm) — one heatmap per keypoint
        model_w: model input width (e.g. 192)
        model_h: model input height (e.g. 256)

    Returns:
        keypoints: shape (K, 3) — [x, y, visibility] in model input coords
    """
    K, H, W = heatmaps.shape
    keypoints = np.zeros((K, 3), dtype=np.float32)

    for k in range(K):
        hm = heatmaps[k]
        # Find peak location
        idx = np.argmax(hm)
        y_hm, x_hm = np.unravel_index(idx, (H, W))

        # Sub-pixel refinement via neighbor difference (Taylor expansion)
        # Clamp to avoid boundary issues
        if 0 < x_hm < W - 1:
            dx = (float(hm[y_hm, x_hm + 1]) - float(hm[y_hm, x_hm - 1])) * 0.25
        else:
            dx = 0.0
        if 0 < y_hm < H - 1:
            dy = (float(hm[y_hm + 1, x_hm]) - float(hm[y_hm - 1, x_hm])) * 0.25
        else:
            dy = 0.0

        # Convert heatmap coords to model input coords
        x = (x_hm + dx + 0.5) * model_w / W
        y = (y_hm + dy + 0.5) * model_h / H

        # Use peak value as visibility/confidence score
        vis = float(hm[y_hm, x_hm])

        keypoints[k] = [x, y, vis]

    return keypoints


class DecodeHeatmapPose(AxOperator):
    """Decode heatmap-based pose model output to CocoBodyKeypointsMeta.

    Expects model output tensor of shape [1, 17, H_hm, W_hm].
    Performs argmax + sub-pixel refinement on CPU to get keypoint coordinates.
    """

    # Visibility threshold: below this, keypoint is considered not detected
    visibility_threshold: float = 0.0

    def _post_init(self):
        super()._post_init()

    def configure_model_and_context_info(
        self,
        model_info: types.ModelInfo,
        context: PipelineContext,
        task_name: str,
        taskn: int,
        compiled_model_dir: Path | None,
        task_graph,
    ):
        super().configure_model_and_context_info(
            model_info, context, task_name, taskn, compiled_model_dir, task_graph
        )
        self.meta_type_name = "CocoBodyKeypointsMeta"
        self.scaled = context.resize_status
        self.model_width = model_info.input_width
        self.model_height = model_info.input_height
        self._association = context.association or None

        if model_info.manifest and model_info.manifest.is_compiled():
            self._deq_scales, self._deq_zeropoints = zip(*model_info.manifest.dequantize_params)
            self._n_padded_ch_outputs = model_info.manifest.n_padded_ch_outputs

    def exec_torch(self, image, predict, meta):
        """Decode heatmaps in the torch/torch-aipu pipeline."""
        if isinstance(predict, torch.Tensor):
            predict = predict.cpu().detach().numpy()

        # predict shape: [1, 17, H_hm, W_hm]
        if predict.ndim == 4:
            heatmaps = predict[0]  # [17, H, W]
        elif predict.ndim == 3:
            heatmaps = predict  # [17, H, W]
        else:
            LOG.warning(f"Unexpected heatmap shape: {predict.shape}")
            model_meta = CocoBodyKeypointsMeta(
                keypoints=np.array([]),
                boxes=np.array([]),
                scores=np.array([]),
            )
            meta.add_instance(self.task_name, model_meta, self._where)
            return image, predict, meta

        # Decode keypoints from heatmaps
        kpts = heatmaps_to_keypoints(heatmaps, self.model_width, self.model_height)

        # If in a cascade, get the parent bounding box
        if self._where:
            master_meta = meta[self._where]
            box_idx = master_meta.get_next_secondary_frame_index(self.task_name)
            base_box = master_meta.boxes[box_idx]
            box_w = base_box[2] - base_box[0]
            box_h = base_box[3] - base_box[1]

            # Scale keypoints from model coords to ROI coords, then to image coords
            kpts[:, 0] = kpts[:, 0] * box_w / self.model_width + base_box[0]
            kpts[:, 1] = kpts[:, 1] * box_h / self.model_height + base_box[1]

            box = base_box
            score = np.max(kpts[:, 2])  # max keypoint visibility as person score
        else:
            # Standalone (no cascade): use BBoxState to reverse resize transform
            if hasattr(image, 'size'):
                img_w, img_h = image.size[0], image.size[1]
            elif hasattr(image, 'shape'):
                if image.ndim == 3 and image.shape[0] in (1, 3, 4):
                    # CHW tensor — this is the preprocessed input, not original
                    img_h, img_w = image.shape[1], image.shape[2]
                else:
                    img_h, img_w = image.shape[:2]
            else:
                img_h, img_w = self.model_height, self.model_width

            # kpts are in model coords — create a full-model box
            box = np.array([[0, 0, self.model_width, self.model_height]], dtype=np.float32)
            kpts_2d = kpts.reshape(1, 17, 3)

            # Use BBoxState.rescale to properly undo letterbox/stretch/etc.
            if self.scaled != types.ResizeMode.ORIGINAL:
                box, kpts_2d, _ = BBoxState.rescale(
                    box, types.BoxFormat.XYXY,
                    (self.model_height, self.model_width),   # ori_shape (model)
                    (img_h, img_w),                          # target_shape (image)
                    resize_mode=self.scaled,
                    kpts=kpts_2d,
                )
            else:
                # No resize info — fall back to proportional scaling
                kpts_2d[:, :, 0] *= img_w / self.model_width
                kpts_2d[:, :, 1] *= img_h / self.model_height
                box = np.array([[0, 0, img_w, img_h]], dtype=np.float32)

            kpts = kpts_2d[0]
            box = box[0]
            score = np.max(kpts[:, 2])

        # Pack as CocoBodyKeypointsMeta (N=1 person per crop)
        model_meta = CocoBodyKeypointsMeta(
            keypoints=kpts.reshape(1, 17, 3),               # [1, 17, 3]
            boxes=np.array([box], dtype=np.float32),         # [1, 4]
            scores=np.array([score], dtype=np.float32),      # [1]
        )
        meta.add_instance(self.task_name, model_meta, self._where)
        return image, predict, meta

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        """GStreamer pipeline: output raw tensors via libdecode_to_raw_tensor.so.

        The heatmap argmax is done in Python application code (see run_heatmap_pose.py).
        NOTE: We intentionally omit master_meta here because libdecode_to_raw_tensor.so
        has a bug where it fails dynamic_cast in cascade mode. Instead, the raw tensor
        goes into the top-level meta map, and we correlate with detections in Python.
        """
        gst.decode_muxer(
            name=f'decoder_task{self._taskn}{stream_idx}',
            lib='libdecode_to_raw_tensor.so',
            options=f'meta_key:{str(self.task_name)};',
        )
