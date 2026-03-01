# Copyright Axelera AI, 2025
# Decoder for RTMPose SimCC-based top-down pose estimation.
# The AIPU runs the backbone and outputs a compact feature tensor [1,17,8,6].
# This decoder runs the SimCC head (GAU + linear) on CPU via ONNX Runtime
# to produce keypoint coordinates with sub-pixel precision.
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

# SimCC split ratio: coordinates are encoded at 2× model resolution
SIMCC_SPLIT_RATIO = 2.0


def _softmax(x, axis=-1):
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def _decode_simcc(simcc_x, simcc_y, simcc_split_ratio=SIMCC_SPLIT_RATIO):
    """Decode SimCC coordinate representations to keypoints.

    Args:
        simcc_x: [B, K, Wx] logits for x coordinates (Wx = model_w * split_ratio)
        simcc_y: [B, K, Wy] logits for y coordinates (Wy = model_h * split_ratio)
        simcc_split_ratio: coordinate encoding ratio (default 2.0)

    Returns:
        keypoints: [B, K, 3] — (x, y, confidence) in model input pixel coords
    """
    B, K, Wx = simcc_x.shape
    Wy = simcc_y.shape[2]

    # Get peak positions
    x_locs = np.argmax(simcc_x, axis=-1).astype(np.float32)  # [B, K]
    y_locs = np.argmax(simcc_y, axis=-1).astype(np.float32)  # [B, K]

    # Convert to model pixel coords
    x_coords = x_locs / simcc_split_ratio
    y_coords = y_locs / simcc_split_ratio

    # Confidence: max of softmax probabilities
    x_conf = np.max(_softmax(simcc_x, axis=-1), axis=-1)  # [B, K]
    y_conf = np.max(_softmax(simcc_y, axis=-1), axis=-1)  # [B, K]
    conf = np.minimum(x_conf, y_conf)

    # Stack into [B, K, 3]
    keypoints = np.stack([x_coords, y_coords, conf], axis=-1)
    return keypoints


class DecodeRTMPose(AxOperator):
    """Decode RTMPose backbone output via SimCC head on CPU.

    The AIPU outputs a compact feature tensor [1, 17, 8, 6] (backbone).
    This decoder runs the SimCC head ONNX (41 nodes, ~0.65ms on ARM64)
    to get [1, 17, 384] x-logits and [1, 17, 512] y-logits,
    then decodes to keypoint coordinates via argmax.

    Parameters:
        conf_threshold: Minimum keypoint confidence to consider valid.
        simcc_split_ratio: Coordinate encoding ratio (default 2.0 = 2× model res).
        simcc_head_onnx: Path to the SimCC head ONNX file (auto-resolved if empty).
    """

    conf_threshold: float = 0.3
    simcc_split_ratio: float = SIMCC_SPLIT_RATIO
    simcc_head_onnx: str = ''

    def _post_init(self):
        super()._post_init()
        self._head_session = None

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
        self.model_width = model_info.input_width
        self.model_height = model_info.input_height
        self.association = context.association or None

        # Dequant params for when running on AIPU (INT8 output)
        self._deq_scales = None
        self._deq_zeropoints = None
        if model_info.manifest and model_info.manifest.is_compiled():
            self._deq_scales, self._deq_zeropoints = zip(
                *model_info.manifest.dequantize_params
            )

    def _get_head_session(self):
        """Lazy-load the SimCC head ONNX Runtime session."""
        if self._head_session is not None:
            return self._head_session

        # Resolve SimCC head ONNX path
        if self.simcc_head_onnx:
            head_path = self.simcc_head_onnx
        else:
            # Look next to the backbone ONNX or in customers/heatmap_pose/
            candidates = [
                Path(self._compiled_model_dir or '.') / 'rtmpose-m-simcc-head.onnx',
                Path(__file__).parent / 'rtmpose-m-simcc-head.onnx',
            ]
            head_path = None
            for c in candidates:
                if c.exists():
                    head_path = str(c)
                    break
            if head_path is None:
                raise RuntimeError(
                    f'SimCC head ONNX not found. Searched: {candidates}. '
                    f'Set simcc_head_onnx parameter to the path.'
                )

        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 2
        self._head_session = ort.InferenceSession(
            head_path, opts, providers=['CPUExecutionProvider']
        )
        LOG.info(
            f'Loaded SimCC head: {head_path}, '
            f'input={self._head_session.get_inputs()[0].name}'
        )
        return self._head_session

    def _run_simcc_head(self, features):
        """Run SimCC head on backbone features.

        Args:
            features: numpy array [1, 17, 8, 6] or [17, 8, 6]

        Returns:
            simcc_x [1, 17, Wx], simcc_y [1, 17, Wy]
        """
        sess = self._get_head_session()
        if features.ndim == 3:
            features = features[np.newaxis]
        input_name = sess.get_inputs()[0].name
        simcc_x, simcc_y = sess.run(
            None, {input_name: features.astype(np.float32)}
        )
        return simcc_x, simcc_y

    def _decode_outputs(self, predict):
        """Decode model outputs to keypoints.

        Handles both:
        - Direct SimCC outputs: [simcc_x, simcc_y] from end2end model
        - Backbone features: [1,17,8,6] from split AIPU model -> run SimCC head
        """
        # Handle various input formats
        if isinstance(predict, (list, tuple)):
            if len(predict) == 2:
                # Two outputs: simcc_x, simcc_y (end2end model)
                simcc_x, simcc_y = predict
                if hasattr(simcc_x, 'cpu'):
                    simcc_x = simcc_x.cpu().detach().numpy()
                if hasattr(simcc_y, 'cpu'):
                    simcc_y = simcc_y.cpu().detach().numpy()
            else:
                features = predict[0]
                if hasattr(features, 'cpu'):
                    features = features.cpu().detach().numpy()
                if not isinstance(features, np.ndarray):
                    features = np.array(features)
                simcc_x, simcc_y = self._run_simcc_head(features)
        else:
            # Single tensor: backbone features
            if hasattr(predict, 'cpu'):
                predict = predict.cpu().detach().numpy()
            if not isinstance(predict, np.ndarray):
                predict = np.array(predict)

            # Check if this is backbone output [1,17,8,6] or SimCC output
            if predict.ndim == 4 and predict.shape[-2:] == (8, 6):
                # Backbone features -> run SimCC head
                simcc_x, simcc_y = self._run_simcc_head(predict)
            elif predict.ndim == 3 and predict.shape[-1] in (384, 512):
                # Already SimCC output
                raise ValueError(
                    f'Single SimCC tensor not supported; need both simcc_x and simcc_y. '
                    f'Got shape {predict.shape}'
                )
            else:
                # Try as backbone features
                simcc_x, simcc_y = self._run_simcc_head(predict)

        # Ensure batch dimension
        if simcc_x.ndim == 2:
            simcc_x = simcc_x[np.newaxis]
        if simcc_y.ndim == 2:
            simcc_y = simcc_y[np.newaxis]

        return simcc_x.astype(np.float32), simcc_y.astype(np.float32)

    def _map_to_image_coords(self, keypoints, image, meta):
        """Map keypoints from model coords to image coords.

        In cascade mode: scale to ROI box, then offset by box position.
        In standalone mode: use BBoxState.rescale() for letterbox correction.
        """
        kpts = keypoints.copy()  # [K, 3]

        if self._where:
            # Cascade mode: scale to parent detection box
            master_meta = meta[self._where]
            box_idx = master_meta.get_next_secondary_frame_index(self.task_name)
            base_box = master_meta.boxes[box_idx]
            roi_w = float(base_box[2] - base_box[0])
            roi_h = float(base_box[3] - base_box[1])

            kpts[:, 0] = kpts[:, 0] * roi_w / self.model_width + base_box[0]
            kpts[:, 1] = kpts[:, 1] * roi_h / self.model_height + base_box[1]
        else:
            # Standalone mode: use image dimensions
            if hasattr(image, 'size'):
                src_w, src_h = image.size[0], image.size[1]
            elif hasattr(image, 'shape'):
                if image.ndim == 3 and image.shape[0] in (1, 3, 4):
                    src_h, src_w = image.shape[1], image.shape[2]
                else:
                    src_h, src_w = image.shape[:2]
            else:
                src_h, src_w = self.model_height, self.model_width

            kpts[:, 0] *= src_w / self.model_width
            kpts[:, 1] *= src_h / self.model_height

        return kpts

    def _compute_bbox_and_score(self, keypoints, image, meta):
        """Compute bounding box from keypoints and overall score."""
        if self._where:
            master_meta = meta[self._where]
            idx = master_meta.get_next_secondary_frame_index(self.task_name)
            bbox = master_meta.boxes[idx].copy()
        else:
            # Derive bbox from valid keypoints
            valid = keypoints[:, 2] > self.conf_threshold
            if valid.any():
                valid_kpts = keypoints[valid]
                bbox = np.array([
                    valid_kpts[:, 0].min(),
                    valid_kpts[:, 1].min(),
                    valid_kpts[:, 0].max(),
                    valid_kpts[:, 1].max(),
                ], dtype=np.float32)
            else:
                bbox = np.zeros(4, dtype=np.float32)

        # Score: mean confidence of valid keypoints
        valid_conf = keypoints[:, 2][keypoints[:, 2] > self.conf_threshold]
        score = float(valid_conf.mean()) if len(valid_conf) > 0 else 0.0

        return bbox, score

    def exec_torch(self, image, predict, meta):
        """Decode RTMPose output in the torch/torch-aipu pipeline."""
        # 1. Run SimCC head on backbone features
        simcc_x, simcc_y = self._decode_outputs(predict)

        # 2. Decode to keypoints [B, K, 3]
        keypoints = _decode_simcc(simcc_x, simcc_y, self.simcc_split_ratio)
        kpts = keypoints[0]  # [17, 3]

        # 3. Map to image coordinates
        kpts = self._map_to_image_coords(kpts, image, meta)

        # 4. Compute bbox and score
        bbox, score = self._compute_bbox_and_score(kpts, image, meta)

        # 5. Pack as CocoBodyKeypointsMeta
        model_meta = CocoBodyKeypointsMeta(
            keypoints=kpts.reshape(1, 17, 3),
            boxes=np.array([bbox], dtype=np.float32),
            scores=np.array([score], dtype=np.float32),
        )
        meta.add_instance(self.task_name, model_meta, self._where)
        return image, predict, meta

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        """GStreamer pipeline: output raw tensor, decode SimCC in Python."""
        gst.decode_muxer(
            name=f'decoder_task{self._taskn}{stream_idx}',
            lib='libdecode_to_raw_tensor.so',
            options=f'meta_key:{str(self.task_name)};',
        )
