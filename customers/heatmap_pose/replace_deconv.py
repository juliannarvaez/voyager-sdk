#!/usr/bin/env python3
"""
Replace ConvTranspose2d nodes with Resize(bilinear,2x) + Conv2d for AIPU compatibility.

The Axelera Metis AIPU can tile Conv2d across input channels but ConvTranspose2d
with large channel counts fails to map to the MVM unit. The backbone already has
Conv2d with up to 960 input channels that compile fine via tiling.

For each ConvTranspose(C_in, C_out, k=4, s=2, p=1) this creates:
  - Resize(bilinear, scale=2x, half_pixel) — confirmed AIPU-supported
  - Conv(C_in, C_out, k=3, s=1, p=1) — AIPU-supported with tiling

Weight transfer: transpose channels + flip spatial + CENTERED 3x3 from 4x4.
(Average of two possible crops to eliminate systematic spatial offset.)
This is APPROXIMATE — for exact accuracy, retraining needed.
"""
import onnx
import numpy as np
from onnx import helper, TensorProto, numpy_helper

MODEL_IN = "customers/heatmap_pose/mobilenetv2_pose_256x192.onnx"
MODEL_OUT = "customers/heatmap_pose/mobilenetv2_pose_no_deconv.onnx"

model = onnx.load(MODEL_IN)
graph = model.graph

# Build lookup
init_map = {init.name: init for init in graph.initializer}
def get_np(name):
    return numpy_helper.to_array(init_map[name])

# Find ConvTranspose nodes
deconv_info = []
for i, node in enumerate(graph.node):
    if node.op_type == "ConvTranspose":
        w = get_np(node.input[1])
        print(f"ConvTranspose [{i}]: {node.name}")
        print(f"  weight: {node.input[1]} shape={list(w.shape)}")
        deconv_info.append((i, node, w))

print(f"\nReplacing {len(deconv_info)} ConvTranspose nodes...\n")

nodes = list(graph.node)
new_initializers = []

# Process in reverse order so indices stay valid
for idx, node, deconv_w in reversed(deconv_info):
    name_prefix = node.name.replace("/", "_").strip("_")
    C_in, C_out, kH, kW = deconv_w.shape

    print(f"  [{idx}] ConvTranspose({C_in}->{C_out}, k={kH}, s=2)")

    # --- Resize node (nearest, 2x) ---
    roi_name = f"{name_prefix}_roi"
    scales_name = f"{name_prefix}_scales"
    resize_out = f"{name_prefix}_resize_out"

    new_initializers.append(numpy_helper.from_array(
        np.array([], dtype=np.float32), name=roi_name))
    new_initializers.append(numpy_helper.from_array(
        np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32), name=scales_name))

    resize_node = helper.make_node(
        'Resize',
        inputs=[node.input[0], roi_name, scales_name],
        outputs=[resize_out],
        name=f"{name_prefix}_resize",
        mode='linear',
        coordinate_transformation_mode='half_pixel',
    )

    # --- Conv node with adapted weights ---
    # ConvTranspose weight: (C_in, C_out, 4, 4)
    # Conv weight:          (C_out, C_in, kH, kW)
    # Transpose channels, flip spatial (180° rotate), centered 3x3 crop
    conv_w = deconv_w.transpose(1, 0, 2, 3)   # (C_out, C_in, 4, 4)
    conv_w = conv_w[:, :, ::-1, ::-1]          # flip spatial
    # Average two possible 3x3 crops for centered kernel (eliminates offset)
    crop_a = conv_w[:, :, 0:3, 0:3]  # top-left
    crop_b = conv_w[:, :, 1:4, 1:4]  # bottom-right
    conv_w = ((crop_a + crop_b) / 2.0).copy()  # centered average

    conv_w_name = f"{name_prefix}_conv_w"
    new_initializers.append(numpy_helper.from_array(
        conv_w.astype(np.float32), name=conv_w_name))

    conv_inputs = [resize_out, conv_w_name]
    if len(node.input) > 2 and node.input[2]:
        conv_inputs.append(node.input[2])

    conv_node = helper.make_node(
        'Conv',
        inputs=conv_inputs,
        outputs=list(node.output),
        name=f"{name_prefix}_conv",
        kernel_shape=[3, 3],
        strides=[1, 1],
        pads=[1, 1, 1, 1],
        dilations=[1, 1],
        group=1,
    )

    print(f"    -> Resize(nearest,2x) + Conv({C_in}->{C_out}, k=3, p=1)")
    print(f"       Conv weight: {list(conv_w.shape)}")

    nodes[idx:idx+1] = [resize_node, conv_node]

# Rebuild model
all_inits = list(graph.initializer) + new_initializers
new_graph = helper.make_graph(
    nodes, graph.name, list(graph.input), list(graph.output), all_inits)
new_graph.value_info.extend(graph.value_info)

new_model = helper.make_model(new_graph, opset_imports=model.opset_import)
new_model.ir_version = model.ir_version

try:
    onnx.checker.check_model(new_model)
    print("\n[OK] ONNX validation passed")
except Exception as e:
    print(f"\n[WARN] ONNX validation: {e}")

onnx.save(new_model, MODEL_OUT)
import os
print(f"Saved: {MODEL_OUT}")
print(f"Size: {os.path.getsize(MODEL_OUT) / 1024 / 1024:.1f} MB")

# --- Quick inference comparison (optional, needs onnxruntime) ---
try:
    import onnxruntime as ort

    np.random.seed(42)
    inp = np.random.randn(1, 3, 256, 192).astype(np.float32)

    sess_orig = ort.InferenceSession(MODEL_IN, providers=['CPUExecutionProvider'])
    out_orig = sess_orig.run(None, {sess_orig.get_inputs()[0].name: inp})[0]

    sess_new = ort.InferenceSession(MODEL_OUT, providers=['CPUExecutionProvider'])
    out_new = sess_new.run(None, {sess_new.get_inputs()[0].name: inp})[0]

    print(f"\n--- Inference comparison (256x192, random input) ---")
    print(f"Original output: shape={out_orig.shape}, range=[{out_orig.min():.3f}, {out_orig.max():.3f}]")
    print(f"Modified output: shape={out_new.shape}, range=[{out_new.min():.3f}, {out_new.max():.3f}]")

    diff = np.abs(out_new - out_orig)
    orig_range = out_orig.max() - out_orig.min()
    print(f"Diff: mean={diff.mean():.4f}, max={diff.max():.4f}")
    if orig_range > 0:
        print(f"Relative diff: {diff.mean() / orig_range * 100:.1f}% of output range")

    assert out_new.shape == out_orig.shape, f"Shape mismatch: {out_new.shape} vs {out_orig.shape}"
    print(f"\n[OK] Output shapes match: {out_new.shape}")
except ImportError:
    print("\n[SKIP] onnxruntime not installed, skipping inference comparison")
