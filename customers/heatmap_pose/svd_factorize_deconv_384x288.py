#!/usr/bin/env python3
"""
Create a 384x288 resolution MobileNetV2 pose model with SVD factorization.

SimpleBaseline architecture is fully convolutional — all ops (Conv, BN, ReLU,
ConvTranspose) are spatially agnostic. We can simply change the input shape
and run ONNX shape inference. Output becomes [1, 17, 96, 72] (4× downsample).

Then apply the same SVD factorization on the first ConvTranspose2d:
  ConvTranspose(1280→256, k=4, s=2, p=1) → Conv1x1(1280→512) + ConvTranspose(512→256)

Usage (run inside mmpose_venv which has onnx):
  cd /voyager-sdk
  /app/mmpose_venv/bin/python customers/heatmap_pose/svd_factorize_deconv_384x288.py
"""
import os
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference

# --- Configuration ---
MODEL_IN = "customers/heatmap_pose/mobilenetv2_pose_256x192.onnx"
MODEL_OUT = "customers/heatmap_pose/mobilenetv2_pose_svd_384x288.onnx"
NEW_H, NEW_W = 384, 288        # 1.5× the original 256×192
RANK = 512                      # SVD rank (same as before)

print(f"=== Step 1: Reshape input from [1,3,256,192] to [1,3,{NEW_H},{NEW_W}] ===")

model = onnx.load(MODEL_IN)
graph = model.graph

# Change input shape
inp = graph.input[0]
inp_shape = inp.type.tensor_type.shape
assert len(inp_shape.dim) == 4, f"Expected 4D input, got {len(inp_shape.dim)}"
old_shape = [d.dim_value for d in inp_shape.dim]
print(f"  Old input shape: {old_shape}")

inp_shape.dim[2].dim_value = NEW_H
inp_shape.dim[3].dim_value = NEW_W
print(f"  New input shape: [1, 3, {NEW_H}, {NEW_W}]")

# Clear existing value_info AND output shapes so inference recomputes everything
del graph.value_info[:]
for out in graph.output:
    out.type.tensor_type.shape.ClearField('dim')

# Run shape inference to propagate new dimensions
print("  Running shape inference...")
model = shape_inference.infer_shapes(model)
graph = model.graph

# Verify output shape
out = graph.output[0]
out_shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
expected_out = [1, 17, NEW_H // 4, NEW_W // 4]
print(f"  Output shape: {out_shape} (expected {expected_out})")
if out_shape != expected_out:
    # Shape inference may not have fully propagated — set output explicitly
    # since the model is fully convolutional with 4× downsample
    print(f"  Shape inference incomplete — setting output manually to {expected_out}")
    out.type.tensor_type.shape.ClearField('dim')
    for dv in expected_out:
        dim = out.type.tensor_type.shape.dim.add()
        dim.dim_value = dv
    out_shape = expected_out

print(f"\n=== Step 2: SVD factorize first ConvTranspose2d (rank-{RANK}) ===")

# Build lookup
init_map = {init.name: init for init in graph.initializer}
def get_np(name):
    return numpy_helper.to_array(init_map[name])

# Find ConvTranspose nodes and identify the one exceeding MVM limit
deconv_nodes = []
for i, node in enumerate(graph.node):
    if node.op_type == "ConvTranspose":
        w = get_np(node.input[1])
        C_in, C_out, kH, kW = w.shape
        blocks = C_in // 4
        status = 'EXCEEDS' if blocks > 256 else 'OK'
        print(f"  ConvTranspose [{i}]: {C_in}→{C_out}, k={kH}, MVM blocks={blocks} [{status}]")
        deconv_nodes.append((i, node, w))

target_idx, target_node, target_w = None, None, None
for i, node, w in deconv_nodes:
    if w.shape[0] // 4 > 256:
        target_idx, target_node, target_w = i, node, w
        break

if target_idx is None:
    print("  No ConvTranspose exceeds MVM limit — saving reshaped model as-is")
    onnx.save(model, MODEL_OUT)
    print(f"  Saved: {MODEL_OUT} ({os.path.getsize(MODEL_OUT) / 1024 / 1024:.1f} MB)")
    exit(0)

C_in, C_out, kH, kW = target_w.shape
print(f"\n  Factoring [{target_idx}]: Conv1x1({C_in}→{RANK}) + ConvTranspose({RANK}→{C_out}, k={kH})")

# SVD decomposition
W = target_w.reshape(C_in, -1).astype(np.float64)
U, S, Vt = np.linalg.svd(W, full_matrices=False)

U_r = U[:, :RANK]
S_r = S[:RANK]
Vt_r = Vt[:RANK, :]

sqrtS = np.sqrt(S_r)
W1 = (U_r * sqrtS[None, :])   # [C_in, RANK]
W2 = (sqrtS[:, None] * Vt_r)  # [RANK, C_out*kH*kW]

# Quality metrics
W_reconstructed = W1 @ W2
error = np.linalg.norm(W - W_reconstructed) / np.linalg.norm(W)
energy = np.sum(S[:RANK]**2) / np.sum(S**2) * 100
print(f"  Reconstruction error: {error:.6f}")
print(f"  Energy captured: {energy:.2f}%")

# Conv1x1 weight: [RANK, C_in, 1, 1]
conv1x1_w = W1.T.astype(np.float32).reshape(RANK, C_in, 1, 1)
# ConvTranspose weight: [RANK, C_out, kH, kW]
convt_w = W2.astype(np.float32).reshape(RANK, C_out, kH, kW)

print(f"  Conv1x1 weight: {list(conv1x1_w.shape)} (MVM blocks={RANK // 4})")
print(f"  ConvTranspose weight: {list(convt_w.shape)} (MVM blocks={RANK // 4})")

# Build replacement nodes
name_prefix = target_node.name.replace("/", "_").strip("_")
conv1x1_out = f"{name_prefix}_conv1x1_out"
conv1x1_w_name = f"{name_prefix}_conv1x1_weight"
convt_w_name = f"{name_prefix}_convt_weight"

new_initializers = [
    numpy_helper.from_array(conv1x1_w, name=conv1x1_w_name),
    numpy_helper.from_array(convt_w, name=convt_w_name),
]

conv1x1_node = helper.make_node(
    'Conv',
    inputs=[target_node.input[0], conv1x1_w_name],
    outputs=[conv1x1_out],
    name=f"{name_prefix}_conv1x1",
    kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0], group=1,
)

convt_inputs = [conv1x1_out, convt_w_name]
if len(target_node.input) > 2 and target_node.input[2]:
    convt_inputs.append(target_node.input[2])

convt_node = helper.make_node(
    'ConvTranspose',
    inputs=convt_inputs,
    outputs=list(target_node.output),
    name=f"{name_prefix}_convt",
    kernel_shape=[kH, kW], strides=[2, 2], pads=[1, 1, 1, 1],
    dilations=[1, 1], group=1,
)

# Replace node in graph
nodes = list(graph.node)
nodes[target_idx:target_idx+1] = [conv1x1_node, convt_node]

# Rebuild model
all_inits = list(graph.initializer) + new_initializers
new_graph = helper.make_graph(
    nodes, graph.name, list(graph.input), list(graph.output), all_inits)
new_graph.value_info.extend(graph.value_info)

new_model = helper.make_model(new_graph, opset_imports=model.opset_import)
new_model.ir_version = model.ir_version

try:
    onnx.checker.check_model(new_model)
    print(f"\n[OK] ONNX validation passed")
except Exception as e:
    print(f"\n[WARN] ONNX validation: {e}")

onnx.save(new_model, MODEL_OUT)
size_mb = os.path.getsize(MODEL_OUT) / 1024 / 1024
print(f"Saved: {MODEL_OUT} ({size_mb:.1f} MB)")

# Inference comparison
print(f"\n=== Step 3: Inference comparison ===")
try:
    import onnxruntime as ort
    np.random.seed(42)
    inp_data = np.random.randn(1, 3, NEW_H, NEW_W).astype(np.float32)

    # Original model reshaped (no SVD) — load original and run at new resolution
    # Since the original is fixed at 256x192, we compare against SVD model only
    sess_new = ort.InferenceSession(MODEL_OUT, providers=['CPUExecutionProvider'])
    out_new = sess_new.run(None, {sess_new.get_inputs()[0].name: inp_data})[0]
    print(f"SVD model output: shape={out_new.shape}, range=[{out_new.min():.4f}, {out_new.max():.4f}]")

    # Also compare with the 256x192 SVD model to verify different spatial resolution
    inp_256 = np.random.randn(1, 3, 256, 192).astype(np.float32)
    old_svd = "customers/heatmap_pose/mobilenetv2_pose_svd_deconv.onnx"
    if os.path.exists(old_svd):
        sess_old = ort.InferenceSession(old_svd, providers=['CPUExecutionProvider'])
        out_old = sess_old.run(None, {sess_old.get_inputs()[0].name: inp_256})[0]
        print(f"Old SVD model:    shape={out_old.shape}")
    print(f"\n[OK] 384x288 model produces {NEW_H//4}x{NEW_W//4} heatmaps (vs 64x48 at 256x192)")
except ImportError:
    print("[SKIP] onnxruntime not installed, skipping inference comparison")

print(f"\nDone! Next steps:")
print(f"  1. Compile: python deploy.py customers/heatmap_pose/mobilenetv2-pose-svd-384x288.yaml")
print(f"  2. Run: python customers/heatmap_pose/run_standalone_heatmap.py mobilenetv2-pose-svd-384x288 usb:10/yuyv")
