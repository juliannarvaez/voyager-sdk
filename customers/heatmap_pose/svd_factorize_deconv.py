#!/usr/bin/env python3
"""
Fix MobileNetV2 pose ONNX for AIPU by SVD-factoring the first ConvTranspose2d.

Problem: ConvTranspose(1280→256, k=4, s=2, p=1) needs 1280/4 = 320 MVM blocks > 256 limit.
The other two ConvTranspose(256→256) only need 64 blocks — they're fine as-is.

Solution: Replace ONLY the first ConvTranspose with:
  - Conv1x1(1280→512) — pointwise channel reduction (128 blocks, compiles fine)
  - ConvTranspose(512→256, k=4, s=2, p=1) — spatial upsampling (128 blocks, under limit)

Weight decomposition via SVD (optimal low-rank approximation):
  W_orig [1280, 256, 4, 4] ≈ W1^T @ W2
  where W1 [512, 1280] and W2 [512, 256*4*4]

This preserves the ConvTranspose spatial alignment EXACTLY — no pixel offset.
"""
import os
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

MODEL_IN = "customers/heatmap_pose/mobilenetv2_pose_256x192.onnx"
MODEL_OUT = "customers/heatmap_pose/mobilenetv2_pose_svd_deconv.onnx"
RANK = 512  # intermediate channels (512/4 = 128 MVM blocks, well under 256)

model = onnx.load(MODEL_IN)
graph = model.graph

# Build lookup
init_map = {init.name: init for init in graph.initializer}
def get_np(name):
    return numpy_helper.to_array(init_map[name])

# Find ConvTranspose nodes
deconv_nodes = []
for i, node in enumerate(graph.node):
    if node.op_type == "ConvTranspose":
        w = get_np(node.input[1])
        C_in, C_out, kH, kW = w.shape
        blocks = C_in // 4
        print(f"ConvTranspose [{i}]: {node.name}")
        print(f"  weight: {list(w.shape)}, MVM blocks={blocks}, {'EXCEEDS' if blocks > 256 else 'OK'}")
        deconv_nodes.append((i, node, w))

# Only replace the first ConvTranspose (1280→256) that exceeds MVM limit
target_idx, target_node, target_w = None, None, None
for i, node, w in deconv_nodes:
    if w.shape[0] // 4 > 256:
        target_idx, target_node, target_w = i, node, w
        break

if target_idx is None:
    print("No ConvTranspose exceeds MVM limit — nothing to do")
    exit(0)

C_in, C_out, kH, kW = target_w.shape
print(f"\nFactoring ConvTranspose [{target_idx}]: ({C_in}→{C_out}, k={kH}) via SVD rank-{RANK}")
print(f"  Conv1x1({C_in}→{RANK}) + ConvTranspose({RANK}→{C_out}, k={kH}, s=2, p=1)")

# SVD factorization
# W_orig: [C_in, C_out, kH, kW] = [1280, 256, 4, 4]
# Reshape to [C_in, C_out * kH * kW] = [1280, 4096]
W = target_w.reshape(C_in, -1).astype(np.float64)

U, S, Vt = np.linalg.svd(W, full_matrices=False)
# U: [1280, 1280], S: [1280], Vt: [1280, 4096] (since C_in < C_out*k*k is False here)
# Actually: min(1280, 4096) = 1280, so S has 1280 values

# Keep top RANK singular values
U_r = U[:, :RANK]          # [1280, 512]
S_r = S[:RANK]             # [512]
Vt_r = Vt[:RANK, :]        # [512, 4096]

# Split sqrt(S) between the two factors for numerical stability
sqrtS = np.sqrt(S_r)
W1 = (U_r * sqrtS[None, :])   # [1280, 512] — Conv1x1 weights (transposed)
W2 = (sqrtS[:, None] * Vt_r)  # [512, 4096] — ConvTranspose weights

# Verify decomposition quality
W_reconstructed = W1 @ W2
error = np.linalg.norm(W - W_reconstructed) / np.linalg.norm(W)
energy = np.sum(S[:RANK]**2) / np.sum(S**2) * 100
print(f"  SVD reconstruction error: {error:.6f} (relative Frobenius norm)")
print(f"  Singular value energy captured: {energy:.2f}%")

# Conv1x1 weight: [C_out_conv1x1, C_in, 1, 1] = [RANK, C_in, 1, 1]
conv1x1_w = W1.T.astype(np.float32).reshape(RANK, C_in, 1, 1)

# ConvTranspose weight: [C_in_ct, C_out, kH, kW] = [RANK, C_out, 4, 4]
convt_w = W2.astype(np.float32).reshape(RANK, C_out, kH, kW)

print(f"  Conv1x1 weight: {list(conv1x1_w.shape)} (MVM blocks={RANK // 4})")
print(f"  ConvTranspose weight: {list(convt_w.shape)} (MVM blocks={RANK // 4})")

# Build new nodes
name_prefix = target_node.name.replace("/", "_").strip("_")
conv1x1_out = f"{name_prefix}_conv1x1_out"
conv1x1_w_name = f"{name_prefix}_conv1x1_weight"
convt_w_name = f"{name_prefix}_convt_weight"

new_initializers = [
    numpy_helper.from_array(conv1x1_w, name=conv1x1_w_name),
    numpy_helper.from_array(convt_w, name=convt_w_name),
]

# Conv1x1 node: (B, 1280, H, W) → (B, 512, H, W)
conv1x1_node = helper.make_node(
    'Conv',
    inputs=[target_node.input[0], conv1x1_w_name],
    outputs=[conv1x1_out],
    name=f"{name_prefix}_conv1x1",
    kernel_shape=[1, 1],
    strides=[1, 1],
    pads=[0, 0, 0, 0],
    group=1,
)

# ConvTranspose node: (B, 512, H, W) → (B, 256, 2H, 2W)
# Reuses the original bias if present
convt_inputs = [conv1x1_out, convt_w_name]
if len(target_node.input) > 2 and target_node.input[2]:
    convt_inputs.append(target_node.input[2])

convt_node = helper.make_node(
    'ConvTranspose',
    inputs=convt_inputs,
    outputs=list(target_node.output),
    name=f"{name_prefix}_convt",
    kernel_shape=[kH, kW],
    strides=[2, 2],
    pads=[1, 1, 1, 1],
    dilations=[1, 1],
    group=1,
)

# Replace the target node in the graph
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
print(f"Saved: {MODEL_OUT}")
print(f"Size: {os.path.getsize(MODEL_OUT) / 1024 / 1024:.1f} MB")

# Inference comparison (if onnxruntime available)
try:
    import onnxruntime as ort
    np.random.seed(42)
    inp = np.random.randn(1, 3, 256, 192).astype(np.float32)

    sess_orig = ort.InferenceSession(MODEL_IN, providers=['CPUExecutionProvider'])
    out_orig = sess_orig.run(None, {sess_orig.get_inputs()[0].name: inp})[0]

    sess_new = ort.InferenceSession(MODEL_OUT, providers=['CPUExecutionProvider'])
    out_new = sess_new.run(None, {sess_new.get_inputs()[0].name: inp})[0]

    print(f"\n--- Inference comparison (256x192, random input) ---")
    print(f"Original:  shape={out_orig.shape}, range=[{out_orig.min():.4f}, {out_orig.max():.4f}]")
    print(f"SVD-fixed: shape={out_new.shape}, range=[{out_new.min():.4f}, {out_new.max():.4f}]")

    diff = np.abs(out_new - out_orig)
    print(f"Abs diff:  mean={diff.mean():.6f}, max={diff.max():.6f}")

    # Check heatmap peak alignment (most important for pose)
    for k in range(17):
        orig_peak = np.unravel_index(out_orig[0, k].argmax(), out_orig[0, k].shape)
        new_peak = np.unravel_index(out_new[0, k].argmax(), out_new[0, k].shape)
        if orig_peak != new_peak:
            print(f"  Keypoint {k}: peak moved {orig_peak} → {new_peak}")
    print("[OK] Peak comparison done")
except ImportError:
    print("\n[SKIP] onnxruntime not installed, skipping inference comparison")
