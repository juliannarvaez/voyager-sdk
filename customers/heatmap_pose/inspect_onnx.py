#!/usr/bin/env python3
"""Inspect ConvTranspose nodes in the MobileNetV2 pose ONNX model."""
import onnx
from onnx import numpy_helper

MODEL = "customers/heatmap_pose/mobilenetv2_pose_256x192.onnx"
model = onnx.load(MODEL)
graph = model.graph

print(f"Opset: {[o.version for o in model.opset_import]}")
print(f"IR version: {model.ir_version}")

# Build weight lookup
weights = {}
for init in graph.initializer:
    weights[init.name] = numpy_helper.to_array(init)

print(f"\n=== Model IO ===")
for inp in graph.input:
    shape = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
    print(f"  Input:  {inp.name}: {shape}")
for out in graph.output:
    shape = [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]
    print(f"  Output: {out.name}: {shape}")

print(f"\n=== All node types ===")
from collections import Counter
op_counts = Counter(n.op_type for n in graph.node)
for op, cnt in sorted(op_counts.items()):
    print(f"  {op}: {cnt}")

print(f"\n=== ConvTranspose nodes ===")
for i, node in enumerate(graph.node):
    if node.op_type == "ConvTranspose":
        attrs = {}
        for attr in node.attribute:
            if attr.ints:
                attrs[attr.name] = list(attr.ints)
            elif attr.i:
                attrs[attr.name] = attr.i
            elif attr.f:
                attrs[attr.name] = attr.f
        w_name = node.input[1]
        w_shape = list(weights[w_name].shape) if w_name in weights else "?"
        b_name = node.input[2] if len(node.input) > 2 else None
        b_shape = list(weights[b_name].shape) if b_name and b_name in weights else None
        print(f"\n  Node [{i}]: {node.name or '(unnamed)'}")
        print(f"    Input tensor:  {node.input[0]}")
        print(f"    Output tensor: {node.output[0]}")
        print(f"    Weight: {w_name} shape={w_shape}")
        if b_shape:
            print(f"    Bias:   {b_name} shape={b_shape}")
        print(f"    Attrs:  {attrs}")

        # What comes after this node?
        next_nodes = [n for n in graph.node if node.output[0] in n.input]
        for nn in next_nodes:
            print(f"    → followed by: {nn.op_type} ({nn.name or ''})")

print(f"\n=== Conv nodes with large input channels ===")
for i, node in enumerate(graph.node):
    if node.op_type == "Conv":
        w_name = node.input[1]
        if w_name in weights:
            w = weights[w_name]
            if w.shape[1] > 256:  # C_in > 256
                print(f"  [{i}] Conv: weight={list(w.shape)} (C_out={w.shape[0]}, C_in={w.shape[1]})")
