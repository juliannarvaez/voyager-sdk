#!/usr/bin/env python3
# Copyright Axelera AI, 2025

"""
Auto-calibrate One Euro Filter parameters from stroke data files.

Analyzes stroke_*.json files from rowing_ergometer_recording.py to estimate:
- sample rate (Hz)
- noise level (low-speed jitter)
- max speed (high-speed motion)

Then grid-searches min_cutoff and beta to satisfy target precision and minimize lag.

Usage:
  python calibrate_from_strokes.py --input-dir /tmp/stroke_data --keypoint-idx 9
"""

import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

from one_euro_filter import OneEuroFilter1D


def _parse_range(range_str: str) -> Tuple[float, float, float]:
    parts = [p.strip() for p in range_str.split(",")]
    if len(parts) != 3:
        raise ValueError("Range must be formatted as 'start,end,step'")
    return float(parts[0]), float(parts[1]), float(parts[2])


def _load_stroke_file(json_path: Path, keypoint_idx: int, min_conf: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load keypoint trajectories from a stroke JSON file."""
    try:
        data = json.loads(json_path.read_text())
    except Exception:
        return np.array([]), np.array([]), np.array([])

    keypoints = data.get("keypoints", [])
    timestamps = data.get("frame_timestamps", [])

    if len(keypoints) != len(timestamps):
        return np.array([]), np.array([]), np.array([])

    xs: List[float] = []
    ys: List[float] = []
    ts: List[float] = []

    for kpts, t in zip(keypoints, timestamps):
        if not isinstance(kpts, list) or keypoint_idx >= len(kpts):
            continue
        k = kpts[keypoint_idx]
        if not isinstance(k, dict):
            continue

        x = k.get("x")
        y = k.get("y")
        conf = k.get("confidence", 1.0)

        if x is None or y is None or conf < min_conf:
            continue

        xs.append(float(x))
        ys.append(float(y))
        ts.append(float(t))

    return np.array(xs), np.array(ys), np.array(ts)


def _estimate_sample_rate(timestamps: np.ndarray) -> float:
    if len(timestamps) < 2:
        return 0.0
    deltas = np.diff(timestamps)
    deltas = deltas[deltas > 0]
    if len(deltas) == 0:
        return 0.0
    return 1.0 / float(np.mean(deltas))


def _compute_speeds(xs: np.ndarray, ys: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    speeds = np.zeros_like(xs)
    if len(xs) < 2:
        return speeds
    dt = np.diff(timestamps)
    dx = np.diff(xs)
    dy = np.diff(ys)
    dt = np.where(dt <= 0, np.nan, dt)
    v = np.sqrt(dx * dx + dy * dy) / dt
    speeds[1:] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    return speeds


def _apply_filter(series: np.ndarray, timestamps: np.ndarray, min_cutoff: float, beta: float, d_cutoff: float) -> np.ndarray:
    filt = OneEuroFilter1D(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
    out = np.zeros_like(series)
    for i, (x, t) in enumerate(zip(series, timestamps)):
        out[i], _ = filt.filter(float(x), float(t))
    return out


def _estimate_lag_frames(raw: np.ndarray, filt: np.ndarray, max_lag_frames: int) -> int:
    if len(raw) < 2 * max_lag_frames + 3:
        return 0
    raw = raw - np.mean(raw)
    filt = filt - np.mean(filt)
    best_lag = 0
    best_corr = -np.inf
    for lag in range(0, max_lag_frames + 1):
        if lag == 0:
            a = raw
            b = filt
        else:
            a = raw[lag:]
            b = filt[:-lag]
        if len(a) < 5:
            continue
        corr = float(np.dot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
    return best_lag


def calibrate(xs: np.ndarray,
              ys: np.ndarray,
              timestamps: np.ndarray,
              min_cutoff_range: Tuple[float, float, float],
              beta_range: Tuple[float, float, float],
              d_cutoff: float,
              target_precision_px: float,
              max_lag_s: float,
              low_speed_percentile: float,
              high_speed_percentile: float) -> dict:
    sample_hz = _estimate_sample_rate(timestamps)
    speeds = _compute_speeds(xs, ys, timestamps)

    low_thresh = np.percentile(speeds, low_speed_percentile)
    high_thresh = np.percentile(speeds, high_speed_percentile)

    low_mask = speeds <= low_thresh
    high_mask = speeds >= high_thresh

    max_lag_frames = int(max_lag_s * sample_hz) if sample_hz > 0 else 0

    best = {
        "min_cutoff": None,
        "beta": None,
        "jitter": math.inf,
        "lag_s": math.inf,
        "sample_hz": sample_hz,
        "low_speed_threshold": low_thresh,
        "high_speed_threshold": high_thresh,
    }

    min_start, min_end, min_step = min_cutoff_range
    beta_start, beta_end, beta_step = beta_range

    min_values = np.arange(min_start, min_end + 1e-9, min_step)
    beta_values = np.arange(beta_start, beta_end + 1e-9, beta_step)

    for min_cutoff in min_values:
        for beta in beta_values:
            fx = _apply_filter(xs, timestamps, min_cutoff, beta, d_cutoff)
            fy = _apply_filter(ys, timestamps, min_cutoff, beta, d_cutoff)

            if low_mask.any():
                jitter_x = float(np.std(fx[low_mask]))
                jitter_y = float(np.std(fy[low_mask]))
                jitter = (jitter_x + jitter_y) / 2.0
            else:
                jitter = float(np.std(fx) + np.std(fy)) / 2.0

            if max_lag_frames > 0 and high_mask.any():
                lag_x = _estimate_lag_frames(xs[high_mask], fx[high_mask], max_lag_frames)
                lag_y = _estimate_lag_frames(ys[high_mask], fy[high_mask], max_lag_frames)
                lag_frames = max(lag_x, lag_y)
                lag_s = lag_frames / sample_hz if sample_hz > 0 else 0.0
            else:
                lag_s = 0.0

            meets_precision = jitter <= target_precision_px
            if meets_precision:
                if lag_s < best["lag_s"] or (math.isclose(lag_s, best["lag_s"]) and jitter < best["jitter"]):
                    best.update({"min_cutoff": float(min_cutoff), "beta": float(beta), "jitter": jitter, "lag_s": lag_s})
            else:
                if best["min_cutoff"] is None and jitter < best["jitter"]:
                    best.update({"min_cutoff": float(min_cutoff), "beta": float(beta), "jitter": jitter, "lag_s": lag_s})

    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-calibrate One Euro Filter from stroke data files")
    parser.add_argument("--input-dir", default="/tmp/stroke_data", help="Directory containing stroke_*.json files")
    parser.add_argument("--keypoint-idx", type=int, default=9, help="Keypoint index to use (default: 9 = right wrist)")
    parser.add_argument("--min-confidence", type=float, default=0.3, help="Minimum keypoint confidence")
    parser.add_argument("--min-cutoff-range", default="0.1,5.0,0.05", help="min_cutoff range: start,end,step")
    parser.add_argument("--beta-range", default="0.0,0.05,0.001", help="beta range: start,end,step")
    parser.add_argument("--d-cutoff", type=float, default=1.0, help="Derivative cutoff (default: 1.0)")
    parser.add_argument("--target-precision-px", type=float, default=1.0, help="Target jitter precision (px)")
    parser.add_argument("--max-lag-s", type=float, default=0.08, help="Maximum acceptable lag (seconds)")
    parser.add_argument("--low-speed-percentile", type=float, default=20.0, help="Percentile for low-speed jitter")
    parser.add_argument("--high-speed-percentile", type=float, default=80.0, help="Percentile for high-speed lag")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Directory not found: {input_dir}")

    stroke_files = sorted(input_dir.glob("stroke_*.json"))
    if not stroke_files:
        raise SystemExit(f"No stroke_*.json files found in {input_dir}")

    print(f"Found {len(stroke_files)} stroke files")

    all_xs: List[float] = []
    all_ys: List[float] = []
    all_ts: List[float] = []

    for stroke_file in stroke_files:
        xs, ys, ts = _load_stroke_file(stroke_file, args.keypoint_idx, args.min_confidence)
        if len(xs) == 0:
            continue
        all_xs.extend(xs.tolist())
        all_ys.extend(ys.tolist())
        all_ts.extend(ts.tolist())

    if len(all_xs) < 20:
        raise SystemExit("Not enough valid samples for calibration (need at least 20)")

    xs_arr = np.array(all_xs, dtype=np.float64)
    ys_arr = np.array(all_ys, dtype=np.float64)
    ts_arr = np.array(all_ts, dtype=np.float64)

    print(f"Loaded {len(xs_arr)} samples from {len(stroke_files)} strokes")

    best = calibrate(
        xs_arr,
        ys_arr,
        ts_arr,
        _parse_range(args.min_cutoff_range),
        _parse_range(args.beta_range),
        args.d_cutoff,
        args.target_precision_px,
        args.max_lag_s,
        args.low_speed_percentile,
        args.high_speed_percentile,
    )

    print("=" * 60)
    print("One Euro Filter Auto-Calibration Results")
    print("=" * 60)
    print(f"Samples used: {len(xs_arr)}")
    print(f"Estimated sample rate: {best['sample_hz']:.2f} Hz")
    print(f"Low-speed threshold: {best['low_speed_threshold']:.3f} px/s")
    print(f"High-speed threshold: {best['high_speed_threshold']:.3f} px/s")
    print("-" * 60)
    print(f"Best min_cutoff: {best['min_cutoff']}")
    print(f"Best beta: {best['beta']}")
    print(f"Jitter (px): {best['jitter']:.3f}")
    print(f"Lag (s): {best['lag_s']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
