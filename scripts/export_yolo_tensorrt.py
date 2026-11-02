#!/usr/bin/env python3
"""Export an Ultralytics YOLO model to TensorRT (.engine) for fast inference on NVIDIA GPUs.

Requires:
  - CUDA + cuDNN matching your PyTorch / TensorRT install
  - ``pip install ultralytics tensorrt`` (TensorRT often ships with the NVIDIA CUDA toolkit bundle)

Usage (from repo root):
  python scripts/export_yolo_tensorrt.py --weights models/best.pt --imgsz 320 --half

Then set in config:
  vision:
    model_path: "models/best.engine"
    device: "cuda"
    input_size: 320
    half_precision: true

Notes for RTX 3060 8GB:
  - ``--workspace 4`` (GiB) is usually safe; reduce to 2 if export OOMs.
  - First ``predict`` after export may take longer (TensorRT runtime tuning).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Export YOLO to TensorRT engine")
    parser.add_argument(
        "--weights",
        type=str,
        default="models/best.pt",
        help="Path to .pt checkpoint",
    )
    parser.add_argument("--imgsz", type=int, default=320, help="Square inference size (must match runtime)")
    parser.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="FP16 engine (recommended on RTX 3060)",
    )
    parser.add_argument(
        "--workspace",
        type=float,
        default=4.0,
        help="TensorRT workspace size in GiB",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device id for export",
    )
    parser.add_argument(
        "--simplify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="ONNX simplify step (used before TRT build)",
    )
    args = parser.parse_args()

    w = Path(args.weights)
    if not w.is_file():
        print(f"ERROR: weights not found: {w.resolve()}", file=sys.stderr)
        return 1

    try:
        from ultralytics import YOLO
    except ImportError as e:
        print("ERROR: pip install ultralytics", file=sys.stderr)
        raise e

    print(f"Loading {w} …")
    model = YOLO(str(w))

    print(
        f"Exporting TensorRT: imgsz={args.imgsz} half={args.half} "
        f"workspace={args.workspace}GiB device=cuda:{args.device}"
    )
    # Ultralytics builds ONNX → TensorRT internally for format="engine"
    out = model.export(
        format="engine",
        imgsz=args.imgsz,
        half=args.half,
        device=args.device,
        workspace=args.workspace,
        simplify=args.simplify,
    )
    print(f"Done: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
