#!/usr/bin/env python3
"""unified_data_process.py — Convert raw datasets to pipeline-ready format.

Two-stage pipeline:
  Stage 1: Dataset-specific parser reads raw data → RawSample (OpenCV c2w)
  Stage 2: This script applies resize/crop + writes fixed-format disk files

Output per sample: <output_root>/<dataset>/<sample_id>/
  start.png       – first frame
  gt.mp4          – GT video at target resolution, 16 fps
  camera.txt      – fixed format: idx fx/W fy/H cx/W cy/H 0 0 <w2c 3x4>
  metadata.json   – img_w, img_h, caption, dataset, camera_convention, etc.
  gt_depth.npz    – (optional) depth maps

Usage:
    python unified_data_process.py \\
        --dataset re10k \\
        --dataset_path /path/to/raw_data \\
        --model gen3r \\
        --output /path/to/processed \\
        [--num_frames 49] [--target_size 560] \\
        [--sample_mode fixed] [--skip_done] [--max_samples 50]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

import imageio.v3 as iio
import numpy as np
from PIL import Image
from tqdm import tqdm

from datasets import DATASET_REGISTRY
from datasets.base import (
    RawSample,
    compute_K_after_resize_crop,
    resize_center_crop,
    resize_center_crop_depth,
    sample_frame_indices,
)

# ═══════════════════════ Model Configs ═══════════════════════
# Each model defines default target_size and num_frames.
# These can be overridden via CLI arguments.

MODEL_DEFAULTS = {
    "gen3r": {"target_size": 560, "num_frames": 49, "fps": 16},
    # Future models:
    # "wan22": {"target_size": 512, "num_frames": 81, "fps": 16},
}

TARGET_FPS = 16


# ═══════════════════════ Disk Writers ═══════════════════════


def write_camera_txt(
    out_path: str,
    c2ws: np.ndarray,
    Ks: np.ndarray,
    target_w: int,
    target_h: int,
) -> None:
    """Write camera.txt in fixed format.

    Format per line (19 space-separated values):
        idx  fx/W  fy/H  cx/W  cy/H  0  0  <w2c[0:3,:] row-major 12 values>

    c2ws are OpenCV convention camera-to-world matrices (N, 4, 4).
    They are inverted to w2c for storage.
    Ks are pixel intrinsics (N, 3, 3) already at target resolution.
    Intrinsics are normalised by target_w / target_h for storage.
    """
    N = c2ws.shape[0]
    lines = []
    for i in range(N):
        K = Ks[i]
        fx_n = K[0, 0] / target_w
        fy_n = K[1, 1] / target_h
        cx_n = K[0, 2] / target_w
        cy_n = K[1, 2] / target_h

        w2c = np.linalg.inv(c2ws[i])
        w2c_flat = w2c[:3, :].flatten()

        vals = (
            f"{i} {fx_n:.10f} {fy_n:.10f} {cx_n:.10f} {cy_n:.10f} 0 0 "
            + " ".join(f"{v:.10f}" for v in w2c_flat)
        )
        lines.append(vals)

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_metadata_json(
    out_path: str,
    img_w: int,
    img_h: int,
    orig_w: int,
    orig_h: int,
    caption: str,
    dataset: str,
    orig_id: str,
    start_frame: int = 0,
    num_frames: int = 49,
    model: str = "",
) -> None:
    meta = {
        "img_w": img_w,
        "img_h": img_h,
        "orig_w": orig_w,
        "orig_h": orig_h,
        "num_frames": num_frames,
        "caption": caption,
        "dataset": dataset,
        "orig_id": orig_id,
        "start_frame": start_frame,
        "model": model,
        "camera_convention": "opencv",
        "camera_format": "w2c",
    }
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)


def save_video_mp4(
    frames: List[np.ndarray],
    out_path: str,
    fps: int = TARGET_FPS,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    iio.imwrite(
        out_path,
        np.stack(frames, axis=0),
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
    )


# ═══════════════════════ Core Conversion ═══════════════════════


def convert_sample(
    sample: RawSample,
    output_dir: Path,
    target_size: int,
    num_frames: int,
    fixed_start: bool = False,
    include_depth: bool = False,
    model: str = "",
) -> bool:
    """Convert a RawSample to disk files at target resolution.

    Returns True on success, False if skipped/failed.
    """
    total_frames = len(sample.frames)
    if total_frames < num_frames:
        print(f"  [skip] {sample.sample_id}: only {total_frames} frames, need {num_frames}")
        return False

    indices = sample_frame_indices(total_frames, num_frames, fixed_start)

    # Resize + center crop frames
    frames_out = []
    for idx in indices:
        cropped = resize_center_crop(sample.frames[idx], target_size, target_size)
        frames_out.append(cropped)

    # Recompute K at target resolution for selected frames
    Ks_out = []
    for idx in indices:
        K = sample.Ks[idx]
        fx_t, fy_t, cx_t, cy_t = compute_K_after_resize_crop(
            sample.orig_w, sample.orig_h,
            K[0, 0], K[1, 1], K[0, 2], K[1, 2],
            target_size, target_size,
        )
        K_new = np.array([
            [fx_t, 0.0, cx_t],
            [0.0, fy_t, cy_t],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        Ks_out.append(K_new)

    c2ws_out = sample.c2ws[indices]
    Ks_out = np.stack(Ks_out)

    # Write files
    output_dir.mkdir(parents=True, exist_ok=True)

    Image.fromarray(frames_out[0]).save(str(output_dir / "start.png"))
    save_video_mp4(frames_out, str(output_dir / "gt.mp4"))
    write_camera_txt(
        str(output_dir / "camera.txt"),
        c2ws_out, Ks_out, target_size, target_size,
    )
    write_metadata_json(
        str(output_dir / "metadata.json"),
        img_w=target_size,
        img_h=target_size,
        orig_w=sample.orig_w,
        orig_h=sample.orig_h,
        caption=sample.caption,
        dataset=sample.dataset,
        orig_id=sample.orig_id,
        start_frame=indices[0],
        num_frames=num_frames,
        model=model,
    )

    # Depth (optional)
    if include_depth and sample.depths is not None:
        depths_out = []
        for idx in indices:
            if idx < sample.depths.shape[0]:
                d = resize_center_crop_depth(sample.depths[idx], target_size, target_size)
                depths_out.append(d)
        if depths_out:
            np.savez_compressed(
                str(output_dir / "gt_depth.npz"),
                depth=np.stack(depths_out, axis=0),
            )

    return True


# ═══════════════════════ Main ═══════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw datasets to pipeline-ready format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=list(DATASET_REGISTRY.keys()),
        help=f"Dataset to process. Choices: {list(DATASET_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True,
        help="Root directory of the raw dataset",
    )
    parser.add_argument(
        "--model", type=str, default="gen3r",
        choices=list(MODEL_DEFAULTS.keys()),
        help="Target model config (controls default resolution/frames)",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output root directory",
    )
    parser.add_argument(
        "--target_size", type=int, default=None,
        help="Target resolution (overrides model default)",
    )
    parser.add_argument(
        "--num_frames", type=int, default=None,
        help="Number of frames to sample (overrides model default)",
    )
    parser.add_argument(
        "--sample_mode", choices=["random", "fixed"], default="fixed",
        help="Frame sampling: random (training) or fixed (eval, default)",
    )
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Max samples to process (0 = all)")
    parser.add_argument("--skip_done", action="store_true",
                        help="Skip samples whose gt.mp4 already exists")
    parser.add_argument("--include_depth", action="store_true",
                        help="Also process depth maps if available")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    # Resolve target_size and num_frames from model defaults
    model_cfg = MODEL_DEFAULTS[args.model]
    target_size = args.target_size or model_cfg["target_size"]
    num_frames = args.num_frames or model_cfg["num_frames"]
    fixed_start = (args.sample_mode == "fixed")

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Dataset:     {args.dataset}")
    print(f"  Model:       {args.model}")
    print(f"  Target size: {target_size}x{target_size}")
    print(f"  Num frames:  {num_frames}")
    print(f"  Sample mode: {args.sample_mode}")
    print(f"  Output:      {output_root}")
    print(f"{'='*60}\n")

    # Stage 1: Parse raw dataset → RawSample iterator
    parse_fn = DATASET_REGISTRY[args.dataset]
    raw_samples = parse_fn(
        Path(args.dataset_path),
        max_samples=args.max_samples,
        verbose=not args.quiet,
    )

    # Stage 2: Convert each sample
    success, skipped, failed = 0, 0, 0
    for sample in tqdm(raw_samples, desc=args.dataset):
        out_dir = output_root / args.dataset / sample.sample_id

        if args.skip_done and (out_dir / "gt.mp4").exists():
            skipped += 1
            continue

        try:
            ok = convert_sample(
                sample, out_dir, target_size, num_frames,
                fixed_start=fixed_start,
                include_depth=args.include_depth,
                model=args.model,
            )
            if ok:
                success += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  [error] {sample.sample_id}: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"  Done: {success} converted, {skipped} skipped, {failed} failed")
    print(f"  Output: {output_root / args.dataset}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
