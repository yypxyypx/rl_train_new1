#!/usr/bin/env python3
"""scannetpp_data_process.py — Build a 1600-train / 200-test set from ScanNet++ iPhone.

Outputs (per the plan)
----------------------
::

    <out>/
      manifest.json
      train.txt
      test.txt
      train/
        <scene_id>_clip<k>/
          1280x704/{start.png, gt.mp4, camera.txt, metadata.json}
          560x560/ {start.png, gt.mp4, camera.txt, metadata.json}
        ...
      test/
        ...

Camera convention is OpenCV + w2c -- ``camera.txt`` follows the unified 19
values-per-line format used by ``unified_data_process.py``::

    idx  fx/W  fy/H  cx/W  cy/H  0  0  <w2c[0:3,:] row-major 12 vals>

Reproducibility
---------------
- Master seed (``--seed``) drives both scene-pool shuffling and per-clip start
  offsets (deterministic; same seed -> same outputs).
- Each ``metadata.json`` records the master seed, the per-clip rng seed, the
  COLMAP window, the offset, and the raw frame indices/filenames.
- A top-level ``manifest.json`` captures the global config + git commit so
  others can reproduce the run end-to-end.

Usage
-----
::

    python scannetpp_data_process.py \\
        [--dataset_path /horizon-bucket/.../scannetpp] \\
        [--output       /home/users/puxin.yan-labs/RL_code/datasets/scannet] \\
        [--num_train 1600] [--num_test 200] [--clips_per_scene 2] \\
        [--num_frames 49] [--seed 0] [--skip_done] \\
        [--resolutions 1280x704,560x560]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import subprocess
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import imageio.v3 as iio
import numpy as np
from PIL import Image
from tqdm import tqdm

from datasets import scannetpp as scannetpp_parser
from datasets.base import (
    RawSample,
    compute_K_after_resize_crop,
    resize_center_crop,
)


# ════════════════════ defaults ═════════════════════════

DEFAULT_DATASET_PATH = "/horizon-bucket/robot_lab/users/haoyi.jiang/data/scannetpp"
DEFAULT_OUTPUT = "/home/users/puxin.yan-labs/RL_code/datasets/scannet++/gen3r/scannet++"

# (target_h, target_w) -- height first to match numpy ordering.
DEFAULT_RESOLUTIONS: List[Tuple[int, int]] = [
    (704, 1280),   # near user-requested 702, 32-aligned (model-friendly)
    (560, 560),    # square
]
TARGET_FPS = 16
NUM_FRAMES_DEFAULT = 49
STRIDE_DEFAULT = 2


# ════════════════════ writers (rectangular-aware) ═════════════════════════


def write_camera_txt(
    out_path: Path,
    c2ws: np.ndarray,
    Ks: np.ndarray,
    target_w: int,
    target_h: int,
) -> None:
    """idx fx/W fy/H cx/W cy/H 0 0 <w2c 3x4>  (OpenCV w2c)."""
    lines = []
    for i in range(c2ws.shape[0]):
        K = Ks[i]
        fx_n = K[0, 0] / target_w
        fy_n = K[1, 1] / target_h
        cx_n = K[0, 2] / target_w
        cy_n = K[1, 2] / target_h

        w2c = np.linalg.inv(c2ws[i])
        w2c_flat = w2c[:3, :].flatten()

        lines.append(
            f"{i} {fx_n:.10f} {fy_n:.10f} {cx_n:.10f} {cy_n:.10f} 0 0 "
            + " ".join(f"{v:.10f}" for v in w2c_flat)
        )
    out_path.write_text("\n".join(lines) + "\n")


def write_metadata_json(
    out_path: Path,
    *,
    img_w: int,
    img_h: int,
    orig_w: int,
    orig_h: int,
    caption: str,
    dataset: str,
    orig_id: str,
    sample_id: str,
    num_frames: int,
    split: str,
    extra: dict,
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
        "sample_id": sample_id,
        "split": split,
        "camera_convention": "opencv",
        "camera_format": "w2c",
    }
    # Reproducibility / scannetpp-specific fields.
    if extra:
        meta.update(extra)
    out_path.write_text(json.dumps(meta, indent=2))


def save_video_mp4(frames: List[np.ndarray], out_path: Path, fps: int = TARGET_FPS) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(
        str(out_path),
        np.stack(frames, axis=0),
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
    )


# ════════════════════ per-resolution writer ═════════════════════════


def _write_one_resolution(
    sample: RawSample,
    out_dir: Path,
    target_h: int,
    target_w: int,
    *,
    sample_id: str,
    split: str,
) -> None:
    """Resize-crop frames + recompute K, then dump to ``out_dir``."""
    frames_out: List[np.ndarray] = []
    Ks_out: List[np.ndarray] = []

    for i, raw_frame in enumerate(sample.frames):
        frames_out.append(resize_center_crop(raw_frame, target_h, target_w))

        K = sample.Ks[i]
        fx_t, fy_t, cx_t, cy_t = compute_K_after_resize_crop(
            sample.orig_w, sample.orig_h,
            float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]),
            target_w, target_h,
        )
        Ks_out.append(np.array(
            [[fx_t, 0.0, cx_t], [0.0, fy_t, cy_t], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ))

    Ks_arr = np.stack(Ks_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    Image.fromarray(frames_out[0]).save(out_dir / "start.png")
    save_video_mp4(frames_out, out_dir / "gt.mp4")
    write_camera_txt(out_dir / "camera.txt", sample.c2ws, Ks_arr, target_w, target_h)
    write_metadata_json(
        out_dir / "metadata.json",
        img_w=target_w, img_h=target_h,
        orig_w=sample.orig_w, orig_h=sample.orig_h,
        caption=sample.caption,
        dataset=sample.dataset,
        orig_id=sample.orig_id,
        sample_id=sample_id,
        num_frames=sample.c2ws.shape[0],
        split=split,
        extra=sample.extra or {},
    )


# ════════════════════ scene splitting ═════════════════════════


def _read_split_file(p: Path) -> List[str]:
    with open(p) as f:
        return [ln.strip() for ln in f if ln.strip()]


def _build_scene_pools(
    data_root: Path,
    num_train_clips: int,
    num_test_clips: int,
    clips_per_scene: int,
    seed: int,
) -> Tuple[List[str], List[str], dict]:
    """Pick disjoint train / test scene pools.

    Test pool = nvs_sem_val + nvs_test (deduped, sorted, then seeded shuffle,
    then trimmed).  Train pool = nvs_sem_train minus test pool, seeded
    shuffle, trimmed.
    """
    splits_dir = data_root / "splits"
    train_pool_all = _read_split_file(splits_dir / "nvs_sem_train.txt")
    val_scenes = _read_split_file(splits_dir / "nvs_sem_val.txt")
    test_scenes = _read_split_file(splits_dir / "nvs_test.txt")

    test_pool_full = sorted(set(val_scenes) | set(test_scenes))
    train_pool_all = [s for s in train_pool_all if s not in set(test_pool_full)]

    rng = random.Random(seed)
    rng.shuffle(train_pool_all)
    rng.shuffle(test_pool_full)

    n_train_scenes = (num_train_clips + clips_per_scene - 1) // clips_per_scene
    n_test_scenes = (num_test_clips + clips_per_scene - 1) // clips_per_scene

    if n_train_scenes > len(train_pool_all):
        raise RuntimeError(
            f"need {n_train_scenes} train scenes but pool has only "
            f"{len(train_pool_all)} (try smaller --num_train or larger "
            f"--clips_per_scene)"
        )
    if n_test_scenes > len(test_pool_full):
        raise RuntimeError(
            f"need {n_test_scenes} test scenes but pool has only "
            f"{len(test_pool_full)}"
        )

    pool_info = {
        "train_pool_file": str(splits_dir / "nvs_sem_train.txt"),
        "test_pool_files": [
            str(splits_dir / "nvs_sem_val.txt"),
            str(splits_dir / "nvs_test.txt"),
        ],
        "n_train_scenes": n_train_scenes,
        "n_test_scenes": n_test_scenes,
    }
    return train_pool_all[:n_train_scenes], test_pool_full[:n_test_scenes], pool_info


# ════════════════════ split runner ═════════════════════════


def _process_one_scene(args_tuple) -> Tuple[str, List[str], Optional[str]]:
    """Worker: process every clip in one scene; return (scene_id, sample_ids, err).

    Designed for ``multiprocessing.Pool.imap_unordered``.  All heavy work --
    parser invocation, jpeg decoding, undistort, resize, mp4 encode -- happens
    inside the worker process.
    """
    (
        scene_id, split_name, data_root, out_root,
        clips_per_scene, num_frames, master_seed,
        skip_done, target_resolutions, stride, cache_root,
    ) = args_tuple

    sample_ids: List[str] = []

    # Fast path: if all expected outputs already exist, skip the heavy parser.
    if skip_done:
        all_present = True
        provisional_ids = [f"{scene_id}_clip{k}" for k in range(clips_per_scene)]
        for sid in provisional_ids:
            for (h, w) in target_resolutions:
                if not (out_root / split_name / sid / f"{h}x{w}" / "gt.mp4").exists():
                    all_present = False
                    break
            if not all_present:
                break
        if all_present:
            return scene_id, provisional_ids, None

    try:
        parser_iter = scannetpp_parser.parse(
            data_root,
            max_samples=0,
            verbose=False,
            scene_ids=[scene_id],
            master_seed=master_seed,
            num_clips_per_scene=clips_per_scene,
            num_frames=num_frames,
            stride=stride,
            cache_root=cache_root,
        )
        for sample in parser_iter:
            sid = sample.sample_id
            sample_dir = out_root / split_name / sid
            for (target_h, target_w) in target_resolutions:
                res_dir = sample_dir / f"{target_h}x{target_w}"
                _write_one_resolution(
                    sample, res_dir, target_h, target_w,
                    sample_id=sid, split=split_name,
                )
            sample_ids.append(sid)
    except Exception as e:
        return scene_id, sample_ids, repr(e)

    return scene_id, sample_ids, None


def _process_split(
    *,
    split_name: str,
    scene_ids: List[str],
    target_clips: int,
    data_root: Path,
    out_root: Path,
    clips_per_scene: int,
    num_frames: int,
    master_seed: int,
    skip_done: bool,
    target_resolutions: List[Tuple[int, int]],
    workers: int,
    stride: int,
    cache_root: Optional[Path],
) -> List[str]:
    """Process every scene in parallel; stop accepting once we reach ``target_clips``."""
    written: List[str] = []
    errors: List[Tuple[str, str]] = []

    job_args = [
        (
            scene_id, split_name, data_root, out_root,
            clips_per_scene, num_frames, master_seed,
            skip_done, target_resolutions, stride, cache_root,
        )
        for scene_id in scene_ids
    ]

    pbar = tqdm(total=target_clips, desc=f"scannetpp[{split_name}]")

    if workers <= 1:
        results_iter = (_process_one_scene(a) for a in job_args)
        for scene_id, sids, err in results_iter:
            if err:
                errors.append((scene_id, err))
                continue
            for sid in sids:
                if len(written) >= target_clips:
                    break
                written.append(sid)
                pbar.update(1)
            if len(written) >= target_clips:
                break
    else:
        # ``spawn`` start method avoids inheriting numpy / cv2 thread pools that
        # can deadlock with fork on glibc.
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            for scene_id, sids, err in pool.imap_unordered(
                _process_one_scene, job_args
            ):
                if err:
                    errors.append((scene_id, err))
                    continue
                for sid in sids:
                    if len(written) >= target_clips:
                        break
                    written.append(sid)
                    pbar.update(1)
                if len(written) >= target_clips:
                    pool.terminate()
                    break
    pbar.close()

    if errors:
        print(f"  [{split_name}] {len(errors)} scene(s) failed:")
        for sc, err in errors[:10]:
            print(f"    {sc}: {err}")

    return written


# ════════════════════ manifest ═════════════════════════


def _git_commit_short() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        return out.decode().strip()
    except Exception:
        return ""


def _write_manifest(
    out_root: Path,
    *,
    args: argparse.Namespace,
    target_resolutions: List[Tuple[int, int]],
    pool_info: dict,
    train_ids: List[str],
    test_ids: List[str],
) -> None:
    real_dur = round((args.num_frames - 1) * args.stride / 60.0, 4)
    manifest = {
        "dataset": "scannetpp",
        "dataset_path": args.dataset_path,
        "output": str(out_root),
        "master_seed": args.seed,
        "num_train_target": args.num_train,
        "num_test_target": args.num_test,
        "num_train_actual": len(train_ids),
        "num_test_actual": len(test_ids),
        "clips_per_scene": args.clips_per_scene,
        "num_frames": args.num_frames,
        "stride": args.stride,
        "real_duration_sec": real_dur,
        "output_fps": TARGET_FPS,
        "resolutions": [f"{h}x{w}" for (h, w) in target_resolutions],
        "camera_convention": "opencv",
        "camera_format": "w2c",
        "frame_base": (
            f"iphone/rgb/ 60fps full frames, stride={args.stride}, "
            f"49 frames -> {real_dur:.2f}s real time, played back @ {TARGET_FPS}fps"
        ),
        "pose_source": "arkit pose_intrinsic_imu.json `pose` field (per-frame, 60fps)",
        "intrinsic_source": "colmap cameras.txt OPENCV model (scene-shared)",
        "undistort": True,
        "rgb_mkv_fallback_cache": str(args.cache_root) if args.cache_root else None,
        "scene_pool": pool_info,
        "code_version": _git_commit_short(),
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))


# ════════════════════ main ═════════════════════════


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset_path", default=DEFAULT_DATASET_PATH,
                   help="ScanNet++ root (contains data/ and splits/)")
    p.add_argument("--output", default=DEFAULT_OUTPUT,
                   help="Output root directory (default: %(default)s)")
    p.add_argument("--num_train", type=int, default=1600)
    p.add_argument("--num_test", type=int, default=200)
    p.add_argument("--clips_per_scene", type=int, default=2)
    p.add_argument("--num_frames", type=int, default=NUM_FRAMES_DEFAULT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip_done", action="store_true",
                   help="Skip clips where all target_res gt.mp4 already exist")
    p.add_argument("--resolutions", type=str, default="",
                   help="Override target resolutions, e.g. '704x1280,560x560'. "
                        f"Default: {','.join(f'{h}x{w}' for h,w in DEFAULT_RESOLUTIONS)}")
    p.add_argument("--workers", type=int, default=8,
                   help="Number of parallel scene workers (default 8). "
                        "Set 1 to disable multiprocessing.")
    p.add_argument("--stride", type=int, default=STRIDE_DEFAULT,
                   help=f"Frame stride within a clip in original 60fps frames "
                        f"(default {STRIDE_DEFAULT}). "
                        f"49 frames @ stride 2 = 1.6s real time.")
    p.add_argument("--cache_root", default="",
                   help="Cache directory for RGB jpgs extracted from rgb.mkv "
                        "(only used for scenes lacking iphone/rgb/). "
                        "Default: <output>/_rgb_cache")
    args = p.parse_args()

    data_root = Path(args.dataset_path)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "test").mkdir(parents=True, exist_ok=True)

    cache_root = Path(args.cache_root) if args.cache_root else (out_root / "_rgb_cache")
    cache_root.mkdir(parents=True, exist_ok=True)
    args.cache_root = str(cache_root)

    if args.resolutions.strip():
        target_resolutions: List[Tuple[int, int]] = []
        for token in args.resolutions.split(","):
            h_str, w_str = token.lower().split("x")
            target_resolutions.append((int(h_str), int(w_str)))
    else:
        target_resolutions = DEFAULT_RESOLUTIONS

    train_scenes, test_scenes, pool_info = _build_scene_pools(
        data_root,
        args.num_train, args.num_test, args.clips_per_scene, args.seed,
    )

    real_dur = round((args.num_frames - 1) * args.stride / 60.0, 4)
    print(f"\n{'='*60}")
    print(f"  ScanNet++ pipeline")
    print(f"  Dataset:      {args.dataset_path}")
    print(f"  Output:       {args.output}")
    print(f"  Cache (mkv):  {args.cache_root}")
    print(f"  Train scenes: {len(train_scenes)} -> target {args.num_train} clips")
    print(f"  Test  scenes: {len(test_scenes)} -> target {args.num_test} clips")
    print(f"  Clips/scene:  {args.clips_per_scene}")
    print(f"  Frames/clip:  {args.num_frames}  stride: {args.stride}  "
          f"(real {real_dur:.2f}s, played @ {TARGET_FPS}fps = "
          f"{args.num_frames/TARGET_FPS:.2f}s slow-mo)")
    print(f"  Resolutions:  {target_resolutions}")
    print(f"  Master seed:  {args.seed}")
    print(f"  Workers:      {args.workers}")
    print(f"{'='*60}\n")

    train_ids = _process_split(
        split_name="train", scene_ids=train_scenes, target_clips=args.num_train,
        data_root=data_root, out_root=out_root,
        clips_per_scene=args.clips_per_scene, num_frames=args.num_frames,
        master_seed=args.seed, skip_done=args.skip_done,
        target_resolutions=target_resolutions, workers=args.workers,
        stride=args.stride, cache_root=cache_root,
    )
    test_ids = _process_split(
        split_name="test", scene_ids=test_scenes, target_clips=args.num_test,
        data_root=data_root, out_root=out_root,
        clips_per_scene=args.clips_per_scene, num_frames=args.num_frames,
        master_seed=args.seed, skip_done=args.skip_done,
        target_resolutions=target_resolutions, workers=args.workers,
        stride=args.stride, cache_root=cache_root,
    )

    (out_root / "train.txt").write_text("\n".join(train_ids) + "\n")
    (out_root / "test.txt").write_text("\n".join(test_ids) + "\n")

    _write_manifest(
        out_root,
        args=args, target_resolutions=target_resolutions, pool_info=pool_info,
        train_ids=train_ids, test_ids=test_ids,
    )

    print(f"\n{'='*60}")
    print(f"  Done: {len(train_ids)} train, {len(test_ids)} test")
    print(f"  Output: {out_root}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
