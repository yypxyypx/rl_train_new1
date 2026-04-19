"""
Step: SAM3 Video Segmentation (Phase B, after step_qwen.py)
===========================================================
Conda env: SAM3

读 step_qwen.py 写出的 objects.json + frames，调用 SAM3 视频追踪模式做语义
分割，输出 label_maps.npz（int16）。如果 objects 列表为空，直接写零矩阵。

与 step_qwen_sam3.py 的差异：本脚本不加载 Qwen-VL，单卡显存 ≈ 6 GB。

Usage:
    conda run -n SAM3 python step_sam3.py \
        --video_frames_dir /path/to/frames/ \
        --objects_json /path/to/objects.json \
        --output /path/to/label_maps.npz \
        --gpu 0
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

_REWARD_DIR = Path(__file__).resolve().parent.parent
_RL_TRAIN_DIR = _REWARD_DIR.parent
_RL_CODE_DIR = _RL_TRAIN_DIR.parent
_THIRD_PARTY_DIR = _RL_CODE_DIR / "third_party"

_SAM3_SRC = _THIRD_PARTY_DIR / "repos" / "SAM3" / "sam3"
_SAM3_PKG = _THIRD_PARTY_DIR / "repos" / "SAM3"
for _p in [str(_SAM3_SRC), str(_SAM3_PKG)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

if str(_REWARD_DIR / "steps") not in sys.path:
    sys.path.insert(0, str(_REWARD_DIR / "steps"))

from step_qwen_sam3 import segment_frames_video  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="SAM3 video segmentation (consume objects.json)")
    parser.add_argument("--video_frames_dir", required=True)
    parser.add_argument("--objects_json", required=True,
                        help="objects.json produced by step_qwen.py")
    parser.add_argument("--output", required=True,
                        help="Output label_maps .npz path")
    parser.add_argument("--masks_output", default=None,
                        help="Optionally save raw masks .npz")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"

    frame_paths = sorted([
        os.path.join(args.video_frames_dir, f)
        for f in os.listdir(args.video_frames_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    if not frame_paths:
        raise FileNotFoundError(f"No image frames in {args.video_frames_dir}")
    print(f"[step_sam3] Found {len(frame_paths)} frames", flush=True)

    if not os.path.exists(args.objects_json):
        raise FileNotFoundError(
            f"objects_json not found: {args.objects_json}. Run step_qwen.py first."
        )
    with open(args.objects_json, "r") as f:
        obj_data = json.load(f)
    if isinstance(obj_data, dict):
        object_names = obj_data.get("objects", obj_data.get("object_names", []))
    else:
        object_names = obj_data
    print(f"[step_sam3] Loaded objects: {object_names}", flush=True)

    if not object_names:
        print("[step_sam3] 物体列表为空，写零 label_maps", flush=True)
        T = len(frame_paths)
        first_bgr = cv2.imread(frame_paths[0])
        H, W = first_bgr.shape[:2]
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        np.savez_compressed(
            args.output,
            label_maps=np.zeros((T, H, W), dtype=np.int16),
            object_names=np.array([], dtype=object),
        )
        return

    masks, label_maps = segment_frames_video(args.video_frames_dir, object_names, device)

    sys.path.insert(0, str(_REWARD_DIR))
    from reward_metrics import filter_unstable_masks
    filtered_masks, removed = filter_unstable_masks(masks, object_names=object_names)
    if removed:
        print(f"[step_sam3] Filtered {len(removed)} unstable objects:", flush=True)
        for idx, name, reason, areas in removed:
            print(f"  - {name} (#{idx}): {reason}", flush=True)
        T, H, W = label_maps.shape
        label_maps = np.zeros((T, H, W), dtype=np.int16)
        for i in range(len(object_names) - 1, -1, -1):
            label_maps[filtered_masks[i]] = i + 1
        masks = filtered_masks

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(
        args.output,
        label_maps=label_maps,
        object_names=np.array(object_names, dtype=object),
    )
    print(f"[step_sam3] Saved label_maps to {args.output}  "
          f"物体数: {len(object_names)}  覆盖率: {(label_maps > 0).mean():.1%}",
          flush=True)

    if args.masks_output:
        os.makedirs(os.path.dirname(os.path.abspath(args.masks_output)), exist_ok=True)
        pixel_total = float(label_maps.shape[1] * label_maps.shape[2])
        mean_areas = masks.sum(axis=(2, 3)).mean(axis=1) / pixel_total
        np.savez_compressed(
            args.masks_output,
            object_names=np.array(object_names, dtype=object),
            masks=masks,
            mean_areas=mean_areas,
        )


if __name__ == "__main__":
    main()
