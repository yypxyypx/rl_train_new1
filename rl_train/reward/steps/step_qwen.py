"""
Step: Qwen-VL First-Frame Object Identification (Phase A)
=========================================================
Conda env: SAM3

仅用 Qwen3-VL 识别视频首帧的物体，写出 objects.json，然后立即退出
（subprocess 结束时 Qwen 模型自动从显存释放）。

与 step_qwen_sam3.py 的关系：
- step_qwen_sam3.py 在同一进程内先跑 Qwen 再跑 SAM3，单卡峰值 ≈ 18-22 GB
- 拆分后 step_qwen.py 只占 Qwen-VL 显存（~14 GB），写完 objects.json 就退出
- 配合 step_sam3.py（只占 ~6 GB）顺序运行，单卡峰值降到 max(14, 6) = 14 GB
- 旧的 step_qwen_sam3.py 仍保留以兼容 per_rank dispatch 模式

Usage:
    conda run -n SAM3 python step_qwen.py \
        --video_frames_dir /path/to/frames/ \
        --objects_output /path/to/objects.json \
        --gpu 0
"""

import argparse
import json
import os
import sys
from pathlib import Path

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

from step_qwen_sam3 import identify_objects  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Qwen-VL first-frame object identification")
    parser.add_argument("--video_frames_dir", required=True)
    parser.add_argument("--objects_output", required=True,
                        help="Output objects.json path")
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
    print(f"[step_qwen] Found {len(frame_paths)} frames", flush=True)

    object_names = identify_objects(frame_paths[0], device)

    os.makedirs(os.path.dirname(os.path.abspath(args.objects_output)), exist_ok=True)
    with open(args.objects_output, "w") as f:
        json.dump({"objects": object_names}, f, ensure_ascii=False, indent=2)
    print(f"[step_qwen] Saved {len(object_names)} objects to {args.objects_output}",
          flush=True)


if __name__ == "__main__":
    main()
