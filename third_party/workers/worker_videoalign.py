#!/usr/bin/env python3
"""
worker_videoalign.py
====================
VideoAlign 批量 worker —— 在 Videoalign conda 环境中运行，一次性加载模型处理所有视频。

路径配置（VIDEOALIGN_ROOT / CHECKPOINT_PATH）从同目录的 step_videoalign.py 引入。
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

# ── 从 third_party/reward_code/step_videoalign.py 引入路径配置 ─────────────────
_THIRD_PARTY_DIR = Path(__file__).resolve().parent.parent
_REWARD_CODE = _THIRD_PARTY_DIR / "reward_code"
if str(_REWARD_CODE) not in sys.path:
    sys.path.insert(0, str(_REWARD_CODE))

from step_videoalign import VIDEOALIGN_ROOT, CHECKPOINT_PATH  # noqa: E402

if VIDEOALIGN_ROOT not in sys.path:
    sys.path.insert(0, VIDEOALIGN_ROOT)

os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")


def main():
    parser = argparse.ArgumentParser(description="VideoAlign 批量 worker（单次模型加载）")
    parser.add_argument("--batch_manifest", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    with open(args.batch_manifest, "r") as f:
        entries = json.load(f)

    device = f"cuda:{args.gpu}"
    print(f"[worker_videoalign] 共 {len(entries)} 条视频  device={device}")

    # ── 加载模型（仅一次）────────────────────────────────────────────────────
    print(f"[worker_videoalign] 加载 VideoAlign 模型 from {CHECKPOINT_PATH}")
    from inference import VideoVLMRewardInference
    inferencer = VideoVLMRewardInference(
        load_from_pretrained=CHECKPOINT_PATH,
        device=device,
        dtype=torch.bfloat16,
    )
    print("[worker_videoalign] 模型加载完成")

    # ── 逐条处理 ─────────────────────────────────────────────────────────────
    n_ok, n_err = 0, 0
    for i, entry in enumerate(entries):
        video_path = entry["video_path"]
        prompt     = entry["prompt"]
        out_json   = Path(entry["output_json"])
        skip_done  = entry.get("skip_done", True)

        print(f"\n[{i+1}/{len(entries)}] {Path(video_path).name}")

        if skip_done and out_json.exists():
            print(f"  [跳过] 已存在: {out_json}")
            n_ok += 1
            continue

        out_json.parent.mkdir(parents=True, exist_ok=True)

        try:
            with torch.no_grad():
                rewards = inferencer.reward([video_path], [prompt], use_norm=True)
            result = rewards[0]
            result["video"]  = video_path
            result["prompt"] = prompt

            with open(str(out_json), "w") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            print(f"  VQ={result['VQ']:.4f}  MQ={result['MQ']:.4f}  "
                  f"TA={result['TA']:.4f}  Overall={result['Overall']:.4f}")
            n_ok += 1

        except Exception as e:
            import traceback
            print(f"  [错误] {e}")
            traceback.print_exc()
            n_err += 1

    # ── 卸载 ─────────────────────────────────────────────────────────────────
    del inferencer
    torch.cuda.empty_cache()

    print(f"\n[worker_videoalign] 完成: 成功 {n_ok}  失败 {n_err}")


if __name__ == "__main__":
    main()
