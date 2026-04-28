"""
Step: VideoAlign Video Quality Scoring
========================================
Conda env: Videoalign

Usage:
    conda run -n Videoalign python step_videoalign.py \
        --video_path /path/to/video.mp4 \
        --prompt "camera moving through a room" \
        --output /path/to/videoalign_scores.json \
        --gpu 0
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

_REWARD_DIR = Path(__file__).resolve().parent.parent
_RL_TRAIN_DIR = _REWARD_DIR.parent
_RL_CODE_DIR = _RL_TRAIN_DIR.parent
_THIRD_PARTY_DIR = _RL_CODE_DIR / "third_party"

VIDEOALIGN_ROOT = str(_THIRD_PARTY_DIR / "repos" / "VideoAlign" / "VideoAlign")
CHECKPOINT_PATH = str(
    Path(os.environ.get("RL_MODEL_ROOT", str(_RL_CODE_DIR / "model")))
    / "Videoalign" / "VideoReward"
)


def score_video(video_path: str, prompt: str, device: str = "cuda:0") -> dict:
    """Score a video using VideoAlign (VideoReward)."""
    sys.path.insert(0, VIDEOALIGN_ROOT)
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"

    from inference import VideoVLMRewardInference

    print(f"[VideoAlign] Loading model from {CHECKPOINT_PATH}")
    inferencer = VideoVLMRewardInference(
        load_from_pretrained=CHECKPOINT_PATH,
        device=device,
        dtype=torch.bfloat16,
    )
    print(f"[VideoAlign] Model loaded on {device}")

    with torch.no_grad():
        rewards = inferencer.reward([video_path], [prompt], use_norm=True)

    result = rewards[0]
    print(f"[VideoAlign] Scores: VQ={result['VQ']:.4f}, MQ={result['MQ']:.4f}, "
          f"TA={result['TA']:.4f}, Overall={result['Overall']:.4f}")

    del inferencer
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description="VideoAlign scoring")
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True, help="Output .json path")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    result = score_video(args.video_path, args.prompt, device=f"cuda:{args.gpu}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[step_videoalign] Saved VideoAlign scores to {args.output}")


if __name__ == "__main__":
    main()
