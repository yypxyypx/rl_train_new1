#!/usr/bin/env python3
"""
videoalign_eval.py — 从预计算的 videoalign.json 提取 VideoAlign 评分。

复用 RL/eval/reward_eval/reward_metrics.py 的 compute_reward_video_quality 逻辑。
"""

import json
from pathlib import Path
from typing import Dict, Optional


def evaluate_videoalign(videoalign_json: str) -> Optional[Dict]:
    """
    从 videoalign.json 提取 Overall / VQ / MQ / TA 分数。

    返回: {"Overall": float, "VQ": float, "MQ": float, "TA": float}
    """
    path = Path(videoalign_json)
    if not path.exists():
        return None

    try:
        with open(str(path), "r") as f:
            data = json.load(f)
    except Exception:
        return None

    return {
        "Overall": float(data.get("Overall", 0.0)),
        "VQ": float(data.get("VQ", 0.0)),
        "MQ": float(data.get("MQ", 0.0)),
        "TA": float(data.get("TA", 0.0)),
    }
