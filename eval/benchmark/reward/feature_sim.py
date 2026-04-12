#!/usr/bin/env python3
"""
feature_sim.py — 从预计算的 feature_sim_reward.json 提取 DINOv2 特征相似度。
"""

import json
from pathlib import Path
from typing import Dict, Optional


def evaluate_feature_sim(feature_sim_json: str) -> Optional[Dict]:
    """
    读取预计算的 feature_sim_reward.json。

    返回: {"reward_feature_sim": float, "mode": str, "details": dict}
    """
    path = Path(feature_sim_json)
    if not path.exists():
        return None

    try:
        with open(str(path), "r") as f:
            data = json.load(f)
    except Exception:
        return None

    return {
        "reward_feature_sim": float(data.get("reward_feature_sim", 0.0)),
        "mode": data.get("mode", "unknown"),
        "mean_dissimilarity": data.get("details", {}).get("mean_dissimilarity", float("nan")),
    }
