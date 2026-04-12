#!/usr/bin/env python3
"""
vbench_eval.py — VBench 指标评估（从预计算的 vbench.json 读取）。

VBench 实际推理由 IntermediateManager 通过 conda run worker_vbench.py 完成。
本模块仅从已生成的 JSON 中提取分数。
"""

import json
from pathlib import Path
from typing import Dict, Optional


def evaluate_vbench(vbench_json: str) -> Optional[Dict]:
    """
    从 vbench.json 中提取三项指标。

    返回 {"i2v_subject": float, "i2v_background": float, "imaging_quality": float}
    """
    path = Path(vbench_json)
    if not path.exists():
        return None

    try:
        with open(str(path), "r") as f:
            data = json.load(f)
    except Exception:
        return None

    result = {}
    for key in ("i2v_subject", "i2v_background", "imaging_quality"):
        val = data.get(key)
        if val is not None:
            result[key] = float(val)
        else:
            result[key] = float("nan")

    return result
