"""reward_bridge.py — Reward 桥接层。

调用 RL_code/rl_train/reward/reward_pipeline.py 的 run_pipeline，
对每条 rollout 视频逐条计算 reward，返回标量列表。

接口设计：
    compute_rewards_for_rollouts(
        video_paths, camera_txt_path, prompt, output_dir,
        rewards, weights, gpu_id, keep_intermediates, skip_done
    ) -> list[dict]

--dry_run 模式：不调用 reward pipeline，返回随机 reward，
               用于快速调通训练流程而不等待慢速 reward 计算。
"""

from __future__ import annotations

import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

# ─── 将 reward 代码加入路径 ───────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_RL_CODE = _HERE.parent.parent.parent  # RL_code/
_REWARD_DIR = _RL_CODE / "rl_train" / "reward"

if str(_REWARD_DIR) not in sys.path:
    sys.path.insert(0, str(_REWARD_DIR))


# ══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════════════

def _parse_weights(weights_str: str | None) -> dict | None:
    """解析 'key:value,key:value' 格式的权重字符串。"""
    if not weights_str:
        return None
    result = {}
    for part in weights_str.split(","):
        part = part.strip()
        if ":" in part:
            k, v = part.split(":", 1)
            result[k.strip()] = float(v.strip())
    return result if result else None


def _cleanup_intermediates(work_dir: str) -> None:
    """删除中间值目录（frames/ 和 intermediates/）。"""
    for sub in ("frames", "intermediates"):
        d = Path(work_dir) / sub
        if d.exists():
            shutil.rmtree(str(d))


# ══════════════════════════════════════════════════════════════════════════════
# 核心接口
# ══════════════════════════════════════════════════════════════════════════════

def compute_rewards_for_rollouts(
    video_paths: List[str],
    camera_txt_path: str,
    prompt: str,
    output_dir: str,
    rewards: List[str] | str = "all",
    weights: Dict | str | None = None,
    gpu_id: int = 0,
    keep_intermediates: bool = True,
    skip_done: bool = True,
    dry_run: bool = False,
) -> List[Dict]:
    """批量计算多条 rollout 视频的 reward。

    对每条 video_path 调用 run_pipeline，逐条处理。

    Args:
        video_paths         : N 条 rollout 视频路径 [gen_0.mp4, ...]
        camera_txt_path     : GT camera.txt 路径（直接传给 reward pipeline）
        prompt              : 文本 prompt（VideoAlign 使用）
        output_dir          : reward 中间值和结果输出根目录
        rewards             : 消融配置，"all" 或列表 ["camera_traj", "feature_sim"]
        weights             : reward 权重覆盖，dict 或 "key:v,key:v" 字符串
        gpu_id              : 使用的 GPU 编号
        keep_intermediates  : False 时计算完删除 frames/ 和 intermediates/
        skip_done           : True 时跳过已有的中间值（断点续算）
        dry_run             : True 时不实际计算，返回随机 reward（调试用）

    Returns:
        list of dicts，每条 rollout 对应一个 dict，包含：
            reward_total, reward_geo_semantic, reward_geo_global,
            reward_feature_sim, reward_camera_traj, reward_video_quality,
            gen_id, video_path
    """
    if dry_run:
        return _dry_run_rewards(video_paths)

    # 解析 rewards 列表
    if isinstance(rewards, str):
        if rewards == "all":
            from reward_pipeline import ALL_REWARDS
            rewards_list = ALL_REWARDS
        else:
            rewards_list = [r.strip() for r in rewards.split(",") if r.strip()]
    else:
        rewards_list = list(rewards)

    # 解析权重
    if isinstance(weights, str):
        weights = _parse_weights(weights)

    from reward_pipeline import run_pipeline  # noqa: E402

    results = []
    for k, video_path in enumerate(video_paths):
        work_dir = os.path.join(output_dir, f"gen_{k}")
        os.makedirs(work_dir, exist_ok=True)

        print(f"[Reward] Computing reward for gen_{k}: {video_path}")
        try:
            result = run_pipeline(
                video_path=video_path,
                gt_camera_txt=camera_txt_path,
                work_dir=work_dir,
                rewards=rewards_list,
                gpu=gpu_id,
                prompt=prompt if prompt else None,
                metadata_json=None,
                skip_done=skip_done,
            )
        except Exception as e:
            print(f"[Reward] ERROR for gen_{k}: {e}")
            result = {
                "reward_total": 0.0,
                "reward_geo_semantic": float("nan"),
                "reward_geo_global": float("nan"),
                "reward_feature_sim": float("nan"),
                "reward_camera_traj": float("nan"),
                "reward_video_quality": float("nan"),
            }

        # 如果 weights 不为 None，重新计算 reward_total
        if weights is not None:
            total = 0.0
            for rname, w in weights.items():
                key = f"reward_{rname}"
                val = result.get(key, float("nan"))
                if val == val:  # not nan
                    total += w * val
            result["reward_total"] = total

        # 附加元信息
        result["gen_id"] = k
        result["video_path"] = video_path

        results.append(result)

        if not keep_intermediates:
            _cleanup_intermediates(work_dir)

        print(f"[Reward] gen_{k}: reward_total={result.get('reward_total', float('nan')):.4f}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# dry_run 模式
# ══════════════════════════════════════════════════════════════════════════════

def _dry_run_rewards(video_paths: List[str]) -> List[Dict]:
    """返回随机 reward（用于调通训练流程）。"""
    print(f"[Reward] DRY RUN: returning random rewards for {len(video_paths)} rollouts")
    results = []
    for k, vp in enumerate(video_paths):
        r = random.gauss(0.0, 1.0)
        results.append({
            "gen_id": k,
            "video_path": vp,
            "reward_total": r,
            "reward_geo_semantic": float("nan"),
            "reward_geo_global": float("nan"),
            "reward_feature_sim": float("nan"),
            "reward_camera_traj": float("nan"),
            "reward_video_quality": float("nan"),
        })
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 从 reward 结果提取标量 tensor
# ══════════════════════════════════════════════════════════════════════════════

def extract_reward_scalars(reward_results: List[Dict], device) -> List[float]:
    """从 reward_results 列表提取 reward_total 标量。

    NaN 替换为 0.0，并打印警告。
    """
    scalars = []
    for res in reward_results:
        val = res.get("reward_total", float("nan"))
        if val != val:  # nan check
            print(f"[Reward] WARNING: reward_total is nan for gen_{res.get('gen_id', '?')}, using 0.0")
            val = 0.0
        scalars.append(float(val))
    return scalars


# ══════════════════════════════════════════════════════════════════════════════
# 将每 step 的 reward 写入日志
# ══════════════════════════════════════════════════════════════════════════════

def log_rewards(
    reward_results: List[Dict],
    step: int,
    log_path: str,
    rank: int = 0,
) -> None:
    """将 reward 结果追加写入 JSON Lines 文件（仅 rank 0）。"""
    if rank != 0:
        return
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    with open(log_path, "a") as f:
        record = {
            "step": step,
            "rollouts": [
                {k: v for k, v in r.items() if k != "details"}
                for r in reward_results
            ],
        }
        f.write(json.dumps(record, default=str) + "\n")
