"""reward_aggregator.py — 单 rollout reward 聚合（主进程 GPU）。

由 train_grpo_v2.py 在 Phase 2 调用：worker 把 DA3/DINOv2 npz 和
VideoAlign json 写到 /dev/shm，主进程在自己的 cuda:0 上做：
  - DINOv2 patch tokens → FeatUp 上采样到 pixel-level
  - DA3 depth + extrinsics → 做 cross-frame warping
  - 调 compute_all_rewards 算 geo_global / feature_sim / camera_rot/trans / video_quality
  - 序列化 reward.json + reward_details.json 到 work_dir
  - 立即 unlink /dev/shm 里的中间产物，释放 tmpfs

设计要点：
  - 不在主进程加载 reward 模型（DA3/DINOv2/VideoAlign），那些活全在 worker 里
  - 中间 GPU tensor 在每条 rollout 结束时 del + empty_cache，控制峰值显存
  - 单 rollout failure 不影响其它 rollout：返回 nan，外层照样能 advantage 归一化
"""

from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
_REWARD = _HERE.parent.parent / "reward"
if str(_REWARD) not in sys.path:
    sys.path.insert(0, str(_REWARD))


def aggregate_one_rollout(
    *,
    work_dir: str,
    video_path: str,
    gt_camera_txt: str,
    da3_npz: Optional[str],
    dinov2_npz: Optional[str],
    videoalign_json: Optional[str],
    rewards_to_compute: list[str],
    weights: Optional[dict] = None,
    device: str = "cuda:0",
    conf_threshold: float = 0.0,
    geo_compare_mode: str = "all_pairs",
    feature_compare_mode: str = "first_frame",
    cleanup_shm: bool = True,
) -> dict:
    """聚合一条 rollout 的 reward，返回完整 result dict（含 reward_total / details）。

    da3_npz / dinov2_npz / videoalign_json 任一为 None 或不存在 → 该项 reward = nan。
    cleanup_shm=True：聚合完立即 unlink /dev/shm 里的中间产物。
    """
    import cv2
    import torch
    from reward_metrics import compute_all_rewards  # type: ignore

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"aggregator: cannot open video {video_path}")
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    # compute_all_rewards 会按 path 是否存在自动跳过
    da3_p = str(da3_npz) if da3_npz else "/nonexistent/da3.npz"
    dino_p = str(dinov2_npz) if dinov2_npz else "/nonexistent/dinov2.npz"
    va_p = str(videoalign_json) if videoalign_json else "/nonexistent/va.json"
    fs_p = str(work / "_unused_feature_sim.json")  # 强制走新流程（npz + DA3 warping）

    try:
        result = compute_all_rewards(
            da3_path=da3_p,
            label_maps_path="/nonexistent/labels.npz",  # geo_semantic 弃用
            feature_sim_path=fs_p,
            dinov2_features_path=dino_p,
            videoalign_path=va_p,
            gt_camera_txt=str(gt_camera_txt),
            H_img=H, W_img=W,
            rewards_to_compute=rewards_to_compute,
            weights=weights,
            device=device,
            conf_threshold=conf_threshold,
            geo_compare_mode=geo_compare_mode,
            feature_compare_mode=feature_compare_mode,
        )
    except Exception as e:
        print(f"[Aggregator] compute_all_rewards FAILED for {work_dir}: {e}")
        result = {"reward_total": float("nan"), "_error": str(e)}
    finally:
        # 严格清理 GPU：FeatUp 上采样产生的 ~5GB tensor 必须立即释放
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        gc.collect()

    # 序列化
    try:
        serializable = {k: v for k, v in result.items() if k != "details"}
        with open(work / "reward.json", "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        with open(work / "reward_details.json", "w") as f:
            json.dump(result, f, indent=2, default=str)
    except Exception as e:
        print(f"[Aggregator] save reward.json failed: {e}")

    # 清理 /dev/shm 中间产物
    if cleanup_shm:
        for p in (da3_npz, dinov2_npz, videoalign_json):
            if not p:
                continue
            try:
                if str(p).startswith("/dev/shm") and os.path.isfile(p):
                    os.unlink(p)
            except Exception as e:
                print(f"[Aggregator] unlink {p} failed: {e}")

    return result
