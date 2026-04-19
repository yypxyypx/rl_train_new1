#!/usr/bin/env python3
"""
run_reward_recompute.py
=======================
Phase 2: 使用新的 SAM label_maps 重新计算所有 gen 视频的 reward。

依赖:
  - gen_k/intermediates/da3_pred.npz         (已有)
  - gen_k/intermediates/label_maps.npz       (SAM3 重新生成)
  - gen_k/intermediates/feature_sim_reward.json  (已有)
  - gen_k/intermediates/videoalign.json      (已有)
  - sample/camera.txt                        (已有)

输出:
  - gen_k/reward.json         (标量 reward，覆盖旧文件)
  - gen_k/reward_details.json (含 details)

用法:
    python run_reward_recompute.py \
        --output_root /horizon-bucket/robot_lab/users/puxin.yan-labs/test_output1 \
        [--device cuda:0] [--force]
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

_THIS_DIR = Path(__file__).resolve().parent
_RL_CODE_DIR = _THIS_DIR.parent.parent
_BENCHMARK_DIR = _RL_CODE_DIR / "eval" / "benchmark"
_REWARD_DIR = _RL_CODE_DIR / "rl_train" / "reward"

sys.path.insert(0, str(_BENCHMARK_DIR))
sys.path.insert(0, str(_REWARD_DIR))

from common.scan import scan_output_root
from common.utils import log
from reward_metrics import compute_all_rewards


def get_video_size(video_path: str):
    cap = cv2.VideoCapture(str(video_path))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    return H, W


def recompute_one(entry: dict, gv: dict, device: str) -> dict | None:
    """重新计算单个 gen 视频的所有 reward。"""
    inter_dir = gv["gen_dir"] / "intermediates"
    da3_npz = inter_dir / "da3_pred.npz"
    label_maps_npz = inter_dir / "label_maps.npz"
    feature_sim_json = inter_dir / "feature_sim_reward.json"
    videoalign_json = inter_dir / "videoalign.json"
    camera_txt = entry.get("camera_txt") or ""

    if not da3_npz.exists():
        log(f"  [SKIP] da3_pred.npz 不存在: {gv['video_path']}")
        return None

    H, W = get_video_size(gv["video_path"])
    if H == 0 or W == 0:
        log(f"  [SKIP] 无法读取视频尺寸: {gv['video_path']}")
        return None

    result = compute_all_rewards(
        da3_path=str(da3_npz),
        label_maps_path=str(label_maps_npz) if label_maps_npz.exists() else "",
        feature_sim_path=str(feature_sim_json) if feature_sim_json.exists() else "",
        videoalign_path=str(videoalign_json) if videoalign_json.exists() else "",
        gt_camera_txt=str(camera_txt),
        H_img=H,
        W_img=W,
        rewards_to_compute=None,   # 计算全部
        device=device,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Recompute rewards")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true",
                        help="强制重算（即使 reward.json 已存在）")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    log(f"[Phase 2] 扫描 {output_root} ...")
    entries = scan_output_root(str(output_root))
    log(f"找到 {len(entries)} 个样本，使用设备: {args.device}")

    total = ok = err = skip = 0
    for entry in entries:
        for gv in entry["gen_videos"]:
            total += 1
            reward_json = gv["gen_dir"] / "reward.json"

            if reward_json.exists() and not args.force:
                skip += 1
                continue

            try:
                result = recompute_one(entry, gv, args.device)
                if result is None:
                    err += 1
                    continue

                # 保存 reward.json（不含 details）
                scalars = {k: v for k, v in result.items()
                           if k != "details"}
                with open(str(reward_json), "w") as f:
                    json.dump(scalars, f, ensure_ascii=False,
                              indent=2, default=str)

                # 保存 reward_details.json（含 details）
                details_json = gv["gen_dir"] / "reward_details.json"
                with open(str(details_json), "w") as f:
                    json.dump(result, f, ensure_ascii=False,
                              indent=2, default=str)

                ok += 1
                rt = result.get("reward_total", float("nan"))
                log(f"  [{ok}/{total - skip}] "
                    f"{entry['sample_id']}/gen_{gv['idx']}: "
                    f"reward_total={rt:.4f}")

            except Exception as e:
                import traceback
                log(f"  [ERROR] {gv['video_path']}: {e}")
                traceback.print_exc()
                err += 1

    log(f"[Phase 2] 完成: total={total} ok={ok} err={err} skip={skip}")


if __name__ == "__main__":
    main()
