#!/usr/bin/env python3
"""
run_reward_multimode.py
=======================
Phase 2 (multi-mode): 用三种 geo 帧对比模式重算 geo_semantic / geo_global，
其余 reward（feature_sim / camera_traj / video_quality）各算一次。

三种 geo_compare_mode：
  first_frame  — 每帧 vs 第 0 帧
  first_three  — 每帧 vs 前三帧（ref=0,1,2，有效像素数为分母）
  all_pairs    — 所有无序帧对

输出 gen_k/reward_multimode.json:
{
  "modes": {
    "first_frame":  {"geo_semantic": 0.53, "geo_global": 0.89, "total": -32.5},
    "first_three":  {"geo_semantic": 0.51, "geo_global": 0.87, "total": -33.1},
    "all_pairs":    {"geo_semantic": 0.48, "geo_global": 0.85, "total": -34.2}
  },
  "feature_sim":  0.59,
  "camera_traj":  -4.0,
  "video_quality": -3.3
}

多 GPU 并行：
  --gpu_ids 1,2,3,5   手动指定 GPU
  主进程构建 entry list，按 GPU 数均分，每个 shard 作为子进程运行。
  子进程通过 --shard_idx 和 --n_shards 参数识别自己的分片。

用法（主进程）:
    python run_reward_multimode.py \\
        --output_root /horizon-bucket/robot_lab/users/puxin.yan-labs/test_output1 \\
        --gpu_ids 0,1,2,3 [--force]
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_RL_CODE_DIR = _THIS_DIR.parent.parent
_BENCHMARK_DIR = _RL_CODE_DIR / "eval" / "benchmark"
_REWARD_DIR = _RL_CODE_DIR / "rl_train" / "reward"

sys.path.insert(0, str(_BENCHMARK_DIR))
sys.path.insert(0, str(_REWARD_DIR))

from common.scan import scan_output_root
from common.utils import log
from reward_metrics import (
    compute_reward_geo_semantic,
    compute_reward_geo_global,
    compute_reward_camera_traj,
    compute_reward_video_quality,
    parse_camera_txt,
)

GEO_MODES = ["first_frame", "first_three", "all_pairs"]
GEO_MODE_LOG = {"first_frame": "FF", "first_three": "F3", "all_pairs": "AP"}

# reward_total 权重（与训练一致）
WEIGHTS = {
    "geo_semantic": 3.0,
    "geo_global":   2.0,
    "feature_sim":  5.0,
    "camera_traj":  8.0,
    "video_quality": 1.5,
}


def get_video_size(video_path: str):
    cap = cv2.VideoCapture(str(video_path))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    return H, W


def compute_total(geo_sem: float, geo_glob: float, feat_sim: float,
                  cam_traj: float, vid_qual: float) -> float:
    return (
        WEIGHTS["geo_semantic"]  * geo_sem
        + WEIGHTS["geo_global"]  * geo_glob
        + WEIGHTS["feature_sim"] * feat_sim
        + WEIGHTS["camera_traj"] * cam_traj
        + WEIGHTS["video_quality"] * vid_qual
    )


def recompute_one(entry: dict, gv: dict, device: str,
                  only_camera: bool = False) -> dict | None:
    """计算单个 gen 视频的多模式 reward，返回 reward_multimode dict 或 None。

    only_camera=True 时：只重算 camera_rot / camera_trans / camera_traj，
    其他字段（geo_sem/glob/feature_sim/video_quality）从已有 reward_multimode.json
    读取并 merge；total 用新 camera_traj + 旧 geo/feat/vq 重算以保持自洽。
    """
    inter_dir = gv["gen_dir"] / "intermediates"
    da3_npz = inter_dir / "da3_pred.npz"
    label_maps_npz = inter_dir / "label_maps.npz"
    feature_sim_json = inter_dir / "feature_sim_reward.json"
    videoalign_json = inter_dir / "videoalign.json"
    camera_txt = entry.get("camera_txt") or ""

    if not da3_npz.exists():
        log(f"  [SKIP] da3_pred.npz 不存在: {gv['video_path']}")
        return None

    # ── only_camera: 仅算 camera_traj，其他字段从已有 json merge ─
    if only_camera:
        old_json = gv["gen_dir"] / "reward_multimode.json"
        if not old_json.exists():
            log(f"  [SKIP only_camera] 未找到已有 reward_multimode.json: "
                f"{gv['video_path']}")
            return None
        try:
            with open(str(old_json), "r") as f:
                old = json.load(f)
        except Exception as e:
            log(f"  [SKIP only_camera] 读取 reward_multimode.json 失败: {e}")
            return None

        # 只读 npz 里的 extrinsics，避免载入整个 ~80MB 文件
        try:
            npz_h = np.load(str(da3_npz), allow_pickle=True)
            da3_ext = {"extrinsics": npz_h["extrinsics"]}
        except Exception as e:
            log(f"  [WARN only_camera] 读取 extrinsics 失败: {e}")
            return None

        cam_traj = float("nan")
        cam_rot = float("nan")
        cam_trans = float("nan")
        if camera_txt:
            try:
                # parse_camera_txt 需要 H, W 做 intrinsic 归一化反推，
                # 但 camera_traj 只用 w2c 做刚体变换（w2c 不依赖 H, W），
                # 所以传任意正值即可。
                _, gt_w2c = parse_camera_txt(camera_txt, 1, 1)
                r, ct_details = compute_reward_camera_traj(da3_ext, gt_w2c)
                cam_traj = r
                cam_rot = float(ct_details.get("rot_reward", float("nan")))
                cam_trans = float(ct_details.get("trans_reward", float("nan")))
            except Exception as e:
                log(f"  [WARN only_camera] camera_traj FAILED: {e}")

        # 更新 shared rewards 部分，其余字段保留
        feat_sim = float(old.get("feature_sim", float("nan")))
        vid_qual = float(old.get("video_quality", float("nan")))
        modes_result = old.get("modes", {})

        # 用新 camera_traj + 旧其它量重算每种 mode 的 total
        for mode in GEO_MODES:
            md = modes_result.get(mode, {}) or {}
            geo_sem_v = float(md.get("geo_semantic", float("nan")))
            geo_glob_v = float(md.get("geo_global", float("nan")))
            gs = 0.0 if np.isnan(geo_sem_v) else geo_sem_v
            gg = 0.0 if np.isnan(geo_glob_v) else geo_glob_v
            fs = 0.0 if np.isnan(feat_sim) else feat_sim
            ct = 0.0 if np.isnan(cam_traj) else cam_traj
            vq = 0.0 if np.isnan(vid_qual) else vid_qual
            md["geo_semantic"] = geo_sem_v
            md["geo_global"]   = geo_glob_v
            md["total"] = compute_total(gs, gg, fs, ct, vq)
            modes_result[mode] = md

        return {
            "modes":         modes_result,
            "feature_sim":   feat_sim,
            "camera_traj":   cam_traj,
            "camera_rot":    cam_rot,
            "camera_trans":  cam_trans,
            "video_quality": vid_qual,
        }

    H, W = get_video_size(gv["video_path"])
    if H == 0 or W == 0:
        log(f"  [SKIP] 无法读取视频尺寸: {gv['video_path']}")
        return None

    da3_data = dict(np.load(str(da3_npz), allow_pickle=True))

    # ── label_maps（geo_semantic 需要）─────────────────────────────
    label_maps = None
    if label_maps_npz.exists():
        try:
            lm = np.load(str(label_maps_npz), allow_pickle=True)
            label_maps = lm["label_maps"]
        except Exception as e:
            log(f"  [WARN] label_maps 加载失败: {e}")

    # ── 三种 geo 模式 ────────────────────────────────────────────────
    modes_result = {}
    for mode in GEO_MODES:
        geo_sem = float("nan")
        geo_glob = float("nan")

        if label_maps is not None:
            try:
                r, _ = compute_reward_geo_semantic(
                    da3_data, label_maps, H, W,
                    device=device, compare_mode=mode,
                )
                geo_sem = r
            except Exception as e:
                log(f"  [WARN] geo_semantic ({mode}) FAILED: {e}")
        else:
            log(f"  [WARN] label_maps 不存在，geo_semantic ({mode}) 跳过")

        try:
            r, _ = compute_reward_geo_global(
                da3_data, H, W,
                device=device, compare_mode=mode,
            )
            geo_glob = r
        except Exception as e:
            log(f"  [WARN] geo_global ({mode}) FAILED: {e}")

        modes_result[mode] = {
            "geo_semantic": geo_sem,
            "geo_global":   geo_glob,
        }

    # ── feature_sim（读已有 JSON）───────────────────────────────────
    feat_sim = float("nan")
    if feature_sim_json.exists():
        try:
            with open(str(feature_sim_json), "r") as f:
                fs = json.load(f)
            feat_sim = float(fs["reward_feature_sim"])
        except Exception as e:
            log(f"  [WARN] feature_sim 读取失败: {e}")

    # ── camera_traj（同时保存 rot / trans 分量，便于 Phase 4 分析）─
    cam_traj = float("nan")
    cam_rot = float("nan")
    cam_trans = float("nan")
    if camera_txt:
        try:
            _, gt_w2c = parse_camera_txt(camera_txt, H, W)
            r, ct_details = compute_reward_camera_traj(da3_data, gt_w2c)
            cam_traj = r
            cam_rot = float(ct_details.get("rot_reward", float("nan")))
            cam_trans = float(ct_details.get("trans_reward", float("nan")))
        except Exception as e:
            log(f"  [WARN] camera_traj FAILED: {e}")

    # ── video_quality（读 videoalign.json）─────────────────────────
    vid_qual = float("nan")
    if videoalign_json.exists():
        try:
            with open(str(videoalign_json), "r") as f:
                va = json.load(f)
            r, _ = compute_reward_video_quality(va)
            vid_qual = r
        except Exception as e:
            log(f"  [WARN] video_quality FAILED: {e}")

    # ── 每种 geo 模式的 total ────────────────────────────────────────
    for mode in GEO_MODES:
        md = modes_result[mode]
        geo_sem_v = md["geo_semantic"]
        geo_glob_v = md["geo_global"]
        if any(np.isnan(v) for v in [geo_sem_v, geo_glob_v, feat_sim,
                                      cam_traj, vid_qual]):
            # 用能算的算，nan 当 0 处理（与训练期间跳过异常项一致）
            gs = 0.0 if np.isnan(geo_sem_v) else geo_sem_v
            gg = 0.0 if np.isnan(geo_glob_v) else geo_glob_v
            fs = 0.0 if np.isnan(feat_sim) else feat_sim
            ct = 0.0 if np.isnan(cam_traj) else cam_traj
            vq = 0.0 if np.isnan(vid_qual) else vid_qual
            total = compute_total(gs, gg, fs, ct, vq)
        else:
            total = compute_total(geo_sem_v, geo_glob_v, feat_sim,
                                  cam_traj, vid_qual)
        md["total"] = total

    return {
        "modes":        modes_result,
        "feature_sim":  feat_sim,
        "camera_traj":  cam_traj,
        "camera_rot":   cam_rot,
        "camera_trans": cam_trans,
        "video_quality": vid_qual,
    }


# ═══════════════════════════════════════════════════════════════════
# 子进程 worker（通过 --shard_idx / --n_shards 区分分片）
# ═══════════════════════════════════════════════════════════════════

def run_shard(entries: list, device: str, force: bool, shard_label: str,
              only_camera: bool = False):
    """在当前进程内处理 entries 子集。"""
    total = ok = err = skip = 0
    for entry in entries:
        for gv in entry["gen_videos"]:
            total += 1
            out_json = gv["gen_dir"] / "reward_multimode.json"
            # only_camera 模式永远覆盖（否则没意义）；非 only_camera 沿用 force 逻辑
            if not only_camera and out_json.exists() and not force:
                skip += 1
                continue
            try:
                result = recompute_one(entry, gv, device,
                                        only_camera=only_camera)
                if result is None:
                    err += 1
                    continue
                with open(str(out_json), "w") as f:
                    json.dump(result, f, ensure_ascii=False,
                              indent=2, default=str)
                ok += 1
                # log progress per-mode total
                totals = " | ".join(
                    f"{GEO_MODE_LOG[m]}={result['modes'][m]['total']:.4f}"
                    for m in GEO_MODES
                )
                log(f"  [{shard_label}] [{ok}/{total - skip}] "
                    f"{entry['sample_id']}/gen_{gv['idx']}: {totals}")
            except Exception as e:
                import traceback
                log(f"  [ERROR] {gv['video_path']}: {e}")
                traceback.print_exc()
                err += 1

    log(f"[{shard_label}] 完成: total={total} ok={ok} "
        f"err={err} skip={skip}")


# ═══════════════════════════════════════════════════════════════════
# 主进程：分片 + 启动子进程
# ═══════════════════════════════════════════════════════════════════

def split_entries(entries: list, n: int) -> list:
    """将 entries 均分为 n 份（按条目循环分配）。"""
    shards = [[] for _ in range(n)]
    for i, entry in enumerate(entries):
        shards[i % n].append(entry)
    return shards


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 multi-mode: 三种 geo 对比模式重算 reward")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--device", default="cuda:0",
                        help="单进程时使用的设备（多进程时由 gpu_ids 决定）")
    parser.add_argument("--gpu_ids", type=str, default=None,
                        help="手动指定 GPU ID，逗号分隔，如 '0,1,2,3'")
    parser.add_argument("--shards_per_gpu", type=int, default=1,
                        help="每块 GPU 启动的子进程数（I/O 密集场景下可>1）")
    parser.add_argument("--force", action="store_true",
                        help="强制重算（即使 reward_multimode.json 已存在）")
    parser.add_argument("--only_camera", action="store_true",
                        help="只重算 camera_traj / camera_rot / camera_trans，"
                             "其它字段从已有 reward_multimode.json 读取并 merge。"
                             "速度很快（纯 numpy + 小文件 I/O），默认就覆盖已有 json。")
    # 子进程专用参数（主进程不传）
    parser.add_argument("--shard_idx", type=int, default=None)
    parser.add_argument("--n_shards",  type=int, default=None)
    parser.add_argument("--shard_device", type=str, default=None)
    parser.add_argument("--shard_only_camera", action="store_true",
                        help="子进程专用: 对齐主进程的 --only_camera")
    args = parser.parse_args()

    output_root = Path(args.output_root)

    # ── 子进程模式 ─────────────────────────────────────────────────
    if args.shard_idx is not None:
        entries = scan_output_root(str(output_root))
        shards = split_entries(entries, args.n_shards)
        my_entries = shards[args.shard_idx]
        device = args.shard_device or "cuda:0"
        label = f"shard{args.shard_idx}@{device}"
        log(f"[{label}] 处理 {len(my_entries)} 个样本，设备 {device}"
            + (", only_camera" if args.shard_only_camera else ""))
        run_shard(my_entries, device, args.force, label,
                  only_camera=args.shard_only_camera)
        return

    # ── 主进程模式 ─────────────────────────────────────────────────
    log(f"[Phase 2 multimode] 扫描 {output_root} ...")
    entries = scan_output_root(str(output_root))
    log(f"找到 {len(entries)} 个样本")

    if args.gpu_ids:
        gpu_ids = [g.strip() for g in args.gpu_ids.split(",")]
    else:
        gpu_ids = [args.device.replace("cuda:", "")]
        gpu_ids = [gpu_ids[0]] if gpu_ids[0].isdigit() else ["0"]

    n_gpus = len(gpu_ids)
    shards_per_gpu = max(1, int(args.shards_per_gpu))
    n_shards = n_gpus * shards_per_gpu
    log(f"[Phase 2 multimode] 使用 {n_gpus} 个 GPU: {gpu_ids}，"
        f"每卡 {shards_per_gpu} shard，合计 {n_shards} 子进程")

    if n_shards == 1:
        # 单进程直接跑
        run_shard(entries, f"cuda:{gpu_ids[0]}", args.force,
                  f"gpu{gpu_ids[0]}", only_camera=args.only_camera)
        return

    # 多进程：shard round-robin 到 GPU
    procs = []
    for i in range(n_shards):
        gid = gpu_ids[i % n_gpus]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gid)
        cmd = [
            sys.executable, "-u", __file__,
            "--output_root", str(output_root),
            "--shard_idx",   str(i),
            "--n_shards",    str(n_shards),
            "--shard_device", "cuda:0",   # CUDA_VISIBLE_DEVICES 已映射
        ]
        if args.force:
            cmd.append("--force")
        if args.only_camera:
            cmd.append("--shard_only_camera")
        log(f"  启动子进程 shard {i} on GPU {gid}: {' '.join(cmd[-6:])}")
        p = subprocess.Popen(cmd, env=env)
        procs.append(p)

    # 等待所有子进程完成
    exit_codes = []
    for i, p in enumerate(procs):
        code = p.wait()
        exit_codes.append(code)
        if code != 0:
            log(f"  [WARN] shard {i} 退出码 {code}")

    if all(c == 0 for c in exit_codes):
        log("[Phase 2 multimode] 所有 shard 完成")
    else:
        log(f"[Phase 2 multimode] 部分 shard 失败: {exit_codes}")
        sys.exit(1)


if __name__ == "__main__":
    main()
