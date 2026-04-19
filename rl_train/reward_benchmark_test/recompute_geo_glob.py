"""只重算 geo_global（三种 mode），更新 reward_multimode.json 中：
   - modes[mode].geo_global
   - modes[mode].total（用新 geo_global + 旧 geo_sem/feat_sim/cam_traj/vid_qual）

同时收集 rel_err 量级统计写入 results/geo_glob_rel_err_stats.json，便于评估
新公式 1-clip(x,0,1)^2 vs 旧 exp(-x) 的差异。
"""
from __future__ import annotations

import json
import sys
import time
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

_THIS = Path(__file__).resolve()
_RL_TRAIN = _THIS.parents[1]
sys.path.insert(0, str(_RL_TRAIN))

from reward.reward_metrics import compute_reward_geo_global  # noqa: E402

GEO_MODES = ["first_frame", "first_three", "all_pairs"]

WEIGHTS = {
    "geo_semantic": 3.0, "geo_global": 2.0,
    "feature_sim": 4.0, "camera_traj": 4.0,
    "video_quality": 2.0,
}


def compute_total(gs, gg, fs, ct, vq):
    return (
        WEIGHTS["geo_semantic"] * gs
        + WEIGHTS["geo_global"]   * gg
        + WEIGHTS["feature_sim"]  * fs
        + WEIGHTS["camera_traj"]  * ct
        + WEIGHTS["video_quality"] * vq
    )


def get_video_size(video_path: str):
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, 0
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    return H, W


def scan_samples(root: Path):
    """Layout: root/{dl3dv,re10k}/<sample>/gen_*/"""
    out = []
    for ds_dir in sorted(root.iterdir()):
        if not ds_dir.is_dir() or ds_dir.name not in ("dl3dv", "re10k"):
            continue
        for sample_dir in sorted(ds_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            for gen_dir in sorted(sample_dir.glob("gen_*")):
                if gen_dir.is_dir():
                    out.append(gen_dir)
    return out


def process_one(gen_dir_str: str, device: str):
    gen_dir = Path(gen_dir_str)
    rm_json = gen_dir / "reward_multimode.json"
    da3_npz = gen_dir / "intermediates" / "da3_pred.npz"
    sample_dir = gen_dir.parent
    video_path = sample_dir / f"{gen_dir.name}.mp4"

    if not rm_json.exists() or not da3_npz.exists() or not video_path.exists():
        return {"status": "skip", "reason": "missing files",
                "gen_dir": gen_dir_str}

    H, W = get_video_size(str(video_path))
    if H == 0:
        return {"status": "skip", "reason": "video read fail",
                "gen_dir": gen_dir_str}

    try:
        with open(str(rm_json), "r") as f:
            old = json.load(f)
    except Exception as e:
        return {"status": "skip", "reason": f"json read fail: {e}",
                "gen_dir": gen_dir_str}

    try:
        da3_data = dict(np.load(str(da3_npz), allow_pickle=True))
    except Exception as e:
        return {"status": "skip", "reason": f"npz fail: {e}",
                "gen_dir": gen_dir_str}

    feat_sim = float(old.get("feature_sim", float("nan")))
    cam_traj = float(old.get("camera_traj", float("nan")))
    vid_qual = float(old.get("video_quality", float("nan")))
    modes = old.get("modes", {})

    err_stats_per_mode = {}
    for mode in GEO_MODES:
        try:
            r, det = compute_reward_geo_global(
                da3_data, H, W, device=device, compare_mode=mode,
            )
            geo_glob_new = r
            err_stats_per_mode[mode] = det.get("rel_err_stats", {})
        except Exception as e:
            return {"status": "err",
                    "reason": f"geo_glob {mode}: {e}",
                    "gen_dir": gen_dir_str}

        md = modes.get(mode, {}) or {}
        md["geo_global"] = geo_glob_new
        gs = float(md.get("geo_semantic", 0.0) or 0.0)
        if np.isnan(gs):
            gs = 0.0
        gg = 0.0 if np.isnan(geo_glob_new) else geo_glob_new
        fs = 0.0 if np.isnan(feat_sim) else feat_sim
        ct = 0.0 if np.isnan(cam_traj) else cam_traj
        vq = 0.0 if np.isnan(vid_qual) else vid_qual
        md["total"] = compute_total(gs, gg, fs, ct, vq)
        modes[mode] = md

    old["modes"] = modes

    try:
        with open(str(rm_json), "w") as f:
            json.dump(old, f, indent=2)
    except Exception as e:
        return {"status": "err", "reason": f"write fail: {e}",
                "gen_dir": gen_dir_str}

    return {
        "status": "ok",
        "gen_dir": gen_dir_str,
        "err_stats": err_stats_per_mode,
    }


def gpu_worker(gen_dirs: list, device: str, label: str,
               result_queue=None):
    """串行处理 gen_dirs（同一进程占用一张 GPU）。"""
    n = len(gen_dirs)
    rows = []
    for i, gd in enumerate(gen_dirs):
        res = process_one(gd, device)
        rows.append(res)
        if (i + 1) % 50 == 0 or i + 1 == n:
            ok = sum(1 for r in rows if r["status"] == "ok")
            print(f"[{label}] {i+1}/{n}  ok={ok}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--gpu_ids", default="0,1,2,3")
    ap.add_argument("--out_stats",
                    default="results/geo_glob_rel_err_stats.json")
    args = ap.parse_args()

    root = Path(args.output_root)
    gpus = [int(x) for x in args.gpu_ids.split(",")]
    n_gpus = len(gpus)

    print(f"[scan] {root} ...", flush=True)
    gen_dirs = scan_samples(root)
    print(f"[scan] 共 {len(gen_dirs)} 个 gen_dir，{n_gpus} GPU 并行", flush=True)

    # 切分到各 GPU
    shards = [[] for _ in range(n_gpus)]
    for i, gd in enumerate(gen_dirs):
        shards[i % n_gpus].append(str(gd))

    t0 = time.time()
    all_rows = []
    with ProcessPoolExecutor(max_workers=n_gpus) as ex:
        futs = {
            ex.submit(gpu_worker, shards[i], f"cuda:{gpus[i]}",
                      f"gpu{gpus[i]}"): i
            for i in range(n_gpus)
        }
        for fut in as_completed(futs):
            rows = fut.result()
            all_rows.extend(rows)

    elapsed = time.time() - t0
    n_ok = sum(1 for r in all_rows if r["status"] == "ok")
    print(f"[done] {n_ok}/{len(all_rows)} success, "
          f"elapsed={elapsed/60:.1f} min", flush=True)

    # ── 聚合 rel_err 量级统计（across all samples, all modes, all pairs）
    by_mode = {m: {k: [] for k in
                   ["mean", "median", "p25", "p75", "p90", "max",
                    "frac_lt_0p1", "frac_lt_0p3", "frac_lt_0p5",
                    "frac_gt_1p0"]} for m in GEO_MODES}
    for r in all_rows:
        if r["status"] != "ok":
            continue
        for mode, es in r.get("err_stats", {}).items():
            for k, v in es.items():
                by_mode[mode][k].append(v)

    summary = {}
    for mode in GEO_MODES:
        stats = {}
        for k, vals in by_mode[mode].items():
            if vals:
                stats[k] = {
                    "mean": float(np.mean(vals)),
                    "p50":  float(np.percentile(vals, 50)),
                    "p90":  float(np.percentile(vals, 90)),
                    "n_samples": len(vals),
                }
        summary[mode] = stats

    out_path = Path(args.out_stats)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_path), "w") as f:
        json.dump({
            "n_processed": n_ok,
            "score_formula": "1 - clip(rel_err, 0, 1)^2",
            "by_mode": summary,
        }, f, indent=2)
    print(f"[save] err 量级统计 → {out_path}", flush=True)


if __name__ == "__main__":
    main()
