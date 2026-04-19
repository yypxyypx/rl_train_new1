"""读取所有 reward_multimode.json，计算每个 reward 的：
   - mean / global_std
   - within_sample_std（核心，决定贡献度）
   - 估计的 w * within_std（用于权重设计）
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

OUTPUT_ROOT = Path("/horizon-bucket/robot_lab/users/puxin.yan-labs/test_output1")

# 当前生产权重
WEIGHTS = {
    "geo_semantic": 3.0,
    "geo_global":   2.0,
    "feature_sim":  4.0,
    "camera_rot":   4.0,
    "camera_trans": 4.0,
    "video_quality": 2.0,
}


def collect():
    """返回 dict: reward_name -> { sample_id -> [values across rollouts] }"""
    by_reward = {
        "geo_sem_ff": {},  "geo_sem_f3": {},  "geo_sem_ap": {},
        "geo_glob_ff": {}, "geo_glob_f3": {}, "geo_glob_ap": {},
        "feature_sim": {}, "camera_rot": {},  "camera_trans": {},
        "video_quality": {},
        "total_ff": {},   "total_f3": {},   "total_ap": {},
    }
    n_files = 0
    for ds in ["dl3dv", "re10k"]:
        for sample_dir in sorted((OUTPUT_ROOT / ds).iterdir()):
            if not sample_dir.is_dir():
                continue
            sid = f"{ds}/{sample_dir.name}"
            for gen_dir in sorted(sample_dir.glob("gen_*")):
                rj = gen_dir / "reward_multimode.json"
                if not rj.exists():
                    continue
                try:
                    d = json.load(open(rj))
                except Exception:
                    continue
                n_files += 1
                modes = d.get("modes", {})
                ff = modes.get("first_frame", {})
                f3 = modes.get("first_three", {})
                ap = modes.get("all_pairs", {})

                def push(name, val):
                    if val is None or (isinstance(val, float) and
                                       (np.isnan(val) or np.isinf(val))):
                        return
                    by_reward[name].setdefault(sid, []).append(float(val))

                push("geo_sem_ff",  ff.get("geo_semantic"))
                push("geo_sem_f3",  f3.get("geo_semantic"))
                push("geo_sem_ap",  ap.get("geo_semantic"))
                push("geo_glob_ff", ff.get("geo_global"))
                push("geo_glob_f3", f3.get("geo_global"))
                push("geo_glob_ap", ap.get("geo_global"))
                push("total_ff",    ff.get("total"))
                push("total_f3",    f3.get("total"))
                push("total_ap",    ap.get("total"))
                push("feature_sim",   d.get("feature_sim"))
                push("camera_rot",    d.get("camera_rot"))
                push("camera_trans",  d.get("camera_trans"))
                push("video_quality", d.get("video_quality"))

    print(f"[scan] read {n_files} reward_multimode.json files")
    return by_reward


def stats(values_per_sample: dict):
    """组内 std vs 整体 std。"""
    all_vals = []
    within_stds = []
    for sid, vs in values_per_sample.items():
        if len(vs) < 2:
            continue
        a = np.array(vs)
        all_vals.extend(vs)
        within_stds.append(float(a.std()))
    if not all_vals:
        return None
    a = np.array(all_vals)
    return {
        "n_samples":    len(within_stds),
        "n_rollouts":   len(all_vals),
        "mean":         float(a.mean()),
        "global_std":   float(a.std()),
        "within_std":   float(np.mean(within_stds)),
        "within_std_p50": float(np.median(within_stds)),
        "abs_mean":     float(np.mean(np.abs(a))),
    }


def main():
    data = collect()
    print()
    name_map = {
        "geo_sem_ff": "geo_semantic", "geo_sem_f3": "geo_semantic",
        "geo_sem_ap": "geo_semantic",
        "geo_glob_ff": "geo_global",  "geo_glob_f3": "geo_global",
        "geo_glob_ap": "geo_global",
        "feature_sim": "feature_sim",
        "camera_rot": "camera_rot",   "camera_trans": "camera_trans",
        "video_quality": "video_quality",
    }
    print(f"{'reward':18s} | {'mean':>8s} {'within_std':>10s} "
          f"{'glob_std':>9s} | {'w':>5s} | {'w*within':>10s}")
    print("-" * 80)
    rows = {}
    for name, vps in data.items():
        if name.startswith("total_"):
            continue
        s = stats(vps)
        if s is None:
            continue
        wname = name_map.get(name)
        w = WEIGHTS.get(wname, 1.0)
        w_within = w * s["within_std"]
        rows[name] = {**s, "w": w, "w_within": w_within}
        print(f"{name:18s} | {s['mean']:+.4f} {s['within_std']:10.5f} "
              f"{s['global_std']:9.5f} | {w:5.2f} | {w_within:10.5f}")
    print("-" * 80)
    for name in ["total_ff", "total_f3", "total_ap"]:
        s = stats(data[name])
        if s is None:
            continue
        print(f"{name:18s} | {s['mean']:+.4f} {s['within_std']:10.5f} "
              f"{s['global_std']:9.5f}")
    print()

    # 计算"权重均衡建议": 让所有 reward 的 w*within_std 都达到目标 t
    print("=" * 80)
    print("【方案 A: 完全均衡（每个 reward 的 w*within_std 都 = t）】")
    print("=" * 80)
    for t in [0.05, 0.10, 0.15]:
        print(f"\n--- 目标 w*within_std = {t} ---")
        print(f"{'reward (FF)':18s} | {'within_std':>10s} | {'new_w':>8s}")
        for name in ["geo_sem_ff", "geo_glob_ff", "feature_sim",
                     "camera_rot", "camera_trans", "video_quality"]:
            if name not in rows:
                continue
            new_w = t / max(rows[name]["within_std"], 1e-8)
            print(f"{name:18s} | {rows[name]['within_std']:10.5f} | "
                  f"{new_w:8.3f}")

    # 分层方案
    print()
    print("=" * 80)
    print("【方案 B: 分层 prior + 层内 w*within_std 均衡】")
    print("=" * 80)
    layers = {
        "主信号 (50%)":   [("camera_trans", 0.25), ("camera_rot", 0.25)],
        "辅助 (30%)":     [("geo_glob_ff", 0.15), ("feature_sim", 0.15)],
        "软约束 (20%)":   [("video_quality", 0.20)],
    }
    # 假定 |adv_total|_target 不变；scale 各 reward 让 layer 内 w*within 等于该层占比
    # 简化做法：层内每个 reward 均等占比 → 均等 w*within
    print(f"\n{'reward (FF)':18s} | {'layer':12s} | "
          f"{'target_share':>12s} | {'within_std':>10s} | {'new_w':>8s}")
    REF_TOTAL = 0.5  # 总信号目标，可调
    for layer_name, items in layers.items():
        for name, share in items:
            if name not in rows:
                continue
            target_w_within = share * REF_TOTAL
            new_w = target_w_within / max(rows[name]["within_std"], 1e-8)
            print(f"{name:18s} | {layer_name:12s} | "
                  f"{share*100:11.1f}% | "
                  f"{rows[name]['within_std']:10.5f} | {new_w:8.3f}")


if __name__ == "__main__":
    main()
