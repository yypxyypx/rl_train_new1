"""用新权重重算 reward_multimode.json 中的 total（不重跑任何 reward 模型）。

新设计：
  - 拆开 camera_rot / camera_trans（不再用合并的 camera_traj）
  - geo_semantic 弃用（weight = 0）
  - 目标 w*within_std ≈ 0.10（5 项均衡贡献）
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

OUTPUT_ROOT = Path("/horizon-bucket/robot_lab/users/puxin.yan-labs/test_output1")

NEW_WEIGHTS = {
    "geo_semantic":   0.0,    # 弃用
    "geo_global":     7.7,    # 2.0 → 7.7
    "feature_sim":    5.3,    # 5.0 → 5.3
    "camera_rot":     0.92,   # 4.0 → 0.92
    "camera_trans":   3.6,    # 4.0 → 3.6
    "video_quality":  0.67,   # 1.5 → 0.67
}


def safe(v):
    if v is None:
        return 0.0
    try:
        v = float(v)
    except Exception:
        return 0.0
    if np.isnan(v) or np.isinf(v):
        return 0.0
    return v


def main():
    n_total = 0
    n_ok = 0
    for ds in ["dl3dv", "re10k"]:
        for sample_dir in sorted((OUTPUT_ROOT / ds).iterdir()):
            if not sample_dir.is_dir():
                continue
            for gen_dir in sorted(sample_dir.glob("gen_*")):
                rj = gen_dir / "reward_multimode.json"
                if not rj.exists():
                    continue
                n_total += 1
                try:
                    d = json.load(open(rj))
                except Exception:
                    continue

                cam_rot   = safe(d.get("camera_rot"))
                cam_trans = safe(d.get("camera_trans"))
                feat_sim  = safe(d.get("feature_sim"))
                vid_qual  = safe(d.get("video_quality"))

                modes = d.get("modes", {})
                for mname in ["first_frame", "first_three", "all_pairs"]:
                    md = modes.get(mname, {}) or {}
                    geo_sem = safe(md.get("geo_semantic"))
                    geo_glob = safe(md.get("geo_global"))
                    new_total = (
                        NEW_WEIGHTS["geo_semantic"]  * geo_sem
                        + NEW_WEIGHTS["geo_global"]  * geo_glob
                        + NEW_WEIGHTS["feature_sim"] * feat_sim
                        + NEW_WEIGHTS["camera_rot"]  * cam_rot
                        + NEW_WEIGHTS["camera_trans"] * cam_trans
                        + NEW_WEIGHTS["video_quality"] * vid_qual
                    )
                    md["total"] = float(new_total)
                    modes[mname] = md
                d["modes"] = modes
                d["weights_used"] = NEW_WEIGHTS

                with open(rj, "w") as f:
                    json.dump(d, f, indent=2)
                n_ok += 1
                if n_ok % 500 == 0:
                    print(f"  updated {n_ok}/{n_total}", flush=True)

    print(f"[done] updated {n_ok}/{n_total} reward_multimode.json files")


if __name__ == "__main__":
    main()
