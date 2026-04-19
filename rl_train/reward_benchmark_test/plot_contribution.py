#!/usr/bin/env python3
"""
plot_contribution.py
====================
Plot reward contribution to Total advantage.

For each dataset (re10k / dl3dv / combined), render a grouped bar chart where:
  - x-axis: reward item (GeoSem, GeoGlob, FeatSim, CamRot, CamTrans, VidQual)
  - 3 bars per item: contribution to Total/FF, Total/F3, Total/AP
  - value = mean(|w_r * adv_r|) / mean(|adv_total|)

Notes:
- GeoSem/F3/AP are "mode-bound" (only contribute to their matching Total),
  so we aggregate them into a single "GeoSem" row where each bar corresponds
  to its own mode's Total (geo_sem_ff → Total/FF, etc.).
- FeatSim, CamRot, CamTrans, VidQual are shared across all 3 Totals, so each
  bar is their contribution to the respective Total.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Reward rows to show ────────────────────────────────────────────
# Each row: (display label, mapping from mode-index {0:FF, 1:F3, 2:AP} -> col name)
MODES = ["ff", "f3", "ap"]
MODE_LBL = ["FF", "F3", "AP"]
TOTAL_KEYS = [f"total_{m}" for m in MODES]

# For "mode-bound" rewards: the same reward has 3 variants (ff/f3/ap),
# each tied to its own Total. Each bar uses the variant matching the x-mode.
# For "shared" rewards: same col for all bars, but contribution is read under
# each Total's key.
ROWS = [
    ("GeoSem",  "mode_bound",  [f"geo_sem_{m}"  for m in MODES], 3.0),
    ("GeoGlob", "mode_bound",  [f"geo_glob_{m}" for m in MODES], 2.0),
    ("FeatSim", "shared",      ["feature_sim"] * 3,              5.0),
    ("CamRot",  "shared",      ["camera_rot"]  * 3,              4.0),
    ("CamTrans","shared",      ["camera_trans"] * 3,             4.0),
    ("VidQual", "shared",      ["video_quality"] * 3,            1.5),
]

COLORS = {
    "FF": "#c94b4b",  # deep red
    "F3": "#eaa84a",  # orange
    "AP": "#4a7fc9",  # blue
}


def load_json(p: Path) -> dict:
    with open(p) as f:
        return json.load(f)


def build_contrib_matrix(contrib: dict) -> tuple[np.ndarray, list[str]]:
    """
    Build (n_rows, 3) matrix.
      M[i, j] = contribution of ROWS[i] to total_{MODES[j]}.
    """
    M = np.full((len(ROWS), 3), np.nan)
    labels = []
    for i, (lbl, kind, cols, _w) in enumerate(ROWS):
        labels.append(lbl)
        for j, tk in enumerate(TOTAL_KEYS):
            col = cols[j]
            entry = contrib.get(col, {})
            v = entry.get(tk, None)
            M[i, j] = float(v) if v is not None else np.nan
    return M, labels


def plot_one(M: np.ndarray, labels: list[str], n_samples: int,
             n_records: int, ds_label: str, out_path: Path):
    n_rows = len(labels)
    x = np.arange(n_rows)
    bar_w = 0.26

    fig, ax = plt.subplots(figsize=(11, 5.2))
    # Bars for each mode
    for j, mode_lbl in enumerate(MODE_LBL):
        ax.bar(
            x + (j - 1) * bar_w, M[:, j],
            width=bar_w, label=f"Total/{mode_lbl}",
            color=COLORS[mode_lbl], edgecolor="black", linewidth=0.5,
        )
        # Value labels
        for i in range(n_rows):
            v = M[i, j]
            if np.isnan(v):
                continue
            ax.text(x[i] + (j - 1) * bar_w, v + 0.005, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8,
                    color=COLORS[mode_lbl], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(r"Contribution $= \dfrac{\mathrm{mean}|w_r \cdot \mathrm{adv}_r|}{\mathrm{mean}|\mathrm{adv}_{\mathrm{total}}|}$",
                  fontsize=10)
    top = float(np.nanmax(M)) if np.isfinite(M).any() else 1.0
    ax.set_ylim(0, top * 1.18 + 0.02)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_title(
        f"Reward Contribution to Total advantage — {ds_label}  "
        f"[{n_samples} samples, {n_records} rollouts]\n"
        f"Weights: GeoSem=3  GeoGlob=2  FeatSim=5  CamRot=4  CamTrans=4  VidQual=1.5  "
        f"(CamTraj=8 split evenly into Rot+Trans)",
        fontsize=10, pad=10,
    )
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir",
                    default="/home/users/puxin.yan-labs/RL_code/rl_train/reward_benchmark_test/results")
    args = ap.parse_args()

    results = Path(args.results_dir)
    json_dir = results / "json"

    for ds in ["re10k", "dl3dv", "combined"]:
        p = json_dir / f"correlation_{ds}.json"
        if not p.exists():
            print(f"[skip] {p} not found")
            continue
        d = load_json(p)
        contrib = d["contrib"]
        n_records = d.get("n_records", 0)

        # number of samples = distinct (dataset, sample_id) — already embedded
        # into n_records/8 approximately; but we can just report rollouts.
        # For a more accurate count, we recover it from the table's p-values N.
        # Fallback to n_records // 8
        n_samples = n_records // 8

        M, labels = build_contrib_matrix(contrib)
        out = results / f"contribution_{ds}.png"
        plot_one(M, labels, n_samples, n_records, ds, out)


if __name__ == "__main__":
    main()
