#!/usr/bin/env python3
"""
diagnose_geo_sem.py — 揭示 geo_semantic vs geo_global 的方差来源。

挑几个 DL3DV sample × 8 rollouts，重算两个 reward 的全 details，
- 比较 within-sample std(geo_sem) vs std(geo_glob)
- 输出 valid_pixels, label_match_rate, fg_match_rate, per-frame score 的分布
- 对每个 sample 的"最差 rollout"输出一对 (src=mid, ref=0) 的标签直方图

跑在 GPU 上，每个 rollout 仅 first_frame mode（最快，N-1 对 warp）。
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
import cv2

_THIS = Path(__file__).resolve()
_RL_TRAIN = _THIS.parents[1]
sys.path.insert(0, str(_RL_TRAIN))

from reward.reward_metrics import (  # noqa: E402
    compute_reward_geo_semantic,
    compute_reward_geo_global,
    _prepare_frames,
    _get_pairs,
    build_warp_grid,
)
import torch.nn.functional as F  # noqa: E402


def _vid_size(p: Path):
    cap = cv2.VideoCapture(str(p))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    return H, W


def diagnose_one_pair(
    da3_data: dict, label_maps: np.ndarray,
    H_img: int, W_img: int, src: int, ref: int, device: str,
):
    """对单个 (src, ref) 输出 mask / label / match 详细统计。"""
    depths, Ks, c2w_all, confs, N = _prepare_frames(
        da3_data, H_img, W_img, device)
    label_raw = torch.from_numpy(label_maps.astype(np.int32)).to(device)
    if label_raw.shape[1] != H_img or label_raw.shape[2] != W_img:
        label_t = F.interpolate(
            label_raw.unsqueeze(1).float(), size=(H_img, W_img),
            mode="nearest").squeeze(1).long()
    else:
        label_t = label_raw

    flow_grid, valid_mask, proj_depth = build_warp_grid(
        H_img, W_img,
        depth=depths[src], K_src=Ks[src], c2w_src=c2w_all[src],
        K_ref=Ks[ref], c2w_ref=c2w_all[ref],
        conf=confs[src], conf_threshold=0.0,
    )
    mask_2d = valid_mask.squeeze().bool()
    valid_count = int(mask_2d.sum().item())

    ref_label_sampled = F.grid_sample(
        label_t[ref].float()[None, None], flow_grid, mode="nearest",
        padding_mode="zeros", align_corners=True,
    ).squeeze().long()
    label_src = label_t[src]

    fg_src = (label_src > 0) & mask_2d
    fg_ref_warped = (ref_label_sampled > 0) & mask_2d
    bg_src = (label_src == 0) & mask_2d
    bg_ref_warped = (ref_label_sampled == 0) & mask_2d

    fg_match = fg_src & fg_ref_warped & (label_src == ref_label_sampled)
    bg_match = bg_src & bg_ref_warped

    n_pix = H_img * W_img
    out = {
        "frame_size": (H_img, W_img),
        "n_pix": n_pix,
        "valid_count": valid_count,
        "valid_frac": valid_count / n_pix,
        "fg_src_frac": fg_src.sum().item() / valid_count if valid_count else 0,
        "bg_src_frac": bg_src.sum().item() / valid_count if valid_count else 0,
        "fg_ref_warped_frac": (fg_ref_warped.sum().item() / valid_count
                                if valid_count else 0),
        "fg_match_frac": fg_match.sum().item() / valid_count if valid_count else 0,
        "bg_match_frac": bg_match.sum().item() / valid_count if valid_count else 0,
        "label_match_rate": ((fg_match | bg_match).sum().item() / valid_count
                             if valid_count else 0),
    }

    # Per-label diagnostics
    src_labels, src_counts = torch.unique(
        label_src[mask_2d], return_counts=True)
    ref_labels, ref_counts = torch.unique(
        ref_label_sampled[mask_2d], return_counts=True)
    out["unique_labels_src"] = sorted(src_labels.cpu().tolist())
    out["unique_labels_ref_warped"] = sorted(ref_labels.cpu().tolist())
    out["src_label_pix"] = {int(k): int(v) for k, v in
                             zip(src_labels.cpu().tolist(),
                                 src_counts.cpu().tolist())}

    # For each fg label k>0 in src, how many are matched in ref
    per_label_match = {}
    for k in src_labels.cpu().tolist():
        if k == 0:
            continue
        k_src = (label_src == k) & mask_2d
        k_ref = (ref_label_sampled == k) & mask_2d
        intersect = (k_src & k_ref).sum().item()
        per_label_match[int(k)] = {
            "src_pix": int(k_src.sum().item()),
            "ref_warped_pix": int(k_ref.sum().item()),
            "intersect": int(intersect),
            "iou": float(intersect / max((k_src | k_ref).sum().item(), 1)),
        }
    out["per_label_match"] = per_label_match
    return out


def diagnose_sample(sample_dir: Path, mode: str, device: str):
    """对一个 sample 的 8 个 rollouts，跑 geo_sem + geo_global，收集 details。"""
    name = sample_dir.name
    gen_dirs = sorted(d for d in sample_dir.glob("gen_*") if d.is_dir())
    rollouts = []
    for gd in gen_dirs:
        gv = sample_dir / f"{gd.name}.mp4"
        da3 = gd / "intermediates" / "da3_pred.npz"
        lm  = gd / "intermediates" / "label_maps.npz"
        if not (da3.exists() and lm.exists() and gv.exists()):
            print(f"  [skip] {gd.name}: da3={da3.exists()} lm={lm.exists()} mp4={gv.exists()}")
            continue
        H_img, W_img = _vid_size(gv)
        if H_img == 0:
            continue
        da3_data = dict(np.load(str(da3), allow_pickle=True))
        try:
            label_maps = np.load(str(lm), allow_pickle=True)["label_maps"]
        except Exception as e:
            print(f"  [WARN] load label_maps failed: {e}")
            continue

        sem_r, sem_d = compute_reward_geo_semantic(
            da3_data, label_maps, H_img, W_img,
            device=device, compare_mode=mode)
        glob_r, glob_d = compute_reward_geo_global(
            da3_data, H_img, W_img,
            device=device, compare_mode=mode)
        rollouts.append({
            "gen_idx": int(gd.name.split("_")[1]),
            "sem": sem_r,
            "glob": glob_r,
            "sem_details": {
                "mean_score": sem_d.get("mean_score"),
                "mean_label_match_rate": sem_d.get("mean_label_match_rate"),
                "mean_fg_match_rate": sem_d.get("mean_fg_match_rate"),
                "mean_valid_pixels": sem_d.get("mean_valid_pixels"),
                "per_frame_score": sem_d.get("per_frame_score"),
                "per_frame_label_match_rate": sem_d.get(
                    "per_frame_label_match_rate"),
                "per_frame_fg_match_rate": sem_d.get(
                    "per_frame_fg_match_rate"),
                "per_frame_valid_pixels": sem_d.get(
                    "per_frame_valid_pixels"),
            },
            "glob_details": {
                "mean_score": glob_d.get("mean_score"),
                "mean_valid_pixels": glob_d.get("mean_valid_pixels"),
                "per_frame_score": glob_d.get("per_frame_score"),
                "per_frame_valid_pixels": glob_d.get(
                    "per_frame_valid_pixels"),
            },
            "_da3_path": str(da3),
            "_lm_path":  str(lm),
            "_HW": (H_img, W_img),
        })

    return name, rollouts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="+", default=None,
                    help="DL3DV sample dirs to diagnose (full path). "
                         "If empty: pick 4 random.")
    ap.add_argument("--mode", default="first_frame",
                    choices=["first_frame", "first_three", "all_pairs"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out_json", default="/tmp/diagnose_geo_sem.json")
    args = ap.parse_args()

    base = Path("/horizon-bucket/robot_lab/users/puxin.yan-labs/test_output1/dl3dv")
    if args.samples:
        sdirs = [Path(s) for s in args.samples]
    else:
        import random
        random.seed(7)
        all_s = sorted(d for d in base.iterdir() if d.is_dir())
        sdirs = random.sample(all_s, 4)

    print(f"Diagnosing {len(sdirs)} samples on {args.device}, mode={args.mode}")
    out = {"mode": args.mode, "samples": {}}
    for sd in sdirs:
        print(f"\n--- {sd.name} ---")
        name, rolls = diagnose_sample(sd, args.mode, args.device)
        if not rolls:
            print("  (no rollouts loadable)")
            continue
        sems = np.array([r["sem"] for r in rolls])
        globs = np.array([r["glob"] for r in rolls])
        print(f"  geo_sem  : n={len(sems)}  mean={sems.mean():.4f}  std={sems.std():.4f}  range=[{sems.min():.3f}, {sems.max():.3f}]")
        print(f"  geo_glob : n={len(globs)} mean={globs.mean():.4f}  std={globs.std():.4f}  range=[{globs.min():.3f}, {globs.max():.3f}]")
        print(f"  std ratio (sem/glob) = {sems.std()/max(globs.std(),1e-12):.2f}x")
        print(f"  per-rollout:")
        print(f"    {'gen':>4} {'sem':>8} {'glob':>8} {'lblMR':>7} {'fgMR':>7} {'valid':>10}")
        for r in rolls:
            d = r["sem_details"]
            gd = r["glob_details"]
            print(f"    {r['gen_idx']:>4d} {r['sem']:+.4f} {r['glob']:+.4f} "
                  f"{d['mean_label_match_rate']:.3f} {d['mean_fg_match_rate']:.3f} "
                  f"{d['mean_valid_pixels']:>9}")
        # Save details (drop heavy paths after diagnose_pair)
        out["samples"][name] = {
            "stats": {
                "sem_mean": float(sems.mean()), "sem_std": float(sems.std()),
                "glob_mean": float(globs.mean()), "glob_std": float(globs.std()),
                "ratio": float(sems.std()/max(globs.std(), 1e-12)),
            },
            "rollouts": [
                {
                    "gen_idx": r["gen_idx"],
                    "sem": r["sem"], "glob": r["glob"],
                    "sem_details": r["sem_details"],
                    "glob_details": r["glob_details"],
                }
                for r in rolls
            ],
        }

        # Pick the worst rollout (lowest sem) and the best, dump pair detail
        sems_list = [(r["sem"], i) for i, r in enumerate(rolls)]
        sems_list.sort()
        for tag, i in [("worst_sem", sems_list[0][1]),
                        ("best_sem",  sems_list[-1][1])]:
            r = rolls[i]
            H, W = r["_HW"]
            da3_data = dict(np.load(r["_da3_path"], allow_pickle=True))
            lm = np.load(r["_lm_path"], allow_pickle=True)["label_maps"]
            N = lm.shape[0]
            mid = N // 2
            pair_diag = diagnose_one_pair(
                da3_data, lm, H, W, src=mid, ref=0, device=args.device)
            print(f"\n  [{tag} = gen_{r['gen_idx']}] pair (src={mid}, ref=0):")
            print(f"    valid_frac={pair_diag['valid_frac']:.3f}  "
                  f"fg_src={pair_diag['fg_src_frac']:.3f}  "
                  f"bg_src={pair_diag['bg_src_frac']:.3f}")
            print(f"    fg_ref_warped={pair_diag['fg_ref_warped_frac']:.3f}  "
                  f"fg_match={pair_diag['fg_match_frac']:.3f}  "
                  f"bg_match={pair_diag['bg_match_frac']:.3f}  "
                  f"label_match_rate={pair_diag['label_match_rate']:.3f}")
            print(f"    unique src labels: {pair_diag['unique_labels_src']}")
            print(f"    unique ref-warp labels: {pair_diag['unique_labels_ref_warped']}")
            print(f"    per-label IoU:")
            for k, v in sorted(pair_diag["per_label_match"].items()):
                print(f"      label {k}: src_pix={v['src_pix']:>7}  "
                      f"ref_pix={v['ref_warped_pix']:>7}  "
                      f"inter={v['intersect']:>7}  IoU={v['iou']:.3f}")
            out["samples"][name][f"pair_diag_{tag}"] = pair_diag

    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved details: {args.out_json}")


if __name__ == "__main__":
    main()
