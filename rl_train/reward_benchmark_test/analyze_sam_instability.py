"""
全样本统计：SAM3 grounding 不稳定性是否是 geo_semantic 失效的根本原因？

对 DL3DV (200) + Re10K (187) 的所有 sample × 8 rollout 做：
  1) 验证 Qwen 输出是否在 8 rollout 间一致（pred_objects.json 比对）
  2) 量化 SAM3 输出的 within-sample 不稳定（label 数 / fg 比例的 std/range）
  3) 计算 geo_sem / geo_glob 的 within-sample std
  4) 求相关：SAM 不稳定度 vs geo_sem 不稳定度

输出：
  - results/sam_instability_per_sample.json     每个 sample 的统计明细
  - results/sam_instability_summary.json        全局汇总
  - results/sam_instability_report.txt          可读报告
  - results/sam_instability_scatter.png         散点图：SAM 不稳定 vs geo_sem 不稳定
"""

import json
import os
import sys
from pathlib import Path
import concurrent.futures as cf
import numpy as np

ROOT = Path("/horizon-bucket/robot_lab/users/puxin.yan-labs/test_output1")
OUT = Path("/home/users/puxin.yan-labs/RL_code/rl_train/reward_benchmark_test/results")
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = ["dl3dv", "re10k"]
N_WORKERS = 32


def per_rollout_stats(gen_dir: Path) -> dict | None:
    """单个 rollout：读 label_maps.npz + pred_objects.json + reward_multimode.json"""
    lm_p = gen_dir / "intermediates" / "label_maps.npz"
    obj_p = gen_dir / "intermediates" / "pred_objects.json"
    rm_p = gen_dir / "reward_multimode.json"
    if not (lm_p.exists() and obj_p.exists() and rm_p.exists()):
        return None
    try:
        lm = np.load(str(lm_p), allow_pickle=True)["label_maps"]  # (T, H, W) int16
    except Exception:
        return None
    T, H, W = lm.shape
    fg_mask = lm > 0
    per_frame_fg = fg_mask.reshape(T, -1).mean(axis=1)  # (T,)
    unique_labels = np.unique(lm[fg_mask]).tolist() if fg_mask.any() else []
    n_unique = len(unique_labels)

    with open(obj_p) as f:
        obj_data = json.load(f)
    objects = obj_data.get("objects", [])

    with open(rm_p) as f:
        rm = json.load(f)
    geo_sem_ff = rm.get("modes", {}).get("first_frame", {}).get("geo_semantic")
    geo_glob_ff = rm.get("modes", {}).get("first_frame", {}).get("geo_global")
    geo_sem_ap = rm.get("modes", {}).get("all_pairs", {}).get("geo_semantic")
    geo_glob_ap = rm.get("modes", {}).get("all_pairs", {}).get("geo_global")

    return {
        "T": int(T),
        "frame0_fg": float(per_frame_fg[0]),
        "mean_fg": float(per_frame_fg.mean()),
        "max_fg": float(per_frame_fg.max()),
        "n_unique_labels": int(n_unique),
        "qwen_objects": objects,
        "qwen_n_objects": len(objects),
        "geo_sem_ff": geo_sem_ff,
        "geo_glob_ff": geo_glob_ff,
        "geo_sem_ap": geo_sem_ap,
        "geo_glob_ap": geo_glob_ap,
    }


def per_sample_stats(sample_dir: Path) -> dict:
    """单个 sample：聚合 8 个 rollout 的统计"""
    gens = sorted([p for p in sample_dir.iterdir() if p.is_dir() and p.name.startswith("gen_")])
    rollouts = []
    for g in gens:
        s = per_rollout_stats(g)
        if s is not None:
            rollouts.append(s)
    if len(rollouts) < 2:
        return {"sample": sample_dir.name, "n_rollouts_ok": len(rollouts), "skip": True}

    # Qwen consistency: 是否所有 rollout 的 objects list 完全相同
    qwen_lists = [tuple(r["qwen_objects"]) for r in rollouts]
    qwen_identical = len(set(qwen_lists)) == 1
    qwen_unique_lists = len(set(qwen_lists))
    qwen_n_objs = [r["qwen_n_objects"] for r in rollouts]

    # SAM 输出统计
    n_labels = np.array([r["n_unique_labels"] for r in rollouts], dtype=np.float64)
    f0_fg = np.array([r["frame0_fg"] for r in rollouts], dtype=np.float64)
    mean_fg = np.array([r["mean_fg"] for r in rollouts], dtype=np.float64)
    max_fg = np.array([r["max_fg"] for r in rollouts], dtype=np.float64)

    # 失败定义：mean_fg < 0.02（整段视频 fg<2% 视为 SAM 完全失败）
    sam_fail = (mean_fg < 0.02)
    sam_fail_rate = float(sam_fail.mean())

    # geo_sem / geo_glob 的 within-sample 统计
    def _arr(key):
        v = np.array([r.get(key) for r in rollouts], dtype=np.float64)
        return v if not np.isnan(v).all() else None

    gs_ff = _arr("geo_sem_ff");  gg_ff = _arr("geo_glob_ff")
    gs_ap = _arr("geo_sem_ap");  gg_ap = _arr("geo_glob_ap")

    def _stats(v):
        if v is None or np.isnan(v).any():
            return None
        return {
            "mean": float(v.mean()),
            "std":  float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "range": float(v.max() - v.min()),
        }

    return {
        "sample": sample_dir.name,
        "n_rollouts_ok": len(rollouts),
        "skip": False,
        "qwen": {
            "identical_across_rollouts": bool(qwen_identical),
            "n_unique_lists": int(qwen_unique_lists),
            "n_objects_per_rollout": qwen_n_objs,  # list of 8 ints
        },
        "sam": {
            "n_labels_per_rollout":   n_labels.tolist(),
            "n_labels_std":           float(n_labels.std(ddof=1)),
            "n_labels_range":         float(n_labels.max() - n_labels.min()),
            "frame0_fg_per_rollout":  f0_fg.tolist(),
            "frame0_fg_std":          float(f0_fg.std(ddof=1)),
            "frame0_fg_range":        float(f0_fg.max() - f0_fg.min()),
            "mean_fg_per_rollout":    mean_fg.tolist(),
            "mean_fg_std":            float(mean_fg.std(ddof=1)),
            "mean_fg_range":          float(mean_fg.max() - mean_fg.min()),
            "max_fg_per_rollout":     max_fg.tolist(),
            "fail_rate":              sam_fail_rate,  # 占 8 个 rollout 的几个完全失败
        },
        "geo": {
            "sem_ff": _stats(gs_ff),
            "glob_ff": _stats(gg_ff),
            "sem_ap": _stats(gs_ap),
            "glob_ap": _stats(gg_ap),
        },
    }


def collect_dataset(ds: str) -> list:
    base = ROOT / ds
    sample_dirs = sorted([p for p in base.iterdir() if p.is_dir()])
    print(f"[{ds}] {len(sample_dirs)} samples ...", flush=True)
    out = []
    with cf.ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(per_sample_stats, sd): sd for sd in sample_dirs}
        for i, fut in enumerate(cf.as_completed(futs)):
            try:
                r = fut.result()
                r["dataset"] = ds
                out.append(r)
            except Exception as e:
                sd = futs[fut]
                print(f"  ERR {sd.name}: {e}", flush=True)
            if (i + 1) % 50 == 0 or (i + 1) == len(sample_dirs):
                print(f"  [{ds}] {i+1}/{len(sample_dirs)} done", flush=True)
    return out


def summarize(per_sample: list) -> dict:
    """全局汇总：Qwen 一致率、SAM 不稳定分布、与 geo_sem std 的相关性"""
    by_ds = {ds: [r for r in per_sample if r.get("dataset") == ds and not r.get("skip")] for ds in DATASETS}
    by_ds["all"] = [r for r in per_sample if not r.get("skip")]

    summary = {}
    for ds, rs in by_ds.items():
        n = len(rs)
        if n == 0:
            continue
        qwen_id = sum(1 for r in rs if r["qwen"]["identical_across_rollouts"])
        n_lab_std = np.array([r["sam"]["n_labels_std"] for r in rs])
        f0_std    = np.array([r["sam"]["frame0_fg_std"] for r in rs])
        mean_std  = np.array([r["sam"]["mean_fg_std"] for r in rs])
        fail_rate = np.array([r["sam"]["fail_rate"] for r in rs])

        gs_std = np.array([r["geo"]["sem_ff"]["std"]  if r["geo"]["sem_ff"]  else np.nan for r in rs])
        gg_std = np.array([r["geo"]["glob_ff"]["std"] if r["geo"]["glob_ff"] else np.nan for r in rs])

        # Pearson correlation (drop NaN)
        def _corr(a, b):
            m = ~(np.isnan(a) | np.isnan(b))
            if m.sum() < 3:
                return None
            return float(np.corrcoef(a[m], b[m])[0, 1])

        summary[ds] = {
            "n_samples": n,
            "qwen_identical_rate": qwen_id / n,
            "qwen_identical_count": qwen_id,
            "sam_n_labels_std":    {"mean": float(np.nanmean(n_lab_std)), "median": float(np.nanmedian(n_lab_std)), "p90": float(np.nanpercentile(n_lab_std, 90))},
            "sam_frame0_fg_std":   {"mean": float(np.nanmean(f0_std)),    "median": float(np.nanmedian(f0_std)),    "p90": float(np.nanpercentile(f0_std, 90))},
            "sam_mean_fg_std":     {"mean": float(np.nanmean(mean_std)),  "median": float(np.nanmedian(mean_std)),  "p90": float(np.nanpercentile(mean_std, 90))},
            "sam_fail_rate":       {"mean": float(np.nanmean(fail_rate)), "median": float(np.nanmedian(fail_rate)), "p90": float(np.nanpercentile(fail_rate, 90))},
            "geo_sem_within_std":  {"mean": float(np.nanmean(gs_std)), "median": float(np.nanmedian(gs_std))},
            "geo_glob_within_std": {"mean": float(np.nanmean(gg_std)), "median": float(np.nanmedian(gg_std))},
            "corr_sam_fail_vs_geo_sem_std":   _corr(fail_rate, gs_std),
            "corr_sam_mean_fg_std_vs_geo_sem_std": _corr(mean_std, gs_std),
            "corr_sam_n_labels_std_vs_geo_sem_std": _corr(n_lab_std, gs_std),
            # 反事实：去掉高 SAM 不稳定的样本后，geo_sem 的 within-std 是否显著降低
            "geo_sem_std_among_stable_sam":   float(np.nanmean(gs_std[mean_std < np.nanmedian(mean_std)])) if n > 5 else None,
            "geo_sem_std_among_unstable_sam": float(np.nanmean(gs_std[mean_std >= np.nanmedian(mean_std)])) if n > 5 else None,
            "geo_glob_std_among_stable_sam":  float(np.nanmean(gg_std[mean_std < np.nanmedian(mean_std)])) if n > 5 else None,
            "geo_glob_std_among_unstable_sam":float(np.nanmean(gg_std[mean_std >= np.nanmedian(mean_std)])) if n > 5 else None,
        }
    return summary


def make_scatter(per_sample: list, summary: dict, out_path: Path):
    """4 子图：分别 plot SAM 各种不稳定度 vs geo_sem within-std；按 dataset 着色"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    rs_all = [r for r in per_sample if not r.get("skip")]
    colors = {"dl3dv": "tab:blue", "re10k": "tab:orange"}

    def _plot(ax, x_key_path, x_label):
        for ds in DATASETS:
            rs = [r for r in rs_all if r["dataset"] == ds and r["geo"]["sem_ff"] is not None]
            xs, ys_sem, ys_glob = [], [], []
            for r in rs:
                # nested key resolve
                v = r
                for k in x_key_path:
                    v = v[k]
                xs.append(v)
                ys_sem.append(r["geo"]["sem_ff"]["std"])
                ys_glob.append(r["geo"]["glob_ff"]["std"] if r["geo"]["glob_ff"] else np.nan)
            xs = np.array(xs); ys_sem = np.array(ys_sem); ys_glob = np.array(ys_glob)
            ax.scatter(xs, ys_sem, s=14, alpha=0.45, c=colors[ds], label=f"{ds} (geo_sem)", edgecolors="none")
            ax.scatter(xs, ys_glob, s=12, alpha=0.25, c=colors[ds], marker="x", label=f"{ds} (geo_glob)")
        ax.set_xlabel(x_label)
        ax.set_ylabel("within-sample std")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    _plot(axes[0,0], ["sam","fail_rate"],     "SAM fail rate (mean_fg<0.02 fraction over 8 rollouts)")
    _plot(axes[0,1], ["sam","mean_fg_std"],   "SAM mean_fg std across 8 rollouts")
    _plot(axes[1,0], ["sam","n_labels_std"],  "SAM n_unique_labels std across 8 rollouts")
    _plot(axes[1,1], ["sam","frame0_fg_std"], "SAM frame0_fg std across 8 rollouts")

    # title with corr summary
    s = summary.get("all", {})
    c1 = s.get("corr_sam_fail_vs_geo_sem_std")
    c2 = s.get("corr_sam_mean_fg_std_vs_geo_sem_std")
    c3 = s.get("corr_sam_n_labels_std_vs_geo_sem_std")
    fig.suptitle(
        f"SAM3 instability vs geo_semantic within-sample std (DL3DV+Re10K, n={s.get('n_samples')})\n"
        f"Pearson:  fail_rate→{c1:+.3f}   mean_fg_std→{c2:+.3f}   n_labels_std→{c3:+.3f}"
        if c1 is not None else "SAM3 instability vs geo_semantic within-sample std",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(str(out_path), dpi=130, bbox_inches="tight")
    print(f"saved {out_path}", flush=True)


def make_report(per_sample: list, summary: dict, out_path: Path):
    """文字报告"""
    lines = []
    lines.append("=" * 80)
    lines.append("SAM3 grounding 不稳定性 → geo_semantic 失效  全样本验证报告")
    lines.append("=" * 80)
    lines.append("")
    for ds in ["dl3dv", "re10k", "all"]:
        s = summary.get(ds)
        if s is None:
            continue
        lines.append(f"### Dataset: {ds}  (n_samples={s['n_samples']})")
        lines.append("")
        lines.append(f"  Qwen 8-rollout 完全一致率: {s['qwen_identical_count']}/{s['n_samples']}  "
                     f"= {s['qwen_identical_rate']*100:.1f}%   <-- 验证 Qwen 是否稳定")
        lines.append("")
        lines.append("  SAM3 within-sample 不稳定度 (across 8 rollouts):")
        lines.append(f"    n_unique_labels  std :  mean={s['sam_n_labels_std']['mean']:.2f}   median={s['sam_n_labels_std']['median']:.2f}   p90={s['sam_n_labels_std']['p90']:.2f}")
        lines.append(f"    frame0_fg        std :  mean={s['sam_frame0_fg_std']['mean']:.3f}   median={s['sam_frame0_fg_std']['median']:.3f}   p90={s['sam_frame0_fg_std']['p90']:.3f}")
        lines.append(f"    mean_fg          std :  mean={s['sam_mean_fg_std']['mean']:.3f}   median={s['sam_mean_fg_std']['median']:.3f}   p90={s['sam_mean_fg_std']['p90']:.3f}")
        lines.append(f"    fail_rate (mean_fg<0.02):  mean={s['sam_fail_rate']['mean']:.2%}   median={s['sam_fail_rate']['median']:.2%}   p90={s['sam_fail_rate']['p90']:.2%}")
        lines.append("")
        lines.append("  Reward within-sample std:")
        lines.append(f"    geo_semantic (FF) :  mean={s['geo_sem_within_std']['mean']:.4f}   median={s['geo_sem_within_std']['median']:.4f}")
        lines.append(f"    geo_global   (FF) :  mean={s['geo_glob_within_std']['mean']:.4f}   median={s['geo_glob_within_std']['median']:.4f}")
        lines.append("")
        lines.append("  ★ 相关性 (SAM 不稳定 → geo_sem within-std)  ←   核心证据")
        lines.append(f"    Pearson(SAM fail_rate,    geo_sem std) = {s['corr_sam_fail_vs_geo_sem_std']:+.3f}")
        lines.append(f"    Pearson(SAM mean_fg_std,  geo_sem std) = {s['corr_sam_mean_fg_std_vs_geo_sem_std']:+.3f}")
        lines.append(f"    Pearson(SAM n_labels_std, geo_sem std) = {s['corr_sam_n_labels_std_vs_geo_sem_std']:+.3f}")
        lines.append("")
        lines.append("  反事实对比 (按 SAM mean_fg_std 中位数二分):")
        lines.append(f"    geo_sem  std  (SAM 稳定的样本): {s['geo_sem_std_among_stable_sam']:.4f}")
        lines.append(f"    geo_sem  std  (SAM 不稳定样本): {s['geo_sem_std_among_unstable_sam']:.4f}")
        lines.append(f"    geo_glob std  (SAM 稳定的样本): {s['geo_glob_std_among_stable_sam']:.4f}")
        lines.append(f"    geo_glob std  (SAM 不稳定样本): {s['geo_glob_std_among_unstable_sam']:.4f}")
        if s['geo_glob_std_among_stable_sam'] and s['geo_glob_std_among_stable_sam'] > 0:
            ratio_sem = s['geo_sem_std_among_unstable_sam'] / max(s['geo_sem_std_among_stable_sam'], 1e-9)
            ratio_glob = s['geo_glob_std_among_unstable_sam'] / max(s['geo_glob_std_among_stable_sam'], 1e-9)
            lines.append(f"    -> 不稳定/稳定 比值:  geo_sem={ratio_sem:.2f}x   geo_glob={ratio_glob:.2f}x")
            lines.append(f"       (geo_sem 应远大于 geo_glob，证明 SAM 不稳定主要污染了 sem，没有污染 glob)")
        lines.append("")
        lines.append("-" * 80)
    out_path.write_text("\n".join(lines))
    print(f"saved {out_path}", flush=True)


def main():
    all_records = []
    for ds in DATASETS:
        rs = collect_dataset(ds)
        all_records.extend(rs)

    # save per-sample
    per_p = OUT / "sam_instability_per_sample.json"
    with open(per_p, "w") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=1)
    print(f"saved {per_p}  ({len(all_records)} samples)", flush=True)

    summary = summarize(all_records)
    sum_p = OUT / "sam_instability_summary.json"
    with open(sum_p, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"saved {sum_p}", flush=True)

    make_report(all_records, summary, OUT / "sam_instability_report.txt")
    make_scatter(all_records, summary, OUT / "sam_instability_scatter.png")


if __name__ == "__main__":
    main()
