#!/usr/bin/env python3
"""
run_correlation_analysis.py
============================
Phase 4: 收集 reward_multimode + benchmark 数据，计算组内优势(advantage)，
计算 Pearson 相关系数，生成可视化表格图片。

Reward 来自 reward_multimode.json（3 种 geo 对比模式）：
  geo_sem_ff / geo_sem_f3 / geo_sem_ap    — geo_semantic (3 modes)
  geo_glob_ff / geo_glob_f3 / geo_glob_ap — geo_global  (3 modes)
  feature_sim                             — 单一 first_frame
  camera_traj                             — 单一
  video_quality                           — 单一
  total_ff / total_f3 / total_ap          — reward_total (3 modes)
共 14 列。

Benchmark 来自 eval/ 目录：
  PSNR, SSIM, LPIPS (3)
  VBench/i2v_subj, VBench/i2v_bg, VBench/img_qual (3)
  CamPose/rot_auc30, trans_auc30, pose_auc30, trans_met (4)
  VideoAlign/Overall (1)
  GlobalPC/{Cam,FF,Ume,ICP} x {accuracy, completeness, chamfer} (12)
  ObjPC/{Cam,FF,Ume,ICP} x {accuracy, completeness, chamfer} (12)
共 35 列。

三份独立分析：re10k / dl3dv / combined（各自热力图 + JSON + 贡献权重）。

同时可作为总调度脚本，通过 --phase 参数选择运行哪些阶段。

用法（单独运行 Phase 4）:
    python run_correlation_analysis.py \\
        --output_root /horizon-bucket/robot_lab/users/puxin.yan-labs/test_output1 \\
        --out_dir ./results \\
        --phase 4

用法（运行 Phase 2,3,4）:
    python run_correlation_analysis.py \\
        --output_root /path/to/test_output1 \\
        --out_dir ./results \\
        --phase 2,3,4 \\
        --gpu_ids 0,1,2,3 --gpu 0 --align all_align
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_RL_CODE_DIR = _THIS_DIR.parent.parent
_BENCHMARK_DIR = _RL_CODE_DIR / "eval" / "benchmark"

sys.path.insert(0, str(_BENCHMARK_DIR))
from common.scan import scan_output_root
from common.utils import log


# ═══════════════════════════════════════════════════════════════════
# 1. 常量定义
# ═══════════════════════════════════════════════════════════════════

# geo 对比模式及其列名后缀
GEO_MODES = ["first_frame", "first_three", "all_pairs"]
GEO_MODE_SUFFIX = {
    "first_frame":  "ff",
    "first_three":  "f3",
    "all_pairs":    "ap",
}
GEO_MODE_LABEL = {
    "first_frame":  "FF",
    "first_three":  "F3",
    "all_pairs":    "AP",
}

# (列名, 显示名, 对应的 total key 或 None, 贡献权重或 None)
#   对于 total 行，贡献权重=None（不计算自身贡献）
REWARD_COLS = []
# NEW WEIGHTS (target w*within_std ≈ 0.10, geo_sem dropped):
#   geo_global=7.7, feature_sim=5.3, camera_rot=0.92, camera_trans=3.6,
#   video_quality=0.67, geo_semantic=0
# geo_semantic x 3 modes (weight=0, kept for visualization only)
for _m in GEO_MODES:
    _sfx = GEO_MODE_SUFFIX[_m]
    _lbl = GEO_MODE_LABEL[_m]
    REWARD_COLS.append((f"geo_sem_{_sfx}",  f"GeoSem/{_lbl}",   f"total_{_sfx}", 0.0))
# geo_global x 3 modes
for _m in GEO_MODES:
    _sfx = GEO_MODE_SUFFIX[_m]
    _lbl = GEO_MODE_LABEL[_m]
    REWARD_COLS.append((f"geo_glob_{_sfx}", f"GeoGlob/{_lbl}",  f"total_{_sfx}", 7.7))
# shared rewards (contribute to all 3 totals)
REWARD_COLS.append(("feature_sim",   "FeatSim",  None,  5.3))
REWARD_COLS.append(("camera_rot",    "CamRot",   None,  0.92))
REWARD_COLS.append(("camera_trans",  "CamTrans", None,  3.6))
REWARD_COLS.append(("video_quality", "VidQual",  None,  0.67))
# totals
for _m in GEO_MODES:
    _sfx = GEO_MODE_SUFFIX[_m]
    _lbl = GEO_MODE_LABEL[_m]
    REWARD_COLS.append((f"total_{_sfx}", f"Total/{_lbl}", None, None))

# Set of all reward column names (for separating from benchmark cols)
ALL_REWARD_KEYS = {col for col, _, _, _ in REWARD_COLS}
# The 3 total keys
TOTAL_KEYS = [f"total_{GEO_MODE_SUFFIX[m]}" for m in GEO_MODES]

# Benchmark keys: (filename, nested key path, display name)
BENCHMARK_KEYS = [
    # ── Video Quality ──────────────────────────────────────────
    ("psnr_ssim_lpips.json", ("psnr",),            "PSNR"),
    ("psnr_ssim_lpips.json", ("ssim",),            "SSIM"),
    ("psnr_ssim_lpips.json", ("lpips",),           "LPIPS"),
    # ── VBench ─────────────────────────────────────────────────
    ("vbench.json",          ("i2v_subject",),     "VBench/i2v_subj"),
    ("vbench.json",          ("i2v_background",),  "VBench/i2v_bg"),
    ("vbench.json",          ("imaging_quality",), "VBench/img_qual"),
    # ── Camera Pose AUC ────────────────────────────────────────
    ("camera_pose.json",     ("rotation_auc30",),     "CamPose/rot_auc30"),
    ("camera_pose.json",     ("translation_auc30",),  "CamPose/trans_auc30"),
    ("camera_pose.json",     ("pose_auc30",),          "CamPose/pose_auc30"),
    ("camera_pose.json",     ("translation_metric",),  "CamPose/trans_met"),
    # ── VideoAlign ─────────────────────────────────────────────
    ("videoalign.json",      ("Overall",),          "VideoAlign/Overall"),
]

# Fallback filenames for backward compatibility (搜索 eval/ 目录内的替代文件)
_BENCH_FILE_FALLBACK = {
    "psnr_ssim_lpips.json": "video_quality.json",
}

# 如果 eval/ 下找不到该 benchmark 文件，则再到 intermediates/ 下找同名文件
# （VBench 目前只在 intermediates/vbench.json 里；videoalign 两处都可能有）
_BENCH_INTERMEDIATE_FALLBACK = {
    "vbench.json",
    "videoalign.json",
}

# 方向标记：+1 = 越大越好（保持 r 原值），-1 = 越小越好（绘图时对 r 取负）
# 目的：可视化上统一为 "r > 0 = reward 与 better benchmark 正相关"，红色 = 好。
# 生成的 JSON 保留原始 r（未翻转），只有绘图/对比图使用翻转后的值。
BENCHMARK_DIRECTION = {
    # Video Quality
    "PSNR":  +1,  "SSIM":  +1,  "LPIPS": -1,
    # VBench
    "VBench/i2v_subj": +1, "VBench/i2v_bg": +1, "VBench/img_qual": +1,
    # Camera Pose: AUC 越大越好；translation_metric = mean(-exp(d/0.3)) ∈ (−∞, −1]，
    # less-negative = better，同样是 higher-is-better (+1)
    "CamPose/rot_auc30":   +1,
    "CamPose/trans_auc30": +1,
    "CamPose/pose_auc30":  +1,
    "CamPose/trans_met":   +1,
    # VideoAlign
    "VideoAlign/Overall": +1,
    # Point cloud metrics：accuracy / completeness / chamfer_distance
    # 在这份代码里都是 "mean distance (mm)" 语义，越小越好。
}
# 点云指标全部 lower-is-better
for _prefix in ("GlobalPC", "ObjPC"):
    for _align_lbl in ("Cam", "FF", "Ume", "ICP"):
        for _ms in ("acc", "comp", "chamfer"):
            BENCHMARK_DIRECTION[f"{_prefix}/{_align_lbl}/{_ms}"] = -1

# Phase 3 写入的点云 eval 文件后缀（见 run_benchmark_recompute.py: DEFAULT_RECON_SUFFIX="mm"）
# 读取顺序： global_point_cloud_{RECON_EVAL_SUFFIX}.json → global_point_cloud.json
RECON_EVAL_SUFFIX = "mm"

# Point cloud: 4 align modes × 3 metrics × 2 PC types
PC_ALIGN_MODES  = ["camera", "first_frame", "umeyama", "icp"]
PC_ALIGN_LABELS = {"camera": "Cam", "first_frame": "FF",
                   "umeyama": "Ume", "icp": "ICP"}
PC_METRICS = ["accuracy", "completeness", "chamfer_distance"]
PC_METRIC_SHORT = {"accuracy": "acc", "completeness": "comp",
                   "chamfer_distance": "chamfer"}


# ═══════════════════════════════════════════════════════════════════
# 2. 数据收集
# ═══════════════════════════════════════════════════════════════════

def _get_nested(d: dict, keys: tuple):
    """从嵌套 dict 中按 key 序列取值，失败返回 None。"""
    val = d
    for k in keys:
        if not isinstance(val, dict):
            return None
        val = val.get(k)
        if val is None:
            return None
    return val


def load_json_safe(path) -> dict:
    try:
        with open(str(path)) as f:
            return json.load(f)
    except Exception:
        return {}


def collect_record(entry: dict, gv: dict) -> dict | None:
    """收集单个 gen 视频的所有 reward + benchmark 值。"""
    gen_dir = gv["gen_dir"]
    record = {
        "dataset":   entry["dataset"],
        "sample_id": entry["sample_id"],
        "gen_idx":   gv["idx"],
    }

    # ── Reward (multimode) ────────────────────────────────────
    mm_json = gen_dir / "reward_multimode.json"
    if not mm_json.exists():
        return None
    mm = load_json_safe(mm_json)
    if not mm:
        return None

    modes_data = mm.get("modes", {})
    for m in GEO_MODES:
        sfx = GEO_MODE_SUFFIX[m]
        md = modes_data.get(m, {})
        record[f"geo_sem_{sfx}"]  = float(md.get("geo_semantic",  float("nan")))
        record[f"geo_glob_{sfx}"] = float(md.get("geo_global",    float("nan")))
        record[f"total_{sfx}"]    = float(md.get("total",         float("nan")))

    record["feature_sim"]   = float(mm.get("feature_sim",   float("nan")))
    # camera_traj 合并值不再进 record（避免被误当作 benchmark 列）；
    # 只保留 rot/trans 两分量。
    record["camera_rot"]    = float(mm.get("camera_rot",    float("nan")))
    record["camera_trans"]  = float(mm.get("camera_trans",  float("nan")))
    record["video_quality"] = float(mm.get("video_quality", float("nan")))

    # ── Standard benchmarks ───────────────────────────────────
    eval_dir = gen_dir / "eval"
    inter_dir = gen_dir / "intermediates"
    for fname, path_keys, col in BENCHMARK_KEYS:
        fdata = load_json_safe(eval_dir / fname)
        if not fdata:
            old = _BENCH_FILE_FALLBACK.get(fname)
            if old:
                fdata = load_json_safe(eval_dir / old)
        # intermediates/ 作为最后 fallback（VBench 目前只在 intermediates 下）
        if not fdata and fname in _BENCH_INTERMEDIATE_FALLBACK:
            fdata = load_json_safe(inter_dir / fname)
        val = _get_nested(fdata, path_keys)
        record[col] = float(val) if val is not None else float("nan")

    # ── Global point cloud (4 align × 3 metrics) ──────────────
    gpc_data = load_json_safe(
        eval_dir / f"global_point_cloud_{RECON_EVAL_SUFFIX}.json")
    if not gpc_data:
        gpc_data = load_json_safe(eval_dir / "global_point_cloud.json")
    if not gpc_data:
        old = load_json_safe(eval_dir / "point_cloud.json")
        if old:
            gpc_data = {"umeyama": old}

    for mode in PC_ALIGN_MODES:
        lbl = PC_ALIGN_LABELS[mode]
        if isinstance(gpc_data, dict):
            mode_data = gpc_data.get(mode, {})
            if not mode_data and "chamfer_distance" in gpc_data:
                mode_data = gpc_data  # 单 align 兼容
        else:
            mode_data = {}
        for metric in PC_METRICS:
            short = PC_METRIC_SHORT[metric]
            val = mode_data.get(metric, float("nan"))
            record[f"GlobalPC/{lbl}/{short}"] = float(val)

    # ── Object point cloud (4 align × 3 metrics) ──────────────
    opc_data = load_json_safe(
        eval_dir / f"object_point_cloud_{RECON_EVAL_SUFFIX}.json")
    if not opc_data:
        opc_data = load_json_safe(eval_dir / "object_point_cloud.json")
    for mode in PC_ALIGN_MODES:
        lbl = PC_ALIGN_LABELS[mode]
        if isinstance(opc_data, dict):
            mode_data = opc_data.get(mode, {})
            if not mode_data and "summary" in opc_data:
                mode_data = opc_data  # 单 align 兼容
        else:
            mode_data = {}
        summary = mode_data.get("summary", {}) if isinstance(mode_data, dict) else {}
        for metric in PC_METRICS:
            short = PC_METRIC_SHORT[metric]
            v = summary.get(metric)
            if isinstance(v, dict):
                v = v.get("mean")
            record[f"ObjPC/{lbl}/{short}"] = float(v) if v is not None else float("nan")

    return record


def collect_all_records(entries: list, n_workers: int = 32) -> list:
    """收集所有 gen 视频的 record（bucket I/O 并行化）。"""
    import concurrent.futures as _cf
    tasks = []
    for entry in entries:
        for gv in entry["gen_videos"]:
            tasks.append((entry, gv))
    records = []
    n_skip = 0
    n_total = len(tasks)
    log(f"[Phase 4] 并行读取 {n_total} 条记录 (workers={n_workers}) ...")
    with _cf.ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(collect_record, e, gv) for e, gv in tasks]
        for i, fut in enumerate(_cf.as_completed(futs)):
            rec = fut.result()
            if rec is None:
                n_skip += 1
            else:
                records.append(rec)
            if (i + 1) % 500 == 0 or (i + 1) == n_total:
                log(f"  [{i+1}/{n_total}] 已处理 (ok={len(records)}, skip={n_skip})")
    log(f"[Phase 4] 收集到 {len(records)} 条记录，跳过 {n_skip} 条")
    return records


# ═══════════════════════════════════════════════════════════════════
# 3. Pearson 相关性计算
# ═══════════════════════════════════════════════════════════════════

def compute_advantages(records: list) -> list:
    """
    对每个 sample 的所有 rollout 计算组内优势：
    advantage = value - mean(values in same sample)
    返回包含 _adv_<key> 字段的记录副本。
    """
    from collections import defaultdict

    groups = defaultdict(list)
    for i, rec in enumerate(records):
        groups[(rec["dataset"], rec["sample_id"])].append(i)

    num_fields = [k for k in records[0].keys()
                  if k not in ("dataset", "sample_id", "gen_idx")]

    adv_records = [dict(r) for r in records]

    for idx_list in groups.values():
        for field in num_fields:
            vals = [records[i].get(field, float("nan")) for i in idx_list]
            arr = np.array(vals, dtype=float)
            valid = arr[~np.isnan(arr)]
            mean_val = float(np.mean(valid)) if len(valid) > 0 else float("nan")
            for idx in idx_list:
                v = records[idx].get(field, float("nan"))
                if np.isnan(v) or np.isnan(mean_val):
                    adv_records[idx][f"_adv_{field}"] = float("nan")
                else:
                    adv_records[idx][f"_adv_{field}"] = float(v - mean_val)

    return adv_records


def pearson_r_p(x: np.ndarray, y: np.ndarray):
    """计算 Pearson r 和 p-value，自动过滤 NaN 对。"""
    from scipy import stats
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return float("nan"), float("nan")
    r, p = stats.pearsonr(x[mask], y[mask])
    return float(r), float(p)


def sig_stars(p: float) -> str:
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def compute_contribution_weights(adv_records: list) -> dict:
    """
    计算每个 reward 分项对各自 total 的贡献权重。

    对于绑定到单个 total (total_key 非 None) 的列：
      contrib[col][total_key] = mean(|weight * adv_col|) / mean(|adv_total|)
    对于共享 reward (feature_sim, camera_traj, video_quality)，
    分别对 3 个 total 计算贡献。
    对于 total 行本身，不计算贡献。

    返回:
      {
        "geo_sem_ff":  {"total_ff": 0.xx},
        "feature_sim": {"total_ff": 0.xx, "total_f3": 0.xx, "total_ap": 0.xx},
        ...
      }
    """
    result = {}
    # pre-compute adv arrays for each total
    total_adv = {}
    for tk in TOTAL_KEYS:
        arr = np.array([r.get(f"_adv_{tk}", float("nan")) for r in adv_records],
                       dtype=float)
        denom = np.nanmean(np.abs(arr))
        total_adv[tk] = (arr, float(denom) if not np.isnan(denom) else 1.0)

    for col, _lbl, total_key, weight in REWARD_COLS:
        if weight is None:
            continue  # skip total rows

        adv_col = np.array([r.get(f"_adv_{col}", float("nan")) for r in adv_records],
                           dtype=float)
        weighted = np.abs(adv_col * weight)
        numer = float(np.nanmean(weighted)) if not np.all(np.isnan(weighted)) else 0.0

        if total_key is not None:
            # bound to a specific total
            _, denom = total_adv[total_key]
            result[col] = {total_key: numer / (denom + 1e-12)}
        else:
            # shared: compute for all 3 totals
            result[col] = {}
            for tk in TOTAL_KEYS:
                _, denom = total_adv[tk]
                result[col][tk] = numer / (denom + 1e-12)

    return result


def _within_sample_pearson_fisher(
    records: list, col: str, bm: str, groups: dict,
    min_pairs: int = 4, min_samples: int = 3,
):
    """
    Within-sample Pearson + Fisher-z 加权聚合。

    对每个 sample（prompt）内部的 rollouts 独立算 1 个 Pearson r_s，
    然后用 Fisher-z 变换加权聚合（权重 = n_s - 3，即每个 r 的自由度）。

    返回 (r_agg, p_agg, n_valid_samples)。
    """
    from scipy import stats

    zs = []
    ws = []
    for sid, idx_list in groups.items():
        r_vals = np.array([records[i].get(col, float("nan")) for i in idx_list],
                          dtype=float)
        b_vals = np.array([records[i].get(bm,  float("nan")) for i in idx_list],
                          dtype=float)
        mask = ~(np.isnan(r_vals) | np.isnan(b_vals))
        if mask.sum() < min_pairs:
            continue
        x, y = r_vals[mask], b_vals[mask]
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            continue
        r_s = float(np.corrcoef(x, y)[0, 1])
        if not np.isfinite(r_s):
            continue
        # Fisher z 变换（clip 防 inf）
        z_s = 0.5 * np.log(
            (1 + np.clip(r_s, -0.9999, 0.9999))
            / (1 - np.clip(r_s, -0.9999, 0.9999))
        )
        zs.append(z_s)
        ws.append(max(int(mask.sum()) - 3, 1))

    if len(zs) < min_samples:
        return float("nan"), float("nan"), len(zs)

    zs = np.asarray(zs, dtype=float)
    ws = np.asarray(ws, dtype=float)
    z_agg = float(np.sum(zs * ws) / np.sum(ws))
    r_agg = float(np.tanh(z_agg))

    # 聚合 z 的标准误 ≈ 1 / sqrt(sum(ws))；双侧正态近似 p 值
    se = 1.0 / np.sqrt(np.sum(ws))
    z_score = z_agg / se if se > 0 else 0.0
    p_agg = float(2.0 * (1.0 - stats.norm.cdf(abs(z_score))))

    return r_agg, p_agg, int(len(zs))


def compute_correlation_table(adv_records: list) -> tuple:
    """
    计算 14 × 35 的 **within-sample** Pearson 相关矩阵。

    对每个 (reward_col, benchmark_col)：
      1. 对每个 sample 的 8 个 rollouts 独立算 Pearson r_s（n=8）
      2. 把各 sample 的 r_s 通过 Fisher-z 加权聚合（权重 = 每个 r_s 的自由度 n_s-3）
      3. 反变换回 r_agg；p 由加权 z 的近似标准误给出

    注：这里和旧版（对 adv 做 pooled Pearson）不同。新版每个 sample 内独立归一化，
       跨 sample 平等聚合，避免 "难场景方差大导致信号被稀释" 的问题。
       advantage 字段 (`_adv_*`) 仍然用于 compute_contribution_weights，不影响相关性计算。

    返回 (table, bm_cols):
      table = {reward_col: {bm_col: {"r": r_agg, "p": p_agg, "n_samples": n}, ...}, ...}
      bm_cols = sorted list of benchmark column names
    """
    from collections import defaultdict

    reward_keys_set = ALL_REWARD_KEYS
    meta_keys = {"dataset", "sample_id", "gen_idx"}

    all_keys = set(adv_records[0].keys())
    bm_cols = sorted([
        k for k in all_keys
        if not k.startswith("_adv_")
        and k not in meta_keys
        and k not in reward_keys_set
    ])

    # 按 sample 分组（索引列表）
    groups = defaultdict(list)
    for i, rec in enumerate(adv_records):
        groups[(rec["dataset"], rec["sample_id"])].append(i)

    table = {}
    for col, _, _, _ in REWARD_COLS:
        table[col] = {}
        for bm in bm_cols:
            r_agg, p_agg, n_s = _within_sample_pearson_fisher(
                adv_records, col, bm, groups)
            table[col][bm] = {"r": r_agg, "p": p_agg, "n_samples": n_s}

    return table, bm_cols


def run_analysis(records: list, dataset_label: str) -> dict:
    """对一组 records 跑完整分析，返回结果 dict。"""
    if len(records) < 2:
        log(f"  [WARN] {dataset_label}: 记录数不足 ({len(records)})，跳过")
        return {}

    adv_records = compute_advantages(records)
    table, bm_cols = compute_correlation_table(adv_records)
    contrib = compute_contribution_weights(adv_records)

    return {
        "dataset":   dataset_label,
        "n_records": len(records),
        "table":     table,
        "bm_cols":   bm_cols,
        "contrib":   contrib,
    }


# ═══════════════════════════════════════════════════════════════════
# 4. 可视化
# ═══════════════════════════════════════════════════════════════════

def _annotate_bars(ax, bars):
    """在柱状图上标注数值。"""
    for bar in bars:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h + (0.02 if h >= 0 else -0.07),
                    f"{h:.2f}", ha="center", va="bottom",
                    fontsize=7, rotation=0)


def make_heatmap(analysis_result: dict, out_path: Path):
    """
    生成 Pearson 热力图 + 贡献权重列（3 列，每 total 模式一列）。

    布局:
      行 = 14 个 reward 列
      列 = 35 个 benchmark 列 + 3 列贡献权重（FF / F3 / AP）
      颜色 = Pearson r
      文字 = r 值 + 显著性星号
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    if not analysis_result:
        return

    ds_label  = analysis_result["dataset"]
    table     = analysis_result["table"]
    bm_cols   = analysis_result["bm_cols"]
    contrib   = analysis_result["contrib"]
    n_records = analysis_result["n_records"]

    reward_cols   = [col for col, _, _, _ in REWARD_COLS]
    reward_labels = {col: lbl for col, lbl, _, _ in REWARD_COLS}

    n_rows = len(reward_cols)
    n_bm   = len(bm_cols)
    n_contrib = len(TOTAL_KEYS)  # 3 contrib columns

    # ── 构建矩阵（对 "越小越好" 的 benchmark 列翻转 r 的符号）────────
    # 目的：可视化上统一成 "r > 0 / 红色 = reward 与 benchmark 的 good 方向正相关"。
    # 原始 r 仍保留在 JSON 中未动，只在此处用于绘图。
    r_matrix = np.full((n_rows, n_bm), float("nan"))
    p_matrix = np.full((n_rows, n_bm), float("nan"))
    bm_direction = np.array(
        [BENCHMARK_DIRECTION.get(bm, +1) for bm in bm_cols], dtype=float)

    for i, col in enumerate(reward_cols):
        for j, bm in enumerate(bm_cols):
            cell = table.get(col, {}).get(bm, {})
            r_raw = cell.get("r", float("nan"))
            r_matrix[i, j] = r_raw * bm_direction[j]
            p_matrix[i, j] = cell.get("p", float("nan"))

    # x 轴 label：对翻转过的列加 "↓" 标签，方便一眼识别原方向
    bm_display = [
        (bm + "  (↓)") if BENCHMARK_DIRECTION.get(bm, +1) < 0 else bm
        for bm in bm_cols
    ]

    # contrib matrix: (n_rows, 3)  — contrib of row col wrt each total
    contrib_matrix = np.full((n_rows, n_contrib), float("nan"))
    for i, (col, _, total_key, weight) in enumerate(REWARD_COLS):
        if weight is None:
            continue  # total rows: no contrib
        col_contribs = contrib.get(col, {})
        for j, tk in enumerate(TOTAL_KEYS):
            v = col_contribs.get(tk)
            if v is not None:
                contrib_matrix[i, j] = v

    # ── 绘图 ─────────────────────────────────────────────────────────
    fig_w = max(28, n_bm * 0.52 + n_contrib * 1.2 + 4)
    fig_h = max(8, n_rows * 0.85 + 3)
    width_ratios = [n_bm] + [1] * n_contrib
    fig, axes = plt.subplots(
        1, 1 + n_contrib,
        figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": width_ratios, "wspace": 0.02},
        constrained_layout=False,
    )
    ax_main = axes[0]
    ax_contribs = axes[1:]

    # ── 主热力图 ─────────────────────────────────────────────────────
    cmap = plt.cm.RdBu_r
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im = ax_main.imshow(r_matrix, cmap=cmap, norm=norm,
                        aspect="auto", interpolation="nearest")

    for i in range(n_rows):
        for j in range(n_bm):
            r = r_matrix[i, j]
            p = p_matrix[i, j]
            if not np.isnan(r):
                txt = f"{r:.2f}{sig_stars(p)}"
                color = "white" if abs(r) > 0.5 else "black"
                ax_main.text(j, i, txt, ha="center", va="center",
                             fontsize=6, color=color, fontweight="bold")

    ax_main.set_xticks(range(n_bm))
    ax_main.set_xticklabels(bm_display, rotation=55, ha="right", fontsize=6.5)
    ax_main.set_yticks(range(n_rows))
    ax_main.set_yticklabels([reward_labels[c] for c in reward_cols], fontsize=8)
    # 取任一格子的 n_samples 作为 sample 总数（每格相同或相近）
    n_samples_any = 0
    for bm in bm_cols:
        cell = table.get(reward_cols[0], {}).get(bm, {})
        n_samples_any = max(n_samples_any, int(cell.get("n_samples", 0) or 0))

    ax_main.set_title(
        f"Within-Sample Pearson (Fisher-z aggregated) — {ds_label}  "
        f"[{n_samples_any} samples, {n_records} rollouts]\n"
        f"Per-sample Pearson on 8 rollouts → Fisher-z weighted mean over samples; "
        f"sign-flipped for ↓ metrics; red = reward aligns with better benchmark",
        fontsize=10, pad=10)
    cbar = plt.colorbar(im, ax=ax_main, fraction=0.015, pad=0.01)
    cbar.set_label("Pearson r", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # ── 贡献权重列（每个 total 模式一列）────────────────────────────
    for j, (ax_c, tk) in enumerate(zip(ax_contribs, TOTAL_KEYS)):
        col_data = contrib_matrix[:, j].copy()
        col_disp = np.where(np.isnan(col_data), 0.0, col_data)

        vmax = max(1.0, float(np.nanmax(col_data[~np.isnan(col_data)]))
                   if not np.all(np.isnan(col_data)) else 1.0)
        ax_c.imshow(col_disp[:, np.newaxis],
                    cmap="YlOrRd", vmin=0, vmax=vmax,
                    aspect="auto", interpolation="nearest")
        for i, v in enumerate(col_data):
            if not np.isnan(v):
                ax_c.text(0, i, f"{v:.2f}x", ha="center", va="center",
                          fontsize=7, fontweight="bold")
        mode_lbl = GEO_MODE_LABEL[GEO_MODES[j]]
        ax_c.set_xticks([0])
        ax_c.set_xticklabels([f"Contrib\n/{mode_lbl}"], fontsize=7, rotation=30)
        ax_c.set_yticks([])

    fig.subplots_adjust(left=0.10, right=0.97, top=0.92, bottom=0.28)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  保存热力图: {out_path}")


def make_pc_alignment_comparison(results_by_dataset: dict, out_path: Path):
    """
    点云对齐方式对比图：
    4 种对齐方式 × 3 个 total 的 Pearson r（chamfer / accuracy / completeness）。
    每个数据集（re10k / dl3dv / combined）一行，Global PC / Object PC 各一列。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = [k for k in ["re10k", "dl3dv", "combined"]
                if k in results_by_dataset and results_by_dataset[k]]
    if not datasets:
        return

    # 每个数据集 2 列 (global / object)，每个 total 模式 3 组柱状图
    n_total = len(TOTAL_KEYS)
    align_labels = [PC_ALIGN_LABELS[m] for m in PC_ALIGN_MODES]

    fig, axes = plt.subplots(
        len(datasets) * n_total, 2,
        figsize=(16, 3.5 * len(datasets) * n_total),
        squeeze=False,
    )
    fig.suptitle("Point Cloud Alignment Comparison\n"
                 "(Pearson r with each total_mode advantage, "
                 "sign-flipped so positive = reward agrees with better PC)",
                 fontsize=11)

    row_idx = 0
    for ds in datasets:
        res = results_by_dataset[ds]
        table = res.get("table", {})

        for tk in TOTAL_KEYS:
            tk_row = table.get(tk, {})
            m_lbl = GEO_MODE_LABEL[GEO_MODES[TOTAL_KEYS.index(tk)]]

            for col_idx, (pc_prefix, pc_label) in enumerate(
                    [("GlobalPC", "Global PC"), ("ObjPC", "Object PC")]):
                ax = axes[row_idx][col_idx]
                x = np.arange(len(PC_ALIGN_MODES))
                w = 0.25
                colors = ["steelblue", "salmon", "seagreen"]
                metrics_short = [PC_METRIC_SHORT[m] for m in PC_METRICS]
                metric_labels = ["accuracy", "completeness", "chamfer"]

                for mi, (ms, ml) in enumerate(zip(metrics_short, metric_labels)):
                    vals = []
                    for mode in PC_ALIGN_MODES:
                        lbl = PC_ALIGN_LABELS[mode]
                        col_name = f"{pc_prefix}/{lbl}/{ms}"
                        r_raw = tk_row.get(col_name, {}).get("r", float("nan"))
                        # 点云指标都是 lower-is-better → 翻转符号
                        r_disp = r_raw * BENCHMARK_DIRECTION.get(col_name, +1)
                        vals.append(r_disp)
                    offset = (mi - 1) * w
                    bars = ax.bar(x + offset, vals, w,
                                  label=ml, color=colors[mi], alpha=0.8)
                    _annotate_bars(ax, bars)

                ax.set_xticks(x)
                ax.set_xticklabels(align_labels, fontsize=8)
                ax.set_ylim(-1, 1)
                ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
                ax.set_title(f"{ds} / Total({m_lbl}) — {pc_label}", fontsize=9)
                ax.set_ylabel("Pearson r", fontsize=8)
                ax.legend(fontsize=7)

            row_idx += 1

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  保存点云对齐对比图: {out_path}")


def make_geo_mode_comparison(results_by_dataset: dict, out_path: Path):
    """
    Geo 对比模式比较图：
    对每个 benchmark 指标，显示 3 种 geo 模式的 geo_semantic / geo_global Pearson r，
    方便直观对比哪种帧对比方式与 benchmark 相关性最高。

    每个数据集一行，geo_semantic / geo_global 各一列。
    X 轴：benchmark 指标（有代表性的子集，避免过多）
    柱组：3 种 geo 模式
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = [k for k in ["re10k", "dl3dv", "combined"]
                if k in results_by_dataset and results_by_dataset[k]]
    if not datasets:
        return

    # 选取有代表性的 benchmark 指标子集（避免图太宽）
    BM_SUBSET = [
        "PSNR", "SSIM", "LPIPS",
        "VBench/i2v_subj", "VBench/img_qual",
        "CamPose/rot_auc30", "CamPose/pose_auc30",
        "VideoAlign/Overall",
        "GlobalPC/Cam/chamfer", "GlobalPC/FF/chamfer",
        "GlobalPC/Ume/chamfer", "GlobalPC/ICP/chamfer",
    ]

    x = np.arange(len(BM_SUBSET))
    w = 0.25
    mode_colors = ["steelblue", "darkorange", "seagreen"]
    mode_labels_list = [GEO_MODE_LABEL[m] for m in GEO_MODES]

    fig, axes = plt.subplots(
        len(datasets), 2,
        figsize=(max(18, len(BM_SUBSET) * 1.4), 5 * len(datasets)),
        squeeze=False,
    )
    fig.suptitle("Geo Compare Mode Comparison\n"
                 "(Pearson r with benchmark metrics, FF vs F3 vs AP; "
                 "sign-flipped for lower-is-better metrics marked ↓)",
                 fontsize=11)

    for row_idx, ds in enumerate(datasets):
        res = results_by_dataset[ds]
        table = res.get("table", {})

        for col_idx, geo_type in enumerate(["sem", "glob"]):
            ax = axes[row_idx][col_idx]
            geo_label = "GeoSem" if geo_type == "sem" else "GeoGlob"

            for mi, m in enumerate(GEO_MODES):
                sfx = GEO_MODE_SUFFIX[m]
                col_key = f"geo_{geo_type}_{sfx}"
                col_row = table.get(col_key, {})
                vals = []
                for bm in BM_SUBSET:
                    r_raw = col_row.get(bm, {}).get("r", float("nan"))
                    vals.append(r_raw * BENCHMARK_DIRECTION.get(bm, +1))
                offset = (mi - 1) * w
                bars = ax.bar(x + offset, vals, w,
                              label=mode_labels_list[mi],
                              color=mode_colors[mi], alpha=0.85)
                _annotate_bars(ax, bars)

            bm_subset_disp = [
                (bm + " (↓)") if BENCHMARK_DIRECTION.get(bm, +1) < 0 else bm
                for bm in BM_SUBSET
            ]
            ax.set_xticks(x)
            ax.set_xticklabels(bm_subset_disp, rotation=45, ha="right", fontsize=7.5)
            ax.set_ylim(-1, 1)
            ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
            ax.set_title(f"{ds} — {geo_label} vs Benchmark (mode comparison)",
                         fontsize=9)
            ax.set_ylabel("Pearson r", fontsize=8)
            ax.legend(fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  保存 geo 模式对比图: {out_path}")


# ═══════════════════════════════════════════════════════════════════
# 5. 保存 JSON 结果
# ═══════════════════════════════════════════════════════════════════

def save_json_results(results_by_dataset: dict, out_dir: Path):
    """将相关性分析结果保存为 JSON 文件。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    for ds, res in results_by_dataset.items():
        if not res:
            continue
        out_path = out_dir / f"correlation_{ds}.json"
        serializable = {
            "dataset":   res["dataset"],
            "n_records": res["n_records"],
            "table":     res["table"],
            "bm_cols":   res["bm_cols"],
            "contrib":   res["contrib"],
        }
        with open(str(out_path), "w") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2,
                      default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else x)
        log(f"  保存 JSON: {out_path}")


# ═══════════════════════════════════════════════════════════════════
# 6. 主流程
# ═══════════════════════════════════════════════════════════════════

def phase4_main(output_root: Path, out_dir: Path):
    """Phase 4 主函数：数据收集 -> 分析 -> 可视化。"""
    log(f"[Phase 4] 扫描 {output_root} ...")
    entries = scan_output_root(str(output_root))
    log(f"找到 {len(entries)} 个样本")

    records = collect_all_records(entries)
    if not records:
        log("[Phase 4] 没有有效记录，退出")
        return

    re10k_recs    = [r for r in records if r["dataset"] == "re10k"]
    dl3dv_recs    = [r for r in records if r["dataset"] == "dl3dv"]
    combined_recs = records

    log(f"[Phase 4] re10k={len(re10k_recs)}  "
        f"dl3dv={len(dl3dv_recs)}  combined={len(combined_recs)}")

    results_by_dataset = {}
    for ds, recs in [("re10k", re10k_recs),
                     ("dl3dv", dl3dv_recs),
                     ("combined", combined_recs)]:
        log(f"[Phase 4] 分析 {ds} ({len(recs)} 条) ...")
        results_by_dataset[ds] = run_analysis(recs, ds)

    # 保存 JSON（三份）
    save_json_results(results_by_dataset, out_dir / "json")

    # 生成可视化
    log("[Phase 4] 生成可视化 ...")
    for ds, res in results_by_dataset.items():
        if not res:
            continue
        make_heatmap(res, out_dir / f"heatmap_{ds}.png")

    make_pc_alignment_comparison(results_by_dataset,
                                 out_dir / "pc_alignment_comparison.png")
    make_geo_mode_comparison(results_by_dataset,
                             out_dir / "geo_mode_comparison.png")

    log(f"[Phase 4] 完成！输出目录: {out_dir}")


def run_phase(phase_num: int, args):
    """运行单个阶段。"""
    script_dir = _THIS_DIR

    if phase_num == 1:
        cmd = [
            sys.executable, "-u",
            str(script_dir / "run_sam3_recompute.py"),
            "--output_root", args.output_root,
            "--n_gpus", str(args.n_gpus),
        ]
        if args.gpu_ids:
            cmd += ["--gpu_ids", args.gpu_ids]
        subprocess.run(cmd, check=True)

    elif phase_num == 2:
        # 多模式 geo reward
        cmd = [
            sys.executable, "-u",
            str(script_dir / "run_reward_multimode.py"),
            "--output_root", args.output_root,
        ]
        if args.gpu_ids:
            cmd += ["--gpu_ids", args.gpu_ids]
        else:
            cmd += ["--device", args.device]
        if args.force_reward:
            cmd.append("--force")
        subprocess.run(cmd, check=True)

    elif phase_num == 3:
        cmd = [
            sys.executable, "-u",
            str(script_dir / "run_benchmark_recompute.py"),
            "--output_root", args.output_root,
            "--gpu", str(args.gpu),
            "--align", args.align,
            "--n_fps", str(args.n_fps),
            "--n_workers", str(args.n_workers),
        ]
        subprocess.run(cmd, check=True)

    elif phase_num == 4:
        phase4_main(Path(args.output_root), Path(args.out_dir))


def main():
    parser = argparse.ArgumentParser(
        description="Reward-Benchmark Pearson 相关性分析（多模式 geo）总调度脚本")
    parser.add_argument("--output_root", required=True,
                        help="测试数据根目录")
    parser.add_argument("--out_dir",
                        default=str(_THIS_DIR / "results"),
                        help="可视化输出目录（默认 ./results）")
    parser.add_argument("--phase", default="4",
                        help="运行阶段: 1/2/3/4/all（默认 4）")

    # Phase 1 参数
    parser.add_argument("--n_gpus", type=int, default=4)
    parser.add_argument("--gpu_ids", type=str, default=None,
                        help="手动指定 GPU ID，逗号分隔，如 '0,1,2,3'")

    # Phase 2 参数
    parser.add_argument("--device", default="cuda:0",
                        help="单 GPU 时的设备（有 gpu_ids 则忽略）")
    parser.add_argument("--force_reward", action="store_true",
                        help="Phase 2: 强制重算（忽略已有 reward_multimode.json）")

    # Phase 3 参数
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--align", default="all_align")
    parser.add_argument("--n_fps", type=int, default=20000)
    parser.add_argument("--n_workers", type=int, default=8,
                        help="Phase 3 点云/camera_pose 并行进程数（默认 8）")

    args = parser.parse_args()

    if args.phase == "all":
        phases = [1, 2, 3, 4]
    else:
        phases = [int(p.strip()) for p in args.phase.split(",")]

    for p in phases:
        log(f"\n{'='*60}")
        log(f">>> 开始 Phase {p}")
        log(f"{'='*60}")
        run_phase(p, args)
        log(f">>> Phase {p} 完成")


if __name__ == "__main__":
    main()
