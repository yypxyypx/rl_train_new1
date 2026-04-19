#!/usr/bin/env python3
"""
run_benchmark.py — Benchmark 主调度器。

支持选择性指标评估与智能模型复用。

指标选择粒度：
  all                              全部评测
  video_quality                    PSNR + VBench
  video_quality.psnr               仅 PSNR/SSIM/LPIPS
  video_quality.vbench             仅 VBench
  reward                           所有 reward 指标
  reward.camera_pose               旋转 AUC + 平移
  reward.depth_reprojection        物体 + 全局
  reward.depth_reprojection.object 仅物体级
  reward.depth_reprojection.global 仅全局
  reward.videoalign                VideoAlign
  reward.feature_sim               DINOv2 特征相似度
  reconstruction                   全局 + 物体级点云
  reconstruction.global            仅全局点云
  reconstruction.object            仅物体级点云
  reconstruction.both              全局 + 物体级（同 reconstruction）

用法：
  python run_benchmark.py --output_root /path/to/output --metrics all --gpu 0
  python run_benchmark.py --output_root /path/to/output \\
      --metrics video_quality.psnr,reward.camera_pose,reconstruction.global
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np

from common import scan_output_root, save_json, load_json, log
from common.utils import parse_camera_txt, to_4x4, extract_frames
from common.intermediate import IntermediateManager, METRIC_TO_DEPS


# ═══════════════════ 指标展开 ══════════════════════════════════

EXPAND_RULES = {
    "all": [
        "video_quality.psnr", "video_quality.vbench",
        "reward.camera_pose", "reward.depth_reprojection.both",
        "reward.videoalign", "reward.feature_sim",
        "reconstruction.both",
    ],
    "video_quality": ["video_quality.psnr", "video_quality.vbench"],
    "reward": [
        "reward.camera_pose", "reward.depth_reprojection.both",
        "reward.videoalign", "reward.feature_sim",
    ],
    "reward.depth_reprojection": ["reward.depth_reprojection.both"],
    "reconstruction": ["reconstruction.both"],
}


def expand_metrics(metrics_str: str) -> list:
    """将逗号分隔的指标字符串展开为最终指标列表。"""
    raw = [m.strip() for m in metrics_str.split(",") if m.strip()]
    expanded = []
    for m in raw:
        if m in EXPAND_RULES:
            expanded.extend(EXPAND_RULES[m])
        else:
            expanded.append(m)
    return list(dict.fromkeys(expanded))


# ═══════════════════ 评估执行器 ═══════════════════════════════

def _get_video_size(video_path: str):
    cap = cv2.VideoCapture(video_path)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    return H, W


def _recon_eval_names(recon_json_suffix: str) -> tuple:
    """返回 (global_point_cloud 文件名, object_point_cloud 文件名)。"""
    suf = (recon_json_suffix or "").strip()
    if suf and not suf.startswith("_"):
        suf = "_" + suf
    if suf:
        return f"global_point_cloud{suf}.json", f"object_point_cloud{suf}.json"
    return "global_point_cloud.json", "object_point_cloud.json"


def run_metrics(metrics: list, entries: list, device: str, align: str,
                n_fps: int = 20000, conf_thresh: float = 0.0,
                force_recon: bool = False,
                recon_json_suffix: str = ""):
    """按指标列表逐一执行评估。

    recon_json_suffix: 非空时写入新文件名（如 mm -> global_point_cloud_mm.json）， 适合 bucket 禁止删除旧 eval 的场景。
    force_recon: 为 True 时即使输出文件已存在也重算（覆盖写入）。
    """
    g_fname, o_fname = _recon_eval_names(recon_json_suffix)

    if "video_quality.psnr" in metrics:
        log("── 评估 video_quality.psnr ──")
        from video_quality.psnr_ssim_lpips import evaluate_psnr_ssim_lpips, build_lpips_model
        lpips_model = build_lpips_model(device)
        for entry in entries:
            for gv in entry["gen_videos"]:
                out = gv["gen_dir"] / "eval" / "psnr_ssim_lpips.json"
                if out.exists():
                    continue
                try:
                    result = evaluate_psnr_ssim_lpips(
                        entry["gt_video"], gv["video_path"], device, lpips_model)
                    save_json(str(out), result)
                    log(f"  PSNR={result['psnr']:.2f}  {Path(gv['video_path']).name}")
                except Exception as e:
                    log(f"  [错误] psnr {Path(gv['video_path']).name}: {e}")
        del lpips_model

    if "video_quality.vbench" in metrics:
        log("── 评估 video_quality.vbench ──")
        from video_quality.vbench_eval import evaluate_vbench
        for entry in entries:
            for gv in entry["gen_videos"]:
                out = gv["gen_dir"] / "eval" / "vbench.json"
                if out.exists():
                    continue
                vb = evaluate_vbench(str(gv["gen_dir"] / "intermediates" / "vbench.json"))
                if vb:
                    save_json(str(out), vb)

    if "reward.camera_pose" in metrics:
        log("── 评估 reward.camera_pose ──")
        from reward.camera_pose import evaluate_camera_pose
        for entry in entries:
            if not entry.get("camera_txt"):
                continue
            H, W = _get_video_size(entry["gt_video"])
            _, gt_c2w = parse_camera_txt(entry["camera_txt"], H, W)
            for gv in entry["gen_videos"]:
                out = gv["gen_dir"] / "eval" / "camera_pose.json"
                if out.exists():
                    continue
                da3_npz = gv["gen_dir"] / "intermediates" / "da3_pred.npz"
                if not da3_npz.exists():
                    continue
                try:
                    da3 = dict(np.load(str(da3_npz), allow_pickle=True))
                    result = evaluate_camera_pose(da3, gt_c2w)
                    save_json(str(out), result)
                    log(f"  rot_auc30={result.get('rotation_auc30', 'N/A'):.4f}  "
                        f"{Path(gv['video_path']).name}")
                except Exception as e:
                    log(f"  [错误] camera_pose: {e}")
                    traceback.print_exc()

    # Depth reprojection
    dr_modes = set()
    if "reward.depth_reprojection.both" in metrics:
        dr_modes = {"object", "global"}
    if "reward.depth_reprojection.object" in metrics:
        dr_modes.add("object")
    if "reward.depth_reprojection.global" in metrics:
        dr_modes.add("global")

    if dr_modes:
        log(f"── 评估 reward.depth_reprojection ({dr_modes}) ──")
        from reward.depth_reprojection import (
            evaluate_object_reprojection, evaluate_global_reprojection)
        for entry in entries:
            H, W = _get_video_size(entry["gt_video"])
            for gv in entry["gen_videos"]:
                out = gv["gen_dir"] / "eval" / "depth_reprojection.json"
                if out.exists():
                    continue
                da3_npz = gv["gen_dir"] / "intermediates" / "da3_pred.npz"
                if not da3_npz.exists():
                    continue
                da3 = dict(np.load(str(da3_npz), allow_pickle=True))
                result = {}
                try:
                    if "object" in dr_modes:
                        lm_npz = gv["gen_dir"] / "intermediates" / "label_maps.npz"
                        if lm_npz.exists():
                            lm = np.load(str(lm_npz), allow_pickle=True)["label_maps"]
                            r, d = evaluate_object_reprojection(
                                da3, lm, H, W, device=device)
                            result["object"] = {"reward": r, "details": d}
                    if "global" in dr_modes:
                        r, d = evaluate_global_reprojection(
                            str(da3_npz), H, W, device=device)
                        result["global"] = {"reward": r, "details": d}
                    save_json(str(out), result)
                except Exception as e:
                    log(f"  [错误] depth_reproj: {e}")
                    traceback.print_exc()

    if "reward.videoalign" in metrics:
        log("── 评估 reward.videoalign ──")
        from reward.videoalign_eval import evaluate_videoalign
        for entry in entries:
            for gv in entry["gen_videos"]:
                out = gv["gen_dir"] / "eval" / "videoalign.json"
                if out.exists():
                    continue
                va = evaluate_videoalign(str(gv["gen_dir"] / "intermediates" / "videoalign.json"))
                if va:
                    save_json(str(out), va)

    if "reward.feature_sim" in metrics:
        log("── 评估 reward.feature_sim ──")
        from reward.feature_sim import evaluate_feature_sim
        for entry in entries:
            for gv in entry["gen_videos"]:
                out = gv["gen_dir"] / "eval" / "feature_sim.json"
                if out.exists():
                    continue
                fs = evaluate_feature_sim(
                    str(gv["gen_dir"] / "intermediates" / "feature_sim_reward.json"))
                if fs:
                    save_json(str(out), fs)

    # Reconstruction
    recon_modes = set()
    if "reconstruction.both" in metrics:
        recon_modes = {"global", "object"}
    if "reconstruction.global" in metrics:
        recon_modes.add("global")
    if "reconstruction.object" in metrics:
        recon_modes.add("object")

    if recon_modes:
        log(f"── 评估 reconstruction ({recon_modes}, align={align}) ──")
        from reconstruction.global_point_cloud import (
            evaluate_global, evaluate_global_multi, evaluate_global_all_align,
            ALL_ALIGN_MODES,
        )
        from reconstruction.object_point_cloud import (
            evaluate_object, evaluate_object_multi, evaluate_object_all_align,
        )

        # 将 align 快捷名展开为具体模式列表
        _ALIGN_EXPAND = {
            "both_align": ("camera", "first_frame"),
            "all_align":  ALL_ALIGN_MODES,
        }
        align_modes = _ALIGN_EXPAND.get(align, (align,))
        is_multi = len(align_modes) > 1

        for entry in entries:
            camera_txt = entry.get("camera_txt")
            gt_depth_npz = entry.get("gt_depth_npz")
            gt_inter = entry["sample_dir"] / "gt_intermediates"
            gt_masks_npz = gt_inter / "gt_masks.npz"
            da3_gt_npz = gt_inter / "da3_gt.npz"

            has_gt_depth = gt_depth_npz and Path(gt_depth_npz).exists()
            has_da3_gt = da3_gt_npz.exists()
            if not has_gt_depth and not has_da3_gt:
                continue

            if has_gt_depth and camera_txt:
                H, W = _get_video_size(entry["gt_video"])
                gt_K, gt_c2w = parse_camera_txt(camera_txt, H, W)
                gt_depth = np.load(gt_depth_npz, allow_pickle=True)["depth"].astype(np.float64)
            else:
                da3_gt = dict(np.load(str(da3_gt_npz), allow_pickle=True))
                gt_depth = da3_gt["depth"].astype(np.float64)
                gt_K = da3_gt["intrinsics"].astype(np.float64)
                w2c = to_4x4(da3_gt["extrinsics"].astype(np.float64))
                gt_c2w = np.array([np.linalg.inv(w2c[i])[:3, :] for i in range(len(w2c))])

            for gv in entry["gen_videos"]:
                da3_pred = gv["gen_dir"] / "intermediates" / "da3_pred.npz"
                pred_masks = gv["gen_dir"] / "intermediates" / "pred_masks.npz"
                if not da3_pred.exists():
                    continue

                try:
                    if "global" in recon_modes:
                        out = gv["gen_dir"] / "eval" / g_fname
                        if force_recon or not out.exists():
                            if is_multi:
                                result = evaluate_global_multi(
                                    str(da3_pred), gt_depth, gt_K, gt_c2w,
                                    aligns=align_modes, n_fps=n_fps,
                                    conf_thresh=conf_thresh, device=device)
                            else:
                                result = evaluate_global(
                                    str(da3_pred), gt_depth, gt_K, gt_c2w,
                                    align=align_modes[0], n_fps=n_fps,
                                    conf_thresh=conf_thresh, device=device)
                            save_json(str(out), result)
                            log(f"  global_pc: {Path(gv['video_path']).name}")

                    if "object" in recon_modes:
                        out = gv["gen_dir"] / "eval" / o_fname
                        if (force_recon or not out.exists()) and pred_masks.exists() and gt_masks_npz.exists():
                            if is_multi:
                                result = evaluate_object_multi(
                                    str(pred_masks), str(gt_masks_npz),
                                    str(da3_pred), gt_depth, gt_K, gt_c2w,
                                    aligns=align_modes, n_fps_global=n_fps,
                                    conf_thresh=conf_thresh, device=device)
                            else:
                                result = evaluate_object(
                                    str(pred_masks), str(gt_masks_npz),
                                    str(da3_pred), gt_depth, gt_K, gt_c2w,
                                    align=align_modes[0], n_fps_global=n_fps,
                                    conf_thresh=conf_thresh, device=device)
                            save_json(str(out), result)
                            log(f"  object_pc: {Path(gv['video_path']).name}")
                except Exception as e:
                    log(f"  [错误] reconstruction: {e}")
                    traceback.print_exc()


# ═══════════════════ 汇总 ═════════════════════════════════════

def aggregate(entries: list, output_root: Path, recon_json_suffix: str = ""):
    """汇总所有评测结果到 results.jsonl 和 summary.json。"""
    log("汇总结果 ...")
    results_dir = output_root / "benchmark_results"
    results_dir.mkdir(exist_ok=True)

    g_fname, o_fname = _recon_eval_names(recon_json_suffix)

    all_records = []
    for entry in entries:
        for gv in entry["gen_videos"]:
            gen_dir = gv["gen_dir"]
            eval_dir = gen_dir / "eval"

            record = {
                "dataset": entry["dataset"],
                "sample_id": entry["sample_id"],
                "gen_idx": gv["idx"],
                "video_path": gv["video_path"],
            }

            for name in [
                "psnr_ssim_lpips", "vbench", "camera_pose",
                "depth_reprojection", "videoalign", "feature_sim",
            ]:
                data = load_json(str(eval_dir / f"{name}.json"))
                if data:
                    record[name] = data

            gpc = load_json(str(eval_dir / g_fname))
            if not gpc:
                gpc = load_json(str(eval_dir / "global_point_cloud.json"))
            if gpc:
                record["global_point_cloud"] = gpc

            opc = load_json(str(eval_dir / o_fname))
            if not opc:
                opc = load_json(str(eval_dir / "object_point_cloud.json"))
            if opc:
                record["object_point_cloud"] = opc

            all_records.append(record)

    # results.jsonl
    with open(str(results_dir / "results.jsonl"), "w") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    # summary
    summary = _compute_summary(all_records)
    save_json(str(results_dir / "summary.json"), summary)
    log(f"汇总完成: {len(all_records)} 条记录 → {results_dir}")

    # 打印关键指标
    for k, v in summary.items():
        if isinstance(v, dict) and "mean" in v:
            log(f"  {k}: mean={v['mean']:.4f}")

    return summary


def _compute_summary(records: list) -> dict:
    def _agg(vals):
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and v != v)]
        if not vals:
            return {}
        a = np.array(vals, dtype=float)
        return {"mean": float(np.mean(a)), "median": float(np.median(a)),
                "std": float(np.std(a)), "n": len(vals)}

    def _collect(records, *keys):
        vals = []
        for r in records:
            d = r
            for k in keys:
                if isinstance(d, dict):
                    d = d.get(k)
                else:
                    d = None
                    break
            vals.append(d)
        return vals

    summary = {}
    fields = [
        ("psnr_ssim_lpips", "psnr"),
        ("psnr_ssim_lpips", "ssim"),
        ("psnr_ssim_lpips", "lpips"),
        ("vbench", "i2v_subject"),
        ("vbench", "i2v_background"),
        ("vbench", "imaging_quality"),
        ("camera_pose", "rotation_auc30"),
        ("camera_pose", "rotation_auc15"),
        ("camera_pose", "rotation_auc05"),
        ("camera_pose", "translation_auc30"),
        ("camera_pose", "translation_auc15"),
        ("camera_pose", "pose_auc30"),
        ("camera_pose", "pose_auc15"),
        ("camera_pose", "translation_metric"),
        ("depth_reprojection", "object", "reward"),
        ("depth_reprojection", "global", "reward"),
        ("videoalign", "Overall"),
        ("feature_sim", "reward_feature_sim"),
    ]
    for keys in fields:
        vals = _collect(records, *keys)
        label = ".".join(keys)
        summary[label] = _agg(vals)

    return summary


# ═══════════════════ 主入口 ═══════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Benchmark 主调度器")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--metrics", default="all",
                        help="逗号分隔的指标列表，或 all/video_quality/reward/reconstruction")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", default=None, help="覆盖 GPU 设备 (默认 cuda:<gpu>)")
    parser.add_argument("--align", default="both_align",
                        choices=["camera", "first_frame", "umeyama", "icp",
                                 "both_align", "all_align"],
                        help="重建指标的对齐方式：单模式(camera/first_frame/umeyama/icp)"
                             " 或组合(both_align=camera+first_frame, all_align=全部四种)")
    parser.add_argument("--n_fps", type=int, default=20000,
                        help="重建点云 FPS 采样点数")
    parser.add_argument("--conf_thresh", type=float, default=0.0,
                        help="深度置信度阈值（0.0 = 不过滤）")
    parser.add_argument("--vbench_cache", default=None)
    parser.add_argument("--skip_intermediates", action="store_true",
                        help="跳过中间产物生成（假设已存在）")
    parser.add_argument("--aggregate_only", action="store_true",
                        help="只做汇总，不运行评测")
    parser.add_argument("--force_recon", action="store_true",
                        help="强制重算点云重建 JSON（覆盖已存在输出文件）")
    parser.add_argument("--recon_json_suffix", type=str, default="",
                        help="点云 eval 文件名后缀，如 mm -> global_point_cloud_mm.json（不写旧文件名，适合禁止删除的 bucket）")
    # 并行分片参数：由 run_benchmark_recompute.py 多进程时注入
    parser.add_argument("--shard_idx", type=int, default=None,
                        help="当前进程处理的分片索引（0-based）")
    parser.add_argument("--n_shards", type=int, default=None,
                        help="总分片数（与 --shard_idx 配合使用）")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if not output_root.exists():
        raise FileNotFoundError(f"output_root 不存在: {output_root}")

    device = args.device or f"cuda:{args.gpu}"
    metrics = expand_metrics(args.metrics)

    entries = scan_output_root(args.output_root)
    if not entries:
        log("未找到推理结果，退出")
        return

    # 分片：只处理属于本进程的 entries
    if args.shard_idx is not None and args.n_shards and args.n_shards > 1:
        entries = entries[args.shard_idx::args.n_shards]
        log(f"[分片 {args.shard_idx}/{args.n_shards}] 处理 {len(entries)} 个样本")
    else:
        log(f"展开后的指标: {metrics}")

    if args.aggregate_only:
        # aggregate 需要全量 entries，分片模式下不支持
        if args.shard_idx is None:
            aggregate(entries, output_root, recon_json_suffix=args.recon_json_suffix)
        return

    # 中间产物调度（分片模式下跳过，由主进程或 skip_intermediates 保证）
    if not args.skip_intermediates and args.shard_idx is None:
        manager = IntermediateManager(gpu=args.gpu, vbench_cache=args.vbench_cache)
        needed_deps = manager.resolve_deps(metrics)
        if needed_deps:
            log(f"需要的中间产物: {needed_deps}")
            tmp_dir = output_root / "_benchmark_tmp"
            manager.prepare(needed_deps, entries, tmp_dir)

    run_metrics(metrics, entries, device, args.align,
                n_fps=args.n_fps, conf_thresh=args.conf_thresh,
                force_recon=args.force_recon,
                recon_json_suffix=args.recon_json_suffix)

    # 汇总只在非分片模式（或最后的主进程汇总阶段）执行
    if args.shard_idx is None:
        aggregate(entries, output_root, recon_json_suffix=args.recon_json_suffix)
        log("Benchmark 完成")


if __name__ == "__main__":
    main()
