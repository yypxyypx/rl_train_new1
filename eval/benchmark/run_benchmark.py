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


def run_metrics(metrics: list, entries: list, device: str, align: str):
    """按指标列表逐一执行评估。"""

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
        log(f"── 评估 reconstruction ({recon_modes}) ──")
        from reconstruction.global_point_cloud import evaluate_global_both as eval_gl_both
        from reconstruction.object_point_cloud import evaluate_object_both_align

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
                        out = gv["gen_dir"] / "eval" / "global_point_cloud.json"
                        if not out.exists():
                            if align == "both_align":
                                result = eval_gl_both(
                                    str(da3_pred), gt_depth, gt_K, gt_c2w,
                                    device=device)
                            else:
                                from reconstruction.global_point_cloud import evaluate_global
                                result = evaluate_global(
                                    str(da3_pred), gt_depth, gt_K, gt_c2w,
                                    align=align, device=device)
                            save_json(str(out), result)
                            log(f"  global_pc: {Path(gv['video_path']).name}")

                    if "object" in recon_modes:
                        out = gv["gen_dir"] / "eval" / "object_point_cloud.json"
                        if not out.exists() and pred_masks.exists() and gt_masks_npz.exists():
                            if align == "both_align":
                                result = evaluate_object_both_align(
                                    str(pred_masks), str(gt_masks_npz),
                                    str(da3_pred), gt_depth, gt_K, gt_c2w,
                                    device=device)
                            else:
                                from reconstruction.object_point_cloud import evaluate_object
                                result = evaluate_object(
                                    str(pred_masks), str(gt_masks_npz),
                                    str(da3_pred), gt_depth, gt_K, gt_c2w,
                                    align=align, device=device)
                            save_json(str(out), result)
                            log(f"  object_pc: {Path(gv['video_path']).name}")
                except Exception as e:
                    log(f"  [错误] reconstruction: {e}")
                    traceback.print_exc()


# ═══════════════════ 汇总 ═════════════════════════════════════

def aggregate(entries: list, output_root: Path):
    """汇总所有评测结果到 results.jsonl 和 summary.json。"""
    log("汇总结果 ...")
    results_dir = output_root / "benchmark_results"
    results_dir.mkdir(exist_ok=True)

    all_records = []
    for entry in entries:
        for gv in entry["gen_videos"]:
            gen_dir = gv["gen_dir"]
            inter_dir = gen_dir / "intermediates"
            eval_dir = gen_dir / "eval"

            record = {
                "dataset": entry["dataset"],
                "sample_id": entry["sample_id"],
                "gen_idx": gv["idx"],
                "video_path": gv["video_path"],
            }

            # 从各 eval JSON 读取
            for name in [
                "psnr_ssim_lpips", "vbench", "camera_pose",
                "depth_reprojection", "videoalign", "feature_sim",
                "global_point_cloud", "object_point_cloud",
            ]:
                data = load_json(str(eval_dir / f"{name}.json"))
                if data:
                    record[name] = data

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
        ("camera_pose", "translation_metric"),
        ("camera_pose", "translation_auc30"),
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
                        choices=["camera", "first_frame", "both_align"],
                        help="重建指标的对齐方式")
    parser.add_argument("--vbench_cache", default=None)
    parser.add_argument("--skip_intermediates", action="store_true",
                        help="跳过中间产物生成（假设已存在）")
    parser.add_argument("--aggregate_only", action="store_true",
                        help="只做汇总，不运行评测")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if not output_root.exists():
        raise FileNotFoundError(f"output_root 不存在: {output_root}")

    device = args.device or f"cuda:{args.gpu}"
    metrics = expand_metrics(args.metrics)
    log(f"展开后的指标: {metrics}")

    entries = scan_output_root(args.output_root)
    if not entries:
        log("未找到推理结果，退出")
        return

    if args.aggregate_only:
        aggregate(entries, output_root)
        return

    # 中间产物调度
    if not args.skip_intermediates:
        manager = IntermediateManager(gpu=args.gpu, vbench_cache=args.vbench_cache)
        needed_deps = manager.resolve_deps(metrics)
        if needed_deps:
            log(f"需要的中间产物: {needed_deps}")
            tmp_dir = output_root / "_benchmark_tmp"
            manager.prepare(needed_deps, entries, tmp_dir)

    # 运行评测
    run_metrics(metrics, entries, device, args.align)

    # 汇总
    aggregate(entries, output_root)
    log("Benchmark 完成")


if __name__ == "__main__":
    main()
