#!/usr/bin/env python3
"""
run_reconstruction.py — 重建指标的独立运行入口。

三接口：--mode object / global / both
对齐方式：--align camera / first_frame / both_align
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import scan_output_root, save_json, log
from common.utils import parse_camera_txt
from reconstruction.global_point_cloud import (
    evaluate_global, evaluate_global_multi,
    ALL_ALIGN_MODES, VALID_ALIGN_CHOICES,
)
from reconstruction.object_point_cloud import (
    evaluate_object, evaluate_object_multi,
)


def _load_gt_depth(path: str) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    return data["depth"].astype(np.float64)


def main():
    parser = argparse.ArgumentParser(description="重建指标评估")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--mode", default="both", choices=["object", "global", "both"])
    parser.add_argument("--align", default="both_align",
                        choices=list(VALID_ALIGN_CHOICES),
                        help="对齐方式：单模式(camera/first_frame/umeyama/icp)"
                             " 或组合(both_align=camera+first_frame, all_align=全部四种)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_fps", type=int, default=20000)
    parser.add_argument("--conf_thresh", type=float, default=0.0,
                        help="深度置信度阈值（0.0 = 不过滤）")
    args = parser.parse_args()

    _ALIGN_EXPAND = {
        "both_align": ("camera", "first_frame"),
        "all_align":  ALL_ALIGN_MODES,
    }
    align_modes = _ALIGN_EXPAND.get(args.align, (args.align,))
    is_multi = len(align_modes) > 1

    entries = scan_output_root(args.output_root)
    if not entries:
        log("未找到推理结果")
        return

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

        # 解析 GT
        if has_gt_depth and camera_txt:
            cap = cv2.VideoCapture(entry["gt_video"])
            H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap.release()
            gt_K, gt_c2w = parse_camera_txt(camera_txt, H, W)
            gt_depth = _load_gt_depth(gt_depth_npz)
        else:
            da3_gt = dict(np.load(str(da3_gt_npz), allow_pickle=True))
            gt_depth = da3_gt["depth"].astype(np.float64)
            gt_K = da3_gt["intrinsics"].astype(np.float64)
            # DA3 extrinsics 为 w2c，取逆得 c2w
            from reconstruction.global_point_cloud import _to_4x4
            w2c = _to_4x4(da3_gt["extrinsics"].astype(np.float64))
            gt_c2w = np.array([np.linalg.inv(w2c[i])[:3, :] for i in range(len(w2c))])

        for gv in entry["gen_videos"]:
            da3_pred = gv["gen_dir"] / "intermediates" / "da3_pred.npz"
            pred_masks = gv["gen_dir"] / "intermediates" / "pred_masks.npz"
            if not da3_pred.exists():
                continue

            # 全局点云
            if args.mode in ("global", "both"):
                out_json = gv["gen_dir"] / "eval" / "global_point_cloud.json"
                if not out_json.exists():
                    if is_multi:
                        result = evaluate_global_multi(
                            str(da3_pred), gt_depth, gt_K, gt_c2w,
                            aligns=align_modes, n_fps=args.n_fps,
                            conf_thresh=args.conf_thresh, device=args.device)
                    else:
                        result = evaluate_global(
                            str(da3_pred), gt_depth, gt_K, gt_c2w,
                            align=align_modes[0], n_fps=args.n_fps,
                            conf_thresh=args.conf_thresh, device=args.device)
                    save_json(str(out_json), result)
                    log(f"  global: {Path(gv['video_path']).name}")

            # 物体级点云
            if args.mode in ("object", "both"):
                out_json = gv["gen_dir"] / "eval" / "object_point_cloud.json"
                if not out_json.exists() and pred_masks.exists() and gt_masks_npz.exists():
                    if is_multi:
                        result = evaluate_object_multi(
                            str(pred_masks), str(gt_masks_npz), str(da3_pred),
                            gt_depth, gt_K, gt_c2w, aligns=align_modes,
                            conf_thresh=args.conf_thresh, device=args.device)
                    else:
                        result = evaluate_object(
                            str(pred_masks), str(gt_masks_npz), str(da3_pred),
                            gt_depth, gt_K, gt_c2w, align=align_modes[0],
                            conf_thresh=args.conf_thresh, device=args.device)
                    save_json(str(out_json), result)
                    log(f"  object: {Path(gv['video_path']).name}")

    log("reconstruction 评估完成")


if __name__ == "__main__":
    main()
