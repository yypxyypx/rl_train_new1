#!/usr/bin/env python3
"""
run_reward.py — reward 类指标的独立运行入口。
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import scan_output_root, save_json, load_json, log
from common.utils import parse_camera_txt, to_4x4
from reward.camera_pose import evaluate_camera_pose, da3_to_c2w
from reward.depth_reprojection import (
    evaluate_object_reprojection,
    evaluate_global_reprojection,
    evaluate_depth_reprojection,
)
from reward.videoalign_eval import evaluate_videoalign
from reward.feature_sim import evaluate_feature_sim


def _get_video_size(video_path: str):
    cap = cv2.VideoCapture(video_path)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    return H, W


def run_camera_pose(entries: list, device: str):
    log("评估 camera_pose ...")
    for entry in entries:
        if not entry.get("camera_txt"):
            continue
        H, W = _get_video_size(entry["gt_video"])
        _, gt_c2w = parse_camera_txt(entry["camera_txt"], H, W)

        for gv in entry["gen_videos"]:
            out_json = gv["gen_dir"] / "eval" / "camera_pose.json"
            if out_json.exists():
                continue
            da3_npz = gv["gen_dir"] / "intermediates" / "da3_pred.npz"
            if not da3_npz.exists():
                continue
            da3_data = dict(np.load(str(da3_npz), allow_pickle=True))
            result = evaluate_camera_pose(da3_data, gt_c2w)
            save_json(str(out_json), result)
            log(f"  rot_auc30={result.get('rotation_auc30', 'N/A'):.4f}  "
                f"{Path(gv['video_path']).name}")


def run_depth_reprojection(entries: list, device: str, mode: str = "both"):
    log(f"评估 depth_reprojection (mode={mode}) ...")
    for entry in entries:
        H, W = _get_video_size(entry["gt_video"])
        for gv in entry["gen_videos"]:
            out_json = gv["gen_dir"] / "eval" / "depth_reprojection.json"
            if out_json.exists():
                continue
            da3_npz = gv["gen_dir"] / "intermediates" / "da3_pred.npz"
            if not da3_npz.exists():
                continue
            da3_data = dict(np.load(str(da3_npz), allow_pickle=True))
            result = {}

            if mode in ("object", "both"):
                label_maps_npz = gv["gen_dir"] / "intermediates" / "label_maps.npz"
                if label_maps_npz.exists():
                    label_data = np.load(str(label_maps_npz), allow_pickle=True)
                    label_maps = label_data["label_maps"]
                    r, d = evaluate_object_reprojection(
                        da3_data, label_maps, H, W, device=device)
                    result["object"] = {"reward": r, "details": d}

            if mode in ("global", "both"):
                r, d = evaluate_global_reprojection(str(da3_npz), H, W, device=device)
                result["global"] = {"reward": r, "details": d}

            save_json(str(out_json), result)
            scores = [f"{k}={v.get('reward', 'N/A'):.4f}" for k, v in result.items()]
            log(f"  {' '.join(scores)}  {Path(gv['video_path']).name}")


def run_videoalign(entries: list):
    log("评估 videoalign ...")
    for entry in entries:
        for gv in entry["gen_videos"]:
            out_json = gv["gen_dir"] / "eval" / "videoalign.json"
            if out_json.exists():
                continue
            va_json = gv["gen_dir"] / "intermediates" / "videoalign.json"
            result = evaluate_videoalign(str(va_json))
            if result:
                save_json(str(out_json), result)
                log(f"  Overall={result['Overall']:.4f}  {Path(gv['video_path']).name}")


def run_feature_sim(entries: list):
    log("评估 feature_sim ...")
    for entry in entries:
        for gv in entry["gen_videos"]:
            out_json = gv["gen_dir"] / "eval" / "feature_sim.json"
            if out_json.exists():
                continue
            fs_json = gv["gen_dir"] / "intermediates" / "feature_sim_reward.json"
            result = evaluate_feature_sim(str(fs_json))
            if result:
                save_json(str(out_json), result)
                log(f"  feat_sim={result['reward_feature_sim']:.4f}  "
                    f"{Path(gv['video_path']).name}")


def main():
    parser = argparse.ArgumentParser(description="Reward 评估")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", default="all",
                        choices=["camera_pose", "depth_reprojection",
                                 "depth_reprojection.object", "depth_reprojection.global",
                                 "videoalign", "feature_sim", "all"])
    args = parser.parse_args()

    entries = scan_output_root(args.output_root)
    if not entries:
        log("未找到推理结果")
        return

    if args.mode in ("camera_pose", "all"):
        run_camera_pose(entries, args.device)
    if args.mode in ("depth_reprojection", "depth_reprojection.object",
                      "depth_reprojection.global", "all"):
        dr_mode = "both"
        if args.mode == "depth_reprojection.object":
            dr_mode = "object"
        elif args.mode == "depth_reprojection.global":
            dr_mode = "global"
        run_depth_reprojection(entries, args.device, dr_mode)
    if args.mode in ("videoalign", "all"):
        run_videoalign(entries)
    if args.mode in ("feature_sim", "all"):
        run_feature_sim(entries)

    log("reward 评估完成")


if __name__ == "__main__":
    main()
