#!/usr/bin/env python3
"""
run_video_quality.py — video_quality 类指标的独立运行入口。
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import scan_output_root, save_json, log
from video_quality.psnr_ssim_lpips import (
    evaluate_psnr_ssim_lpips, build_lpips_model,
)
from video_quality.vbench_eval import evaluate_vbench


def run_psnr(entries: list, device: str):
    log("评估 PSNR / SSIM / LPIPS ...")
    lpips_model = build_lpips_model(device)
    for entry in entries:
        for gv in entry["gen_videos"]:
            out_json = gv["gen_dir"] / "eval" / "psnr_ssim_lpips.json"
            if out_json.exists():
                continue
            result = evaluate_psnr_ssim_lpips(
                entry["gt_video"], gv["video_path"], device, lpips_model,
            )
            save_json(str(out_json), result)
            log(f"  PSNR={result['psnr']:.2f}  {Path(gv['video_path']).name}")
    del lpips_model


def run_vbench(entries: list):
    log("评估 VBench ...")
    for entry in entries:
        for gv in entry["gen_videos"]:
            vbench_json = gv["gen_dir"] / "intermediates" / "vbench.json"
            out_json = gv["gen_dir"] / "eval" / "vbench.json"
            if out_json.exists():
                continue
            result = evaluate_vbench(str(vbench_json))
            if result:
                save_json(str(out_json), result)
                log(f"  VBench: {result}  {Path(gv['video_path']).name}")


def main():
    parser = argparse.ArgumentParser(description="Video Quality 评估")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", default="both", choices=["psnr", "vbench", "both"])
    args = parser.parse_args()

    entries = scan_output_root(args.output_root)
    if not entries:
        log("未找到推理结果")
        return

    if args.mode in ("psnr", "both"):
        run_psnr(entries, args.device)
    if args.mode in ("vbench", "both"):
        run_vbench(entries)

    log("video_quality 评估完成")


if __name__ == "__main__":
    main()
