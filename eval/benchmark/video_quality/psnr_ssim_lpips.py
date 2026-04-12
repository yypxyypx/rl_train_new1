#!/usr/bin/env python3
"""
psnr_ssim_lpips.py — PSNR / SSIM / LPIPS 视频质量评估。

复用 RL/eval/video_eval/appearance_metrics.py 的核心逻辑。
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def read_video(path: Path, max_frames: int = 0) -> np.ndarray:
    """读取视频，返回 (N, H, W, 3) uint8。"""
    arr = imageio.v3.imread(str(path))
    if arr.ndim != 4:
        raise ValueError(f"unexpected video shape {arr.shape} for {path}")
    if max_frames > 0:
        arr = arr[:max_frames]
    return arr[..., :3]


def compute_psnr_ssim(gt: np.ndarray, pred: np.ndarray) -> Tuple[float, float]:
    """逐帧计算 PSNR 和 SSIM，返回均值。"""
    n = min(len(gt), len(pred))
    gt, pred = gt[:n], pred[:n]
    psnr_list, ssim_list = [], []
    for i in range(n):
        psnr_list.append(float(peak_signal_noise_ratio(gt[i], pred[i], data_range=255)))
        ssim_list.append(float(structural_similarity(
            gt[i], pred[i], channel_axis=2, data_range=255)))
    return float(np.mean(psnr_list)), float(np.mean(ssim_list))


def build_lpips_model(device: str):
    """初始化 LPIPS 模型（AlexNet）。"""
    try:
        import lpips
        return lpips.LPIPS(net="alex").to(device).eval()
    except Exception as e:
        print(f"[警告] LPIPS 模型初始化失败: {e}")
        return None


def compute_lpips(
    gt: np.ndarray, pred: np.ndarray, model, device: str,
) -> Optional[float]:
    """逐帧计算 LPIPS，返回均值。"""
    if model is None:
        return None
    import torch
    n = min(len(gt), len(pred))
    vals = []
    with torch.no_grad():
        for i in range(n):
            g = torch.from_numpy(gt[i]).permute(2, 0, 1).float() / 127.5 - 1.0
            p = torch.from_numpy(pred[i]).permute(2, 0, 1).float() / 127.5 - 1.0
            vals.append(float(model(g.unsqueeze(0).to(device),
                                    p.unsqueeze(0).to(device)).item()))
    return float(np.mean(vals))


def evaluate_psnr_ssim_lpips(
    gt_video: str, pred_video: str, device: str = "cuda",
    lpips_model=None, max_frames: int = 0,
) -> Dict:
    """
    计算单对视频的 PSNR / SSIM / LPIPS。

    返回 {"psnr": float, "ssim": float, "lpips": float}
    """
    gt = read_video(Path(gt_video), max_frames)
    pred = read_video(Path(pred_video), max_frames)
    psnr, ssim = compute_psnr_ssim(gt, pred)
    lpips_val = compute_lpips(gt, pred, lpips_model, device)
    return {
        "psnr": psnr,
        "ssim": ssim,
        "lpips": lpips_val if lpips_val is not None else float("nan"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PSNR / SSIM / LPIPS")
    parser.add_argument("--gt_video", required=True)
    parser.add_argument("--pred_video", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    lpips_m = build_lpips_model(args.device)
    result = evaluate_psnr_ssim_lpips(args.gt_video, args.pred_video, args.device, lpips_m)
    print(f"PSNR={result['psnr']:.4f}  SSIM={result['ssim']:.4f}  LPIPS={result['lpips']:.4f}")
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
