"""
Step: DINOv2 Feature Extraction (geometry-free)
================================================
Conda env: rl_da3

只做 per-frame DINOv2 特征提取，不依赖 DA3 输出。
输出保存为 dinov2_features.npz，warping 和相似度计算由
compute_all_rewards() 在聚合阶段完成（届时 da3_output.npz 也已就绪）。

这样 DA3 / Qwen+SAM3 / DINOv2-extract / VideoAlign 四组可以完全并行。

Usage:
    conda run -n rl_da3 python step_dinov2_extract.py \\
        --video_frames_dir /path/to/frames/ \\
        --output /path/to/dinov2_features.npz \\
        --gpu 0
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

_REWARD_DIR = Path(__file__).resolve().parent.parent
_RL_TRAIN_DIR = _REWARD_DIR.parent
_RL_CODE_DIR = _RL_TRAIN_DIR.parent
_THIRD_PARTY_DIR = _RL_CODE_DIR / "third_party"

FEATUP_ROOT = str(_THIRD_PARTY_DIR / "repos" / "FeatUp")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
PATCH_SIZE = 14


def make_divisible(size: int, divisor: int = PATCH_SIZE) -> int:
    return ((size + divisor - 1) // divisor) * divisor


# ── DINOv2 model（与 step_dinov2_featup.py 一致，直接复用） ──────────────────

class DINOv2BilinearExtractor(nn.Module):
    """DINOv2 ViT-S/14 with bilinear upsample to pixel resolution."""

    def __init__(self):
        super().__init__()
        self.patch_size = 14
        self.dim = 384
        _model_root = Path(os.environ.get(
            "RL_MODEL_ROOT", str(_RL_CODE_DIR.parent / "RL" / "model")))
        _dinov2_local = _model_root / "dinov2_repo"
        if _dinov2_local.is_dir():
            print(f"[DINOv2Extract] Loading from local repo: {_dinov2_local}")
            self.model = torch.hub.load(
                str(_dinov2_local), "dinov2_vits14", source="local")
        else:
            print("[DINOv2Extract] Local repo not found, loading from torch hub")
            self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.model.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        feat_h = H // self.patch_size
        feat_w = W // self.patch_size
        out = self.model.forward_features(x)
        patch_tokens = out["x_norm_patchtokens"]
        feat = patch_tokens.reshape(B, feat_h, feat_w, self.dim).permute(0, 3, 1, 2)
        return F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=False)


def load_model(device: str = "cuda:0"):
    """Try FeatUp DINOv2, fall back to DINOv2 + bilinear."""
    try:
        ckpt_path = os.path.expanduser(
            "~/.cache/torch/hub/checkpoints/dinov2_jbu_stack_cocostuff.ckpt"
        )
        if os.path.exists(ckpt_path) and os.path.getsize(ckpt_path) > 1024:
            model = torch.hub.load(FEATUP_ROOT, "dinov2", source="local", use_norm=True)
            model = model.to(device).eval()
            for p in model.parameters():
                p.requires_grad = False
            print(f"[DINOv2Extract] Loaded with FeatUp JBU, dim={model.dim}")
            return model, "featup"
        else:
            print("[DINOv2Extract] JBU ckpt missing, bilinear fallback")
    except Exception as e:
        print(f"[DINOv2Extract] FeatUp failed ({e}), bilinear fallback")

    model = DINOv2BilinearExtractor().to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"[DINOv2Extract] Loaded DINOv2 ViT-S/14 bilinear, dim={model.dim}")
    return model, "bilinear"


@torch.no_grad()
def extract_single_frame(
    model, img_path: str, H_model: int, W_model: int,
    normalize: T.Normalize, device: str,
) -> torch.Tensor:
    """提取单帧特征 -> (1, C, H_model, W_model)。"""
    img = Image.open(img_path).convert("RGB")
    img = img.resize((W_model, H_model), Image.BILINEAR)
    t = T.ToTensor()(img).unsqueeze(0).to(device)
    normed = normalize(t)
    feat = model(normed)
    if feat.shape[2] != H_model or feat.shape[3] != W_model:
        feat = F.interpolate(feat, size=(H_model, W_model), mode="bilinear", align_corners=False)
    return feat


def extract_all_frames(
    model,
    frame_paths: list[str],
    device: str,
) -> tuple[np.ndarray, int, int, int, int]:
    """提取所有帧的 DINOv2 特征。

    Returns:
        features  : np.ndarray [N, C, H_feat, W_feat] float16
        H_feat, W_feat, H_model, W_model
    """
    if not frame_paths:
        raise ValueError("frame_paths is empty")

    normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    # 从第一帧确定分辨率
    ref_img = Image.open(frame_paths[0]).convert("RGB")
    H_orig, W_orig = ref_img.height, ref_img.width
    H_model = make_divisible(H_orig)
    W_model = make_divisible(W_orig)
    if H_model != H_orig or W_model != W_orig:
        print(f"[DINOv2Extract] Resizing {H_orig}x{W_orig} -> {H_model}x{W_model}")

    # 首帧确定特征分辨率
    first_feat = extract_single_frame(model, frame_paths[0], H_model, W_model, normalize, device)
    H_feat, W_feat = first_feat.shape[2], first_feat.shape[3]
    C = first_feat.shape[1]
    N = len(frame_paths)

    print(f"[DINOv2Extract] N={N} frames, feat={H_feat}x{W_feat}, C={C}")

    features = np.zeros((N, C, H_feat, W_feat), dtype=np.float16)
    features[0] = first_feat.squeeze(0).cpu().to(torch.float16).numpy()

    for i in range(1, N):
        feat = extract_single_frame(model, frame_paths[i], H_model, W_model, normalize, device)
        features[i] = feat.squeeze(0).cpu().to(torch.float16).numpy()
        if i % 10 == 0 or i == N - 1:
            print(f"[DINOv2Extract] {i}/{N - 1} frames extracted")

    return features, H_feat, W_feat, H_model, W_model


def main():
    parser = argparse.ArgumentParser(description="DINOv2 Feature Extraction (no DA3)")
    parser.add_argument("--video_frames_dir", required=True, help="帧图像目录")
    parser.add_argument("--output", required=True, help="输出 .npz 路径")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    frames_dir = Path(args.video_frames_dir)
    frame_paths = sorted([
        str(p) for p in frames_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    ])
    if not frame_paths:
        raise RuntimeError(f"No image files found in {frames_dir}")

    print(f"[DINOv2Extract] Found {len(frame_paths)} frames in {frames_dir}")

    model, mode = load_model(device)
    features, H_feat, W_feat, H_model, W_model = extract_all_frames(model, frame_paths, device)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(out_path),
        features=features,    # [N, C, H_feat, W_feat] float16
        H_feat=H_feat,
        W_feat=W_feat,
        H_model=H_model,
        W_model=W_model,
        mode=mode,
    )
    print(f"[DINOv2Extract] Saved to {out_path}  "
          f"shape={features.shape}  dtype=float16")


if __name__ == "__main__":
    main()
