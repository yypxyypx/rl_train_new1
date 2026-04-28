"""
Step: DINOv2 Patch Token Extraction (geometry-free)
====================================================
Conda env: rl_da3

只跑一次 DINOv2 ViT-S/14 forward，保存 **patch-level** token (small)。
**不在这里做** FeatUp / bilinear 上采样到 pixel-level。
上采样推迟到 reward_metrics._featup_upsample_patch_tokens 阶段在 GPU 上做。

输出 npz 大小（49 帧 × 384 dim × 36 × 36 × 2B fp16 ≈ 96 MB），
比旧版 pixel-level (≈ 11 GB) 小 ~100×，quarkfs IO 友好。

设计：完全本地、无下载。
  - dinov2_vits14 模型结构：从 third_party/repos 内置 dinov2_repo 加载（torch.hub source="local"）
  - dinov2_vits14 权重：从 model/dinov2/dinov2_vits14_pretrain.pth 直接 torch.load
  - 不依赖 ~/.cache/torch/hub 下载缓存

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
import torchvision.transforms as T
from PIL import Image

_REWARD_DIR = Path(__file__).resolve().parent.parent
_RL_TRAIN_DIR = _REWARD_DIR.parent
_RL_CODE_DIR = _RL_TRAIN_DIR.parent

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
PATCH_SIZE = 14


def make_divisible(size: int, divisor: int = PATCH_SIZE) -> int:
    return ((size + divisor - 1) // divisor) * divisor


# ── DINOv2 Patch-Token Extractor ─────────────────────────────────────────────
# 仅返回 patch tokens (B, dim, H_feat, W_feat)；不做任何上采样。

def _model_root() -> Path:
    return Path(os.environ.get("RL_MODEL_ROOT", str(_RL_CODE_DIR / "model")))


def _build_dinov2_vits14() -> nn.Module:
    """从本地 dinov2_repo + 本地权重构建 DINOv2 ViT-S/14。完全离线。"""
    repo_dir = _model_root() / "dinov2_repo"
    weights_path = _model_root() / "dinov2" / "dinov2_vits14_pretrain.pth"

    if not repo_dir.is_dir():
        raise FileNotFoundError(
            f"DINOv2 repo not found: {repo_dir}\n"
            f"Expected: {repo_dir}/hubconf.py (a git clone of facebookresearch/dinov2)."
        )
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"DINOv2 ViT-S/14 weights not found: {weights_path}\n"
            f"Expected ~85MB .pth at this path."
        )

    print(f"[DINOv2Extract] Building dinov2_vits14 from {repo_dir} (no weights)")
    model = torch.hub.load(str(repo_dir), "dinov2_vits14",
                           source="local", pretrained=False)
    print(f"[DINOv2Extract] Loading weights from {weights_path}")
    state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"DINOv2 state_dict load mismatch: missing={missing}, unexpected={unexpected}"
        )
    return model.eval()


class DINOv2PatchExtractor(nn.Module):
    """DINOv2 ViT-S/14, returns patch tokens (no upsampling)."""

    def __init__(self):
        super().__init__()
        self.patch_size = PATCH_SIZE
        self.dim = 384
        self.model = _build_dinov2_vits14()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        feat_h = H // self.patch_size
        feat_w = W // self.patch_size
        out = self.model.forward_features(x)
        patch_tokens = out["x_norm_patchtokens"]
        # (B, num_patches, dim) → (B, dim, feat_h, feat_w)
        return patch_tokens.reshape(B, feat_h, feat_w, self.dim).permute(0, 3, 1, 2)


def load_extractor(device: str = "cuda:0") -> DINOv2PatchExtractor:
    model = DINOv2PatchExtractor().to(device).eval()
    print(f"[DINOv2Extract] DINOv2 ViT-S/14 patch extractor on {device}, dim={model.dim}")
    return model


@torch.no_grad()
def extract_single_frame(
    model: DINOv2PatchExtractor,
    img_path: str, H_model: int, W_model: int,
    normalize: T.Normalize, device: str,
) -> torch.Tensor:
    """提取单帧 patch tokens -> (1, C, H_feat, W_feat)."""
    img = Image.open(img_path).convert("RGB")
    img = img.resize((W_model, H_model), Image.BILINEAR)
    t = T.ToTensor()(img).unsqueeze(0).to(device)
    normed = normalize(t)
    return model(normed)


def extract_all_frames(
    model: DINOv2PatchExtractor,
    frame_paths: list[str],
    device: str,
) -> tuple[np.ndarray, int, int, int, int]:
    """提取所有帧的 DINOv2 patch tokens.

    Returns:
        features  : np.ndarray [N, C, H_feat, W_feat] float16
        H_feat, W_feat, H_model, W_model
    """
    if not frame_paths:
        raise ValueError("frame_paths is empty")

    normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    ref_img = Image.open(frame_paths[0]).convert("RGB")
    H_orig, W_orig = ref_img.height, ref_img.width
    H_model = make_divisible(H_orig)
    W_model = make_divisible(W_orig)
    if H_model != H_orig or W_model != W_orig:
        print(f"[DINOv2Extract] Resizing {H_orig}x{W_orig} -> {H_model}x{W_model}")

    first_feat = extract_single_frame(model, frame_paths[0], H_model, W_model, normalize, device)
    H_feat, W_feat = first_feat.shape[2], first_feat.shape[3]
    C = first_feat.shape[1]
    N = len(frame_paths)

    print(f"[DINOv2Extract] N={N} frames, patch={H_feat}x{W_feat}, C={C}, "
          f"target_pixel={H_model}x{W_model}")

    features = np.zeros((N, C, H_feat, W_feat), dtype=np.float16)
    features[0] = first_feat.squeeze(0).cpu().to(torch.float16).numpy()

    for i in range(1, N):
        feat = extract_single_frame(model, frame_paths[i], H_model, W_model, normalize, device)
        features[i] = feat.squeeze(0).cpu().to(torch.float16).numpy()
        if i % 10 == 0 or i == N - 1:
            print(f"[DINOv2Extract] {i}/{N - 1} frames extracted")

    return features, H_feat, W_feat, H_model, W_model


def main():
    parser = argparse.ArgumentParser(description="DINOv2 Patch Token Extraction (offline)")
    parser.add_argument("--video_frames_dir", required=True, help="帧图像目录")
    parser.add_argument("--output", required=True, help="输出 .npz 路径")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    frames_dir = Path(args.video_frames_dir).resolve()
    frame_paths = sorted([
        str(p) for p in frames_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    ])
    if not frame_paths:
        raise RuntimeError(f"No image files found in {frames_dir}")

    print(f"[DINOv2Extract] Found {len(frame_paths)} frames in {frames_dir}")

    model = load_extractor(device)
    features, H_feat, W_feat, H_model, W_model = extract_all_frames(model, frame_paths, device)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(out_path),
        features=features,             # [N, C, H_feat, W_feat] float16  (patch-level)
        H_feat=H_feat,
        W_feat=W_feat,
        H_model=H_model,
        W_model=W_model,
        mode="patch_tokens",
        frames_dir=str(frames_dir),
    )
    size_mb = features.nbytes / 1024 / 1024
    print(f"[DINOv2Extract] Saved to {out_path}  "
          f"shape={features.shape}  dtype=float16  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
