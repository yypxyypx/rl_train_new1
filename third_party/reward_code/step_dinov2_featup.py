"""
Step 3: DINOv2 + FeatUp Feature Similarity Reward
====================================================
Conda env: rl_da3

Extracts DINOv2 (+FeatUp JBU upsampler) features for each frame, then uses
DA3 geometry (from step 2's da3_output.npz) to warp each frame's features to
the reference frame and compute cosine similarity — all in one process,
no large intermediate files.

Usage:
    conda run -n rl_da3 python step_dinov2_featup.py \
        --video_frames_dir /path/to/frames/ \
        --da3_output /path/to/da3_output.npz \
        --output /path/to/feature_sim_reward.json \
        --gpu 0
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

_THIRD_PARTY_DIR = Path(__file__).resolve().parent.parent
_WORKSPACE = _THIRD_PARTY_DIR.parent.parent  # /home/users/.../

FEATUP_ROOT = str(_THIRD_PARTY_DIR / "repos" / "FeatUp")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

PATCH_SIZE = 14


def make_divisible(size: int, divisor: int = PATCH_SIZE) -> int:
    return ((size + divisor - 1) // divisor) * divisor


# ═══════════════════════ DINOv2 Model ═══════════════════════


class DINOv2BilinearExtractor(nn.Module):
    """DINOv2 ViT-S/14 with bilinear upsample to pixel resolution."""

    def __init__(self):
        super().__init__()
        self.patch_size = 14
        self.dim = 384
        _model_root = Path(os.environ.get(
            "RL_MODEL_ROOT", str(_WORKSPACE / "RL" / "model")))
        _dinov2_local = _model_root / "dinov2_repo"
        if _dinov2_local.is_dir():
            print(f"[DINOv2] Loading from local repo: {_dinov2_local}")
            self.model = torch.hub.load(
                str(_dinov2_local), "dinov2_vits14", source="local")
        else:
            print("[DINOv2] Local repo not found, loading from torch hub (needs internet)")
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
            print(f"[DINOv2+FeatUp] Loaded with JBU upsampler, dim={model.dim}")
            return model, "featup"
        else:
            print("[DINOv2+FeatUp] JBU upsampler ckpt missing or empty, using bilinear fallback")
    except Exception as e:
        print(f"[DINOv2+FeatUp] FeatUp load failed ({e}), using bilinear fallback")

    model = DINOv2BilinearExtractor().to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"[DINOv2+Bilinear] Loaded DINOv2 ViT-S/14, dim={model.dim}")
    return model, "bilinear"


@torch.no_grad()
def extract_single_frame(
    model, img_path: str, H_model: int, W_model: int,
    normalize: T.Normalize, device: str,
) -> torch.Tensor:
    """Extract features for a single frame. Returns (1, C, H_model, W_model) on GPU."""
    img = Image.open(img_path).convert("RGB")
    img = img.resize((W_model, H_model), Image.BILINEAR)
    t = T.ToTensor()(img).unsqueeze(0).to(device)
    normed = normalize(t)
    feat = model(normed)
    if feat.shape[2] != H_model or feat.shape[3] != W_model:
        feat = F.interpolate(feat, size=(H_model, W_model), mode="bilinear", align_corners=False)
    return feat


# ═══════════════════════ Geometry Utilities ═══════════════════════


def align_depth_to_feat(
    depth: torch.Tensor, K: torch.Tensor, H_feat: int, W_feat: int,
    conf: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Resize DA3 depth and adjust intrinsics to match feature resolution."""
    H_d, W_d = depth.shape
    if H_d == H_feat and W_d == W_feat:
        return depth, K, conf

    K_new = K.clone()
    K_new[0, :] *= W_feat / W_d
    K_new[1, :] *= H_feat / H_d

    depth_resized = F.interpolate(
        depth[None, None].float(), size=(H_feat, W_feat),
        mode="bilinear", align_corners=False
    ).squeeze()

    conf_resized = None
    if conf is not None:
        conf_resized = F.interpolate(
            conf[None, None].float(), size=(H_feat, W_feat),
            mode="bilinear", align_corners=False
        ).squeeze()
    return depth_resized, K_new, conf_resized


def build_warp_grid(
    H: int, W: int,
    depth: torch.Tensor, K_src: torch.Tensor, c2w_src: torch.Tensor,
    K_ref: torch.Tensor, c2w_ref: torch.Tensor,
    conf: Optional[torch.Tensor] = None, conf_threshold: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build sampling grid that warps source pixels to reference frame via 3D reprojection."""
    device = depth.device

    v_coords, u_coords = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    ones = torch.ones_like(u_coords)
    uv1 = torch.stack([u_coords, v_coords, ones], dim=-1).reshape(-1, 3).T

    K_inv = torch.inverse(K_src)
    pts_cam = K_inv @ uv1 * depth.reshape(1, -1)

    R_src, t_src = c2w_src[:3, :3], c2w_src[:3, 3:]
    pts_world = R_src @ pts_cam + t_src

    R_ref, t_ref = c2w_ref[:3, :3], c2w_ref[:3, 3:]
    pts_ref_cam = R_ref.T @ (pts_world - t_ref)

    pts_2d = K_ref @ pts_ref_cam
    z = pts_2d[2:, :]
    pts_2d = pts_2d[:2, :] / (z + 1e-8)

    grid_x = (pts_2d[0].reshape(H, W) / (W - 1)) * 2 - 1
    grid_y = (pts_2d[1].reshape(H, W) / (H - 1)) * 2 - 1
    flow_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

    valid = (
        (grid_x >= -1) & (grid_x <= 1) &
        (grid_y >= -1) & (grid_y <= 1) &
        (z.reshape(H, W) > 0) & (depth > 0)
    )
    if conf is not None:
        valid = valid & (conf > conf_threshold)
    valid_mask = valid.float().unsqueeze(0).unsqueeze(0)

    return flow_grid, valid_mask


def to_4x4(ext: np.ndarray) -> np.ndarray:
    if ext.ndim == 3 and ext.shape[1:] == (3, 4):
        N = ext.shape[0]
        out = np.zeros((N, 4, 4), dtype=ext.dtype)
        out[:, :3, :] = ext
        out[:, 3, 3] = 1.0
        return out
    return ext


# ═══════════════════════ Main ═══════════════════════


@torch.no_grad()
def compute_feature_similarity(
    model, frame_paths: list[str], da3_data: dict,
    device: str = "cuda:0", conf_threshold: float = 0.0,
) -> Tuple[float, dict]:
    """
    Extract DINOv2 features frame-by-frame and compute warped cosine similarity
    against frame 0, using DA3 geometry. No large intermediate files.
    """
    normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    N = len(frame_paths)

    ref_img = Image.open(frame_paths[0]).convert("RGB")
    H_orig, W_orig = ref_img.height, ref_img.width
    H_model = make_divisible(H_orig)
    W_model = make_divisible(W_orig)
    if H_model != H_orig or W_model != W_orig:
        print(f"[DINOv2] Resizing {H_orig}x{W_orig} -> {H_model}x{W_model} (patch_size={PATCH_SIZE})")

    depth_all = torch.from_numpy(da3_data["depth"]).to(device)
    c2w_all = torch.from_numpy(to_4x4(da3_data["extrinsics"]).astype(np.float32)).to(device)
    K_all = torch.from_numpy(da3_data["intrinsics"].astype(np.float32)).to(device)
    conf_all = torch.from_numpy(da3_data["conf"]).to(device) if da3_data["conf"].size > 0 else None

    ref_feat = extract_single_frame(model, frame_paths[0], H_model, W_model, normalize, device)
    H_feat, W_feat = ref_feat.shape[2], ref_feat.shape[3]
    print(f"[DINOv2] Feature resolution: {H_feat}x{W_feat}, dim={ref_feat.shape[1]}")

    depth_ref, K_ref, _ = align_depth_to_feat(
        depth_all[0], K_all[0], H_feat, W_feat,
        conf_all[0] if conf_all is not None else None,
    )

    dissim_scores = []
    overlap_ratios = []

    for i in range(1, N):
        gen_feat = extract_single_frame(model, frame_paths[i], H_model, W_model, normalize, device)

        depth_i, K_i, conf_i = align_depth_to_feat(
            depth_all[i], K_all[i], H_feat, W_feat,
            conf_all[i] if conf_all is not None else None,
        )

        flow_grid, valid_mask = build_warp_grid(
            H_feat, W_feat,
            depth=depth_i, K_src=K_i, c2w_src=c2w_all[i],
            K_ref=K_ref, c2w_ref=c2w_all[0],
            conf=conf_i, conf_threshold=conf_threshold,
        )

        ref_sampled = F.grid_sample(
            ref_feat, flow_grid, mode="bilinear",
            padding_mode="zeros", align_corners=True,
        )

        cos_sim = F.cosine_similarity(gen_feat, ref_sampled, dim=1).squeeze()
        score_map = 1.0 - cos_sim
        mask_2d = valid_mask.squeeze()

        valid_count = mask_2d.sum().item()
        if valid_count < 10:
            dissim_scores.append(1.0)
            overlap_ratios.append(0.0)
        else:
            dissim = (score_map * mask_2d).sum().item() / (valid_count + 1e-8)
            dissim_scores.append(dissim)
            overlap_ratios.append(valid_count / (H_feat * W_feat))

        if (i % 10 == 0) or (i == N - 1):
            print(f"[DINOv2] Processed frame {i}/{N-1}")

    if not dissim_scores:
        return 0.0, {}

    mean_dissim = float(np.mean(dissim_scores))
    reward = 1.0 - mean_dissim

    details = {
        "per_frame_dissimilarity": dissim_scores,
        "per_frame_overlap_ratio": overlap_ratios,
        "mean_dissimilarity": mean_dissim,
        "feature_resolution": [H_feat, W_feat],
    }
    return reward, details


def main():
    parser = argparse.ArgumentParser(description="Step 3: DINOv2 feature similarity reward")
    parser.add_argument("--video_frames_dir", required=True)
    parser.add_argument("--da3_output", required=True, help="DA3 output .npz from step 2")
    parser.add_argument("--output", required=True, help="Output .json path")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"

    frame_paths = sorted([
        os.path.join(args.video_frames_dir, f)
        for f in os.listdir(args.video_frames_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    if not frame_paths:
        raise FileNotFoundError(f"No image frames in {args.video_frames_dir}")
    print(f"[Step3] Found {len(frame_paths)} frames")

    da3_data = dict(np.load(args.da3_output, allow_pickle=True))
    print(f"[Step3] Loaded DA3 output: depth {da3_data['depth'].shape}")

    model, mode = load_model(device=device)
    reward, details = compute_feature_similarity(
        model, frame_paths, da3_data, device=device,
    )

    del model
    torch.cuda.empty_cache()

    result = {
        "reward_feature_sim": reward,
        "mode": mode,
        "details": details,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[Step3] reward_feature_sim = {reward:.4f} (mode={mode})")
    print(f"[Step3] Saved to {args.output}")


if __name__ == "__main__":
    main()
