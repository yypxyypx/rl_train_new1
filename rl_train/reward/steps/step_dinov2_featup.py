"""
Step: DINOv2 + FeatUp Feature Similarity Reward
=================================================
Conda env: rl_da3

Extracts DINOv2 (+FeatUp JBU upsampler) features for each frame, then uses
DA3 geometry (from da3_output.npz) to warp each frame's features to
the reference frame and compute cosine similarity.

DA3 extrinsics are w2c OpenCV; inverted to c2w internally for warping.

compare_mode:
  first_frame : each frame vs frame 0
  adjacent    : each frame vs previous frame
  all_pairs   : each frame vs all others, per-frame mean then global mean

Usage:
    conda run -n rl_da3 python step_dinov2_featup.py \
        --video_frames_dir /path/to/frames/ \
        --da3_output /path/to/da3_output.npz \
        --output /path/to/feature_sim_reward.json \
        --compare_mode first_frame \
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


# ======================== DINOv2 Model ========================


class DINOv2BilinearExtractor(nn.Module):
    """DINOv2 ViT-S/14 with bilinear upsample to pixel resolution."""

    def __init__(self):
        super().__init__()
        self.patch_size = 14
        self.dim = 384
        _model_root = Path(os.environ.get(
            "RL_MODEL_ROOT", str(_RL_CODE_DIR / "model")))
        _dinov2_local = _model_root / "dinov2_repo"
        if _dinov2_local.is_dir():
            print(f"[DINOv2] Loading from local repo: {_dinov2_local}")
            self.model = torch.hub.load(
                str(_dinov2_local), "dinov2_vits14", source="local")
        else:
            print("[DINOv2] Local repo not found, loading from torch hub")
            self.model = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vits14")
        self.model.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        feat_h = H // self.patch_size
        feat_w = W // self.patch_size
        out = self.model.forward_features(x)
        patch_tokens = out["x_norm_patchtokens"]
        feat = patch_tokens.reshape(
            B, feat_h, feat_w, self.dim).permute(0, 3, 1, 2)
        return F.interpolate(
            feat, size=(H, W), mode="bilinear", align_corners=False)


def load_model(device: str = "cuda:0"):
    """Try FeatUp DINOv2, fall back to DINOv2 + bilinear."""
    try:
        ckpt_path = os.path.expanduser(
            "~/.cache/torch/hub/checkpoints/"
            "dinov2_jbu_stack_cocostuff.ckpt"
        )
        if os.path.exists(ckpt_path) and os.path.getsize(ckpt_path) > 1024:
            model = torch.hub.load(
                FEATUP_ROOT, "dinov2", source="local", use_norm=True)
            model = model.to(device).eval()
            for p in model.parameters():
                p.requires_grad = False
            print(f"[DINOv2+FeatUp] Loaded with JBU upsampler, "
                  f"dim={model.dim}")
            return model, "featup"
        else:
            print("[DINOv2+FeatUp] JBU ckpt missing, bilinear fallback")
    except Exception as e:
        print(f"[DINOv2+FeatUp] FeatUp load failed ({e}), "
              "bilinear fallback")

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
    """Extract features for a single frame -> (1, C, H, W) on GPU."""
    img = Image.open(img_path).convert("RGB")
    img = img.resize((W_model, H_model), Image.BILINEAR)
    t = T.ToTensor()(img).unsqueeze(0).to(device)
    normed = normalize(t)
    feat = model(normed)
    if feat.shape[2] != H_model or feat.shape[3] != W_model:
        feat = F.interpolate(
            feat, size=(H_model, W_model),
            mode="bilinear", align_corners=False)
    return feat


# ======================== Geometry Utilities ========================


def align_depth_to_feat(
    depth: torch.Tensor, K: torch.Tensor, H_feat: int, W_feat: int,
    conf: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
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


def _to_4x4(ext: np.ndarray) -> np.ndarray:
    if ext.ndim == 3 and ext.shape[1:] == (3, 4):
        N = ext.shape[0]
        out = np.zeros((N, 4, 4), dtype=ext.dtype)
        out[:, :3, :] = ext
        out[:, 3, 3] = 1.0
        return out
    if ext.ndim == 2 and ext.shape == (3, 4):
        out = np.eye(4, dtype=ext.dtype)
        out[:3, :] = ext
        return out
    return ext


# ======================== Core Computation ========================


def _compute_single_pair(
    src_feat: torch.Tensor, ref_feat: torch.Tensor,
    flow_grid: torch.Tensor, valid_mask: torch.Tensor,
    H_feat: int, W_feat: int,
) -> Tuple[float, float]:
    """Compute dissimilarity and overlap for one (src, ref) pair."""
    ref_sampled = F.grid_sample(
        ref_feat, flow_grid, mode="bilinear",
        padding_mode="zeros", align_corners=True,
    )
    cos_sim = F.cosine_similarity(src_feat, ref_sampled, dim=1).squeeze()
    score_map = 1.0 - cos_sim
    mask_2d = valid_mask.squeeze()

    valid_count = mask_2d.sum().item()
    if valid_count < 10:
        return 1.0, 0.0

    dissim = (score_map * mask_2d).sum().item() / (valid_count + 1e-8)
    overlap = valid_count / (H_feat * W_feat)
    return dissim, overlap


@torch.no_grad()
def compute_feature_similarity(
    model, frame_paths: list, da3_data: dict,
    device: str = "cuda:0", conf_threshold: float = 0.0,
    compare_mode: str = "first_frame",
) -> Tuple[float, dict]:
    """
    Extract DINOv2 features and compute warped cosine similarity using
    DA3 geometry, supporting three comparison modes.
    """
    normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    N = len(frame_paths)

    ref_img = Image.open(frame_paths[0]).convert("RGB")
    H_orig, W_orig = ref_img.height, ref_img.width
    H_model = make_divisible(H_orig)
    W_model = make_divisible(W_orig)
    if H_model != H_orig or W_model != W_orig:
        print(f"[DINOv2] Resizing {H_orig}x{W_orig} -> "
              f"{H_model}x{W_model} (patch={PATCH_SIZE})")

    depth_all = torch.from_numpy(da3_data["depth"]).to(device)
    w2c_all = torch.from_numpy(
        _to_4x4(da3_data["extrinsics"]).astype(np.float32)).to(device)
    c2w_all = torch.linalg.inv(w2c_all)
    K_all = torch.from_numpy(
        da3_data["intrinsics"].astype(np.float32)).to(device)
    conf_all = (
        torch.from_numpy(da3_data["conf"]).to(device)
        if da3_data.get("conf") is not None
        and np.asarray(da3_data["conf"]).size > 0
        else None
    )

    # Determine feature resolution from first frame
    first_feat = extract_single_frame(
        model, frame_paths[0], H_model, W_model, normalize, device)
    H_feat, W_feat = first_feat.shape[2], first_feat.shape[3]
    print(f"[DINOv2] Feature resolution: {H_feat}x{W_feat}, "
          f"dim={first_feat.shape[1]}, mode={compare_mode}")

    # Pre-align all depths to feature resolution
    depths, Ks, confs_a = [], [], []
    for i in range(N):
        d, k, c = align_depth_to_feat(
            depth_all[i], K_all[i], H_feat, W_feat,
            conf_all[i] if conf_all is not None else None,
        )
        depths.append(d)
        Ks.append(k)
        confs_a.append(c)

    # ── first_frame mode ──
    if compare_mode == "first_frame":
        ref_feat = first_feat
        dissim_scores, overlap_ratios = [], []

        for i in range(1, N):
            gen_feat = extract_single_frame(
                model, frame_paths[i],
                H_model, W_model, normalize, device)
            flow_grid, valid_mask = build_warp_grid(
                H_feat, W_feat,
                depth=depths[i], K_src=Ks[i], c2w_src=c2w_all[i],
                K_ref=Ks[0], c2w_ref=c2w_all[0],
                conf=confs_a[i], conf_threshold=conf_threshold,
            )
            d, o = _compute_single_pair(
                gen_feat, ref_feat, flow_grid, valid_mask,
                H_feat, W_feat)
            dissim_scores.append(d)
            overlap_ratios.append(o)

            if (i % 10 == 0) or (i == N - 1):
                print(f"[DINOv2] Processed frame {i}/{N-1}")

    # ── adjacent mode ──
    elif compare_mode == "adjacent":
        prev_feat = first_feat
        dissim_scores, overlap_ratios = [], []

        for i in range(1, N):
            curr_feat = extract_single_frame(
                model, frame_paths[i],
                H_model, W_model, normalize, device)
            flow_grid, valid_mask = build_warp_grid(
                H_feat, W_feat,
                depth=depths[i], K_src=Ks[i], c2w_src=c2w_all[i],
                K_ref=Ks[i - 1], c2w_ref=c2w_all[i - 1],
                conf=confs_a[i], conf_threshold=conf_threshold,
            )
            d, o = _compute_single_pair(
                curr_feat, prev_feat, flow_grid, valid_mask,
                H_feat, W_feat)
            dissim_scores.append(d)
            overlap_ratios.append(o)
            prev_feat = curr_feat

            if (i % 10 == 0) or (i == N - 1):
                print(f"[DINOv2] Processed frame {i}/{N-1}")

    # ── all_pairs mode ──
    elif compare_mode == "all_pairs":
        all_feats = [first_feat.cpu()]
        for i in range(1, N):
            feat = extract_single_frame(
                model, frame_paths[i],
                H_model, W_model, normalize, device)
            all_feats.append(feat.cpu())
            if (i % 10 == 0) or (i == N - 1):
                print(f"[DINOv2] Extracted features {i}/{N-1}")

        per_frame_dissim = [[] for _ in range(N)]
        per_frame_overlap = [[] for _ in range(N)]

        for i in range(N):
            src_feat = all_feats[i].to(device)
            for j in range(N):
                if i == j:
                    continue
                ref_feat_j = all_feats[j].to(device)
                flow_grid, valid_mask = build_warp_grid(
                    H_feat, W_feat,
                    depth=depths[i], K_src=Ks[i], c2w_src=c2w_all[i],
                    K_ref=Ks[j], c2w_ref=c2w_all[j],
                    conf=confs_a[i], conf_threshold=conf_threshold,
                )
                d, o = _compute_single_pair(
                    src_feat, ref_feat_j, flow_grid, valid_mask,
                    H_feat, W_feat)
                per_frame_dissim[i].append(d)
                per_frame_overlap[i].append(o)
                del ref_feat_j
            del src_feat
            if (i % 5 == 0) or (i == N - 1):
                print(f"[DINOv2] All-pairs: frame {i}/{N-1} done")

        dissim_scores = [float(np.mean(d)) for d in per_frame_dissim]
        overlap_ratios = [float(np.mean(o)) for o in per_frame_overlap]

    else:
        raise ValueError(
            f"Unknown compare_mode: {compare_mode}. "
            "Valid: first_frame, adjacent, all_pairs.")

    if not dissim_scores:
        return 0.0, {}

    mean_dissim = float(np.mean(dissim_scores))
    reward = 1.0 - mean_dissim

    details = {
        "compare_mode": compare_mode,
        "per_frame_dissimilarity": dissim_scores,
        "per_frame_overlap_ratio": overlap_ratios,
        "mean_dissimilarity": mean_dissim,
        "feature_resolution": [H_feat, W_feat],
        "num_pairs": (
            N * (N - 1) if compare_mode == "all_pairs" else N - 1),
    }
    return reward, details


# ======================== Main ========================


def main():
    parser = argparse.ArgumentParser(
        description="DINOv2 feature similarity reward")
    parser.add_argument("--video_frames_dir", required=True)
    parser.add_argument("--da3_output", required=True,
                        help="DA3 output .npz from step_da3")
    parser.add_argument("--output", required=True,
                        help="Output .json path")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--compare_mode", type=str, default="first_frame",
                        choices=["first_frame", "adjacent", "all_pairs"],
                        help="Comparison mode for feature similarity")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"

    frame_paths = sorted([
        os.path.join(args.video_frames_dir, f)
        for f in os.listdir(args.video_frames_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    if not frame_paths:
        raise FileNotFoundError(
            f"No image frames in {args.video_frames_dir}")
    print(f"[step_dinov2_featup] Found {len(frame_paths)} frames")

    da3_data = dict(np.load(args.da3_output, allow_pickle=True))
    print(f"[step_dinov2_featup] Loaded DA3 output: "
          f"depth {da3_data['depth'].shape}")

    model, mode = load_model(device=device)
    reward, details = compute_feature_similarity(
        model, frame_paths, da3_data, device=device,
        compare_mode=args.compare_mode,
    )

    del model
    torch.cuda.empty_cache()

    result = {
        "reward_feature_sim": reward,
        "mode": mode,
        "details": details,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)),
                exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[step_dinov2_featup] reward_feature_sim = {reward:.4f} "
          f"(mode={mode}, compare={args.compare_mode})")
    print(f"[step_dinov2_featup] Saved to {args.output}")


if __name__ == "__main__":
    main()
