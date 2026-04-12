#!/usr/bin/env python3
"""
depth_reprojection.py — 深度重投影一致性评估。

三个接口：
  evaluate_object_reprojection  — 物体级（语义引导几何一致性）
  evaluate_global_reprojection  — 全局（纯深度一致性）
  evaluate_depth_reprojection   — 两者都跑

复用 RL/eval/reward_eval/reward_metrics.py 的核心逻辑。

坐标系：DA3 extrinsics 为 w2c，内部取逆为 c2w 后用于 warp。
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ═══════════════════ 工具函数 ═══════════════════════════════════


def _to_4x4(ext: np.ndarray) -> np.ndarray:
    if ext.ndim == 3 and ext.shape[1:] == (3, 4):
        N = ext.shape[0]
        out = np.zeros((N, 4, 4), dtype=ext.dtype)
        out[:, :3, :] = ext
        out[:, 3, 3] = 1.0
        return out
    return ext


def align_depth_to_image(
    depth: torch.Tensor, K: torch.Tensor,
    H_img: int, W_img: int,
    conf: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    H_d, W_d = depth.shape
    if H_d == H_img and W_d == W_img:
        return depth, K, conf
    K_new = K.clone()
    K_new[0, :] *= W_img / W_d
    K_new[1, :] *= H_img / H_d
    depth_resized = F.interpolate(
        depth[None, None].float(), size=(H_img, W_img),
        mode="bilinear", align_corners=False,
    ).squeeze()
    conf_resized = None
    if conf is not None:
        conf_resized = F.interpolate(
            conf[None, None].float(), size=(H_img, W_img),
            mode="bilinear", align_corners=False,
        ).squeeze()
    return depth_resized, K_new, conf_resized


def build_warp_grid(
    H: int, W: int,
    depth: torch.Tensor, K_src: torch.Tensor, c2w_src: torch.Tensor,
    K_ref: torch.Tensor, c2w_ref: torch.Tensor,
    conf: Optional[torch.Tensor] = None, conf_threshold: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    proj_depth = z.reshape(H, W)

    valid = (
        (grid_x >= -1) & (grid_x <= 1) &
        (grid_y >= -1) & (grid_y <= 1) &
        (z.reshape(H, W) > 0) & (depth > 0)
    )
    if conf is not None:
        valid = valid & (conf > conf_threshold)
    valid_mask = valid.float().unsqueeze(0).unsqueeze(0)
    return flow_grid, valid_mask, proj_depth


def filter_unstable_masks(
    masks: np.ndarray, area_change_thresh: float = 0.30,
    min_area: float = 0.001, object_names: list = None,
) -> Tuple[np.ndarray, list]:
    N_obj, N_frames, H, W = masks.shape
    filtered = masks.copy()
    pixel_total = float(H * W)
    removed = []

    for i in range(N_obj):
        areas = masks[i].sum(axis=(1, 2)) / pixel_total
        present = areas > min_area

        # flickering
        was_present = was_absent_after = False
        flickering = False
        for t in range(N_frames):
            if present[t]:
                if was_absent_after:
                    flickering = True
                    break
                was_present = True
            elif was_present:
                was_absent_after = True
        if flickering:
            filtered[i] = False
            name = object_names[i] if object_names else str(i)
            removed.append((i, name, "flickering"))
            continue

        # area jump
        for t in range(1, N_frames):
            prev, curr = float(areas[t - 1]), float(areas[t])
            if prev < min_area or curr < min_area:
                continue
            denom = max(prev, curr)
            if abs(curr - prev) / denom > area_change_thresh:
                filtered[i] = False
                name = object_names[i] if object_names else str(i)
                removed.append((i, name, "area_jump"))
                break

    return filtered, removed


# ═══════════════════ 物体级深度重投影 ═══════════════════════════


def evaluate_object_reprojection(
    da3_data: dict, label_maps: np.ndarray,
    H_img: int, W_img: int,
    conf_threshold: float = 0.0, device: str = "cpu",
) -> Tuple[float, Dict]:
    """
    语义引导的几何一致性（物体级深度重投影）。

    DA3 extrinsics 为 w2c，内部取逆为 c2w。
    """
    depth_all = torch.from_numpy(da3_data["depth"]).to(device)
    # DA3 extrinsics 是 w2c，需取逆得 c2w
    w2c_all = torch.from_numpy(
        _to_4x4(da3_data["extrinsics"]).astype(np.float32)).to(device)
    c2w_all = torch.linalg.inv(w2c_all)
    K_all = torch.from_numpy(da3_data["intrinsics"].astype(np.float32)).to(device)
    conf_all = (torch.from_numpy(da3_data["conf"]).to(device)
                if da3_data.get("conf") is not None
                and np.asarray(da3_data["conf"]).size > 0 else None)

    label_raw = torch.from_numpy(label_maps.astype(np.int32)).to(device)
    if label_raw.shape[1] != H_img or label_raw.shape[2] != W_img:
        label_t = F.interpolate(
            label_raw.unsqueeze(1).float(), size=(H_img, W_img), mode="nearest"
        ).squeeze(1).long()
    else:
        label_t = label_raw

    N = depth_all.shape[0]
    depth_ref, K_ref, conf_ref = align_depth_to_image(
        depth_all[0], K_all[0], H_img, W_img,
        conf_all[0] if conf_all is not None else None,
    )
    label_ref = label_t[0]

    frame_scores = []
    for i in range(1, N):
        depth_i, K_i, conf_i = align_depth_to_image(
            depth_all[i], K_all[i], H_img, W_img,
            conf_all[i] if conf_all is not None else None,
        )
        flow_grid, valid_mask, proj_depth = build_warp_grid(
            H_img, W_img, depth=depth_i, K_src=K_i, c2w_src=c2w_all[i],
            K_ref=K_ref, c2w_ref=c2w_all[0],
            conf=conf_i, conf_threshold=conf_threshold,
        )
        mask_2d = valid_mask.squeeze().bool()
        valid_count = mask_2d.sum().item()
        if valid_count < 10:
            frame_scores.append(0.0)
            continue

        ref_depth_sampled = F.grid_sample(
            depth_ref[None, None], flow_grid, mode="bilinear",
            padding_mode="zeros", align_corners=True,
        ).squeeze()
        ref_label_sampled = F.grid_sample(
            label_ref.float()[None, None], flow_grid, mode="nearest",
            padding_mode="zeros", align_corners=True,
        ).squeeze().long()

        label_i = label_t[i]
        fg_match = mask_2d & (label_i > 0) & (ref_label_sampled > 0) & (label_i == ref_label_sampled)
        bg_match = mask_2d & (label_i == 0) & (ref_label_sampled == 0)

        rel_err = torch.abs(proj_depth - ref_depth_sampled) / (ref_depth_sampled + 1e-8)
        per_pixel_score = torch.exp(-rel_err)

        numerator = (per_pixel_score * fg_match.float() +
                     per_pixel_score * bg_match.float() * 0.8).sum()
        frame_scores.append(float(numerator / (valid_count + 1e-8)))

    if not frame_scores:
        return 0.0, {}

    reward = float(np.mean(frame_scores))
    return reward, {"per_frame_score": frame_scores, "mean_score": reward}


# ═══════════════════ 全局深度重投影 ═════════════════════════════


def evaluate_global_reprojection(
    da3_path: str, H_img: int, W_img: int,
    conf_threshold: float = 0.5, device: str = "cpu",
) -> Tuple[float, Dict]:
    """
    全局深度一致性（无语义标签）。

    DA3 extrinsics 为 w2c，内部取逆为 c2w。
    """
    da3 = np.load(da3_path, allow_pickle=True)
    depth_np = da3["depth"]
    K_np = da3["intrinsics"]
    w2c_all = torch.from_numpy(
        _to_4x4(da3["extrinsics"]).astype(np.float32)).to(device)
    c2w_all = torch.linalg.inv(w2c_all)
    conf_np = da3["conf"] if "conf" in da3 else None

    N = depth_np.shape[0]
    depth_ref, K_ref_t, conf_ref = align_depth_to_image(
        torch.from_numpy(depth_np[0].astype(np.float32)).to(device),
        torch.from_numpy(K_np[0].astype(np.float32)).to(device),
        H_img, W_img,
        torch.from_numpy(conf_np[0].astype(np.float32)).to(device) if conf_np is not None else None,
    )
    depth_ref = depth_ref.float()

    frame_scores = []
    for i in range(1, N):
        depth_i, K_i_t, conf_i = align_depth_to_image(
            torch.from_numpy(depth_np[i].astype(np.float32)).to(device),
            torch.from_numpy(K_np[i].astype(np.float32)).to(device),
            H_img, W_img,
            torch.from_numpy(conf_np[i].astype(np.float32)).to(device) if conf_np is not None else None,
        )
        flow_grid, valid_mask, proj_depth = build_warp_grid(
            H_img, W_img, depth=depth_i, K_src=K_i_t, c2w_src=c2w_all[i],
            K_ref=K_ref_t, c2w_ref=c2w_all[0],
            conf=conf_i, conf_threshold=conf_threshold,
        )
        mask_2d = valid_mask.squeeze().bool()
        valid_count = mask_2d.sum().item()
        if valid_count < 10:
            frame_scores.append(0.0)
            continue

        ref_depth_sampled = F.grid_sample(
            depth_ref[None, None], flow_grid, mode="bilinear",
            padding_mode="zeros", align_corners=True,
        ).squeeze()

        rel_err = torch.abs(proj_depth - ref_depth_sampled) / (ref_depth_sampled + 1e-8)
        per_pixel_score = torch.exp(-rel_err)
        frame_scores.append(float(
            (per_pixel_score * mask_2d.float()).sum().item() / (valid_count + 1e-8)))

    if not frame_scores:
        return 0.0, {}

    reward = float(np.mean(frame_scores))
    return reward, {"per_frame_score": frame_scores, "mean_score": reward}


# ═══════════════════ 组合接口 ══════════════════════════════════


def evaluate_depth_reprojection(
    da3_data: dict, da3_path: str,
    label_maps: Optional[np.ndarray],
    H_img: int, W_img: int,
    device: str = "cpu",
) -> Dict:
    """两者都跑，返回 {object: ..., global: ...}。"""
    result = {}

    if label_maps is not None:
        r_obj, d_obj = evaluate_object_reprojection(
            da3_data, label_maps, H_img, W_img, device=device)
        result["object"] = {"reward": r_obj, "details": d_obj}

    r_glob, d_glob = evaluate_global_reprojection(
        da3_path, H_img, W_img, device=device)
    result["global"] = {"reward": r_glob, "details": d_glob}

    return result
