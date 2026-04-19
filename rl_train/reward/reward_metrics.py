#!/usr/bin/env python3
"""
reward_metrics.py
=================
Reward computation module.

Rewards:
  1 : Geo + Semantic (DA3 depth + SAM3 labels)
  1b: Geo Global (depth only)
  2 : DINOv2 Feature Similarity (from pre-computed JSON)
  3 : Camera Trajectory (DA3 pred vs GT)
  4 : VideoAlign Video Quality

Convention:
  DA3 extrinsics = w2c OpenCV (N, 3, 4)
  GT camera.txt  = w2c OpenCV

compare_mode:
  first_frame : each frame vs frame 0
  adjacent    : each frame vs previous frame
  all_pairs   : each frame vs all others, per-frame mean then global mean
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ==================== Utilities ====================


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
    removed_list = []
    for i in range(N_obj):
        areas = masks[i].sum(axis=(1, 2)) / pixel_total
        present = areas > min_area
        flickering = False
        was_present = was_absent_after = False
        for t in range(N_frames):
            if present[t]:
                if was_absent_after:
                    flickering = True
                    break
                was_present = True
            else:
                if was_present:
                    was_absent_after = True
        if flickering:
            filtered[i] = False
            name = object_names[i] if object_names else str(i)
            removed_list.append((i, name, "flickering", areas.tolist()))
            continue
        for t in range(1, N_frames):
            prev, curr = float(areas[t - 1]), float(areas[t])
            if prev < min_area or curr < min_area:
                continue
            denom = max(prev, curr)
            if abs(curr - prev) / denom > area_change_thresh:
                filtered[i] = False
                name = object_names[i] if object_names else str(i)
                removed_list.append((i, name, "area_jump", areas.tolist()))
                break
    return filtered, removed_list


def _get_pairs(N: int, compare_mode: str) -> List[Tuple[int, int]]:
    """Generate (source, reference) index pairs."""
    if compare_mode == "first_frame":
        return [(i, 0) for i in range(1, N)]
    elif compare_mode == "adjacent":
        return [(i, i - 1) for i in range(1, N)]
    elif compare_mode == "first_three":
        ref_count = min(3, N)
        return [(i, j) for j in range(ref_count) for i in range(N) if i != j]
    elif compare_mode == "all_pairs":
        return [(i, j) for i in range(N) for j in range(N) if i != j]
    raise ValueError(
        f"Unknown compare_mode: {compare_mode}. "
        "Valid: first_frame, adjacent, first_three, all_pairs."
    )


def _prepare_frames(
    da3_data: dict, H_img: int, W_img: int, device: str,
):
    """Load DA3 data and align all frames to image resolution."""
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

    N = depth_all.shape[0]
    depths, Ks, confs = [], [], []
    for i in range(N):
        d, k, c = align_depth_to_image(
            depth_all[i], K_all[i], H_img, W_img,
            conf_all[i] if conf_all is not None else None,
        )
        depths.append(d.float())
        Ks.append(k)
        confs.append(c)

    return depths, Ks, c2w_all, confs, N


def _aggregate_scores(
    pair_results: list, N: int, compare_mode: str,
) -> Tuple[float, list]:
    """
    Aggregate pair scores.

    pair_results: list of tuples, first three elements are (src, ref, score).
    Returns (mean_reward, per_frame_scores).
    """
    if not pair_results:
        return 0.0, []

    if compare_mode == "all_pairs":
        per_frame = [[] for _ in range(N)]
        for item in pair_results:
            src, ref, score = item[0], item[1], item[2]
            per_frame[src].append(score)
        per_frame_means = [
            float(np.mean(s)) if s else 0.0 for s in per_frame
        ]
        return float(np.mean(per_frame_means)), per_frame_means
    else:
        scores = [item[2] for item in pair_results]
        return (float(np.mean(scores)) if scores else 0.0), scores


# ==================== Reward 1: Geo + Semantic ====================


def compute_reward_geo_semantic(
    da3_data: dict, label_maps: np.ndarray,
    H_img: int, W_img: int,
    conf_threshold: float = 0.0, device: str = "cpu",
    compare_mode: str = "first_frame",
) -> Tuple[float, dict]:
    """DA3 extrinsics = w2c, inverted to c2w internally."""
    depths, Ks, c2w_all, confs, N = _prepare_frames(
        da3_data, H_img, W_img, device)

    label_raw = torch.from_numpy(label_maps.astype(np.int32)).to(device)
    if label_raw.shape[1] != H_img or label_raw.shape[2] != W_img:
        label_t = F.interpolate(
            label_raw.unsqueeze(1).float(), size=(H_img, W_img),
            mode="nearest",
        ).squeeze(1).long()
    else:
        label_t = label_raw

    pairs = _get_pairs(N, compare_mode)

    # (src, ref, score, match_rate, valid_count, fg_rate)
    pair_results = []

    for src, ref in pairs:
        flow_grid, valid_mask, proj_depth = build_warp_grid(
            H_img, W_img,
            depth=depths[src], K_src=Ks[src], c2w_src=c2w_all[src],
            K_ref=Ks[ref], c2w_ref=c2w_all[ref],
            conf=confs[src], conf_threshold=conf_threshold,
        )
        mask_2d = valid_mask.squeeze().bool()
        valid_count = mask_2d.sum().item()
        if valid_count < 10:
            pair_results.append((src, ref, 0.0, 0.0, 0, 0.0))
            continue

        ref_depth_sampled = F.grid_sample(
            depths[ref][None, None], flow_grid, mode="bilinear",
            padding_mode="zeros", align_corners=True,
        ).squeeze()
        ref_label_sampled = F.grid_sample(
            label_t[ref].float()[None, None], flow_grid, mode="nearest",
            padding_mode="zeros", align_corners=True,
        ).squeeze().long()

        label_src = label_t[src]
        fg_match = (
            mask_2d & (label_src > 0) & (ref_label_sampled > 0)
            & (label_src == ref_label_sampled)
        )
        bg_match = mask_2d & (label_src == 0) & (ref_label_sampled == 0)
        label_match = fg_match | bg_match

        rel_err = (
            torch.abs(proj_depth - ref_depth_sampled)
            / (ref_depth_sampled + 1e-8)
        )
        per_pixel_score = torch.exp(-rel_err)

        numerator = (
            per_pixel_score * fg_match.float()
            + per_pixel_score * bg_match.float() * 0.8
        ).sum()
        score = (numerator / (valid_count + 1e-8)).item()
        match_rate = float(
            label_match.sum().item() / (valid_count + 1e-8))
        fg_total = (mask_2d & (label_src > 0)).sum().item()
        fg_rate = float(fg_match.sum().item() / (fg_total + 1e-8))

        pair_results.append(
            (src, ref, score, match_rate, int(valid_count), fg_rate))

    if not pair_results:
        return 0.0, {}

    reward, per_frame_scores = _aggregate_scores(
        pair_results, N, compare_mode)

    # Aggregate auxiliary metrics per frame
    if compare_mode == "all_pairs":
        grp_mr = [[] for _ in range(N)]
        grp_fr = [[] for _ in range(N)]
        grp_vc = [[] for _ in range(N)]
        for s, r, sc, mr, vc, fr in pair_results:
            grp_mr[s].append(mr)
            grp_fr[s].append(fr)
            grp_vc[s].append(vc)
        match_rates = [float(np.mean(x)) if x else 0.0 for x in grp_mr]
        fg_rates = [float(np.mean(x)) if x else 0.0 for x in grp_fr]
        valid_counts = [int(np.mean(x)) if x else 0 for x in grp_vc]
    else:
        match_rates = [mr for _, _, _, mr, _, _ in pair_results]
        fg_rates = [fr for _, _, _, _, _, fr in pair_results]
        valid_counts = [int(vc) for _, _, _, _, vc, _ in pair_results]

    details = {
        "compare_mode": compare_mode,
        "per_frame_score": per_frame_scores,
        "per_frame_label_match_rate": match_rates,
        "per_frame_fg_match_rate": fg_rates,
        "per_frame_valid_pixels": valid_counts,
        "mean_score": reward,
        "mean_label_match_rate": float(np.mean(match_rates)),
        "mean_fg_match_rate": (
            float(np.mean(fg_rates)) if fg_rates else float("nan")),
        "mean_valid_pixels": (
            int(np.mean(valid_counts)) if valid_counts else 0),
        "num_pairs": len(pair_results),
    }
    return reward, details


# ==================== Reward 1b: Geo Global ====================


def compute_reward_geo_global(
    da3_data: dict, H_img: int, W_img: int,
    conf_threshold: float = 0.5, device: str = "cpu",
    compare_mode: str = "first_frame",
) -> Tuple[float, dict]:
    """DA3 extrinsics = w2c, inverted to c2w internally."""
    depths, Ks, c2w_all, confs, N = _prepare_frames(
        da3_data, H_img, W_img, device)

    pairs = _get_pairs(N, compare_mode)
    pair_results = []  # (src, ref, score, valid_count)

    for src, ref in pairs:
        flow_grid, valid_mask, proj_depth = build_warp_grid(
            H_img, W_img,
            depth=depths[src], K_src=Ks[src], c2w_src=c2w_all[src],
            K_ref=Ks[ref], c2w_ref=c2w_all[ref],
            conf=confs[src], conf_threshold=conf_threshold,
        )
        mask_2d = valid_mask.squeeze().bool()
        valid_count = mask_2d.sum().item()
        if valid_count < 10:
            pair_results.append((src, ref, 0.0, 0))
            continue

        ref_depth_sampled = F.grid_sample(
            depths[ref][None, None], flow_grid, mode="bilinear",
            padding_mode="zeros", align_corners=True,
        ).squeeze()

        rel_err = (
            torch.abs(proj_depth - ref_depth_sampled)
            / (ref_depth_sampled + 1e-8)
        )
        per_pixel_score = torch.exp(-rel_err)
        score = (
            (per_pixel_score * mask_2d.float()).sum().item()
            / (valid_count + 1e-8)
        )
        pair_results.append((src, ref, score, int(valid_count)))

    if not pair_results:
        return 0.0, {}

    reward, per_frame_scores = _aggregate_scores(
        pair_results, N, compare_mode)

    if compare_mode == "all_pairs":
        grp_vc = [[] for _ in range(N)]
        for s, r, sc, vc in pair_results:
            grp_vc[s].append(vc)
        valid_list = [int(np.mean(x)) if x else 0 for x in grp_vc]
    else:
        valid_list = [int(vc) for _, _, _, vc in pair_results]

    return reward, {
        "compare_mode": compare_mode,
        "per_frame_score": per_frame_scores,
        "per_frame_valid_pixels": valid_list,
        "mean_score": reward,
        "mean_valid_pixels": int(np.mean(valid_list)) if valid_list else 0,
        "num_pairs": len(pair_results),
    }


# ==================== Reward 3: Camera Trajectory ====================


def _trajectory_length(positions):
    diffs = np.diff(positions, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def compute_reward_camera_traj(
    da3_data: dict,
    gt_w2c,
    rot_weight: float = 0.5,
    trans_weight: float = 0.5,
):
    """
    Compare DA3 predicted trajectory vs GT (both w2c OpenCV).

    Pipeline:
      1. First-frame pose alignment (w2c space)
      2. Convert to c2w, extract camera positions
      3. Trajectory length scale alignment
      4. Rotation reward: adjacent-frame relative rotation angle diff
      5. Translation reward: per-frame absolute distance, normalized
    """
    pred_w2c = _to_4x4(da3_data["extrinsics"]).astype(np.float64)
    gt_w2c_4 = _to_4x4(gt_w2c).astype(np.float64)

    N = min(len(pred_w2c), len(gt_w2c_4))
    pred_w2c = pred_w2c[:N]
    gt_w2c_4 = gt_w2c_4[:N]

    if N < 2:
        return 0.0, {"error": "frames < 2"}

    # 首帧 pose 对齐：把 pred 表达到 GT 的 world 系下。
    #   设 X_gt = T @ X_pred (两套 world 系之间的刚体变换)，则在 GT world 下
    #   表达 pred 相机为 aligned[i] = pred_w2c[i] @ T_inv，
    #   约束 aligned[0] = gt_w2c[0] => T_inv = inv(pred_w2c[0]) @ gt_w2c[0]。
    #   注意必须右乘（不是左乘），否则会把 pred 的相对运动嫁接到 GT 的初始
    #   相机系上，导致轨迹形状和长度都被扭曲。
    T_inv = np.linalg.inv(pred_w2c[0]) @ gt_w2c_4[0]
    aligned_pred_w2c = np.stack(
        [pred_w2c[i] @ T_inv for i in range(N)])

    pred_c2w = np.array(
        [np.linalg.inv(aligned_pred_w2c[i]) for i in range(N)])
    gt_c2w = np.array(
        [np.linalg.inv(gt_w2c_4[i]) for i in range(N)])

    pred_pos = pred_c2w[:, :3, 3]
    gt_pos = gt_c2w[:, :3, 3]

    gt_traj_len = _trajectory_length(gt_pos)
    pred_traj_len = _trajectory_length(pred_pos)
    s = (gt_traj_len / (pred_traj_len + 1e-8)
         if pred_traj_len > 1e-8 else 1.0)

    origin = pred_pos[0].copy()
    scaled_pos = origin + s * (pred_pos - origin)

    # === Rotation: per-pair adjacent-frame relative rotation error (SO(3) geodesic), in degrees ===
    # 旧公式：比较 pred 与 gt 的相邻帧相对旋转
    n_pairs = N - 1
    rot_errors = np.zeros(n_pairs)
    for i in range(n_pairs):
        pred_rel_R = pred_c2w[i, :3, :3].T @ pred_c2w[i + 1, :3, :3]
        gt_rel_R = gt_c2w[i, :3, :3].T @ gt_c2w[i + 1, :3, :3]
        diff_R = pred_rel_R.T @ gt_rel_R
        cos_angle = np.clip((np.trace(diff_R) - 1) / 2, -1.0, 1.0)
        rot_errors[i] = np.degrees(np.arccos(cos_angle))

    rot_reward = -float(np.mean(rot_errors))

    # === Translation: per-frame absolute position error, normalized, linear mean ===
    per_frame_dist = np.linalg.norm(scaled_pos - gt_pos, axis=1)
    per_frame_dist_norm = per_frame_dist / (gt_traj_len + 1e-8)
    trans_reward = -float(np.mean(per_frame_dist_norm[1:]))  # 跳过 i=0（≈0）

    w_sum = rot_weight + trans_weight
    reward = (rot_weight * rot_reward + trans_weight * trans_reward) / w_sum

    details = {
        "scale": s,
        "gt_trajectory_length": gt_traj_len,
        "pred_trajectory_length": pred_traj_len,
        "rot_formula": "adjacent_relative_rotation_geodesic_deg_linear_mean",
        "trans_formula": "absolute_pos_norm_linear_mean",
        "rot_mean_error_deg": float(np.mean(rot_errors)),
        "rot_median_error_deg": float(np.median(rot_errors)),
        "rot_max_error_deg": float(np.max(rot_errors)),
        "rot_reward": rot_reward,
        "trans_mean_dist": float(np.mean(per_frame_dist[1:])),
        "trans_median_dist": float(np.median(per_frame_dist[1:])),
        "trans_mean_dist_norm": float(np.mean(per_frame_dist_norm[1:])),
        "trans_max_dist_norm": float(np.max(per_frame_dist_norm)),
        "trans_reward": trans_reward,
        "rot_weight": rot_weight,
        "trans_weight": trans_weight,
        "per_frame_rot_errors_deg": rot_errors.tolist(),
        "per_frame_trans_dists": per_frame_dist.tolist(),
        "per_frame_trans_dists_norm": per_frame_dist_norm.tolist(),
    }
    return reward, details


# ==================== Reward 4: VideoAlign ====================


def compute_reward_video_quality(videoalign_scores: dict):
    reward = videoalign_scores.get("Overall", 0.0)
    return reward, {
        "VQ": videoalign_scores.get("VQ", 0.0),
        "MQ": videoalign_scores.get("MQ", 0.0),
        "TA": videoalign_scores.get("TA", 0.0),
        "Overall": reward,
    }


# ==================== GT Camera Parsing ====================


def parse_camera_txt(path: str, H: int, W: int):
    """
    Parse camera.txt -> (intrinsics, w2c).
    Each line: frame fx fy cx cy d1 d2 w2c(3x4 row-major)
    fx/fy/cx/cy are normalized. w2c is OpenCV convention.
    """
    raw_entries = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = list(map(float, line.split()))
            raw_entries.append(vals)

    raw_entries.sort(key=lambda v: v[0])
    N = len(raw_entries)
    intrinsics = np.zeros((N, 3, 3), dtype=np.float64)
    w2c_arr = np.zeros((N, 3, 4), dtype=np.float64)

    for idx, vals in enumerate(raw_entries):
        fx, fy = vals[1] * W, vals[2] * H
        cx, cy = vals[3] * W, vals[4] * H
        intrinsics[idx] = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        w2c_arr[idx] = np.array(vals[7:19], dtype=np.float64).reshape(3, 4)

    return intrinsics, w2c_arr


# ==================== Aggregate ====================


def _compute_feature_sim_from_npz(
    dinov2_features_path: str,
    da3_data: dict,
    compare_mode: str = "first_frame",
    conf_threshold: float = 0.0,
    device: str = "cpu",
) -> tuple:
    """利用 dinov2_features.npz + da3_data 计算 warping cosine similarity。

    等价于旧 step_dinov2_featup.py 的计算逻辑，但输入来自分离的 npz 文件，
    使得特征提取可以与 DA3 完全并行。

    Returns:
        (reward_feature_sim: float, details: dict)
    """
    npz = np.load(dinov2_features_path, allow_pickle=True)
    # 加载为 fp16 直接搬 GPU（[49,384,560,560] fp16 ≈ 5.9 GB；fp32 会爆到 11.8 GB）。
    # warping/grid_sample 在 fp16 上数值稳定（cosine similarity 用 dim=1 归一化）。
    raw = npz["features"]
    if raw.dtype == np.float16:
        features = torch.from_numpy(raw).to(device)  # [N,C,Hf,Wf] fp16
    else:
        features = torch.from_numpy(raw).to(device).to(torch.float16)
    H_feat = int(npz["H_feat"])
    W_feat = int(npz["W_feat"])
    N = features.shape[0]

    depth_all = torch.from_numpy(np.array(da3_data["depth"]).astype(np.float32)).to(device)
    w2c_all = torch.from_numpy(
        _to_4x4(np.array(da3_data["extrinsics"])).astype(np.float32)).to(device)
    c2w_all = torch.linalg.inv(w2c_all)
    K_all = torch.from_numpy(np.array(da3_data["intrinsics"]).astype(np.float32)).to(device)
    conf_raw = da3_data.get("conf")
    conf_all = (
        torch.from_numpy(np.asarray(conf_raw).astype(np.float32)).to(device)
        if conf_raw is not None and np.asarray(conf_raw).size > 0 else None
    )

    def _align_depth(depth, K, conf):
        H_d, W_d = depth.shape
        if H_d == H_feat and W_d == W_feat:
            return depth, K, conf
        K_new = K.clone()
        K_new[0, :] *= W_feat / W_d
        K_new[1, :] *= H_feat / H_d
        d_r = torch.nn.functional.interpolate(
            depth[None, None], size=(H_feat, W_feat), mode="bilinear", align_corners=False
        ).squeeze()
        c_r = None
        if conf is not None:
            c_r = torch.nn.functional.interpolate(
                conf[None, None], size=(H_feat, W_feat), mode="bilinear", align_corners=False
            ).squeeze()
        return d_r, K_new, c_r

    def _warp_grid(depth, K_src, c2w_src, K_ref, c2w_ref, conf):
        H, W = H_feat, W_feat
        v_c, u_c = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )
        ones = torch.ones_like(u_c)
        uv1 = torch.stack([u_c, v_c, ones], dim=-1).reshape(-1, 3).T
        pts_cam = torch.inverse(K_src) @ uv1 * depth.reshape(1, -1)
        pts_w = c2w_src[:3, :3] @ pts_cam + c2w_src[:3, 3:]
        pts_r = c2w_ref[:3, :3].T @ (pts_w - c2w_ref[:3, 3:])
        pts_2d = K_ref @ pts_r
        z = pts_2d[2:, :]
        pts_2d = pts_2d[:2, :] / (z + 1e-8)
        gx = (pts_2d[0].reshape(H, W) / (W - 1)) * 2 - 1
        gy = (pts_2d[1].reshape(H, W) / (H - 1)) * 2 - 1
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
        valid = (gx >= -1) & (gx <= 1) & (gy >= -1) & (gy <= 1) & (z.reshape(H, W) > 0) & (depth > 0)
        if conf is not None:
            valid = valid & (conf > conf_threshold)
        return grid, valid.float().unsqueeze(0).unsqueeze(0)

    def _pair_score(src_f, ref_f, grid, mask):
        # grid 是 fp32（来自 K^-1 投影计算），features 可能是 fp16。
        # grid_sample 要求 input/grid 同 dtype，统一到 features.dtype。
        if grid.dtype != ref_f.dtype:
            grid = grid.to(ref_f.dtype)
        ref_s = torch.nn.functional.grid_sample(
            ref_f, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        cos = torch.nn.functional.cosine_similarity(src_f, ref_s, dim=1).squeeze()
        score_map = 1.0 - cos
        m2d = mask.squeeze()
        vc = m2d.sum().item()
        if vc < 10:
            return 1.0, 0.0
        dissim = (score_map * m2d).sum().item() / (vc + 1e-8)
        overlap = vc / (H_feat * W_feat)
        return dissim, overlap

    depths_a, Ks_a, confs_a = [], [], []
    for i in range(N):
        d, k, c = _align_depth(depth_all[i], K_all[i],
                                conf_all[i] if conf_all is not None else None)
        depths_a.append(d); Ks_a.append(k); confs_a.append(c)

    dissim_scores, overlap_ratios = [], []

    if compare_mode == "first_frame":
        ref_f = features[0:1]
        for i in range(1, N):
            grid, mask = _warp_grid(depths_a[i], Ks_a[i], c2w_all[i], Ks_a[0], c2w_all[0], confs_a[i])
            d, o = _pair_score(features[i:i+1], ref_f, grid, mask)
            dissim_scores.append(d); overlap_ratios.append(o)
    elif compare_mode == "adjacent":
        for i in range(1, N):
            grid, mask = _warp_grid(depths_a[i], Ks_a[i], c2w_all[i], Ks_a[i-1], c2w_all[i-1], confs_a[i])
            d, o = _pair_score(features[i:i+1], features[i-1:i], grid, mask)
            dissim_scores.append(d); overlap_ratios.append(o)
    else:  # all_pairs
        for i in range(N):
            row_d, row_o = [], []
            for j in range(N):
                if i == j:
                    continue
                grid, mask = _warp_grid(depths_a[i], Ks_a[i], c2w_all[i], Ks_a[j], c2w_all[j], confs_a[i])
                d, o = _pair_score(features[i:i+1], features[j:j+1], grid, mask)
                row_d.append(d); row_o.append(o)
            dissim_scores.append(float(np.mean(row_d)) if row_d else 1.0)
            overlap_ratios.append(float(np.mean(row_o)) if row_o else 0.0)

    mean_dissim = float(np.nanmean(dissim_scores)) if dissim_scores else 1.0
    reward = max(0.0, 1.0 - mean_dissim)
    details = {
        "compare_mode": compare_mode,
        "mean_dissim": mean_dissim,
        "mean_overlap": float(np.nanmean(overlap_ratios)) if overlap_ratios else 0.0,
        "num_pairs": len(dissim_scores),
        "source": "dinov2_features_npz",
    }
    return reward, details


def compute_all_rewards(
    da3_path: str,
    label_maps_path: str,
    feature_sim_path: str,
    videoalign_path: str,
    gt_camera_txt: str,
    H_img: int, W_img: int,
    rewards_to_compute=None,
    weights=None,
    device: str = "cpu",
    rot_weight: float = 0.5,
    trans_weight: float = 0.5,
    conf_threshold: float = 0.0,
    geo_compare_mode: str = "all_pairs",  # 默认 AP（最终选定）
    feature_compare_mode: str = "first_frame",
    dinov2_features_path: Optional[str] = None,
) -> dict:
    """Load intermediates and compute selected rewards.

    feature_sim 的两种流程：
      - 新流程（推荐）：提供 dinov2_features_path 和 da3_path，在此聚合阶段完成 warping。
      - 旧流程（兼容）：feature_sim_path 存在（step_dinov2_featup 输出），直接读取 JSON。
    """

    ALL_REWARDS = [
        "geo_semantic", "geo_global", "feature_sim",
        "camera_traj", "video_quality",
    ]
    if rewards_to_compute is None:
        rewards_to_compute = ALL_REWARDS

    if weights is None:
        # 经过 387 sample × 8 rollout 的 within-sample std 标定，
        # 目标 w·within_std ≈ 0.10，让 5 项 reward 在 advantage 中
        # 贡献度均衡（约各 ~28%~39%）。geo_semantic 因 SAM3 不稳定弃用。
        weights = {
            "geo_semantic":  0.0,    # 弃用
            "geo_global":    7.7,
            "feature_sim":   5.3,
            "camera_rot":    0.92,
            "camera_trans":  3.6,
            "video_quality": 0.67,
            # 兼容旧调用：camera_traj = 0.5*(rot+trans)，等价于
            # 0.92*rot + 3.6*trans 的近似时使用。新流程已不再用此 key。
            "camera_traj":   0.0,
        }

    result = {"details": {}}

    # DA3 data 在需要地理/相机/feature_sim（新流程）时加载
    da3_data = None
    need_da3 = any(r in rewards_to_compute
                   for r in ["geo_semantic", "geo_global", "camera_traj"])
    # 新流程的 feature_sim 也需要 DA3（warping）
    need_da3_for_feat = (
        "feature_sim" in rewards_to_compute
        and dinov2_features_path is not None
        and os.path.isfile(str(dinov2_features_path))
        and not (feature_sim_path is not None and os.path.isfile(str(feature_sim_path)))
    )
    if need_da3 or need_da3_for_feat:
        if os.path.isfile(str(da3_path)):
            da3_data = dict(np.load(da3_path, allow_pickle=True))

    if "geo_semantic" in rewards_to_compute:
        try:
            lm = np.load(label_maps_path, allow_pickle=True)
            r, d = compute_reward_geo_semantic(
                da3_data, lm["label_maps"], H_img, W_img,
                conf_threshold=conf_threshold, device=device,
                compare_mode=geo_compare_mode,
            )
            result["reward_geo_semantic"] = r
            result["details"]["geo_semantic"] = d
            print(f"[Reward] Geo+Semantic ({geo_compare_mode}): {r:.4f}")
        except Exception as e:
            print(f"[Reward] Geo+Semantic FAILED: {e}")
            result["reward_geo_semantic"] = float("nan")

    if "geo_global" in rewards_to_compute:
        try:
            r, d = compute_reward_geo_global(
                da3_data, H_img, W_img,
                conf_threshold=conf_threshold, device=device,
                compare_mode=geo_compare_mode,
            )
            result["reward_geo_global"] = r
            result["details"]["geo_global"] = d
            print(f"[Reward] Geo Global ({geo_compare_mode}): {r:.4f}")
        except Exception as e:
            print(f"[Reward] Geo Global FAILED: {e}")
            result["reward_geo_global"] = float("nan")

    if "feature_sim" in rewards_to_compute:
        try:
            _use_new_flow = (
                dinov2_features_path is not None
                and os.path.isfile(str(dinov2_features_path))
                and da3_data is not None
            )
            _use_old_flow = (
                feature_sim_path is not None
                and os.path.isfile(str(feature_sim_path))
            )

            if _use_new_flow:
                # 新流程：DINOv2 features npz + DA3 warping
                print(f"[Reward] Feature sim: using new flow (dinov2_features.npz + DA3 warping)")
                r, d = _compute_feature_sim_from_npz(
                    str(dinov2_features_path), da3_data,
                    compare_mode=feature_compare_mode,
                    conf_threshold=conf_threshold,
                    device=device,
                )
            elif _use_old_flow:
                # 旧流程：直接读取 step_dinov2_featup 输出的 JSON
                print(f"[Reward] Feature sim: using legacy flow (feature_sim_reward.json)")
                with open(feature_sim_path, "r") as f:
                    fs = json.load(f)
                r = fs["reward_feature_sim"]
                d = fs.get("details", {})
            else:
                raise FileNotFoundError(
                    f"No feature_sim intermediate found: "
                    f"dinov2_features_path={dinov2_features_path}, "
                    f"feature_sim_path={feature_sim_path}"
                )

            result["reward_feature_sim"] = r
            result["details"]["feature_sim"] = d
            print(f"[Reward] Feature sim ({feature_compare_mode}): {r:.4f}")
        except Exception as e:
            print(f"[Reward] Feature sim FAILED: {e}")
            result["reward_feature_sim"] = float("nan")

    if "camera_traj" in rewards_to_compute:
        try:
            _, gt_w2c = parse_camera_txt(gt_camera_txt, H_img, W_img)
            r, d = compute_reward_camera_traj(
                da3_data, gt_w2c,
                rot_weight=rot_weight, trans_weight=trans_weight)
            result["reward_camera_traj"] = r
            # 新流程：rot / trans 单独暴露，便于 Total 用独立权重聚合
            result["reward_camera_rot"] = float(d.get("rot_reward", float("nan")))
            result["reward_camera_trans"] = float(d.get("trans_reward", float("nan")))
            result["details"]["camera_traj"] = d
            print(f"[Reward] Camera traj: {r:.4f} "
                  f"(rot={d['rot_mean_error_deg']:.2f}deg, "
                  f"trans={d['trans_mean_dist']:.4f}, "
                  f"scale={d['scale']:.4f})")
        except Exception as e:
            print(f"[Reward] Camera traj FAILED: {e}")
            result["reward_camera_traj"] = float("nan")
            result["reward_camera_rot"] = float("nan")
            result["reward_camera_trans"] = float("nan")

    if "video_quality" in rewards_to_compute:
        try:
            with open(videoalign_path, "r") as f:
                va = json.load(f)
            r, d = compute_reward_video_quality(va)
            result["reward_video_quality"] = r
            result["details"]["video_quality"] = d
            print(f"[Reward] Video quality: {r:.4f}")
        except Exception as e:
            print(f"[Reward] Video quality FAILED: {e}")
            result["reward_video_quality"] = float("nan")

    # Weighted total — 新方案：camera_rot / camera_trans 独立加权，
    # camera_traj 不再直接进入 total（只作为兼容字段保留）；
    # geo_semantic 默认权重 0（弃用），如需启用请显式传 weights。
    _agg_keys = [
        ("reward_geo_semantic", "geo_semantic"),
        ("reward_geo_global",   "geo_global"),
        ("reward_feature_sim",  "feature_sim"),
        ("reward_camera_rot",   "camera_rot"),
        ("reward_camera_trans", "camera_trans"),
        ("reward_video_quality", "video_quality"),
    ]
    # 兼容旧调用：若调用方未单独设置 camera_rot/trans 权重，但传了 camera_traj，
    # 则 fallback 到旧聚合方式（weights["camera_traj"] * (rot+trans)/2）。
    use_legacy_camera = (
        "camera_rot" not in weights and "camera_trans" not in weights
        and "camera_traj" in weights
    )
    if use_legacy_camera:
        _agg_keys = [
            ("reward_geo_semantic", "geo_semantic"),
            ("reward_geo_global",   "geo_global"),
            ("reward_feature_sim",  "feature_sim"),
            ("reward_camera_traj",  "camera_traj"),
            ("reward_video_quality", "video_quality"),
        ]

    total = 0.0
    contribs = {}
    for key, wk in _agg_keys:
        if key not in result:
            continue
        val = result[key]
        w = weights.get(wk, 0.0)
        if val is not None and not np.isnan(val) and w != 0.0:
            contribs[wk] = w * val
            total += w * val
    result["reward_total"] = total
    result["reward_contributions"] = contribs
    print(f"[Reward] Total: {total:.4f}  "
          + " ".join(f"{k}={v:+.3f}" for k, v in contribs.items()))

    # reward_total_global：仅几何/相机/视觉组件（不含 geo_semantic），
    # 用于诊断"semantic 信号失效时的纯几何 total"
    total_global = 0.0
    _global_keys = [
        ("reward_geo_global",   "geo_global"),
        ("reward_feature_sim",  "feature_sim"),
        ("reward_camera_rot",   "camera_rot"),
        ("reward_camera_trans", "camera_trans"),
        ("reward_video_quality", "video_quality"),
    ]
    if use_legacy_camera:
        _global_keys = [
            ("reward_geo_global",   "geo_global"),
            ("reward_feature_sim",  "feature_sim"),
            ("reward_camera_traj",  "camera_traj"),
            ("reward_video_quality", "video_quality"),
        ]
    for key, wk in _global_keys:
        if key not in result:
            continue
        val = result[key]
        w = weights.get(wk, 0.0)
        if val is not None and not np.isnan(val) and w != 0.0:
            total_global += w * val
    result["reward_total_global"] = total_global

    return result


# ==================== CLI ====================


def main():
    parser = argparse.ArgumentParser(description="Reward metrics")
    parser.add_argument("--da3_path", type=Path, required=True)
    parser.add_argument("--label_maps_path", type=Path, required=True)
    parser.add_argument("--feature_sim_path", type=Path, required=True)
    parser.add_argument("--videoalign_path", type=Path, required=True)
    parser.add_argument("--gt_camera_txt", type=Path, required=True)
    parser.add_argument("--H_img", type=int, required=True)
    parser.add_argument("--W_img", type=int, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--rewards", type=str, default="all",
                        help="Comma-separated reward names or 'all'")
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--conf_threshold", type=float, default=0.0)
    parser.add_argument("--geo_compare_mode", type=str,
                        default="first_frame",
                        choices=["first_frame", "adjacent", "all_pairs"])
    args = parser.parse_args()

    rlist = (None if args.rewards == "all"
             else [r.strip() for r in args.rewards.split(",")])
    result = compute_all_rewards(
        da3_path=str(args.da3_path),
        label_maps_path=str(args.label_maps_path),
        feature_sim_path=str(args.feature_sim_path),
        videoalign_path=str(args.videoalign_path),
        gt_camera_txt=str(args.gt_camera_txt),
        H_img=args.H_img, W_img=args.W_img,
        rewards_to_compute=rlist, device=args.device,
        conf_threshold=args.conf_threshold,
        geo_compare_mode=args.geo_compare_mode,
    )

    if args.output_json:
        out = args.output_json
        out.parent.mkdir(parents=True, exist_ok=True)
        ser = {k: v for k, v in result.items() if k != "details"}
        out.write_text(
            json.dumps(ser, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"Saved to {out}")


if __name__ == "__main__":
    main()
