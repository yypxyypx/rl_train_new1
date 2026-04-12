#!/usr/bin/env python3
"""
reward_metrics.py
=================
Reward computation module.

Rewards:
  1 : Geo + Semantic (DA3 depth + SAM3 labels)
  1b: Geo Global (depth only)
  2 : DINOv2 Feature Similarity (from pre-computed JSON)
  3 : Camera Trajectory (DA3 pred vs GT) -- rewritten
  4 : VideoAlign Video Quality

Convention:
  DA3 extrinsics = w2c OpenCV (N, 3, 4)
  GT camera.txt  = w2c OpenCV
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


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


# ==================== Reward 1: Geo + Semantic ====================


def compute_reward_geo_semantic(
    da3_data: dict, label_maps: np.ndarray,
    H_img: int, W_img: int,
    conf_threshold: float = 0.0, device: str = "cpu",
) -> Tuple[float, dict]:
    """DA3 extrinsics = w2c, inverted to c2w internally."""
    depth_all = torch.from_numpy(da3_data["depth"]).to(device)
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

    frame_scores, matched_pixel_rates = [], []
    valid_pixel_counts, fg_match_rates = [], []

    for i in range(1, N):
        depth_i, K_i, conf_i = align_depth_to_image(
            depth_all[i], K_all[i], H_img, W_img,
            conf_all[i] if conf_all is not None else None,
        )
        flow_grid, valid_mask, proj_depth_in_ref = build_warp_grid(
            H_img, W_img, depth=depth_i, K_src=K_i, c2w_src=c2w_all[i],
            K_ref=K_ref, c2w_ref=c2w_all[0],
            conf=conf_i, conf_threshold=conf_threshold,
        )
        mask_2d = valid_mask.squeeze().bool()
        valid_count = mask_2d.sum().item()
        if valid_count < 10:
            frame_scores.append(0.0)
            matched_pixel_rates.append(0.0)
            valid_pixel_counts.append(0)
            fg_match_rates.append(0.0)
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
        label_match = fg_match | bg_match

        rel_err = torch.abs(proj_depth_in_ref - ref_depth_sampled) / (ref_depth_sampled + 1e-8)
        per_pixel_score = torch.exp(-rel_err)

        numerator = (per_pixel_score * fg_match.float()
                     + per_pixel_score * bg_match.float() * 0.8).sum()
        frame_scores.append((numerator / (valid_count + 1e-8)).item())
        matched_pixel_rates.append(float(label_match.sum().item() / (valid_count + 1e-8)))
        fg_total = (mask_2d & (label_i > 0)).sum().item()
        fg_match_rates.append(float(fg_match.sum().item() / (fg_total + 1e-8)))
        valid_pixel_counts.append(int(valid_count))

    if not frame_scores:
        return 0.0, {}

    reward = float(np.mean(frame_scores))
    details = {
        "per_frame_score": frame_scores,
        "per_frame_label_match_rate": matched_pixel_rates,
        "per_frame_fg_match_rate": fg_match_rates,
        "per_frame_valid_pixels": valid_pixel_counts,
        "mean_score": reward,
        "mean_label_match_rate": float(np.mean(matched_pixel_rates)),
        "mean_fg_match_rate": float(np.mean(fg_match_rates)) if fg_match_rates else float("nan"),
        "mean_valid_pixels": int(np.mean(valid_pixel_counts)) if valid_pixel_counts else 0,
    }
    return reward, details


# ==================== Reward 1b: Geo Global ====================


def compute_reward_geo_global(
    da3_path: str, H_img: int, W_img: int,
    conf_threshold: float = 0.5, device: str = "cpu",
) -> Tuple[float, dict]:
    """DA3 extrinsics = w2c, inverted to c2w internally."""
    da3 = np.load(da3_path, allow_pickle=True)
    depth_np, K_np = da3["depth"], da3["intrinsics"]
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

    frame_scores, valid_px_list = [], []

    for i in range(1, N):
        depth_i, K_i_t, conf_i = align_depth_to_image(
            torch.from_numpy(depth_np[i].astype(np.float32)).to(device),
            torch.from_numpy(K_np[i].astype(np.float32)).to(device),
            H_img, W_img,
            torch.from_numpy(conf_np[i].astype(np.float32)).to(device) if conf_np is not None else None,
        )
        flow_grid, valid_mask, proj_depth_in_ref = build_warp_grid(
            H_img, W_img, depth=depth_i, K_src=K_i_t, c2w_src=c2w_all[i],
            K_ref=K_ref_t, c2w_ref=c2w_all[0],
            conf=conf_i, conf_threshold=conf_threshold,
        )
        mask_2d = valid_mask.squeeze().bool()
        valid_count = mask_2d.sum().item()
        if valid_count < 10:
            frame_scores.append(0.0)
            valid_px_list.append(0)
            continue

        ref_depth_sampled = F.grid_sample(
            depth_ref[None, None], flow_grid, mode="bilinear",
            padding_mode="zeros", align_corners=True,
        ).squeeze()

        rel_err = torch.abs(proj_depth_in_ref - ref_depth_sampled) / (ref_depth_sampled + 1e-8)
        per_pixel_score = torch.exp(-rel_err)
        frame_scores.append(
            (per_pixel_score * mask_2d.float()).sum().item() / (valid_count + 1e-8))
        valid_px_list.append(int(valid_count))

    if not frame_scores:
        return 0.0, {}

    reward = float(np.mean(frame_scores))
    return reward, {
        "per_frame_score": frame_scores,
        "per_frame_valid_pixels": valid_px_list,
        "mean_score": reward,
        "mean_valid_pixels": int(np.mean(valid_px_list)) if valid_px_list else 0,
    }


# ==================== Reward 3: Camera Trajectory (rewritten) ====================


def _trajectory_length(positions):
    import numpy as np
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
      4. Rotation reward: adjacent-frame relative rotation angle diff, -mean(theta)
      5. Translation reward: per-frame absolute distance, mean(-exp(dist/0.3))
    """
    import numpy as np

    pred_w2c = _to_4x4(da3_data["extrinsics"]).astype(np.float64)
    gt_w2c_4 = _to_4x4(gt_w2c).astype(np.float64)

    N = min(len(pred_w2c), len(gt_w2c_4))
    pred_w2c = pred_w2c[:N]
    gt_w2c_4 = gt_w2c_4[:N]

    if N < 2:
        return 0.0, {"error": "frames < 2"}

    # 1. First-frame pose alignment
    transform = gt_w2c_4[0] @ np.linalg.inv(pred_w2c[0])
    aligned_pred_w2c = np.array([transform @ pred_w2c[i] for i in range(N)])

    # 2. Convert to c2w
    pred_c2w = np.array([np.linalg.inv(aligned_pred_w2c[i]) for i in range(N)])
    gt_c2w = np.array([np.linalg.inv(gt_w2c_4[i]) for i in range(N)])

    pred_pos = pred_c2w[:, :3, 3]
    gt_pos = gt_c2w[:, :3, 3]

    # 3. Trajectory length scale alignment
    gt_traj_len = _trajectory_length(gt_pos)
    pred_traj_len = _trajectory_length(pred_pos)
    s = gt_traj_len / (pred_traj_len + 1e-8) if pred_traj_len > 1e-8 else 1.0

    origin = pred_pos[0].copy()
    scaled_pos = origin + s * (pred_pos - origin)

    # 4. Rotation reward (adjacent-frame relative rotation angle diff)
    n_pairs = N - 1
    rot_errors = np.zeros(n_pairs)

    for i in range(n_pairs):
        pred_rel_R = pred_c2w[i, :3, :3].T @ pred_c2w[i + 1, :3, :3]
        gt_rel_R = gt_c2w[i, :3, :3].T @ gt_c2w[i + 1, :3, :3]
        diff_R = pred_rel_R.T @ gt_rel_R
        cos_angle = np.clip((np.trace(diff_R) - 1) / 2, -1.0, 1.0)
        rot_errors[i] = np.degrees(np.arccos(cos_angle))

    rot_reward = -float(np.mean(rot_errors))

    # 5. Translation reward (per-frame absolute distance)
    per_frame_dist = np.linalg.norm(scaled_pos - gt_pos, axis=1)
    trans_rewards = -np.exp(per_frame_dist / 0.3)
    trans_reward = float(np.mean(trans_rewards))

    # 6. Weighted combination
    w_sum = rot_weight + trans_weight
    reward = (rot_weight * rot_reward + trans_weight * trans_reward) / w_sum

    details = {
        "scale": s,
        "gt_trajectory_length": gt_traj_len,
        "pred_trajectory_length": pred_traj_len,
        "rot_mean_error_deg": float(np.mean(rot_errors)),
        "rot_median_error_deg": float(np.median(rot_errors)),
        "rot_reward": rot_reward,
        "trans_mean_dist": float(np.mean(per_frame_dist)),
        "trans_median_dist": float(np.median(per_frame_dist)),
        "trans_reward": trans_reward,
        "rot_weight": rot_weight,
        "trans_weight": trans_weight,
        "per_frame_rot_errors_deg": rot_errors.tolist(),
        "per_frame_trans_dists": per_frame_dist.tolist(),
        "per_frame_trans_rewards": trans_rewards.tolist(),
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
    import numpy as np

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
) -> dict:
    """Load intermediates and compute selected rewards."""
    import json
    import numpy as np

    ALL_REWARDS = [
        "geo_semantic", "geo_global", "feature_sim",
        "camera_traj", "video_quality",
    ]
    if rewards_to_compute is None:
        rewards_to_compute = ALL_REWARDS

    if weights is None:
        weights = {
            "geo_semantic": 3.0, "geo_global": 2.0,
            "feature_sim": 5.0, "camera_traj": 8.0,
            "video_quality": 1.5,
        }

    result = {"details": {}}

    da3_data = None
    if any(r in rewards_to_compute for r in ["geo_semantic", "geo_global", "camera_traj"]):
        da3_data = dict(np.load(da3_path, allow_pickle=True))

    if "geo_semantic" in rewards_to_compute:
        try:
            lm = np.load(label_maps_path, allow_pickle=True)
            r, d = compute_reward_geo_semantic(
                da3_data, lm["label_maps"], H_img, W_img, device=device)
            result["reward_geo_semantic"] = r
            result["details"]["geo_semantic"] = d
            print(f"[Reward] Geo+Semantic: {r:.4f}")
        except Exception as e:
            print(f"[Reward] Geo+Semantic FAILED: {e}")
            result["reward_geo_semantic"] = float("nan")

    if "geo_global" in rewards_to_compute:
        try:
            r, d = compute_reward_geo_global(da3_path, H_img, W_img, device=device)
            result["reward_geo_global"] = r
            result["details"]["geo_global"] = d
            print(f"[Reward] Geo Global: {r:.4f}")
        except Exception as e:
            print(f"[Reward] Geo Global FAILED: {e}")
            result["reward_geo_global"] = float("nan")

    if "feature_sim" in rewards_to_compute:
        try:
            with open(feature_sim_path, "r") as f:
                fs = json.load(f)
            result["reward_feature_sim"] = fs["reward_feature_sim"]
            result["details"]["feature_sim"] = fs.get("details", {})
            print(f"[Reward] Feature sim: {result['reward_feature_sim']:.4f}")
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
            result["details"]["camera_traj"] = d
            print(f"[Reward] Camera traj: {r:.4f} "
                  f"(rot={d['rot_mean_error_deg']:.2f}deg, "
                  f"trans={d['trans_mean_dist']:.4f}, scale={d['scale']:.4f})")
        except Exception as e:
            print(f"[Reward] Camera traj FAILED: {e}")
            result["reward_camera_traj"] = float("nan")

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

    total = 0.0
    for key, wk in [("reward_geo_semantic", "geo_semantic"),
                     ("reward_geo_global", "geo_global"),
                     ("reward_feature_sim", "feature_sim"),
                     ("reward_camera_traj", "camera_traj"),
                     ("reward_video_quality", "video_quality")]:
        val = result.get(key, 0.0)
        w = weights.get(wk, 0.0)
        if val is not None and not np.isnan(val):
            total += w * val
    result["reward_total"] = total
    print(f"[Reward] Total: {total:.4f}")

    total_global = 0.0
    for key, wk in [("reward_geo_global", "geo_global"),
                     ("reward_feature_sim", "feature_sim"),
                     ("reward_camera_traj", "camera_traj"),
                     ("reward_video_quality", "video_quality")]:
        val = result.get(key, 0.0)
        w = weights.get(wk, 0.0)
        if val is not None and not np.isnan(val):
            total_global += w * val
    result["reward_total_global"] = total_global

    return result


# ==================== CLI ====================


def main():
    import json
    from pathlib import Path

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
    args = parser.parse_args()

    rlist = None if args.rewards == "all" else [r.strip() for r in args.rewards.split(",")]
    result = compute_all_rewards(
        da3_path=str(args.da3_path),
        label_maps_path=str(args.label_maps_path),
        feature_sim_path=str(args.feature_sim_path),
        videoalign_path=str(args.videoalign_path),
        gt_camera_txt=str(args.gt_camera_txt),
        H_img=args.H_img, W_img=args.W_img,
        rewards_to_compute=rlist, device=args.device,
    )

    if args.output_json:
        out = args.output_json
        out.parent.mkdir(parents=True, exist_ok=True)
        ser = {k: v for k, v in result.items() if k != "details"}
        out.write_text(json.dumps(ser, ensure_ascii=False, indent=2, default=str),
                       encoding="utf-8")
        print(f"Saved to {out}")


if __name__ == "__main__":
    main()
