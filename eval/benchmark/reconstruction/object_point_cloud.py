#!/usr/bin/env python3
"""
object_point_cloud.py — 物体级点云一致性评估。

复用 RL/eval/object_point_eval/object_point_eval.py 的核心逻辑，
新增 first_frame 对齐模式。

三个接口：
  evaluate_object  — 仅物体级
  evaluate_global  — 仅全局（委托给 global_point_cloud.py）
  evaluate_both    — 两者都跑

坐标系：DA3 extrinsics 为 w2c，取逆得 c2w。
"""

from typing import Optional, Tuple

import cv2
import numpy as np

from .global_point_cloud import (
    _to_4x4, depth_to_pointcloud, umeyama_align,
    apply_similarity_transform, align_pointclouds_via_cameras,
    fps_sample, compute_metrics, icp_refine, first_frame_align,
    evaluate_global, evaluate_global_both,
)


# ═══════════════ mask 工具 ══════════════════════════════════════

def resize_masks_to_depth(masks: np.ndarray, depth_h: int, depth_w: int) -> np.ndarray:
    N_obj, T, H_f, W_f = masks.shape
    if H_f == depth_h and W_f == depth_w:
        return masks
    masks_r = np.zeros((N_obj, T, depth_h, depth_w), dtype=bool)
    for i in range(N_obj):
        for t in range(T):
            m = masks[i, t].astype(np.uint8)
            masks_r[i, t] = cv2.resize(m, (depth_w, depth_h),
                                        interpolation=cv2.INTER_NEAREST).astype(bool)
    return masks_r


def select_top_objects(pred_masks, gt_masks, object_names, top_k=8, min_area=0.005):
    N_obj, T, H, W = gt_masks.shape
    pixel_total = H * W
    gt_areas = gt_masks.sum(axis=(2, 3)).mean(axis=1) / pixel_total
    valid_idx = np.where(gt_areas >= min_area)[0]
    if len(valid_idx) == 0:
        return pred_masks[:0], gt_masks[:0], []
    sorted_valid = valid_idx[np.argsort(gt_areas[valid_idx])[::-1]]
    sel_idx = sorted_valid[:top_k]
    selected_names = [object_names[i] for i in sel_idx]
    return pred_masks[sel_idx], gt_masks[sel_idx], selected_names


def masked_depth_to_pointcloud(
    depth, mask, intrinsics, c2w,
    conf=None, conf_thresh=0.0, depth_min=1e-3, depth_max=1e6,
):
    depth_masked = depth.copy()
    depth_masked[~mask] = 0.0
    return depth_to_pointcloud(
        depth_masked, intrinsics, c2w,
        conf=conf, conf_thresh=conf_thresh,
        depth_min=depth_min, depth_max=depth_max)


# ═══════════════ 物体级评估 ════════════════════════════════════

def evaluate_objects_core(
    pred_masks_npz: str, gt_masks_npz: str,
    pred_depth_npz: str,
    gt_depth: np.ndarray, intrinsics: np.ndarray, c2w: np.ndarray,
    align: str = "camera",
    top_k: int = 8, min_area: float = 0.005,
    n_fps_global: int = 20000, n_fps_min: int = 200,
    conf_thresh: float = 0.0, device: str = "cpu",
) -> dict:
    """物体级点云评估核心函数。"""

    pred_m_data = dict(np.load(pred_masks_npz, allow_pickle=True))
    gt_m_data = dict(np.load(gt_masks_npz, allow_pickle=True))
    pred_masks_raw = pred_m_data["masks"]
    gt_masks_raw = gt_m_data["masks"]
    object_names = list(gt_m_data["object_names"])

    pred_d_data = dict(np.load(pred_depth_npz, allow_pickle=True))
    pred_depth = pred_d_data["depth"].astype(np.float64)
    pred_intr = pred_d_data["intrinsics"].astype(np.float64)
    pred_conf = pred_d_data.get("conf", None)
    if pred_conf is not None and hasattr(pred_conf, "size") and pred_conf.size == 0:
        pred_conf = None

    # DA3 extrinsics 为 w2c，取逆得 c2w
    pred_w2c = _to_4x4(pred_d_data["extrinsics"].astype(np.float64))
    pred_c2w = np.array([np.linalg.inv(pred_w2c[i]) for i in range(len(pred_w2c))])

    T = min(pred_depth.shape[0], gt_depth.shape[0], intrinsics.shape[0],
            c2w.shape[0], pred_masks_raw.shape[1], gt_masks_raw.shape[1])
    pred_depth = pred_depth[:T]
    gt_depth_n = gt_depth[:T].astype(np.float64)
    intrinsics = intrinsics[:T]
    c2w = c2w[:T]
    pred_intr = pred_intr[:T]
    pred_c2w = pred_c2w[:T]
    pred_masks_raw = pred_masks_raw[:, :T]
    gt_masks_raw = gt_masks_raw[:, :T]
    if pred_conf is not None:
        pred_conf = pred_conf[:T]

    depth_H, depth_W = pred_depth.shape[1], pred_depth.shape[2]
    pred_masks = resize_masks_to_depth(pred_masks_raw, depth_H, depth_W)
    gt_masks = resize_masks_to_depth(gt_masks_raw, depth_H, depth_W)

    gt_valid = gt_depth_n[gt_depth_n > 1e-3]
    depth_cap = float(np.percentile(gt_valid, 98)) * 2.0 if len(gt_valid) > 0 else 1e6

    # 尺度对齐
    if align == "first_frame":
        aligned_c2w_34, scaled_depth, global_s = first_frame_align(
            pred_c2w[:, :3, :], c2w, pred_depth)
        scaled_depth = np.clip(scaled_depth, 1e-3, depth_cap)
        use_gt_cam = True
        use_depth = scaled_depth
        use_c2w = aligned_c2w_34
        use_K = intrinsics
    elif align == "camera":
        _, _, global_s = umeyama_align(pred_c2w[:, :3, 3], _to_4x4(c2w)[:, :3, 3])
        scaled_depth = np.clip(pred_depth * global_s, 1e-3, depth_cap)
        use_gt_cam = True
        use_depth = scaled_depth
        use_c2w = c2w
        use_K = intrinsics
    else:
        global_s = 1.0
        use_gt_cam = False
        use_depth = pred_depth
        use_c2w = pred_c2w[:, :3, :]
        use_K = pred_intr

    pred_masks_sel, gt_masks_sel, sel_names = select_top_objects(
        pred_masks, gt_masks, object_names, top_k=top_k, min_area=min_area)
    K = len(sel_names)
    if K == 0:
        return {"per_object": [], "summary": {}, "align_mode": align, "n_objects": 0}

    frame0_areas = gt_masks_sel[:, 0, :, :].sum(axis=(1, 2))
    frame0_ratios = frame0_areas / (depth_H * depth_W)
    n_fps_per_obj = np.maximum(n_fps_min, (frame0_ratios * n_fps_global).astype(int)).tolist()

    per_object_results = []
    for k in range(K):
        name = sel_names[k]
        pred_mask_k = pred_masks_sel[k]
        gt_mask_k = gt_masks_sel[k]
        n_fps_k = n_fps_per_obj[k]

        gt_pc = masked_depth_to_pointcloud(
            gt_depth_n, gt_mask_k, intrinsics, c2w, depth_max=depth_cap)

        if use_gt_cam or align == "first_frame":
            pred_pc = masked_depth_to_pointcloud(
                use_depth, pred_mask_k, use_K, use_c2w,
                conf=pred_conf, conf_thresh=conf_thresh, depth_max=depth_cap)
        else:
            pred_pc_raw = masked_depth_to_pointcloud(
                use_depth, pred_mask_k, use_K, use_c2w,
                conf=pred_conf, conf_thresh=conf_thresh)
            if len(pred_pc_raw) > 0 and len(gt_pc) > 0:
                pred_pc = align_pointclouds_via_cameras(pred_c2w[:, :3, :], c2w, pred_pc_raw)
                if align == "icp" and len(pred_pc) > 3 and len(gt_pc) > 3:
                    icp_limit = min(20000, len(pred_pc), len(gt_pc))
                    src_icp = fps_sample(pred_pc, icp_limit, device=device)
                    dst_icp = fps_sample(gt_pc, icp_limit, device=device)
                    src_refined, _, _ = icp_refine(src_icp, dst_icp)
                    mu_s, mu_d = src_icp.mean(0), src_refined.mean(0)
                    A_icp = (src_icp - mu_s).T @ (src_refined - mu_d)
                    U_i, _, Vt_i = np.linalg.svd(A_icp)
                    Ss = np.eye(3)
                    if np.linalg.det(U_i) * np.linalg.det(Vt_i) < 0:
                        Ss[2, 2] = -1
                    R_icp = (U_i @ Ss @ Vt_i).T
                    t_icp = mu_d - R_icp @ mu_s
                    pred_pc = (R_icp @ pred_pc.T).T + t_icp
            else:
                pred_pc = pred_pc_raw if len(pred_pc_raw) > 0 else np.zeros((0, 3))

        if len(pred_pc) == 0 or len(gt_pc) == 0:
            nan = float("nan")
            per_object_results.append({
                "object": name, "accuracy": nan, "completeness": nan,
                "chamfer_distance": nan, "n_fps": n_fps_k,
            })
            continue

        pred_fps = fps_sample(pred_pc, n_fps_k, device=device)
        gt_fps = fps_sample(gt_pc, n_fps_k, device=device)
        m = compute_metrics(pred_fps, gt_fps)
        m.update({"object": name, "n_pred_raw": len(pred_pc),
                  "n_gt_raw": len(gt_pc), "n_fps": n_fps_k})
        per_object_results.append(m)

    summary = {}
    for key in ("accuracy", "completeness", "chamfer_distance", "precision", "recall", "fscore"):
        vals = np.array([r.get(key, float("nan")) for r in per_object_results], dtype=np.float64)
        valid = vals[~np.isnan(vals)]
        summary[key] = {
            "mean": float(valid.mean()) if len(valid) else float("nan"),
            "std": float(valid.std()) if len(valid) else float("nan"),
            "count": int(len(valid)),
        }

    return {
        "per_object": per_object_results,
        "summary": summary,
        "align_mode": align,
        "depth_scale": float(global_s),
        "depth_cap": float(depth_cap),
        "n_objects": K,
    }


# ═══════════════ 三个公开接口 ══════════════════════════════════


def evaluate_object(
    pred_masks_npz: str, gt_masks_npz: str,
    pred_depth_npz: str, gt_depth: np.ndarray,
    intrinsics: np.ndarray, c2w: np.ndarray,
    align: str = "camera", device: str = "cpu",
    **kwargs,
) -> dict:
    """仅物体级评估。"""
    return evaluate_objects_core(
        pred_masks_npz, gt_masks_npz, pred_depth_npz,
        gt_depth, intrinsics, c2w, align=align, device=device, **kwargs)


def evaluate_object_both_align(
    pred_masks_npz: str, gt_masks_npz: str,
    pred_depth_npz: str, gt_depth: np.ndarray,
    intrinsics: np.ndarray, c2w: np.ndarray,
    existing_align: str = "camera", device: str = "cpu",
    **kwargs,
) -> dict:
    """同时运行现有模式 + 首帧对齐。"""
    result_existing = evaluate_objects_core(
        pred_masks_npz, gt_masks_npz, pred_depth_npz,
        gt_depth, intrinsics, c2w, align=existing_align, device=device, **kwargs)
    result_ff = evaluate_objects_core(
        pred_masks_npz, gt_masks_npz, pred_depth_npz,
        gt_depth, intrinsics, c2w, align="first_frame", device=device, **kwargs)
    return {existing_align: result_existing, "first_frame": result_ff}


def evaluate_reconstruction_both(
    pred_npz: str, gt_depth: np.ndarray,
    K: np.ndarray, c2w: np.ndarray,
    pred_masks_npz: Optional[str] = None,
    gt_masks_npz: Optional[str] = None,
    align: str = "camera", device: str = "cpu",
    **kwargs,
) -> dict:
    """全局 + 物体级都跑。"""
    result = {"global": evaluate_global(pred_npz, gt_depth, K, c2w, align=align, device=device)}
    if pred_masks_npz and gt_masks_npz:
        result["object"] = evaluate_objects_core(
            pred_masks_npz, gt_masks_npz, pred_npz,
            gt_depth, K, c2w, align=align, device=device, **kwargs)
    return result
