#!/usr/bin/env python3
"""
global_point_cloud.py — 全局点云一致性评估。

完全复用 RL/eval/point_eval/point_eval.py 的核心逻辑，
新增 "first_frame" 对齐模式。

两种对齐模式：
  1. 现有模式（camera / umeyama / icp）
  2. 首帧对齐（first_frame）：首帧 pose 对齐 + 轨迹长度尺度对齐

三个接口：
  evaluate_global          — 单一对齐模式
  evaluate_global_both     — 同时运行现有模式 + 首帧对齐
  evaluate_global_firstframe_only — 仅首帧对齐

坐标系：DA3 extrinsics 为 w2c，取逆得 c2w。
"""

from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree


# ═══════════════ 从 point_eval.py 复制的核心函数 ═══════════════

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


def depth_to_pointcloud(
    depth: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray,
    conf: Optional[np.ndarray] = None, conf_thresh: float = 0.0,
    depth_min: float = 1e-3, depth_max: float = 1e6,
) -> np.ndarray:
    c2w_all = _to_4x4(extrinsics.astype(np.float64))
    N, H, W = depth.shape
    v_idx, u_idx = np.meshgrid(
        np.arange(H, dtype=np.float64), np.arange(W, dtype=np.float64), indexing="ij")

    all_pts = []
    for i in range(N):
        d = depth[i].astype(np.float64)
        K = intrinsics[i].astype(np.float64)
        valid = (d > depth_min) & (d < depth_max)
        if conf is not None:
            valid &= conf[i] > conf_thresh
        if not valid.any():
            continue
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        x_cam = (u_idx[valid] - cx) / fx * d[valid]
        y_cam = (v_idx[valid] - cy) / fy * d[valid]
        z_cam = d[valid]
        pts_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(z_cam)], axis=1)
        pts_world = (c2w_all[i] @ pts_cam.T).T[:, :3]
        all_pts.append(pts_world)

    return np.concatenate(all_pts, axis=0) if all_pts else np.zeros((0, 3), dtype=np.float64)


def umeyama_align(src_pts: np.ndarray, dst_pts: np.ndarray):
    N = src_pts.shape[0]
    mu_src = src_pts.mean(axis=0)
    mu_dst = dst_pts.mean(axis=0)
    src_c = src_pts - mu_src
    dst_c = dst_pts - mu_dst
    var_src = np.mean(np.sum(src_c ** 2, axis=1))
    cov = (dst_c.T @ src_c) / N
    U, sigma, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.sum(sigma * np.diag(S)) / var_src if var_src > 1e-12 else 1.0
    t = mu_dst - s * (R @ mu_src)
    return R, t, float(s)


def apply_similarity_transform(pts, R, t, s):
    return (s * (R @ pts.T)).T + t


def align_pointclouds_via_cameras(pred_c2w, gt_c2w, pred_pc):
    pred_c2w_4 = _to_4x4(pred_c2w.astype(np.float64))
    gt_c2w_4 = _to_4x4(gt_c2w.astype(np.float64))
    R, t, s = umeyama_align(pred_c2w_4[:, :3, 3], gt_c2w_4[:, :3, 3])
    return apply_similarity_transform(pred_pc, R, t, s)


def fps_sample(pts: np.ndarray, n_samples: int, seed: int = 0,
               pre_sample_factor: int = 10, device: str = "cpu") -> np.ndarray:
    N = pts.shape[0]
    if N <= n_samples:
        return pts
    rng = np.random.default_rng(seed)
    pre_limit = n_samples * pre_sample_factor
    if N > pre_limit:
        pts = pts[rng.choice(N, size=pre_limit, replace=False)]
        N = pre_limit
    try:
        import torch
        _dev = torch.device(device if torch.cuda.is_available() else "cpu")
        pts_t = torch.from_numpy(pts.astype(np.float32)).to(_dev)
        selected = torch.zeros(n_samples, dtype=torch.long, device=_dev)
        selected[0] = int(rng.integers(0, N))
        min_dists = torch.full((N,), float("inf"), dtype=torch.float32, device=_dev)
        d = ((pts_t - pts_t[selected[0]]) ** 2).sum(dim=1)
        torch.minimum(min_dists, d, out=min_dists)
        for i in range(1, n_samples):
            idx = int(min_dists.argmax())
            selected[i] = idx
            d = ((pts_t - pts_t[idx]) ** 2).sum(dim=1)
            torch.minimum(min_dists, d, out=min_dists)
        return pts_t[selected].cpu().numpy().astype(pts.dtype)
    except Exception:
        pass
    selected_np = np.zeros(n_samples, dtype=np.int64)
    selected_np[0] = rng.integers(0, N)
    pts_f = pts.astype(np.float32)
    min_dists = np.full(N, np.inf, dtype=np.float32)
    diff = pts_f - pts_f[selected_np[0]]
    min_dists = np.sum(diff ** 2, axis=1)
    for i in range(1, n_samples):
        idx = int(np.argmax(min_dists))
        selected_np[i] = idx
        diff = pts_f - pts_f[idx]
        np.minimum(min_dists, np.sum(diff ** 2, axis=1), out=min_dists)
    return pts[selected_np]


def compute_metrics(pred_pc: np.ndarray, gt_pc: np.ndarray,
                    fscore_threshold: float = 0.05) -> dict:
    if len(pred_pc) == 0 or len(gt_pc) == 0:
        nan = float("nan")
        return {"accuracy": nan, "completeness": nan, "chamfer_distance": nan,
                "precision": nan, "recall": nan, "fscore": nan}
    tree_gt = cKDTree(gt_pc)
    tree_pred = cKDTree(pred_pc)
    dist_p2g, _ = tree_gt.query(pred_pc, workers=-1)
    dist_g2p, _ = tree_pred.query(gt_pc, workers=-1)
    accuracy = float(np.mean(dist_p2g))
    completeness = float(np.mean(dist_g2p))
    cd = (accuracy + completeness) / 2.0
    precision = float(np.mean((dist_p2g < fscore_threshold).astype(float)))
    recall = float(np.mean((dist_g2p < fscore_threshold).astype(float)))
    fscore = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"accuracy": accuracy, "completeness": completeness, "chamfer_distance": cd,
            "precision": precision, "recall": recall, "fscore": fscore}


def icp_refine(pred_pc, gt_pc, max_iter=50, tol=1e-5, inlier_ratio=0.9):
    tree_gt = cKDTree(gt_pc)
    pred_cur = pred_pc.copy()
    n_iters = 0
    final_rmse = float("inf")
    for i in range(max_iter):
        dists, _ = tree_gt.query(pred_cur, workers=-1)
        thresh_idx = max(1, int(len(dists) * inlier_ratio))
        inlier_mask = np.argsort(dists)[:thresh_idx]
        src = pred_cur[inlier_mask]
        dst_idx = tree_gt.query(src, workers=-1)[1]
        dst = gt_pc[dst_idx]
        rmse = float(np.sqrt(np.mean(np.sum((src - dst) ** 2, axis=1))))
        mu_src, mu_dst = src.mean(0), dst.mean(0)
        A = (src - mu_src).T @ (dst - mu_dst)
        U, _, Vt = np.linalg.svd(A)
        Ss = np.eye(3)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            Ss[2, 2] = -1
        R_step = (U @ Ss @ Vt).T
        t_step = mu_dst - R_step @ mu_src
        pred_cur = (R_step @ pred_cur.T).T + t_step
        n_iters = i + 1
        final_rmse = rmse
        if np.linalg.norm(t_step) < tol and np.abs(np.trace(R_step) - 3) < tol:
            break
    return pred_cur, n_iters, final_rmse


# ═══════════════ 首帧对齐模式 ══════════════════════════════════

def _trajectory_length(positions: np.ndarray) -> float:
    diffs = np.diff(positions, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def first_frame_align(
    pred_c2w: np.ndarray, gt_c2w: np.ndarray,
    pred_depth: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    首帧对齐：
    1. 将 pred c2w[0] 对齐到 GT c2w[0]
    2. 尺度 s = GT 轨迹总长度 / pred 轨迹总长度
    3. 返回对齐后的 c2w 和缩放后的深度

    返回: (aligned_c2w, scaled_depth, scale)
    """
    pred_4 = _to_4x4(pred_c2w.astype(np.float64))
    gt_4 = _to_4x4(gt_c2w.astype(np.float64))

    transform = gt_4[0] @ np.linalg.inv(pred_4[0])
    aligned = np.array([transform @ pred_4[i] for i in range(len(pred_4))])

    gt_pos = gt_4[:, :3, 3]
    aligned_pos = aligned[:, :3, 3]

    gt_len = _trajectory_length(gt_pos)
    pred_len = _trajectory_length(aligned_pos)
    s = gt_len / (pred_len + 1e-8) if pred_len > 1e-8 else 1.0

    # 缩放平移分量
    origin = aligned[0, :3, 3].copy()
    for i in range(len(aligned)):
        aligned[i, :3, 3] = origin + s * (aligned[i, :3, 3] - origin)

    scaled_depth = pred_depth * s

    return aligned[:, :3, :], scaled_depth, s


# ═══════════════ 统一评估函数 ══════════════════════════════════

def _evaluate_core(
    pred_npz_path: str, gt_depth: np.ndarray,
    intrinsics: np.ndarray, c2w: np.ndarray,
    align: str, n_fps: int, conf_thresh: float,
    icp_max_iter: int, icp_inlier_ratio: float, device: str,
) -> dict:
    """内部统一评估入口。"""
    pred_data = dict(np.load(pred_npz_path, allow_pickle=True))
    pred_depth = pred_data["depth"].astype(np.float64)
    pred_conf = pred_data.get("conf", None)
    if pred_conf is not None and hasattr(pred_conf, "size") and pred_conf.size == 0:
        pred_conf = None

    N = min(pred_depth.shape[0], gt_depth.shape[0], intrinsics.shape[0], c2w.shape[0])
    pred_depth = pred_depth[:N]
    gt_depth_n = gt_depth[:N]
    K_n = intrinsics[:N]
    c2w_n = c2w[:N]
    if pred_conf is not None:
        pred_conf = pred_conf[:N]

    gt_valid = gt_depth_n[gt_depth_n > 1e-3]
    depth_cap = float(np.percentile(gt_valid, 98)) * 2.0 if len(gt_valid) > 0 else 1e6

    extra = {"align_mode": align, "n_fps": n_fps, "depth_cap": depth_cap}

    # DA3 extrinsics 为 w2c，取逆得 c2w
    pred_w2c = _to_4x4(pred_data["extrinsics"].astype(np.float64))[:N]
    pred_c2w = np.array([np.linalg.inv(pred_w2c[i]) for i in range(N)])
    pred_intr = pred_data["intrinsics"].astype(np.float64)[:N]

    if align == "first_frame":
        aligned_c2w, scaled_depth, s = first_frame_align(
            pred_c2w[:, :3, :], c2w_n, pred_depth)
        scaled_depth = np.clip(scaled_depth, 1e-3, depth_cap)
        pred_pc = depth_to_pointcloud(
            scaled_depth, pred_intr, aligned_c2w,
            conf=pred_conf, conf_thresh=conf_thresh, depth_max=depth_cap)
        gt_pc = depth_to_pointcloud(gt_depth_n, K_n, c2w_n, depth_max=depth_cap)
        extra["depth_scale"] = s

    elif align == "camera":
        _, _, s = umeyama_align(pred_c2w[:, :3, 3], _to_4x4(c2w_n)[:, :3, 3])
        pred_depth_scaled = np.clip(pred_depth * s, 1e-3, depth_cap)
        pred_pc = depth_to_pointcloud(
            pred_depth_scaled, pred_intr, c2w_n,
            conf=pred_conf, conf_thresh=conf_thresh, depth_max=depth_cap)
        gt_pc = depth_to_pointcloud(gt_depth_n, K_n, c2w_n, depth_max=depth_cap)
        extra["depth_scale"] = s

    else:  # umeyama / icp
        pred_pc_raw = depth_to_pointcloud(
            pred_depth, pred_intr, pred_c2w[:, :3, :],
            conf=pred_conf, conf_thresh=conf_thresh, depth_max=depth_cap)
        gt_pc = depth_to_pointcloud(gt_depth_n, K_n, c2w_n, depth_max=depth_cap)

        if len(pred_pc_raw) == 0 or len(gt_pc) == 0:
            nan = float("nan")
            return {**extra, "accuracy": nan, "completeness": nan, "chamfer_distance": nan}

        pred_pc = align_pointclouds_via_cameras(pred_c2w[:, :3, :], c2w_n, pred_pc_raw)

        if align == "icp":
            icp_limit = 50000
            src_icp = fps_sample(pred_pc, min(icp_limit, len(pred_pc)), device=device)
            dst_icp = fps_sample(gt_pc, min(icp_limit, len(gt_pc)), device=device)
            src_refined, n_iters, rmse = icp_refine(
                src_icp, dst_icp, max_iter=icp_max_iter, inlier_ratio=icp_inlier_ratio)
            mu_s, mu_d = src_icp.mean(0), src_refined.mean(0)
            A_icp = (src_icp - mu_s).T @ (src_refined - mu_d)
            U_i, _, Vt_i = np.linalg.svd(A_icp)
            Ss_i = np.eye(3)
            if np.linalg.det(U_i) * np.linalg.det(Vt_i) < 0:
                Ss_i[2, 2] = -1
            R_icp = (U_i @ Ss_i @ Vt_i).T
            t_icp = mu_d - R_icp @ mu_s
            pred_pc = (R_icp @ pred_pc.T).T + t_icp
            extra.update({"icp_iterations": n_iters, "icp_final_rmse": rmse})

    if len(pred_pc) == 0 or len(gt_pc) == 0:
        nan = float("nan")
        return {**extra, "accuracy": nan, "completeness": nan, "chamfer_distance": nan}

    pred_fps = fps_sample(pred_pc, n_fps, device=device)
    gt_fps = fps_sample(gt_pc, n_fps, device=device)
    metrics = compute_metrics(pred_fps, gt_fps)
    metrics.update({**extra, "n_pred_raw": len(pred_pc), "n_gt_raw": len(gt_pc)})
    return metrics


# ═══════════════ 四种对齐模式说明 ══════════════════════════════
#
#  camera      — Umeyama 对齐相机位置求 scale → 用 pred 内参 + GT 外参重投影
#                （假设坐标系方向一致，只修正尺度）
#  first_frame — 首帧 pose 对齐 + 轨迹长度求 scale → 用 pred 内参 + 对齐后外参重投影
#                （最接近 RL 训练 reward 的对齐方式）
#  umeyama     — 各用自己相机参数重投影 → Umeyama 相似变换对齐点云
#                （无坐标系假设，允许旋转+缩放）
#  icp         — umeyama 对齐后再做 ICP 精化（最精确，耗时最长）
#
#  组合快捷：
#  both_align  — 同时运行 camera + first_frame（默认推荐）
#  all_align   — 同时运行全部四种模式

# ═══════════════ 公开接口 ════════════════════════════════════════

ALL_ALIGN_MODES = ("camera", "first_frame", "umeyama", "icp")
VALID_ALIGN_CHOICES = ("camera", "first_frame", "umeyama", "icp",
                       "both_align", "all_align")


def evaluate_global(
    pred_npz: str, gt_depth: np.ndarray,
    K: np.ndarray, c2w: np.ndarray,
    align: str = "camera", n_fps: int = 20000,
    conf_thresh: float = 0.0, device: str = "cpu",
) -> dict:
    """单一对齐模式评估。align 须为四种基础模式之一。"""
    assert align in ALL_ALIGN_MODES, \
        f"align 须为 {ALL_ALIGN_MODES} 之一，组合模式请用 evaluate_global_multi"
    return _evaluate_core(
        pred_npz, gt_depth, K, c2w, align, n_fps, conf_thresh,
        icp_max_iter=50, icp_inlier_ratio=0.9, device=device)


def evaluate_global_multi(
    pred_npz: str, gt_depth: np.ndarray,
    K: np.ndarray, c2w: np.ndarray,
    aligns: tuple = ("camera", "first_frame"),
    n_fps: int = 20000, conf_thresh: float = 0.0, device: str = "cpu",
) -> dict:
    """同时运行多种对齐模式，返回 {align_name: result} 字典。"""
    return {
        mode: _evaluate_core(
            pred_npz, gt_depth, K, c2w, mode, n_fps, conf_thresh,
            icp_max_iter=50, icp_inlier_ratio=0.9, device=device)
        for mode in aligns
    }


def evaluate_global_both(
    pred_npz: str, gt_depth: np.ndarray,
    K: np.ndarray, c2w: np.ndarray,
    n_fps: int = 20000, conf_thresh: float = 0.0, device: str = "cpu",
) -> dict:
    """同时运行 camera + first_frame 两种模式。"""
    return evaluate_global_multi(
        pred_npz, gt_depth, K, c2w,
        aligns=("camera", "first_frame"),
        n_fps=n_fps, conf_thresh=conf_thresh, device=device)


def evaluate_global_all_align(
    pred_npz: str, gt_depth: np.ndarray,
    K: np.ndarray, c2w: np.ndarray,
    n_fps: int = 20000, conf_thresh: float = 0.0, device: str = "cpu",
) -> dict:
    """同时运行全部四种对齐模式：camera / first_frame / umeyama / icp。"""
    return evaluate_global_multi(
        pred_npz, gt_depth, K, c2w,
        aligns=ALL_ALIGN_MODES,
        n_fps=n_fps, conf_thresh=conf_thresh, device=device)


def evaluate_global_firstframe_only(
    pred_npz: str, gt_depth: np.ndarray,
    K: np.ndarray, c2w: np.ndarray,
    n_fps: int = 20000, conf_thresh: float = 0.0, device: str = "cpu",
) -> dict:
    """仅首帧对齐模式。"""
    return _evaluate_core(
        pred_npz, gt_depth, K, c2w, "first_frame", n_fps, conf_thresh,
        icp_max_iter=50, icp_inlier_ratio=0.9, device=device)
