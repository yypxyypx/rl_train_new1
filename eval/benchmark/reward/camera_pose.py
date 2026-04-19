#!/usr/bin/env python3
"""
camera_pose.py — 相机轨迹评估：旋转 AUC + 平移指标。

旋转指标：全帧对 AUC（借鉴 DA3 bench/utils.py，独立重写）。
平移指标：首帧对齐 + 尺度对齐 + 逐帧绝对距离 + AUC。

坐标系约定：
  DA3 extrinsics 为 w2c + OpenCV，需取逆得 c2w。
  camera.txt 通过 parse_camera_txt 返回 c2w。
  两者同为 OpenCV 约定，无需转换。
"""

from typing import Dict, Tuple

import numpy as np


# ═══════════════════ 工具函数 ═══════════════════════════════════


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


def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    """(N, 3, 3) rotation matrices -> (N, 4) quaternions [w, x, y, z]."""
    N = R.shape[0]
    q = np.zeros((N, 4), dtype=np.float64)
    for i in range(N):
        m = R[i]
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if tr > 0:
            s = 2.0 * np.sqrt(tr + 1.0)
            q[i] = [0.25 * s, (m[2, 1] - m[1, 2]) / s,
                     (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            q[i] = [(m[2, 1] - m[1, 2]) / s, 0.25 * s,
                     (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            q[i] = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
                     0.25 * s, (m[1, 2] + m[2, 1]) / s]
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            q[i] = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
                     (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    return q


# ═══════════════════ 首帧对齐 ══════════════════════════════════


def align_to_first_camera(poses: np.ndarray) -> np.ndarray:
    """
    将所有 c2w pose 归一化到首帧坐标系。
    poses: (N, 4, 4) c2w
    aligned[i] = inv(c2w[0]) @ c2w[i]，使得 aligned[0] = I。
    """
    first_inv = np.linalg.inv(poses[0])
    return np.array([first_inv @ pose for pose in poses])


# ═══════════════════ 全帧对误差 ═════════════════════════════════


def _build_pair_indices(N: int):
    """构建所有 N*(N-1)/2 帧对的索引。"""
    i1, i2 = [], []
    for a in range(N):
        for b in range(a + 1, N):
            i1.append(a)
            i2.append(b)
    return np.array(i1), np.array(i2)


def _rotation_angle(R_gt: np.ndarray, R_pred: np.ndarray) -> np.ndarray:
    """
    计算旋转角度误差（度）— 测地线距离。
    R_gt, R_pred: (M, 3, 3)
    返回: (M,) 度，范围 [0, 180]
    """
    q_gt = _mat_to_quat(R_gt)
    q_pred = _mat_to_quat(R_pred)
    dot = np.abs(np.sum(q_gt * q_pred, axis=1))
    dot = np.clip(dot, 0.0, 1.0)
    err = 2.0 * np.arccos(dot)
    return np.degrees(err)


def _translation_angle(t_gt: np.ndarray, t_pred: np.ndarray) -> np.ndarray:
    """
    计算平移方向角度误差（度）。
    t_gt, t_pred: (M, 3)
    返回: (M,) 度
    """
    eps = 1e-15
    t_gt_n = t_gt / (np.linalg.norm(t_gt, axis=1, keepdims=True) + eps)
    t_pred_n = t_pred / (np.linalg.norm(t_pred, axis=1, keepdims=True) + eps)
    dot2 = np.sum(t_gt_n * t_pred_n, axis=1) ** 2
    loss = np.clip(1.0 - dot2, eps, 1.0)
    err_rad = np.arccos(np.clip(np.sqrt(1 - loss), -1.0, 1.0))
    err_deg = np.degrees(err_rad)
    err_deg = np.minimum(err_deg, np.abs(180 - err_deg))
    return err_deg


def compute_all_pairs_errors(
    pred_c2w: np.ndarray, gt_c2w: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算所有帧对的旋转和平移角度误差。
    pred_c2w, gt_c2w: (N, 4, 4) c2w

    返回: (rot_errors_deg, trans_errors_deg)  各 (M,)
    """
    N = len(pred_c2w)
    i1, i2 = _build_pair_indices(N)

    gt_rel = np.linalg.inv(gt_c2w[i1]) @ gt_c2w[i2]
    pred_rel = np.linalg.inv(pred_c2w[i1]) @ pred_c2w[i2]

    rot_err = _rotation_angle(gt_rel[:, :3, :3], pred_rel[:, :3, :3])
    trans_err = _translation_angle(gt_rel[:, :3, 3], pred_rel[:, :3, 3])

    return rot_err, trans_err


# ═══════════════════ AUC 计算 ══════════════════════════════════


def _auc_from_errors(
    errors: np.ndarray, max_threshold: int = 30,
) -> float:
    """单一误差序列的 AUC。"""
    bins = np.arange(max_threshold + 1)
    histogram, _ = np.histogram(errors, bins=bins)
    num_pairs = float(len(errors))
    normalized = histogram.astype(float) / num_pairs
    return float(np.mean(np.cumsum(normalized)))


def calculate_auc(
    r_error: np.ndarray, t_error: np.ndarray, max_threshold: int = 30,
) -> Tuple[float, float, float]:
    """
    分别计算旋转 AUC、平移 AUC 和组合 AUC（max(rot, trans)）。

    返回: (rot_auc, trans_auc, pose_auc)
    """
    rot_auc = _auc_from_errors(r_error, max_threshold)
    trans_auc = _auc_from_errors(t_error, max_threshold)
    max_errors = np.max(np.column_stack([r_error, t_error]), axis=1)
    pose_auc = _auc_from_errors(max_errors, max_threshold)
    return rot_auc, trans_auc, pose_auc


# ═══════════════════ 平移距离指标 ═══════════════════════════════


def _trajectory_length(positions: np.ndarray) -> float:
    """计算轨迹总长度 = sum(||p[i+1] - p[i]||)。"""
    diffs = np.diff(positions, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def compute_translation_metric(
    pred_c2w: np.ndarray, gt_c2w: np.ndarray,
) -> Tuple[float, Dict]:
    """
    平移指标：首帧 pose 对齐 + 尺度对齐 + 逐帧距离（按 GT 轨迹长度归一化）。

    与 RL 训练 reward (compute_reward_camera_traj) 对齐：
    1. 将 pred 首帧对齐到 GT 首帧
    2. 尺度 s = GT 轨迹总长度 / pred 轨迹总长度
    3. 逐帧距离 dist_i = ||s * aligned_pred_pos[i] - gt_pos[i]||
    4. 按 GT 轨迹长度归一化 dist_norm_i = dist_i / gt_traj_len
       —— 消除不同数据集 / 样本绝对尺度差异的影响
    5. 最终指标 = mean(-exp(dist_norm_i / 0.3))
       —— 先 exp 再 mean，与训练 reward 完全一致

    pred_c2w, gt_c2w: (N, 4, 4) c2w
    """
    N = min(len(pred_c2w), len(gt_c2w))
    pred_c2w = pred_c2w[:N]
    gt_c2w = gt_c2w[:N]

    # 首帧对齐：将 pred 全部变换到 GT 首帧坐标系
    # aligned_pred[i] = gt_c2w[0] @ inv(pred_c2w[0]) @ pred_c2w[i]
    transform = gt_c2w[0] @ np.linalg.inv(pred_c2w[0])
    aligned_pred = np.array([transform @ pred_c2w[i] for i in range(N)])

    gt_pos = gt_c2w[:, :3, 3]
    aligned_pos = aligned_pred[:, :3, 3]

    gt_traj_len = _trajectory_length(gt_pos)
    pred_traj_len = _trajectory_length(aligned_pos)

    s = gt_traj_len / (pred_traj_len + 1e-8) if pred_traj_len > 1e-8 else 1.0

    # 尺度对齐：只缩放平移分量（相对于首帧）
    origin = aligned_pos[0].copy()
    scaled_pos = origin + s * (aligned_pos - origin)

    per_frame_dist = np.linalg.norm(scaled_pos - gt_pos, axis=1)
    # 按 GT 轨迹长度归一化，消除数据集间绝对尺度差异
    per_frame_dist_norm = per_frame_dist / (gt_traj_len + 1e-8)
    # 与训练 reward 完全一致：先逐帧 exp 再求均值
    per_frame_rewards = -np.exp(per_frame_dist_norm / 0.3)
    metric = float(np.mean(per_frame_rewards))

    details = {
        "scale": s,
        "gt_trajectory_length": gt_traj_len,
        "pred_trajectory_length": pred_traj_len,
        "mean_distance": float(np.mean(per_frame_dist)),
        "median_distance": float(np.median(per_frame_dist)),
        "mean_distance_norm": float(np.mean(per_frame_dist_norm)),
        "metric": metric,
        "per_frame_distance": per_frame_dist.tolist(),
        "per_frame_distance_norm": per_frame_dist_norm.tolist(),
        "per_frame_rewards": per_frame_rewards.tolist(),
    }
    return metric, details


# ═══════════════════ DA3 数据加载 ═══════════════════════════════


def da3_to_c2w(da3_data: dict) -> np.ndarray:
    """
    从 DA3 数据中提取 c2w pose。
    DA3 extrinsics 为 w2c (N, 3, 4)，需取逆得 c2w。
    """
    w2c = _to_4x4(da3_data["extrinsics"].astype(np.float64))
    c2w = np.array([np.linalg.inv(w2c[i]) for i in range(len(w2c))])
    return c2w


# ═══════════════════ 三个接口 ══════════════════════════════════


def evaluate_camera_pose(da3_data: dict, gt_c2w: np.ndarray) -> Dict:
    """
    组合评估：旋转 AUC + 平移方向 AUC + 组合 Pose AUC + 平移距离指标。

    返回: {rotation_auc30, translation_auc30, pose_auc30, ...,
           translation_metric, mean_rotation_error_deg, ..., details}
    """
    pred_c2w = da3_to_c2w(da3_data)
    gt_c2w_4 = _to_4x4(gt_c2w.astype(np.float64))

    N = min(len(pred_c2w), len(gt_c2w_4))

    # 全帧对误差（对齐对 compute_all_pairs_errors 是 no-op，保留以防后续用途）
    rot_err, trans_err = compute_all_pairs_errors(pred_c2w[:N], gt_c2w_4[:N])

    result = {}
    for thresh in [30, 15, 5, 3]:
        r_auc, t_auc, p_auc = calculate_auc(rot_err, trans_err, max_threshold=thresh)
        result[f"rotation_auc{thresh:02d}"] = r_auc
        result[f"translation_auc{thresh:02d}"] = t_auc
        result[f"pose_auc{thresh:02d}"] = p_auc

    result["mean_rotation_error_deg"] = float(np.mean(rot_err))
    result["mean_translation_angle_error_deg"] = float(np.mean(trans_err))

    # 平移距离指标
    metric, dist_details = compute_translation_metric(pred_c2w[:N], gt_c2w_4[:N])
    result["translation_metric"] = metric
    result["details"] = dist_details

    return result
