"""dataset_rl.py — Wan2.2 RL 训练数据集适配器。

读取 unified_data_process 的输出格式（camera.txt + gt.mp4 / frames/），
返回 wan2.2 5B 训练所需字段：
    text, pixel_values [F,C,H,W], c2ws [F,4,4], Ks [F,3,3]

与 gen3r dataset_rl.py 的差异：
    1. 支持非方形分辨率 (resolution_h, resolution_w)。
    2. 不调用 gen3r.utils.data_utils.preprocess_poses（首帧对齐）；
       wan2.2 5B 在 build_camera_control 内通过官方 get_relative_pose 实现。
    3. caption 默认 fallback 为 "camera moving through a scene"，与 reward
       dispatcher 的默认 prompt 一致，便于 VideoAlign。

额外字段（供 reward 使用，原始路径不做任何坐标变换）：
    camera_txt_path, gt_video_path, sample_id, dataset_name
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, Any, List, Optional

import cv2
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset


# ══════════════════════════════════════════════════════════════════════════════
# 相机解析
# ══════════════════════════════════════════════════════════════════════════════

def _pad_to_4x4(mat34: np.ndarray) -> np.ndarray:
    """将 (3,4) 矩阵补全为 (4,4)。"""
    out = np.eye(4, dtype=mat34.dtype)
    out[:3, :] = mat34
    return out


def parse_camera_txt(camera_txt: str, H: int, W: int):
    """解析 unified_data_process 输出的 camera.txt。

    每行格式（19 个浮点数）：
        idx  fx/W  fy/H  cx/W  cy/H  d1  d2  w2c[0,0]...w2c[2,3]

    Returns:
        c2ws : Tensor [N, 4, 4] OpenCV c2w（**未做首帧对齐**，由 wan22_encode 处理）
        Ks   : Tensor [N, 3, 3] 像素坐标内参
    """
    entries = []
    with open(camera_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = list(map(float, line.split()))
            if len(vals) < 19:
                # 兼容含 header 的版本（如 RealEstate10K 原始格式），跳过
                continue
            entries.append(vals)

    entries.sort(key=lambda v: v[0])

    c2ws_np = []
    Ks_np = []
    for vals in entries:
        fx_n, fy_n, cx_n, cy_n = vals[1], vals[2], vals[3], vals[4]
        K = np.array([
            [fx_n * W, 0.0,      cx_n * W],
            [0.0,      fy_n * H, cy_n * H],
            [0.0,      0.0,      1.0     ],
        ], dtype=np.float64)
        w2c_34 = np.array(vals[7:19], dtype=np.float64).reshape(3, 4)
        w2c_44 = _pad_to_4x4(w2c_34)
        c2w = np.linalg.inv(w2c_44)
        c2ws_np.append(c2w)
        Ks_np.append(K)

    c2ws_t = torch.from_numpy(np.stack(c2ws_np, axis=0)).float()  # [N,4,4]
    Ks_t = torch.from_numpy(np.stack(Ks_np, axis=0)).float()       # [N,3,3]
    return c2ws_t, Ks_t


# ══════════════════════════════════════════════════════════════════════════════
# 帧读取（aspect-preserving resize + center crop）
# ══════════════════════════════════════════════════════════════════════════════

def _resize_center_crop(frame_np: np.ndarray, H: int, W: int) -> np.ndarray:
    """aspect-preserving resize + center crop to (H, W)，输入 HxWx3 uint8。"""
    h, w = frame_np.shape[:2]
    ratio = max(H / h, W / w)
    new_h = max(int(round(h * ratio)), H)
    new_w = max(int(round(w * ratio)), W)
    import PIL.Image as PILImage
    pil = PILImage.fromarray(frame_np).resize((new_w, new_h), PILImage.LANCZOS)
    arr = np.array(pil)
    top = (arr.shape[0] - H) // 2
    left = (arr.shape[1] - W) // 2
    return arr[top: top + H, left: left + W]


def load_frames_from_video(video_path: str, indices: List[int], H: int, W: int) -> torch.Tensor:
    """从 mp4 按帧索引读取帧，返回 [F, 3, H, W] float32 [0,1]。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    frame_idx = 0
    target_set = set(indices)
    idx_map: Dict[int, np.ndarray] = {}

    while len(idx_map) < len(target_set):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in target_set:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            idx_map[frame_idx] = frame_rgb
        frame_idx += 1
    cap.release()

    for i in indices:
        if i not in idx_map:
            last_available = max(idx_map.keys()) if idx_map else 0
            frame_np = idx_map.get(last_available, np.zeros((H, W, 3), dtype=np.uint8))
        else:
            frame_np = idx_map[i]
        cropped = _resize_center_crop(frame_np, H, W)
        frames.append(cropped)

    arr = np.stack(frames, axis=0)
    tensor = torch.from_numpy(arr).float().permute(0, 3, 1, 2) / 255.0
    return tensor


def load_frames_from_dir(frames_dir: str, indices: List[int], H: int, W: int) -> torch.Tensor:
    """从预提取的帧目录按索引读取，返回 [F, 3, H, W] float32 [0,1]。"""
    fdir = Path(frames_dir)
    all_files = sorted([
        f for f in fdir.iterdir()
        if f.suffix.lower() in (".png", ".jpg", ".jpeg")
    ])
    frames = []
    for i in indices:
        if i < len(all_files):
            frame_np = imageio.imread(str(all_files[i]))
            if frame_np.ndim == 2:
                frame_np = np.stack([frame_np] * 3, axis=-1)
            if frame_np.shape[-1] == 4:
                frame_np = frame_np[..., :3]
        else:
            frame_np = np.zeros((H, W, 3), dtype=np.uint8)
        cropped = _resize_center_crop(frame_np, H, W)
        frames.append(cropped)

    arr = np.stack(frames, axis=0)
    tensor = torch.from_numpy(arr).float().permute(0, 3, 1, 2) / 255.0
    return tensor


# ══════════════════════════════════════════════════════════════════════════════
# 帧索引采样
# ══════════════════════════════════════════════════════════════════════════════

def sample_frame_indices(total: int, num_frames: int, stride: int) -> List[int]:
    """从 total 帧中按 stride 采样 num_frames 帧，随机起始点。"""
    need = num_frames * stride
    if total < need:
        stride = max(1, total // num_frames)
        need = num_frames * stride

    if total < num_frames:
        indices = list(range(total)) + [total - 1] * (num_frames - total)
        return indices[:num_frames]

    max_start = total - (num_frames - 1) * stride - 1
    max_start = max(0, max_start)
    start = random.randint(0, max_start)
    return [start + i * stride for i in range(num_frames)]


def get_video_total_frames(video_path: str) -> int:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total


# ══════════════════════════════════════════════════════════════════════════════
# 数据集
# ══════════════════════════════════════════════════════════════════════════════

class RLDataset(Dataset):
    """Wan2.2 RL 训练数据集。

    目录结构（与 gen3r 完全一致）：
        data_root/
            <dataset_name>/
                <sample_id>/
                    camera.txt
                    gt.mp4           (frame_mode="video")
                    frames/          (frame_mode="frames")
                    metadata.json
                    start.png

    Args:
        data_root      : 数据根目录
        datasets       : 数据集名列表
        num_frames     : 每样本采样帧数（wan2.2 默认 49）
        stride         : 帧间隔（默认 1）
        resolution_h   : 目标高度（默认 704）
        resolution_w   : 目标宽度（默认 1280）
        frame_mode     : "video" or "frames"
        dataset_weights: 各数据集采样权重
    """

    def __init__(
        self,
        data_root: str,
        datasets: List[str],
        num_frames: int = 49,
        stride: int = 1,
        resolution_h: int = 704,
        resolution_w: int = 1280,
        frame_mode: str = "video",
        dataset_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.stride = stride
        self.resolution_h = resolution_h
        self.resolution_w = resolution_w
        self.frame_mode = frame_mode

        assert frame_mode in ("video", "frames"), f"Unknown frame_mode: {frame_mode}"

        self.samples: List[Dict] = []
        self.dataset_names: List[str] = []
        per_dataset: Dict[str, List[Dict]] = {}

        for ds_name in datasets:
            ds_dir = Path(data_root) / ds_name
            if not ds_dir.exists():
                print(f"[RLDataset] WARNING: {ds_dir} not found, skipping")
                continue
            ds_samples = []
            for sample_dir in sorted(ds_dir.iterdir()):
                if not sample_dir.is_dir():
                    continue
                cam_txt = sample_dir / "camera.txt"
                meta_json = sample_dir / "metadata.json"
                if not cam_txt.exists() or not meta_json.exists():
                    continue
                if frame_mode == "video":
                    media = sample_dir / "gt.mp4"
                else:
                    media = sample_dir / "frames"
                if not media.exists():
                    continue
                ds_samples.append({
                    "sample_id": sample_dir.name,
                    "dataset_name": ds_name,
                    "camera_txt": str(cam_txt),
                    "metadata_json": str(meta_json),
                    "media": str(media),
                })
            if ds_samples:
                per_dataset[ds_name] = ds_samples
                self.dataset_names.append(ds_name)
                print(f"[RLDataset] {ds_name}: {len(ds_samples)} samples")

        if not per_dataset:
            raise ValueError(f"No valid samples found in {data_root} for datasets {datasets}")

        if dataset_weights is None:
            dataset_weights = [1.0] * len(self.dataset_names)
        assert len(dataset_weights) == len(self.dataset_names)
        total_w = sum(dataset_weights)
        self.dataset_probs = [w / total_w for w in dataset_weights]
        self.per_dataset = per_dataset

        for ds_samples in per_dataset.values():
            self.samples.extend(ds_samples)

        print(f"[RLDataset] Total: {len(self.samples)} samples across {len(self.dataset_names)} datasets")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        ds_name = np.random.choice(self.dataset_names, p=self.dataset_probs)
        ds_samples = self.per_dataset[ds_name]
        info = ds_samples[np.random.randint(0, len(ds_samples))]
        return self._load_sample(info)

    def _load_sample(self, info: Dict) -> Dict[str, Any]:
        camera_txt = info["camera_txt"]
        media = info["media"]
        H, W = self.resolution_h, self.resolution_w

        # ── 解析相机 ─────────────────────────────────────────────────────────
        c2ws, Ks = parse_camera_txt(camera_txt, H, W)
        total_cam = c2ws.shape[0]

        # ── 帧索引采样 ────────────────────────────────────────────────────────
        if self.frame_mode == "video":
            total_frames = get_video_total_frames(media)
        else:
            fdir = Path(media)
            total_frames = len([
                f for f in fdir.iterdir()
                if f.suffix.lower() in (".png", ".jpg", ".jpeg")
            ])

        total_avail = min(total_frames, total_cam)
        indices = sample_frame_indices(total_avail, self.num_frames, self.stride)

        # ── 读取帧 ────────────────────────────────────────────────────────────
        if self.frame_mode == "video":
            pixel_values = load_frames_from_video(media, indices, H, W)
        else:
            pixel_values = load_frames_from_dir(media, indices, H, W)

        c2ws_sampled = c2ws[indices]
        Ks_sampled = Ks[indices]

        # ── 读取 caption ──────────────────────────────────────────────────────
        with open(info["metadata_json"], "r") as f:
            meta = json.load(f)
        caption = meta.get("caption", meta.get("prompt", "")) or "camera moving through a scene"

        return {
            "text": caption,
            "pixel_values": pixel_values,
            "c2ws": c2ws_sampled,
            "Ks": Ks_sampled,
            "sample_id": info["sample_id"],
            "dataset_name": info["dataset_name"],
            "camera_txt_path": camera_txt,
            "gt_video_path": media if self.frame_mode == "video" else str(Path(media).parent / "gt.mp4"),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Collate
# ══════════════════════════════════════════════════════════════════════════════

def collate_fn(batch: List[Dict]) -> Dict:
    """DataLoader collate：保留 list 格式（帧数可变），不做 stack。"""
    return {
        "text":             [s["text"] for s in batch],
        "pixel_values":     [s["pixel_values"] for s in batch],
        "c2ws":             [s["c2ws"] for s in batch],
        "Ks":               [s["Ks"] for s in batch],
        "sample_id":        [s["sample_id"] for s in batch],
        "dataset_name":     [s["dataset_name"] for s in batch],
        "camera_txt_path":  [s["camera_txt_path"] for s in batch],
        "gt_video_path":    [s["gt_video_path"] for s in batch],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 工厂函数
# ══════════════════════════════════════════════════════════════════════════════

def build_rl_dataset(args) -> RLDataset:
    """从 argparse.Namespace 构建 RLDataset。"""
    dataset_list = [d.strip() for d in args.datasets.split(",") if d.strip()]
    return RLDataset(
        data_root=args.data_root,
        datasets=dataset_list,
        num_frames=args.num_frames,
        stride=args.frame_stride,
        resolution_h=args.resolution_h,
        resolution_w=args.resolution_w,
        frame_mode=args.frame_mode,
    )
