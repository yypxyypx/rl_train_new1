"""dataset_rl.py — RL 训练数据集适配器。

读取 unified_data_process 的输出格式（camera.txt + gt.mp4 / frames/），
做坐标系转换后返回与 Gen3R 原始训练相同的字段格式：
    text, pixel_values [F,C,H,W], c2ws [F,4,4], Ks [F,3,3]

坐标系转换链：
    camera.txt (OpenCV w2c, 归一化内参)
        → invert → OpenCV c2w, 像素内参
        → preprocess_poses → 首帧 identity

额外字段（供 reward 使用，原始路径不做任何坐标变换）：
    camera_txt_path, gt_video_path, sample_id, dataset_name
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import cv2
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize, center_crop as tv_center_crop, InterpolationMode

# ─── 将 Gen3R 包加入路径 ───────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_GEN3R_PKG = _HERE / "Gen3R"
if str(_GEN3R_PKG) not in sys.path:
    sys.path.insert(0, str(_GEN3R_PKG))

from gen3r.utils.data_utils import preprocess_poses  # noqa: E402


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
        c2ws : Tensor [N, 4, 4] OpenCV c2w（已首帧对齐）
        Ks   : Tensor [N, 3, 3] 像素坐标内参
    """
    entries = []
    with open(camera_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = list(map(float, line.split()))
            entries.append(vals)

    entries.sort(key=lambda v: v[0])

    c2ws_np = []
    Ks_np = []
    for vals in entries:
        fx_n, fy_n, cx_n, cy_n = vals[1], vals[2], vals[3], vals[4]
        # 归一化 -> 像素
        K = np.array([
            [fx_n * W, 0.0,      cx_n * W],
            [0.0,      fy_n * H, cy_n * H],
            [0.0,      0.0,      1.0     ],
        ], dtype=np.float64)
        # w2c (3,4) -> (4,4) -> c2w
        w2c_34 = np.array(vals[7:19], dtype=np.float64).reshape(3, 4)
        w2c_44 = _pad_to_4x4(w2c_34)
        c2w = np.linalg.inv(w2c_44)
        c2ws_np.append(c2w)
        Ks_np.append(K)

    c2ws_t = torch.from_numpy(np.stack(c2ws_np, axis=0)).float()  # [N,4,4]
    Ks_t = torch.from_numpy(np.stack(Ks_np, axis=0)).float()       # [N,3,3]
    # 首帧对齐到 identity（与 Gen3R 原始训练一致）
    c2ws_t = preprocess_poses(c2ws_t)
    return c2ws_t, Ks_t


# ══════════════════════════════════════════════════════════════════════════════
# 帧读取
# ══════════════════════════════════════════════════════════════════════════════

def _resize_center_crop(frame_np: np.ndarray, H: int, W: int) -> np.ndarray:
    """aspect-preserving resize + center crop to (H, W), input HxWx3 uint8."""
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
            # 边界情况：索引超出视频长度，用最后一帧填充
            last_available = max(idx_map.keys()) if idx_map else 0
            frame_np = idx_map.get(last_available, np.zeros((H, W, 3), dtype=np.uint8))
        else:
            frame_np = idx_map[i]
        cropped = _resize_center_crop(frame_np, H, W)
        frames.append(cropped)

    arr = np.stack(frames, axis=0)  # [F, H, W, 3]
    tensor = torch.from_numpy(arr).float().permute(0, 3, 1, 2) / 255.0  # [F,3,H,W]
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
    """从 total 帧中按 stride 采样 num_frames 帧，随机起始点。

    若帧数不足，自动降低 stride 并补充重复帧。
    """
    need = num_frames * stride
    if total < need:
        stride = max(1, total // num_frames)
        need = num_frames * stride

    if total < num_frames:
        # 极端情况：直接重复
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
    """RL 训练数据集，读取 unified_data_process 输出格式。

    目录结构：
        data_root/
            <dataset_name>/
                <sample_id>/
                    camera.txt
                    gt.mp4           (frame_mode="video")
                    frames/          (frame_mode="frames", 预提取)
                    metadata.json
                    start.png

    Args:
        data_root    : 数据根目录
        datasets     : 数据集名列表，如 ["re10k", "dl3dv"]
        num_frames   : 每样本采样帧数
        stride       : 帧间隔
        resolution   : 图像分辨率（H=W，Gen3R 为 560）
        frame_mode   : "video"（从 gt.mp4 提取）或 "frames"（预提取目录）
        dataset_weights : 各数据集采样权重（None=均等）
    """

    def __init__(
        self,
        data_root: str,
        datasets: List[str],
        num_frames: int = 17,
        stride: int = 2,
        resolution: int = 560,
        frame_mode: str = "video",
        dataset_weights: Optional[List[float]] = None,
        train_manifest: Optional[str] = None,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.stride = stride
        self.resolution = resolution
        self.frame_mode = frame_mode

        assert frame_mode in ("video", "frames"), f"Unknown frame_mode: {frame_mode}"

        # 收集所有样本
        self.samples: List[Dict] = []
        self.dataset_names: List[str] = []
        per_dataset: Dict[str, List[Dict]] = {}

        # ── 模式 A：manifest 驱动（推荐，与 t5 预计算/数据归档保持一致）───────
        if train_manifest:
            manifest_path = Path(train_manifest)
            if not manifest_path.is_file():
                raise FileNotFoundError(f"train_manifest not found: {manifest_path}")
            allow = set(datasets) if datasets else None
            print(f"[RLDataset] loading manifest: {manifest_path}"
                  f"  (filter datasets={sorted(allow) if allow else 'ALL'})")
            with open(manifest_path, "r") as f:
                for line_idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"[RLDataset] skip malformed line {line_idx}: {e}")
                        continue
                    ds_name = rec.get("dataset")
                    sample_dir = rec.get("sample_dir")
                    if not ds_name or not sample_dir:
                        continue
                    if allow is not None and ds_name not in allow:
                        continue
                    sd = Path(sample_dir)
                    cam_txt = sd / "camera.txt"
                    meta_json = Path(rec.get("metadata_json") or (sd / "metadata.json"))
                    if not cam_txt.exists() or not meta_json.exists():
                        continue
                    if frame_mode == "video":
                        media = sd / "gt.mp4"
                    else:
                        media = sd / "frames"
                    if not media.exists():
                        continue
                    per_dataset.setdefault(ds_name, []).append({
                        "sample_id": rec.get("sample_id", sd.name),
                        "dataset_name": ds_name,
                        "sample_dir": str(sd),
                        "camera_txt": str(cam_txt),
                        "metadata_json": str(meta_json),
                        "media": str(media),
                    })
            for ds_name, ds_samples in per_dataset.items():
                self.dataset_names.append(ds_name)
                print(f"[RLDataset] {ds_name}: {len(ds_samples)} samples (manifest)")
            # 保持 datasets 入参的顺序（影响采样权重映射）
            if datasets:
                self.dataset_names = [d for d in datasets if d in per_dataset]

        # ── 模式 B：目录扫描（旧逻辑，递归到 train/ 子目录下找 metadata.json）─
        else:
            for ds_name in datasets:
                ds_dir = Path(data_root) / ds_name
                if not ds_dir.exists():
                    print(f"[RLDataset] WARNING: {ds_dir} not found, skipping")
                    continue
                # 递归找 metadata.json，匹配 unified_data_process 输出的多层目录
                ds_samples = []
                for meta_json in sorted(ds_dir.rglob("metadata.json")):
                    sample_dir = meta_json.parent
                    cam_txt = sample_dir / "camera.txt"
                    if not cam_txt.exists():
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
                        "sample_dir": str(sample_dir),
                        "camera_txt": str(cam_txt),
                        "metadata_json": str(meta_json),
                        "media": str(media),
                    })
                per_dataset[ds_name] = ds_samples
                self.dataset_names.append(ds_name)
                print(f"[RLDataset] {ds_name}: {len(ds_samples)} samples (rglob)")

        if not per_dataset or not any(per_dataset.values()):
            src = f"manifest={train_manifest}" if train_manifest else f"data_root={data_root}"
            raise ValueError(
                f"No valid samples found ({src}, datasets={datasets}). "
                f"Each sample dir must contain camera.txt + metadata.json + "
                f"({'gt.mp4' if frame_mode == 'video' else 'frames/'})."
            )

        # 计算采样权重
        if dataset_weights is None:
            dataset_weights = [1.0] * len(self.dataset_names)
        assert len(dataset_weights) == len(self.dataset_names)
        total_w = sum(dataset_weights)
        self.dataset_probs = [w / total_w for w in dataset_weights]
        self.per_dataset = per_dataset

        # 合并所有样本（用于 __len__）
        for ds_samples in per_dataset.values():
            self.samples.extend(ds_samples)

        print(f"[RLDataset] Total: {len(self.samples)} samples across {len(self.dataset_names)} datasets")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # 确定性 index 查表：DistributedSampler 给出的 index 必须真实生效，
        # 否则 (1) 同 sub-group 的两张卡拿到的不是同一条 prompt
        #      (2) 中断后无法靠 step 计数器复现样本顺序做 resume
        # 跨 dataset 的均衡由 DistributedSampler 的 shuffle 在全集 3200 条上保证
        # （dl3dv 1600 + scannet++ 1600，本身就是 50/50 平衡）。
        info = self.samples[index % len(self.samples)]
        return self._load_sample(info)

    def _load_sample(self, info: Dict) -> Dict[str, Any]:
        camera_txt = info["camera_txt"]
        media = info["media"]
        H = W = self.resolution

        # ── 解析相机 ─────────────────────────────────────────────────────────
        c2ws, Ks = parse_camera_txt(camera_txt, H, W)  # [N,4,4], [N,3,3]
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

        # 同步 camera
        c2ws_sampled = c2ws[indices]  # [F,4,4]
        Ks_sampled = Ks[indices]       # [F,3,3]

        # ── 读取 caption ──────────────────────────────────────────────────────
        with open(info["metadata_json"], "r") as f:
            meta = json.load(f)
        caption = meta.get("caption", meta.get("prompt", ""))

        return {
            "text": caption,
            "pixel_values": pixel_values,    # [F,3,H,W] float32 [0,1]
            "c2ws": c2ws_sampled,             # [F,4,4] OpenCV c2w, 首帧 identity
            "Ks": Ks_sampled,                 # [F,3,3] 像素内参
            "sample_id": info["sample_id"],
            "dataset_name": info["dataset_name"],
            "sample_dir": info.get("sample_dir", ""),  # 用于 in-place T5 embed 查找
            "camera_txt_path": camera_txt,    # 原始 camera.txt（reward 直接使用）
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
        "sample_dir":       [s.get("sample_dir", "") for s in batch],
        "camera_txt_path":  [s["camera_txt_path"] for s in batch],
        "gt_video_path":    [s["gt_video_path"] for s in batch],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 工厂函数
# ══════════════════════════════════════════════════════════════════════════════

def build_rl_dataset(args) -> RLDataset:
    """从 argparse.Namespace 构建 RLDataset。

    期望 args 字段：
        data_root, datasets (逗号分隔字符串),
        num_frames, frame_stride, resolution, frame_mode
        train_manifest (可选；为空则走目录扫描)
    """
    dataset_list = [d.strip() for d in args.datasets.split(",") if d.strip()]
    return RLDataset(
        data_root=args.data_root,
        datasets=dataset_list,
        num_frames=args.num_frames,
        stride=args.frame_stride,
        resolution=args.resolution,
        frame_mode=args.frame_mode,
        train_manifest=getattr(args, "train_manifest", None) or None,
    )
