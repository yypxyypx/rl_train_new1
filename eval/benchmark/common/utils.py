#!/usr/bin/env python3
"""
utils.py — IO 工具函数、conda 环境运行器、抽帧等。
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def log(msg: str):
    print(f"[benchmark] {msg}", flush=True)


# ─────────────────────── JSON IO ──────────────────────────────────

def save_json(path: str, data: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def load_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


# ─────────────────────── 抽帧 ─────────────────────────────────────

def extract_frames(video_path: str, out_dir: str, max_frames: int = 0) -> list:
    """抽帧为 PNG，返回已排序的帧路径列表。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    existing = sorted(out.glob("frame_*.png"))
    if existing:
        return [str(p) for p in existing]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    paths, idx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        p = out / f"frame_{idx:05d}.png"
        cv2.imwrite(str(p), frame)
        paths.append(str(p))
        idx += 1
        if max_frames > 0 and idx >= max_frames:
            break
    cap.release()
    return paths


# ─────────────────────── conda 环境管理 ──────────────────────────

def find_conda() -> str:
    for candidate in [
        "/opt/conda/bin/conda",
        os.environ.get("CONDA_EXE", ""),
        shutil.which("conda") or "",
    ]:
        if candidate and os.path.isfile(candidate):
            return candidate
    return "conda"


def env_python(env: str) -> str:
    candidates = [
        f"/opt/conda/envs/{env}/bin/python",
        os.path.expanduser(f"~/miniconda3/envs/{env}/bin/python"),
        os.path.expanduser(f"~/anaconda3/envs/{env}/bin/python"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return f"/opt/conda/envs/{env}/bin/python"


def run_conda(env: str, script: str, extra_args: list = None, cwd: str = None,
              cuda_visible: str = None):
    py = env_python(env)
    cmd = [py, "-u", script] + (extra_args or [])
    log(f"{env}: {Path(script).name} ...")
    env_vars = dict(os.environ)
    if cuda_visible is not None:
        env_vars["CUDA_VISIBLE_DEVICES"] = str(cuda_visible)
    result = subprocess.run(cmd, cwd=cwd, capture_output=False, env=env_vars)
    if result.returncode != 0:
        raise RuntimeError(f"subprocess 失败 (code={result.returncode}): {' '.join(cmd)}")


# ─────────────────────── 相机参数解析 ────────────────────────────

def parse_camera_txt(
    path: str, H: int, W: int,
) -> tuple:
    """
    解析 camera.txt，返回像素坐标系内参和 c2w 外参。

    camera.txt 格式（每行）：
      frame  fx  fy  cx  cy  d1  d2  w2c(3x4 row-major)
    其中 fx/fy/cx/cy 已按图像尺寸归一化。

    所有数据已统一为 w2c + OpenCV 约定。
    本函数读取 w2c 并求逆返回 c2w。

    返回
    ----
    intrinsics : (N, 3, 3)  像素坐标系内参
    c2w        : (N, 3, 4)  相机到世界变换
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
    c2w_arr = np.zeros((N, 3, 4), dtype=np.float64)

    for idx, vals in enumerate(raw_entries):
        fx = vals[1] * W
        fy = vals[2] * H
        cx = vals[3] * W
        cy = vals[4] * H
        intrinsics[idx] = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]

        w2c_flat = np.array(vals[7:19], dtype=np.float64)
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :] = w2c_flat.reshape(3, 4)
        c2w = np.linalg.inv(w2c)
        c2w_arr[idx] = c2w[:3, :]

    return intrinsics, c2w_arr


def to_4x4(ext: np.ndarray) -> np.ndarray:
    """(N,3,4) -> (N,4,4) 或 (3,4) -> (4,4)"""
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


def get_prompt(entry: dict) -> str:
    """从 metadata.json 读 prompt。"""
    meta_path = entry["sample_dir"] / "metadata.json"
    if meta_path.exists():
        meta = load_json(str(meta_path)) or {}
        for key in ("prompt", "caption", "description", "text"):
            if key in meta:
                return str(meta[key])
    return "camera moving through a scene"
