"""Canonical intermediate representation for all datasets.

Every dataset-specific parser converts raw data into RawSample objects
with a unified coordinate convention (OpenCV: X-right, Y-down, Z-forward).
Resolution is left at original — resize/crop is handled by the converter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


@dataclass
class RawSample:
    """Dataset-agnostic sample, all cameras in OpenCV c2w convention."""

    sample_id: str
    frames: List[np.ndarray]          # original-resolution RGB (H, W, 3) uint8
    c2ws: np.ndarray                  # (N, 4, 4) camera-to-world, OpenCV convention
    Ks: np.ndarray                    # (N, 3, 3) pixel intrinsics at original res
    caption: str
    orig_w: int
    orig_h: int
    dataset: str                      # "re10k", "dl3dv", etc.
    orig_id: str = ""                 # original identifier in the raw dataset
    depths: Optional[np.ndarray] = None   # (N, H, W) float32 in meters, if available
    extra: Optional[Dict[str, Any]] = None  # free-form per-sample metadata for writers


# ═══════════════════════ Shared Geometry Utilities ═══════════════════════


OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


def c2w_opengl_to_opencv(c2w: np.ndarray) -> np.ndarray:
    """Convert c2w from OpenGL (Y-up, Z-back) to OpenCV (Y-down, Z-forward).

    Works for both single (4, 4) and batched (N, 4, 4) inputs.
    """
    return c2w @ OPENGL_TO_OPENCV


def w2c_to_c2w(w2c: np.ndarray) -> np.ndarray:
    """Invert w2c to c2w. Works for (4, 4) or (N, 4, 4)."""
    if w2c.ndim == 2:
        return np.linalg.inv(w2c)
    return np.linalg.inv(w2c)


def pad_3x4_to_4x4(mat: np.ndarray) -> np.ndarray:
    """Pad (3, 4) or (N, 3, 4) matrix to (4, 4) or (N, 4, 4)."""
    if mat.ndim == 2 and mat.shape == (3, 4):
        out = np.eye(4, dtype=mat.dtype)
        out[:3, :] = mat
        return out
    if mat.ndim == 3 and mat.shape[1:] == (3, 4):
        N = mat.shape[0]
        out = np.zeros((N, 4, 4), dtype=mat.dtype)
        out[:, :3, :] = mat
        out[:, 3, 3] = 1.0
        return out
    return mat


# ═══════════════════════ Shared Image Utilities ═══════════════════════


def compute_K_after_resize_crop(
    orig_w: int,
    orig_h: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    target_w: int,
    target_h: int,
) -> Tuple[float, float, float, float]:
    """Compute pixel intrinsics at (target_w, target_h) after aspect-preserving
    resize (cover) + center crop.

    Mirrors the logic in gen3r/Gen3R/gen3r/utils/data_utils.py::get_K().

    Parameters
    ----------
    orig_w, orig_h : original image dimensions
    fx, fy, cx, cy : pixel intrinsics at original resolution
    target_w, target_h : target resolution after resize+crop

    Returns
    -------
    (fx', fy', cx', cy') in pixel units at target resolution
    """
    w_ratio = target_w / orig_w
    h_ratio = target_h / orig_h
    resize_ratio = max(h_ratio, w_ratio)

    fx_new = fx * resize_ratio
    fy_new = fy * resize_ratio
    cx_new = cx * resize_ratio
    cy_new = cy * resize_ratio

    new_h = int(orig_h * resize_ratio)
    new_w = int(orig_w * resize_ratio)
    crop_h = (new_h - target_h) // 2
    crop_w = (new_w - target_w) // 2

    cx_new -= crop_w
    cy_new -= crop_h

    return fx_new, fy_new, cx_new, cy_new


def resize_center_crop(
    img: np.ndarray, target_h: int, target_w: int,
) -> np.ndarray:
    """Aspect-preserving resize (cover) + center crop to (target_h, target_w).

    Input:  H x W x 3 uint8.
    Output: target_h x target_w x 3 uint8.
    """
    h, w = img.shape[:2]
    resize_ratio = max(target_h / h, target_w / w)
    new_h = max(int(round(h * resize_ratio)), target_h)
    new_w = max(int(round(w * resize_ratio)), target_w)

    pil = Image.fromarray(img).resize((new_w, new_h), Image.LANCZOS)
    arr = np.array(pil)

    top = (arr.shape[0] - target_h) // 2
    left = (arr.shape[1] - target_w) // 2
    return arr[top: top + target_h, left: left + target_w]


def resize_center_crop_depth(
    depth: np.ndarray, target_h: int, target_w: int,
) -> np.ndarray:
    """Same spatial transform for depth maps (nearest-neighbour)."""
    h, w = depth.shape[:2]
    resize_ratio = max(target_h / h, target_w / w)
    new_h = max(int(round(h * resize_ratio)), target_h)
    new_w = max(int(round(w * resize_ratio)), target_w)

    pil = Image.fromarray(depth).resize((new_w, new_h), Image.NEAREST)
    arr = np.array(pil)

    top = (arr.shape[0] - target_h) // 2
    left = (arr.shape[1] - target_w) // 2
    return arr[top: top + target_h, left: left + target_w]


def sample_frame_indices(
    total: int, num_frames: int, fixed_start: bool = False,
) -> List[int]:
    """Sample a contiguous block of num_frames from a sequence of length total.

    fixed_start=False: random start (for training diversity).
    fixed_start=True:  always start at frame 0 (for reproducible eval).
    """
    if total <= num_frames:
        return list(range(total))
    if fixed_start:
        return list(range(num_frames))
    max_start = total - num_frames
    start = int(np.random.randint(0, max_start + 1))
    return list(range(start, start + num_frames))
