"""DL3DV dataset parser -> RawSample (OpenCV c2w).

Raw format: transforms.json (global intrinsics at full res + per-frame c2w)
            images_2/ (PNG/JPG; half resolution vs transforms w,h)

Coordinate convention: transforms.json stores c2w in OpenGL/NeRF convention
(Y-up, Z-backward). Converted to OpenCV via c2w @ diag(1, -1, -1, 1).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from PIL import Image

from .base import RawSample, c2w_opengl_to_opencv


def _find_image(images_dir: Path, fname: str) -> Optional[Path]:
    p = images_dir / fname
    if p.exists():
        return p
    stem = fname.rsplit(".", 1)[0]
    for ext in (".jpg", ".jpeg", ".png"):
        alt = images_dir / (stem + ext)
        if alt.exists():
            return alt
    return None


def parse(
    data_root: Path,
    max_samples: int = 0,
    verbose: bool = True,
) -> Iterator[RawSample]:
    dl3dv_dir = Path(data_root) / "DL3DV"
    if not dl3dv_dir.exists():
        dl3dv_dir = Path(data_root) / "dl3dv"
    if not dl3dv_dir.exists():
        warnings.warn(f"DL3DV: not found at {data_root}/DL3DV")
        return

    sample_dirs = sorted(d for d in dl3dv_dir.iterdir() if d.is_dir())
    if max_samples > 0:
        sample_dirs = sample_dirs[:max_samples]

    for sample_dir in sample_dirs:
        sample_id = sample_dir.name[:16]
        transforms_path = sample_dir / "transforms.json"
        images_dir = sample_dir / "images_2"

        if not transforms_path.exists() or not images_dir.exists():
            if verbose:
                print(f"  [dl3dv] Skip {sample_id}: missing files")
            continue

        with open(transforms_path) as f:
            tfm = json.load(f)

        orig_w_full = int(tfm["w"])
        orig_h_full = int(tfm["h"])
        fl_x_full = float(tfm["fl_x"])
        fl_y_full = float(tfm["fl_y"])
        cx_full = float(tfm["cx"])
        cy_full = float(tfm["cy"])

        frames_meta = tfm["frames"]
        N = len(frames_meta)
        if N == 0:
            continue

        first_fname = frames_meta[0]["file_path"].split("/")[-1]
        first_path = _find_image(images_dir, first_fname)
        if first_path is None:
            continue

        first_img = np.array(Image.open(first_path).convert("RGB"))
        orig_h_img, orig_w_img = first_img.shape[:2]

        scale_x = orig_w_img / orig_w_full
        scale_y = orig_h_img / orig_h_full

        K = np.array([
            [fl_x_full * scale_x, 0.0, cx_full * scale_x],
            [0.0, fl_y_full * scale_y, cy_full * scale_y],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        frames, c2ws_list, Ks_list = [], [], []
        skip = False

        for i, fm in enumerate(frames_meta):
            fname = fm["file_path"].split("/")[-1]
            img_path = _find_image(images_dir, fname)
            if img_path is None:
                skip = True
                break

            img = np.array(Image.open(img_path).convert("RGB"))
            frames.append(img)

            c2w_gl = np.array(fm["transform_matrix"], dtype=np.float64)
            if c2w_gl.shape == (3, 4):
                tmp = np.eye(4, dtype=np.float64)
                tmp[:3, :] = c2w_gl
                c2w_gl = tmp
            c2ws_list.append(c2w_opengl_to_opencv(c2w_gl))
            Ks_list.append(K.copy())

        if skip:
            continue

        yield RawSample(
            sample_id=sample_id,
            frames=frames,
            c2ws=np.stack(c2ws_list),
            Ks=np.stack(Ks_list),
            caption="A 3D scene",
            orig_w=orig_w_img,
            orig_h=orig_h_img,
            dataset="dl3dv",
            orig_id=sample_dir.name,
        )
