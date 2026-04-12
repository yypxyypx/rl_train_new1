"""RealEstate10K dataset parser → RawSample (OpenCV c2w).

Raw format
----------
Each .torch file contains a list of video sequences. We take index 0 from
every file (200 files → 200 samples).

cameras  [N, 18]  where:
  [0:4]  = [fx/W, fy/H, cx/W, cy/H]  (normalised intrinsics, W=640, H=360)
  [4:6]  = distortion k1, k2  (ignored)
  [6:18] = w2c 3×4 row-major  (OpenCV convention, verified det(R)=1)

images   list of JPEG bytes stored as uint8 tensors.

Coordinate convention: cameras are already OpenCV (X-right, Y-down, Z-forward),
so w2c → inv → c2w is directly in OpenCV convention. No conversion needed.
"""

from __future__ import annotations

import io
import warnings
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

from .base import RawSample, pad_3x4_to_4x4, w2c_to_c2w


def parse(
    data_root: Path,
    max_samples: int = 0,
    verbose: bool = True,
) -> Iterator[RawSample]:
    """Yield RawSample for each valid sequence in re10k .torch files.

    Parameters
    ----------
    data_root : directory containing re10k/ subdirectory with .torch files
    max_samples : limit number of samples (0 = all)
    """
    import torch as _torch

    torch_dir = Path(data_root) / "re10k"
    torch_files = sorted(torch_dir.glob("*.torch"))
    if not torch_files:
        warnings.warn(f"re10k: no .torch files found in {torch_dir}")
        return

    if max_samples > 0:
        torch_files = torch_files[:max_samples]

    for torch_file in torch_files:
        try:
            sequences = _torch.load(torch_file, map_location="cpu")
        except Exception as e:
            if verbose:
                print(f"  [re10k] Failed to load {torch_file.name}: {e}")
            continue

        if not sequences:
            continue

        seq = sequences[0]
        key = seq["key"]
        sample_id = f"{torch_file.stem}_{key}"

        images_bytes: list = seq["images"]
        cameras: np.ndarray = seq["cameras"].numpy()  # (N, 18)
        N = len(images_bytes)

        # Decode all frames to get original resolution
        frames = []
        orig_w, orig_h = None, None
        for idx in range(N):
            img_bytes = images_bytes[idx].numpy().tobytes()
            img = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
            if orig_w is None:
                orig_h, orig_w = img.shape[:2]
            frames.append(img)

        # Build c2w (OpenCV convention) and K arrays
        c2ws_list = []
        Ks_list = []
        for idx in range(N):
            cam = cameras[idx]

            # Denormalise intrinsics: normalised → pixel at (orig_w, orig_h)
            fx_px = float(cam[0]) * orig_w
            fy_px = float(cam[1]) * orig_h
            cx_px = float(cam[2]) * orig_w
            cy_px = float(cam[3]) * orig_h

            K = np.array([
                [fx_px, 0.0, cx_px],
                [0.0, fy_px, cy_px],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)
            Ks_list.append(K)

            # w2c 3x4 → 4x4 → invert → c2w (already OpenCV)
            w2c_34 = cam[6:18].reshape(3, 4)
            w2c_44 = pad_3x4_to_4x4(w2c_34.astype(np.float64))
            c2w = w2c_to_c2w(w2c_44)
            c2ws_list.append(c2w)

        yield RawSample(
            sample_id=sample_id,
            frames=frames,
            c2ws=np.stack(c2ws_list),
            Ks=np.stack(Ks_list),
            caption="A real estate scene",
            orig_w=orig_w,
            orig_h=orig_h,
            dataset="re10k",
            orig_id=f"{torch_file.stem}/{key}",
        )
