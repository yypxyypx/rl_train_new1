"""
Step: DA3 Depth Estimation + Camera Pose
=========================================
Conda env: rl_da3

输出 extrinsics 为 w2c OpenCV (N, 3, 4)。

Usage:
    conda run -n rl_da3 python step_da3.py \
        --video_frames_dir /path/to/frames/ \
        --output /path/to/da3_output.npz \
        --gpu 0
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_REWARD_DIR = Path(__file__).resolve().parent.parent
_RL_TRAIN_DIR = _REWARD_DIR.parent
_RL_CODE_DIR = _RL_TRAIN_DIR.parent
_THIRD_PARTY_DIR = _RL_CODE_DIR / "third_party"

DA3_SRC = str(_THIRD_PARTY_DIR / "repos" / "DA3" / "Depth-Anything-3" / "src")
DA3_WEIGHTS = str(
    Path(os.environ.get("RL_MODEL_ROOT", str(_RL_CODE_DIR / "model")))
    / "DA3NESTED-GIANT-LARGE-1.1"
)


def _load_da3_local(model_dir: str):
    """Load DepthAnything3 from a local directory, bypassing HF Hub validation."""
    import json
    from depth_anything_3.api import DepthAnything3

    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)
    model_name = cfg.get("model_name", "da3nested-giant-large")

    model = DepthAnything3(model_name=model_name)

    weights_path = os.path.join(model_dir, "model.safetensors")
    try:
        from safetensors.torch import load_file
        state_dict = load_file(weights_path)
    except ImportError:
        import torch as _torch
        state_dict = _torch.load(weights_path, map_location="cpu")

    model.load_state_dict(state_dict, strict=False)
    return model


def run_da3(frame_paths: list[str], device: str = "cuda:0") -> dict:
    """
    Run DA3 on all frames together for globally consistent depth + pose.
    Returns dict with numpy arrays: depth, extrinsics (w2c OpenCV), intrinsics, conf.
    """
    if DA3_SRC not in sys.path:
        sys.path.insert(0, DA3_SRC)

    print(f"[DA3] Loading model from {DA3_WEIGHTS}")
    model = _load_da3_local(DA3_WEIGHTS)
    model = model.to(device=torch.device(device)).eval()
    print(f"[DA3] Model loaded on {device}")

    images = []
    for p in frame_paths:
        img = Image.open(p).convert("RGB")
        images.append(np.array(img))
    print(f"[DA3] Processing {len(images)} frames, resolution {images[0].shape[:2]}")

    with torch.no_grad():
        prediction = model.inference(images)

    result = {
        "depth": prediction.depth,           # (N, H_d, W_d)
        "extrinsics": prediction.extrinsics,  # (N, 3, 4) w2c OpenCV
        "intrinsics": prediction.intrinsics,  # (N, 3, 3)
        "conf": prediction.conf,              # (N, H_d, W_d)
    }

    print(f"[DA3] Output shapes:")
    for k, v in result.items():
        if v is not None:
            print(f"  {k}: {v.shape}")

    del model
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description="DA3 depth + camera pose")
    parser.add_argument("--video_frames_dir", required=True)
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    frame_paths = sorted([
        os.path.join(args.video_frames_dir, f)
        for f in os.listdir(args.video_frames_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    if not frame_paths:
        raise FileNotFoundError(f"No image frames in {args.video_frames_dir}")
    print(f"[step_da3] Found {len(frame_paths)} frames")

    result = run_da3(frame_paths, device=f"cuda:{args.gpu}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(
        args.output,
        depth=result["depth"],
        extrinsics=result["extrinsics"],
        intrinsics=result["intrinsics"],
        conf=result["conf"] if result["conf"] is not None else np.array([]),
    )
    print(f"[step_da3] Saved DA3 output to {args.output}")


if __name__ == "__main__":
    main()
