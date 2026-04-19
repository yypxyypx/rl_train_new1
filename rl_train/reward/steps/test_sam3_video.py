"""
SAM3 Video Segmentation 快速测试脚本
====================================
Conda env: SAM3

验证 build_sam3_video_predictor 接口能否对帧目录/mp4 做跨帧跟踪分割，
并与当前逐帧图片模式做对比输出。

Usage:
    conda run -n SAM3 python test_sam3_video.py \
        --video_frames_dir /path/to/frames/ \
        --objects "person,chair,table" \
        --gpu 0 \
        --save_dir /tmp/sam3_video_test
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

_REWARD_DIR = Path(__file__).resolve().parent.parent
_RL_CODE_DIR = _REWARD_DIR.parent.parent
_THIRD_PARTY_DIR = _RL_CODE_DIR / "third_party"
_SAM3_SRC = _THIRD_PARTY_DIR / "repos" / "SAM3" / "sam3"
_SAM3_PKG = _THIRD_PARTY_DIR / "repos" / "SAM3" / "sam3"

for _p in [str(_SAM3_SRC), str(_SAM3_SRC / "sam3"), str(_SAM3_PKG)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

_MODEL_DIR = Path(os.environ.get("RL_MODEL_ROOT", str(_RL_CODE_DIR.parent / "RL" / "model")))
SAM3_CKPT = str(_MODEL_DIR / "SAM3" / "sam3.pt")
SAM3_BPE = str(_SAM3_PKG / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz")


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette for visualisation
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = np.array([
    [0,   0,   0  ],  # 0 = background (black)
    [255, 80,  80 ],  # 1
    [80,  255, 80 ],  # 2
    [80,  80,  255],  # 3
    [255, 255, 80 ],  # 4
    [255, 80,  255],  # 5
    [80,  255, 255],  # 6
    [255, 160, 80 ],  # 7
    [160, 80,  255],  # 8
], dtype=np.uint8)


def overlay_masks(frame_bgr: np.ndarray, label_map: np.ndarray, alpha=0.5) -> np.ndarray:
    """Overlay a colour-coded label map onto an BGR frame."""
    n_labels = int(label_map.max()) + 1
    palette = np.zeros((max(n_labels, len(PALETTE)), 3), dtype=np.uint8)
    palette[:len(PALETTE)] = PALETTE
    colour_mask = palette[label_map]       # (H, W, 3) BGR order would be reversed, but cv2 uses BGR
    colour_mask_bgr = colour_mask[:, :, ::-1].copy()
    out = frame_bgr.copy().astype(np.float32)
    fg = label_map > 0
    out[fg] = out[fg] * (1 - alpha) + colour_mask_bgr[fg].astype(np.float32) * alpha
    return out.astype(np.uint8)


def load_frames(frames_dir: str) -> list:
    exts = {".jpg", ".jpeg", ".png"}
    paths = sorted(
        [os.path.join(frames_dir, f) for f in os.listdir(frames_dir)
         if os.path.splitext(f)[1].lower() in exts]
    )
    if not paths:
        raise FileNotFoundError(f"No image frames found in {frames_dir}")
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Video-mode segmentation  (build_sam3_video_predictor)
# ─────────────────────────────────────────────────────────────────────────────
def run_video_mode(frames_dir: str, object_names: list, gpu: int) -> np.ndarray:
    """
    Use SAM3 video predictor: detect on frame-0, track across all frames.
    Returns label_maps (T, H, W) int16.
    """
    import torch
    from sam3.model_builder import build_sam3_video_predictor

    print(f"\n[VideoMode] Loading SAM3 video predictor from {SAM3_CKPT} ...")
    predictor = build_sam3_video_predictor(
        checkpoint_path=SAM3_CKPT,
        bpe_path=SAM3_BPE,
        gpus_to_use=[gpu],
    )

    # ── open session ─────────────────────────────────────────────────────────
    resp = predictor.handle_request(dict(type="start_session", resource_path=frames_dir))
    session_id = resp["session_id"]
    print(f"[VideoMode] Session opened: {session_id}")

    # ── per-concept: add text prompt → propagate ──────────────────────────────
    # We accumulate object masks from each concept separately, then combine.
    # obj_id_offset lets different concepts get different label values.
    frame_paths = load_frames(frames_dir)
    T = len(frame_paths)
    first = cv2.imread(frame_paths[0])
    H, W = first.shape[:2]
    label_maps = np.zeros((T, H, W), dtype=np.int16)

    for concept_idx, concept in enumerate(object_names, start=1):
        print(f"[VideoMode] Propagating concept [{concept_idx}/{len(object_names)}]: '{concept}' ...")

        # reset then set new text prompt on frame 0
        _ = predictor.handle_request(dict(type="reset_session", session_id=session_id))
        _ = predictor.handle_request(dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=0,
            text=concept,
        ))

        # propagate across the full video
        per_frame_out = {}
        for resp in predictor.handle_stream_request(
            dict(type="propagate_in_video", session_id=session_id)
        ):
            per_frame_out[resp["frame_index"]] = resp["outputs"]

        # merge masks into label_maps (later concept overwrites earlier for overlap)
        for t in range(T):
            out = per_frame_out.get(t)
            if out is None:
                continue
            obj_ids = out.get("out_obj_ids", [])
            masks   = out.get("out_binary_masks", [])  # list of (1, H_out, W_out) bool tensors
            for _oid, mask_t in zip(obj_ids, masks):
                import torch
                if isinstance(mask_t, torch.Tensor):
                    mask_np = mask_t.cpu().numpy()
                else:
                    mask_np = np.asarray(mask_t)
                # squeeze all size-1 dims: could be (1,H,W), (H,W), etc.
                mask_np = mask_np.squeeze()
                if mask_np.ndim == 0:
                    continue  # scalar, skip
                if mask_np.ndim == 3:
                    mask_np = mask_np[0]  # take first if still 3D
                mask_np = mask_np.astype(bool)
                if mask_np.shape != (H, W):
                    mask_np = cv2.resize(
                        mask_np.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                # only write where not yet occupied (priority: earlier concept wins)
                free = label_maps[t] == 0
                label_maps[t][free & mask_np] = concept_idx

    _ = predictor.handle_request(dict(type="close_session", session_id=session_id))
    predictor.shutdown()
    return label_maps


# ─────────────────────────────────────────────────────────────────────────────
# Save comparison video
# ─────────────────────────────────────────────────────────────────────────────
def _write_frames_to_h264(frames_bgr: list, out_path: str, fps: int):
    """Write a list of BGR frames to an h264-encoded mp4 via ffmpeg."""
    import subprocess, tempfile
    H, W = frames_bgr[0].shape[:2]
    tmp = out_path + ".raw.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp, fourcc, fps, (W, H))
    for f in frames_bgr:
        writer.write(f)
    writer.release()
    # Re-encode to h264 for broad compatibility
    cmd = [
        "ffmpeg", "-y", "-i", tmp,
        "-vcodec", "libx264", "-crf", "23", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", out_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    os.remove(tmp)
    if result.returncode != 0:
        # fallback: keep mp4v version
        os.rename(tmp, out_path) if os.path.exists(tmp) else None
        print(f"[WARN] ffmpeg re-encode failed, keeping raw: {result.stderr.decode()[:200]}")


def save_comparison_video(frame_paths: list, label_maps: np.ndarray,
                          object_names: list, out_path: str, fps: int = 10):
    first = cv2.imread(frame_paths[0])
    H, W = first.shape[:2]

    legend_lines = [f"{i+1}: {n}" for i, n in enumerate(object_names)]
    frames_out = []

    for t, fp in enumerate(frame_paths):
        frame = cv2.imread(fp)
        if frame is None:
            frame = np.zeros((H, W, 3), dtype=np.uint8)
        seg = overlay_masks(frame, label_maps[t])

        # draw legend on segmented side
        for li, line in enumerate(legend_lines[:8]):
            colour = PALETTE[(li + 1) % len(PALETTE)].tolist()[::-1]  # BGR
            cv2.putText(seg, line, (10, 25 + li * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)

        frames_out.append(np.concatenate([frame, seg], axis=1))

    _write_frames_to_h264(frames_out, out_path, fps)
    print(f"[Saved] comparison video → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_frames_dir", required=True, help="帧图像目录")
    parser.add_argument("--objects", default="person,chair",
                        help="逗号分隔的物体名称列表，例如 'person,chair,table'")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--save_dir", default="/tmp/sam3_video_test")
    parser.add_argument("--fps", type=int, default=10, help="输出视频帧率")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    object_names = [o.strip() for o in args.objects.split(",") if o.strip()]
    print(f"[test_sam3_video] 物体列表: {object_names}")
    print(f"[test_sam3_video] 权重路径: {SAM3_CKPT}")

    frame_paths = load_frames(args.video_frames_dir)
    print(f"[test_sam3_video] 帧数: {len(frame_paths)}")

    # ── Video mode ────────────────────────────────────────────────────────────
    label_maps_video = run_video_mode(args.video_frames_dir, object_names, args.gpu)

    coverage = (label_maps_video > 0).mean()
    print(f"\n[VideoMode] 覆盖率: {coverage:.1%}")

    per_concept_presence = {
        name: (label_maps_video == i + 1).any(axis=(1, 2)).mean()
        for i, name in enumerate(object_names)
    }
    print("[VideoMode] 各物体出现帧比例:")
    for name, ratio in per_concept_presence.items():
        print(f"  {name}: {ratio:.1%}")

    # save npz
    npz_path = os.path.join(args.save_dir, "label_maps_video.npz")
    np.savez_compressed(npz_path,
                        label_maps=label_maps_video,
                        object_names=np.array(object_names, dtype=object))
    print(f"[Saved] label_maps → {npz_path}")

    # save visualisation video
    vid_path = os.path.join(args.save_dir, "video_mode_comparison.mp4")
    save_comparison_video(frame_paths, label_maps_video, object_names, vid_path, fps=args.fps)


if __name__ == "__main__":
    main()
