"""
SAM3 逐帧 vs 视频追踪 批量对比
================================
Conda env: SAM3

对每个样本生成三列对比视频:
  左列: 原始帧
  中列: 逐帧分割 (来自 vlow_conf_videos 已有结果，若无则显示灰底说明)
  右列: SAM3 视频追踪 (新生成)

数据源:
  gen_0.mp4 ← RL/reward_weight_experiments/video_comparison/<sample>/gen_0.mp4
  逐帧分割  ← reward_debug_output/SAM/vlow_conf_videos/<sample>_gen_0_lowconf0.1_comparison.mp4

Usage:
    conda run -n SAM3 python run_sam3_video_comparison.py \
        --gpu 0 \
        --output_dir /path/to/output
"""

import argparse
import gc
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

# ─── path setup ──────────────────────────────────────────────────────────────
_REWARD_DIR = Path(__file__).resolve().parent.parent
_RL_CODE_DIR = _REWARD_DIR.parent.parent
_THIRD_PARTY_DIR = _RL_CODE_DIR / "third_party"
_SAM3_SRC = _THIRD_PARTY_DIR / "repos" / "SAM3" / "sam3"
_SAM3_PKG = _THIRD_PARTY_DIR / "repos" / "SAM3" / "sam3"

for _p in [str(_SAM3_SRC), str(_SAM3_SRC / "sam3"), str(_SAM3_PKG)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

_MODEL_DIR = Path(os.environ.get("RL_MODEL_ROOT", str(_RL_CODE_DIR / "model")))
SAM3_CKPT   = str(_MODEL_DIR / "SAM3" / "sam3.pt")
SAM3_BPE    = str(_SAM3_PKG / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz")
QWEN_MODEL  = str(_MODEL_DIR / "Qwen3-VL-8B-Instruct")

VIDEO_CMP_ROOT = Path("/home/users/puxin.yan-labs/RL/reward_weight_experiments/video_comparison")
VLOW_CONF_ROOT = Path("/home/users/puxin.yan-labs/reward_debug_output/SAM/vlow_conf_videos")

# ─── sample registry ─────────────────────────────────────────────────────────
# Format: (display_name, sample_dir_name, dataset)
# display_name = short label shown in video
# sample_dir_name = folder inside VIDEO_CMP_ROOT

_ALL_SAMPLES = [
    # ── dl3dv ──────────────────────────────────────────────────────────────
    ("dl3dv_0a4151528bffc2ce",  "0a4151528bffc2ce",           "dl3dv"),
    ("dl3dv_00def8c08b0a092e",  "00def8c08b0a092e",           "dl3dv"),
    ("dl3dv_0b328f5b77415725",  "0b328f5b77415725",           "dl3dv"),
    ("dl3dv_0f638de8c5c80d3d",  "0f638de8c5c80d3d",           "dl3dv"),
    ("dl3dv_0343987d0a3f4438",  "dl3dv_0343987d0a3f4438",     "dl3dv"),
    ("dl3dv_1061e37c81beab5b",  "dl3dv_1061e37c81beab5b",     "dl3dv"),
    ("dl3dv_01cd1a4633b366ce",  "dl3dv_01cd1a4633b366ce",     "dl3dv"),
    ("dl3dv_0a6c01ac32127687",  "dl3dv_0a6c01ac32127687",     "dl3dv"),
    # ── re10k ──────────────────────────────────────────────────────────────
    ("re10k_000075",  "re10k_000075_d942e48c948b3546",  "re10k"),
    ("re10k_000078",  "re10k_000078_342ea495bd00435d",  "re10k"),
    ("re10k_000096",  "re10k_000096_cc8e95a1e8d0489c",  "re10k"),
    ("re10k_000097",  "re10k_000097_2b5b5b4f4fc526ba",  "re10k"),
    ("re10k_000102",  "re10k_000102_7c1d611595e6c833",  "re10k"),
    ("re10k_000105",  "re10k_000105_84a7bc993f35b6b4",  "re10k"),
    ("re10k_000114",  "re10k_000114_a978e4b3eaf80a1a",  "re10k"),
    ("re10k_000089",  "re10k_000089_0c916bcc9351521e",  "re10k"),
]

# ─── colour palette ───────────────────────────────────────────────────────────
PALETTE = np.array([
    [0,   0,   0  ],
    [255, 80,  80 ],
    [80,  255, 80 ],
    [80,  80,  255],
    [255, 255, 80 ],
    [255, 80,  255],
    [80,  255, 255],
    [255, 160, 80 ],
    [160, 80,  255],
], dtype=np.uint8)


# ─── helpers ─────────────────────────────────────────────────────────────────
def overlay_masks(frame_bgr: np.ndarray, label_map: np.ndarray, alpha=0.5) -> np.ndarray:
    n_labels = int(label_map.max()) + 1
    palette = np.zeros((max(n_labels, len(PALETTE)), 3), dtype=np.uint8)
    palette[:len(PALETTE)] = PALETTE
    colour_mask = palette[label_map][:, :, ::-1].copy()
    out = frame_bgr.copy().astype(np.float32)
    fg = label_map > 0
    out[fg] = out[fg] * (1 - alpha) + colour_mask[fg].astype(np.float32) * alpha
    return out.astype(np.uint8)


def extract_frames_from_video(video_path: str, out_dir: str) -> list:
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    paths = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        p = os.path.join(out_dir, f"frame_{idx:05d}.jpg")
        cv2.imwrite(p, frame)
        paths.append(p)
        idx += 1
    cap.release()
    return paths


def read_video_frames(video_path: str) -> list:
    """Read all BGR frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    return frames


def make_placeholder_frame(H: int, W: int, text: str) -> np.ndarray:
    """Dark frame with text (for 'no per-frame result' cases)."""
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[:] = (40, 40, 40)
    lines = [text[i:i+22] for i in range(0, len(text), 22)]
    for li, line in enumerate(lines):
        cv2.putText(frame, line, (10, 30 + li * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    return frame


def write_h264_video(frames_bgr: list, out_path: str, fps: int = 10):
    """Write frames to h264 mp4 via ffmpeg."""
    H, W = frames_bgr[0].shape[:2]
    tmp = out_path + ".tmp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp, fourcc, fps, (W, H))
    for f in frames_bgr:
        writer.write(f)
    writer.release()
    cmd = ["ffmpeg", "-y", "-i", tmp,
           "-vcodec", "libx264", "-crf", "23", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", out_path]
    r = subprocess.run(cmd, capture_output=True)
    if os.path.exists(tmp):
        os.remove(tmp)
    if r.returncode != 0:
        print(f"[WARN] ffmpeg failed: {r.stderr.decode()[:300]}")


def find_vlow_conf_video(sample_dir_name: str) -> str | None:
    """Find the per-frame comparison video in vlow_conf_videos."""
    # sample_dir_name may be like '0a4151528bffc2ce' or 'dl3dv_0a4151528bffc2ce'
    # vlow_conf filename: dl3dv_<id>_gen_0_lowconf0.1_comparison.mp4
    # Try full match first, then short ID match
    for f in VLOW_CONF_ROOT.glob("*.mp4"):
        name = f.name
        # strip leading 'dl3dv_' or 're10k_' from sample_dir_name for matching
        clean = sample_dir_name.lstrip("dl3dv_").lstrip("re10k_")
        if clean in name and "gen_0" in name:
            return str(f)
    return None


# ─── Qwen identification ─────────────────────────────────────────────────────
def identify_objects_qwen(first_frame_path: str, device: str) -> list:
    """Use Qwen3-VL to identify objects in the first frame."""
    import json, re, torch, gc
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText

    processor = AutoProcessor.from_pretrained(QWEN_MODEL, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        QWEN_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()

    pil_img = Image.open(first_frame_path).convert("RGB")
    system_prompt = (
        "You are a visual object detection assistant. "
        "List the most important and distinct object categories visible in the image. "
        "Rules: "
        "1. Return ONLY a valid JSON array, no explanation, no markdown. "
        "2. Maximum 12 entries. Pick the most salient objects only. "
        "3. Each entry must be a simple singular English noun or short noun phrase (2-3 words max). "
        "4. No duplicates. No articles (use 'chair' not 'a chair'). "
        'Example output: ["person", "dog", "bench", "tree"]'
    )
    user_text = (
        "List up to 12 most important object categories in this image. "
        'Return ONLY a JSON array like ["chair", "table", "lamp"]. No markdown. No explanation.'
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image", "image": pil_img},
            {"type": "text", "text": user_text},
        ]},
    ]
    try:
        from qwen_vl_utils import process_vision_info
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to(device)
    except Exception:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[pil_img], padding=True, return_tensors="pt").to(device)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=256)
    raw = processor.batch_decode(
        generated_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0].strip()

    # parse JSON
    if "```" in raw:
        for part in raw.split("```"):
            s = part.strip()
            if s.startswith("json"):
                s = s[4:].strip()
            if s.startswith("["):
                raw = s; break
    s, e = raw.find("["), raw.rfind("]")
    try:
        objs = json.loads(raw[s:e+1]) if s != -1 and e > s else []
    except Exception:
        objs = re.findall(r'"([^"]+)"', raw[s:] if s != -1 else raw)

    objs = list(dict.fromkeys(o.strip().lower() for o in objs if o.strip()))
    print(f"  [Qwen] 识别到 {len(objs)} 个物体: {objs}")

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return objs


# ─── SAM3 video mode ─────────────────────────────────────────────────────────
def run_sam3_video_mode(frames_dir: str, object_names: list, gpu: int) -> np.ndarray:
    """Run SAM3 video predictor and return label_maps (T, H, W) int16."""
    import torch
    from sam3.model_builder import build_sam3_video_predictor

    predictor = build_sam3_video_predictor(
        checkpoint_path=SAM3_CKPT,
        bpe_path=SAM3_BPE,
        gpus_to_use=[gpu],
    )

    resp = predictor.handle_request(dict(type="start_session", resource_path=frames_dir))
    session_id = resp["session_id"]

    # get H, W from first frame
    frame_paths = sorted([
        os.path.join(frames_dir, f) for f in os.listdir(frames_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    T = len(frame_paths)
    first = cv2.imread(frame_paths[0])
    H, W = first.shape[:2]
    label_maps = np.zeros((T, H, W), dtype=np.int16)

    for concept_idx, concept in enumerate(object_names, start=1):
        _ = predictor.handle_request(dict(type="reset_session", session_id=session_id))
        _ = predictor.handle_request(dict(
            type="add_prompt", session_id=session_id, frame_index=0, text=concept,
        ))
        per_frame_out = {}
        for resp in predictor.handle_stream_request(
            dict(type="propagate_in_video", session_id=session_id)
        ):
            per_frame_out[resp["frame_index"]] = resp["outputs"]

        for t in range(T):
            out = per_frame_out.get(t)
            if out is None:
                continue
            obj_ids = out.get("out_obj_ids", [])
            masks   = out.get("out_binary_masks", [])
            for _oid, mask_t in zip(obj_ids, masks):
                import torch as _torch
                mask_np = mask_t.cpu().numpy() if isinstance(mask_t, _torch.Tensor) else np.asarray(mask_t)
                mask_np = mask_np.squeeze()
                if mask_np.ndim == 0:
                    continue
                if mask_np.ndim == 3:
                    mask_np = mask_np[0]
                mask_np = mask_np.astype(bool)
                if mask_np.shape != (H, W):
                    mask_np = cv2.resize(mask_np.astype(np.uint8), (W, H),
                                         interpolation=cv2.INTER_NEAREST).astype(bool)
                free = label_maps[t] == 0
                label_maps[t][free & mask_np] = concept_idx

    _ = predictor.handle_request(dict(type="close_session", session_id=session_id))
    predictor.shutdown()
    import gc; gc.collect()
    torch.cuda.empty_cache()
    return label_maps


# ─── build 3-panel comparison video ─────────────────────────────────────────
def build_comparison_video(
    raw_frames: list,        # list of BGR ndarray (original frames)
    perframe_video: str | None,  # path to existing per-frame comparison mp4 (or None)
    label_maps: np.ndarray,  # (T, H, W) int16 from video mode
    object_names: list,
    sample_name: str,
    out_path: str,
    fps: int = 10,
):
    T = len(raw_frames)
    H, W = raw_frames[0].shape[:2]

    # Load per-frame frames (left 560px of the comparison video = original+overlay)
    # The vlow_conf comparison video is 2-panel: left = raw SAM overlay, right = filtered
    # We extract frames from left half of that video
    pf_frames = []
    if perframe_video and os.path.exists(perframe_video):
        all_pf = read_video_frames(perframe_video)
        if all_pf:
            pf_H, pf_W2 = all_pf[0].shape[:2]
            pf_W = pf_W2 // 2
            for f in all_pf:
                # resize left half to match H, W
                left = f[:, :pf_W, :]
                if left.shape[:2] != (H, W):
                    left = cv2.resize(left, (W, H))
                pf_frames.append(left)
            # repeat/trim to match T
            if len(pf_frames) < T:
                pf_frames += [pf_frames[-1]] * (T - len(pf_frames))
            pf_frames = pf_frames[:T]
        else:
            pf_frames = [make_placeholder_frame(H, W, "per-frame: video read failed")] * T
    else:
        pf_frames = [make_placeholder_frame(H, W, "per-frame: not available")] * T

    legend_lines = [f"{i+1}: {n}" for i, n in enumerate(object_names[:8])]
    out_frames = []

    for t in range(T):
        raw = raw_frames[t].copy()
        pf  = pf_frames[t].copy()
        vid = overlay_masks(raw_frames[t], label_maps[t], alpha=0.5)

        # add legend to video mode panel
        for li, line in enumerate(legend_lines):
            colour = PALETTE[(li + 1) % len(PALETTE)].tolist()[::-1]
            cv2.putText(vid, line, (8, 22 + li * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)

        # header bar
        bar_h = 28
        bar = np.zeros((bar_h, W * 3, 3), dtype=np.uint8)
        bar[:] = (30, 30, 30)
        labels = [
            (f"[{sample_name}]  Original", 0),
            ("Per-Frame SAM3", W),
            ("Video Tracking SAM3 (NEW)", W * 2),
        ]
        for txt, x in labels:
            cv2.putText(bar, txt, (x + 6, 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

        combined = np.concatenate([raw, pf, vid], axis=1)
        full = np.concatenate([bar, combined], axis=0)
        out_frames.append(full)

    write_h264_video(out_frames, out_path, fps)
    print(f"  [Done] → {out_path}")


# ─── process one sample ───────────────────────────────────────────────────────
def process_sample(sample_name: str, sample_dir_name: str, output_dir: str, gpu: int):
    print(f"\n{'='*60}")
    print(f"[Sample] {sample_name}")
    print(f"{'='*60}")

    sample_dir = VIDEO_CMP_ROOT / sample_dir_name
    gen0_mp4   = sample_dir / "gen_0.mp4"
    if not gen0_mp4.exists():
        print(f"  [SKIP] gen_0.mp4 not found: {gen0_mp4}")
        return

    os.makedirs(output_dir, exist_ok=True)
    out_video = os.path.join(output_dir, f"{sample_name}_comparison.mp4")

    # ── step 1: extract frames ──────────────────────────────────────────────
    frames_dir = os.path.join(output_dir, f"_frames_{sample_name}")
    print(f"  [1/4] Extracting frames from gen_0.mp4 ...")
    frame_paths = extract_frames_from_video(str(gen0_mp4), frames_dir)
    print(f"        {len(frame_paths)} frames extracted")

    # Load raw BGR frames
    raw_frames = [cv2.imread(p) for p in frame_paths]
    raw_frames = [f for f in raw_frames if f is not None]

    # ── step 2: Qwen identify objects ───────────────────────────────────────
    print(f"  [2/4] Qwen identifying objects ...")
    device = f"cuda:{gpu}"
    try:
        object_names = identify_objects_qwen(frame_paths[0], device)
    except Exception as e:
        print(f"  [WARN] Qwen failed: {e}, using defaults")
        object_names = ["object"]
    if not object_names:
        object_names = ["object"]

    # ── step 3: SAM3 video mode ──────────────────────────────────────────────
    print(f"  [3/4] SAM3 video mode ({len(object_names)} concepts) ...")
    try:
        label_maps = run_sam3_video_mode(frames_dir, object_names, gpu)
        coverage = (label_maps > 0).mean()
        print(f"        coverage={coverage:.1%}")
    except Exception as e:
        print(f"  [ERROR] SAM3 video mode failed: {e}")
        import traceback; traceback.print_exc()
        label_maps = np.zeros((len(raw_frames),
                                raw_frames[0].shape[0],
                                raw_frames[0].shape[1]), dtype=np.int16)

    # ── step 4: find per-frame comparison video ──────────────────────────────
    pf_video = find_vlow_conf_video(sample_dir_name)
    if pf_video:
        print(f"  [4/4] Per-frame video found: {Path(pf_video).name}")
    else:
        print(f"  [4/4] Per-frame video NOT found (will show placeholder)")

    # ── step 5: build comparison video ──────────────────────────────────────
    build_comparison_video(
        raw_frames=raw_frames,
        perframe_video=pf_video,
        label_maps=label_maps,
        object_names=object_names,
        sample_name=sample_name,
        out_path=out_video,
    )

    # cleanup extracted frames
    import shutil
    shutil.rmtree(frames_dir, ignore_errors=True)


# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",        type=int,   default=0)
    parser.add_argument("--output_dir", type=str,
                        default="/home/users/puxin.yan-labs/reward_debug_output/SAM/video_vs_perframe")
    parser.add_argument("--dataset",    type=str,   default="all",
                        help="Filter by dataset: 'dl3dv', 're10k', or 'all'")
    parser.add_argument("--samples",    type=str,   default=None,
                        help="Comma-separated sample names to process (overrides --dataset)")
    args = parser.parse_args()

    samples = _ALL_SAMPLES
    if args.samples:
        names = set(args.samples.split(","))
        samples = [s for s in samples if s[0] in names]
    elif args.dataset != "all":
        samples = [s for s in samples if s[2] == args.dataset]

    print(f"[run_sam3_video_comparison] Processing {len(samples)} samples → {args.output_dir}")
    print(f"  SAM3 checkpoint: {SAM3_CKPT}")
    print(f"  GPU: {args.gpu}")

    for sample_name, sample_dir_name, dataset in samples:
        try:
            process_sample(
                sample_name=sample_name,
                sample_dir_name=sample_dir_name,
                output_dir=args.output_dir,
                gpu=args.gpu,
            )
        except Exception as e:
            print(f"\n[ERROR] {sample_name}: {e}")
            import traceback; traceback.print_exc()

    print(f"\n[Done] All results in: {args.output_dir}")


if __name__ == "__main__":
    main()
