#!/usr/bin/env python3
"""
worker_sam3.py
==============
SAM3 批量 worker（视频追踪模式）—— 在 SAM3 conda 环境中运行，一次性处理所有视频。

设计原则：Qwen 加载一次做物体识别，SAM3 视频预测器加载一次，
顺序处理 batch_manifest 中的所有条目。
GT 视频先跑 Qwen 识别物体，pred 视频复用 GT 的物体列表（跳过 Qwen）。
SAM3 使用 propagate_in_video 视频追踪模式，比逐帧分割更稳定。

运行方式（由 run_benchmark.py 通过 conda run 调用）：
    conda run -n SAM3 python worker_sam3.py --batch_manifest /path/to/batch.json

batch_manifest.json 格式：
[
  {
    "video_path":           "/path/to/gt.mp4",
    "output_masks_npz":     "/path/to/gt_masks.npz",
    "output_objects_json":  "/path/to/gt_objects.json",
    "output_label_maps_npz":"/path/to/gt_label_maps.npz",
    "ref_objects_json":     null,         # null = 用 Qwen 识别
    "is_gt":                true,
    "max_frames":           0
  },
  ...
]
"""

import argparse
import gc
import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# ── 路径注入 ──────────────────────────────────────────────────────────────────
_THIRD_PARTY_DIR = Path(__file__).resolve().parent.parent
_SAM3_SRC = _THIRD_PARTY_DIR / "repos" / "SAM3" / "sam3"
_SAM3_PKG = _THIRD_PARTY_DIR / "repos" / "SAM3"
for _p in [str(_SAM3_SRC), str(_SAM3_PKG)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── 默认模型路径 ──────────────────────────────────────────────────────────────
_WORKSPACE = _THIRD_PARTY_DIR.parent.parent
_MODEL_DIR  = Path(os.environ.get("RL_MODEL_ROOT", str(_WORKSPACE / "RL" / "model")))
QWEN_MODEL  = str(_MODEL_DIR / "Qwen3-VL-8B-Instruct")
SAM3_CKPT   = str(_MODEL_DIR / "SAM3" / "sam3.pt")
SAM3_BPE    = str(_SAM3_PKG / "sam3" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz")
SAM3_CONF   = 0.5


# ── 帧提取（写入磁盘，视频预测器需要帧目录）─────────────────────────────────

def _extract_frames_to_dir(video_path: str, out_dir: str, max_frames: int = 0) -> int:
    """将视频帧提取为 PNG 文件保存到 out_dir，返回提取帧数。"""
    os.makedirs(out_dir, exist_ok=True)
    # 若帧已存在则跳过
    existing = sorted(Path(out_dir).glob("frame_*.png"))
    if existing:
        return len(existing)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(out_dir, f"frame_{idx:05d}.png"), frame)
        idx += 1
        if max_frames > 0 and idx >= max_frames:
            break
    cap.release()
    return idx


# ── Qwen 物体识别 ─────────────────────────────────────────────────────────────

def _identify_objects(first_frame_path: str, device: str, qwen_model, qwen_processor) -> list:
    """用 Qwen3-VL 识别首帧中的显著物体，返回物体名称列表。"""
    from qwen_vl_utils import process_vision_info

    pil_img = Image.open(first_frame_path).convert("RGB")

    system_prompt = (
        "You are a visual object detection assistant. "
        "List the most important and distinct object categories visible in the image. "
        "Rules: "
        "1. Return ONLY a valid JSON array, no explanation, no markdown. "
        "2. Maximum 15 entries. Pick the most salient objects only. "
        "3. Each entry must be a simple singular English noun or short noun phrase (2-3 words max). "
        "4. No duplicates, no overly specific variants (use 'chair' not 'wooden chair'). "
        "5. No articles (use 'chair' not 'a chair'). "
        'Example output: ["person", "dog", "bench", "tree", "bicycle"]'
    )
    user_text = (
        "List up to 15 most important object categories in this image. "
        'Return ONLY a JSON array like ["chair", "table", "lamp"]. '
        "No duplicates. No markdown. No explanation."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image", "image": pil_img},
            {"type": "text",  "text": user_text},
        ]},
    ]

    try:
        text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = qwen_processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(device)
    except Exception:
        text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = qwen_processor(
            text=[text], images=[pil_img], padding=True, return_tensors="pt",
        ).to(device)

    with torch.inference_mode():
        generated_ids = qwen_model.generate(**inputs, max_new_tokens=256)
    input_len = inputs["input_ids"].shape[1]
    raw = qwen_processor.batch_decode(
        generated_ids[:, input_len:], skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    import re
    if "```" in raw:
        for part in raw.split("```"):
            s = part.strip()
            if s.startswith("json"):
                s = s[4:].strip()
            if s.startswith("["):
                raw = s
                break
    start, end = raw.find("["), raw.rfind("]")
    if start != -1 and end > start:
        try:
            objs = json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            objs = re.findall(r'"([^"]+)"', raw[start:])
    else:
        objs = re.findall(r'"([^"]+)"', raw)

    objs = list(dict.fromkeys(o.strip().lower() for o in objs if o.strip()))
    print(f"  [Qwen] 识别到 {len(objs)} 个物体: {objs}")
    return objs


# ── SAM3 视频追踪 ─────────────────────────────────────────────────────────────

def _track_video(
    frames_dir: str,
    object_names: list,
    predictor,
    gpu_idx: int,
) -> tuple:
    """
    SAM3 视频追踪模式，对一个视频的帧目录做多物体追踪。

    返回
    ----
    label_maps : (T, H, W) int16
    masks      : (N_obj, T, H, W) bool
    """
    frame_paths = sorted([
        os.path.join(frames_dir, f)
        for f in os.listdir(frames_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    T = len(frame_paths)
    if T == 0:
        raise RuntimeError(f"帧目录无有效帧: {frames_dir}")

    first = cv2.imread(frame_paths[0])
    H, W = first.shape[:2]
    N_obj = len(object_names)
    label_maps = np.zeros((T, H, W), dtype=np.int16)

    resp = predictor.handle_request(dict(type="start_session", resource_path=frames_dir))
    sid = resp["session_id"]

    for cidx, concept in enumerate(object_names, start=1):
        print(f"  [SAM3-Video] concept [{cidx}/{N_obj}]: {concept!r}", flush=True)
        predictor.handle_request(dict(type="reset_session", session_id=sid))
        predictor.handle_request(dict(
            type="add_prompt", session_id=sid, frame_index=0, text=concept,
        ))
        pfo = {}
        for r in predictor.handle_stream_request(
            dict(type="propagate_in_video", session_id=sid)
        ):
            pfo[r["frame_index"]] = r["outputs"]

        for t in range(T):
            out = pfo.get(t)
            if out is None:
                continue
            for _oid, mask_t in zip(out.get("out_obj_ids", []),
                                    out.get("out_binary_masks", [])):
                if isinstance(mask_t, torch.Tensor):
                    mask_np = mask_t.cpu().numpy()
                else:
                    mask_np = np.asarray(mask_t)
                mask_np = mask_np.squeeze()
                if mask_np.ndim == 0:
                    continue
                if mask_np.ndim == 3:
                    mask_np = mask_np[0]
                mask_np = mask_np.astype(bool)
                if mask_np.shape != (H, W):
                    mask_np = cv2.resize(
                        mask_np.astype(np.uint8), (W, H),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                free = label_maps[t] == 0
                label_maps[t][free & mask_np] = cidx

    predictor.handle_request(dict(type="close_session", session_id=sid))

    # 从 label_maps 反推 per-object masks，兼容原有输出格式
    masks = np.zeros((N_obj, T, H, W), dtype=bool)
    for i in range(N_obj):
        masks[i] = (label_maps == i + 1)

    cov = (label_maps > 0).mean()
    print(f"  [SAM3-Video] 完成  coverage={cov:.1%}", flush=True)
    return label_maps, masks


# ── masks → mean_areas ────────────────────────────────────────────────────────

def _compute_mean_areas(masks: np.ndarray) -> np.ndarray:
    """masks (N_obj, T, H, W) bool → mean_areas (N_obj,) float"""
    if masks.shape[0] == 0:
        return np.array([])
    N_obj, T, H, W = masks.shape
    pixel_total = H * W
    return masks.sum(axis=(2, 3)).mean(axis=1) / pixel_total


# ── 单条处理 ─────────────────────────────────────────────────────────────────

def _process_one(
    entry: dict,
    qwen_model,
    qwen_processor,
    predictor,
    qwen_device: str,
    sam3_gpu: int,
    tmp_root: str,
) -> None:
    """处理 batch_manifest 中的一条记录。"""
    video_path  = entry["video_path"]
    out_masks   = Path(entry["output_masks_npz"])
    out_objects = Path(entry["output_objects_json"])
    out_labels  = Path(entry["output_label_maps_npz"])
    ref_objects = entry.get("ref_objects_json")
    max_frames  = entry.get("max_frames", 0)

    def _valid(p: Path) -> bool:
        return p.exists() and p.stat().st_size > 64

    if _valid(out_masks) and _valid(out_objects) and _valid(out_labels):
        print(f"  [跳过] 已存在: {out_masks.name}")
        return

    print(f"\n  [处理] {Path(video_path).name}")
    for p in [out_masks.parent, out_objects.parent, out_labels.parent]:
        p.mkdir(parents=True, exist_ok=True)

    # 提取帧到磁盘（视频预测器需要帧目录）
    frames_dir = os.path.join(tmp_root, Path(video_path).stem + "_frames")
    n_frames = _extract_frames_to_dir(video_path, frames_dir, max_frames)
    if n_frames == 0:
        raise RuntimeError(f"视频无有效帧: {video_path}")

    first_frame_path = sorted([
        os.path.join(frames_dir, f)
        for f in os.listdir(frames_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])[0]
    first_bgr = cv2.imread(first_frame_path)
    H, W = first_bgr.shape[:2]
    T = n_frames
    print(f"  帧数: {T}  分辨率: {H}×{W}")

    # 物体识别
    if ref_objects is not None and Path(ref_objects).exists():
        with open(ref_objects, "r") as f:
            obj_data = json.load(f)
        if isinstance(obj_data, dict):
            object_names = obj_data.get("objects", obj_data.get("object_names", []))
        else:
            object_names = obj_data
        print(f"  [物体列表] 复用 {Path(ref_objects).name}: {object_names}")
    else:
        object_names = _identify_objects(
            first_frame_path, qwen_device, qwen_model, qwen_processor,
        )

    if not object_names:
        print(f"  [警告] 未识别到物体，保存空 masks")
        np.savez_compressed(str(out_masks),
                            object_names=np.array([], dtype=object),
                            masks=np.zeros((0, T, H, W), dtype=bool),
                            mean_areas=np.array([]))
        with open(str(out_objects), "w") as f:
            json.dump({"objects": [], "video": video_path}, f, ensure_ascii=False)
        np.savez_compressed(str(out_labels),
                            label_maps=np.zeros((T, H, W), dtype=np.int16),
                            object_names=np.array([], dtype=object))
        return

    with open(str(out_objects), "w") as f:
        json.dump({"objects": object_names, "video": video_path}, f,
                  ensure_ascii=False, indent=2)

    label_maps, masks = _track_video(frames_dir, object_names, predictor, sam3_gpu)
    mean_areas = _compute_mean_areas(masks)

    np.savez_compressed(
        str(out_masks),
        object_names=np.array(object_names, dtype=object),
        masks=masks,
        mean_areas=mean_areas,
    )

    np.savez_compressed(
        str(out_labels),
        label_maps=label_maps,
        object_names=np.array(object_names, dtype=object),
    )

    print(f"  [完成] {Path(video_path).name}  "
          f"物体数: {len(object_names)}  覆盖率: {(label_maps > 0).mean():.1%}")


# ── 主函数 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SAM3 batch segmentation worker (video mode)")
    parser.add_argument("--batch_manifest", required=True, help="batch manifest JSON path")
    parser.add_argument("--sam3_gpu",  type=int, default=0)
    parser.add_argument("--qwen_gpu",  type=int, default=0)
    args = parser.parse_args()

    with open(args.batch_manifest, "r") as f:
        entries = json.load(f)

    print(f"[worker_sam3] total {len(entries)} videos  (video-tracking mode)")

    # ── Pass 1: Qwen 物体识别（仅 GT 视频）─────────────────────────────────
    need_qwen_entries = [e for e in entries
                         if not e.get("ref_objects_json") or
                         not Path(e.get("ref_objects_json", "")).exists()]

    if need_qwen_entries:
        print(f"\n[worker_sam3] Pass 1: Qwen object ID  "
              f"device=cuda:{args.qwen_gpu}  ({len(need_qwen_entries)} GT videos)")
        from transformers import AutoProcessor, AutoModelForImageTextToText
        qwen_processor = AutoProcessor.from_pretrained(QWEN_MODEL, trust_remote_code=True)
        qwen_model = AutoModelForImageTextToText.from_pretrained(
            QWEN_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True,
        ).to(f"cuda:{args.qwen_gpu}").eval()
        print(f"[worker_sam3] Qwen3-VL loaded")

        with tempfile.TemporaryDirectory(prefix="worker_sam3_qwen_") as tmp_root:
            for entry in need_qwen_entries:
                out_objects = Path(entry["output_objects_json"])
                if out_objects.exists() and out_objects.stat().st_size > 64:
                    print(f"  [skip] objects already exist: {out_objects.name}")
                    continue
                try:
                    frames_dir = os.path.join(tmp_root,
                                              Path(entry["video_path"]).stem + "_frames")
                    n = _extract_frames_to_dir(entry["video_path"], frames_dir,
                                               entry.get("max_frames", 0))
                    if n == 0:
                        raise RuntimeError("no frames extracted")
                    first_frame = sorted([
                        os.path.join(frames_dir, f)
                        for f in os.listdir(frames_dir)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))
                    ])[0]
                    object_names = _identify_objects(
                        first_frame, f"cuda:{args.qwen_gpu}",
                        qwen_model, qwen_processor,
                    )
                except Exception as exc:
                    import traceback
                    print(f"  [error] Qwen failed for {Path(entry['video_path']).name}: {exc}")
                    traceback.print_exc()
                    object_names = []
                out_objects.parent.mkdir(parents=True, exist_ok=True)
                with open(str(out_objects), "w") as f:
                    json.dump({"objects": object_names, "video": entry["video_path"]},
                              f, ensure_ascii=False, indent=2)
                print(f"  Qwen done: {out_objects.name}  objects={object_names}")

        del qwen_model, qwen_processor
        gc.collect()
        torch.cuda.empty_cache()
        print("[worker_sam3] Qwen unloaded, VRAM released")
    else:
        print("[worker_sam3] Pass 1: skipped (all entries have ref_objects_json)")
        qwen_model = qwen_processor = None

    # ref_objects_json が未解決のエントリを自分の objects_json に向ける
    for e in entries:
        if not e.get("ref_objects_json") or \
                not Path(e.get("ref_objects_json", "")).exists():
            e["ref_objects_json"] = e["output_objects_json"]

    # ── Pass 2: SAM3 视频追踪 ─────────────────────────────────────────────
    print(f"\n[worker_sam3] Pass 2: SAM3 video tracking  device=cuda:{args.sam3_gpu}")
    from sam3.model_builder import build_sam3_video_predictor

    predictor = build_sam3_video_predictor(
        checkpoint_path=SAM3_CKPT,
        bpe_path=SAM3_BPE,
        gpus_to_use=[args.sam3_gpu],
    )
    print(f"[worker_sam3] SAM3 video predictor loaded")

    n_ok, n_err = 0, 0
    with tempfile.TemporaryDirectory(prefix="worker_sam3_frames_") as tmp_root:
        for i, entry in enumerate(entries):
            print(f"\n{'='*55}")
            print(f"[{i+1}/{len(entries)}] {Path(entry['video_path']).name}")
            try:
                _process_one(
                    entry,
                    qwen_model=None,
                    qwen_processor=None,
                    predictor=predictor,
                    qwen_device=f"cuda:{args.qwen_gpu}",
                    sam3_gpu=args.sam3_gpu,
                    tmp_root=tmp_root,
                )
                n_ok += 1
            except Exception as e:
                import traceback
                print(f"  [error] {e}")
                traceback.print_exc()
                n_err += 1

    predictor.shutdown()
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n[worker_sam3] done: ok={n_ok}  err={n_err}")


if __name__ == "__main__":
    main()
