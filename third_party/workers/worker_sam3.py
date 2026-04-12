#!/usr/bin/env python3
"""
worker_sam3.py
==============
SAM3 批量 worker —— 在 SAM3 conda 环境中运行，一次性处理所有视频。

设计原则：Qwen + SAM3 模型各加载一次，顺序处理 batch_manifest 中的所有条目。
GT 视频先跑 Qwen 识别物体，pred 视频复用 GT 的物体列表（跳过 Qwen）。

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


# ── 帧提取 ────────────────────────────────────────────────────────────────────

def _extract_frames_bgr(video_path: str, max_frames: int = 0) -> list:
    """从视频提取所有帧（BGR numpy），返回 list of (H, W, 3) uint8。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if max_frames > 0 and len(frames) >= max_frames:
            break
    cap.release()
    return frames


# ── Qwen 物体识别 ─────────────────────────────────────────────────────────────

def _identify_objects(first_frame_bgr: np.ndarray, device: str, qwen_model, qwen_processor) -> list:
    """用 Qwen3-VL 识别首帧中的显著物体，返回物体名称列表。"""
    from qwen_vl_utils import process_vision_info

    pil_img = Image.fromarray(cv2.cvtColor(first_frame_bgr, cv2.COLOR_BGR2RGB))

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


# ── SAM3 分割 ─────────────────────────────────────────────────────────────────

def _segment_video(
    frames_bgr: list,
    object_names: list,
    sam3_processor,
    device: str,
) -> tuple:
    """
    对所有帧按物体列表做 SAM3 分割（逐帧精确推理）。

    返回
    ----
    masks     : (N_obj, T, H, W) bool
    mean_areas: (N_obj,) float
    """
    import copy

    T = len(frames_bgr)
    if T == 0:
        return np.zeros((0, 0, 0, 0), dtype=bool), np.array([])

    H, W = frames_bgr[0].shape[:2]
    N_obj = len(object_names)
    all_masks = np.zeros((N_obj, T, H, W), dtype=bool)

    for t, frame_bgr in enumerate(frames_bgr):
        pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        state_base = sam3_processor.set_image(pil_img)

        for i, obj_name in enumerate(object_names):
            state = copy.copy(state_base)
            state = sam3_processor.set_text_prompt(state=state, prompt=obj_name)

            masks_t = state.get("masks")
            scores_t = state.get("scores")

            if masks_t is None or masks_t.shape[0] == 0:
                continue

            best_idx = int(scores_t.argmax())
            score = float(scores_t[best_idx])
            if score < SAM3_CONF:
                continue

            mask_np = masks_t[best_idx, 0].cpu().numpy().astype(np.uint8)
            if mask_np.shape != (H, W):
                mask_np = cv2.resize(mask_np, (W, H), interpolation=cv2.INTER_NEAREST)
            all_masks[i, t] = mask_np.astype(bool)

        if (t + 1) % 10 == 0 or t == T - 1:
            print(f"  [SAM3] 帧 {t+1}/{T} 完成")

    pixel_total = H * W
    mean_areas = all_masks.sum(axis=(2, 3)).mean(axis=1) / pixel_total
    return all_masks, mean_areas


# ── masks → label_maps ────────────────────────────────────────────────────────

def masks_to_label_maps(masks: np.ndarray) -> np.ndarray:
    """
    将 per-object masks (N_obj, T, H, W) bool 转换为 label_maps (T, H, W) int16。
    优先级：object index 越小优先级越高（先出现的物体不被后出现的覆盖）。

    0 = 背景，1~K = 物体类别（对应 object_names[class_id - 1]）
    """
    N_obj, T, H, W = masks.shape
    label_maps = np.zeros((T, H, W), dtype=np.int16)
    for i in range(N_obj - 1, -1, -1):  # 逆序，低 index 优先
        label_maps[masks[i]] = i + 1
    return label_maps


# ── 单条处理 ─────────────────────────────────────────────────────────────────

def _process_one(entry: dict, qwen_model, qwen_processor, sam3_processor, qwen_device: str,
                 sam3_device: str = "cuda:0") -> None:
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

    frames_bgr = _extract_frames_bgr(video_path, max_frames)
    T = len(frames_bgr)
    if T == 0:
        raise RuntimeError(f"视频无有效帧: {video_path}")
    H, W = frames_bgr[0].shape[:2]
    print(f"  抽帧: {T} 帧  分辨率: {H}×{W}")

    if ref_objects is not None and Path(ref_objects).exists():
        with open(ref_objects, "r") as f:
            obj_data = json.load(f)
        if isinstance(obj_data, dict):
            object_names = obj_data.get("objects", obj_data.get("object_names", []))
        else:
            object_names = obj_data
        print(f"  [物体列表] 复用 {Path(ref_objects).name}: {object_names}")
    else:
        object_names = _identify_objects(frames_bgr[0], qwen_device, qwen_model, qwen_processor)

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

    masks, mean_areas = _segment_video(frames_bgr, object_names, sam3_processor, device=sam3_device)

    np.savez_compressed(
        str(out_masks),
        object_names=np.array(object_names, dtype=object),
        masks=masks,
        mean_areas=mean_areas,
    )

    label_maps = masks_to_label_maps(masks)
    np.savez_compressed(
        str(out_labels),
        label_maps=label_maps,
        object_names=np.array(object_names, dtype=object),
    )

    print(f"  [完成] {Path(video_path).name}  "
          f"物体数: {len(object_names)}  覆盖率: {(label_maps > 0).mean():.1%}")


# ── 主函数 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SAM3 batch segmentation worker")
    parser.add_argument("--batch_manifest", required=True, help="batch manifest JSON path")
    parser.add_argument("--sam3_gpu",  type=int, default=0)
    parser.add_argument("--qwen_gpu",  type=int, default=0)
    args = parser.parse_args()

    import gc

    with open(args.batch_manifest, "r") as f:
        entries = json.load(f)

    print(f"[worker_sam3] total {len(entries)} videos")

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

        for entry in need_qwen_entries:
            out_objects = Path(entry["output_objects_json"])
            if out_objects.exists() and out_objects.stat().st_size > 64:
                print(f"  [skip] objects already exist: {out_objects.name}")
                continue
            try:
                frames_bgr = _extract_frames_bgr(entry["video_path"],
                                                 entry.get("max_frames", 0))
                object_names = (
                    _identify_objects(frames_bgr[0], f"cuda:{args.qwen_gpu}",
                                      qwen_model, qwen_processor)
                    if frames_bgr else []
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

    for e in entries:
        if not e.get("ref_objects_json") or \
                not Path(e.get("ref_objects_json", "")).exists():
            e["ref_objects_json"] = e["output_objects_json"]

    print(f"\n[worker_sam3] Pass 2: SAM3 segmentation  device=cuda:{args.sam3_gpu}")
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    sam3_model = build_sam3_image_model(
        bpe_path=SAM3_BPE,
        device=f"cuda:{args.sam3_gpu}",
        eval_mode=True,
        checkpoint_path=SAM3_CKPT,
        load_from_HF=False,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        compile=False,
    ).to(f"cuda:{args.sam3_gpu}").eval()
    sam3_processor = Sam3Processor(sam3_model, device=f"cuda:{args.sam3_gpu}",
                                   confidence_threshold=SAM3_CONF)
    print(f"[worker_sam3] SAM3 loaded")

    n_ok, n_err = 0, 0
    for i, entry in enumerate(entries):
        print(f"\n{'='*55}")
        print(f"[{i+1}/{len(entries)}] {Path(entry['video_path']).name}")
        try:
            _process_one(entry, None, None, sam3_processor,
                         qwen_device=f"cuda:{args.qwen_gpu}",
                         sam3_device=f"cuda:{args.sam3_gpu}")
            n_ok += 1
        except Exception as e:
            import traceback
            print(f"  [error] {e}")
            traceback.print_exc()
            n_err += 1

    del sam3_model, sam3_processor
    torch.cuda.empty_cache()

    print(f"\n[worker_sam3] done: ok={n_ok}  err={n_err}")


if __name__ == "__main__":
    main()
