"""
Step: Qwen-VL + SAM3 Semantic Segmentation
============================================
Conda env: SAM3

对视频帧进行语义分割：
  1. Qwen3-VL 识别首帧物体
  2. SAM3 逐帧分割所有物体
  3. 输出 label_maps.npz (T, H, W) int16

Usage:
    conda run -n SAM3 python step_qwen_sam3.py \
        --video_frames_dir /path/to/frames/ \
        --output /path/to/label_maps.npz \
        --objects_json /path/to/objects.json \
        --gpu 0
"""

import argparse
import copy
import gc
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

_REWARD_DIR = Path(__file__).resolve().parent.parent
_RL_TRAIN_DIR = _REWARD_DIR.parent
_RL_CODE_DIR = _RL_TRAIN_DIR.parent
_THIRD_PARTY_DIR = _RL_CODE_DIR / "third_party"

_SAM3_SRC = _THIRD_PARTY_DIR / "repos" / "SAM3" / "sam3"
_SAM3_PKG = _THIRD_PARTY_DIR / "repos" / "SAM3"
for _p in [str(_SAM3_SRC), str(_SAM3_PKG)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

_MODEL_DIR = Path(os.environ.get("RL_MODEL_ROOT", str(_RL_CODE_DIR.parent / "RL" / "model")))
QWEN_MODEL = str(_MODEL_DIR / "Qwen3-VL-8B-Instruct")
SAM3_CKPT = str(_MODEL_DIR / "SAM3" / "sam3.pt")
SAM3_BPE = str(_SAM3_PKG / "sam3" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz")
SAM3_CONF = 0.5


# ═══════════════════════ Qwen Object Identification ═══════════════════════


def identify_objects(first_frame_path: str, device: str) -> list:
    """用 Qwen3-VL 识别首帧中的显著物体，返回物体名称列表。"""
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
            {"type": "text", "text": user_text},
        ]},
    ]

    try:
        from qwen_vl_utils import process_vision_info
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(device)
    except Exception:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text], images=[pil_img], padding=True, return_tensors="pt",
        ).to(device)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=256)
    input_len = inputs["input_ids"].shape[1]
    raw = processor.batch_decode(
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
    print(f"[Qwen] 识别到 {len(objs)} 个物体: {objs}")

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return objs


# ═══════════════════════ SAM3 Segmentation ═══════════════════════


def segment_frames(
    frame_paths: list, object_names: list, device: str,
) -> tuple:
    """
    SAM3 逐帧分割，返回 (masks, label_maps)。

    masks      : (N_obj, T, H, W) bool
    label_maps : (T, H, W) int16
    """
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    sam3_model = build_sam3_image_model(
        bpe_path=SAM3_BPE,
        device=device,
        eval_mode=True,
        checkpoint_path=SAM3_CKPT,
        load_from_HF=False,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        compile=False,
    ).to(device).eval()
    sam3_processor = Sam3Processor(sam3_model, device=device,
                                   confidence_threshold=SAM3_CONF)

    T = len(frame_paths)
    first_img = Image.open(frame_paths[0]).convert("RGB")
    W, H = first_img.size
    N_obj = len(object_names)
    all_masks = np.zeros((N_obj, T, H, W), dtype=bool)

    for t, fp in enumerate(frame_paths):
        pil_img = Image.open(fp).convert("RGB")
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
            print(f"[SAM3] 帧 {t+1}/{T} 完成")

    label_maps = np.zeros((T, H, W), dtype=np.int16)
    for i in range(N_obj - 1, -1, -1):
        label_maps[all_masks[i]] = i + 1

    del sam3_model, sam3_processor
    gc.collect()
    torch.cuda.empty_cache()

    return all_masks, label_maps


# ═══════════════════════ Main ═══════════════════════


def main():
    parser = argparse.ArgumentParser(description="Qwen-VL + SAM3 semantic segmentation")
    parser.add_argument("--video_frames_dir", required=True)
    parser.add_argument("--output", required=True, help="Output label_maps .npz path")
    parser.add_argument("--objects_json", default=None,
                        help="Pre-computed objects JSON (skip Qwen if provided)")
    parser.add_argument("--objects_output", default=None,
                        help="Save identified objects to this JSON path")
    parser.add_argument("--masks_output", default=None,
                        help="Optionally save raw masks .npz")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"

    frame_paths = sorted([
        os.path.join(args.video_frames_dir, f)
        for f in os.listdir(args.video_frames_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    if not frame_paths:
        raise FileNotFoundError(f"No image frames in {args.video_frames_dir}")
    print(f"[step_qwen_sam3] Found {len(frame_paths)} frames")

    if args.objects_json and os.path.exists(args.objects_json):
        with open(args.objects_json, "r") as f:
            obj_data = json.load(f)
        if isinstance(obj_data, dict):
            object_names = obj_data.get("objects", obj_data.get("object_names", []))
        else:
            object_names = obj_data
        print(f"[step_qwen_sam3] Loaded objects from {args.objects_json}: {object_names}")
    else:
        object_names = identify_objects(frame_paths[0], device)

    if args.objects_output:
        os.makedirs(os.path.dirname(os.path.abspath(args.objects_output)), exist_ok=True)
        with open(args.objects_output, "w") as f:
            json.dump({"objects": object_names}, f, ensure_ascii=False, indent=2)

    if not object_names:
        print("[step_qwen_sam3] 未识别到物体，保存空 label_maps")
        T = len(frame_paths)
        first_img = Image.open(frame_paths[0]).convert("RGB")
        W, H = first_img.size
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        np.savez_compressed(args.output,
                            label_maps=np.zeros((T, H, W), dtype=np.int16),
                            object_names=np.array([], dtype=object))
        return

    masks, label_maps = segment_frames(frame_paths, object_names, device)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(
        args.output,
        label_maps=label_maps,
        object_names=np.array(object_names, dtype=object),
    )
    print(f"[step_qwen_sam3] Saved label_maps to {args.output}  "
          f"物体数: {len(object_names)}  覆盖率: {(label_maps > 0).mean():.1%}")

    if args.masks_output:
        os.makedirs(os.path.dirname(os.path.abspath(args.masks_output)), exist_ok=True)
        pixel_total = float(label_maps.shape[1] * label_maps.shape[2])
        mean_areas = masks.sum(axis=(2, 3)).mean(axis=1) / pixel_total
        np.savez_compressed(
            args.masks_output,
            object_names=np.array(object_names, dtype=object),
            masks=masks,
            mean_areas=mean_areas,
        )


if __name__ == "__main__":
    main()
