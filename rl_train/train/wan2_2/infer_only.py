"""infer_only.py — Wan2.2-Fun-5B-Control-Camera 推理调试脚本。

用我们 wan2_2 框架完整链路（model_loader + wan22_encode + grpo_core.run_sample_step）
跑端到端推理，输出格式与 gen3r 完全一致：
    <output_dir>/<dataset>/<sample_id>/
        infer_info.json
        camera.txt           (从 GT 复制)
        gen_0.mp4 ... gen_N.mp4

用法：
    方式 A：批量
        python infer_only.py --data_root /path --datasets dl3dv \\
            --output_dir ./out --num_rollouts 4 --max_samples 2

    方式 B：单样本
        python infer_only.py --sample_dir /path/to/sample \\
            --output_dir ./out --num_rollouts 4

为了让同一样本不同 rollout 视觉上有差异，seed = seed_base + k * 1000。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config import WAN22_DEFAULT_NEG_PROMPT  # noqa: E402
from paths import default_wan22_config_path  # noqa: E402
from dataset_rl import (get_video_total_frames, load_frames_from_dir,  # noqa: E402
                        load_frames_from_video, parse_camera_txt,
                        sample_frame_indices)
from grpo_core import run_sample_step, sd3_time_shift  # noqa: E402
from model_loader import load_all_models  # noqa: E402
from wan22_encode import (build_camera_control, chunk_camera_control,  # noqa: E402
                          decode_rgb_video, encode_inpaint_conditions,
                          encode_text, transformer_forward)


# ══════════════════════════════════════════════════════════════════════════════
# 样本收集
# ══════════════════════════════════════════════════════════════════════════════

def collect_samples(args) -> list[dict]:
    samples = []

    if args.sample_dir:
        # 支持多个 --sample_dir，避免每次重新 load 22GB 模型
        sample_dirs = args.sample_dir if isinstance(args.sample_dir, list) else [args.sample_dir]
        for sd in sample_dirs:
            sample_dir = Path(sd)
            meta_path = sample_dir / "metadata.json"
            cam_path = sample_dir / "camera.txt"
            if not cam_path.exists():
                raise FileNotFoundError(f"camera.txt not found in {sample_dir}")

            meta = {}
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)

            dataset_name = meta.get("dataset", sample_dir.parent.name)
            sample_id = sample_dir.name

            media = str(sample_dir / "gt.mp4") if (sample_dir / "gt.mp4").exists() \
                else str(sample_dir / "frames")

            samples.append({
                "sample_id": sample_id,
                "dataset_name": dataset_name,
                "camera_txt": str(cam_path),
                "media": media,
                "caption": meta.get("caption", meta.get("prompt", "")) or "camera moving through a scene",
            })
    else:
        dataset_list = [d.strip() for d in args.datasets.split(",") if d.strip()]
        count = 0
        for ds_name in dataset_list:
            ds_dir = Path(args.data_root) / ds_name
            if not ds_dir.exists():
                print(f"[Infer] WARNING: {ds_dir} not found, skipping")
                continue
            for sample_dir in sorted(ds_dir.iterdir()):
                if not sample_dir.is_dir():
                    continue
                cam_path = sample_dir / "camera.txt"
                meta_path = sample_dir / "metadata.json"
                if not cam_path.exists():
                    continue
                meta = {}
                if meta_path.exists():
                    with open(meta_path) as f:
                        meta = json.load(f)

                frame_mode = getattr(args, "frame_mode", "video")
                media = str(sample_dir / "gt.mp4") if frame_mode == "video" \
                    else str(sample_dir / "frames")
                if not Path(media).exists():
                    media = str(sample_dir / "gt.mp4")
                if not Path(media).exists():
                    continue

                samples.append({
                    "sample_id": sample_dir.name,
                    "dataset_name": ds_name,
                    "camera_txt": str(cam_path),
                    "media": media,
                    "caption": meta.get("caption", meta.get("prompt", "")) or "camera moving through a scene",
                })
                count += 1
                if args.max_samples > 0 and count >= args.max_samples:
                    break
            if args.max_samples > 0 and count >= args.max_samples:
                break

    print(f"[Infer] Collected {len(samples)} samples")
    return samples


# ══════════════════════════════════════════════════════════════════════════════
# 单样本推理
# ══════════════════════════════════════════════════════════════════════════════

def infer_one_sample(args, models, sample_info: dict, device) -> str:
    tokenizer = models["tokenizer"]
    text_encoder = models["text_encoder"]
    vae = models["vae"]
    transformer = models["transformer"]

    weight_dtype = torch.bfloat16

    sample_id = sample_info["sample_id"]
    dataset_name = sample_info["dataset_name"]
    camera_txt = sample_info["camera_txt"]
    media = sample_info["media"]
    caption = sample_info["caption"]

    out_dir = os.path.join(args.output_dir, dataset_name, sample_id)
    os.makedirs(out_dir, exist_ok=True)

    if args.skip_done and (Path(out_dir) / "infer_info.json").exists():
        existing = list(Path(out_dir).glob("gen_*.mp4"))
        if len(existing) >= args.num_rollouts:
            print(f"[Infer] Skip {dataset_name}/{sample_id} (already done)")
            return out_dir

    H, W = args.resolution_h, args.resolution_w

    # ── 解析相机 ─────────────────────────────────────────────────────────────
    c2ws, Ks = parse_camera_txt(camera_txt, H, W)
    total_cam = c2ws.shape[0]

    # ── 采样帧 ────────────────────────────────────────────────────────────────
    if media.endswith(".mp4"):
        total_frames = get_video_total_frames(media)
    else:
        total_frames = len([
            f for f in Path(media).iterdir()
            if f.suffix.lower() in (".png", ".jpg", ".jpeg")
        ])

    total_avail = min(total_frames, total_cam)
    # 推理时固定从 0 开始，连续 num_frames 帧（可复现，便于 reward 比对）
    indices = list(range(0, min(args.num_frames * args.frame_stride, total_avail), args.frame_stride))
    if len(indices) < args.num_frames:
        indices = indices + [indices[-1]] * (args.num_frames - len(indices))
    indices = indices[: args.num_frames]

    if media.endswith(".mp4"):
        pixel_values = load_frames_from_video(media, indices, H, W)
    else:
        pixel_values = load_frames_from_dir(media, indices, H, W)

    pixel_values = pixel_values.to(device)
    c2ws_s = c2ws[indices].to(device)
    Ks_s = Ks[indices].to(device)
    F = pixel_values.shape[0]

    # ── 文本编码（或加载预算 cache） ─────────────────────────────────────────
    if tokenizer is not None and text_encoder is not None:
        text_encoder.to(device=device)
        with torch.no_grad():
            prompt_embed = encode_text(caption, tokenizer, text_encoder, device,
                                       max_length=args.tokenizer_max_length).to(weight_dtype)
            neg_embed = encode_text(WAN22_DEFAULT_NEG_PROMPT, tokenizer, text_encoder, device,
                                    max_length=args.tokenizer_max_length).to(weight_dtype)
        # T5 用完直接 offload
        text_encoder.cpu()
        torch.cuda.empty_cache()
    else:
        from model_loader import load_t5_embeds
        prompt_cpu, neg_cpu = load_t5_embeds(args, sample_id, dataset_name)
        prompt_embed = prompt_cpu.to(device=device, dtype=weight_dtype)
        neg_embed = neg_cpu.to(device=device, dtype=weight_dtype)

    prompt_embeds = [prompt_embed]
    neg_embeds = [neg_embed]

    # ── 控制条件 ──────────────────────────────────────────────────────────────
    vae.to(device=device, dtype=weight_dtype)
    with torch.no_grad():
        control_latents, masked_video_lat, latent_mask = encode_inpaint_conditions(
            pixel_values[0], vae, F, H, W, device, weight_dtype,
        )
        plucker_fchw = build_camera_control(c2ws_s, Ks_s, H, W, device, weight_dtype)
        control_camera_latents = chunk_camera_control(plucker_fchw, F)

    # seq_len
    SPATIAL_DS = 16
    TEMPORAL_DS = 4
    IN_CHANNELS = 48
    latent_t = ((F - 1) // TEMPORAL_DS) + 1
    latent_h = H // SPATIAL_DS
    latent_w = W // SPATIAL_DS
    patch = transformer.config.patch_size if hasattr(transformer, "config") else (1, 2, 2)
    patch_h, patch_w = patch[1], patch[2]
    seq_len = math.ceil((latent_h * latent_w) / (patch_h * patch_w) * latent_t)

    sigma_schedule = sd3_time_shift(
        args.shift, torch.linspace(1, 0, args.sampling_steps + 1)
    )

    # ── 生成 num_rollouts 条 rollout ────────────────────────────────────────
    transformer.to(device=device, dtype=weight_dtype)
    transformer.eval()

    seed_base = args.seed_base
    for k in range(args.num_rollouts):
        seed_k = seed_base + k * 1000
        torch.manual_seed(seed_k)
        torch.cuda.manual_seed_all(seed_k)
        gen = torch.Generator(device=device).manual_seed(seed_k)

        z0 = torch.randn(
            (1, IN_CHANNELS, latent_t, latent_h, latent_w),
            generator=gen, device=device, dtype=weight_dtype,
        )

        t0 = time.time()
        with torch.no_grad():
            final_z, _pred_x0, _, _ = run_sample_step(
                args, z0, sigma_schedule, transformer,
                prompt_embeds, neg_embeds, seq_len,
                control_latents, control_camera_latents,
                masked_video_lat, latent_mask,
                transformer_forward,
            )

        # 解码 final_z（已 mask_clamp，等价于 official pipeline 的 `latents`）。
        # **不要**用 pred_x0 — pred_x0 = latents - sigma*model_output 没有 mask_clamp，
        # 首帧颜色会出错，整体也会发生漂移。
        video_path = os.path.join(out_dir, f"gen_{k}.mp4")
        decode_rgb_video(final_z, vae, video_path, fps=16)
        elapsed = time.time() - t0
        print(f"[Infer] Saved {dataset_name}/{sample_id}/gen_{k}.mp4 "
              f"(seed={seed_k}, took {elapsed:.1f}s)")

        del z0, _pred_x0, final_z
        torch.cuda.empty_cache()

    # ── 复制 GT camera.txt ──────────────────────────────────────────────────
    shutil.copy2(camera_txt, os.path.join(out_dir, "camera.txt"))

    # ── infer_info.json ─────────────────────────────────────────────────────
    infer_info = {
        "dataset": dataset_name,
        "sample_id": sample_id,
        "num_rollouts": args.num_rollouts,
        "num_frames": F,
        "resolution_h": H,
        "resolution_w": W,
        "sampling_steps": args.sampling_steps,
        "eta": args.eta,
        "cfg": args.cfg_infer,
        "shift": args.shift,
        "seed_base": seed_base,
        "caption": caption,
        "negative_prompt": WAN22_DEFAULT_NEG_PROMPT,
    }
    with open(os.path.join(out_dir, "infer_info.json"), "w") as f:
        json.dump(infer_info, f, indent=2, ensure_ascii=False)

    return out_dir


# ══════════════════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════════════════

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = OmegaConf.load(args.config_path)
    weight_dtype = torch.bfloat16
    models = load_all_models(args, config, device, weight_dtype)
    models["transformer"].eval()

    samples = collect_samples(args)
    if not samples:
        print("[Infer] No samples found.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    for i, sample_info in enumerate(samples):
        print(f"\n[Infer] === Sample {i+1}/{len(samples)}: "
              f"{sample_info['dataset_name']}/{sample_info['sample_id']} ===")
        try:
            infer_one_sample(args, models, sample_info, device)
        except Exception as e:
            print(f"[Infer] ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n[Infer] Done. Output at: {args.output_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Wan2.2-Fun-5B-Control-Camera 推理调试脚本")

    # 模型
    p.add_argument("--pretrained_model_path", type=str,
                   default="/mnt/afs/visitor16/rl_train_new/model/Wan2.2-Fun-5B-Control-Camera")
    p.add_argument("--config_path", type=str,
                   default=default_wan22_config_path())
    p.add_argument("--transformer_path", type=str, default=None)
    p.add_argument("--t5_embed_dir", type=str, default=None,
                   help="若设置则跳过加载 T5 模型")

    # 数据输入（二选一）
    p.add_argument("--sample_dir", type=str, default=None, nargs="+",
                   help="[方式B] 一个或多个样本目录（可重复传，模型只加载一次）")
    p.add_argument("--data_root", type=str, default=None,
                   help="[方式A] 数据根目录")
    p.add_argument("--datasets", type=str, default="dl3dv")
    p.add_argument("--max_samples", type=int, default=5)
    p.add_argument("--frame_mode", type=str, default="video", choices=["video", "frames"])

    # 输出
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--skip_done", action="store_true", default=True)
    p.add_argument("--no_skip_done", dest="skip_done", action="store_false")

    # 推理参数（与 config.py 命名一致）
    p.add_argument("--num_rollouts", type=int, default=4)
    p.add_argument("--num_frames", type=int, default=49)
    p.add_argument("--frame_stride", type=int, default=1)
    p.add_argument("--resolution_h", type=int, default=704)
    p.add_argument("--resolution_w", type=int, default=1280)
    p.add_argument("--sampling_steps", type=int, default=50)
    p.add_argument("--eta", type=float, default=0.0,
                   help="推理时设为 0 = ODE")
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--cfg_infer", type=float, default=6.0)
    p.add_argument("--cfg_rollout", type=float, default=-1.0)
    p.add_argument("--tokenizer_max_length", type=int, default=512)
    p.add_argument("--seed_base", type=int, default=42,
                   help="rollout k 用 seed = seed_base + k * 1000，保证不同 rollout 噪声不同")

    # GRPO 相关字段（run_sample_step 需要）—— 推理用默认值
    p.add_argument("--train_timestep_strategy", type=str, default="random",
                   choices=["front", "random"])
    p.add_argument("--sde_fraction", type=float, default=1.0)
    p.add_argument("--init_same_noise", action="store_true")

    args = p.parse_args()

    if args.sample_dir is None and args.data_root is None:
        p.error("需要提供 --sample_dir 或 --data_root")

    if args.cfg_rollout < 0:
        args.cfg_rollout = args.cfg_infer

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
