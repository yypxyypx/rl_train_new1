"""infer_only.py — 推理调试脚本。

用于调试 Gen3R 推理是否正确，不涉及训练。
输出格式与 batch_reward.sh 的输入格式完全兼容，推理完可直接跑 reward。

使用方式：

方式 A: 指定数据根目录（批量推理 N 个样本）
    bash run_infer.sh \\
        --data_root /path/to/processed \\
        --datasets re10k,dl3dv \\
        --max_samples 5

方式 B: 指定单个样本目录（精确调试）
    bash run_infer.sh \\
        --sample_dir /path/to/processed/dl3dv/0a6c01ac32127687

输出结构（兼容 batch_reward.sh 输入）：
    <output>/<dataset>/<sample_id>/
        infer_info.json          # 推理参数记录
        camera.txt               # GT 相机轨迹（直接复制）
        gen_0.mp4 ... gen_N.mp4  # 生成视频
"""

import argparse
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

_HERE = Path(__file__).resolve().parent
_GEN3R_PKG = _HERE / "Gen3R"
if str(_GEN3R_PKG) not in sys.path:
    sys.path.insert(0, str(_GEN3R_PKG))

from dataset_rl import RLDataset, parse_camera_txt  # noqa: E402
from gen3r_encode import (                           # noqa: E402
    encode_text,
    build_plucker_embeds,
    encode_control_latents,
    transformer_forward,
    decode_rgb_video,
)
from grpo_core import sd3_time_shift, run_sample_step  # noqa: E402
from model_loader import load_all_models               # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# 样本列表构建
# ══════════════════════════════════════════════════════════════════════════════

def collect_samples(args) -> list[dict]:
    """收集需要推理的样本信息。"""
    samples = []

    if args.sample_dir:
        # 方式 B：单个样本
        sample_dir = Path(args.sample_dir)
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

        # 视频/帧媒体
        media = str(sample_dir / "gt.mp4") if (sample_dir / "gt.mp4").exists() \
            else str(sample_dir / "frames")

        samples.append({
            "sample_id": sample_id,
            "dataset_name": dataset_name,
            "camera_txt": str(cam_path),
            "media": media,
            "caption": meta.get("caption", meta.get("prompt", "")),
        })

    else:
        # 方式 A：批量
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

                frame_mode = args.frame_mode if hasattr(args, "frame_mode") else "video"
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
                    "caption": meta.get("caption", meta.get("prompt", "")),
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
    """对单个样本推理 num_rollouts 条视频，输出到 output_dir。

    Returns:
        out_dir : 输出目录路径
    """
    tokenizer = models["tokenizer"]
    text_encoder = models["text_encoder"]
    wan_vae = models["wan_vae"]
    geo_adapter = models["geo_adapter"]
    clip_image_encoder = models["clip_image_encoder"]
    transformer = models["transformer"]

    weight_dtype = torch.bfloat16

    sample_id = sample_info["sample_id"]
    dataset_name = sample_info["dataset_name"]
    camera_txt = sample_info["camera_txt"]
    media = sample_info["media"]
    caption = sample_info["caption"]

    out_dir = os.path.join(args.output_dir, dataset_name, sample_id)
    os.makedirs(out_dir, exist_ok=True)

    # 跳过已完成
    if args.skip_done and (Path(out_dir) / "infer_info.json").exists():
        existing = list(Path(out_dir).glob("gen_*.mp4"))
        if len(existing) >= args.num_rollouts:
            print(f"[Infer] Skip {dataset_name}/{sample_id} (already done)")
            return out_dir

    # ── 解析相机 ─────────────────────────────────────────────────────────────
    H = W = args.resolution
    c2ws, Ks = parse_camera_txt(camera_txt, H, W)
    total_cam = c2ws.shape[0]

    # ── 采样帧 ────────────────────────────────────────────────────────────────
    from dataset_rl import (
        load_frames_from_video,
        load_frames_from_dir,
        sample_frame_indices,
        get_video_total_frames,
    )

    if media.endswith(".mp4"):
        total_frames = get_video_total_frames(media)
    else:
        total_frames = len([
            f for f in Path(media).iterdir()
            if f.suffix.lower() in (".png", ".jpg", ".jpeg")
        ])

    total_avail = min(total_frames, total_cam)
    # 推理时固定从 0 开始，保证可复现
    indices = list(range(0, min(args.num_frames * args.frame_stride, total_avail), args.frame_stride))
    if len(indices) < args.num_frames:
        indices = indices + [indices[-1]] * (args.num_frames - len(indices))
    indices = indices[:args.num_frames]

    if media.endswith(".mp4"):
        pixel_values = load_frames_from_video(media, indices, H, W)
    else:
        pixel_values = load_frames_from_dir(media, indices, H, W)

    pixel_values = pixel_values.to(device)
    c2ws_s = c2ws[indices].to(device)
    Ks_s = Ks[indices].to(device)
    F = pixel_values.shape[0]

    # ── 条件编码 ──────────────────────────────────────────────────────────────
    with torch.no_grad():
        prompt_embed = encode_text(caption, tokenizer, text_encoder, device,
                                   max_length=args.tokenizer_max_length)
        neg_embed = encode_text("bad detailed", tokenizer, text_encoder, device,
                                max_length=args.tokenizer_max_length)

    prompt_embeds = [prompt_embed]
    neg_embeds = [neg_embed]

    control_index = [0]
    control_images = torch.zeros_like(pixel_values).unsqueeze(0)
    control_images[0, control_index] = pixel_values[control_index]

    with torch.no_grad():
        control_latents, clip_context = encode_control_latents(
            control_images, wan_vae, geo_adapter, clip_image_encoder,
            control_index, F, device, weight_dtype,
        )
        plucker_embeds = build_plucker_embeds(
            c2ws_s, Ks_s, h=H, w=W, num_frames=F, device=device, dtype=weight_dtype,
        )

    # seq_len
    SPATIAL_DS = 8
    TEMPORAL_DS = 4
    latent_t = ((F - 1) // TEMPORAL_DS) + 1
    latent_h = H // SPATIAL_DS
    latent_w = W // SPATIAL_DS
    patch_h = transformer.config.patch_size[1] if hasattr(transformer.config, "patch_size") else 2
    patch_w = transformer.config.patch_size[2] if hasattr(transformer.config, "patch_size") else 2
    seq_len = math.ceil((latent_w * 2 * latent_h) / (patch_h * patch_w) * latent_t)

    sigma_schedule = sd3_time_shift(
        args.shift, torch.linspace(1, 0, args.sampling_steps + 1)
    )

    # ── 生成 num_rollouts 条 rollout ──────────────────────────────────────────
    transformer.eval()
    IN_CHANNELS = 16
    for k in range(args.num_rollouts):
        z0 = torch.randn(
            (1, IN_CHANNELS, latent_t, latent_h, latent_w * 2),
            device=device, dtype=weight_dtype,
        )

        with torch.no_grad():
            _, pred_x0, _, _ = run_sample_step(
                args, z0, sigma_schedule, transformer,
                prompt_embeds, neg_embeds, seq_len,
                control_latents, plucker_embeds, clip_context,
                transformer_forward,
            )

        video_path = os.path.join(out_dir, f"gen_{k}.mp4")
        decode_rgb_video(pred_x0, wan_vae, video_path, fps=16)
        print(f"[Infer] Saved {dataset_name}/{sample_id}/gen_{k}.mp4")

    # ── 复制 GT camera.txt ────────────────────────────────────────────────────
    shutil.copy2(camera_txt, os.path.join(out_dir, "camera.txt"))

    # ── infer_info.json ───────────────────────────────────────────────────────
    infer_info = {
        "dataset": dataset_name,
        "sample_id": sample_id,
        "num_rollouts": args.num_rollouts,
        "num_frames": F,
        "resolution": H,
        "sampling_steps": args.sampling_steps,
        "eta": args.eta,
        "cfg": args.cfg_infer,
        "shift": args.shift,
        "caption": caption,
    }
    with open(os.path.join(out_dir, "infer_info.json"), "w") as f:
        json.dump(infer_info, f, indent=2)

    return out_dir


# ══════════════════════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════════════════════

def main(args):
    device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")

    # 加载配置和模型
    config = OmegaConf.load(args.config_path)
    weight_dtype = torch.bfloat16
    models = load_all_models(args, config, device, weight_dtype)
    models["transformer"].eval()

    # 收集样本
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
    print(f"  You can now run reward:\n"
          f"  bash rl_train/reward/batch_reward.sh "
          f"--sample_dir <output>/<dataset>/<sample_id> "
          f"--output_dir <reward_output>")


# ══════════════════════════════════════════════════════════════════════════════
# 命令行参数
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Gen3R 推理调试脚本")

    # ── 模型路径 ──────────────────────────────────────────────────────────────
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--vggt_path", type=str, required=True)
    parser.add_argument("--geo_adapter_path", type=str, required=True)
    parser.add_argument("--config_path", type=str,
                        default=str(_HERE / "Gen3R" / "gen3r" / "config" / "gen3r.yaml"))
    parser.add_argument("--transformer_path", type=str, default=None,
                        help="从额外 checkpoint 加载 Transformer 权重")

    # ── 数据输入（二选一） ────────────────────────────────────────────────────
    parser.add_argument("--sample_dir", type=str, default=None,
                        help="[方式B] 单个样本目录，含 camera.txt + gt.mp4")
    parser.add_argument("--data_root", type=str, default=None,
                        help="[方式A] 数据根目录（unified_data_process 输出）")
    parser.add_argument("--datasets", type=str, default="re10k,dl3dv",
                        help="[方式A] 逗号分隔的数据集名称")
    parser.add_argument("--max_samples", type=int, default=5,
                        help="[方式A] 最多推理多少个样本（0=全部）")
    parser.add_argument("--frame_mode", type=str, default="video",
                        choices=["video", "frames"])

    # ── 输出 ──────────────────────────────────────────────────────────────────
    parser.add_argument("--output_dir", type=str, required=True,
                        help="推理结果输出目录")
    parser.add_argument("--skip_done", action="store_true", default=True,
                        help="跳过已完成的样本")
    parser.add_argument("--no_skip_done", dest="skip_done", action="store_false")

    # ── 推理参数 ──────────────────────────────────────────────────────────────
    parser.add_argument("--num_rollouts", type=int, default=8,
                        help="每个样本生成多少条 rollout 视频")
    parser.add_argument("--num_frames", type=int, default=17,
                        help="采样帧数")
    parser.add_argument("--frame_stride", type=int, default=2)
    parser.add_argument("--resolution", type=int, default=560)
    parser.add_argument("--sampling_steps", type=int, default=50,
                        help="去噪步数（推理时建议 50）")
    parser.add_argument("--eta", type=float, default=0.0,
                        help="SDE 噪声强度（推理时设为 0 使用 ODE）")
    parser.add_argument("--shift", type=float, default=2.0)
    parser.add_argument("--cfg_infer", type=float, default=5.0)
    parser.add_argument("--tokenizer_max_length", type=int, default=512)

    args = parser.parse_args()

    if args.sample_dir is None and args.data_root is None:
        parser.error("需要提供 --sample_dir 或 --data_root")

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
