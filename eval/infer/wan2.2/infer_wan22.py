#!/usr/bin/env python3
"""Wan2.2-Fun-5B-Control-Camera 官方 pipeline 推理（ODE）。

使用 Wan2_2FunControlPipeline.__call__ 的官方 ODE 推理路径，不依赖任何 GRPO 代码。
每条样本产出一个 gen_0.mp4，跳过 T5 文本编码器（从样本目录读取预编码的
prompt_embed.pt / neg_embed.pt）。

相机参数处理：
  camera.txt 格式同 gen3r（OpenCV w2c，每行:
    idx fx_n fy_n cx_n cy_n 0 0 r00..r22 tx ty tz）
  直接调用 videox_fun.data.utils 里的 Camera / get_relative_pose / ray_condition，
  与 process_pose_file 等价但跳过了该函数里的 header-line 跳过逻辑。

多卡支持：按样本分片，torchrun --nproc_per_node=N 启动。

输出目录结构：
  <output_root>/<dataset>/<sample_id>/
    gen_0.mp4
    start.png, gt.mp4, camera.txt, metadata.json  (从输入目录复制)
    infer_info.json

用法：
  python infer_wan22.py \\
      --manifest /path/to/manifest.jsonl \\
      --checkpoint /path/to/Wan2.2-Fun-5B-Control-Camera \\
      --output_root /path/to/output

  torchrun --nproc_per_node=4 infer_wan22.py \\
      --manifest /path/to/manifest.jsonl \\
      --checkpoint /path/to/Wan2.2-Fun-5B-Control-Camera \\
      --output_root /path/to/output
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from PIL import Image

# ── VideoX-Fun 路径 ────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_VIDEOX_ROOT = Path(os.environ.get("VIDEOX_ROOT", str(_HERE / "VideoX-Fun")))
if str(_VIDEOX_ROOT) not in sys.path:
    sys.path.insert(0, str(_VIDEOX_ROOT))

from diffusers import FlowMatchEulerDiscreteScheduler
from videox_fun.models import (AutoencoderKLWan, AutoencoderKLWan3_8,
                                Wan2_2Transformer3DModel)
from videox_fun.pipeline import Wan2_2FunControlPipeline
from videox_fun.utils.utils import (filter_kwargs, get_image_to_video_latent,
                                     save_videos_grid)
from videox_fun.data.utils import Camera, get_relative_pose, ray_condition


# ═══════════════════════ 相机参数解析 ══════════════════════════════════════════


def build_plucker_from_camera_txt(camera_txt: Path, img_w: int, img_h: int,
                                   target_w: int, target_h: int,
                                   video_length: int) -> torch.Tensor:
    """把 camera.txt 转换成 Wan2.2 pipeline 期望的 Plücker 嵌入。

    camera.txt 每行：idx fx_n fy_n cx_n cy_n 0 0 r00..r22 tx ty tz（无 header）

    返回：[1, 6, F, H, W] float32 tensor，与官方 process_pose_file 输出等价。
    """
    with open(camera_txt, "r") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    cam_params_raw = [[float(x) for x in line.split()] for line in lines]
    cam_params = [Camera(row) for row in cam_params_raw]

    # ── 内参缩放（与 process_pose_file 相同逻辑）────────────────────────────────
    sample_wh_ratio = target_w / target_h
    pose_wh_ratio   = img_w / img_h
    if pose_wh_ratio > sample_wh_ratio:
        resized_ori_w = target_h * pose_wh_ratio
        for cp in cam_params:
            cp.fx = resized_ori_w * cp.fx / target_w
    else:
        resized_ori_h = target_w / pose_wh_ratio
        for cp in cam_params:
            cp.fy = resized_ori_h * cp.fy / target_h

    # 截断 / 补帧
    if len(cam_params) < video_length:
        cam_params = cam_params + [cam_params[-1]] * (video_length - len(cam_params))
    cam_params = cam_params[:video_length]

    intrinsic = np.array(
        [[cp.fx * target_w, cp.fy * target_h, cp.cx * target_w, cp.cy * target_h]
         for cp in cam_params], dtype=np.float32)

    K    = torch.as_tensor(intrinsic)[None]             # [1, F, 4]
    c2ws = torch.as_tensor(get_relative_pose(cam_params))[None]  # [1, F, 4, 4]

    # [1, F, H, W, 6] → [F, H, W, 6] → [6, F, H, W] → [1, 6, F, H, W]
    plucker = ray_condition(K, c2ws, target_h, target_w, device="cpu")[0]  # [F, H, W, 6]
    plucker = plucker.permute(3, 0, 1, 2).unsqueeze(0).float()             # [1, 6, F, H, W]
    return plucker


# ═══════════════════════ 文件工具 ══════════════════════════════════════════════


def copy_input_files(sample_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for fname in ["start.png", "gt.mp4", "camera.txt", "metadata.json", "gt_depth.npz"]:
        src = sample_dir / fname
        dst = output_dir / fname
        if src.exists() and not dst.exists():
            shutil.copy2(str(src), str(dst))


def load_manifest(manifest_path: str) -> list:
    return [
        json.loads(line)
        for line in Path(manifest_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_manifest_from_dir(sample_dir: str) -> list:
    p = Path(sample_dir)
    meta_path = p / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return [{
        "dataset":    meta.get("dataset", p.parent.name),
        "sample_id":  meta.get("sample_id", p.name),
        "sample_dir": str(p),
    }]


# ═══════════════════════ 主流程 ═══════════════════════════════════════════════


def load_pipeline(checkpoint: str, config_path: str, weight_dtype,
                  device: torch.device) -> Wan2_2FunControlPipeline:
    """加载 Wan2.2 官方 pipeline，跳过 text encoder / tokenizer。"""
    config = OmegaConf.load(config_path)

    # Transformer
    transformer = Wan2_2Transformer3DModel.from_pretrained(
        os.path.join(checkpoint,
                     config["transformer_additional_kwargs"].get(
                         "transformer_low_noise_model_subpath", "transformer")),
        transformer_additional_kwargs=OmegaConf.to_container(
            config["transformer_additional_kwargs"]),
        torch_dtype=weight_dtype,
    )
    # 5B 是单模型，不需要 transformer_2
    transformer_combination = config["transformer_additional_kwargs"].get(
        "transformer_combination_type", "single")
    transformer_2 = None

    # VAE
    Chosen_VAE = {
        "AutoencoderKLWan":     AutoencoderKLWan,
        "AutoencoderKLWan3_8":  AutoencoderKLWan3_8,
    }[config["vae_kwargs"].get("vae_type", "AutoencoderKLWan")]
    vae = Chosen_VAE.from_pretrained(
        os.path.join(checkpoint,
                     config["vae_kwargs"].get("vae_subpath", "vae")),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    ).to(weight_dtype)

    # Scheduler（Flow = FlowMatchEulerDiscreteScheduler）
    scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler,
                        OmegaConf.to_container(config["scheduler_kwargs"]))
    )

    # Pipeline（不加载 text_encoder / tokenizer）
    pipeline = Wan2_2FunControlPipeline(
        transformer=transformer,
        transformer_2=transformer_2,
        vae=vae,
        tokenizer=None,
        text_encoder=None,
        scheduler=scheduler,
    )
    pipeline = pipeline.to(device)
    return pipeline, config


def main(args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank       = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device      = torch.device(f"cuda:{local_rank}")
    weight_dtype = torch.bfloat16

    # ── 任务列表 ──────────────────────────────────────────────────────────────
    if args.manifest:
        all_entries = load_manifest(args.manifest)
    elif args.sample_dir:
        all_entries = build_manifest_from_dir(args.sample_dir)
    else:
        raise ValueError("需要 --manifest 或 --sample_dir")

    my_entries = all_entries[rank::world_size]

    if rank == 0:
        print(f"\n[Wan2.2 Infer] 共 {len(all_entries)} 条样本，本进程处理 {len(my_entries)} 条")
        print(f"  checkpoint:  {args.checkpoint}")
        print(f"  config:      {args.config_path}")
        print(f"  resolution:  {args.width}x{args.height}")
        print(f"  video_length:{args.video_length}")
        print(f"  steps:       {args.num_inference_steps}")
        print(f"  guidance:    {args.guidance_scale}")
        print(f"  shift:       {args.shift}")
        print(f"  seed:        {args.seed}")
        print(f"  output_root: {args.output_root}\n")

    # ── 加载官方 pipeline ─────────────────────────────────────────────────────
    pipeline, config = load_pipeline(
        args.checkpoint, args.config_path, weight_dtype, device)
    boundary = config["transformer_additional_kwargs"].get("boundary", 0.875)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    sample_size = [args.height, args.width]

    # video_length 对齐到 VAE temporal compression ratio
    t_ratio = pipeline.vae.config.temporal_compression_ratio
    video_length = int((args.video_length - 1) // t_ratio * t_ratio) + 1

    for entry_idx, entry in enumerate(my_entries):
        dataset   = entry.get("dataset", "unknown")
        sample_id = entry.get("sample_id", f"sample_{entry_idx}")
        sample_dir = Path(entry["sample_dir"])
        output_dir = Path(args.output_root) / dataset / sample_id

        out_video = output_dir / "gen_0.mp4"
        if args.skip_done and out_video.exists():
            print(f"[{rank}] skip (done): {dataset}/{sample_id}")
            continue

        print(f"[{rank}] [{entry_idx+1}/{len(my_entries)}] {dataset}/{sample_id}")
        copy_input_files(sample_dir, output_dir)

        # ── 读取元数据 ────────────────────────────────────────────────────────
        meta_path = sample_dir / "metadata.json"
        if not meta_path.exists():
            print(f"  [rank {rank}] metadata.json 缺失，跳过")
            continue
        meta   = json.loads(meta_path.read_text(encoding="utf-8"))
        img_w  = meta["img_w"]
        img_h  = meta["img_h"]

        # ── 预编码 T5 embedding ───────────────────────────────────────────────
        embed_path = sample_dir / "prompt_embed.pt"
        neg_path   = sample_dir / "neg_embed.pt"
        if not embed_path.exists() or not neg_path.exists():
            print(f"  [rank {rank}] prompt_embed.pt / neg_embed.pt 缺失，跳过")
            continue

        prompt_embed = torch.load(embed_path, map_location="cpu", weights_only=True)
        neg_embed    = torch.load(neg_path,   map_location="cpu", weights_only=True)
        # 期望 [L, 4096] → [1, L, 4096]
        if prompt_embed.dim() == 2:
            prompt_embed = prompt_embed.unsqueeze(0)
        if neg_embed.dim() == 2:
            neg_embed = neg_embed.unsqueeze(0)
        prompt_embed = prompt_embed.to(device, weight_dtype)
        neg_embed    = neg_embed.to(device, weight_dtype)

        # ── 起始图像（inpaint latent） ─────────────────────────────────────────
        start_img_path = str(sample_dir / "start.png")
        try:
            inpaint_video, inpaint_mask, clip_image = get_image_to_video_latent(
                start_img_path, None, video_length=video_length,
                sample_size=sample_size)
        except Exception as e:
            print(f"  [rank {rank}] 图像准备失败: {e}")
            continue

        # ── 相机控制信号 ───────────────────────────────────────────────────────
        try:
            control_camera_video = build_plucker_from_camera_txt(
                sample_dir / "camera.txt",
                img_w=img_w, img_h=img_h,
                target_w=args.width, target_h=args.height,
                video_length=video_length,
            ).to(device, weight_dtype)
        except Exception as e:
            print(f"  [rank {rank}] 相机参数处理失败: {e}")
            continue

        # ── 官方 pipeline 调用 ────────────────────────────────────────────────
        try:
            with torch.no_grad():
                output = pipeline(
                    prompt=None,
                    negative_prompt=None,
                    prompt_embeds=prompt_embed,
                    negative_prompt_embeds=neg_embed,
                    num_frames=video_length,
                    height=args.height,
                    width=args.width,
                    generator=generator,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    video=inpaint_video,
                    mask_video=inpaint_mask,
                    control_video=None,
                    control_camera_video=control_camera_video,
                    ref_image=None,
                    boundary=boundary,
                    shift=args.shift,
                ).videos  # [1, 3, F, H, W]

            save_videos_grid(output, str(out_video), fps=args.fps)
            print(f"  [rank {rank}] gen_0.mp4 保存成功")

        except Exception as e:
            import traceback
            print(f"  [rank {rank}] 推理失败: {e}")
            traceback.print_exc()
            continue

        info = {
            "checkpoint":          args.checkpoint,
            "config_path":         args.config_path,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale":      args.guidance_scale,
            "shift":               args.shift,
            "video_length":        video_length,
            "height":              args.height,
            "width":               args.width,
            "fps":                 args.fps,
            "seed":                args.seed,
            "sampler":             "Flow (FlowMatchEulerDiscrete)",
        }
        (output_dir / "infer_info.json").write_text(
            json.dumps(info, indent=2), encoding="utf-8")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

    if rank == 0:
        print(f"\n[Wan2.2 Infer] 完成。输出: {args.output_root}")


if __name__ == "__main__":
    _HERE_DEFAULT_CONFIG = str(
        Path(__file__).resolve().parent
        / "VideoX-Fun/config/wan2.2/wan_civitai_5b.yaml"
    )
    _HERE_DEFAULT_CKPT = str(
        Path(__file__).resolve().parent.parent.parent.parent
        / "model/Wan2.2-Fun-5B-Control-Camera"
    )

    parser = argparse.ArgumentParser(description="Wan2.2 官方 pipeline ODE 推理")

    input_grp = parser.add_mutually_exclusive_group(required=True)
    input_grp.add_argument("--manifest",   type=str, help="manifest.jsonl 路径")
    input_grp.add_argument("--sample_dir", type=str, help="单条样本目录（调试用）")

    parser.add_argument("--checkpoint",  type=str, default=_HERE_DEFAULT_CKPT,
                        help="Wan2.2 模型目录")
    parser.add_argument("--config_path", type=str, default=_HERE_DEFAULT_CONFIG,
                        help="wan_civitai_5b.yaml 路径")
    parser.add_argument("--output_root", type=str, required=True)

    parser.add_argument("--height",              type=int,   default=704)
    parser.add_argument("--width",               type=int,   default=1280)
    parser.add_argument("--video_length",        type=int,   default=49,
                        help="帧数（会自动对齐到 VAE temporal ratio）")
    parser.add_argument("--fps",                 type=int,   default=16)
    parser.add_argument("--num_inference_steps", type=int,   default=50)
    parser.add_argument("--guidance_scale",      type=float, default=6.0)
    parser.add_argument("--shift",               type=float, default=5.0)
    parser.add_argument("--seed",                type=int,   default=42)
    parser.add_argument("--skip_done",           action="store_true")

    main(parser.parse_args())
