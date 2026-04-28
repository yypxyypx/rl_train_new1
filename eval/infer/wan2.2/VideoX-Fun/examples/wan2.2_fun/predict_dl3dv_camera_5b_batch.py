import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from transformers import AutoTokenizer

current_file_path = os.path.abspath(__file__)
project_roots = [
    os.path.dirname(current_file_path),
    os.path.dirname(os.path.dirname(current_file_path)),
    os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))),
]
for project_root in project_roots:
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from videox_fun.models import (  # noqa: E402
    AutoencoderKLWan,
    AutoencoderKLWan3_8,
    Wan2_2Transformer3DModel,
    WanT5EncoderModel,
)
from videox_fun.models.cache_utils import get_teacache_coefficients  # noqa: E402
from videox_fun.pipeline.pipeline_wan2_2_fun_control import Wan2_2FunControlPipeline  # noqa: E402
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler  # noqa: E402
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler  # noqa: E402
from videox_fun.utils.utils import (  # noqa: E402
    filter_kwargs,
    get_image_to_video_latent,
    save_videos_grid,
)


class Camera:
    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)


def get_relative_pose(cam_params):
    abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
    abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
    target_cam_c2w = np.eye(4)
    abs2rel = target_cam_c2w @ abs_w2cs[0]
    ret_poses = [target_cam_c2w] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
    return np.array(ret_poses, dtype=np.float32)


def ray_condition(K, c2w, H, W, device):
    B = K.shape[0]
    j, i = torch.meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
        indexing="ij",
    )
    i = i.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5
    j = j.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5

    fx, fy, cx, cy = K.chunk(4, dim=-1)
    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)
    rays_o = c2w[..., :3, 3]
    rays_o = rays_o[:, :, None].expand_as(rays_d)
    rays_dxo = torch.cross(rays_o, rays_d)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    return plucker.reshape(B, c2w.shape[1], H, W, 6)


def process_pose_params(cam_params, width=576, height=576, original_pose_width=1280, original_pose_height=720, device="cpu"):
    cam_params = [Camera(cam_param) for cam_param in cam_params]
    sample_wh_ratio = width / height
    pose_wh_ratio = original_pose_width / original_pose_height

    if pose_wh_ratio > sample_wh_ratio:
        resized_ori_w = height * pose_wh_ratio
        for cam_param in cam_params:
            cam_param.fx = resized_ori_w * cam_param.fx / width
    else:
        resized_ori_h = width / pose_wh_ratio
        for cam_param in cam_params:
            cam_param.fy = resized_ori_h * cam_param.fy / height

    intrinsic = np.asarray(
        [[cam_param.fx * width, cam_param.fy * height, cam_param.cx * width, cam_param.cy * height] for cam_param in cam_params],
        dtype=np.float32,
    )
    K = torch.as_tensor(intrinsic)[None]
    c2ws = get_relative_pose(cam_params)
    c2ws = torch.as_tensor(c2ws)[None]
    plucker_embedding = ray_condition(K, c2ws, height, width, device=device)[0].permute(0, 3, 1, 2).contiguous()
    return plucker_embedding.permute(0, 2, 3, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Batch inference for dl3dv with Wan2.2-Fun-5B-Control-Camera.")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/mnt/afs/visitor16/Wan2.2_RL/test_moxingxingneng_data/dl3dv",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/mnt/afs/visitor16/Wan2.2_RL/Model/Wan2.2-Fun-5B-Control-Camera",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/wan2.2/wan_civitai_5b.yaml",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="samples/wan2_2_fun_5b_dl3dv_576x576_49f",
    )
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--video_length", type=int, default=49)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampler_name", type=str, default="Flow", choices=["Flow", "Flow_Unipc", "Flow_DPM++"])
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--gpu_memory_mode", type=str, default="model_full_load", choices=["model_full_load", "model_cpu_offload"])
    parser.add_argument("--enable_teacache", action="store_true", default=True)
    parser.add_argument("--disable_teacache", action="store_true")
    return parser.parse_args()


def load_pipeline(args, config, weight_dtype, device):
    transformer = Wan2_2Transformer3DModel.from_pretrained(
        os.path.join(args.model_path, config["transformer_additional_kwargs"].get("transformer_low_noise_model_subpath", "transformer")),
        transformer_additional_kwargs=OmegaConf.to_container(config["transformer_additional_kwargs"]),
        torch_dtype=weight_dtype,
    )

    chosen_vae = {
        "AutoencoderKLWan": AutoencoderKLWan,
        "AutoencoderKLWan3_8": AutoencoderKLWan3_8,
    }[config["vae_kwargs"].get("vae_type", "AutoencoderKLWan")]
    vae = chosen_vae.from_pretrained(
        os.path.join(args.model_path, config["vae_kwargs"].get("vae_subpath", "vae")),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    ).to(weight_dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.model_path, config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer")),
    )
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(args.model_path, config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder")),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    ).eval()

    chosen_scheduler = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }[args.sampler_name]
    if args.sampler_name in ("Flow_Unipc", "Flow_DPM++"):
        config["scheduler_kwargs"]["shift"] = 1
    scheduler = chosen_scheduler(
        **filter_kwargs(chosen_scheduler, OmegaConf.to_container(config["scheduler_kwargs"]))
    )

    pipeline = Wan2_2FunControlPipeline(
        transformer=transformer,
        transformer_2=None,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
    )
    if args.gpu_memory_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    else:
        pipeline.to(device=device)
    return pipeline, vae


def load_cam_params(camera_path):
    rows = []
    with open(camera_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 19:
                continue
            try:
                _ = float(parts[0])
                row = [float(x) for x in parts[:19]]
                rows.append(row)
            except ValueError:
                continue
    return rows


def pick_samples(data_root, num_samples):
    root = Path(data_root)
    candidates = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        start_png = d / "start.png"
        camera_txt = d / "camera.txt"
        metadata_json = d / "metadata.json"
        if start_png.exists() and camera_txt.exists() and metadata_json.exists():
            candidates.append(d)
        if len(candidates) >= num_samples:
            break
    return candidates


def main():
    args = parse_args()
    if args.disable_teacache:
        args.enable_teacache = False

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = OmegaConf.load(args.config_path)
    boundary = config["transformer_additional_kwargs"].get("boundary", 0.875)
    weight_dtype = torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipeline, vae = load_pipeline(args, config, weight_dtype, device)
    if args.enable_teacache:
        coefficients = get_teacache_coefficients(args.model_path)
        if coefficients is not None:
            pipeline.transformer.enable_teacache(
                coefficients,
                args.num_inference_steps,
                0.10,
                num_skip_start_steps=5,
                offload=False,
            )

    samples = pick_samples(args.data_root, args.num_samples)
    if len(samples) < args.num_samples:
        raise RuntimeError(f"Only found {len(samples)} valid samples under {args.data_root}.")

    results = []
    with torch.no_grad():
        for idx, sample_dir in enumerate(samples, start=1):
            sample_id = sample_dir.name
            metadata_path = sample_dir / "metadata.json"
            camera_path = sample_dir / "camera.txt"
            start_image = sample_dir / "start.png"

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            prompt = metadata.get("caption", "A realistic scene with natural camera motion.")
            negative_prompt = (
                "static frame, blur, low quality, bad anatomy, distorted, jpeg artifacts, "
                "subtitles, watermark, noisy"
            )
            orig_w = int(metadata.get("img_w", metadata.get("orig_w", 1280)))
            orig_h = int(metadata.get("img_h", metadata.get("orig_h", 720)))

            cam_params = load_cam_params(camera_path)
            if not cam_params:
                results.append({"sample_id": sample_id, "status": "failed", "reason": "empty_or_invalid_camera"})
                continue

            video_length = int((args.video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1
            generator = torch.Generator(device=device).manual_seed(args.seed + idx)

            inpaint_video, inpaint_video_mask, _ = get_image_to_video_latent(
                str(start_image),
                None,
                video_length=video_length,
                sample_size=[args.height, args.width],
            )

            control_camera_video = process_pose_params(
                cam_params,
                width=args.width,
                height=args.height,
                original_pose_width=orig_w,
                original_pose_height=orig_h,
            )
            control_camera_video = control_camera_video[:video_length].permute([3, 0, 1, 2]).unsqueeze(0)

            sample = pipeline(
                prompt,
                num_frames=video_length,
                negative_prompt=negative_prompt,
                height=args.height,
                width=args.width,
                generator=generator,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                video=inpaint_video,
                mask_video=inpaint_video_mask,
                control_video=None,
                control_camera_video=control_camera_video,
                ref_image=None,
                boundary=boundary,
                shift=args.shift,
            ).videos

            out_path = output_dir / f"{idx:02d}_{sample_id}.mp4"
            save_videos_grid(sample, str(out_path), fps=args.fps)
            results.append({"sample_id": sample_id, "status": "ok", "output": str(out_path)})
            print(f"[{idx}/{len(samples)}] done: {sample_id} -> {out_path}")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
