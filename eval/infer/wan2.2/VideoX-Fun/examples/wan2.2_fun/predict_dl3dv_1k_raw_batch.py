#!/usr/bin/env python3
"""
DL3DV 1K raw data -> Wan2.2-Fun-5B-Control-Camera 推理脚本

核心约定：
  DL3DV transforms.json 的 transform_matrix 是 c2w，但用的是
  nerfstudio/OpenGL 惯例 (X右, Y上, Z后)。
  模型 ray_condition 期待 OpenCV 惯例 (X右, Y下, Z前)。
  转换：c2w[:3, 1:3] *= -1  (翻转 Y、Z 列)

  DL3DV 内参在 transforms.json 里对应的是原始高分辨率 (3840x2160)，
  images_2/ 是 2x 降采样后的 1920x1080。
  推理分辨率：704x1280 (H x W)，需 H 和 W 均被 32 整除。
  图像从 1920x1080 → 1280 宽 (scale=2/3) → 高 720 → 中心裁剪到 704。

  平移归一化：DL3DV 是米制尺度，统一缩放使最大帧位移 = 1.0。

用法:
  python predict_dl3dv_1k_raw_batch.py \
      --data_root /mnt/afs/visitor16/RL_Pipeline/data/dl3dv/train_2k/1K \
      --model_path /mnt/afs/visitor16/Wan2.2_RL/Model/Wan2.2-Fun-5B-Control-Camera \
      --config_path config/wan2.2/wan_civitai_5b.yaml \
      --output_dir samples/wan2_2_fun_5b_dl3dv_1k_704x1280_49f \
      --num_frames 49
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import imageio
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
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

from videox_fun.data.utils import process_pose_file            # noqa: E402
from videox_fun.models import (                                # noqa: E402
    AutoencoderKLWan3_8,
    Wan2_2Transformer3DModel,
    WanT5EncoderModel,
)
from videox_fun.models.cache_utils import get_teacache_coefficients  # noqa: E402
from videox_fun.pipeline.pipeline_wan2_2_fun_control import (  # noqa: E402
    Wan2_2FunControlPipeline,
)
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler  # noqa: E402
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler  # noqa: E402
from videox_fun.utils.utils import (                          # noqa: E402
    filter_kwargs,
    get_image_to_video_latent,
    save_videos_grid,
)

# ─────────────────────────── image helpers ───────────────────────────────────

def load_and_resize_image(path: str, out_w: int, out_h: int) -> np.ndarray:
    """Load image, resize keeping W=out_w, then center-crop to out_h."""
    img = Image.open(path).convert("RGB")
    orig_w, orig_h = img.size          # e.g. 1920 x 1080
    scale = out_w / orig_w             # e.g. 1280/1920 = 2/3
    scaled_w = out_w
    scaled_h = max(out_h, int(round(orig_h * scale)))
    img = img.resize((scaled_w, scaled_h), Image.BICUBIC)
    top = (scaled_h - out_h) // 2
    img = img.crop((0, top, out_w, top + out_h))
    return np.array(img)


def get_intrinsics_for_target(tf: dict, images_2_w: int, images_2_h: int,
                               out_w: int, out_h: int):
    """
    Scale DL3DV intrinsics from original resolution (tf['w'] x tf['h'])
    down to images_2 resolution, then to (out_w x scaled_h), then adjust
    principal point for center crop to out_h.
    Returns (fl_x, fl_y, cx, cy) in pixels for the final out_w x out_h image.
    """
    orig_w, orig_h = float(tf["w"]), float(tf["h"])

    # Step 1: scale from original to images_2
    s_w = images_2_w / orig_w      # 1920/3840 = 0.5
    s_h = images_2_h / orig_h      # 1080/2160 = 0.5
    fl_x = tf["fl_x"] * s_w
    fl_y = tf["fl_y"] * s_h
    cx   = tf["cx"]   * s_w
    cy   = tf["cy"]   * s_h

    # Step 2: scale from images_2 to (out_w, scaled_h)
    scale = out_w / images_2_w     # 1280/1920 = 2/3
    scaled_h = int(round(images_2_h * scale))   # 720
    fl_x *= scale
    fl_y *= scale
    cx   *= scale
    cy   *= scale

    # Step 3: center-crop height from scaled_h to out_h
    crop_top = (scaled_h - out_h) // 2    # (720 - 704)//2 = 8
    cy -= crop_top
    return fl_x, fl_y, cx, cy


# ─────────────────────────── camera helpers ──────────────────────────────────

def build_camera_txt(frames_meta: list, tf: dict,
                     images_2_w: int, images_2_h: int,
                     out_w: int, out_h: int,
                     num_frames: int,
                     target_max_translation: float = 0.0) -> str:
    """
    Convert DL3DV transforms.json frames to camera.txt content.

    Convention notes:
    - DL3DV/nerfstudio with camera_model=OPENCV stores c2w in OpenCV convention
      already (camera Z-axis points into the scene, same as the CameraCtrl/
      VideoX-Fun training data format). No axis-flip is needed.
    - Intrinsics are stored at the original full resolution in transforms.json;
      we rescale them to the actual images_2 resolution and then to out_w x out_h.
    - target_max_translation > 0 scales all translations so that the maximum
      per-frame displacement from the first frame equals that value.
      Set to 0 to keep raw metric scale (recommended: DL3DV 1K raw scale
      is already compatible with the training data scale).

    Returns the camera.txt as a string.
    """
    fl_x, fl_y, cx, cy = get_intrinsics_for_target(
        tf, images_2_w, images_2_h, out_w, out_h)

    fx_n = fl_x / out_w
    fy_n = fl_y / out_h
    cx_n = cx   / out_w
    cy_n = cy   / out_h

    c2ws = []
    for fmeta in frames_meta[:num_frames]:
        m = np.asarray(fmeta["transform_matrix"], dtype=np.float64)
        if m.shape == (3, 4):
            c4 = np.eye(4, dtype=np.float64)
            c4[:3, :] = m
            m = c4
        # nerfstudio OPENCV c2w → VideoX-Fun training convention:
        # The applied_transform in nerfstudio swaps X/Y and negates Z,
        # causing the camera motion direction to be reversed in relative coords.
        # Flipping Y and Z columns of c2w (OpenGL→OpenCV convention fix) corrects
        # the relative trajectory: later frames become AHEAD (+Z) not BEHIND (-Z).
        m[:3, 1:3] *= -1
        c2ws.append(m)

    # Optional translation normalization
    if target_max_translation > 0:
        pos0 = c2ws[0][:3, 3]
        disps = [np.linalg.norm(c[:3, 3] - pos0) for c in c2ws[1:]]
        max_disp = max(disps) if disps else 1.0
        t_scale = target_max_translation / max_disp if max_disp > 1e-6 else 1.0
    else:
        t_scale = 1.0
        max_disp = np.linalg.norm(c2ws[-1][:3, 3] - c2ws[0][:3, 3])
    print(f"    max_disp={max_disp:.4f}  translation_scale={t_scale:.4f}")

    lines = [f"# frame fx fy cx cy d1 d2 w2c(3x4) | W={out_w} H={out_h}\n"]
    for i, c2w in enumerate(c2ws):
        c2w_scaled = c2w.copy()
        c2w_scaled[:3, 3] *= t_scale
        w2c = np.linalg.inv(c2w_scaled)[:3, :4].reshape(-1).tolist()
        row = ([str(i), f"{fx_n:.8f}", f"{fy_n:.8f}",
                f"{cx_n:.8f}", f"{cy_n:.8f}", "0", "0"]
               + [f"{v:.8f}" for v in w2c])
        lines.append(" ".join(row) + "\n")
    return "".join(lines)


# ─────────────────────────── pipeline loader ─────────────────────────────────

def load_pipeline(args, config, weight_dtype, device):
    transformer = Wan2_2Transformer3DModel.from_pretrained(
        os.path.join(
            args.model_path,
            config["transformer_additional_kwargs"].get(
                "transformer_low_noise_model_subpath", "transformer"),
        ),
        transformer_additional_kwargs=OmegaConf.to_container(
            config["transformer_additional_kwargs"]),
        torch_dtype=weight_dtype,
    )

    vae = AutoencoderKLWan3_8.from_pretrained(
        os.path.join(
            args.model_path,
            config["vae_kwargs"].get("vae_subpath", "vae"),
        ),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    ).to(weight_dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(
            args.model_path,
            config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer"),
        )
    )
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(
            args.model_path,
            config["text_encoder_kwargs"].get(
                "text_encoder_subpath", "text_encoder"),
        ),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    ).eval()

    chosen_scheduler = {
        "Flow":        FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc":  FlowUniPCMultistepScheduler,
        "Flow_DPM++":  FlowDPMSolverMultistepScheduler,
    }[args.sampler_name]
    if args.sampler_name in ("Flow_Unipc", "Flow_DPM++"):
        config["scheduler_kwargs"]["shift"] = 1
    scheduler = chosen_scheduler(
        **filter_kwargs(chosen_scheduler,
                        OmegaConf.to_container(config["scheduler_kwargs"]))
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


# ─────────────────────────── main ────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str,
                   default="/mnt/afs/visitor16/RL_Pipeline/data/dl3dv/train_2k/1K")
    p.add_argument("--model_path", type=str,
                   default="/mnt/afs/visitor16/Wan2.2_RL/Model/Wan2.2-Fun-5B-Control-Camera")
    p.add_argument("--config_path", type=str,
                   default="config/wan2.2/wan_civitai_5b.yaml")
    p.add_argument("--output_dir", type=str,
                   default="samples/wan2_2_fun_5b_dl3dv_1k_704x1280_49f")
    p.add_argument("--num_frames", type=int, default=49)
    p.add_argument("--height", type=int, default=704)
    p.add_argument("--width",  type=int, default=1280)
    p.add_argument("--fps",    type=int, default=24)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale",      type=float, default=6.0)
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--sampler_name", type=str, default="Flow",
                   choices=["Flow", "Flow_Unipc", "Flow_DPM++"])
    p.add_argument("--shift",  type=float, default=5.0)
    p.add_argument("--gpu_memory_mode", type=str, default="model_full_load",
                   choices=["model_full_load", "model_cpu_offload"])
    p.add_argument("--disable_teacache", action="store_true")
    p.add_argument("--target_max_translation", type=float, default=0.0,
                   help="Normalise camera translations so max displacement = this value. "
                        "0 = keep raw metric scale (default, matches training data scale).")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Validate resolution (both must be divisible by 32)
    assert args.height % 32 == 0 and args.width % 32 == 0, (
        f"height={args.height} and width={args.width} must both be divisible by 32. "
        f"Try height=704, width=1280.")

    config   = OmegaConf.load(args.config_path)
    boundary = config["transformer_additional_kwargs"].get("boundary", 0.875)
    weight_dtype = torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading pipeline…")
    pipeline, vae = load_pipeline(args, config, weight_dtype, device)

    if not args.disable_teacache:
        coefficients = get_teacache_coefficients(args.model_path)
        if coefficients is not None:
            pipeline.transformer.enable_teacache(
                coefficients, args.num_inference_steps, 0.10,
                num_skip_start_steps=5, offload=False)
            print("TeaCache enabled.")

    # Collect samples (skip .zip files)
    data_root = Path(args.data_root)
    sample_dirs = sorted([
        d for d in data_root.iterdir()
        if d.is_dir()
        and (d / "transforms.json").exists()
        and (d / "images_2").exists()
    ])
    print(f"Found {len(sample_dirs)} valid sample directories.")

    vl = int((args.num_frames - 1) // vae.config.temporal_compression_ratio
             * vae.config.temporal_compression_ratio) + 1
    print(f"video_length after VAE alignment: {vl}")

    prompt = (
        "A high-quality, realistic video with smooth camera movement "
        "through the scene, sharp details, natural lighting."
    )
    negative_prompt = (
        "static frame, blur, low quality, bad anatomy, distorted, "
        "jpeg artifacts, subtitles, watermark, noisy"
    )

    results = []
    with torch.no_grad():
        for idx, sample_dir in enumerate(sample_dirs, start=1):
            sample_id = sample_dir.name[:16]
            print(f"\n[{idx}/{len(sample_dirs)}] {sample_id}")

            tf = json.loads((sample_dir / "transforms.json").read_text())
            frames_meta = tf["frames"]
            if len(frames_meta) < vl:
                print(f"  Skip: only {len(frames_meta)} frames, need {vl}")
                results.append({"sample_id": sample_id, "status": "skip"})
                continue

            img_dir = sample_dir / "images_2"
            # Detect actual images_2 resolution
            first_img_path = img_dir / Path(frames_meta[0]["file_path"]).name
            if not first_img_path.exists():
                first_img_path = img_dir / frames_meta[0]["file_path"]
            first_img = Image.open(first_img_path)
            img2_w, img2_h = first_img.size   # e.g. 1920 x 1080

            print(f"  images_2: {img2_w}x{img2_h}  original: {tf['w']}x{tf['h']}")
            print(f"  target: {args.width}x{args.height}")

            # ── Build camera.txt ───────────────────────────────────────────
            cam_txt_content = build_camera_txt(
                frames_meta, tf, img2_w, img2_h,
                args.width, args.height, vl,
                target_max_translation=args.target_max_translation,
            )
            with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".txt", delete=False) as f:
                f.write(cam_txt_content)
                cam_txt_path = f.name

            # ── Build start.png and gt.mp4 ─────────────────────────────────
            start_frame_path = img_dir / Path(frames_meta[0]["file_path"]).name
            if not start_frame_path.exists():
                start_frame_path = img_dir / frames_meta[0]["file_path"]
            start_arr = load_and_resize_image(
                str(start_frame_path), args.width, args.height)
            start_img_path = str(out_dir / f"{idx:02d}_{sample_id}_start.png")
            Image.fromarray(start_arr).save(start_img_path)

            gt_frames = []
            for fm in frames_meta[:vl]:
                fp = img_dir / Path(fm["file_path"]).name
                if not fp.exists():
                    fp = img_dir / fm["file_path"]
                gt_frames.append(load_and_resize_image(str(fp), args.width, args.height))
            gt_path = str(out_dir / f"{idx:02d}_{sample_id}_gt.mp4")
            imageio.mimwrite(gt_path, gt_frames, fps=args.fps, quality=8)
            print(f"  GT saved: {gt_path}")

            # ── Plücker embedding ──────────────────────────────────────────
            # original_pose_width/height MUST match the normalisation resolution
            # in camera.txt (we used out_w x out_h, so same aspect ratio -> no fx/fy rescale)
            control_camera_video = process_pose_file(
                cam_txt_path,
                width=args.width, height=args.height,
                original_pose_width=args.width, original_pose_height=args.height,
            )
            os.unlink(cam_txt_path)

            control_camera_video = (
                control_camera_video[:vl]
                .permute([3, 0, 1, 2])
                .unsqueeze(0)
            )
            print(f"  control_camera_video: {control_camera_video.shape}")

            # ── Inpainting latents ─────────────────────────────────────────
            inpaint_video, inpaint_mask, _ = get_image_to_video_latent(
                start_img_path, None,
                video_length=vl,
                sample_size=[args.height, args.width],
            )

            generator = torch.Generator(device=device).manual_seed(
                args.seed + idx)

            # ── Run pipeline ───────────────────────────────────────────────
            sample = pipeline(
                prompt,
                num_frames          = vl,
                negative_prompt     = negative_prompt,
                height              = args.height,
                width               = args.width,
                generator           = generator,
                guidance_scale      = args.guidance_scale,
                num_inference_steps = args.num_inference_steps,
                video               = inpaint_video,
                mask_video          = inpaint_mask,
                control_video       = None,
                control_camera_video= control_camera_video,
                ref_image           = None,
                boundary            = boundary,
                shift               = args.shift,
            ).videos

            out_path = str(out_dir / f"{idx:02d}_{sample_id}_pred.mp4")
            save_videos_grid(sample, out_path, fps=args.fps)
            print(f"  Pred saved: {out_path}")
            results.append({"sample_id": sample_id, "status": "ok",
                             "pred": out_path, "gt": gt_path})

    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDone. Summary → {summary_path}")


if __name__ == "__main__":
    main()
