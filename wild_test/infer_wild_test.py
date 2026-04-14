"""Wild Test 视频推理脚本

读取 generate_trajectories.py 输出的 results/<name>/cameras.json + input.png，
运行 Gen3R SDE 推理（flux_step, eta=0.2）生成视频。

每张图生成 1 个视频，单卡 + CPU offload，顺序处理。
已有 rgb.mp4 的子目录自动跳过。

运行：
    cd /home/users/puxin.yan-labs/wild_test
    conda activate gen3r
    python infer_wild_test.py [--eta 0.2] [--steps 50] [--guidance_scale 5.0]
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import imageio
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from diffusers.utils import export_to_video
from diffusers.video_processor import VideoProcessor
from einops import rearrange
from torchvision.transforms.functional import resize

# ─── 路径配置 ─────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_GEN3R_ROOT = _HERE.parent / "RL" / "gen3r" / "Gen3R"
for _p in [str(_GEN3R_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gen3r.pipeline import Gen3RPipeline
from gen3r.utils.data_utils import center_crop, compute_rays, preprocess_poses
from gen3r.utils.common_utils import save_videos_grid

# ─── 常量 ─────────────────────────────────────────────────────────────────────
CHECKPOINTS = str(_GEN3R_ROOT / "checkpoints")
RESULTS_DIR = _HERE / "results"
NUM_FRAMES = 49
TARGET_H = TARGET_W = 560


# ═══════════════════════════════════════════════════════════════════════════════
# flux_step：ODE→SDE（来自 DanceGRPO/infer_sde_test.py，无修改）
# ═══════════════════════════════════════════════════════════════════════════════

def sd3_time_shift(shift, t):
    return (shift * t) / (1 + (shift - 1) * t)


def flux_step(model_output, latents, eta, sigmas, index, prev_sample=None,
              grpo=False, sde_solver=True):
    """ODE→SDE 单步，纯推理模式（grpo=False）。"""
    sigma = sigmas[index]
    dsigma = sigmas[index + 1] - sigma
    prev_sample_mean = latents + dsigma * model_output
    pred_original_sample = latents - sigma * model_output

    delta_t = sigma - sigmas[index + 1]
    std_dev_t = eta * math.sqrt(delta_t.item())

    if sde_solver:
        score_estimate = -(latents - pred_original_sample * (1 - sigma)) / sigma ** 2
        log_term = -0.5 * eta ** 2 * score_estimate
        prev_sample_mean = prev_sample_mean + log_term * dsigma

    if std_dev_t > 0:
        prev_sample_out = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t
    else:
        prev_sample_out = prev_sample_mean

    return prev_sample_out, pred_original_sample


# ═══════════════════════════════════════════════════════════════════════════════
# decode_latents device-alignment patch（CPU offload 兼容，来自 infer_sde_test.py）
# ═══════════════════════════════════════════════════════════════════════════════

_original_decode_latents = Gen3RPipeline.decode_latents


def _decode_latents_device_aligned(self, latents, min_max_depth_mask=False):
    dev = latents.device
    dt = torch.bfloat16
    for mod in [self.geo_adapter, self.vggt, self.wan_vae]:
        if mod is not None:
            mod.to(device=dev, dtype=dt)
    return _original_decode_latents(self, latents, min_max_depth_mask=min_max_depth_mask)


Gen3RPipeline.decode_latents = _decode_latents_device_aligned


# ═══════════════════════════════════════════════════════════════════════════════
# 相机加载
# ═══════════════════════════════════════════════════════════════════════════════

def load_cameras_json(cameras_path, device, dtype):
    """读取 cameras.json，返回 c2ws [F,4,4] 和 Ks [F,3,3]。

    cameras.json 格式：{"extrinsics": [F×4×4 w2c], "intrinsics": [F×3×3]}
    """
    with open(cameras_path) as f:
        data = json.load(f)
    extrinsics = torch.tensor(data["extrinsics"]).float()  # [F, 4, 4] w2c
    Ks = torch.tensor(data["intrinsics"]).float()           # [F, 3, 3]
    c2ws = preprocess_poses(torch.linalg.inv(extrinsics))   # [F, 4, 4] c2w，第一帧归一化
    return c2ws, Ks


# ═══════════════════════════════════════════════════════════════════════════════
# 主推理函数（SDE 去噪循环，来自 infer_sde_test.py，适配 wild 输入）
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_sde_inference(pipeline, prompt, control_images, c2ws, Ks,
                       device, dtype, eta=0.2, num_inference_steps=50,
                       guidance_scale=5.0, shift=5.0, seed=42,
                       min_max_depth_mask=True):
    """使用 flux_step（eta>0 SDE）去噪推理。

    Args:
        control_images: [1, 1, 3, 560, 560] 仅第一帧为控制图像
        c2ws:           [F, 4, 4] camera-to-world，已 preprocess_poses
        Ks:             [F, 3, 3] 相机内参

    Returns:
        results dict（包含 rgbs, cameras, 可选 pcds）
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    F_ctrl = NUM_FRAMES
    B = 1
    H, W = TARGET_H, TARGET_W

    # 1. sigma schedule
    sigma_schedule = sd3_time_shift(
        shift, torch.linspace(1, 0, num_inference_steps + 1)
    ).to(device)

    # 2. 文本编码
    prompt_embeds, neg_embeds = pipeline.encode_prompt(
        prompt=prompt,
        negative_prompt="bad detailed",
        do_classifier_free_guidance=(guidance_scale > 1.0),
        num_videos_per_prompt=1,
        max_sequence_length=512,
        device=device,
    )
    do_cfg = guidance_scale > 1.0
    in_embeds = neg_embeds + prompt_embeds if do_cfg else prompt_embeds

    # 3. 控制条件：1view 模式，只有第0帧是控制帧
    ctrl_full = torch.cat(
        [control_images,
         torch.zeros(B, F_ctrl - 1, 3, H, W, device=device, dtype=dtype)],
        dim=1,
    )  # [1, F, 3, H, W]
    control_index = [0]
    spatial_ratio = pipeline.geo_adapter.spatial_compression_ratio
    masks = torch.zeros(
        B, F_ctrl, H // spatial_ratio, W // spatial_ratio * 2,
        device=device, dtype=dtype,
    )
    masks[:, control_index, :, : W // spatial_ratio] = 1
    masks_padded = torch.cat([
        torch.repeat_interleave(masks[:, :1], repeats=4, dim=1), masks[:, 1:],
    ], dim=1)
    f_lat = (F_ctrl + 3) // 4
    masks_lat = (
        masks_padded
        .view(B, f_lat, 4, *masks_padded.shape[-2:])
        .contiguous()
        .transpose(1, 2)  # [B, 4, f, h, w*2]
    )
    control_image_latents = pipeline.prepare_control_latents(
        ctrl_full, masks_lat, dtype=dtype, device=device,
    )  # [1, 20, f, h, w*2]

    # 4. Plücker camera latents（来自 infer_sde_test.py）
    rays_o, rays_d = compute_rays(
        c2ws.to(device), Ks.to(device), h=H, w=W, device=device
    )  # [F, 3, H, W]
    plucker = torch.cat([
        torch.cross(rays_o, rays_d, dim=1), rays_d
    ], dim=1).unsqueeze(0)  # [1, F, 6, H, W]

    cam = plucker.to(device, dtype)  # [1, F, 6, H, W]
    cam_lat = cam.transpose(1, 2)    # [1, 6, F, H, W]
    cam_lat = torch.cat([
        torch.repeat_interleave(cam_lat[:, :, 0:1], repeats=4, dim=2),
        cam_lat[:, :, 1:],
    ], dim=2).transpose(1, 2).contiguous()  # [1, F+3, 6, H, W]
    cam_lat = cam_lat.view(
        1, (NUM_FRAMES + 3) // 4, 4, cam_lat.shape[2], H, W
    ).transpose(2, 3).contiguous()
    cam_lat = cam_lat.view(
        1, (NUM_FRAMES + 3) // 4, cam_lat.shape[2] * 4, H, W,
    ).transpose(1, 2).contiguous()       # [1, 24, f, H, W]
    cam_lat = torch.cat([cam_lat, cam_lat], dim=-1)  # [1, 24, f, H, W*2]

    # 5. CLIP context
    clip_img_pil = Image.fromarray(
        (ctrl_full[:, 0].squeeze().permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
    )
    clip_img_t = TF.to_tensor(clip_img_pil).sub_(0.5).div_(0.5).to(device, dtype)
    clip_context = pipeline.clip_image_encoder([clip_img_t[:, None, :, :]])

    # 6. seq_len
    f_lat_size = (NUM_FRAMES - 1) // 4 + 1
    h_lat = H // spatial_ratio
    w_lat = W // spatial_ratio
    seq_len = math.ceil(
        (w_lat * 2 * h_lat) /
        (pipeline.transformer.config.patch_size[1] * pipeline.transformer.config.patch_size[2])
        * f_lat_size
    )

    # 7. 初始噪声
    latent_channels = pipeline.geo_adapter.model.z_dim
    z = torch.randn(
        1, latent_channels, f_lat_size, h_lat, w_lat * 2,
        device=device, dtype=dtype,
    )

    # 8. SDE 去噪循环
    for i in range(num_inference_steps):
        t = sigma_schedule[i]
        timestep = torch.tensor([int(t.item() * 1000)], device=device, dtype=torch.long)

        if do_cfg:
            z_in = torch.cat([z, z], dim=0)
            t_in = timestep.repeat(2)
            ctrl_in = torch.cat([control_image_latents, control_image_latents], dim=0)
            cam_in = torch.cat([cam_lat, cam_lat], dim=0)
            clip_in = torch.cat([clip_context, clip_context], dim=0)
            with torch.autocast("cuda", dtype=dtype):
                pred = pipeline.transformer(
                    x=z_in, context=in_embeds, t=t_in, seq_len=seq_len,
                    y=ctrl_in, y_camera=cam_in, clip_fea=clip_in,
                )
            pred_uncond, pred_cond = pred.chunk(2)
            pred = pred_uncond.float() + guidance_scale * (
                pred_cond.float() - pred_uncond.float()
            )
        else:
            with torch.autocast("cuda", dtype=dtype):
                pred = pipeline.transformer(
                    x=z, context=in_embeds, t=timestep, seq_len=seq_len,
                    y=control_image_latents, y_camera=cam_lat, clip_fea=clip_context,
                )
            pred = pred.float()

        z, _ = flux_step(pred, z.float(), eta, sigma_schedule, i,
                         grpo=False, sde_solver=True)
        z = z.to(dtype)

    # 9. 解码
    results = pipeline.decode_latents(z, min_max_depth_mask=min_max_depth_mask)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 保存结果（来自 infer_sde_test.py）
# ═══════════════════════════════════════════════════════════════════════════════

def save_results(output_dir, results, fps=24):
    os.makedirs(output_dir, exist_ok=True)

    if results.get("rgbs") is not None:
        rgb = rearrange(results["rgbs"], "b f h w c -> b c f h w").float().cpu()
        save_videos_grid(rgb, os.path.join(output_dir, "rgb.mp4"), rescale=False, fps=fps)
        print(f"  [saved] rgb.mp4 ({fps}fps)")

    if results.get("cameras") is not None:
        extrinsics, Ks = results["cameras"]
        extrinsics = torch.cat([
            extrinsics,
            torch.tensor([0, 0, 0, 1], device=extrinsics.device)
            .view(1, 1, 1, 4).repeat(extrinsics.shape[0], extrinsics.shape[1], 1, 1),
        ], dim=2)
        cameras_out = {
            "extrinsics": extrinsics[0].float().cpu().numpy().tolist(),
            "intrinsics": Ks[0].float().cpu().numpy().tolist(),
        }
        with open(os.path.join(output_dir, "cameras_gen3r.json"), "w") as f:
            json.dump(cameras_out, f, indent=4)
        print(f"  [saved] cameras_gen3r.json")


# ═══════════════════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════════════════

def main(args):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    # 收集需要处理的子目录（有 cameras.json 但没有 rgb.mp4 的）
    sample_dirs = sorted([
        d for d in RESULTS_DIR.iterdir()
        if d.is_dir()
        and (d / "cameras.json").exists()
        and (d / "input.png").exists()
    ])

    # 如果指定了 --names，只处理这些样本
    if args.names is not None:
        name_set = set(args.names)
        sample_dirs = [d for d in sample_dirs if d.name in name_set]

    if not sample_dirs:
        print("No sample directories found. Run generate_trajectories.py first.")
        return

    print(f"Found {len(sample_dirs)} samples to process.")
    for d in sample_dirs:
        has_video = (d / "rgb.mp4").exists()
        print(f"  {d.name}: {'DONE' if has_video else 'pending'}")

    print(f"\nLoading Gen3R pipeline from: {CHECKPOINTS}")
    pipeline = Gen3RPipeline.from_pretrained(CHECKPOINTS)
    for name, module in pipeline.components.items():
        if hasattr(module, "to") and module is not None:
            try:
                module.to(torch.bfloat16)
            except Exception:
                pass
    pipeline.enable_model_cpu_offload(gpu_id=0)
    device = pipeline._execution_device
    print(f"Pipeline loaded. Execution device: {device}")

    for sample_dir in sample_dirs:
        name = sample_dir.name
        rgb_path = sample_dir / "rgb.mp4"

        if rgb_path.exists():
            print(f"\n[{name}] rgb.mp4 already exists, skipping.")
            continue

        cameras_path = sample_dir / "cameras.json"
        input_path = sample_dir / "input.png"
        prompt_path = sample_dir / "prompt.txt"

        # 读取 prompt
        if prompt_path.exists():
            prompt = prompt_path.read_text().strip()
        else:
            prompt = "A video exploring an indoor scene from multiple camera angles."

        print(f"\n[{name}] Processing ...")
        print(f"  prompt: {prompt[:80]}...")

        # 加载预处理好的输入图片
        img_tensor = TF.to_tensor(Image.open(str(input_path)).convert("RGB"))  # [3, H, W]
        control_images = img_tensor.unsqueeze(0).unsqueeze(0).to(device, dtype)  # [1,1,3,H,W]

        # 加载相机轨迹
        c2ws, Ks = load_cameras_json(cameras_path, device, dtype)

        # SDE 推理
        try:
            results = run_sde_inference(
                pipeline=pipeline,
                prompt=prompt,
                control_images=control_images,
                c2ws=c2ws,
                Ks=Ks,
                device=device,
                dtype=dtype,
                eta=args.eta,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                shift=args.shift,
                seed=args.seed,
                min_max_depth_mask=True,
            )
            save_results(str(sample_dir), results)
            print(f"[{name}] Done -> {sample_dir}")
        except Exception as e:
            import traceback
            print(f"[{name}] ERROR: {e}")
            traceback.print_exc()
        finally:
            # 清理 GPU 缓存，防止连续样本间内存残留导致 OOM
            del control_images, c2ws, Ks
            if "results" in dir():
                del results
            torch.cuda.empty_cache()
            import gc
            gc.collect()

    print("\n=== All done ===")
    print("Summary:")
    for sample_dir in sample_dirs:
        ok = (sample_dir / "rgb.mp4").exists()
        print(f"  {sample_dir.name}: {'OK' if ok else 'FAILED'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gen3R wild test inference")
    parser.add_argument("--eta", type=float, default=0.2,
                        help="SDE noise strength (default: 0.2)")
    parser.add_argument("--steps", type=int, default=50,
                        help="Number of denoising steps (default: 50)")
    parser.add_argument("--guidance_scale", type=float, default=5.0,
                        help="CFG guidance scale (default: 5.0)")
    parser.add_argument("--shift", type=float, default=5.0,
                        help="Sigma schedule shift (default: 5.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--names", nargs="+", default=None,
                        help="Only process these sample names (e.g. --names shinei1 shinei2)")
    args = parser.parse_args()
    main(args)
