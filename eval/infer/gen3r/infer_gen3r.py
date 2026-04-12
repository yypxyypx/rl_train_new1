#!/usr/bin/env python3
"""Gen3R official pipeline inference for evaluation.

Reads processed data (start.png, camera.txt, metadata.json) produced by
data/unified_data_process.py, runs Gen3R SDE inference with a given
checkpoint, and outputs gen_0.mp4 ~ gen_{N-1}.mp4 in a format aligned
with the eval/reward pipeline.

Supports:
  - manifest.jsonl input (batch mode) or single sample directory
  - Multi-GPU via torchrun (rollouts distributed round-robin across GPUs)
  - Resumable via --skip_done
  - Custom checkpoint path (for evaluating RL-trained checkpoints)

Output per sample: <output_root>/<dataset>/<sample_id>/
  gen_0.mp4 ~ gen_{N-1}.mp4   generated rollouts
  start.png, gt.mp4, camera.txt, gt_depth.npz   copied from input
  infer_info.json              inference parameters

Usage:
    torchrun --nproc_per_node=4 infer_gen3r.py \\
        --manifest /path/to/manifest.jsonl \\
        --checkpoint /path/to/gen3r_checkpoints \\
        --output_root /path/to/output \\
        [--num_rollouts 8] [--eta 0.3] [--num_inference_steps 50]

    # Single sample:
    python infer_gen3r.py \\
        --sample_dir /path/to/processed/re10k/scene_001 \\
        --checkpoint /path/to/gen3r_checkpoints \\
        --output_root /path/to/output
"""

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

import imageio
import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange
from torchvision.transforms.functional import resize

# ─── Gen3R imports ────────────────────────────────────────────────────────────
# Expects gen3r/Gen3R to be available. Set GEN3R_ROOT env var or place
# this script so that the relative path works.

_HERE = Path(__file__).resolve().parent
_GEN3R_ROOT = Path(os.environ.get(
    "GEN3R_ROOT",
    str(_HERE / "Gen3R"),
))
if str(_GEN3R_ROOT) not in sys.path:
    sys.path.insert(0, str(_GEN3R_ROOT))

from gen3r.pipeline import Gen3RPipeline  # noqa: E402
from gen3r.utils.data_utils import (  # noqa: E402
    center_crop, compute_rays, preprocess_poses, get_K,
)
from gen3r.utils.common_utils import save_videos_grid  # noqa: E402


# ─── decode_latents device-alignment patch ────────────────────────────────────
_orig_decode = Gen3RPipeline.decode_latents


def _decode_aligned(self, latents, min_max_depth_mask=False):
    dev, dt = latents.device, torch.bfloat16
    for mod in [self.geo_adapter, self.vggt, self.wan_vae]:
        if mod is not None:
            mod.to(device=dev, dtype=dt)
    return _orig_decode(self, latents, min_max_depth_mask=min_max_depth_mask)


Gen3RPipeline.decode_latents = _decode_aligned


# ═══════════════════════ SDE Sampler ═══════════════════════


def sd3_time_shift(shift, t):
    return (shift * t) / (1 + (shift - 1) * t)


def flux_step(model_output, latents, eta, sigmas, index, sde_solver=True):
    sigma = sigmas[index]
    dsigma = sigmas[index + 1] - sigma
    prev_sample_mean = latents + dsigma * model_output
    pred_original_sample = latents - sigma * model_output

    delta_t = sigma - sigmas[index + 1]
    std_dev_t = eta * math.sqrt(delta_t.item())

    if sde_solver:
        score_estimate = -(latents - pred_original_sample * (1 - sigma)) / sigma ** 2
        prev_sample_mean = prev_sample_mean + (-0.5 * eta ** 2 * score_estimate) * dsigma

    out = (prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t
           if std_dev_t > 0 else prev_sample_mean)
    return out, pred_original_sample


# ═══════════════════════ Data Loading ═══════════════════════


def parse_camera_txt(camera_file, img_w, img_h, target_w, target_h):
    """Parse camera.txt → w2c extrinsics (F,4,4) and intrinsics (F,3,3)."""
    ext_list, int_list = [], []
    with open(camera_file, "r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        vals = list(map(float, line.split()))
        fl_x = vals[1] * img_w
        fl_y = vals[2] * img_h
        cx = vals[3] * img_w
        cy = vals[4] * img_h
        mat34 = np.array(vals[7:19]).reshape(3, 4)
        mat44 = np.eye(4, dtype=np.float64)
        mat44[:3, :] = mat34
        ext_list.append(mat44)
        K = get_K(img_w, img_h, fl_x, fl_y, cx, cy, target_w, target_h)
        int_list.append(K.numpy())
    return np.array(ext_list), np.array(int_list)


def prepare_inputs(sample_dir: Path, prompt: str, device, dtype,
                   num_frames: int, target_h: int, target_w: int):
    """Read a processed sample directory and prepare Gen3R inputs.

    Returns (prompt, ctrl, plucker, c2ws).
    """
    meta_path = sample_dir / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        img_w = meta["img_w"]
        img_h = meta["img_h"]
        if not prompt:
            prompt = meta.get("caption", meta.get("prompt", "")).strip()
    else:
        raw = imageio.v2.imread(str(sample_dir / "start.png"))
        img_h, img_w = raw.shape[:2]

    # Control image
    start_arr = imageio.v2.imread(str(sample_dir / "start.png"))[..., :3]
    ctrl = torch.from_numpy(start_arr).unsqueeze(0).permute(0, 3, 1, 2).unsqueeze(0).float() / 255.0
    fh, fw = ctrl.shape[3], ctrl.shape[4]
    scale = target_h / min(fh, fw)
    ctrl = resize(ctrl[0], [round(fh * scale), round(fw * scale)])
    ctrl = center_crop(ctrl, (target_h, target_w)).unsqueeze(0).to(device, dtype)

    # Camera
    exts_np, ints_np = parse_camera_txt(
        sample_dir / "camera.txt", img_w, img_h, target_w, target_h)

    if len(exts_np) < num_frames:
        pad = num_frames - len(exts_np)
        exts_np = np.concatenate([exts_np, np.tile(exts_np[-1:], (pad, 1, 1))], axis=0)
        ints_np = np.concatenate([ints_np, np.tile(ints_np[-1:], (pad, 1, 1))], axis=0)
    else:
        exts_np = exts_np[:num_frames]
        ints_np = ints_np[:num_frames]

    Ks = torch.from_numpy(ints_np).float()
    ext_t = torch.from_numpy(exts_np).float()
    c2w_abs = torch.linalg.inv(ext_t)

    # camera.txt stores w2c from OpenCV c2w, so c2w is already OpenCV — no flip needed
    c2ws = preprocess_poses(c2w_abs).unsqueeze(0).to(device, dtype)

    rays_o, rays_d = compute_rays(c2ws[0].cpu().float(), Ks, h=target_h, w=target_w, device="cpu")
    plucker = torch.cat([torch.cross(rays_o, rays_d, dim=1), rays_d], dim=1).unsqueeze(0)
    plucker = plucker.to(device, dtype)

    return prompt, ctrl, plucker, c2ws


# ═══════════════════════ SDE Inference ═══════════════════════


@torch.no_grad()
def run_sde_inference(
    pipeline, prompt, control_images, control_cameras,
    device, dtype,
    eta, num_inference_steps, guidance_scale, shift, seed, num_frames,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    sigma_schedule = sd3_time_shift(
        shift, torch.linspace(1, 0, num_inference_steps + 1),
    ).to(device)

    prompt_embeds, neg_embeds = pipeline.encode_prompt(
        prompt=prompt, negative_prompt="bad detailed",
        do_classifier_free_guidance=(guidance_scale > 1.0),
        num_videos_per_prompt=1, max_sequence_length=512, device=device,
    )
    do_cfg = guidance_scale > 1.0
    in_embeds = neg_embeds + prompt_embeds if do_cfg else prompt_embeds

    F_ctrl = control_cameras.shape[1]
    B, _, C, H, W = control_images.shape
    ctrl_full = torch.cat(
        [control_images,
         torch.zeros(B, F_ctrl - 1, C, H, W, device=device, dtype=dtype)],
        dim=1,
    )

    spatial_ratio = pipeline.geo_adapter.spatial_compression_ratio
    masks = torch.zeros(B, F_ctrl, H // spatial_ratio, W // spatial_ratio * 2,
                        device=device, dtype=dtype)
    masks[:, [0], :, :W // spatial_ratio] = 1
    masks_padded = torch.cat(
        [torch.repeat_interleave(masks[:, :1], 4, dim=1), masks[:, 1:]], dim=1,
    )
    f_lat = (F_ctrl + 3) // 4
    masks_lat = (masks_padded
                 .view(B, f_lat, 4, *masks_padded.shape[-2:])
                 .contiguous().transpose(1, 2))
    control_image_latents = pipeline.prepare_control_latents(
        ctrl_full, masks_lat, dtype=dtype, device=device,
    )

    cam = control_cameras.transpose(1, 2)
    cam = torch.cat(
        [torch.repeat_interleave(cam[:, :, 0:1], 4, dim=2), cam[:, :, 1:]], dim=2,
    ).transpose(1, 2).contiguous()
    cam = (cam.view(1, (num_frames + 3) // 4, 4, cam.shape[2], H, W)
           .transpose(2, 3).contiguous())
    cam_lat = (cam.view(1, (num_frames + 3) // 4, cam.shape[2] * 4, H, W)
               .transpose(1, 2).contiguous())
    cam_lat = torch.cat([cam_lat, cam_lat], dim=-1)

    from PIL import Image
    import torchvision.transforms.functional as TF
    clip_pil = Image.fromarray(
        (ctrl_full[:, 0].squeeze().permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
    )
    clip_t = TF.to_tensor(clip_pil).sub_(0.5).div_(0.5).to(device, dtype)
    clip_context = pipeline.clip_image_encoder([clip_t[:, None, :, :]])

    f_lat_size = (num_frames - 1) // 4 + 1
    h_lat = H // spatial_ratio
    w_lat = W // spatial_ratio
    patch_h = pipeline.transformer.config.patch_size[1]
    patch_w = pipeline.transformer.config.patch_size[2]
    seq_len = math.ceil((w_lat * 2 * h_lat) / (patch_h * patch_w) * f_lat_size)

    latent_ch = pipeline.geo_adapter.model.z_dim
    z = torch.randn(1, latent_ch, f_lat_size, h_lat, w_lat * 2, device=device, dtype=dtype)

    pred_x0 = None
    for i in range(num_inference_steps):
        t = sigma_schedule[i]
        timestep = torch.tensor([int(t.item() * 1000)], device=device, dtype=torch.long)

        if do_cfg:
            z_in = torch.cat([z, z])
            t_in = timestep.repeat(2)
            ctrl_in = torch.cat([control_image_latents, control_image_latents])
            cam_in = torch.cat([cam_lat, cam_lat])
            clip_in = torch.cat([clip_context, clip_context])
            with torch.autocast("cuda", dtype=dtype):
                pred = pipeline.transformer(
                    x=z_in, context=in_embeds, t=t_in, seq_len=seq_len,
                    y=ctrl_in, y_camera=cam_in, clip_fea=clip_in,
                )
            pred_u, pred_c = pred.chunk(2)
            pred = pred_u.float() + guidance_scale * (pred_c.float() - pred_u.float())
        else:
            with torch.autocast("cuda", dtype=dtype):
                pred = pipeline.transformer(
                    x=z, context=in_embeds, t=timestep, seq_len=seq_len,
                    y=control_image_latents, y_camera=cam_lat, clip_fea=clip_context,
                )
            pred = pred.float()

        z, pred_x0 = flux_step(pred, z.float(), eta, sigma_schedule, i, sde_solver=True)
        z = z.to(dtype)

    return pipeline.decode_latents(pred_x0.to(dtype), min_max_depth_mask=False)


# ═══════════════════════ Save ═══════════════════════


def save_rollout(output_dir: Path, results: dict, rollout_idx: int) -> bool:
    if results.get("rgbs") is None:
        return False
    rgb = rearrange(results["rgbs"], "b f h w c -> b c f h w").float().cpu()
    save_videos_grid(rgb, str(output_dir / f"gen_{rollout_idx}.mp4"), rescale=False)
    return True


def copy_input_files(sample_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for fname in ["start.png", "gt.mp4", "camera.txt", "gt_depth.npz", "metadata.json"]:
        src = sample_dir / fname
        dst = output_dir / fname
        if src.exists() and not dst.exists():
            shutil.copy2(str(src), str(dst))


# ═══════════════════════ Entry Points ═══════════════════════


def load_manifest(manifest_path: str) -> list:
    return [
        json.loads(line)
        for line in Path(manifest_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_manifest_from_dir(sample_dir: str) -> list:
    """Build a single-entry manifest from a processed sample directory."""
    p = Path(sample_dir)
    meta_path = p / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return [{
        "dataset": meta.get("dataset", p.parent.name),
        "sample_id": meta.get("orig_id", p.name).replace("/", "_"),
        "sample_dir": str(p),
        "prompt": meta.get("caption", ""),
    }]


# ═══════════════════════ Main ═══════════════════════


def main(args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)

    # Build entries list
    if args.manifest:
        entries = load_manifest(args.manifest)
    elif args.sample_dir:
        entries = build_manifest_from_dir(args.sample_dir)
    else:
        raise ValueError("Must specify --manifest or --sample_dir")

    if rank == 0:
        print(f"\n[Gen3R Infer] samples={len(entries)}  rollouts={args.num_rollouts}")
        print(f"  checkpoint: {args.checkpoint}")
        print(f"  eta={args.eta}  steps={args.num_inference_steps}  "
              f"cfg={args.guidance_scale}  shift={args.shift}")
        print(f"  output: {args.output_root}\n")

    # Load pipeline
    pipeline = Gen3RPipeline.from_pretrained(args.checkpoint)
    for _, mod in pipeline.components.items():
        if hasattr(mod, "to") and mod is not None:
            try:
                mod.to(torch.bfloat16)
            except Exception:
                pass

    if args.device_mode == "local":
        pipeline.enable_model_cpu_offload(gpu_id=local_rank)
        device = pipeline._execution_device
    else:
        device = torch.device(f"cuda:{local_rank}")
        pipeline = pipeline.to(device)

    dtype = torch.bfloat16
    my_rollouts = list(range(rank, args.num_rollouts, world_size))

    for entry_idx, entry in enumerate(entries):
        dataset = entry.get("dataset", "unknown")
        sample_id = entry.get("sample_id", f"sample_{entry_idx}")
        sample_dir = Path(entry["sample_dir"])
        prompt = entry.get("prompt", "")

        output_dir = Path(args.output_root) / dataset / sample_id

        if args.skip_done:
            all_done = all(
                (output_dir / f"gen_{i}.mp4").exists()
                for i in range(args.num_rollouts)
            )
            if all_done:
                if rank == 0:
                    print(f"[{entry_idx+1}/{len(entries)}] skip (done): {dataset}/{sample_id}")
                if world_size > 1:
                    dist.barrier()
                continue

        if rank == 0:
            print(f"\n[{entry_idx+1}/{len(entries)}] {dataset}/{sample_id}")
            copy_input_files(sample_dir, output_dir)

        if world_size > 1:
            dist.barrier()

        try:
            prompt_used, ctrl, plucker, c2ws = prepare_inputs(
                sample_dir, prompt, device, dtype,
                args.num_frames, args.target_size, args.target_size,
            )
        except Exception as e:
            print(f"[Rank {rank}] input error: {e}")
            if world_size > 1:
                dist.barrier()
            continue

        for rollout_idx in my_rollouts:
            out_video = output_dir / f"gen_{rollout_idx}.mp4"
            if args.skip_done and out_video.exists():
                continue

            seed = args.base_seed + entry_idx * args.num_rollouts + rollout_idx
            print(f"  [Rank {rank}] gen_{rollout_idx}.mp4  seed={seed}")
            try:
                results = run_sde_inference(
                    pipeline=pipeline, prompt=prompt_used,
                    control_images=ctrl, control_cameras=plucker,
                    device=device, dtype=dtype,
                    eta=args.eta, num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale, shift=args.shift,
                    seed=seed, num_frames=args.num_frames,
                )
                if save_rollout(output_dir, results, rollout_idx):
                    print(f"  [Rank {rank}] gen_{rollout_idx}.mp4 saved")
            except Exception as e:
                print(f"  [Rank {rank}] gen_{rollout_idx} failed: {e}")

        if rank == 0:
            info = {
                "checkpoint": args.checkpoint,
                "num_rollouts": args.num_rollouts,
                "eta": args.eta,
                "num_inference_steps": args.num_inference_steps,
                "guidance_scale": args.guidance_scale,
                "shift": args.shift,
                "num_frames": args.num_frames,
                "target_size": args.target_size,
                "base_seed": args.base_seed,
            }
            (output_dir / "infer_info.json").write_text(
                json.dumps(info, indent=2), encoding="utf-8",
            )

        if world_size > 1:
            dist.barrier()

    if rank == 0:
        print(f"\n[Gen3R Infer] Done. Output: {args.output_root}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gen3R official pipeline inference")

    # Input (one of these required)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--manifest", type=str, help="manifest.jsonl path")
    input_group.add_argument("--sample_dir", type=str, help="Single processed sample directory")

    # Model
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Gen3R checkpoint directory")
    parser.add_argument("--output_root", type=str, required=True)

    # Inference params
    parser.add_argument("--num_rollouts", type=int, default=8)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--target_size", type=int, default=560)
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--shift", type=float, default=2.0)
    parser.add_argument("--base_seed", type=int, default=42)

    # Runtime
    parser.add_argument("--skip_done", action="store_true")
    parser.add_argument("--device_mode", choices=["local", "server"], default="server",
                        help="local=CPU offload, server=full GPU load")

    main(parser.parse_args())
