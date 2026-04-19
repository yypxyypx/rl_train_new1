"""grpo_engine.py — Gen3R GRPO 核心引擎。

负责三个阶段：
  Phase 1: rollout_and_decode  — 采样去噪轨迹 + VAE decode 生成视频
  Phase 2: compute_advantages  — 组内归一化 advantage
  Phase 3: grpo_update         — GRPO policy update + KL 散度正则化

算法：
  GRPO (Group Relative Policy Optimization)
  每组 G 条 rollout 共享同一样本，组内归一化 advantage：
      A_i = (r_i - mean(r)) / (std(r) + eps)

  Loss = -A_i * min(ratio_t, clip(ratio_t, 1-ε, 1+ε)) + β * KL(π_new || π_ref)

  其中 ratio_t = exp(log_π_new(a_t|s_t) - log_π_old(a_t|s_t))
       KL ≈ log_π_new - log_π_ref  （逐时间步估计）
"""

from __future__ import annotations

import math
import os
from typing import Callable, Optional

import torch
import torch.distributed as dist
from tqdm.auto import tqdm

from gen3r_encode import (
    build_plucker_embeds,
    decode_rgb_video,
    encode_control_latents,
    transformer_forward,
)
from grpo_core import flux_step, run_sample_step, sd3_time_shift


# ══════════════════════════════════════════════════════════════════════════════
# 常量
# ══════════════════════════════════════════════════════════════════════════════

IN_CHANNELS = 16
SPATIAL_DS = 8
TEMPORAL_DS = 4
WEIGHT_DTYPE = torch.bfloat16


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: Rollout + Decode
# ══════════════════════════════════════════════════════════════════════════════

def rollout_and_decode(
    args,
    models: dict,
    batch: dict,
    step: int,
    rank: int,
    device: torch.device,
) -> tuple[list[str], list, list, torch.Tensor, list]:
    """对 batch 内每个样本生成 num_generations 条 rollout，解码并保存视频。

    显存优化策略：
      1. 从预计算的 T5 embedding 缓存加载（不加载 T5 模型）
      2. CLIP 编码后立即 offload 到 CPU
      3. Transformer 采样一组 rollout 后 offload
      4. VAE decode 时其他模型在 CPU

    Args:
        args    : argparse.Namespace
        models  : {"wan_vae", "clip_image_encoder", "transformer", "ref_transformer"}
        batch   : collate_fn 输出
        step    : 当前训练步
        rank    : 当前进程 rank

    Returns:
        all_video_paths : list[str]，所有生成视频路径
        all_latents     : list of Tensor [1, K+1, 16, f, h, w*2]
        all_log_probs   : list of Tensor [1, K]
        sigma_schedule  : Tensor [T+1]
        encoded_conds   : list of dicts，每条 rollout 的编码条件（Phase 3 复用）
    """
    from model_loader import load_t5_embeds

    wan_vae = models["wan_vae"]
    clip_image_encoder = models["clip_image_encoder"]
    transformer = models["transformer"]

    sigma_schedule = sd3_time_shift(
        args.shift,
        torch.linspace(1, 0, args.sampling_steps + 1),
    )

    latent_t = ((args.num_frames - 1) // TEMPORAL_DS) + 1
    latent_h = args.resolution // SPATIAL_DS
    latent_w = args.resolution // SPATIAL_DS

    patch_h = transformer.config.patch_size[1] if hasattr(transformer.config, "patch_size") else 2
    patch_w = transformer.config.patch_size[2] if hasattr(transformer.config, "patch_size") else 2
    seq_len = math.ceil((latent_w * 2 * latent_h) / (patch_h * patch_w) * latent_t)

    texts = batch["text"]
    pixel_values_list = batch["pixel_values"]
    c2ws_list = batch["c2ws"]
    Ks_list = batch["Ks"]
    sample_ids = batch["sample_id"]
    dataset_names = batch["dataset_name"]
    camera_txt_paths = batch["camera_txt_path"]

    ng = args.num_generations
    all_video_paths: list[str] = []
    all_latents: list = []
    all_log_probs: list = []
    encoded_conds: list = []

    if args.init_same_noise:
        shared_noise = torch.randn(
            (1, IN_CHANNELS, latent_t, latent_h, latent_w * 2),
            device=device, dtype=WEIGHT_DTYPE,
        )

    for bi in range(len(texts)):
        sample_id = sample_ids[bi]
        dataset_name = dataset_names[bi]
        pixel_values = pixel_values_list[bi].to(device)
        c2ws = c2ws_list[bi].to(device)
        Ks = Ks_list[bi].to(device)
        F_frames = pixel_values.shape[0]
        camera_txt_path = camera_txt_paths[bi]

        # ── 加载预计算的 T5 embedding ────────────────────────────────────────
        prompt_embed_cpu, neg_embed_cpu = load_t5_embeds(args, sample_id, dataset_name)
        prompt_embeds = [prompt_embed_cpu.to(device=device, dtype=WEIGHT_DTYPE)]
        neg_embeds = [neg_embed_cpu.to(device=device, dtype=WEIGHT_DTYPE)]

        # ── 控制图像编码（CLIP + VAE encode），用后 offload ─────────────────
        # 上一个样本结束时 VAE/CLIP 状态可能已被搬到 CPU（offload 节省显存），
        # 这里在 encode 前显式拉回 GPU，避免「Input on CUDA, weight on CPU」错误。
        wan_vae.to(device=device, dtype=WEIGHT_DTYPE)
        clip_image_encoder.to(device=device, dtype=WEIGHT_DTYPE)

        control_index = [0]
        control_images = torch.zeros_like(pixel_values).unsqueeze(0)
        control_images[0, control_index] = pixel_values[control_index]

        with torch.no_grad():
            control_latents, clip_context = encode_control_latents(
                control_images, wan_vae, None, clip_image_encoder,
                control_index, F_frames, device, WEIGHT_DTYPE,
            )
        clip_image_encoder.cpu()
        torch.cuda.empty_cache()

        # ── Plücker ray embedding ────────────────────────────────────────────
        with torch.no_grad():
            plucker_embeds = build_plucker_embeds(
                c2ws, Ks, h=args.resolution, w=args.resolution,
                num_frames=F_frames, device=device, dtype=WEIGHT_DTYPE,
            )

        cond_base = dict(
            prompt_embeds=prompt_embeds,
            neg_embeds=neg_embeds,
            control_latents=control_latents,
            plucker_embeds=plucker_embeds,
            clip_context=clip_context,
            seq_len=seq_len,
            sample_id=sample_id,
            dataset_name=dataset_name,
            camera_txt_path=camera_txt_path,
            F=F_frames,
        )

        eval_dir = os.path.join(args.eval_output_dir, f"step_{step}", f"rank{rank}", sample_id)
        os.makedirs(eval_dir, exist_ok=True)

        # ── 生成 ng 条 rollout ───────────────────────────────────────────────
        transformer.to(device=device, dtype=WEIGHT_DTYPE)

        for gi in range(ng):
            if args.init_same_noise:
                z0 = shared_noise.clone()
            else:
                z0 = torch.randn(
                    (1, IN_CHANNELS, latent_t, latent_h, latent_w * 2),
                    device=device, dtype=WEIGHT_DTYPE,
                )

            with torch.no_grad():
                final_z, pred_x0, batch_latents, batch_log_probs = run_sample_step(
                    args, z0, sigma_schedule, transformer,
                    prompt_embeds, neg_embeds, seq_len,
                    control_latents, plucker_embeds, clip_context,
                    transformer_forward,
                )

            all_latents.append(batch_latents)
            all_log_probs.append(batch_log_probs)
            encoded_conds.append({**cond_base, "gi": gi})

            del final_z, z0

        # ── Transformer offload，再 VAE decode ──────────────────────────────
        transformer.cpu()
        torch.cuda.empty_cache()

        # 重新把 VAE 确保在 GPU
        wan_vae.to(device=device, dtype=WEIGHT_DTYPE)

        for gi in range(ng):
            video_path = os.path.join(eval_dir, f"gen_{gi}.mp4")
            pred_x0 = all_latents[len(all_latents) - ng + gi][:, -1]  # 最后一个 latent = pred_x0

            # 实际拿 run_sample_step 返回的 pred_x0 需要额外存储，用 latent 末帧代替
            # 这里重新从 latents 最后一帧 decode（与原逻辑一致）
            wan_latent = all_latents[len(all_latents) - ng + gi][:, -1, :, :, :latent_w]
            with torch.no_grad():
                decode_rgb_video(wan_latent, wan_vae, video_path)

            all_video_paths.append(video_path)
            torch.cuda.empty_cache()

        # VAE offload，为下一个样本准备
        wan_vae.cpu()
        torch.cuda.empty_cache()

        # CLIP 重新上 GPU 准备下一个样本
        clip_image_encoder.to(device=device, dtype=WEIGHT_DTYPE)

    # 全部完成后把 transformer 放回 GPU（Phase 3 用）
    transformer.to(device=device, dtype=WEIGHT_DTYPE)
    clip_image_encoder.to(device=device, dtype=WEIGHT_DTYPE)
    wan_vae.to(device=device, dtype=WEIGHT_DTYPE)

    return all_video_paths, all_latents, all_log_probs, sigma_schedule, encoded_conds


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: Advantage 计算
# ══════════════════════════════════════════════════════════════════════════════

def compute_advantages(rewards: torch.Tensor, num_generations: int) -> torch.Tensor:
    """GRPO 组内归一化 advantage。

    每 num_generations 条 rollout 为一组，组内做 (r - mean) / (std + eps)。

    Args:
        rewards : [N] float32
        num_generations : 每组大小（= num_generations）

    Returns:
        advantages : [N] float32
    """
    N = rewards.shape[0]
    n_groups = N // num_generations
    advantages = torch.zeros_like(rewards)
    for g in range(n_groups):
        s, e = g * num_generations, (g + 1) * num_generations
        grp = rewards[s:e]
        advantages[s:e] = (grp - grp.mean()) / (grp.std() + 1e-8)
    return advantages


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3: GRPO Policy Update + KL 散度
# ══════════════════════════════════════════════════════════════════════════════

def grpo_update(
    args,
    models: dict,
    all_latents: list,
    all_log_probs: list,
    advantages: torch.Tensor,
    sigma_schedule: torch.Tensor,
    encoded_conds: list,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    device: torch.device,
) -> tuple[float, float, float]:
    """GRPO policy update，返回 (total_loss, kl_mean, grad_norm)。

    Loss_t = -A_i * min(ratio_t, clip(ratio_t, 1-ε, 1+ε)) + β * KL_t
    KL_t ≈ new_log_prob - ref_log_prob  （high-order estimate）

    Args:
        all_latents   : list of Tensor [1, K+1, ...]，每条 rollout 的轨迹
        all_log_probs : list of Tensor [1, K]，rollout 时记录的 log_prob
        advantages    : [N] float32
    """
    transformer = models["transformer"]
    ref_transformer = models["ref_transformer"]

    strategy = getattr(args, "train_timestep_strategy", "random")
    N = len(all_latents)
    K = all_log_probs[0].shape[1]  # SDE 步数

    timestep_values = torch.tensor(
        [[int(sigma_schedule[t] * 1000) for t in range(K)] for _ in range(N)],
        device=device, dtype=torch.long,
    )

    if strategy == "front":
        train_T = K
        order = torch.arange(K, device=device).unsqueeze(0).expand(N, -1)
    else:
        train_T = max(1, int(K * args.timestep_fraction))
        order = torch.stack([torch.randperm(K) for _ in range(N)]).to(device)

    all_latents_t = torch.cat(all_latents, dim=0)     # [N, K+1, ...]
    all_log_probs_t = torch.cat(all_log_probs, dim=0)  # [N, K]

    idx_n = torch.arange(N, device=device)[:, None]
    latents_sel = all_latents_t[:, :-1][idx_n, order]
    next_latents_sel = all_latents_t[:, 1:][idx_n, order]
    log_probs_sel = all_log_probs_t[idx_n, order]
    timesteps_sel = timestep_values[idx_n, order]

    total_loss = 0.0
    total_kl = 0.0
    grad_norm_val = 0.0
    n_steps_total = 0
    optimizer.zero_grad()

    for i in range(N):
        cond = encoded_conds[i]
        adv_i = torch.clamp(
            advantages[i : i + 1],
            -args.adv_clip_max, args.adv_clip_max,
        )

        for t in range(train_T):
            abs_step = int(order[i][t].item())
            z_t = latents_sel[i : i + 1, t].to(device)
            z_next = next_latents_sel[i : i + 1, t].to(device)
            ts = timesteps_sel[i : i + 1, t]

            # ── 新策略 log_prob（有梯度） ─────────────────────────────────────
            transformer.train()
            new_pred = transformer_forward(
                transformer, z_t, ts,
                cond["prompt_embeds"], cond["neg_embeds"], cond["seq_len"],
                cond["control_latents"], cond["plucker_embeds"], cond["clip_context"],
                args.cfg_train, WEIGHT_DTYPE,
            )
            _, _, new_log_prob = flux_step(
                new_pred, z_t.to(torch.float32), args.eta,
                sigma_schedule, abs_step,
                prev_sample=z_next.to(torch.float32),
                grpo=True, sde_solver=True,
            )

            # ── 参考策略 log_prob（无梯度，KL 计算） ──────────────────────────
            kl_loss = torch.tensor(0.0, device=device)
            if args.kl_coeff > 0:
                with torch.no_grad():
                    ref_pred = transformer_forward(
                        ref_transformer, z_t, ts,
                        cond["prompt_embeds"], cond["neg_embeds"], cond["seq_len"],
                        cond["control_latents"], cond["plucker_embeds"], cond["clip_context"],
                        args.cfg_train, WEIGHT_DTYPE,
                    )
                    _, _, ref_log_prob = flux_step(
                        ref_pred, z_t.to(torch.float32), args.eta,
                        sigma_schedule, abs_step,
                        prev_sample=z_next.to(torch.float32),
                        grpo=True, sde_solver=True,
                    )
                # KL(π_new || π_ref) 用 log-ratio 近似
                kl_loss = (new_log_prob - ref_log_prob.detach()).mean()

            # ── GRPO Clipped Loss ──────────────────────────────────────────────
            old_log_prob = log_probs_sel[i : i + 1, t].to(device)
            ratio = torch.exp(new_log_prob - old_log_prob.detach())
            unclipped = -adv_i * ratio
            clipped = -adv_i * torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range)
            grpo_loss = torch.mean(torch.maximum(unclipped, clipped))

            loss = (grpo_loss + args.kl_coeff * kl_loss) / (args.gradient_accumulation_steps * train_T)
            loss.backward()

            avg_loss = loss.detach().clone()
            avg_kl = kl_loss.detach().clone()
            if dist.is_initialized():
                dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
                dist.all_reduce(avg_kl, op=dist.ReduceOp.AVG)

            total_loss += avg_loss.item() * (args.gradient_accumulation_steps * train_T)
            total_kl += avg_kl.item()
            n_steps_total += 1

        if (i + 1) % args.gradient_accumulation_steps == 0 or (i + 1) == N:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in transformer.parameters() if p.requires_grad],
                max_norm=args.max_grad_norm,
            )
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)

        if dist.is_initialized():
            dist.barrier()

    kl_mean = total_kl / max(n_steps_total, 1)
    return total_loss, kl_mean, grad_norm_val


# ══════════════════════════════════════════════════════════════════════════════
# 工具：模型 offload/reload
# ══════════════════════════════════════════════════════════════════════════════

def offload_rollout_models(models: dict) -> None:
    """Phase 1 结束，把 frozen 模型移到 CPU，为 Phase 3 腾出显存。"""
    for key in ("wan_vae", "clip_image_encoder"):
        m = models.get(key)
        if m is not None:
            m.cpu()
    torch.cuda.empty_cache()


def reload_rollout_models(models: dict, device, dtype=WEIGHT_DTYPE) -> None:
    """Phase 3 结束，把 frozen 模型移回 GPU，准备下一轮 rollout。"""
    for key in ("wan_vae", "clip_image_encoder"):
        m = models.get(key)
        if m is not None:
            m.to(device=device, dtype=dtype).eval()
