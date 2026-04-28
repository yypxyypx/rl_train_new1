"""grpo_engine.py — Wan2.2 GRPO 三阶段引擎。

  Phase 1: rollout_and_decode  — 采样去噪轨迹 + VAE decode 生成视频
  Phase 2: compute_advantages  — 组内归一化 advantage
  Phase 3: grpo_update         — GRPO policy update + KL 正则化

与 gen3r 的两处差异（用户指定）：
  1. 每轮 rollout **只触发 1 次** optimizer.step()（gen3r 在 num_generations=8、
     gradient_accumulation_steps=4 时会触发 2 次）。具体做法：把每步 backward
     的归一化分母改为 N*train_T，并在循环结束 i==N-1 时 step 一次。
  2. 形状常量改为 wan2.2 5B：IN_CHANNELS=48, SPATIAL_DS=16, TEMPORAL_DS=4。
"""

from __future__ import annotations

import math
import os
from typing import Optional

import torch
import torch.distributed as dist
from tqdm.auto import tqdm

from grpo_core import flux_step, run_sample_step, sd3_time_shift
from wan22_encode import (build_camera_control, chunk_camera_control,
                          decode_rgb_video, encode_inpaint_conditions,
                          transformer_forward)


# ══════════════════════════════════════════════════════════════════════════════
# Wan2.2 5B 形状常量
# ══════════════════════════════════════════════════════════════════════════════

IN_CHANNELS = 48        # AutoencoderKLWan3_8.latent_channels
SPATIAL_DS = 16
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

    显存策略：
      - T5 由预算 cache 加载（无 T5 模型常驻显存）。
      - VAE / transformer 之间 swap GPU/CPU。

    Returns:
        all_video_paths : list[str]  全部生成视频路径
        all_latents     : list of [1, K+1, 48, F_lat, H_lat, W_lat]
        all_log_probs   : list of [1, K]
        sigma_schedule  : Tensor [T+1]
        encoded_conds   : list of dict（Phase 3 复用）
    """
    from model_loader import load_t5_embeds

    vae = models["vae"]
    transformer = models["transformer"]

    sigma_schedule = sd3_time_shift(
        args.shift,
        torch.linspace(1, 0, args.sampling_steps + 1),
    )

    H, W = args.resolution_h, args.resolution_w
    latent_t = ((args.num_frames - 1) // TEMPORAL_DS) + 1
    latent_h = H // SPATIAL_DS
    latent_w = W // SPATIAL_DS

    patch = transformer.config.patch_size if hasattr(transformer, "config") else (1, 2, 2)
    patch_h, patch_w = patch[1], patch[2]
    seq_len = math.ceil((latent_h * latent_w) / (patch_h * patch_w) * latent_t)

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
            (1, IN_CHANNELS, latent_t, latent_h, latent_w),
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

        # ── 加载预计算 T5 ───────────────────────────────────────────────────
        prompt_embed_cpu, neg_embed_cpu = load_t5_embeds(args, sample_id, dataset_name)
        prompt_embeds = [prompt_embed_cpu.to(device=device, dtype=WEIGHT_DTYPE)]
        neg_embeds = [neg_embed_cpu.to(device=device, dtype=WEIGHT_DTYPE)]

        # ── 控制条件编码 ────────────────────────────────────────────────────
        vae.to(device=device, dtype=WEIGHT_DTYPE)

        with torch.no_grad():
            control_latents, masked_video_lat, latent_mask = encode_inpaint_conditions(
                pixel_values[0], vae, F_frames, H, W, device, WEIGHT_DTYPE,
            )
            plucker_fchw = build_camera_control(c2ws, Ks, H, W, device, WEIGHT_DTYPE)
            control_camera_latents = chunk_camera_control(plucker_fchw, F_frames)

        cond_base = dict(
            prompt_embeds=prompt_embeds,
            neg_embeds=neg_embeds,
            control_latents=control_latents,
            control_camera_latents=control_camera_latents,
            masked_video_lat=masked_video_lat,
            latent_mask=latent_mask,
            seq_len=seq_len,
            sample_id=sample_id,
            dataset_name=dataset_name,
            camera_txt_path=camera_txt_path,
            F=F_frames,
        )

        eval_dir = os.path.join(args.eval_output_dir, f"step_{step}", f"rank{rank}", sample_id)
        os.makedirs(eval_dir, exist_ok=True)

        # ── 生成 ng 条 rollout ──────────────────────────────────────────────
        transformer.to(device=device, dtype=WEIGHT_DTYPE)

        rollout_final_z = []
        for gi in range(ng):
            if args.init_same_noise:
                z0 = shared_noise.clone()
            else:
                z0 = torch.randn(
                    (1, IN_CHANNELS, latent_t, latent_h, latent_w),
                    device=device, dtype=WEIGHT_DTYPE,
                )

            with torch.no_grad():
                final_z, _pred_x0, batch_latents, batch_log_probs = run_sample_step(
                    args, z0, sigma_schedule, transformer,
                    prompt_embeds, neg_embeds, seq_len,
                    control_latents, control_camera_latents,
                    masked_video_lat, latent_mask,
                    transformer_forward,
                )

            all_latents.append(batch_latents)
            all_log_probs.append(batch_log_probs)
            encoded_conds.append({**cond_base, "gi": gi})
            # 用 final_z（含 mask_clamp）解码，对齐 official pipeline；
            # **不要**用 pred_x0：那是 latents-sigma*model_output，没有 mask_clamp，
            # 会导致首帧颜色错位、整体漂移。
            rollout_final_z.append(final_z.detach())

            del _pred_x0, z0

        # ── transformer 让位给 VAE decode ───────────────────────────────────
        transformer.cpu()
        torch.cuda.empty_cache()
        vae.to(device=device, dtype=WEIGHT_DTYPE)

        for gi in range(ng):
            video_path = os.path.join(eval_dir, f"gen_{gi}.mp4")
            with torch.no_grad():
                decode_rgb_video(rollout_final_z[gi], vae, video_path)
            all_video_paths.append(video_path)
            torch.cuda.empty_cache()

        # 把 transformer 放回 GPU 准备 Phase 3
        vae.cpu()
        torch.cuda.empty_cache()
        transformer.to(device=device, dtype=WEIGHT_DTYPE)

    return all_video_paths, all_latents, all_log_probs, sigma_schedule, encoded_conds


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: Advantage
# ══════════════════════════════════════════════════════════════════════════════

def compute_advantages(rewards: torch.Tensor, num_generations: int) -> torch.Tensor:
    """组内归一化 advantage：每 num_generations 条为一组。"""
    N = rewards.shape[0]
    n_groups = N // num_generations
    advantages = torch.zeros_like(rewards)
    for g in range(n_groups):
        s, e = g * num_generations, (g + 1) * num_generations
        grp = rewards[s:e]
        advantages[s:e] = (grp - grp.mean()) / (grp.std() + 1e-8)
    return advantages


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3: GRPO Update（每轮 rollout **只触发 1 次** optimizer.step()）
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
    """GRPO policy update（单步更新版）。

    与 gen3r 的差异：
      - loss 归一化分母从 `gradient_accumulation_steps * train_T` 改为
        `N * train_T`，等价于"先 mean over rollout，再 mean over time-step"。
      - optimizer.step() 改为只在最后一条 rollout 处理完后触发一次。

    Returns:
        (total_loss, kl_mean, grad_norm)
    """
    transformer = models["transformer"]
    ref_transformer = models["ref_transformer"]

    strategy = getattr(args, "train_timestep_strategy", "random")
    N = len(all_latents)
    K = all_log_probs[0].shape[1]

    timestep_values = torch.tensor(
        [[float(sigma_schedule[t] * 1000) for t in range(K)] for _ in range(N)],
        device=device, dtype=torch.float32,
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

    norm_denom = float(N * train_T)  # 单次更新：整组 N 条 rollout × train_T 步

    for i in range(N):
        cond = encoded_conds[i]
        adv_i = torch.clamp(
            advantages[i: i + 1],
            -args.adv_clip_max, args.adv_clip_max,
        )

        for t in range(train_T):
            abs_step = int(order[i][t].item())
            z_t = latents_sel[i: i + 1, t].to(device)
            z_next = next_latents_sel[i: i + 1, t].to(device)
            ts = float(timesteps_sel[i, t].item())

            # ── 新策略 log_prob（含梯度） ─────────────────────────────────
            transformer.train()
            new_pred = transformer_forward(
                transformer, z_t, ts,
                cond["prompt_embeds"], cond["neg_embeds"], cond["seq_len"],
                cond["control_latents"], cond["control_camera_latents"], cond["latent_mask"],
                args.cfg_train, WEIGHT_DTYPE,
            )
            _, _, new_log_prob = flux_step(
                new_pred, z_t.to(torch.float32), args.eta,
                sigma_schedule, abs_step,
                prev_sample=z_next.to(torch.float32),
                grpo=True, sde_solver=True,
            )

            # ── 参考策略 log_prob（KL 项） ────────────────────────────────
            kl_loss = torch.tensor(0.0, device=device)
            if args.kl_coeff > 0 and ref_transformer is not None:
                with torch.no_grad():
                    ref_pred = transformer_forward(
                        ref_transformer, z_t, ts,
                        cond["prompt_embeds"], cond["neg_embeds"], cond["seq_len"],
                        cond["control_latents"], cond["control_camera_latents"], cond["latent_mask"],
                        args.cfg_train, WEIGHT_DTYPE,
                    )
                    _, _, ref_log_prob = flux_step(
                        ref_pred, z_t.to(torch.float32), args.eta,
                        sigma_schedule, abs_step,
                        prev_sample=z_next.to(torch.float32),
                        grpo=True, sde_solver=True,
                    )
                kl_loss = (new_log_prob - ref_log_prob.detach()).mean()

            # ── GRPO Clipped Loss ──────────────────────────────────────────
            old_log_prob = log_probs_sel[i: i + 1, t].to(device)
            ratio = torch.exp(new_log_prob - old_log_prob.detach())
            unclipped = -adv_i * ratio
            clipped = -adv_i * torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range)
            grpo_loss = torch.mean(torch.maximum(unclipped, clipped))

            # **关键改动**：归一化分母 = N * train_T （单次更新）
            loss = (grpo_loss + args.kl_coeff * kl_loss) / norm_denom
            loss.backward()

            avg_loss = loss.detach().clone()
            avg_kl = kl_loss.detach().clone()
            if dist.is_initialized():
                dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
                dist.all_reduce(avg_kl, op=dist.ReduceOp.AVG)

            total_loss += avg_loss.item() * norm_denom  # 还原成 step-level 平均
            total_kl += avg_kl.item()
            n_steps_total += 1

        # **关键改动**：只在最后一条 rollout 处理完后触发 optimizer.step()
        if (i + 1) == N:
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
# 工具
# ══════════════════════════════════════════════════════════════════════════════

def offload_rollout_models(models: dict) -> None:
    """Phase 1 结束，把 frozen 模型移到 CPU。"""
    for key in ("vae",):
        m = models.get(key)
        if m is not None:
            m.cpu()
    torch.cuda.empty_cache()


def reload_rollout_models(models: dict, device, dtype=WEIGHT_DTYPE) -> None:
    """Phase 3 结束，把 frozen 模型移回 GPU。"""
    for key in ("vae",):
        m = models.get(key)
        if m is not None:
            m.to(device=device, dtype=dtype).eval()
