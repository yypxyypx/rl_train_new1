"""grpo_core.py — Wan2.2 GRPO 核心算法。

复用 gen3r [grpo_core.py](rl_train_new/rl_train/train/gen3r/grpo_core.py) 的 SDE/ODE
逐元素去噪逻辑（sd3_time_shift / flux_step），仅替换 transformer 调用接口为
wan22_encode.transformer_forward；并在每步去噪后调用 `apply_mask_clamp` 把
首帧 latent 强制对齐到 GT，对应官方 pipeline 中 spatial_compression_ratio>=16
+ I2V 的标准做法。

包含：
    sd3_time_shift   — 与 gen3r 同
    flux_step        — 与 gen3r 同（任意 latent shape）
    run_sample_step  — 完整去噪 rollout（hybrid SDE/ODE）
    grpo_one_step    — 训练模式单步重计算 log_prob
"""

from __future__ import annotations

import math

import torch
from tqdm.auto import tqdm

from wan22_encode import apply_mask_clamp


# ══════════════════════════════════════════════════════════════════════════════
# Sigma Schedule
# ══════════════════════════════════════════════════════════════════════════════

def sd3_time_shift(shift: float, t: torch.Tensor) -> torch.Tensor:
    """非线性 sigma schedule 偏移（来自 DanceGRPO）。"""
    return (shift * t) / (1 + (shift - 1) * t)


# ══════════════════════════════════════════════════════════════════════════════
# flux_step（与 gen3r 完全一致，逐元素操作支持任意 latent shape）
# ══════════════════════════════════════════════════════════════════════════════

def flux_step(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    eta: float,
    sigmas: torch.Tensor,
    index: int,
    prev_sample: torch.Tensor | None,
    grpo: bool,
    sde_solver: bool,
):
    """ODE→SDE 单步转换 + log_prob（来自 DanceGRPO，无修改）。"""
    sigma = sigmas[index]
    dsigma = sigmas[index + 1] - sigma
    prev_sample_mean = latents + dsigma * model_output
    pred_original_sample = latents - sigma * model_output

    delta_t = sigma - sigmas[index + 1]
    std_dev_t = eta * math.sqrt(delta_t)

    if sde_solver:
        score_estimate = -(latents - pred_original_sample * (1 - sigma)) / sigma ** 2
        log_term = -0.5 * eta ** 2 * score_estimate
        prev_sample_mean = prev_sample_mean + log_term * dsigma

    if grpo and prev_sample is None:
        prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t

    if grpo:
        log_prob = (
            -((prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32)) ** 2)
            / (2 * (std_dev_t ** 2))
            - math.log(std_dev_t)
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        return prev_sample, pred_original_sample, log_prob
    else:
        return prev_sample_mean, pred_original_sample


# ══════════════════════════════════════════════════════════════════════════════
# run_sample_step：完整 Rollout
# ══════════════════════════════════════════════════════════════════════════════

def run_sample_step(
    args,
    z: torch.Tensor,
    sigma_schedule: torch.Tensor,
    transformer,
    prompt_embeds: list,
    neg_embeds: list,
    seq_len: int,
    control_latents: torch.Tensor,
    control_camera_latents: torch.Tensor,
    masked_video_lat: torch.Tensor,
    latent_mask: torch.Tensor,
    transformer_forward_fn,
):
    """完整去噪 Rollout，支持 hybrid SDE/ODE 模式（与 gen3r 同构）。

    与 gen3r 的关键差异：
      1. 删除 plucker / clip_fea 参数，改为传 control_camera_latents（已 chunk）；
      2. 每次 flux_step 之后 apply_mask_clamp，把首帧 latent 锁到 GT；
      3. 初始噪声 z 进入循环前也先 clamp 一次。

    Returns:
        final_z, pred_x0, all_latents [1, K+1, ...], all_log_probs [1, K]
    """
    T = args.sampling_steps
    strategy = getattr(args, "train_timestep_strategy", "random")
    sde_fraction = getattr(args, "sde_fraction", 1.0)
    # eta=0 → SDE 退化为 ODE，不能算 log_prob（math.log(0)）。推理时强制全 ODE。
    if args.eta <= 0:
        sde_steps = 0
    else:
        sde_steps = max(1, int(T * sde_fraction)) if strategy == "front" else T

    # 初始 latent clamp（首帧 = masked_video_lat 第一帧）
    z = apply_mask_clamp(z.to(torch.float32), masked_video_lat.to(torch.float32),
                         latent_mask.to(torch.float32)).to(z.dtype)

    all_latents = [z]
    all_log_probs = []

    cfg_rollout = args.cfg_rollout

    for i in tqdm(range(T), desc="Sampling", leave=False):
        sigma = sigma_schedule[i]
        timestep_val = float(sigma * 1000)

        transformer.eval()
        with torch.no_grad():
            pred = transformer_forward_fn(
                transformer, z, timestep_val,
                prompt_embeds, neg_embeds, seq_len,
                control_latents, control_camera_latents, latent_mask,
                cfg_rollout, torch.bfloat16,
            )

        if i < sde_steps:
            z, pred_original, log_prob = flux_step(
                pred, z.to(torch.float32), args.eta,
                sigma_schedule, i, prev_sample=None,
                grpo=True, sde_solver=True,
            )
        else:
            z, pred_original = flux_step(
                pred, z.to(torch.float32), 0.0,
                sigma_schedule, i, prev_sample=None,
                grpo=False, sde_solver=False,
            )

        # 每步 clamp：首帧 latent 永远等于 masked_video_lat
        z = apply_mask_clamp(z.to(torch.float32), masked_video_lat.to(torch.float32),
                             latent_mask.to(torch.float32))
        z = z.to(torch.bfloat16)

        if i < sde_steps:
            all_latents.append(z)
            all_log_probs.append(log_prob)

    pred_x0 = pred_original
    if len(all_log_probs) > 0:
        all_latents = torch.stack(all_latents, dim=1)      # [1, K+1, ...]
        all_log_probs = torch.stack(all_log_probs, dim=1)  # [1, K]
    else:
        # 纯 ODE 推理路径：返回空占位，infer_only.py 不会消费这两个张量
        all_latents = torch.stack([z], dim=1)              # [1, 1, ...]
        all_log_probs = z.new_zeros((z.shape[0], 0))
    return z, pred_x0, all_latents, all_log_probs


# ══════════════════════════════════════════════════════════════════════════════
# grpo_one_step（训练时重计算 log_prob，含梯度）
# ══════════════════════════════════════════════════════════════════════════════

def grpo_one_step(
    args,
    latents: torch.Tensor,
    pre_latents: torch.Tensor,
    transformer,
    timestep_val: float,
    step_index: int,
    sigma_schedule: torch.Tensor,
    prompt_embeds: list,
    neg_embeds: list,
    seq_len: int,
    control_latents: torch.Tensor,
    control_camera_latents: torch.Tensor,
    latent_mask: torch.Tensor,
    transformer_forward_fn,
) -> torch.Tensor:
    """训练模式单步重计算 log_prob（PPO ratio 分子）。"""
    transformer.train()
    pred = transformer_forward_fn(
        transformer, latents, timestep_val,
        prompt_embeds, neg_embeds, seq_len,
        control_latents, control_camera_latents, latent_mask,
        args.cfg_train, torch.bfloat16,
    )
    _, _, log_prob = flux_step(
        pred, latents.to(torch.float32), args.eta,
        sigma_schedule, step_index,
        prev_sample=pre_latents.to(torch.float32),
        grpo=True, sde_solver=True,
    )
    return log_prob
