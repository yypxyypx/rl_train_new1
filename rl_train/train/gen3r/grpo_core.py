"""grpo_core.py — GRPO 核心算法。

严格复用 RL/gen3r/train_grpo_gen3r.py 的逻辑，只修改 import 路径。
包含：
    sd3_time_shift     — sigma schedule 偏移
    flux_step          — ODE→SDE 单步 + log_prob（逐元素，支持任意 latent shape）
    run_sample_step    — 完整去噪 Rollout（支持 hybrid SDE/ODE 模式）
    grpo_one_step      — 训练模式重计算单步 log_prob（PPO ratio 分子）
"""

import math

import torch
from tqdm.auto import tqdm


# ══════════════════════════════════════════════════════════════════════════════
# Sigma Schedule
# ══════════════════════════════════════════════════════════════════════════════

def sd3_time_shift(shift: float, t: torch.Tensor) -> torch.Tensor:
    """非线性 sigma schedule 偏移（来自 DanceGRPO）。"""
    return (shift * t) / (1 + (shift - 1) * t)


# ══════════════════════════════════════════════════════════════════════════════
# flux_step: ODE → SDE 单步
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
    """ODE→SDE 单步转换，计算 log_prob（来自 DanceGRPO，无修改）。

    适用于任意 latent shape（逐元素操作），直接支持 Gen3R 的 [B, 16, f, h, w*2]。

    Args:
        model_output    : Transformer 预测的速度场 v
        latents         : 当前时刻 z_t
        eta             : SDE 噪声强度（> 0）
        sigmas          : sigma schedule [T+1]
        index           : 当前时间步索引
        prev_sample     : rollout 时已采样的 z_{t+1}（grpo=True 且 prev_sample=None 时新采样）
        grpo            : 是否计算 log_prob
        sde_solver      : 是否启用 SDE 修正项

    Returns:
        grpo=True  : (prev_sample, pred_original_sample, log_prob)
        grpo=False : (prev_sample_mean, pred_original_sample)
    """
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
# run_sample_step: 完整去噪 Rollout
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
    plucker_embeds: torch.Tensor,
    clip_context: torch.Tensor,
    transformer_forward_fn,
):
    """完整去噪轨迹（Rollout），支持 hybrid SDE/ODE 模式。

    当 args.train_timestep_strategy == "front" 时：
        前 sde_steps 步使用 SDE（eta>0），记录 latent + log_prob 用于训练；
        后续步骤切换为 ODE（eta=0），只为生成最终视频，不记录。
    当 args.train_timestep_strategy == "random" 时（默认）：
        所有步骤使用 SDE，和原始 DanceGRPO 一致。

    Args:
        z               : 初始噪声 [1, 16, f, h, w*2]
        sigma_schedule  : [T+1]
        transformer_forward_fn : gen3r_encode.transformer_forward 的函数引用

    Returns:
        final_z        : 去噪后的 z_T
        pred_x0        : 最后一步预测的 x0（用于解码）
        all_latents    : [1, K+1, 16, f, h, w*2]  (K = sde_steps or T)
        all_log_probs  : [1, K]
    """
    T = args.sampling_steps
    strategy = getattr(args, "train_timestep_strategy", "random")
    sde_fraction = getattr(args, "sde_fraction", 1.0)
    sde_steps = max(1, int(T * sde_fraction)) if strategy == "front" else T

    all_latents = [z]
    all_log_probs = []

    for i in tqdm(range(T), desc="Sampling", leave=False):
        sigma = sigma_schedule[i]
        timestep_val = int(sigma * 1000)
        timesteps = torch.full([1], timestep_val, device=z.device, dtype=torch.long)

        transformer.eval()
        with torch.no_grad():
            pred = transformer_forward_fn(
                transformer, z, timesteps,
                prompt_embeds, neg_embeds, seq_len,
                control_latents, plucker_embeds, clip_context,
                args.cfg_rollout, torch.bfloat16,
            )

        if i < sde_steps:
            z, pred_original, log_prob = flux_step(
                pred, z.to(torch.float32), args.eta,
                sigma_schedule, i, prev_sample=None,
                grpo=True, sde_solver=True,
            )
            z = z.to(torch.bfloat16)
            all_latents.append(z)
            all_log_probs.append(log_prob)
        else:
            z, pred_original = flux_step(
                pred, z.to(torch.float32), 0.0,
                sigma_schedule, i, prev_sample=None,
                grpo=False, sde_solver=False,
            )
            z = z.to(torch.bfloat16)

    pred_x0 = pred_original
    all_latents = torch.stack(all_latents, dim=1)      # [1, K+1, ...]
    all_log_probs = torch.stack(all_log_probs, dim=1)  # [1, K]
    return z, pred_x0, all_latents, all_log_probs


# ══════════════════════════════════════════════════════════════════════════════
# grpo_one_step: 训练模式重计算 log_prob
# ══════════════════════════════════════════════════════════════════════════════

def grpo_one_step(
    args,
    latents: torch.Tensor,
    pre_latents: torch.Tensor,
    transformer,
    timesteps: torch.Tensor,
    step_index: int,
    sigma_schedule: torch.Tensor,
    prompt_embeds: list,
    neg_embeds: list,
    seq_len: int,
    control_latents: torch.Tensor,
    plucker_embeds: torch.Tensor,
    clip_context: torch.Tensor,
    transformer_forward_fn,
) -> torch.Tensor:
    """在训练模式下重计算单步 log_prob（用于 PPO ratio 计算）。

    Args:
        latents     : z_t   [1, 16, f, h, w*2]
        pre_latents : z_{t+1}（rollout 时记录的 action）[1, 16, f, h, w*2]
        step_index  : 在 sigma_schedule 中的绝对索引

    Returns:
        log_prob : [1]
    """
    transformer.train()

    pred = transformer_forward_fn(
        transformer, latents, timesteps,
        prompt_embeds, neg_embeds, seq_len,
        control_latents, plucker_embeds, clip_context,
        args.cfg_train, torch.bfloat16,
    )

    _, _, log_prob = flux_step(
        pred, latents.to(torch.float32), args.eta,
        sigma_schedule, step_index,
        prev_sample=pre_latents.to(torch.float32),
        grpo=True, sde_solver=True,
    )
    return log_prob
