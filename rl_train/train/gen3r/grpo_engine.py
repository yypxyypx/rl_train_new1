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
    decode_rgb_videos_batch,
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
    n_per_rank: Optional[int] = None,
) -> tuple[list[str], list, list, torch.Tensor, list]:
    """对 batch 内每个样本生成 num_generations 条 rollout，解码并保存视频。

    新策略（worker 架构）：
      1. T5 embedding 走预计算缓存
      2. CLIP / VAE / Transformer 全程常驻 GPU，不再做 .cpu() offload
         （原因：Phase 2 reward 模型在子进程独占 GPU，没必要把主进程模型搬来搬去）
      3. Rollout 走 ng 一次性 batched forward
      4. VAE decode 改批量：一次 decode N 条 latent，少 N-1 次 CUDA 调用 + Python 开销

    Args:
        args    : argparse.Namespace（需要 vae_decode_micro_batch）
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
    import time as _time

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
    sample_dirs = batch.get("sample_dir", [""] * len(sample_ids))
    camera_txt_paths = batch["camera_txt_path"]

    # ng = "本张 GPU 在这一步要生成多少条 rollout"。
    #   - 旧路径（单卡 / 每卡完整 group）：n_per_rank=None → 用 args.num_generations
    #   - 新路径（sub-group 跨卡 group）：n_per_rank = args.rollouts_per_rank (R)
    ng = int(n_per_rank) if n_per_rank is not None else int(args.num_generations)
    all_video_paths: list[str] = []
    all_latents: list = []
    all_log_probs: list = []
    encoded_conds: list = []

    if args.init_same_noise:
        shared_noise = torch.randn(
            (1, IN_CHANNELS, latent_t, latent_h, latent_w * 2),
            device=device, dtype=WEIGHT_DTYPE,
        )

    vae_mb = max(1, int(getattr(args, "vae_decode_micro_batch", 4)))

    for bi in range(len(texts)):
        sample_id = sample_ids[bi]
        dataset_name = dataset_names[bi]
        sample_dir = sample_dirs[bi] if bi < len(sample_dirs) else ""
        pixel_values = pixel_values_list[bi].to(device)
        c2ws = c2ws_list[bi].to(device)
        Ks = Ks_list[bi].to(device)
        F_frames = pixel_values.shape[0]
        camera_txt_path = camera_txt_paths[bi]

        # ── 加载预计算的 T5 embedding ────────────────────────────────────────
        # 优先 in-place ({sample_dir}/prompt_embed.pt + neg_embed.pt)，
        # 否则 fallback 到 args.t5_embed_dir 老路径
        t_enc0 = _time.time()
        prompt_embed_cpu, neg_embed_cpu = load_t5_embeds(
            args, sample_id, dataset_name, sample_dir=sample_dir,
        )
        prompt_embeds = [prompt_embed_cpu.to(device=device, dtype=WEIGHT_DTYPE)]
        neg_embeds = [neg_embed_cpu.to(device=device, dtype=WEIGHT_DTYPE)]

        # ── 控制图像编码（CLIP + VAE encode）— 模型常驻 GPU，无需 .to() 来回 ─
        control_index = [0]
        control_images = torch.zeros_like(pixel_values).unsqueeze(0)
        control_images[0, control_index] = pixel_values[control_index]

        with torch.no_grad():
            control_latents, clip_context = encode_control_latents(
                control_images, wan_vae, None, clip_image_encoder,
                control_index, F_frames, device, WEIGHT_DTYPE,
            )

        # ── Plücker ray embedding ────────────────────────────────────────────
        with torch.no_grad():
            plucker_embeds = build_plucker_embeds(
                c2ws, Ks, h=args.resolution, w=args.resolution,
                num_frames=F_frames, device=device, dtype=WEIGHT_DTYPE,
            )
        enc_dt = _time.time() - t_enc0
        print(f"[Rollout] [{sample_id}] encode (T5/CLIP/VAE/Plücker) {enc_dt:.1f}s", flush=True)

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

        # ── 批量生成 ng 条 rollout（单次 forward 同时推理所有样本）────────────
        if args.init_same_noise:
            z0_batch = shared_noise.expand(ng, -1, -1, -1, -1).clone()
        else:
            z0_batch = torch.randn(
                (ng, IN_CHANNELS, latent_t, latent_h, latent_w * 2),
                device=device, dtype=WEIGHT_DTYPE,
            )

        t_samp0 = _time.time()
        with torch.no_grad():
            # run_sample_step 支持 z.shape[0]=ng 批量推理
            final_z_batch, pred_x0_batch, batch_latents, batch_log_probs = run_sample_step(
                args, z0_batch, sigma_schedule, transformer,
                prompt_embeds, neg_embeds, seq_len,
                control_latents, plucker_embeds, clip_context,
                transformer_forward,
            )
        samp_dt = _time.time() - t_samp0
        print(f"[Rollout] [{sample_id}] sampling done "
              f"({args.sampling_steps} steps × ng={ng}, {samp_dt:.1f}s, "
              f"{samp_dt/max(args.sampling_steps,1):.2f}s/step)", flush=True)
        # batch_latents : [ng, K+1, 16, f, h, w*2]
        # batch_log_probs: [ng, K]
        # final_z_batch : [ng, 16, f, h, w*2]

        # 拆分回 per-rollout 格式，与 Phase 3 grpo_update 兼容
        for gi in range(ng):
            all_latents.append(batch_latents[gi:gi+1])     # [1, K+1, ...]
            all_log_probs.append(batch_log_probs[gi:gi+1]) # [1, K]
            encoded_conds.append({**cond_base, "gi": gi})

        del z0_batch
        torch.cuda.empty_cache()

        # ── 批量 VAE decode（一次 decode_rgb_videos_batch 替代 ng 次） ───────
        # 用 pred_x0_batch（模型在最后一步预测的干净 x0），而不是 final_z_batch
        # （SDE 终态，最后一步会注入 eta·sqrt(sigma_{T-1}) 量级的噪声，
        #  解码出来肉眼可见雪花/糊脸；与 infer_only.py 行为对齐）。
        video_paths = [os.path.join(eval_dir, f"gen_{gi}.mp4") for gi in range(ng)]
        t_dec0 = _time.time()
        with torch.no_grad():
            decode_rgb_videos_batch(
                pred_x0_batch.to(device=device, dtype=WEIGHT_DTYPE),
                wan_vae,
                video_paths,
                fps=16,
                micro_batch=vae_mb,
            )
        all_video_paths.extend(video_paths)
        dec_dt = _time.time() - t_dec0
        print(f"[Rollout] [{sample_id}] VAE decode ng={ng} (mb={vae_mb}) {dec_dt:.1f}s", flush=True)

        del final_z_batch, pred_x0_batch
        torch.cuda.empty_cache()

    return all_video_paths, all_latents, all_log_probs, sigma_schedule, encoded_conds


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: Advantage 计算
# ══════════════════════════════════════════════════════════════════════════════

def compute_advantages(rewards: torch.Tensor, num_generations: int) -> torch.Tensor:
    """GRPO 组内 advantage（z-score 形式：(r - mean) / (std + eps)）。

    每 num_generations 条 rollout 视为一组，组内做 z-score。这就是 GRPO 的
    critic-free advantage：组均值当 baseline、组标准差做 normalizer。
    归一化范围严格限定在同一 prompt 的若干 rollout 之内，跨 prompt 的
    reward 量纲差异因此被去掉。

    Args:
        rewards : [N] float32
        num_generations : 每组大小（同一 prompt 的 rollout 数）

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


def compute_grpo_advantages_subgroup(
    local_rewards: torch.Tensor,
    sub_group_pg,
    ranks_per_group: int,
    rollouts_per_rank: int,
    local_rank_in_group: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """跨 sub-group 计算 GRPO advantage：同一 prompt 的 G=R*ranks_per_group 条
    rollout 分布在 ranks_per_group 张卡上，需要在 sub-group 内 all_gather 后
    在完整的 G 条上算 z-score，再切回本卡那 R 条。

    Args:
        local_rewards   : [n_local] = [train_batch_size * R] 本卡的 reward
        sub_group_pg    : torch.distributed ProcessGroup（仅含本 sub-group 的 ranks）
        ranks_per_group : 一个 prompt 跨多少张卡（G/R）
        rollouts_per_rank : R
        local_rank_in_group : 本 rank 在 sub-group 内的位置 [0, ranks_per_group)

    Returns:
        my_advantages   : [n_local]   本卡负责的 advantage 切片
        full_rewards    : [n_local * ranks_per_group]  整个 sub-group 的 reward
                          （rank0 顺序 → rank1 顺序 → ...），返回供日志使用。

    数学语义：
        每 (n_local // R) 个样本各自占据 G 条 rollout，在 G 条内独立做 z-score。
        ranks_per_group=1 时退化为单卡 compute_advantages（不做通信）。
    """
    n_local = int(local_rewards.shape[0])
    R = int(rollouts_per_rank)
    G = R * int(ranks_per_group)
    if ranks_per_group <= 1:
        # 退化路径：单卡完整 group，无需通信
        adv_full = compute_advantages(local_rewards, G)
        return adv_full, local_rewards

    # ── sub-group all_gather ─────────────────────────────────────────────────
    gathered = [torch.zeros_like(local_rewards) for _ in range(int(ranks_per_group))]
    dist.all_gather(gathered, local_rewards, group=sub_group_pg)
    full_rewards = torch.cat(gathered, dim=0)  # [ranks_per_group * n_local]

    # ── 重排成 "样本 × G" 顺序 ────────────────────────────────────────────────
    # gathered 是按 rank 顺序拼的：
    #   rank0: [s0_r0..s0_rR-1, s1_r0..s1_rR-1, ...]
    #   rank1: [s0_rR..s0_r2R-1, s1_rR..s1_r2R-1, ...]
    # 按 sample 维 group 起来：
    #   sample 0 完整 G 条 = [rank0[0:R], rank1[0:R], ...]
    n_samples = n_local // R
    grouped = []
    for s in range(n_samples):
        for k in range(int(ranks_per_group)):
            grouped.append(gathered[k][s * R : (s + 1) * R])
    # grouped 长度 = n_samples * ranks_per_group，每段 R；拼起来后形状 [n_samples * G]
    rewards_per_sample = torch.cat(grouped, dim=0)

    adv_full = compute_advantages(rewards_per_sample, G)  # [n_samples * G]

    # ── 取本卡 4 条（每个样本切自己的 R 条） ────────────────────────────────
    my_adv_chunks = []
    for s in range(n_samples):
        offset = s * G + int(local_rank_in_group) * R
        my_adv_chunks.append(adv_full[offset : offset + R])
    my_advantages = torch.cat(my_adv_chunks, dim=0)
    return my_advantages, rewards_per_sample


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3: GRPO Policy Update + KL 散度
# ══════════════════════════════════════════════════════════════════════════════

def _stack_cond_for_microbatch(conds: list[dict], idxs: list[int]) -> dict:
    """从 N 条 rollout 的 cond 列表中挑出 idxs 对应的若干条，沿 batch 维拼起来。

    rollout 同一 sample → 同一 cond_base，所以一个 sample 内 cond 是共享的；
    但跨 sample 时 control_latents/plucker/clip 都不同，必须按 batch 维 cat。

    返回 dict 字段：
      prompt_embeds : list[Tensor[L,C]]  长度 = len(idxs)（transformer_forward 自己处理 ng 份）
      neg_embeds    : 同上
      control_latents : Tensor[mb, 20, f, h, w*2]
      plucker_embeds  : Tensor[mb, 24, f, h, w*2]
      clip_context    : Tensor[mb, 257, 1280]
      seq_len         : int (所有 cond 必须相同)
    """
    if len(idxs) == 1:
        return conds[idxs[0]]

    seq_lens = {conds[i]["seq_len"] for i in idxs}
    assert len(seq_lens) == 1, f"microbatch 内 seq_len 不一致: {seq_lens}"

    def _flat_emb(emb_list):
        # emb_list 是 [Tensor]，可能是 [L,C] 或 [1,L,C]；统一拍成 [L,C]
        e = emb_list[0]
        if e.dim() == 3:
            e = e.squeeze(0)
        return e

    prompt_embeds = [_flat_emb(conds[i]["prompt_embeds"]) for i in idxs]
    neg_embeds = [_flat_emb(conds[i]["neg_embeds"]) for i in idxs]

    def _cat_b(key):
        ts = []
        for i in idxs:
            t = conds[i][key]
            if t.dim() < 5:  # plucker/control/clip 都是 [1,...]
                t = t.unsqueeze(0) if t.shape[0] != 1 else t
            ts.append(t)
        return torch.cat(ts, dim=0)

    return {
        "prompt_embeds":  prompt_embeds,
        "neg_embeds":     neg_embeds,
        "control_latents": _cat_b("control_latents"),
        "plucker_embeds":  _cat_b("plucker_embeds"),
        "clip_context":    _cat_b("clip_context"),
        "seq_len":         conds[idxs[0]]["seq_len"],
    }


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
    ddp_wrapper=None,
) -> tuple[float, float, float]:
    """GRPO policy update，返回 (total_loss, kl_mean, grad_norm)。

    Loss_t = -A_i * min(ratio_t, clip(ratio_t, 1-ε, 1+ε)) + β * KL_t
    KL_t ≈ new_log_prob - ref_log_prob  （high-order estimate, KL=0 时跳过 ref）

    支持 microbatch（args.train_microbatch_size，默认 1）：
      - mb=1: 逐条 rollout 逐 timestep fwd+bwd（N×train_T 次）
      - mb>1: 同 timestep 把 mb 条 rollout 拼到 batch 维一次 fwd+bwd
              （N//mb × train_T 次）

    单 step 一次更新策略：
      gradient_accumulation_steps 默认会被主流程置为 n_groups（即 N//mb），
      使一个 train step 内累加完所有 (n_groups × train_T) 次梯度后只调用一次
      optimizer.step()。Loss 在 backward 前除以 (gradient_accumulation_steps × train_T)
      → 等价于对所有 micro 样本与时间步做平均。

    DDP 通信优化：
      若传入 ddp_wrapper（DistributedDataParallel 实例），除最后一次 backward 外
      全部包在 ddp_wrapper.no_sync() 上下文里，跳过中间 all-reduce。这样 16 卡
      场景下每个 train step 只做 1 次跨卡梯度同步，不是 (N×train_T) 次。

    Args:
        all_latents   : list of Tensor [1, K+1, ...]，每条 rollout 的轨迹
        all_log_probs : list of Tensor [1, K]，rollout 时记录的 log_prob
        advantages    : [N] float32
        ddp_wrapper   : 可选 DDP 包装器；用于 no_sync() 上下文
    """
    import time as _time
    from contextlib import nullcontext
    transformer = models["transformer"]
    ref_transformer = models["ref_transformer"]

    strategy = getattr(args, "train_timestep_strategy", "random")
    N = len(all_latents)
    K = all_log_probs[0].shape[1]  # SDE 步数
    mb = max(1, int(getattr(args, "train_microbatch_size", 1)))
    if N % mb != 0:
        # 不整除时退回 mb=1，保证 advantage 对齐
        print(f"[GRPO] WARN: num_generations={N} 不被 microbatch={mb} 整除，回退 mb=1")
        mb = 1

    timestep_values = torch.tensor(
        [[int(sigma_schedule[t] * 1000) for t in range(K)] for _ in range(N)],
        device=device, dtype=torch.long,
    )

    if strategy == "front":
        # 默认 train_T = K（用全部 SDE 步），但允许 --train_steps_count 显式截断
        # 用法：rollout 全 SDE (sde_fraction=1.0, K=sampling_steps)，但只对前 N 步算梯度
        train_steps_count = int(getattr(args, "train_steps_count", 0))
        train_T = K if train_steps_count <= 0 else min(K, train_steps_count)
        order = torch.arange(train_T, device=device).unsqueeze(0).expand(N, -1)
    else:
        train_T = max(1, int(K * args.timestep_fraction))
        order = torch.stack([torch.randperm(K)[:train_T] for _ in range(N)]).to(device)

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

    n_groups = N // mb  # microbatch 组数
    total_iters = n_groups * train_T
    iter_idx = 0
    t_phase0 = _time.time()
    last_print = t_phase0

    use_no_sync = (ddp_wrapper is not None) and dist.is_initialized()
    print(f"[GRPO] start: N={N}  mb={mb}  train_T={train_T}  "
          f"total_iters={total_iters}  kl_coeff={args.kl_coeff}  "
          f"GAS={args.gradient_accumulation_steps}  ddp_no_sync={use_no_sync}",
          flush=True)

    for g in range(n_groups):
        idxs = list(range(g * mb, (g + 1) * mb))
        cond = _stack_cond_for_microbatch(encoded_conds, idxs)
        adv_g = torch.clamp(
            advantages[idxs[0] : idxs[-1] + 1],          # [mb]
            -args.adv_clip_max, args.adv_clip_max,
        )

        for t in range(train_T):
            # 是否「最后一次 backward」：(g, t) == (n_groups-1, train_T-1)。
            # 只在最后一次同步 all-reduce 跨卡梯度，其它 micro-iter 用 no_sync()
            # 跳过 DDP hook，避免每个 micro 都做一次跨 16 卡 all-reduce。
            is_last_micro = (g == n_groups - 1) and (t == train_T - 1)
            sync_ctx = (
                ddp_wrapper.no_sync() if (use_no_sync and not is_last_micro)
                else nullcontext()
            )
            with sync_ctx:
                # ── 取出该 microbatch 在当前 timestep 的 latents/timesteps ─────
                abs_step = int(order[idxs[0]][t].item())
                z_t = latents_sel[idxs[0] : idxs[-1] + 1, t].to(device)
                z_next = next_latents_sel[idxs[0] : idxs[-1] + 1, t].to(device)
                ts = timesteps_sel[idxs[0] : idxs[-1] + 1, t]
                old_log_prob = log_probs_sel[idxs[0] : idxs[-1] + 1, t].to(device)

                # ── 新策略 log_prob（有梯度） ─────────────────────────────────
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
                )  # [mb]

                # ── 参考策略 log_prob（无梯度，KL 计算）─ kl_coeff=0 时整段跳过 ─
                kl_loss = torch.tensor(0.0, device=device)
                if args.kl_coeff > 0 and ref_transformer is not None:
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
                    kl_loss = (new_log_prob - ref_log_prob.detach()).mean()

                # ── GRPO Clipped Loss ─────────────────────────────────────────
                ratio = torch.exp(new_log_prob - old_log_prob.detach())
                unclipped = -adv_g * ratio
                clipped = -adv_g * torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range)
                grpo_loss = torch.mean(torch.maximum(unclipped, clipped))

                loss = (grpo_loss + args.kl_coeff * kl_loss) / (
                    args.gradient_accumulation_steps * train_T
                )
                loss.backward()

            avg_loss = loss.detach().clone()
            avg_kl = kl_loss.detach().clone()
            if dist.is_initialized():
                dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
                dist.all_reduce(avg_kl, op=dist.ReduceOp.AVG)

            total_loss += avg_loss.item() * (args.gradient_accumulation_steps * train_T)
            total_kl += avg_kl.item()
            n_steps_total += 1

            iter_idx += 1
            now = _time.time()
            if now - last_print >= 30.0 or iter_idx == total_iters:
                elapsed = now - t_phase0
                eta = elapsed / iter_idx * (total_iters - iter_idx)
                print(f"[GRPO] {iter_idx}/{total_iters}  "
                      f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s  "
                      f"last_loss={avg_loss.item():.4f}  last_kl={avg_kl.item():.4f}",
                      flush=True)
                last_print = now

            # 显存压力测试用：跑指定 iter 数后立即退出（不做 optimizer.step）
            _max_iters = int(getattr(args, "p3_max_iters", 0))
            if _max_iters > 0 and iter_idx >= _max_iters:
                _peak = torch.cuda.max_memory_allocated(device)
                print(f"[GRPO] EARLY EXIT after {iter_idx} iters "
                      f"(--p3_max_iters={_max_iters})  "
                      f"peak_alloc={_peak/1e9:.2f}GB", flush=True)
                return total_loss / max(n_steps_total, 1), total_kl / max(n_steps_total, 1), 0.0

        # 一个 microbatch group 走完累计梯度，按 gradient_accumulation_steps 触发 step
        # 主流程通常会把 GAS 设为 n_groups → 每个 train step 只 step 一次，
        # 由 sub-group/DDP 在最后一次 backward 完成跨卡梯度同步。
        if (g + 1) % args.gradient_accumulation_steps == 0 or (g + 1) == n_groups:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in transformer.parameters() if p.requires_grad],
                max_norm=args.max_grad_norm,
            )
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)

    # update 结束时统一 barrier 一次（替代原先 per-group barrier，配合 no_sync 优化）
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
