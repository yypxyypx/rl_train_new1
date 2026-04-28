"""train_grpo.py — Wan2.2-Fun-5B-Control-Camera GRPO 训练主入口。

完整流程（与 gen3r 同构）：
  1. DDP 初始化（torchrun 启动）
  2. 加载 Wan2.2 模型 + 构建 dataset / dataloader / optimizer / lr_scheduler
  3. 主循环：每个 step
     - Phase 1: rollout_and_decode（8 条 rollout）
     - rank0 用 RewardDispatcher.run_centralized() 计算 reward → 广播到全 rank
     - Phase 2: compute_advantages
     - Phase 3: grpo_update（每轮 rollout **只触发 1 次** optimizer.step()）
  4. 周期保存 checkpoint
"""

from __future__ import annotations

import copy
import os
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config import parse_args  # noqa: E402
from dataset_rl import build_rl_dataset, collate_fn  # noqa: E402
from grpo_engine import (compute_advantages, grpo_update,  # noqa: E402
                         rollout_and_decode)
from model_loader import load_all_models  # noqa: E402
from reward_bridge import log_rewards  # noqa: E402
from reward_dispatcher import RewardDispatcher  # noqa: E402


WEIGHT_DTYPE = torch.bfloat16


# ══════════════════════════════════════════════════════════════════════════════
# DDP setup
# ══════════════════════════════════════════════════════════════════════════════

def _setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return rank, world_size, local_rank, device


def _is_main(rank: int) -> bool:
    return rank == 0


# ══════════════════════════════════════════════════════════════════════════════
# Optimizer / LR
# ══════════════════════════════════════════════════════════════════════════════

def _build_optimizer(args, transformer):
    if args.trainable_modules:
        for name, p in transformer.named_parameters():
            p.requires_grad_(any(m in name for m in args.trainable_modules))
    else:
        transformer.requires_grad_(True)

    trainable = [p for p in transformer.parameters() if p.requires_grad]
    print(f"[train] trainable params: {sum(p.numel() for p in trainable)/1e6:.2f} M")

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(trainable, lr=args.learning_rate,
                                      weight_decay=args.weight_decay)
            print("[train] Using bitsandbytes.AdamW8bit")
            return opt
        except ImportError:
            print("[train] bitsandbytes not available, falling back to AdamW")

    return torch.optim.AdamW(trainable, lr=args.learning_rate,
                             weight_decay=args.weight_decay)


def _build_lr_scheduler(args, optimizer):
    from diffusers.optimization import get_scheduler
    return get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Reward 调度（rank0 算完广播）
# ══════════════════════════════════════════════════════════════════════════════

def _compute_rewards_distributed(
    args,
    dispatcher: RewardDispatcher,
    video_paths: list[str],
    encoded_conds: list,
    step: int,
    rank: int,
    world_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, list]:
    """rank0 调用 RewardDispatcher 算 reward；广播到全 rank。

    Returns:
        rewards    : Tensor [N] on device
        reward_results : list[dict]（仅 rank0 有效，其余 rank 为 []）
    """
    N_local = len(video_paths)
    work_dirs = []
    gt_camera_paths = []
    prompts = []
    for cond, vp in zip(encoded_conds, video_paths):
        work_dirs.append(os.path.join(
            args.output_dir, "reward_outputs",
            f"step_{step}", f"rank{rank}",
            cond["sample_id"], f"gen_{cond['gi']}",
        ))
        gt_camera_paths.append(cond["camera_txt_path"])
        prompts.append("camera moving through a scene")  # 与 gen3r 默认一致

    # rank0 计算 reward；其他 rank 等待
    if _is_main(rank):
        scalars = dispatcher.run_centralized(
            video_paths=video_paths,
            gt_camera_paths=gt_camera_paths,
            work_dirs=work_dirs,
            prompts=prompts,
        )
        rewards = torch.tensor(scalars, device=device, dtype=torch.float32)
    else:
        rewards = torch.zeros(N_local, device=device, dtype=torch.float32)

    if dist.is_initialized() and world_size > 1:
        # 广播 rank0 的 rewards 到全 rank
        dist.broadcast(rewards, src=0)

    # 占位空 reward_results（只 rank0 有详细字典；这里简化为标量列表 + 包成 dict）
    reward_results = []
    if _is_main(rank):
        for s, vp, cond in zip(rewards.cpu().tolist(), video_paths, encoded_conds):
            reward_results.append({
                "reward_total": s,
                "video_path": vp,
                "sample_id": cond["sample_id"],
                "gen_id": cond["gi"],
            })

    return rewards, reward_results


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint
# ══════════════════════════════════════════════════════════════════════════════

def _save_checkpoint(args, transformer, step: int, rank: int):
    """与 gen3r 同名输出格式：<output_dir>/checkpoint-<step>/transformer/
    + transformer.safetensors。"""
    if not _is_main(rank):
        return
    ckpt_dir = Path(args.output_dir) / f"checkpoint-{step}" / "transformer"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 解 DDP wrapper
    model = transformer.module if isinstance(transformer, DDP) else transformer
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    try:
        from safetensors.torch import save_file
        save_file(state_dict, str(ckpt_dir / "transformer.safetensors"))
    except ImportError:
        torch.save(state_dict, str(ckpt_dir / "transformer.bin"))

    print(f"[train] Saved checkpoint to {ckpt_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    rank, world_size, local_rank, device = _setup_ddp()

    if args.seed is not None:
        torch.manual_seed(args.seed + rank)

    if _is_main(rank):
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        print(f"[train] Output dir: {args.output_dir}")
        print(f"[train] World size: {world_size}, rank: {rank}, device: {device}")

    # ── 加载模型 ─────────────────────────────────────────────────────────────
    config = OmegaConf.load(args.config_path)
    models = load_all_models(args, config, device, WEIGHT_DTYPE)
    transformer = models["transformer"]

    # 梯度检查点
    if getattr(args, "gradient_checkpointing", False):
        if hasattr(transformer, "enable_gradient_checkpointing"):
            transformer.enable_gradient_checkpointing()
        elif hasattr(transformer, "gradient_checkpointing_enable"):
            transformer.gradient_checkpointing_enable()
        print("[train] gradient_checkpointing enabled")

    # 参考策略（KL 正则化用）
    ref_transformer = None
    if args.kl_coeff > 0:
        ref_transformer = copy.deepcopy(transformer).eval()
        ref_transformer.requires_grad_(False)
        ref_transformer.to(device=device, dtype=WEIGHT_DTYPE)
        print("[train] Built reference transformer for KL regularization")
    models["ref_transformer"] = ref_transformer

    transformer.to(device=device, dtype=WEIGHT_DTYPE)

    # DDP wrap
    if dist.is_initialized() and world_size > 1:
        transformer = DDP(
            transformer,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            broadcast_buffers=False,
        )
    models["transformer"] = transformer

    # ── 数据 / Optimizer ─────────────────────────────────────────────────────
    dataset = build_rl_dataset(args)
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=True,
    )

    # 用未包 DDP 的 transformer 取 trainable params
    raw_transformer = transformer.module if isinstance(transformer, DDP) else transformer
    optimizer = _build_optimizer(args, raw_transformer)
    lr_scheduler = _build_lr_scheduler(args, optimizer)

    # ── Reward dispatcher（仅 rank0 跑） ─────────────────────────────────────
    dispatcher = RewardDispatcher(args)

    # ── 主循环 ───────────────────────────────────────────────────────────────
    global_step = 0
    reward_log_path = os.path.join(args.output_dir, "reward_log.jsonl")

    for epoch in range(10**9):
        for batch in dataloader:
            if global_step >= args.max_train_steps:
                break

            t_start = time.time()

            # Phase 1
            video_paths, all_latents, all_log_probs, sigma_schedule, encoded_conds = \
                rollout_and_decode(args, models, batch, global_step, rank, device)
            t_rollout = time.time() - t_start

            # Reward
            t_r0 = time.time()
            rewards, reward_results = _compute_rewards_distributed(
                args, dispatcher, video_paths, encoded_conds,
                global_step, rank, world_size, device,
            )
            t_reward = time.time() - t_r0

            if _is_main(rank):
                log_rewards(reward_results, global_step, reward_log_path, rank=0)

            # Phase 2
            advantages = compute_advantages(rewards, args.num_generations)

            if _is_main(rank):
                print(f"[train] step {global_step}: rewards mean={rewards.mean().item():.4f} "
                      f"std={rewards.std().item():.4f}; "
                      f"adv mean={advantages.mean().item():.4f} std={advantages.std().item():.4f}")

            # Phase 3
            t_u0 = time.time()
            total_loss, kl_mean, grad_norm = grpo_update(
                args, models, all_latents, all_log_probs, advantages,
                sigma_schedule, encoded_conds, optimizer, lr_scheduler, device,
            )
            t_update = time.time() - t_u0

            global_step += 1
            if _is_main(rank):
                print(f"[train] step {global_step}/{args.max_train_steps}: "
                      f"loss={total_loss:.4f} kl={kl_mean:.4f} grad={grad_norm:.4f} "
                      f"| rollout={t_rollout:.1f}s reward={t_reward:.1f}s update={t_update:.1f}s")

            # Checkpoint
            if global_step % args.checkpointing_steps == 0:
                _save_checkpoint(args, transformer, global_step, rank)

            # 释放本 step 临时显存
            del all_latents, all_log_probs, encoded_conds
            torch.cuda.empty_cache()

        if global_step >= args.max_train_steps:
            break

    # 末次 checkpoint
    _save_checkpoint(args, transformer, global_step, rank)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
