"""train_grpo_v2.py — Gen3R GRPO 强化学习训练主入口（重构版）。

架构：
  config.py        → 参数定义与解析
  model_loader_v2  → 加载 VAE + CLIP + Transformer + Ref（无 T5/geo_adapter/VGGT）
  grpo_engine      → Phase 1 Rollout / Phase 2 Advantage / Phase 3 Update + KL
  reward_dispatcher → 分布式 Reward 调度（4 组 GPU 并行）
  dataset_rl       → 数据集适配器

运行示例（4×4090 单机测试）：
    torchrun --standalone --nproc_per_node=4 train_grpo_v2.py \\
        --pretrained_model_path /path/to/gen3r_ckpts \\
        --config_path /path/to/gen3r.yaml \\
        --t5_embed_dir /path/to/t5_cache \\
        --data_root /path/to/data \\
        --datasets re10k \\
        --output_dir ./outputs/debug \\
        --max_train_steps 20 \\
        --num_generations 2 \\
        --sampling_steps 20 \\
        --dry_run \\
        --gradient_checkpointing

运行示例（16×5090 正式训练）：
    torchrun --nnodes=2 --nproc_per_node=8 train_grpo_v2.py \\
        --pretrained_model_path /path/to/gen3r_ckpts \\
        --config_path /path/to/gen3r.yaml \\
        --t5_embed_dir /path/to/t5_cache \\
        --data_root /path/to/data \\
        --datasets re10k,dl3dv \\
        --output_dir ./outputs/grpo_5090_run1 \\
        --max_train_steps 200 \\
        --num_generations 8 \\
        --sampling_steps 50 \\
        --rewards all \\
        --kl_coeff 0.01 \\
        --checkpointing_steps 50 \\
        --gradient_checkpointing
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

_HERE = Path(__file__).resolve().parent
_GEN3R_PKG = _HERE / "Gen3R"
if str(_GEN3R_PKG) not in sys.path:
    sys.path.insert(0, str(_GEN3R_PKG))

from config import parse_args
from dataset_rl import RLDataset, collate_fn
from grpo_engine import (
    compute_advantages,
    grpo_update,
    rollout_and_decode,
)
from model_loader import (
    create_lr_scheduler,
    create_optimizer,
    load_models,
    save_checkpoint,
    setup_trainable_params,
)
from reward_dispatcher import RewardDispatcher


def setup_dist():
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        # 把 NCCL collective 超时设为 4 小时（默认 10 分钟）。
        # 原因：centralized reward 模式下，rank0 单独协调 4 张卡跑 32 rollout
        # 的 reward（DA3 / DINO / Qwen+SAM3 / VideoAlign），可能耗时 30+ 分钟，
        # 而其它 rank 在 dist.broadcast 等待 reward 期间会被 NCCL watchdog 误杀。
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(hours=4),
        )
    return rank, local_rank, world_size


def set_seed(seed: int, rank: int = 0):
    import random
    import numpy as np
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    random.seed(seed + rank)
    np.random.seed(seed + rank)


def setup_wandb(args, rank: int):
    if rank != 0:
        return None
    if os.environ.get("WANDB_MODE", "").lower() in ("disabled", "off"):
        print("[Train] WANDB_MODE=disabled, skipping wandb")
        return None
    try:
        import wandb
        run = wandb.init(
            project="gen3r-grpo",
            name=Path(args.output_dir).name,
            config=vars(args),
            resume="allow",
        )
        return run
    except Exception as e:
        print(f"[Train] wandb init failed ({type(e).__name__}: {e}), skipping logging")
        return None


def log_metrics(wandb_run, metrics: dict, step: int, rank: int):
    if rank != 0:
        return
    print(f"[Train] Step {step:4d}  " +
          "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in metrics.items()))
    if wandb_run is not None:
        wandb_run.log(metrics, step=step)


def build_work_dirs(args, step: int, rank: int, sample_ids: list, n_gen: int) -> list[list[str]]:
    """为每个 (sample, rollout) 对生成 work_dir。

    返回 [n_samples * n_gen] 的列表（展开后）。
    """
    work_dirs = []
    for sid in sample_ids:
        for gi in range(n_gen):
            wd = os.path.join(
                args.eval_output_dir,
                f"step_{step}", f"rank{rank}", str(sid), "reward", f"gen_{gi}"
            )
            work_dirs.append(wd)
    return work_dirs


def main():
    args = parse_args()

    rank, local_rank, world_size = setup_dist()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if getattr(args, "seed", None) is not None:
        set_seed(args.seed, rank)

    weight_dtype = torch.bfloat16

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.eval_output_dir, exist_ok=True)

    if rank == 0:
        config_snapshot = os.path.join(args.output_dir, "config.json")
        with open(config_snapshot, "w") as f:
            json.dump(vars(args), f, indent=2, default=str)
        print(f"[Train] Config saved to {config_snapshot}")

    # ── 配置加载 ──────────────────────────────────────────────────────────────
    config = OmegaConf.load(args.config_path)

    # ── 模型加载 ──────────────────────────────────────────────────────────────
    if rank == 0:
        print("[Train] Loading models...")
    models = load_models(args, config, device, weight_dtype)
    transformer = models["transformer"]

    # 显存优化：kl_coeff=0 时 ref_transformer 用不到，直接卸到 CPU 释放 ~2.6GB
    if getattr(args, "kl_coeff", 0.0) == 0.0 and models.get("ref_transformer") is not None:
        models["ref_transformer"].cpu()
        torch.cuda.empty_cache()
        if rank == 0:
            print("[Train] kl_coeff=0, ref_transformer offloaded to CPU (~2.6GB freed)")

    # Gradient checkpointing
    if getattr(args, "gradient_checkpointing", False):
        transformer.enable_gradient_checkpointing()
        if rank == 0:
            print("[Train] Gradient checkpointing enabled")

    # 可训练参数
    setup_trainable_params(transformer, getattr(args, "trainable_modules", None))

    # DDP wrap（如果多 GPU）
    if world_size > 1:
        transformer = DDP(transformer, device_ids=[local_rank])

    # Optimizer & scheduler
    raw_transformer = transformer.module if isinstance(transformer, DDP) else transformer
    optimizer = create_optimizer(
        raw_transformer, args.learning_rate, args.weight_decay,
        use_8bit=getattr(args, "use_8bit_adam", False),
    )
    lr_scheduler = create_lr_scheduler(optimizer, args.lr_scheduler, args.lr_warmup_steps)

    # ── 数据集 ────────────────────────────────────────────────────────────────
    if rank == 0:
        print("[Train] Building dataset...")
    dataset_list = [d.strip() for d in args.datasets.split(",") if d.strip()]
    dataset = RLDataset(
        data_root=args.data_root,
        datasets=dataset_list,
        num_frames=args.num_frames,
        stride=args.frame_stride,
        resolution=args.resolution,
        frame_mode=args.frame_mode,
    )

    # ── 可选：固定 seed 抽固定条数 ──────────────────────────────────────────────
    # 通过 --num_samples_subset N 控制（None = 用全量数据）。本机 4 卡 dl3dv smoke
    # test 用 N=4，每条 8 rollout = 32 条 generated videos。
    n_subset = getattr(args, "num_samples_subset", None)
    if n_subset is not None and n_subset > 0 and n_subset < len(dataset.samples):
        import random as _r
        _r.seed(getattr(args, "sampler_seed", 42))
        chosen = _r.sample(dataset.samples, n_subset)
        dataset.samples = chosen
        for ds_name in list(dataset.per_dataset.keys()):
            dataset.per_dataset[ds_name] = [
                s for s in chosen if s["dataset_name"] == ds_name
            ]
        if rank == 0:
            print(f"[Train] Subsampled dataset to {n_subset} samples "
                  f"(seed={getattr(args, 'sampler_seed', 42)})")
            for ds_name, ss in dataset.per_dataset.items():
                print(f"[Train]   {ds_name}: {len(ss)} samples")

    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank,
        shuffle=True, seed=getattr(args, "sampler_seed", 42),
    ) if world_size > 1 else None

    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # ── Reward 调度器 ─────────────────────────────────────────────────────────
    dispatcher = RewardDispatcher(args)

    # ── WandB ─────────────────────────────────────────────────────────────────
    wandb_run = setup_wandb(args, rank)

    # ── 日志文件 ──────────────────────────────────────────────────────────────
    reward_log_path = os.path.join(args.output_dir, "reward_log.jsonl")
    train_log_path = os.path.join(args.output_dir, "training_log.jsonl")

    # ── 训练循环 ──────────────────────────────────────────────────────────────
    global_step = 0
    data_iter = iter(dataloader)
    if rank == 0:
        print(f"[Train] Starting GRPO training. max_steps={args.max_train_steps}")
        print(f"[Train] CFG config: rollout={args.cfg_rollout}, train={args.cfg_train}")
        if abs(args.cfg_rollout - args.cfg_train) > 1e-6:
            print(
                f"[Train] WARNING: cfg_rollout({args.cfg_rollout}) != cfg_train({args.cfg_train}). "
                "Importance ratio between π_old (rollout policy) and π_new (train policy) "
                "will be biased — clip_range/adv_clip_max may need re-tuning. "
                "This trade-off is intentional for 4090 24GB to keep rollout quality "
                "while halving train activation memory."
            )

    while global_step < args.max_train_steps:
        step_start = time.time()

        # ─── 拿下一批数据 ────────────────────────────────────────────────────
        try:
            batch = next(data_iter)
        except StopIteration:
            if sampler is not None:
                sampler.set_epoch(global_step)
            data_iter = iter(dataloader)
            batch = next(data_iter)

        sample_ids = batch["sample_id"]
        gt_camera_paths = batch["camera_txt_path"]

        # ─── Phase 1: Rollout + Decode ───────────────────────────────────────
        if rank == 0:
            print(f"\n[Train] Step {global_step} — Phase 1: Rollout")

        all_video_paths, all_latents, all_log_probs, sigma_schedule, encoded_conds = (
            rollout_and_decode(args, models, batch, global_step, rank, device)
        )

        # ─── Phase 2: Distributed Reward ─────────────────────────────────────
        if rank == 0:
            print(f"[Train] Step {global_step} — Phase 2: Reward")

        ng = args.num_generations
        work_dirs_flat = []
        for sid in sample_ids:
            for gi in range(ng):
                wd = os.path.join(
                    args.eval_output_dir,
                    f"step_{global_step}", f"rank{rank}", str(sid), "reward", f"gen_{gi}"
                )
                work_dirs_flat.append(wd)

        prompts = [
            batch["text"][bi // ng] for bi in range(len(all_video_paths))
        ]
        gt_camera_flat = [
            gt_camera_paths[bi // ng] for bi in range(len(all_video_paths))
        ]

        dispatch_mode = getattr(args, "reward_dispatch_mode", "per_rank")

        if dispatch_mode == "centralized" and world_size > 1:
            # ── 同步 offload 所有训练 / inference 模型，腾出每张卡的全部显存给 reward
            dist.barrier()
            for k in ("transformer", "ref_transformer", "wan_vae", "clip_image_encoder"):
                m = models.get(k)
                if m is not None:
                    m.cpu()
            # DDP wrapper 只是个 module 引用，move 内部 raw_transformer 即可
            torch.cuda.empty_cache()

            # ── all_gather 元数据到全部 rank ──────────────────────────────────
            n_local = len(all_video_paths)
            gathered_videos = [None] * world_size
            gathered_works = [None] * world_size
            gathered_prompts = [None] * world_size
            gathered_gtcam = [None] * world_size
            dist.all_gather_object(gathered_videos, list(all_video_paths))
            dist.all_gather_object(gathered_works, list(work_dirs_flat))
            dist.all_gather_object(gathered_prompts, list(prompts))
            dist.all_gather_object(gathered_gtcam, list(gt_camera_flat))

            # ── rank0 跑 centralized reward ──────────────────────────────────
            if rank == 0:
                full_videos = [v for sub in gathered_videos for v in sub]
                full_works = [w for sub in gathered_works for w in sub]
                full_prompts = [p for sub in gathered_prompts for p in sub]
                full_gtcam = [g for sub in gathered_gtcam for g in sub]
                full_rewards = dispatcher.run_centralized(
                    video_paths=full_videos,
                    gt_camera_paths=full_gtcam,
                    work_dirs=full_works,
                    prompts=full_prompts,
                )
                full_rewards = list(full_rewards)
            else:
                full_rewards = [0.0] * (world_size * n_local)

            full_rewards_t = torch.tensor(full_rewards, dtype=torch.float32, device=device)
            dist.broadcast(full_rewards_t, src=0)

            rewards_t = full_rewards_t[rank * n_local:(rank + 1) * n_local].clone()
            reward_scalars = rewards_t.detach().cpu().tolist()

            # ── reload transformer (Phase 3 用)；KL=0 时 ref/vae/clip 不必上 GPU
            dist.barrier()
            raw_transformer.to(device, dtype=weight_dtype)
            if getattr(args, "kl_coeff", 0.0) != 0.0:
                ref = models.get("ref_transformer")
                if ref is not None:
                    ref.to(device=device, dtype=weight_dtype)
            torch.cuda.empty_cache()
        else:
            # 旧的 per-rank 调度路径（兼容 single-GPU 或老配置）
            reward_scalars = dispatcher.run(
                video_paths=all_video_paths,
                gt_camera_paths=gt_camera_flat,
                work_dirs=work_dirs_flat,
                prompts=prompts,
            )
            rewards_t = torch.tensor(reward_scalars, dtype=torch.float32, device=device)

        # 全局 gather rewards（多卡统计 / 日志）
        if world_size > 1:
            all_rewards = [torch.zeros_like(rewards_t) for _ in range(world_size)]
            dist.all_gather(all_rewards, rewards_t)
            all_rewards_cat = torch.cat(all_rewards)
        else:
            all_rewards_cat = rewards_t

        advantages = compute_advantages(rewards_t, ng)

        reward_mean = rewards_t.mean().item()
        reward_std = rewards_t.std().item()
        reward_min = rewards_t.min().item()
        reward_max = rewards_t.max().item()

        # 日志：reward
        if rank == 0:
            reward_entry = {
                "step": global_step,
                "reward_mean": reward_mean,
                "reward_std": reward_std,
                "reward_min": reward_min,
                "reward_max": reward_max,
                "rewards": reward_scalars,
            }
            with open(reward_log_path, "a") as f:
                f.write(json.dumps(reward_entry, default=str) + "\n")

        # ─── Phase 3: GRPO Update + KL ───────────────────────────────────────
        if rank == 0:
            print(f"[Train] Step {global_step} — Phase 3: GRPO Update")

        models_update = {**models, "transformer": raw_transformer}
        total_loss, kl_mean, grad_norm = grpo_update(
            args, models_update, all_latents, all_log_probs,
            advantages, sigma_schedule, encoded_conds,
            optimizer, lr_scheduler, device,
        )

        global_step += 1
        step_time = time.time() - step_start

        metrics = {
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "reward_min": reward_min,
            "reward_max": reward_max,
            "grpo_loss": total_loss,
            "kl_mean": kl_mean,
            "grad_norm": grad_norm,
            "lr": lr_scheduler.get_last_lr()[0],
            "step_time": step_time,
        }
        log_metrics(wandb_run, metrics, global_step, rank)

        # 日志：训练
        if rank == 0:
            with open(train_log_path, "a") as f:
                f.write(json.dumps({**metrics, "step": global_step}, default=str) + "\n")

        # ─── Checkpoint ──────────────────────────────────────────────────────
        if global_step % args.checkpointing_steps == 0:
            save_checkpoint(raw_transformer, args.output_dir, global_step, rank)

        # ─── 清理 ────────────────────────────────────────────────────────────
        del all_latents, all_log_probs, encoded_conds
        torch.cuda.empty_cache()

        if world_size > 1:
            dist.barrier()

    # ── 最终 checkpoint ──────────────────────────────────────────────────────
    save_checkpoint(raw_transformer, args.output_dir, "final", rank)
    if rank == 0:
        print(f"\n[Train] Training complete. Steps={global_step}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
