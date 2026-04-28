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
from datetime import datetime, timedelta


def _dt_now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")
from pathlib import Path
from typing import Optional

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
    compute_grpo_advantages_subgroup,
    grpo_update,
    rollout_and_decode,
)
from model_loader import (
    create_lr_scheduler,
    create_optimizer,
    find_latest_resume_checkpoint,
    join_ref_transformer,
    load_models,
    load_training_state,
    load_transformer_weights,
    register_existing_rolling_ckpt,
    save_checkpoint,
    save_permanent_checkpoint,
    save_rolling_checkpoint,
    setup_trainable_params,
)
from reward_dispatcher import RewardDispatcher
from reward_workers import RewardWorkerPool, make_shm_path
from reward_aggregator import aggregate_one_rollout


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


def build_sub_groups(world_size: int, ranks_per_group: int, rank: int):
    """构建 sub-group NCCL 通信组：每 ranks_per_group 张卡组成一个 sub-group，
    共享同一个 prompt 的 G=ranks_per_group×R 条 rollout。

    Args:
        world_size      : 总 GPU 数
        ranks_per_group : G/R，一个 prompt 跨多少张卡
        rank            : 本进程的 global rank

    Returns:
        my_sub_group_pg     : 仅含本 sub-group ranks 的 ProcessGroup（None when single-rank or RPG=1）
        group_id            : 本 rank 所属 sub-group 编号 [0, n_groups_total)
        local_rank_in_group : 本 rank 在 sub-group 内的位置 [0, ranks_per_group)
        n_groups_total      : sub-group 总数 = world_size // ranks_per_group

    要求 world_size % ranks_per_group == 0。当 ranks_per_group <= 1 时退化：
    返回 (None, rank, 0, world_size)，即每张卡自己一个 sub-group。
    """
    if ranks_per_group <= 1 or world_size <= 1:
        return None, rank, 0, world_size

    if world_size % ranks_per_group != 0:
        raise ValueError(
            f"world_size ({world_size}) 必须能被 ranks_per_group ({ranks_per_group}) 整除"
        )
    n_groups_total = world_size // ranks_per_group

    # 必须在所有 rank 上同步创建所有 sub-group（dist.new_group 是 collective）
    my_sub_group_pg = None
    for g in range(n_groups_total):
        ranks_in_g = list(range(g * ranks_per_group, (g + 1) * ranks_per_group))
        pg = dist.new_group(ranks=ranks_in_g)
        if rank in ranks_in_g:
            my_sub_group_pg = pg
    group_id = rank // ranks_per_group
    local_rank_in_group = rank % ranks_per_group
    return my_sub_group_pg, group_id, local_rank_in_group, n_groups_total


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
        project = os.environ.get("WANDB_PROJECT") or "gen3r-grpo"
        run_name = (os.environ.get("WANDB_RUN_NAME")
                    or os.environ.get("WANDB_NAME")
                    or Path(args.output_dir).name)
        group = os.environ.get("WANDB_RUN_GROUP")
        notes = os.environ.get("WANDB_NOTES")
        tags_env = os.environ.get("WANDB_TAGS", "")
        tags = [t.strip() for t in tags_env.split(",") if t.strip()] or None
        run = wandb.init(
            project=project,
            name=run_name,
            group=group,
            notes=notes,
            tags=tags,
            config=vars(args),
            resume="allow",
        )
        # 让 wandb 把 step_time 等时间序列以折线展示
        try:
            wandb.define_metric("*", step_metric="train/global_step")
        except Exception:
            pass
        print(f"[Train] wandb init OK  project={project}  name={run_name}")
        return run
    except Exception as e:
        print(f"[Train] wandb init failed ({type(e).__name__}: {e}), skipping logging")
        return None


def log_metrics(wandb_run, metrics: dict, step: int, rank: int):
    if rank != 0:
        return
    say(f"Step {step:4d}  " +
        "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                  for k, v in metrics.items()))
    if wandb_run is not None:
        wandb_run.log(metrics, step=step)


# ════════════════════════════════════════════════════════════════════════════
# 主进程统一日志：[HH:MM:SS Train step=N phase=P t=Xs] msg
# ════════════════════════════════════════════════════════════════════════════

class _Stage:
    """记录当前 step / phase 上下文，给所有 say() 加前缀和阶段计时。"""
    step: int = -1
    phase: str = "init"
    phase_t0: float = 0.0
    rank: int = 0

    @classmethod
    def enter(cls, step: int, phase: str) -> None:
        cls.step = step
        cls.phase = phase
        cls.phase_t0 = time.time()
        say(f"=== Phase {phase} START ===")

    @classmethod
    def leave(cls) -> None:
        dt = time.time() - cls.phase_t0
        say(f"=== Phase {cls.phase} DONE in {dt:.1f}s ===")


def say(msg: str) -> None:
    """主进程打印（rank0 only），带时间戳/step/phase 前缀，立即 flush。

    避免 stdout 行缓冲被 tee/管道缓冲导致日志看似 hang 住。
    """
    if _Stage.rank != 0:
        return
    ts = time.strftime("%H:%M:%S")
    prefix = f"{ts} [Train"
    if _Stage.step >= 0:
        prefix += f" step={_Stage.step}"
    if _Stage.phase:
        prefix += f" {_Stage.phase}"
        if _Stage.phase_t0 > 0:
            prefix += f" +{time.time() - _Stage.phase_t0:.0f}s"
    prefix += "]"
    sys.stdout.write(f"{prefix} {msg}\n")
    sys.stdout.flush()


# 新 worker 架构下 reward 名 → worker group 的映射
# (geo_semantic / qwen_sam3 已弃用，不在 worker 池中)
_REWARD_TO_WORKERS = {
    "geo_global":    {"da3"},
    "feature_sim":   {"da3", "dinov2"},
    "camera_traj":   {"da3"},
    "video_quality": {"videoalign"},
}
_ALL_WORKER_REWARDS = list(_REWARD_TO_WORKERS.keys())


def _resolve_active_workers(rewards_str: str) -> tuple[list[str], set[str]]:
    """从 --rewards 参数推导：(rewards_to_compute, active_worker_groups)。

    geo_semantic（需 SAM3）若被请求会打 warning 并跳过——worker 架构暂不支持。
    """
    if rewards_str == "all":
        rewards = list(_ALL_WORKER_REWARDS)
    else:
        rewards = [r.strip() for r in rewards_str.split(",") if r.strip()]
    keep = []
    skipped = []
    for r in rewards:
        if r in _REWARD_TO_WORKERS:
            keep.append(r)
        else:
            skipped.append(r)
    if skipped:
        print(f"[Train] WARNING: workers 架构不支持 {skipped}，已跳过")
    active = set()
    for r in keep:
        active |= _REWARD_TO_WORKERS[r]
    return keep, active


def _parse_reward_weights(weights_str):
    if not weights_str:
        return None
    w = {}
    for item in weights_str.split(","):
        if ":" in item:
            k, v = item.split(":", 1)
            w[k.strip()] = float(v.strip())
    return w or None


def _extract_frames_to_dir(video_path: str, frames_dir: str) -> int:
    """主进程 cv2 抽帧（CPU 操作，DA3/DINOv2 worker 共享）。返回帧数。"""
    import cv2
    os.makedirs(frames_dir, exist_ok=True)
    existing = sorted(p for p in os.listdir(frames_dir) if p.startswith("frame_"))
    if existing:
        return len(existing)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {video_path}")
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(frames_dir, f"frame_{idx:05d}.png"), frame)
        idx += 1
    cap.release()
    return idx


def _get_blocks(transformer):
    """从 (DDP-wrapped or raw) wan transformer 取出 .blocks ModuleList。"""
    raw = transformer.module if hasattr(transformer, "module") else transformer
    return raw.blocks


def selective_enable_ckpt(transformer, n_layers: int) -> int:
    """选择性 per-block 梯度检查点。

    n_layers > 0  → 开前 n_layers 个 block 的 ckpt；其余正常 forward
    n_layers == -1 → 全部 block 都 ckpt
    n_layers == 0 → 不开

    实现：把 model.gradient_checkpointing 设为一个 list[bool]，长度 == len(blocks)。
    wan_transformer3d.py 的 forward 循环已支持 list 类型的 mask。
    返回实际开 ckpt 的 block 数。
    """
    raw = transformer.module if hasattr(transformer, "module") else transformer
    blocks = raw.blocks
    total = len(blocks)
    if n_layers == -1 or n_layers >= total:
        target = total
    else:
        target = max(0, min(int(n_layers), total))

    if target == 0:
        raw.gradient_checkpointing = False
        return 0

    mask = [i < target for i in range(total)]
    raw.gradient_checkpointing = mask
    return target


def synthetic_rollout(args, models, batch, device):
    """P3-only smoke：完全 mock，不依赖 dataset / T5 cache / 真实样本。

    全部 tensor shape 按 dl3dv 49 帧 560 分辨率的真实 log 数据 hardcode。
    N=8, K=50 时 GPU 占用 ~12 GB（synthetic encoded conds + latents）。
    返回与 rollout_and_decode 完全相同的 5-tuple。
    """
    import math as _math
    from grpo_engine import IN_CHANNELS, SPATIAL_DS, TEMPORAL_DS, WEIGHT_DTYPE
    from grpo_core import sd3_time_shift

    transformer = models["transformer"]
    raw_t = transformer.module if hasattr(transformer, "module") else transformer

    sigma_schedule = sd3_time_shift(
        args.shift, torch.linspace(1, 0, args.sampling_steps + 1),
    )
    K = args.sampling_steps
    ng = args.num_generations
    F_frames = args.num_frames                                # 49
    latent_t = ((F_frames - 1) // TEMPORAL_DS) + 1            # 13
    latent_h = args.resolution // SPATIAL_DS                  # 70
    latent_w = args.resolution // SPATIAL_DS                  # 70 (实际 width = latent_w*2 = 140)

    patch_h = raw_t.config.patch_size[1] if hasattr(raw_t.config, "patch_size") else 2
    patch_w = raw_t.config.patch_size[2] if hasattr(raw_t.config, "patch_size") else 2
    seq_len = _math.ceil((latent_w * 2 * latent_h) / (patch_h * patch_w) * latent_t)

    # ── 真实 shape（来自 0a1b7c20a92c43c6 真实 encode log）──
    # prompt:  (6, 4096) bf16
    # neg:     (6, 4096) bf16  （取 prompt 同 shape，真实 neg 长度可能不同但训练等价）
    # control: (1, 20, 13, 70, 140) bf16  → 16 latent channels + 4 mask/ref channels
    # plucker: (1, 24, 13, 560, 1120) bf16
    # clip:    (1, 257, 1280) bf16
    prompt_embed = torch.randn((6, 4096), device=device, dtype=WEIGHT_DTYPE)
    neg_embed = torch.randn((6, 4096), device=device, dtype=WEIGHT_DTYPE)
    control_latents = torch.randn(
        (1, 20, latent_t, latent_h, latent_w * 2), device=device, dtype=WEIGHT_DTYPE,
    )
    plucker_embeds = torch.randn(
        (1, 24, latent_t, args.resolution, args.resolution * 2),
        device=device, dtype=WEIGHT_DTYPE,
    )
    clip_context = torch.randn((1, 257, 1280), device=device, dtype=WEIGHT_DTYPE)

    say(f"[synthetic] mock conds  prompt={tuple(prompt_embed.shape)}  "
        f"control={tuple(control_latents.shape)}  "
        f"plucker={tuple(plucker_embeds.shape)}  clip={tuple(clip_context.shape)}  "
        f"seq_len={seq_len}")

    cond_base = dict(
        prompt_embeds=[prompt_embed],
        neg_embeds=[neg_embed],
        control_latents=control_latents,
        plucker_embeds=plucker_embeds,
        clip_context=clip_context,
        seq_len=seq_len,
        sample_id="synthetic",
        dataset_name="synthetic",
        camera_txt_path="<mock>",
        F=F_frames,
    )

    say(f"[synthetic] mocking ng={ng} rollouts × K={K} steps "
        f"(latent shape=[1, {K+1}, {IN_CHANNELS}, {latent_t}, {latent_h}, {latent_w*2}])")
    all_latents = []
    all_log_probs = []
    encoded_conds = []
    for gi in range(ng):
        z = torch.randn(
            (1, K + 1, IN_CHANNELS, latent_t, latent_h, latent_w * 2),
            device=device, dtype=WEIGHT_DTYPE,
        )
        lp = torch.randn((1, K), device=device, dtype=torch.float32) * 0.01
        all_latents.append(z)
        all_log_probs.append(lp)
        encoded_conds.append({**cond_base, "gi": gi})

    return [], all_latents, all_log_probs, sigma_schedule, encoded_conds


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
    _Stage.rank = rank

    if getattr(args, "seed", None) is not None:
        set_seed(args.seed, rank)

    # ── Sub-group 通信组（GRPO 跨卡 group 模式）──────────────────────────────
    #   group_size G  = num_generations
    #   per-rank R    = rollouts_per_rank
    #   ranks/group   = G / R  （每个 prompt 跨多少张卡）
    R = int(args.rollouts_per_rank)
    G = int(args.num_generations)
    ranks_per_group = max(1, G // max(R, 1))
    sub_group_pg, group_id, local_rank_in_group, n_groups_total = build_sub_groups(
        world_size, ranks_per_group, rank
    )
    if rank == 0:
        print(f"[Train] GRPO sub-group config:  G={G}  R={R}  "
              f"ranks_per_group={ranks_per_group}  n_groups_total={n_groups_total}  "
              f"world_size={world_size}",
              flush=True)
        print(f"[Train] effective rollouts/step = {R * world_size}  "
              f"= {n_groups_total} groups × {G} rollouts (= {R} rollouts × {world_size} GPUs)",
              flush=True)

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

    # ── Reward worker pool（异步启动，与 Gen3R 加载并行）─────────────────────
    # 用户需求：先把 Gen3R 主模型加载起来，剩下的 reward 模型在后台并行加载，
    # 第一次需要 reward 时再 join 等待。这样 worker 子进程的 ~3min 模型加载
    # 不会拖慢前 1~2 个 step（Gen3R 自身加载也要 ~1-2min，相互覆盖）。
    dispatch_mode = getattr(args, "reward_dispatch_mode", "per_rank")
    dispatcher = None
    worker_pool: Optional[RewardWorkerPool] = None
    workers_rewards: list[str] = []
    workers_weights = None

    if getattr(args, "p3_synthetic", False):
        if rank == 0:
            print("[Train] p3_synthetic=True → skip worker pool / dispatcher entirely")
    elif dispatch_mode == "workers":
        workers_rewards, active_workers = _resolve_active_workers(
            getattr(args, "rewards", "all")
        )
        workers_weights = _parse_reward_weights(getattr(args, "reward_weights", None))
        if not active_workers:
            raise RuntimeError("dispatch_mode=workers 但没有可用的 reward (active_workers 为空)")
        worker_log_dir = os.path.join(args.output_dir, "worker_logs")
        if rank == 0:
            print(f"[Train] dispatch=workers  active_workers={sorted(active_workers)}  "
                  f"rewards={workers_rewards}  weights={workers_weights}")
        worker_pool = RewardWorkerPool(
            active_groups=active_workers,
            gpu_id=local_rank,
            log_dir=worker_log_dir,
            ready_timeout=900.0,
        )
        worker_pool.start_async()  # 立即返回，不阻塞主进程加载 Gen3R
        # 注册清理：进程退出 / SIGTERM 时关闭子进程，避免僵尸
        import atexit as _atexit
        _atexit.register(worker_pool.shutdown)
    else:
        dispatcher = RewardDispatcher(args)

    # ── 模型加载 ──────────────────────────────────────────────────────────────
    if rank == 0:
        print("[Train] Loading models (workers loading in background) ...")
    models = load_models(args, config, device, weight_dtype)
    transformer = models["transformer"]
    # ref_transformer 已由 load_models 按策略处理：
    #   kl_coeff=0 → None（跳过加载，省 ~6GB IO）
    #   kl_coeff>0 → 后台线程异步加载到 CPU，Phase 3 前 join

    # Gradient checkpointing —— 支持选择性 per-block 开
    #   --gradient_checkpoint_layers N（>0） → 开前 N 个 block 的 ckpt
    #   --gradient_checkpoint_layers -1     → 全开（等价 --gradient_checkpointing）
    #   --gradient_checkpointing 单独给     → 全开
    n_ckpt_layers = int(getattr(args, "gradient_checkpoint_layers", 0))
    if n_ckpt_layers == 0 and getattr(args, "gradient_checkpointing", False):
        n_ckpt_layers = -1  # 兼容旧开关：全开
    if n_ckpt_layers != 0:
        n_applied = selective_enable_ckpt(transformer, n_ckpt_layers)
        if rank == 0:
            total = len(_get_blocks(transformer))
            print(f"[Train] Gradient checkpointing: {n_applied}/{total} blocks under ckpt "
                  f"(req={n_ckpt_layers}, -1=all)")

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
    sampler = None
    dataloader = None
    if getattr(args, "p3_synthetic", False):
        if rank == 0:
            print("[Train] p3_synthetic=True → skip dataset/dataloader (mock all tensors)")
    else:
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

        # ── 可选：固定 seed 抽固定条数 ──────────────────────────────────────────
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

        # ── sub-group 模式下，同 sub-group 的 ranks_per_group 张卡必须看到完全
        #    一致的 sample 序列。把 DistributedSampler 的 (num_replicas, rank)
        #    设为 (n_groups_total, group_id) 即可：每个 sub-group 当作一个虚拟
        #    rank，sub-group 内 ranks 共用同一个 sampler 输出。
        sampler = DistributedSampler(
            dataset, num_replicas=n_groups_total, rank=group_id,
            shuffle=True, seed=getattr(args, "sampler_seed", 42),
        ) if n_groups_total > 1 else None

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

    # ── WandB ─────────────────────────────────────────────────────────────────
    wandb_run = setup_wandb(args, rank)

    # ── 日志文件 ──────────────────────────────────────────────────────────────
    reward_log_path = os.path.join(args.output_dir, "reward_log.jsonl")
    train_log_path = os.path.join(args.output_dir, "training_log.jsonl")

    # ── 训练循环 ──────────────────────────────────────────────────────────────
    global_step = 0

    # ── Resume：可选断点恢复 ─────────────────────────────────────────────────
    #   --resume_from "" / "auto" / explicit path
    #   - 还原 transformer 权重 + global_step + sampler_seed
    #   - 跳过已用过的样本（通过 sampler.set_epoch + 消耗 step_in_epoch 个 batch）
    #   - optimizer / scheduler / wandb run id 暂不持久化
    resume_arg = getattr(args, "resume_from", "") or ""
    resume_dir: Optional[Path] = None
    if resume_arg.lower() == "auto":
        resume_dir = find_latest_resume_checkpoint(args.output_dir)
        if rank == 0:
            if resume_dir is None:
                print(f"[Resume] auto: no checkpoint found in {args.output_dir} → 从头训练")
            else:
                print(f"[Resume] auto: found latest = {resume_dir}")
    elif resume_arg:
        resume_dir = Path(resume_arg)
        if not resume_dir.is_dir():
            raise FileNotFoundError(f"--resume_from {resume_dir} 不存在")

    if resume_dir is not None:
        # 1) 加载权重
        if rank == 0:
            print(f"[Resume] loading transformer weights from {resume_dir}")
        load_transformer_weights(raw_transformer, resume_dir, device, strict=True)
        # 2) 加载 training_state 恢复 global_step
        ts = load_training_state(resume_dir)
        if ts is None:
            if rank == 0:
                print(f"[Resume] WARN: {resume_dir}/training_state.json 不存在，"
                      f"global_step 无法恢复，仅加载权重，从 step=0 重新训练。")
        else:
            global_step = int(ts.get("global_step", 0))
            saved_seed = ts.get("sampler_seed", None)
            saved_groups = ts.get("n_groups_total", None)
            if rank == 0:
                print(f"[Resume] training_state: global_step={global_step}, "
                      f"sampler_seed={saved_seed}, n_groups_total={saved_groups}")
            # 一致性校验：seed / 分组数变了，样本顺序就不一样了，需要警告
            cur_seed = getattr(args, "sampler_seed", 42)
            if saved_seed is not None and int(saved_seed) != int(cur_seed):
                if rank == 0:
                    print(f"[Resume] WARN: sampler_seed 改了 (saved={saved_seed} vs current={cur_seed})，"
                          f"样本顺序与原训练不一致；快进逻辑仍按当前 seed 执行。")
            if saved_groups is not None and int(saved_groups) != int(n_groups_total):
                if rank == 0:
                    print(f"[Resume] WARN: n_groups_total 改了 (saved={saved_groups} vs current={n_groups_total})，"
                          f"per-epoch 样本数变了；快进按当前配置执行，已用样本/未用样本边界会偏移。")
        # 3) 把已存在的 rolling-* 注册进模块状态，下次保存时它会被替换
        if resume_dir.name.startswith("rolling-"):
            register_existing_rolling_ckpt(resume_dir)

    # 构建 data_iter，并按 global_step 快进 dataloader 跳过已用样本
    data_iter = None
    n_steps_per_epoch = 1   # synthetic 模式默认占位；真实模式下面会覆盖
    if dataloader is not None:
        n_steps_per_epoch = max(1, len(sampler) if sampler is not None else len(dataloader))
        epoch = global_step // n_steps_per_epoch
        step_in_epoch = global_step % n_steps_per_epoch
        if sampler is not None:
            sampler.set_epoch(epoch)
        data_iter = iter(dataloader)
        if step_in_epoch > 0:
            if rank == 0:
                print(f"[Resume] fast-forwarding dataloader: skip {step_in_epoch} batches "
                      f"(epoch={epoch}, step_in_epoch={step_in_epoch}, "
                      f"steps_per_epoch={n_steps_per_epoch})")
            for _ in range(step_in_epoch):
                try:
                    next(data_iter)
                except StopIteration:
                    if sampler is not None:
                        epoch += 1
                        sampler.set_epoch(epoch)
                    data_iter = iter(dataloader)
                    next(data_iter)
            if rank == 0:
                print(f"[Resume] fast-forward done. resume from global_step={global_step}")
        if rank == 0 and resume_dir is not None:
            print(f"[Resume] note: optimizer/scheduler 状态未持久化，恢复后前几步 loss 可能短暂抖动。")

    # ── 资料化 training_state（每次 ckpt 一起落盘，用于下次 resume）────────
    def _make_training_state(step: int) -> dict:
        return {
            "global_step": step,
            "sampler_seed": int(getattr(args, "sampler_seed", 42)),
            "n_groups_total": int(n_groups_total),
            "ranks_per_group": int(args.num_generations // max(1, args.rollouts_per_rank)),
            "world_size": int(world_size),
            "n_steps_per_epoch": int(n_steps_per_epoch),
            "datasets": getattr(args, "datasets", ""),
            "train_manifest": getattr(args, "train_manifest", ""),
            "saved_at": _dt_now_iso(),
        }
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
        _Stage.step = global_step
        _Stage.phase = ""

        # ─── 拿下一批数据（synthetic 模式跳过）────────────────────────────────
        if getattr(args, "p3_synthetic", False):
            batch = None
            sample_ids = ["synthetic"]
            gt_camera_paths = ["<mock>"]
        else:
            try:
                batch = next(data_iter)
            except StopIteration:
                if sampler is not None:
                    # epoch 与 resume 公式一致：epoch = global_step // n_steps_per_epoch
                    sampler.set_epoch(global_step // n_steps_per_epoch)
                data_iter = iter(dataloader)
                batch = next(data_iter)
            sample_ids = batch["sample_id"]
            gt_camera_paths = batch["camera_txt_path"]

        # ─── P3 SYNTHETIC SMOKE：跳过 worker / rollout / reward，直奔 P3 ─────
        if getattr(args, "p3_synthetic", False):
            _Stage.enter(global_step, "P1-synthetic")
            all_video_paths, all_latents, all_log_probs, sigma_schedule, encoded_conds = (
                synthetic_rollout(args, models, batch, device)
            )
            free_b, total_b = torch.cuda.mem_get_info(device)
            say(f"after synthetic rollout: GPU free={free_b/1e9:.2f}/{total_b/1e9:.2f} GB")
            _Stage.leave()

            # 合成 advantage（已归一化）
            N = args.num_generations
            advantages = torch.randn(N, device=device, dtype=torch.float32)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
            say(f"[synthetic] mock advantages: mean={advantages.mean().item():+.3f} "
                f"std={advantages.std().item():.3f} min={advantages.min().item():+.3f} "
                f"max={advantages.max().item():+.3f}")

            # 主进程主动 empty_cache 模拟真实 P2.5 的释放
            import gc as _gc
            _gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            free_b, total_b = torch.cuda.mem_get_info(device)
            say(f"before P3: GPU free={free_b/1e9:.2f}/{total_b/1e9:.2f} GB "
                f"(no worker pool to unload)")

            # ─── 直接进 P3 ───────────────────────────────────────────────────
            _Stage.enter(global_step, "P3-update")
            join_ref_transformer(models, device, weight_dtype)
            models_update = {**models, "transformer": raw_transformer}
            # 与生产 P3 路径保持一致：自动把 GAS 设为 n_local_groups，整 step 一次 step
            n_local_rollouts = len(all_latents)
            mb_p3 = max(1, int(getattr(args, "train_microbatch_size", 1)))
            if n_local_rollouts % mb_p3 != 0:
                mb_p3 = 1
            args.gradient_accumulation_steps = n_local_rollouts // mb_p3
            ddp_for_no_sync = transformer if isinstance(transformer, DDP) else None
            try:
                total_loss, kl_mean, grad_norm = grpo_update(
                    args, models_update, all_latents, all_log_probs,
                    advantages, sigma_schedule, encoded_conds,
                    optimizer, lr_scheduler, device,
                    ddp_wrapper=ddp_for_no_sync,
                )
                peak_b = torch.cuda.max_memory_allocated(device)
                free_b, total_b = torch.cuda.mem_get_info(device)
                say(f"[synthetic] P3 OK  loss={total_loss:.4f}  kl={kl_mean:.4f}  "
                    f"grad_norm={grad_norm:.4f}")
                say(f"[synthetic] PEAK alloc={peak_b/1e9:.2f} GB  "
                    f"now free={free_b/1e9:.2f}/{total_b/1e9:.2f} GB")
            except torch.cuda.OutOfMemoryError as oom:
                peak_b = torch.cuda.max_memory_allocated(device)
                say(f"[synthetic] P3 OOM!  peak alloc={peak_b/1e9:.2f} GB")
                say(f"[synthetic] error: {oom}")
                raise
            _Stage.leave()

            global_step += 1
            step_time = time.time() - step_start
            say(f"=== synthetic step {global_step-1} done in {step_time:.1f}s ===")

            # ── synthetic 模式也走一遍 ckpt 保存逻辑（smoke test 用）────────
            rolling_every = int(getattr(args, "rolling_ckpt_every", 0))
            permanent_every = int(getattr(args, "permanent_ckpt_every", 0))
            keep_n = int(getattr(args, "keep_last_n_permanent", 0))
            ts_payload = _make_training_state(global_step)
            if rolling_every > 0 and global_step % rolling_every == 0:
                save_rolling_checkpoint(raw_transformer, args.output_dir, global_step, rank,
                                         training_state=ts_payload)
            if permanent_every > 0 and global_step % permanent_every == 0:
                save_permanent_checkpoint(
                    raw_transformer, args.output_dir, global_step, rank,
                    keep_last_n=keep_n, training_state=ts_payload,
                )
            if rolling_every == 0 and permanent_every == 0:
                if global_step % args.checkpointing_steps == 0:
                    save_checkpoint(raw_transformer, args.output_dir, global_step, rank,
                                     training_state=ts_payload)
            continue

        # ─── Phase 1: Rollout + Decode ───────────────────────────────────────
        # 注意：rollout 用 R（本卡份额），不是 G（完整 group）。
        # sub-group 内的 ranks_per_group 张卡共享同一个 sample，每张卡
        # 只生成该 sample 的一份子集（R 条），合起来才是完整 G 条。
        _Stage.enter(global_step, "P1-rollout")
        all_video_paths, all_latents, all_log_probs, sigma_schedule, encoded_conds = (
            rollout_and_decode(
                args, models, batch, global_step, rank, device,
                n_per_rank=R,
            )
        )
        _Stage.leave()

        # ─── Phase 2: Distributed Reward ─────────────────────────────────────
        _Stage.enter(global_step, "P2-reward")

        # work_dir 用 "global rollout 编号"：sub-group 内同一 sample 的
        # ranks_per_group 张卡分别填 [local_rank_in_group*R, (local_rank_in_group+1)*R)
        # 段，避免不同 rank 的产物路径冲突。
        ng_local = R
        work_dirs_flat = []
        for sid in sample_ids:
            for li in range(ng_local):
                gi_global = local_rank_in_group * R + li
                wd = os.path.join(
                    args.eval_output_dir,
                    f"step_{global_step}",
                    f"group{group_id}", f"rank{rank}",
                    str(sid), "reward", f"gen_{gi_global}",
                )
                work_dirs_flat.append(wd)

        prompts = [
            batch["text"][bi // ng_local] for bi in range(len(all_video_paths))
        ]
        gt_camera_flat = [
            gt_camera_paths[bi // ng_local] for bi in range(len(all_video_paths))
        ]

        if dispatch_mode == "workers":
            # ── Stage 化 reward pipeline（用户方案）─────────────────────────
            #   阶段 0  : 抽帧 (CPU)
            #   阶段 1  : DA3 全部 (worker 内部串行 N 个 sub-job; DA3 跨 scene 不能真 batch)
            #   阶段 2  : DINOv2 全部 (worker forward_batch=8 真 batch)
            #   阶段 3  : VideoAlign 全部 (inferencer.reward(list, list) 真 batch)
            #   阶段 4  : 主 GPU 聚合 (FeatUp + warping + reward 计算)
            #             micro_batch=2: 每两条 rollout 算完立刻 empty_cache，
            #             控制 FeatUp 高分辨率特征图的峰值显存
            import gc as _gc

            assert worker_pool is not None
            # 第一次 reward 之前阻塞等 worker 加载完成（async 启动的）。
            if not worker_pool.is_ready():
                say("waiting for async-loading reward workers to be ready...")
                _t_join = time.time()
                worker_pool.wait_ready_join()
                say(f"reward workers ready after {time.time()-_t_join:.1f}s wait")

            n_local = len(all_video_paths)
            phase2_t0 = time.time()
            active_groups = sorted(worker_pool.active_groups)
            say(f"reward pipeline (staged+batched): {n_local} rollouts | "
                f"workers={active_groups} | order: extract → DA3 → DINO → VA → agg(mb=2) | "
                f"geo={getattr(args, 'geo_compare_mode', 'all_pairs')} "
                f"feat={getattr(args, 'feature_compare_mode', 'first_frame')}")

            # ── Stage 0: 抽帧 + 准备 /dev/shm 输出路径 ─────────────────
            t_s0 = time.time()
            frames_dirs: list[str] = []
            da3_outs: list[Optional[str]] = []
            dino_outs: list[Optional[str]] = []
            va_outs: list[Optional[str]] = []
            n_frames_total = 0
            need_frames = ("da3" in active_groups) or ("dinov2" in active_groups)
            for i, (vp, wd) in enumerate(zip(all_video_paths, work_dirs_flat)):
                Path(wd).mkdir(parents=True, exist_ok=True)
                fd = str(Path(wd) / "frames")
                if need_frames:
                    n_frames_total += _extract_frames_to_dir(str(vp), fd)
                frames_dirs.append(fd)
                shm_key_i = rank * 1000 + i  # rank 间隔，多 rank 不撞文件名
                da3_outs.append(
                    make_shm_path(global_step, shm_key_i, "da3", "npz")
                    if "da3" in active_groups else None
                )
                dino_outs.append(
                    make_shm_path(global_step, shm_key_i, "dinov2", "npz")
                    if "dinov2" in active_groups else None
                )
                va_outs.append(
                    make_shm_path(global_step, shm_key_i, "videoalign", "json")
                    if "videoalign" in active_groups else None
                )
            extract_dt = time.time() - t_s0
            say(f"  stage0 extract: {n_local} rollouts ({n_frames_total} frames) "
                f"in {extract_dt:.1f}s")

            # ── Stage 1: DA3 全部 (worker 内串行 N 个 sub-job) ─────────
            da3_dt = 0.0
            if "da3" in active_groups:
                t_s1 = time.time()
                sub_jobs = [
                    {"frames_dir": fd, "output_path": op}
                    for fd, op in zip(frames_dirs, da3_outs) if op is not None
                ]
                try:
                    worker_pool.run_batch("da3", sub_jobs, timeout=1800.0)
                except Exception as e:
                    say(f"  stage1 DA3 batch FAILED: {e}")
                da3_dt = time.time() - t_s1
                say(f"  stage1 DA3 batch DONE in {da3_dt:.1f}s "
                    f"(avg {da3_dt/max(n_local,1):.1f}s/r, serial)")

            # ── Stage 2: DINO 全部 (worker forward batch=8 真 batch) ───
            dino_dt = 0.0
            if "dinov2" in active_groups:
                t_s2 = time.time()
                sub_jobs = [
                    {"frames_dir": fd, "output_path": op}
                    for fd, op in zip(frames_dirs, dino_outs) if op is not None
                ]
                try:
                    worker_pool.run_batch(
                        "dinov2", sub_jobs,
                        extra_args={"forward_batch": 8},
                        timeout=1800.0,
                    )
                except Exception as e:
                    say(f"  stage2 DINO batch FAILED: {e}")
                dino_dt = time.time() - t_s2
                say(f"  stage2 DINO batch DONE in {dino_dt:.1f}s "
                    f"(avg {dino_dt/max(n_local,1):.1f}s/r, forward_batch=8)")

            # ── Stage 3: VideoAlign 全部 ───────────────────────────────
            #   va_micro_batch == 0  → 一次性真 batch=N（多卡，VideoAlign 独占 GPU）
            #   va_micro_batch  > 0  → 切成大小 == va_micro_batch 的 chunk，
            #                          chunk 之间串行调 worker.run_batch
            #                          （单卡 smoke 必须开，否则 Qwen2-VL attention OOM）
            va_dt = 0.0
            if "videoalign" in active_groups:
                t_s3 = time.time()
                all_sub_jobs = [
                    {"video_path": str(vp), "prompt": pr, "output_path": op}
                    for vp, pr, op in zip(all_video_paths, prompts, va_outs)
                    if op is not None
                ]
                va_mb = int(getattr(args, "va_micro_batch", 0))
                if va_mb <= 0 or va_mb >= len(all_sub_jobs):
                    chunks = [all_sub_jobs]
                else:
                    chunks = [
                        all_sub_jobs[i:i + va_mb]
                        for i in range(0, len(all_sub_jobs), va_mb)
                    ]
                n_done = 0
                n_failed = 0
                for ci, chunk in enumerate(chunks):
                    try:
                        worker_pool.run_batch(
                            "videoalign", chunk, timeout=1800.0,
                        )
                        n_done += len(chunk)
                    except Exception as e:
                        n_failed += len(chunk)
                        say(f"  stage3 VA chunk {ci+1}/{len(chunks)} "
                            f"(size={len(chunk)}) FAILED: {e}")
                va_dt = time.time() - t_s3
                if len(chunks) == 1:
                    mode_desc = f"true batch={len(all_sub_jobs)}"
                else:
                    mode_desc = (f"{len(chunks)} chunks × <={va_mb}, "
                                 f"ok={n_done}/{len(all_sub_jobs)}")
                say(f"  stage3 VA DONE in {va_dt:.1f}s "
                    f"(avg {va_dt/max(n_local,1):.1f}s/r, {mode_desc})")

            # ── Stage 4: 主 GPU 聚合 (FeatUp + warping + reward 计算) ──
            #   micro_batch=2：每 2 条 rollout 算完立刻 empty_cache
            AGG_MICRO = int(getattr(args, "reward_agg_micro_batch", 2))
            t_s4 = time.time()
            reward_scalars: list[float] = []
            n_mb = (n_local + AGG_MICRO - 1) // AGG_MICRO
            for s in range(0, n_local, AGG_MICRO):
                e = min(s + AGG_MICRO, n_local)
                t_mb0 = time.time()
                for j in range(s, e):
                    try:
                        result = aggregate_one_rollout(
                            work_dir=work_dirs_flat[j],
                            video_path=str(all_video_paths[j]),
                            gt_camera_txt=str(gt_camera_flat[j]),
                            da3_npz=da3_outs[j],
                            dinov2_npz=dino_outs[j],
                            videoalign_json=va_outs[j],
                            rewards_to_compute=workers_rewards,
                            weights=workers_weights,
                            device=str(device),
                            conf_threshold=getattr(args, "conf_threshold", 0.0),
                            geo_compare_mode=getattr(args, "geo_compare_mode", "all_pairs"),
                            feature_compare_mode=getattr(args, "feature_compare_mode", "first_frame"),
                        )
                        rt = float(result.get("reward_total", float("nan")))
                    except Exception as ex:
                        say(f"  stage4 agg r{j+1} FAILED: {ex}")
                        rt = float("nan")
                    reward_scalars.append(rt)
                # 一组 micro batch 处理完，立刻清显存（FeatUp 中间值很大）
                torch.cuda.empty_cache()
                _gc.collect()
                mb_dt = time.time() - t_mb0
                rts_str = " ".join(f"{r:+.3f}" for r in reward_scalars[s:e])
                say(f"  stage4 agg [mb {s//AGG_MICRO+1}/{n_mb}] r{s+1}-{e} "
                    f"in {mb_dt:.1f}s  rewards={{{rts_str}}}  → empty_cache")
            agg_dt = time.time() - t_s4

            p2_total = time.time() - phase2_t0
            say(f"P2 summary: extract={extract_dt:.1f}s  DA3={da3_dt:.1f}s  "
                f"DINO={dino_dt:.1f}s  VA={va_dt:.1f}s  agg={agg_dt:.1f}s  "
                f"TOTAL={p2_total:.1f}s  avg/r={p2_total/max(n_local,1):.1f}s")

            rewards_t = torch.tensor(reward_scalars, dtype=torch.float32, device=device)

        elif dispatch_mode == "centralized" and world_size > 1:
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

        # ── 全局 gather rewards（多卡统计 / 日志，跨全部 world）─────────────
        if world_size > 1:
            all_rewards = [torch.zeros_like(rewards_t) for _ in range(world_size)]
            dist.all_gather(all_rewards, rewards_t)
            all_rewards_cat = torch.cat(all_rewards)
        else:
            all_rewards_cat = rewards_t

        # ── GRPO advantage：sub-group all_gather → 同一 prompt 的 G 条 z-score ─
        # ranks_per_group=1 时退化为单卡 compute_advantages（无通信）。
        advantages, group_full_rewards = compute_grpo_advantages_subgroup(
            local_rewards=rewards_t,
            sub_group_pg=sub_group_pg,
            ranks_per_group=ranks_per_group,
            rollouts_per_rank=R,
            local_rank_in_group=local_rank_in_group,
        )

        # 日志统计：用全局 all_gather 后的 rewards（覆盖 world 内全部 rollouts），
        # 以前用本卡 rewards_t 是个 bug，单卡时等价、多卡时只反映 1/world_size 数据
        reward_mean = all_rewards_cat.mean().item()
        reward_std = all_rewards_cat.std().item() if all_rewards_cat.numel() > 1 else 0.0
        reward_min = all_rewards_cat.min().item()
        reward_max = all_rewards_cat.max().item()

        # ── per-component reward 分项（rank0 读 reward.json，单步内可统计）──
        # 训练调参时需要看每个 head 单独的趋势：geo_global / feature_sim / camera_*
        # / video_quality / reward_total_global 等
        reward_components: dict[str, float] = {}
        if rank == 0:
            comp_acc: dict[str, list[float]] = {}
            for wd in work_dirs_flat:
                rj = Path(wd) / "reward.json"
                if not rj.exists():
                    continue
                try:
                    with open(rj) as f:
                        rd = json.load(f)
                except Exception:
                    continue
                for k, v in rd.items():
                    if not k.startswith("reward_"):
                        continue
                    if not isinstance(v, (int, float)):
                        continue
                    if isinstance(v, float) and (v != v):  # NaN
                        continue
                    comp_acc.setdefault(k, []).append(float(v))
            for k, vs in comp_acc.items():
                if not vs:
                    continue
                arr = torch.tensor(vs, dtype=torch.float32)
                reward_components[f"{k}_mean"] = arr.mean().item()
                if arr.numel() > 1:
                    reward_components[f"{k}_std"] = arr.std().item()

        # ── advantage 统计（GRPO 健康度诊断）──
        if advantages.numel() > 0:
            adv_mean = advantages.mean().item()
            adv_std = advantages.std().item() if advantages.numel() > 1 else 0.0
            adv_abs_mean = advantages.abs().mean().item()
        else:
            adv_mean = adv_std = adv_abs_mean = 0.0

        # 日志：reward
        if rank == 0:
            reward_entry = {
                "step": global_step,
                "reward_mean": reward_mean,
                "reward_std": reward_std,
                "reward_min": reward_min,
                "reward_max": reward_max,
                "rewards": reward_scalars,
                **reward_components,
            }
            with open(reward_log_path, "a") as f:
                f.write(json.dumps(reward_entry, default=str) + "\n")

        # ─── Phase 2 收尾日志 ────────────────────────────────────────────────
        _Stage.leave()

        # ─── Phase 2.5: 主进程清一道 cache（worker 模型常驻 GPU，不 offload）─
        # 释放 reward 阶段在主进程产生的中间 tensor（FeatUp upsampler、
        # DINO/DA3 warping buffer 等），让 P3 的 caching allocator 起点更干净。
        if rank == 0:
            import gc as _gc
            _gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            free_b, total_b = torch.cuda.mem_get_info(device)
            say(f"P2.5 main-only empty_cache → GPU free={free_b/1e9:.2f}/{total_b/1e9:.2f} GB "
                f"(workers stay on GPU)")

        # ─── Phase 3: GRPO Update + KL ───────────────────────────────────────
        _Stage.enter(global_step, "P3-update")
        say(f"reward stats  mean={reward_mean:+.4f}  std={reward_std:.4f}  "
            f"min={reward_min:+.4f}  max={reward_max:+.4f}  "
            f"kl_coeff={args.kl_coeff}  mb={getattr(args,'train_microbatch_size',1)}  "
            f"ckpt_layers={getattr(args,'gradient_checkpoint_layers',0)}")

        # 确保 ref_transformer 异步加载完成（kl_coeff=0 时为空操作）
        join_ref_transformer(models, device, weight_dtype)

        # ── 单 step 一次更新策略 ────────────────────────────────────────────
        # 自动把 GAS 设为 n_local_groups（n_local // mb），让 grpo_update 在
        # 整个 train step 累加完所有梯度后只调用一次 optimizer.step()。
        # 跨 16 卡的梯度平均由 DDP 在最后一次 backward 时 all-reduce 完成
        # （sub-group 的 reward all_gather 已在 P2 处理，advantage 已经是组内
        #  z-score；DDP all-reduce 把跨 group 的梯度做平均 → 等价于 64 rollout 平均）。
        n_local_rollouts = len(all_latents)
        mb_p3 = max(1, int(getattr(args, "train_microbatch_size", 1)))
        if n_local_rollouts % mb_p3 != 0:
            mb_p3 = 1
        n_local_groups = n_local_rollouts // mb_p3
        prev_gas = args.gradient_accumulation_steps
        args.gradient_accumulation_steps = n_local_groups
        if rank == 0 and prev_gas != n_local_groups:
            print(f"[Train] auto-set gradient_accumulation_steps "
                  f"{prev_gas} → {n_local_groups} (= n_local_rollouts={n_local_rollouts} / mb={mb_p3})",
                  flush=True)

        models_update = {**models, "transformer": raw_transformer}
        ddp_for_no_sync = transformer if isinstance(transformer, DDP) else None
        total_loss, kl_mean, grad_norm = grpo_update(
            args, models_update, all_latents, all_log_probs,
            advantages, sigma_schedule, encoded_conds,
            optimizer, lr_scheduler, device,
            ddp_wrapper=ddp_for_no_sync,
        )
        peak_b = torch.cuda.max_memory_allocated(device)
        free_b, total_b = torch.cuda.mem_get_info(device)
        say(f"P3 done  peak_alloc={peak_b/1e9:.2f}GB  "
            f"now free={free_b/1e9:.2f}/{total_b/1e9:.2f} GB")
        torch.cuda.reset_peak_memory_stats(device)
        _Stage.leave()

        global_step += 1
        step_time = time.time() - step_start
        _Stage.phase = ""

        metrics = {
            # ── reward (global, all rollouts in world) ─────────────────────
            "reward/mean":      reward_mean,
            "reward/std":       reward_std,
            "reward/min":       reward_min,
            "reward/max":       reward_max,
            # ── advantage (GRPO 诊断：abs_mean 越接近 0 说明 group 内 reward
            #    分散度越低 → 策略对该 prompt 的探索越窄；std 太小可能学不动)
            "advantage/mean":     adv_mean,
            "advantage/std":      adv_std,
            "advantage/abs_mean": adv_abs_mean,
            # ── optimization ──────────────────────────────────────────────
            "loss/grpo":   total_loss,
            "loss/kl":     kl_mean,
            "opt/grad_norm": grad_norm,
            "opt/lr":        lr_scheduler.get_last_lr()[0],
            # ── timing & step ─────────────────────────────────────────────
            "time/step_sec":      step_time,
            "train/global_step":  global_step,
            "train/n_rollouts":   int(all_rewards_cat.numel()),
            # 兼容旧字段名，避免之前的曲线断档
            "reward_mean": reward_mean,
            "reward_std":  reward_std,
            "reward_min":  reward_min,
            "reward_max":  reward_max,
            "grpo_loss":   total_loss,
            "kl_mean":     kl_mean,
            "grad_norm":   grad_norm,
            "lr":          lr_scheduler.get_last_lr()[0],
            "step_time":   step_time,
        }
        # per-component reward（rank0 已 populate；非 rank0 为空 dict）
        for k, v in reward_components.items():
            metrics[f"reward/{k}"] = v
        log_metrics(wandb_run, metrics, global_step, rank)

        # 日志：训练
        if rank == 0:
            with open(train_log_path, "a") as f:
                f.write(json.dumps({**metrics, "step": global_step}, default=str) + "\n")

        # ─── Checkpoint：rolling（覆盖最新 1 份） + permanent（永久节点）─────
        rolling_every = int(getattr(args, "rolling_ckpt_every", 0))
        permanent_every = int(getattr(args, "permanent_ckpt_every", 0))
        keep_n = int(getattr(args, "keep_last_n_permanent", 0))
        ts_payload = _make_training_state(global_step)
        if rolling_every > 0 and global_step % rolling_every == 0:
            save_rolling_checkpoint(raw_transformer, args.output_dir, global_step, rank,
                                     training_state=ts_payload)
        if permanent_every > 0 and global_step % permanent_every == 0:
            save_permanent_checkpoint(
                raw_transformer, args.output_dir, global_step, rank,
                keep_last_n=keep_n, training_state=ts_payload,
            )
        # 兼容旧接口：若用户没显式给新参数（都为 0），fallback 到 --checkpointing_steps
        if rolling_every == 0 and permanent_every == 0:
            if global_step % args.checkpointing_steps == 0:
                save_checkpoint(raw_transformer, args.output_dir, global_step, rank,
                                 training_state=ts_payload)

        # ─── 清理 ────────────────────────────────────────────────────────────
        del all_latents, all_log_probs, encoded_conds
        torch.cuda.empty_cache()

        if world_size > 1:
            dist.barrier()

    # ── 最终 checkpoint（无论新旧策略都额外保存一份 final）─────────────────
    save_permanent_checkpoint(raw_transformer, args.output_dir, "final", rank,
                               training_state=_make_training_state(global_step))
    if rank == 0:
        print(f"\n[Train] Training complete. Steps={global_step}")

    # 显式关闭 worker pool（atexit 是兜底）
    if worker_pool is not None:
        worker_pool.shutdown()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
