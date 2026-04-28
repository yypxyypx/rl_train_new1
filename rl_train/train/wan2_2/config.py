"""config.py — Wan2.2-Fun-5B-Control-Camera GRPO 训练全局参数定义。

镜像 gen3r 的 [config.py](rl_train_new/rl_train/train/gen3r/config.py)：
    1. 参数命名一一对齐；
    2. 仅把 --resolution（方形）拆成 --resolution_h / --resolution_w；
    3. 默认值改为 wan2.2 5B 推理推荐：49 帧 / 1280×704 / shift=5 / cfg=6 / 50 steps；
    4. --gradient_accumulation_steps 默认 1（单次更新，对应"每轮 rollout 1 update"）。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from paths import default_wan22_config_path

_DEFAULT_MODEL_ROOT = "/mnt/afs/visitor16/rl_train_new/model/Wan2.2-Fun-5B-Control-Camera"
_DEFAULT_CONFIG_PATH = default_wan22_config_path()

# Wan2.2-Fun-5B-Control-Camera 官方推理脚本默认 negative prompt（中文长版）。
WAN22_DEFAULT_NEG_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)


# ══════════════════════════════════════════════════════════════════════════════
# Dataclass（仅作类型化文档，实际入口走 argparse）
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    pretrained_model_path: str = _DEFAULT_MODEL_ROOT
    transformer_path: Optional[str] = None
    config_path: str = _DEFAULT_CONFIG_PATH
    t5_embed_dir: Optional[str] = None
    trainable_modules: Optional[List[str]] = None
    gradient_checkpointing: bool = True


@dataclass
class DataConfig:
    data_root: str = ""
    datasets: str = "dl3dv"
    num_frames: int = 49
    frame_stride: int = 1
    resolution_h: int = 704
    resolution_w: int = 1280
    frame_mode: str = "video"
    dataloader_num_workers: int = 4
    sampler_seed: int = 1223627


@dataclass
class GRPOConfig:
    num_generations: int = 8
    init_same_noise: bool = False

    sampling_steps: int = 50
    eta: float = 0.2
    shift: float = 5.0
    cfg_infer: float = 6.0
    cfg_rollout: float = -1.0
    cfg_train: float = -1.0
    tokenizer_max_length: int = 512

    train_timestep_strategy: str = "front"
    sde_fraction: float = 0.4
    timestep_fraction: float = 0.5

    clip_range: float = 1e-4
    adv_clip_max: float = 5.0
    kl_coeff: float = 0.01

    boundary: float = 0.875


@dataclass
class TrainConfig:
    train_batch_size: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    lr_scheduler: str = "constant_with_warmup"
    lr_warmup_steps: int = 0
    gradient_accumulation_steps: int = 1
    max_train_steps: int = 200
    seed: Optional[int] = 42
    output_dir: str = "./outputs/grpo_wan22"
    eval_output_dir: Optional[str] = None
    checkpointing_steps: int = 50
    use_8bit_adam: bool = False


@dataclass
class RewardConfig:
    rewards: str = "geo_global,camera_traj,video_quality"
    reward_weights: Optional[str] = None
    reward_model_root: Optional[str] = None
    keep_intermediates: bool = True
    skip_done: bool = True
    dry_run: bool = False
    reward_gpu_assignment: Optional[str] = None
    reward_dispatch_mode: str = "centralized"


@dataclass
class TrainingArgs:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)


# ══════════════════════════════════════════════════════════════════════════════
# argparse 工厂
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Wan2.2-Fun-5B-Control-Camera GRPO 强化学习训练",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── 模型路径 ──────────────────────────────────────────────────────────────
    mg = p.add_argument_group("Model")
    mg.add_argument("--pretrained_model_path", type=str, default=_DEFAULT_MODEL_ROOT,
                    help="Wan2.2-Fun-5B-Control-Camera 模型根目录")
    mg.add_argument("--config_path", type=str, default=_DEFAULT_CONFIG_PATH,
                    help="wan_civitai_5b.yaml 路径")
    mg.add_argument("--transformer_path", type=str, default=None,
                    help="可选：从额外 checkpoint 加载 Transformer 权重")
    mg.add_argument("--t5_embed_dir", type=str, default=None,
                    help="预计算 T5 embedding 目录（由 t5_precompute.py 生成）。"
                         "设置后不加载 T5 模型。")
    mg.add_argument("--trainable_modules", nargs="*", default=None,
                    help="可训练模块名称（含此子串的参数才训练）。None=全部训练")
    mg.add_argument("--gradient_checkpointing", action="store_true",
                    help="开启梯度检查点（时间换显存）")

    # ── 数据 ──────────────────────────────────────────────────────────────────
    dg = p.add_argument_group("Data")
    dg.add_argument("--data_root", type=str, required=True, help="数据根目录")
    dg.add_argument("--datasets", type=str, default="dl3dv",
                    help="逗号分隔的数据集名称")
    dg.add_argument("--num_frames", type=int, default=49, help="每样本采样帧数")
    dg.add_argument("--frame_stride", type=int, default=1, help="帧间隔")
    dg.add_argument("--resolution_h", type=int, default=704, help="目标高度")
    dg.add_argument("--resolution_w", type=int, default=1280, help="目标宽度")
    dg.add_argument("--frame_mode", type=str, default="video",
                    choices=["video", "frames"])
    dg.add_argument("--dataloader_num_workers", type=int, default=4)
    dg.add_argument("--sampler_seed", type=int, default=1223627)
    dg.add_argument("--num_samples_subset", type=int, default=None,
                    help="若设置 N，则用固定 sampler_seed 从全量数据中抽 N 条做 smoke test。")

    # ── GRPO 超参数 ────────────────────────────────────────────────────────────
    gg = p.add_argument_group("GRPO")
    gg.add_argument("--num_generations", type=int, default=8,
                    help="每样本 rollout 数（组内归一化组大小）")
    gg.add_argument("--init_same_noise", action="store_true")
    gg.add_argument("--sampling_steps", type=int, default=50, help="去噪总步数")
    gg.add_argument("--eta", type=float, default=0.2, help="SDE 噪声强度（>0）")
    gg.add_argument("--shift", type=float, default=5.0,
                    help="sigma schedule 偏移；wan2.2 5B 官方默认 5.0")
    gg.add_argument("--cfg_infer", type=float, default=6.0,
                    help="旧接口：CFG guidance scale 默认值（同时给 rollout 和 train 用）。")
    gg.add_argument("--cfg_rollout", type=float, default=-1.0,
                    help="Rollout 阶段 CFG (>=0 时覆盖 cfg_infer)。")
    gg.add_argument("--cfg_train", type=float, default=-1.0,
                    help="GRPO train 阶段重算 logp 用的 CFG (>=0 时覆盖 cfg_infer)。")
    gg.add_argument("--tokenizer_max_length", type=int, default=512)
    gg.add_argument("--train_timestep_strategy", type=str, default="front",
                    choices=["front", "random"],
                    help="front=前 sde_fraction 步训练; random=随机打乱取比例")
    gg.add_argument("--sde_fraction", type=float, default=0.4,
                    help="front 策略下 SDE 步占总步数的比例")
    gg.add_argument("--timestep_fraction", type=float, default=0.5,
                    help="random 策略下训练时间步比例")
    gg.add_argument("--clip_range", type=float, default=1e-4,
                    help="GRPO ratio 裁剪范围 ε")
    gg.add_argument("--adv_clip_max", type=float, default=5.0,
                    help="Advantage 裁剪上限")
    gg.add_argument("--kl_coeff", type=float, default=0.01,
                    help="KL 散度系数 β（0=关闭 KL 正则化）")
    gg.add_argument("--boundary", type=float, default=0.875,
                    help="MoE transformer 切换边界；5B single-stage 时无效，仅保留兼容字段")

    # ── 训练 ──────────────────────────────────────────────────────────────────
    tg = p.add_argument_group("Training")
    tg.add_argument("--train_batch_size", type=int, default=1)
    tg.add_argument("--learning_rate", type=float, default=1e-5)
    tg.add_argument("--weight_decay", type=float, default=1e-4)
    tg.add_argument("--max_grad_norm", type=float, default=1.0)
    tg.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
    tg.add_argument("--lr_warmup_steps", type=int, default=0)
    tg.add_argument("--gradient_accumulation_steps", type=int, default=1,
                    help="DDP 跨 step 累积步数。默认 1：每轮 rollout 完成后只触发 1 次 optimizer.step()")
    tg.add_argument("--max_train_steps", type=int, default=200)
    tg.add_argument("--seed", type=int, default=42)
    tg.add_argument("--output_dir", type=str, default="./outputs/grpo_wan22")
    tg.add_argument("--eval_output_dir", type=str, default=None)
    tg.add_argument("--checkpointing_steps", type=int, default=50,
                    help="每 N 步保存一次 checkpoint")
    tg.add_argument("--use_8bit_adam", action="store_true",
                    help="用 bitsandbytes.optim.AdamW8bit 替代 torch.optim.AdamW，省显存")

    # ── Reward ────────────────────────────────────────────────────────────────
    rg = p.add_argument_group("Reward")
    rg.add_argument("--rewards", type=str, default="geo_global,camera_traj,video_quality",
                    help="'all' 或逗号分隔子集: geo_semantic,geo_global,feature_sim,"
                         "camera_traj,video_quality。"
                         "默认关闭 feature_sim（1280×704 下 DINOv2 特征 fp16≈17GB 会 OOM）。")
    rg.add_argument("--reward_weights", type=str, default=None,
                    help="权重覆盖（key:val 逗号分隔）")
    rg.add_argument("--reward_model_root", type=str, default=None,
                    help="覆盖 RL_MODEL_ROOT 环境变量，指定 reward 模型根目录")
    rg.add_argument("--keep_intermediates", action="store_true", default=True)
    rg.add_argument("--no_keep_intermediates", dest="keep_intermediates",
                    action="store_false")
    rg.add_argument("--skip_done", action="store_true", default=True,
                    help="断点续算：跳过已存在的中间值")
    rg.add_argument("--dry_run", action="store_true", default=False,
                    help="随机 reward（调试训练流程用，不跑 reward 模型）")
    rg.add_argument("--reward_gpu_assignment", type=str, default=None,
                    help='JSON 格式分卡方案')
    rg.add_argument("--reward_dispatch_mode", type=str, default="centralized",
                    choices=["centralized", "per_rank"])

    return p


def parse_args() -> argparse.Namespace:
    args = build_parser().parse_args()

    if args.eval_output_dir is None:
        args.eval_output_dir = str(Path(args.output_dir) / "eval_outputs")

    # CFG 解耦：未显式指定时 fallback 到 cfg_infer，保持向后兼容
    if args.cfg_rollout < 0:
        args.cfg_rollout = args.cfg_infer
    if args.cfg_train < 0:
        args.cfg_train = args.cfg_infer

    return args
