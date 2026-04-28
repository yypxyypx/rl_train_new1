"""config.py — Gen3R GRPO 训练全局参数定义。

使用 dataclass 作为类型化配置载体，同时提供 argparse 工厂函数。
所有子模块（model_loader_v2, grpo_engine, reward_dispatcher, train_grpo_v2）
都从此处导入，避免参数分散在各文件中。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class ModelConfig:
    """模型路径与加载配置。"""
    pretrained_model_path: str = ""
    transformer_path: Optional[str] = None       # 可选：额外 transformer 权重
    vggt_path: str = ""                          # 参数保留，但默认不加载
    geo_adapter_path: str = ""                   # 参数保留，但不加载（全零替代）
    config_path: str = ""                        # gen3r.yaml 路径
    t5_embed_dir: Optional[str] = None          # 预计算 T5 embedding 目录
    trainable_modules: Optional[List[str]] = None  # None=全部可训练
    gradient_checkpointing: bool = True


@dataclass
class DataConfig:
    """数据集与采样配置。"""
    data_root: str = ""
    datasets: str = "re10k,dl3dv"               # 逗号分隔
    num_frames: int = 17
    frame_stride: int = 2
    resolution: int = 560
    frame_mode: str = "video"                    # "video" | "frames"
    dataloader_num_workers: int = 4
    sampler_seed: int = 1223627


@dataclass
class GRPOConfig:
    """GRPO 算法超参数。"""
    # Rollout
    num_generations: int = 8                     # 每样本 rollout 数（组大小）
    init_same_noise: bool = False

    # Diffusion sampling
    sampling_steps: int = 50
    eta: float = 0.2                             # SDE 噪声强度（> 0）
    shift: float = 2.0                           # sigma schedule 偏移
    cfg_infer: float = 5.0                       # 旧接口：默认值（同时控制 rollout 和 train）
    cfg_rollout: float = -1.0                    # rollout/inference CFG（>=0 时覆盖 cfg_infer）
    cfg_train: float = -1.0                      # GRPO train logp 重算 CFG（>=0 时覆盖 cfg_infer）
    tokenizer_max_length: int = 512

    # SDE/ODE 策略
    train_timestep_strategy: str = "front"       # "front" | "random"
    sde_fraction: float = 0.4                    # front 模式下 SDE 步比例
    timestep_fraction: float = 0.5              # random 模式下训练步比例
    train_steps_count: int = 0                   # front 模式下训练阶段截取前 N 个 SDE 步（0=全部）

    # PPO/GRPO ratio clip
    clip_range: float = 1e-4
    adv_clip_max: float = 5.0

    # KL 散度正则化
    kl_coeff: float = 0.01                       # 0 = 关闭 KL


@dataclass
class TrainConfig:
    """训练循环配置。"""
    train_batch_size: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    lr_scheduler: str = "constant_with_warmup"
    lr_warmup_steps: int = 0
    gradient_accumulation_steps: int = 4
    max_train_steps: int = 200
    seed: Optional[int] = 42
    output_dir: str = "./outputs/grpo_gen3r_v2"
    eval_output_dir: Optional[str] = None
    checkpointing_steps: int = 50
    use_8bit_adam: bool = False                  # 4090 24GB 必开；5090 32GB 默认关


@dataclass
class RewardConfig:
    """Reward 管线配置。"""
    rewards: str = "all"                         # "all" 或逗号分隔子集
    reward_weights: Optional[str] = None         # "key:v,key:v" 格式覆盖权重
    reward_model_root: Optional[str] = None      # 覆盖 RL_MODEL_ROOT 环境变量
    keep_intermediates: bool = True              # 是否保留中间值文件
    skip_done: bool = True                       # 断点续算
    dry_run: bool = False                        # 随机 reward（调试用）

    # 分卡策略（可通过 JSON 字符串指定）
    # 例：'{"da3":[0,1],"qwen_sam3":[2,3],"dinov2_extract":[4,5],"videoalign":[6,7]}'
    reward_gpu_assignment: Optional[str] = None

    # 调度模式：
    #   centralized — rank0 起 4 线程，每线程绑定一张 GPU + 一种 reward step，串行处理全部 rollout
    #   per_rank    — 旧逻辑，每个 DDP rank 自己跑 reward 子进程
    reward_dispatch_mode: str = "centralized"


@dataclass
class TrainingArgs:
    """完整训练参数，聚合所有子配置。"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)


# ══════════════════════════════════════════════════════════════════════════════
# argparse 工厂
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    """构建完整的命令行参数解析器。"""
    p = argparse.ArgumentParser(
        description="Gen3R GRPO 强化学习训练",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── 模型路径 ──────────────────────────────────────────────────────────────
    mg = p.add_argument_group("Model")
    mg.add_argument("--pretrained_model_path", type=str, required=True,
                    help="Gen3R checkpoints 目录")
    mg.add_argument("--vggt_path", type=str, default="",
                    help="VGGT 路径（当前流程不使用，保留接口）")
    mg.add_argument("--geo_adapter_path", type=str, default="",
                    help="GeometryAdapter 路径（当前流程不使用，保留接口）")
    mg.add_argument("--config_path", type=str,
                    default=str(Path(__file__).parent / "Gen3R" / "gen3r" / "config" / "gen3r.yaml"),
                    help="gen3r.yaml 配置路径")
    mg.add_argument("--transformer_path", type=str, default=None,
                    help="可选：从额外 checkpoint 加载 Transformer 权重")
    mg.add_argument("--t5_embed_dir", type=str, default=None,
                    help="预计算 T5 embedding 目录（由 t5_precompute.py 生成）。"
                         "设置后不加载 T5 模型，直接读取缓存 tensor。")
    mg.add_argument("--trainable_modules", nargs="*", default=None,
                    help="可训练模块名称（含此子串的参数才训练）。None=全部训练")
    mg.add_argument("--gradient_checkpointing", action="store_true",
                    help="开启梯度检查点（时间换显存）。"
                         "全局开关，等价于 --gradient_checkpoint_layers=-1（所有 block 都 ckpt）。"
                         "若同时给 --gradient_checkpoint_layers，后者优先。")
    mg.add_argument("--gradient_checkpoint_layers", type=int, default=0,
                    help="选择性梯度检查点：开前 N 个 block 的 ckpt，其余正常 forward。"
                         "0（默认）= 全关。 -1 = 全开（等价 --gradient_checkpointing）。"
                         "8 ≈ 在 H100 80G 上恰好放下 N=8/K=50/train_T=30/res=560 的 P3，速度损失 ~8%%。"
                         "颗粒度在 block 级（wan transformer 30 个 block，每开 1 层省 ~1.7GB activation）。")

    # ── 数据 ──────────────────────────────────────────────────────────────────
    dg = p.add_argument_group("Data")
    dg.add_argument("--data_root", type=str, required=True, help="数据根目录")
    dg.add_argument("--datasets", type=str, default="re10k,dl3dv",
                    help="逗号分隔的数据集名称（manifest 模式下用作 dataset 字段过滤）")
    dg.add_argument("--train_manifest", type=str, default="",
                    help="训练样本 manifest jsonl 路径（推荐）。每行 {dataset, sample_id, "
                         "sample_dir, metadata_json, ...}。设置后 RLDataset 直接读 manifest，"
                         "不再扫 data_root/<ds>/<sample> 目录。空字符串 = 走目录扫描兼容模式。")
    dg.add_argument("--num_frames", type=int, default=17, help="每样本采样帧数")
    dg.add_argument("--frame_stride", type=int, default=2, help="帧间隔")
    dg.add_argument("--resolution", type=int, default=560, help="图像分辨率")
    dg.add_argument("--frame_mode", type=str, default="video",
                    choices=["video", "frames"])
    dg.add_argument("--dataloader_num_workers", type=int, default=4)
    dg.add_argument("--sampler_seed", type=int, default=1223627)
    dg.add_argument("--num_samples_subset", type=int, default=None,
                    help="若设置 N，则用固定 sampler_seed 从全量数据中抽 N 条做 smoke test。"
                         "默认 None=全量训练。")
    dg.add_argument("--p3_synthetic", action="store_true",
                    help="P3 显存压力测试模式：跳过 worker pool / rollout / reward，"
                         "用随机 tensor 模拟所有 Phase 1/2 输出，直接进 grpo_update。"
                         "用于快速验证 GRPO update 显存峰值，~3min 一轮。")
    dg.add_argument("--p3_max_iters", type=int, default=0,
                    help="grpo_update 内只跑 N 个 forward+backward iter 后立即返回，"
                         "用于快速测显存峰值。0=正常跑完所有 iter。"
                         "搭配 --p3_synthetic 可在 ~30s 内测出 P3 peak_alloc。")

    # ── GRPO 超参数 ────────────────────────────────────────────────────────────
    gg = p.add_argument_group("GRPO")
    gg.add_argument("--num_generations", type=int, default=8,
                    help="每样本 rollout 数（GRPO 组内归一化组大小，G）。"
                         "16×H100 正式训练：G=8（一个 group 内 8 个 rollout 共享同一 prompt）。")
    gg.add_argument("--rollouts_per_rank", type=int, default=0,
                    help="每张 GPU 负责的 rollout 数（R）。"
                         "0（默认）= 兼容旧路径：R = num_generations，每卡独立完成一个 group。"
                         ">0 时启用 sub-group 模式：ranks_per_group = G/R 张卡共享 1 个 prompt，"
                         "在 sub-group 内做 reward all_gather + 组内 advantage 归一化。"
                         "16×H100 正式训练设 4：8 group × 8 rollout / step，"
                         "每卡 4 rollout × 16 卡 = 64 rollout / step。"
                         "要求 num_generations 必须能被 rollouts_per_rank 整除。")
    gg.add_argument("--init_same_noise", action="store_true")
    gg.add_argument("--sampling_steps", type=int, default=50, help="去噪总步数")
    gg.add_argument("--eta", type=float, default=0.2, help="SDE 噪声强度（>0）")
    gg.add_argument("--shift", type=float, default=2.0, help="sigma schedule 偏移")
    gg.add_argument("--cfg_infer", type=float, default=5.0,
                    help="旧接口：CFG guidance scale 默认值（同时给 rollout 和 train 用）。"
                         "推荐改用 --cfg_rollout / --cfg_train 解耦。")
    gg.add_argument("--cfg_rollout", type=float, default=-1.0,
                    help="Rollout 阶段 CFG (>=0 时覆盖 cfg_infer)。"
                         "建议 5.0 提升生成质量；no_grad 不增加显存。")
    gg.add_argument("--cfg_train", type=float, default=-1.0,
                    help="GRPO train 阶段重算 logp 用的 CFG (>=0 时覆盖 cfg_infer)。"
                         "建议 1.0 节省双倍 batch 激活显存（4090 24GB 必需）。")
    gg.add_argument("--tokenizer_max_length", type=int, default=512)
    gg.add_argument("--train_timestep_strategy", type=str, default="front",
                    choices=["front", "random"],
                    help="front=前 sde_fraction 步训练; random=随机打乱取比例")
    gg.add_argument("--sde_fraction", type=float, default=0.4,
                    help="front 策略下 SDE 步占总步数的比例（rollout 阶段）。"
                         "1.0 = 全部步骤都是 SDE，全部 timestep 都记录 log_prob。")
    gg.add_argument("--timestep_fraction", type=float, default=0.5,
                    help="random 策略下训练时间步比例")
    gg.add_argument("--train_steps_count", type=int, default=0,
                    help="front 策略下，训练阶段对前 N 个 SDE timestep 计算梯度。"
                         "0（默认）= 用全部 K 个 SDE 步（K = sampling_steps × sde_fraction）。"
                         ">0 时取 min(K, train_steps_count)。"
                         "用法举例：sampling_steps=50 + sde_fraction=1.0 + train_steps_count=30 "
                         "→ rollout 50 步全 SDE，但只对前 30 步计算策略梯度。")
    gg.add_argument("--clip_range", type=float, default=1e-4,
                    help="GRPO ratio 裁剪范围 ε")
    gg.add_argument("--adv_clip_max", type=float, default=5.0,
                    help="Advantage 裁剪上限")
    gg.add_argument("--kl_coeff", type=float, default=0.01,
                    help="KL 散度系数 β（0=关闭 KL 正则化）")
    gg.add_argument("--train_microbatch_size", type=int, default=1,
                    help="Phase 3 GRPO update 每个 timestep 一次性塞多少条 rollout 进 transformer。"
                         "1=老逻辑（每条 rollout 单独 fwd/bwd，N×train_T 次）；"
                         ">1=同 timestep 把若干 rollout 拼到 batch 维一次跑。"
                         "受 num_generations 限制（必须整除）。"
                         "建议：ckpt ON 时 4；ckpt OFF 时 1。")
    gg.add_argument("--vae_decode_micro_batch", type=int, default=4,
                    help="Phase 1 末尾批量 VAE decode 时每批多少条 rollout。"
                         "默认 4，过大可能 OOM（H100 80GB 上 8 也能跑）。"
                         "1 = 退化为旧的逐条解码。")

    # ── 训练 ──────────────────────────────────────────────────────────────────
    tg = p.add_argument_group("Training")
    tg.add_argument("--train_batch_size", type=int, default=1)
    tg.add_argument("--learning_rate", type=float, default=1e-5)
    tg.add_argument("--weight_decay", type=float, default=1e-4)
    tg.add_argument("--max_grad_norm", type=float, default=1.0)
    tg.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
    tg.add_argument("--lr_warmup_steps", type=int, default=0)
    tg.add_argument("--gradient_accumulation_steps", type=int, default=4)
    tg.add_argument("--max_train_steps", type=int, default=200)
    tg.add_argument("--seed", type=int, default=42)
    tg.add_argument("--output_dir", type=str, default="./outputs/grpo_gen3r_v2")
    tg.add_argument("--eval_output_dir", type=str, default=None)
    tg.add_argument("--checkpointing_steps", type=int, default=50,
                    help="[旧接口] 每 N 步保存一次 checkpoint。"
                         "若同时给了 --permanent_ckpt_every，以后者为准。")
    tg.add_argument("--rolling_ckpt_every", type=int, default=4,
                    help="滚动 checkpoint 步频（短期）。每 N 步保存一次，"
                         "保存新的同时删除上一个滚动 ckpt（即只保留最新 1 份）。"
                         "用于断电/抢占恢复，不长期占盘。0 = 关闭滚动 ckpt。")
    tg.add_argument("--permanent_ckpt_every", type=int, default=50,
                    help="永久 checkpoint 步频（长期）。每 N 步保存一次且不删除，"
                         "作为里程碑节点。0 = 关闭永久 ckpt（不推荐）。")
    tg.add_argument("--keep_last_n_permanent", type=int, default=0,
                    help="永久 ckpt 最大保留数量。0（默认）= 不限制；>0 时只保留最新 N 个，"
                         "更老的会被删除。用于长跑磁盘紧张时。")
    tg.add_argument("--resume_from", type=str, default="",
                    help="断点恢复路径。"
                         " 空字符串    = 从头训练（默认）。"
                         " 'auto'      = 在 --output_dir 下自动找 step 最大的 ckpt 续训"
                         "              （rolling > permanent > checkpoint 优先级）。"
                         " 显式路径    = 指定一个 ckpt 目录（如 /.../rolling-80）。"
                         "恢复时会还原 model 权重 + global_step + sampler_seed，"
                         "并跳过已用过的样本（按 step × n_groups_total 复现样本顺序）。"
                         "注意：optimizer/scheduler 状态目前不持久化，恢复后前几步 loss "
                         "会有短暂抖动属正常现象。")
    tg.add_argument("--use_8bit_adam", action="store_true",
                    help="用 bitsandbytes.optim.AdamW8bit 替代 torch.optim.AdamW，"
                         "把 optimizer state 从 4B/param 压到 1B/param，"
                         "省 ~4.8GB（1.6B model）。4090 24GB 必开；"
                         "5090 32GB 显存充足时关闭以保数值稳定。")

    # ── Reward ────────────────────────────────────────────────────────────────
    rg = p.add_argument_group("Reward")
    rg.add_argument("--rewards", type=str, default="all",
                    help="'all' 或逗号分隔子集: geo_semantic,geo_global,feature_sim,"
                         "camera_traj,video_quality")
    rg.add_argument("--reward_weights", type=str, default=None,
                    help="权重覆盖（key:val 逗号分隔）。默认权重见 reward_metrics."
                         "compute_all_rewards 中 weights。新方案使用 camera_rot/"
                         "camera_trans 单独加权，例: "
                         "'geo_global:7.7,feature_sim:5.3,camera_rot:0.92,"
                         "camera_trans:3.6,video_quality:0.67,geo_semantic:0.0'")
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
                    help='JSON 格式分卡方案: \'{"da3":[0,1],"qwen_sam3":[2,3],'
                         '"dinov2_extract":[4,5],"videoalign":[6,7]}\'')
    rg.add_argument("--reward_dispatch_mode", type=str, default="centralized",
                    choices=["centralized", "per_rank", "workers"],
                    help="workers=新架构，每个 rank 起 3 个常驻 worker 子进程"
                         "（DA3/DINOv2/VideoAlign），中间产物走 /dev/shm，主进程"
                         "自己 GPU 聚合 reward；centralized=rank0 起 4 线程绑 4 GPU；"
                         "per_rank=旧逻辑（每 rank 各跑 reward 子进程）。")
    rg.add_argument("--geo_compare_mode", type=str, default="all_pairs",
                    choices=["first_frame", "adjacent", "all_pairs"],
                    help="geo_global / geo_semantic 投影时帧对策略。"
                         "all_pairs=每帧 vs 其它所有帧（默认，AP 方案）。")
    rg.add_argument("--feature_compare_mode", type=str, default="first_frame",
                    choices=["first_frame", "adjacent", "all_pairs"],
                    help="feature_sim (DINOv2) warping 时帧对策略。"
                         "first_frame=每帧投影到第 0 帧，按有效像素求 cosine 均值（最终选定）。"
                         "all_pairs=每帧 vs 其它所有帧（与 DA3 几何对齐，但 N^2 计算量太大）。")
    rg.add_argument("--va_micro_batch", type=int, default=0,
                    help="VideoAlign worker 单次 inferencer.reward(...) 接收的视频条数上限。"
                         "0 = 一次性把全部 N 条 rollout 真 batch（多卡训练时 VideoAlign 独占一张 "
                         "H100，足够吃下 N=8）。"
                         ">0 = 把 N 条切成多个大小为 va_micro_batch 的 chunk，串行跑 "
                         "（单卡 smoke 必须用 1 或 2，否则 Qwen2-VL attention 会 OOM）。")
    rg.add_argument("--reward_agg_micro_batch", type=int, default=2,
                    help="主进程聚合 reward 时的 micro batch（每 N 条 rollout empty_cache 一次）。")

    return p


def parse_args() -> argparse.Namespace:
    """解析命令行参数，并做后处理。"""
    args = build_parser().parse_args()

    if args.eval_output_dir is None:
        args.eval_output_dir = str(Path(args.output_dir) / "eval_outputs")

    # CFG 解耦：未显式指定时 fallback 到 cfg_infer，保持向后兼容
    if args.cfg_rollout < 0:
        args.cfg_rollout = args.cfg_infer
    if args.cfg_train < 0:
        args.cfg_train = args.cfg_infer

    # ── rollouts_per_rank 校验 ───────────────────────────────────────────────
    # 0 = 旧路径：每卡独立 1 个完整 group，R = G
    if args.rollouts_per_rank == 0:
        args.rollouts_per_rank = args.num_generations
    if args.num_generations % args.rollouts_per_rank != 0:
        raise ValueError(
            f"num_generations ({args.num_generations}) 必须被 rollouts_per_rank "
            f"({args.rollouts_per_rank}) 整除"
        )

    return args
