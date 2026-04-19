#!/usr/bin/env bash
# =============================================================================
# Gen3R GRPO 训练配置文件
# 所有参数均可在此处修改；run_train.sh 自动 source 本文件。
# =============================================================================

# ── 模型路径 ──────────────────────────────────────────────────────────────────
# Gen3R 主 checkpoint 目录（含 transformer/, wan_vae/, text_encoder/ 等子目录）
GEN3R_CKPT="/path/to/Gen3R/checkpoints"

# VGGT 模型目录（含 config.json 和 pytorch_model.bin 或 model.safetensors）
VGGT_PATH="${GEN3R_CKPT}/vggt"

# Geometry Adapter 模型目录
GEO_ADAPTER_PATH="${GEN3R_CKPT}/geo_adapter"

# Gen3R YAML 配置（默认使用已复制的 gen3r.yaml，通常不需要修改）
CONFIG_PATH="$(dirname "$0")/../rl_train/train/gen3r/Gen3R/gen3r/config/gen3r.yaml"

# 可选：从额外 checkpoint 加载 Transformer 权重（留空 = 使用 pretrained_model_path 下的默认）
# 例如：从上一次 RL 训练 checkpoint 恢复训练
TRANSFORMER_PATH=""

# ── 数据 ──────────────────────────────────────────────────────────────────────
# unified_data_process 输出的根目录
DATA_ROOT="/path/to/processed"

# 逗号分隔的数据集名称（目录名，即 DATA_ROOT/<dataset_name>/）
DATASETS="re10k,dl3dv"

# 帧读取模式：
#   "video"  — 从 gt.mp4 在线提取帧（慢但节省磁盘）
#   "frames" — 从预提取帧目录读取 PNG/JPG（快，需预先提取）
FRAME_MODE="video"

# 每样本采样帧数（Gen3R 训练默认 49，RL 推荐 17 以节省时间）
NUM_FRAMES=17

# 帧间隔（stride=2 则每隔 1 帧取一帧）
FRAME_STRIDE=2

# 图像分辨率（Gen3R 固定 560，unified_data_process 已输出 560x560）
RESOLUTION=560

# DataLoader 工作进程数
DATALOADER_NUM_WORKERS=4

# ── 训练超参数 ────────────────────────────────────────────────────────────────
MAX_STEPS=200
BATCH_SIZE=2                     # 每卡 batch size（样本数，不含 rollout 扩展）
LEARNING_RATE=1e-5
WEIGHT_DECAY=1e-4
MAX_GRAD_NORM=1.0
LR_SCHEDULER="constant_with_warmup"
LR_WARMUP_STEPS=0
GRADIENT_ACCUMULATION=4
GRADIENT_CHECKPOINTING="--gradient_checkpointing"    # 显存不足时开启；H100 可去掉

# 可训练模块：留空 = 全部训练；空格分隔模块名称（如 "attn proj"）
TRAINABLE_MODULES=""

SEED=42
SAMPLER_SEED=1223627

# ── GRPO 超参数 ───────────────────────────────────────────────────────────────
# 去噪步数（训练时推荐 16；推理时用 50）
SAMPLING_STEPS=16

# SDE 噪声强度（必须 > 0；设为 0 退化为 ODE 推理）
ETA=0.2

# Sigma schedule 偏移（Gen3R 默认 2.0）
SHIFT=2.0

# CFG guidance scale（Gen3R 默认 5.0）
CFG_INFER=5.0

# T5 tokenizer 最大序列长度
TOKENIZER_MAX_LENGTH=512

# 每样本生成多少条 rollout（组内归一化的组大小）
NUM_GENERATIONS=8

# 同组样本是否共享初始噪声（消融用，留空=不共享；填 "--init_same_noise" 共享）
INIT_SAME_NOISE=""

# 训练时使用的时间步比例（0.5 表示随机取 50% 时间步训练）
TIMESTEP_FRACTION=0.5

# PPO ratio 裁剪范围
CLIP_RANGE=1e-4

# Advantage 裁剪上限（防止极端 advantage 破坏训练稳定性）
ADV_CLIP_MAX=5.0

# ── Reward 配置 ───────────────────────────────────────────────────────────────
# --dry_run: 不调用 reward pipeline，返回随机 reward（用于调通训练流程）
# 调试时填 "--dry_run"；正式训练留空
DRY_RUN=""

# 计算哪些 reward（消融实验）：
#   "all" = 全部 reward
#   逗号分隔 = 只计算指定 reward，如 "camera_traj,feature_sim"
REWARDS="all"

# Reward 权重覆盖（key:value 格式，留空 = 使用 reward pipeline 的默认权重）
# 格式：REWARD_WEIGHTS="geo_semantic:3.0,geo_global:2.0,feature_sim:5.0,camera_traj:8.0,video_quality:1.5"
REWARD_WEIGHTS=""

# 是否保留 reward 中间值（frames/, intermediates/）
# "--keep_intermediates" 保留（便于 debug）；"--no_keep_intermediates" 删除（节省磁盘）
KEEP_INTERMEDIATES="--keep_intermediates"

# ── Checkpoint & 输出 ─────────────────────────────────────────────────────────
OUTPUT_DIR="/path/to/output/grpo_gen3r"

# 中间推理视频和 reward 输出目录（留空 = OUTPUT_DIR/eval_outputs）
EVAL_OUTPUT_DIR=""

# 每多少步保存一次 checkpoint
CHECKPOINTING_STEPS=50

# ── GPU 配置 ──────────────────────────────────────────────────────────────────
NUM_GPUS=1

# torchrun 主节点地址（单机留空；多机 DDP 时填实际 IP）
MASTER_ADDR="localhost"
MASTER_PORT=29500
