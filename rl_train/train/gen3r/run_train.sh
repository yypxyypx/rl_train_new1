#!/bin/bash
# run_train.sh — 16×5090 正式训练启动脚本
#
# 用法（单机 16 卡）：
#   bash run_train.sh
#
# 用法（2 节点 × 8 卡）：
#   # 节点 0（master）：
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=<node0_ip> bash run_train.sh
#   # 节点 1：
#   NNODES=2 NODE_RANK=1 MASTER_ADDR=<node0_ip> bash run_train.sh

set -euo pipefail

# ── 路径配置（必须修改） ──────────────────────────────────────────────────────
PRETRAINED_MODEL_PATH="${GEN3R_MODEL_PATH:-/path/to/gen3r_ckpts}"
CONFIG_PATH="${GEN3R_CONFIG_PATH:-$(dirname "$0")/Gen3R/gen3r/config/gen3r.yaml}"
DATA_ROOT="${RL_DATA_ROOT:-/path/to/data}"
DATASETS="${RL_DATASETS:-re10k,dl3dv}"
T5_EMBED_DIR="${T5_EMBED_DIR:-/path/to/t5_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/grpo_gen3r_5090_$(date +%Y%m%d_%H%M%S)}"

# ── 分布式参数 ────────────────────────────────────────────────────────────────
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi --list-gpus | wc -l)}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"

# ── 环境 ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN

# Reward GPU 分配（16 卡，每组 4 张）
REWARD_GPU_ASSIGNMENT='{"da3":[0,1,2,3],"qwen_sam3":[4,5,6,7],"dinov2_extract":[8,9,10,11],"videoalign":[12,13,14,15]}'

# ── 打印配置 ──────────────────────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "Gen3R GRPO Training — 16×5090"
echo "========================================"
echo "MODEL:      $PRETRAINED_MODEL_PATH"
echo "DATA:       $DATA_ROOT  [$DATASETS]"
echo "T5_EMBEDS:  $T5_EMBED_DIR"
echo "OUTPUT:     $OUTPUT_DIR"
echo "Nodes:      $NNODES  Rank: $NODE_RANK  NPROC: $NPROC_PER_NODE"
echo ""

# ── 启动 ──────────────────────────────────────────────────────────────────────
torchrun \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    "$SCRIPT_DIR/train_grpo_v2.py" \
    \
    `# ── 模型路径 ──` \
    --pretrained_model_path "$PRETRAINED_MODEL_PATH" \
    --config_path "$CONFIG_PATH" \
    --t5_embed_dir "$T5_EMBED_DIR" \
    \
    `# ── 数据 ──` \
    --data_root "$DATA_ROOT" \
    --datasets "$DATASETS" \
    --num_frames 17 \
    --frame_stride 2 \
    --resolution 560 \
    --frame_mode video \
    --dataloader_num_workers 4 \
    \
    `# ── 训练超参数 ──` \
    --max_train_steps 200 \
    --train_batch_size 1 \
    --learning_rate 1e-5 \
    --weight_decay 1e-4 \
    --gradient_accumulation_steps 4 \
    --checkpointing_steps 50 \
    --max_grad_norm 1.0 \
    --lr_scheduler constant_with_warmup \
    --lr_warmup_steps 10 \
    --seed 42 \
    \
    `# ── GRPO 超参数 ──` \
    --num_generations 8 \
    --sampling_steps 50 \
    --eta 0.2 \
    --shift 2.0 \
    --cfg_infer 5.0 \
    --train_timestep_strategy front \
    --sde_fraction 0.4 \
    --clip_range 1e-4 \
    --adv_clip_max 5.0 \
    --kl_coeff 0.01 \
    \
    `# ── Reward ──` \
    --rewards all \
    `# 新权重：387 sample × 8 rollout 标定，目标 w·within_std ≈ 0.10` \
    `# 5 项 reward 在 advantage 中贡献度均衡 (~28%~39% 每项)` \
    `# geo_semantic 因 SAM3 不稳定弃用 (weight=0)` \
    --reward_weights "geo_global:7.7,feature_sim:5.3,camera_rot:0.92,camera_trans:3.6,video_quality:0.67,geo_semantic:0.0" \
    --reward_gpu_assignment "$REWARD_GPU_ASSIGNMENT" \
    --keep_intermediates \
    --skip_done \
    \
    `# ── 显存优化 ──` \
    --gradient_checkpointing \
    \
    `# ── 输出 ──` \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/train.log"
