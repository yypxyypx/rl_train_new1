#!/bin/bash
# run_train.sh — Wan2.2-Fun-5B-Control-Camera GRPO 训练启动脚本
#
# 用法（单机多卡）：
#   bash run_train.sh
#
# 用法（多节点 × 多卡）：
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=<node0_ip> bash run_train.sh
#   NNODES=2 NODE_RANK=1 MASTER_ADDR=<node0_ip> bash run_train.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
_DEFAULT_WAN22_CONFIG="$REPO_ROOT/eval/infer/wan2.2/VideoX-Fun/config/wan2.2/wan_civitai_5b.yaml"

# ── 路径配置 ──────────────────────────────────────────────────────────────────
PRETRAINED_MODEL_PATH="${WAN22_MODEL_PATH:-/mnt/afs/visitor16/rl_train_new/model/Wan2.2-Fun-5B-Control-Camera}"
CONFIG_PATH="${WAN22_CONFIG_PATH:-$_DEFAULT_WAN22_CONFIG}"
DATA_ROOT="${RL_DATA_ROOT:-/mnt/afs/visitor16/RL_new/datasets_wan}"
DATASETS="${RL_DATASETS:-dl3dv}"
T5_EMBED_DIR="${T5_EMBED_DIR:-/mnt/afs/visitor16/rl_train_new/data/t5_cache_wan22}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/grpo_wan22_$(date +%Y%m%d_%H%M%S)}"

# ── 分布式参数 ────────────────────────────────────────────────────────────────
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi --list-gpus | wc -l)}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"

# ── 环境 ──────────────────────────────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN

# Reward GPU 分配（默认 4 卡方案，每组 1 张；16 卡时改成 [0,1,2,3] 等列表）
REWARD_GPU_ASSIGNMENT="${REWARD_GPU_ASSIGNMENT:-}"

mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "Wan2.2-Fun-5B-Control-Camera GRPO Training"
echo "========================================"
echo "MODEL:      $PRETRAINED_MODEL_PATH"
echo "CONFIG:     $CONFIG_PATH"
echo "DATA:       $DATA_ROOT  [$DATASETS]"
echo "T5_EMBEDS:  $T5_EMBED_DIR"
echo "OUTPUT:     $OUTPUT_DIR"
echo "Nodes:      $NNODES  Rank: $NODE_RANK  NPROC: $NPROC_PER_NODE"
echo ""

EXTRA_ARGS=()
if [[ -n "$REWARD_GPU_ASSIGNMENT" ]]; then
    EXTRA_ARGS+=(--reward_gpu_assignment "$REWARD_GPU_ASSIGNMENT")
fi

# ── 启动 ──────────────────────────────────────────────────────────────────────
torchrun \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    "$SCRIPT_DIR/train_grpo.py" \
    \
    `# ── 模型路径 ──` \
    --pretrained_model_path "$PRETRAINED_MODEL_PATH" \
    --config_path "$CONFIG_PATH" \
    --t5_embed_dir "$T5_EMBED_DIR" \
    \
    `# ── 数据 ──` \
    --data_root "$DATA_ROOT" \
    --datasets "$DATASETS" \
    --num_frames 49 \
    --frame_stride 1 \
    --resolution_h 704 \
    --resolution_w 1280 \
    --frame_mode video \
    --dataloader_num_workers 2 \
    \
    `# ── 训练超参数 ──` \
    --max_train_steps 200 \
    --train_batch_size 1 \
    --learning_rate 1e-5 \
    --weight_decay 1e-4 \
    --gradient_accumulation_steps 1 \
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
    --shift 5.0 \
    --cfg_infer 6.0 \
    --train_timestep_strategy front \
    --sde_fraction 0.4 \
    --clip_range 1e-4 \
    --adv_clip_max 5.0 \
    --kl_coeff 0.01 \
    \
    `# ── Reward（feature_sim 在 1280x704 下会 OOM，先关闭） ──` \
    --rewards "geo_global,camera_traj,video_quality" \
    --reward_weights "geo_global:7.7,camera_rot:0.92,camera_trans:3.6,video_quality:0.67" \
    --keep_intermediates \
    --skip_done \
    \
    `# ── 显存优化 ──` \
    --gradient_checkpointing \
    \
    `# ── 输出 ──` \
    --output_dir "$OUTPUT_DIR" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$OUTPUT_DIR/train.log"
