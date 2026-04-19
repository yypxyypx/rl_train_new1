#!/bin/bash
# run_dl3dv_4gpu.sh — 4×4090 (24GB) 上跑 dl3dv/2K 4 条样本 × 8 rollout 完整 GRPO 1 步。
#
# 关键配置：
#   - CUDA_VISIBLE_DEVICES=1,2,3,5  → 4 张 4090
#   - num_frames=49  resolution=560  cfg_rollout=5 / cfg_train=1  num_generations=8
#   - --use_8bit_adam（4090 24GB 必开，省 4.8GB optimizer state）
#   - 不传 --trainable_modules → 全参微调
#   - --reward_dispatch_mode centralized → rank0 起 4 线程，每线程 1 卡 1 种 reward
#   - --gradient_checkpointing → 30 层全开 ckpt（必须）
#   - --kl_coeff 0 → ref_transformer 不上 GPU
#   - --num_samples_subset 4 → 固定 seed 抽 4 条样本做 smoke test
#
# 5090 (32GB) 启动脚本对照差异：
#   - export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#   - --nproc_per_node=8
#   - 删除 --use_8bit_adam（走原 fp AdamW，数值更稳）
#
# 用法：
#   bash run_dl3dv_4gpu.sh           # 全部参数默认
#   GEN3R_MODEL_PATH=/x bash run_dl3dv_4gpu.sh
#   bash run_dl3dv_4gpu.sh /custom/model/path

set -euo pipefail

# ── 路径配置（按需通过环境变量覆盖） ─────────────────────────────────────────
PRETRAINED_MODEL_PATH="${GEN3R_MODEL_PATH:-/home/users/puxin.yan-labs/RL_code/eval/infer/gen3r/Gen3R/checkpoints}"
CONFIG_PATH="${GEN3R_CONFIG_PATH:-$(dirname $(dirname "$0"))/Gen3R/gen3r/config/gen3r.yaml}"
DATA_ROOT="${RL_DATA_ROOT:-/home/users/puxin.yan-labs/RL_code/data/alsfna_rl_data/gen3r}"
DATASETS="${RL_DATASETS:-dl3dv/2K}"
T5_EMBED_DIR="${T5_EMBED_DIR:-/home/users/puxin.yan-labs/RL_code/data/alsfna_rl_data/gen3r/_t5_cache_dl3dv}"
RL_MODEL_ROOT="${RL_MODEL_ROOT:-/home/users/puxin.yan-labs/RL/model}"
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "$0")/results/dl3dv_4gpu_$(date +%Y%m%d_%H%M%S)}"

if [ $# -ge 1 ]; then
    PRETRAINED_MODEL_PATH="$1"
fi

# ── 路径推导 ─────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$(dirname "$SCRIPT_DIR")"

# ── 环境 ─────────────────────────────────────────────────────────────────────
# 4 张物理卡：1,2,3,5（local rank 0/1/2/3 ↔ 物理 GPU 1/2/3/5）
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,5}"
# CUDA OOM 优化（24GB 紧张时缓解碎片）
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# NCCL watchdog timeout 设大（默认 600s）。
# 因为 centralized reward 模式下 rank0 单线程协调 4 卡跑 32 rollout 的 reward，
# 可能耗时 30+ 分钟，期间 ranks 1/2/3 在 dist.broadcast 等待，必须避免被 watchdog 杀死。
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-14400}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-0}"
export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-0}"
# Reward 子进程读取 reward 模型根目录
export RL_MODEL_ROOT
# 关闭 wandb（smoke test）
export WANDB_MODE="${WANDB_MODE:-disabled}"

# ── 打印配置 ─────────────────────────────────────────────────────────────────
echo "========================================"
echo "Gen3R GRPO 4-GPU dl3dv/2K Smoke Test (4×4090 24GB)"
echo "========================================"
echo "MODEL:         $PRETRAINED_MODEL_PATH"
echo "CONFIG:        $CONFIG_PATH"
echo "DATA:          $DATA_ROOT  [$DATASETS]"
echo "T5_EMBEDS:     $T5_EMBED_DIR"
echo "RL_MODEL_ROOT: $RL_MODEL_ROOT"
echo "OUTPUT:        $OUTPUT_DIR"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

mkdir -p "$OUTPUT_DIR"

# ── 启动 4 卡训练 ────────────────────────────────────────────────────────────
torchrun \
    --standalone \
    --nproc_per_node=4 \
    "$TRAIN_DIR/train_grpo.py" \
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
    --resolution 560 \
    --frame_mode video \
    --num_samples_subset 4 \
    --sampler_seed 1223627 \
    --dataloader_num_workers 2 \
    \
    `# ── 训练超参数（smoke test：1 步 + 立刻 ckpt） ──` \
    --max_train_steps 1 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --weight_decay 1e-4 \
    --max_grad_norm 1.0 \
    --checkpointing_steps 1 \
    --use_8bit_adam \
    \
    `# ── GRPO 参数 ──` \
    --num_generations 8 \
    --sampling_steps 20 \
    --eta 0.2 \
    --shift 2.0 \
    --cfg_rollout 5.0 \
    --cfg_train 1.0 \
    --train_timestep_strategy front \
    --sde_fraction 0.5 \
    --clip_range 1e-4 \
    --adv_clip_max 5.0 \
    --kl_coeff 0 \
    \
    `# ── Reward（centralized：每张卡 1 种 reward × 全部 32 rollout） ──` \
    --rewards all \
    --reward_dispatch_mode centralized \
    --keep_intermediates \
    --skip_done \
    \
    `# ── 显存优化（4090 24GB 必须）──` \
    --gradient_checkpointing \
    \
    `# ── 输出 ──` \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/train.log"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "FIRST RUN PASSED  ✓"
    echo "Output: $OUTPUT_DIR"
    echo ""
    echo "Generated videos:"
    ls "$OUTPUT_DIR/eval_outputs/step_0/" 2>/dev/null | head -20 || true
    echo ""
    echo "Checkpoint:"
    ls "$OUTPUT_DIR/checkpoint-1/" 2>/dev/null || true
    echo ""
    echo "Training log (last 30 lines):"
    tail -30 "$OUTPUT_DIR/training_log.jsonl" 2>/dev/null || true
else
    echo "FIRST RUN FAILED  ✗  (exit code: $EXIT_CODE)"
    echo "Log tail:"
    tail -80 "$OUTPUT_DIR/train.log" 2>/dev/null || true
fi
echo "========================================"

exit $EXIT_CODE
