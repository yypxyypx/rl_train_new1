#!/bin/bash
# test_grpo_4090.sh — 4×4090 端到端 GRPO 训练测试脚本
#
# 此脚本用于验证完整的训练流程在 4×4090 上可以跑通：
#   - 数据加载
#   - Rollout（采样 2 条）
#   - Reward 计算（dry_run 模式跳过实际模型，快速验证流程）
#   - GRPO update + KL
#   - Checkpoint 保存
#
# 正式训练参数见 run_train.sh（16×5090）
#
# 用法：
#   bash test_grpo_4090.sh          # 全部参数使用默认值
#   bash test_grpo_4090.sh /custom/model/path  # 覆盖模型路径

set -euo pipefail

# ── 路径配置（按需修改） ───────────────────────────────────────────────────────
PRETRAINED_MODEL_PATH="${GEN3R_MODEL_PATH:-/path/to/gen3r_ckpts}"
CONFIG_PATH="${GEN3R_CONFIG_PATH:-$(dirname $(dirname "$0"))/Gen3R/gen3r/config/gen3r.yaml}"
DATA_ROOT="${RL_DATA_ROOT:-/path/to/data}"
DATASETS="${RL_DATASETS:-re10k}"
T5_EMBED_DIR="${T5_EMBED_DIR:-/path/to/t5_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "$0")/results/test_grpo_4090_$(date +%Y%m%d_%H%M%S)}"

# 允许第一个参数覆盖 model path
if [ $# -ge 1 ]; then
    PRETRAINED_MODEL_PATH="$1"
fi

# ── 环境 ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$(dirname "$SCRIPT_DIR")"

# CUDA OOM 优化
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# 跳过实际 reward 模型（加速端到端测试）
export RL_DRY_RUN=1

# ── 打印配置 ──────────────────────────────────────────────────────────────────
echo "========================================"
echo "Gen3R GRPO End-to-End Test (4×4090)"
echo "========================================"
echo "MODEL:      $PRETRAINED_MODEL_PATH"
echo "CONFIG:     $CONFIG_PATH"
echo "DATA:       $DATA_ROOT  [$DATASETS]"
echo "T5_EMBEDS:  $T5_EMBED_DIR"
echo "OUTPUT:     $OUTPUT_DIR"
echo ""

N_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "GPUs: $N_GPUS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

# ── 启动 ──────────────────────────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"

torchrun \
    --standalone \
    --nproc_per_node="${N_GPUS}" \
    "$TRAIN_DIR/train_grpo_v2.py" \
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
    \
    `# ── 训练超参数（极小，仅测试流程）──` \
    --max_train_steps 5 \
    --train_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 1 \
    --checkpointing_steps 5 \
    --max_grad_norm 1.0 \
    \
    `# ── GRPO 参数 ──` \
    --num_generations 2 \
    --sampling_steps 10 \
    --eta 0.2 \
    --shift 2.0 \
    --cfg_infer 5.0 \
    --train_timestep_strategy front \
    --sde_fraction 0.4 \
    --clip_range 1e-4 \
    --adv_clip_max 5.0 \
    --kl_coeff 0.01 \
    \
    `# ── Reward（dry_run 跳过实际推理）──` \
    --rewards "camera_traj,feature_sim" \
    --dry_run \
    --keep_intermediates \
    \
    `# ── 显存优化 ──` \
    --gradient_checkpointing \
    \
    `# ── 输出 ──` \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/train.log"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "TEST PASSED  ✓"
    echo "Output: $OUTPUT_DIR"
    ls -la "$OUTPUT_DIR/"
else
    echo "TEST FAILED  ✗  (exit code: $EXIT_CODE)"
    echo "Log tail:"
    tail -50 "$OUTPUT_DIR/train.log" 2>/dev/null || true
fi
echo "========================================"

exit $EXIT_CODE
