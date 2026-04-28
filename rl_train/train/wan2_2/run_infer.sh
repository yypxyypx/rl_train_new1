#!/usr/bin/env bash
# =============================================================================
# Wan2.2-Fun-5B-Control-Camera 推理调试启动脚本
#
# 用法：
#   方式 A：批量推理
#     bash run_infer.sh
#
#   方式 B：单个样本精确调试
#     bash run_infer.sh --sample_dir /path/to/sample
#
#   常用覆盖参数
#     bash run_infer.sh --max_samples 2 --num_rollouts 3 --sampling_steps 20
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
_DEFAULT_WAN22_CONFIG="$REPO_ROOT/eval/infer/wan2.2/VideoX-Fun/config/wan2.2/wan_civitai_5b.yaml"

# ── 默认配置 ──────────────────────────────────────────────────────────────────
WAN22_MODEL_PATH="${WAN22_MODEL_PATH:-/mnt/afs/visitor16/rl_train_new/model/Wan2.2-Fun-5B-Control-Camera}"
WAN22_CONFIG_PATH="${WAN22_CONFIG_PATH:-$_DEFAULT_WAN22_CONFIG}"
DATA_ROOT="${DATA_ROOT:-/mnt/afs/visitor16/RL_new/datasets_wan}"
DATASETS="${DATASETS:-dl3dv}"
OUTPUT_DIR="${OUTPUT_DIR:-./infer_outputs/$(date +%Y%m%d_%H%M%S)}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-2}"
SAMPLING_STEPS="${SAMPLING_STEPS:-50}"
ETA="${ETA:-0.0}"
SHIFT="${SHIFT:-5.0}"
CFG_INFER="${CFG_INFER:-6.0}"
NUM_FRAMES="${NUM_FRAMES:-49}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
RESOLUTION_H="${RESOLUTION_H:-704}"
RESOLUTION_W="${RESOLUTION_W:-1280}"
SEED_BASE="${SEED_BASE:-42}"

EXTRA_ARGS=("$@")

mkdir -p "$OUTPUT_DIR"

echo ""
echo "======================================================"
echo "  Wan2.2-Fun-5B-Control-Camera Inference"
echo "======================================================"
echo "  Model:          $WAN22_MODEL_PATH"
echo "  Data root:      $DATA_ROOT"
echo "  Output:         $OUTPUT_DIR"
echo "  Num rollouts:   $NUM_ROLLOUTS"
echo "  Sampling steps: $SAMPLING_STEPS"
echo "  Resolution:     ${RESOLUTION_W}x${RESOLUTION_H} x ${NUM_FRAMES} frames"
echo "======================================================"
echo ""

cd "$SCRIPT_DIR"
python infer_only.py \
    --pretrained_model_path "$WAN22_MODEL_PATH" \
    --config_path "$WAN22_CONFIG_PATH" \
    --data_root "$DATA_ROOT" \
    --datasets "$DATASETS" \
    --output_dir "$OUTPUT_DIR" \
    --num_rollouts "$NUM_ROLLOUTS" \
    --max_samples "$MAX_SAMPLES" \
    --num_frames "$NUM_FRAMES" \
    --frame_stride "$FRAME_STRIDE" \
    --resolution_h "$RESOLUTION_H" \
    --resolution_w "$RESOLUTION_W" \
    --sampling_steps "$SAMPLING_STEPS" \
    --eta "$ETA" \
    --shift "$SHIFT" \
    --cfg_infer "$CFG_INFER" \
    --seed_base "$SEED_BASE" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$OUTPUT_DIR/infer.log"
