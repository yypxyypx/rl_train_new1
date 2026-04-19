#!/usr/bin/env bash
# =============================================================================
# Gen3R 推理调试启动脚本
#
# 用法：
#   # 方式 A：批量推理（从数据目录取前 N 个样本）
#   bash run_infer.sh [--config /path/to/config.sh] [额外参数]
#
#   # 方式 B：单个样本精确调试
#   bash run_infer.sh --sample_dir /path/to/processed/dl3dv/0a6c01ac32127687
#
#   # 常用覆盖参数
#   bash run_infer.sh --max_samples 2 --num_rollouts 3 --sampling_steps 20
#
# 输出目录结构（兼容 batch_reward.sh）：
#   <OUTPUT_DIR>/<dataset>/<sample_id>/
#       infer_info.json
#       camera.txt
#       gen_0.mp4 ... gen_N.mp4
#
# 推理完成后跑 reward：
#   bash rl_train/reward/batch_reward.sh \
#       --sample_dir <OUTPUT_DIR>/<dataset>/<sample_id> \
#       --output_dir /tmp/reward_out
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_CODE_DIR="$(cd "$SCRIPT_DIR/../../../" && pwd)"

# ── 解析 --config 参数 ────────────────────────────────────────────────────────
CONFIG_FILE="${RL_CODE_DIR}/config/train_gen3r.sh"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "[ERROR] Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "[run_infer.sh] Loading config from: $CONFIG_FILE"
source "$CONFIG_FILE"

# ── 推理专用默认值（覆盖训练配置的部分参数） ─────────────────────────────────
INFER_SAMPLING_STEPS="${INFER_SAMPLING_STEPS:-50}"     # 推理步数（比训练多）
INFER_ETA="${INFER_ETA:-0.0}"                          # 推理时用 ODE（eta=0）
INFER_NUM_ROLLOUTS="${INFER_NUM_ROLLOUTS:-8}"           # 每样本生成几条视频
INFER_MAX_SAMPLES="${INFER_MAX_SAMPLES:-5}"            # 批量模式最多推理几个样本
INFER_OUTPUT_DIR="${INFER_OUTPUT_DIR:-${OUTPUT_DIR}/infer_outputs}"

# ── 构建命令行参数 ────────────────────────────────────────────────────────────
INFER_ARGS=(
    --pretrained_model_path "${GEN3R_CKPT}"
    --vggt_path             "${VGGT_PATH}"
    --geo_adapter_path      "${GEO_ADAPTER_PATH}"
    --config_path           "${CONFIG_PATH}"
    --data_root             "${DATA_ROOT}"
    --datasets              "${DATASETS}"
    --frame_mode            "${FRAME_MODE}"
    --num_frames            "${NUM_FRAMES}"
    --frame_stride          "${FRAME_STRIDE}"
    --resolution            "${RESOLUTION}"
    --output_dir            "${INFER_OUTPUT_DIR}"
    --num_rollouts          "${INFER_NUM_ROLLOUTS}"
    --max_samples           "${INFER_MAX_SAMPLES}"
    --sampling_steps        "${INFER_SAMPLING_STEPS}"
    --eta                   "${INFER_ETA}"
    --shift                 "${SHIFT}"
    --cfg_infer             "${CFG_INFER}"
    --tokenizer_max_length  "${TOKENIZER_MAX_LENGTH}"
    --skip_done
)

[[ -n "${TRANSFORMER_PATH:-}" ]] && INFER_ARGS+=(--transformer_path "${TRANSFORMER_PATH}")
INFER_ARGS+=("${EXTRA_ARGS[@]}")

# ── 打印配置摘要 ──────────────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo "  Gen3R Inference Debug"
echo "======================================================"
echo "  Config:         ${CONFIG_FILE}"
echo "  Checkpoint:     ${GEN3R_CKPT}"
echo "  Data root:      ${DATA_ROOT}"
echo "  Output:         ${INFER_OUTPUT_DIR}"
echo "  Num rollouts:   ${INFER_NUM_ROLLOUTS}"
echo "  Sampling steps: ${INFER_SAMPLING_STEPS}"
echo "======================================================"
echo ""

# ── 启动推理 ──────────────────────────────────────────────────────────────────
cd "$SCRIPT_DIR"
python infer_only.py "${INFER_ARGS[@]}"
