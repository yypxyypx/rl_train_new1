#!/bin/bash
# ============================================================
# infer.sh — Unified inference interface
# ============================================================
#
# Usage:
#   # Gen3R inference with manifest
#   bash infer.sh \
#       --model gen3r \
#       --checkpoint /path/to/gen3r_checkpoints \
#       --data /path/to/processed_data \
#       --datasets re10k,dl3dv \
#       --output /path/to/output \
#       --num_gpus 4 \
#       --num_rollouts 8
#
#   # Gen3R inference with pre-built manifest
#   bash infer.sh \
#       --model gen3r \
#       --checkpoint /path/to/gen3r_checkpoints \
#       --manifest /path/to/manifest.jsonl \
#       --output /path/to/output \
#       --num_gpus 4
#
#   # Single sample
#   bash infer.sh \
#       --model gen3r \
#       --checkpoint /path/to/gen3r_checkpoints \
#       --sample_dir /path/to/processed/re10k/scene_001 \
#       --output /path/to/output
#
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-python3}"

# ── 从中心配置文件加载默认值 ──────────────────────────────────
CONFIG_FILE="${REPO_ROOT}/config/infer.sh"
if [[ -f "$CONFIG_FILE" ]]; then
    source "$CONFIG_FILE"
fi

# ── 确保变量有初始值（config 未定义时兜底）───────────────────
MODEL="${MODEL:-gen3r}"
CHECKPOINT="${CHECKPOINT:-}"
DATA="${DATA:-}"
DATASETS="${DATASETS:-}"
MANIFEST="${MANIFEST:-}"
SAMPLE_DIR="${SAMPLE_DIR:-}"
OUTPUT="${OUTPUT:-}"
NUM_GPUS="${NUM_GPUS:-1}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-8}"
NUM_FRAMES="${NUM_FRAMES:-49}"
TARGET_SIZE="${TARGET_SIZE:-560}"
ETA="${ETA:-0.3}"
STEPS="${STEPS:-50}"
GUIDANCE="${GUIDANCE:-5.0}"
SHIFT="${SHIFT:-2.0}"
SEED="${SEED:-42}"
SKIP_DONE="${SKIP_DONE:-}"
DEVICE_MODE="${DEVICE_MODE:-server}"
REQUIRE_GT="${REQUIRE_GT:-}"
MASTER_PORT="${MASTER_PORT:-29500}"

# ── 命令行参数覆盖 config 值 ─────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)         MODEL="$2"; shift 2 ;;
        --checkpoint)    CHECKPOINT="$2"; shift 2 ;;
        --data)          DATA="$2"; shift 2 ;;
        --datasets)      DATASETS="$2"; shift 2 ;;
        --manifest)      MANIFEST="$2"; shift 2 ;;
        --sample_dir)    SAMPLE_DIR="$2"; shift 2 ;;
        --output)        OUTPUT="$2"; shift 2 ;;
        --num_gpus)      NUM_GPUS="$2"; shift 2 ;;
        --num_rollouts)  NUM_ROLLOUTS="$2"; shift 2 ;;
        --num_frames)    NUM_FRAMES="$2"; shift 2 ;;
        --target_size)   TARGET_SIZE="$2"; shift 2 ;;
        --eta)           ETA="$2"; shift 2 ;;
        --steps)         STEPS="$2"; shift 2 ;;
        --guidance)      GUIDANCE="$2"; shift 2 ;;
        --shift)         SHIFT="$2"; shift 2 ;;
        --seed)          SEED="$2"; shift 2 ;;
        --skip_done)     SKIP_DONE="--skip_done"; shift ;;
        --device_mode)   DEVICE_MODE="$2"; shift 2 ;;
        --require_gt)    REQUIRE_GT="--require_gt_video"; shift ;;
        --master_port)   MASTER_PORT="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

if [[ -z "$CHECKPOINT" || -z "$OUTPUT" ]]; then
    echo "Error: --checkpoint and --output are required."
    exit 1
fi

# ── Build manifest if --data is provided ─────────────────────
if [[ -n "$DATA" && -z "$MANIFEST" && -z "$SAMPLE_DIR" ]]; then
    DS_ARGS=""
    if [[ -n "$DATASETS" ]]; then
        IFS=',' read -ra DS_LIST <<< "$DATASETS"
        DS_ARGS="--datasets ${DS_LIST[*]}"
    fi
    MANIFEST="${OUTPUT}/manifest.jsonl"
    echo "Building manifest from ${DATA} ..."
    ${PYTHON} -u "${SCRIPT_DIR}/${MODEL}/build_manifest.py" \
        --data_root "$DATA" \
        $DS_ARGS \
        --output "$MANIFEST" \
        ${REQUIRE_GT}
    echo ""
fi

# ── Resolve input argument ───────────────────────────────────
INPUT_ARG=""
if [[ -n "$MANIFEST" ]]; then
    INPUT_ARG="--manifest ${MANIFEST}"
elif [[ -n "$SAMPLE_DIR" ]]; then
    INPUT_ARG="--sample_dir ${SAMPLE_DIR}"
else
    echo "Error: need --data, --manifest, or --sample_dir"
    exit 1
fi

# ── Dispatch to model-specific inference script ──────────────
INFER_SCRIPT="${SCRIPT_DIR}/${MODEL}/infer_${MODEL}.py"
if [[ ! -f "$INFER_SCRIPT" ]]; then
    echo "Error: inference script not found: ${INFER_SCRIPT}"
    exit 1
fi

echo "════════════════════════════════════════════════════"
echo "  Model:      ${MODEL}"
echo "  Checkpoint: ${CHECKPOINT}"
echo "  GPUs:       ${NUM_GPUS}"
echo "  Rollouts:   ${NUM_ROLLOUTS}"
echo "  Output:     ${OUTPUT}"
echo "════════════════════════════════════════════════════"

if [[ ${NUM_GPUS} -gt 1 ]]; then
    torchrun --nproc_per_node=${NUM_GPUS} \
        --master_port=${MASTER_PORT} \
        "${INFER_SCRIPT}" \
        ${INPUT_ARG} \
        --checkpoint "${CHECKPOINT}" \
        --output_root "${OUTPUT}" \
        --num_rollouts ${NUM_ROLLOUTS} \
        --num_frames ${NUM_FRAMES} \
        --target_size ${TARGET_SIZE} \
        --eta ${ETA} \
        --num_inference_steps ${STEPS} \
        --guidance_scale ${GUIDANCE} \
        --shift ${SHIFT} \
        --base_seed ${SEED} \
        --device_mode ${DEVICE_MODE} \
        ${SKIP_DONE}
else
    ${PYTHON} -u "${INFER_SCRIPT}" \
        ${INPUT_ARG} \
        --checkpoint "${CHECKPOINT}" \
        --output_root "${OUTPUT}" \
        --num_rollouts ${NUM_ROLLOUTS} \
        --num_frames ${NUM_FRAMES} \
        --target_size ${TARGET_SIZE} \
        --eta ${ETA} \
        --num_inference_steps ${STEPS} \
        --guidance_scale ${GUIDANCE} \
        --shift ${SHIFT} \
        --base_seed ${SEED} \
        --device_mode ${DEVICE_MODE} \
        ${SKIP_DONE}
fi

echo ""
echo "Inference complete. Output: ${OUTPUT}"
