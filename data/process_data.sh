#!/bin/bash
# ============================================================
# process_data.sh — One-stop data processing interface
# ============================================================
#
# Usage examples:
#
#   # Process re10k for Gen3R (560x560, 49 frames)
#   bash process_data.sh \
#       --dataset re10k \
#       --dataset_path /path/to/raw_data \
#       --model gen3r \
#       --output /path/to/processed \
#       --skip_done
#
#   # Process dl3dv with custom resolution
#   bash process_data.sh \
#       --dataset dl3dv \
#       --dataset_path /path/to/raw_data \
#       --model gen3r \
#       --target_size 560 \
#       --num_frames 49 \
#       --output /path/to/processed
#
#   # Process multiple datasets in one go
#   bash process_data.sh \
#       --dataset re10k,dl3dv \
#       --dataset_path /path/to/raw_data \
#       --model gen3r \
#       --output /path/to/processed \
#       --skip_done
#
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-python3}"

# ── 从中心配置文件加载默认值 ──────────────────────────────────
CONFIG_FILE="${REPO_ROOT}/config/data_process.sh"
if [[ -f "$CONFIG_FILE" ]]; then
    source "$CONFIG_FILE"
fi

# ── 确保变量有初始值（config 未定义时兜底）───────────────────
DATASET="${DATASET:-}"
DATASET_PATH="${DATASET_PATH:-}"
MODEL="${MODEL:-gen3r}"
OUTPUT="${OUTPUT:-}"
TARGET_SIZE="${TARGET_SIZE:-}"
NUM_FRAMES="${NUM_FRAMES:-}"
SAMPLE_MODE="${SAMPLE_MODE:-fixed}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SKIP_DONE="${SKIP_DONE:-}"
INCLUDE_DEPTH="${INCLUDE_DEPTH:-}"
QUIET="${QUIET:-}"

# ── 命令行参数覆盖 config 值 ─────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)       DATASET="$2"; shift 2 ;;
        --dataset_path)  DATASET_PATH="$2"; shift 2 ;;
        --model)         MODEL="$2"; shift 2 ;;
        --output)        OUTPUT="$2"; shift 2 ;;
        --target_size)   TARGET_SIZE="$2"; shift 2 ;;
        --num_frames)    NUM_FRAMES="$2"; shift 2 ;;
        --sample_mode)   SAMPLE_MODE="$2"; shift 2 ;;
        --max_samples)   MAX_SAMPLES="$2"; shift 2 ;;
        --skip_done)     SKIP_DONE="--skip_done"; shift ;;
        --include_depth) INCLUDE_DEPTH="--include_depth"; shift ;;
        --quiet)         QUIET="--quiet"; shift ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# ── Validate required arguments ──────────────────────────────
if [[ -z "$DATASET" || -z "$DATASET_PATH" || -z "$OUTPUT" ]]; then
    echo "Error: --dataset, --dataset_path, and --output are required."
    echo ""
    echo "Usage: bash process_data.sh \\"
    echo "    --dataset re10k \\"
    echo "    --dataset_path /path/to/raw_data \\"
    echo "    --model gen3r \\"
    echo "    --output /path/to/processed"
    exit 1
fi

# ── Build optional arguments ─────────────────────────────────
EXTRA_ARGS=""
[[ -n "$TARGET_SIZE" ]]    && EXTRA_ARGS="$EXTRA_ARGS --target_size $TARGET_SIZE"
[[ -n "$NUM_FRAMES" ]]     && EXTRA_ARGS="$EXTRA_ARGS --num_frames $NUM_FRAMES"
[[ -n "$SKIP_DONE" ]]      && EXTRA_ARGS="$EXTRA_ARGS $SKIP_DONE"
[[ -n "$INCLUDE_DEPTH" ]]  && EXTRA_ARGS="$EXTRA_ARGS $INCLUDE_DEPTH"
[[ -n "$QUIET" ]]          && EXTRA_ARGS="$EXTRA_ARGS $QUIET"
[[ "$MAX_SAMPLES" != "0" ]] && EXTRA_ARGS="$EXTRA_ARGS --max_samples $MAX_SAMPLES"

# ── Process each dataset (supports comma-separated list) ─────
IFS=',' read -ra DATASETS <<< "$DATASET"

for DS in "${DATASETS[@]}"; do
    DS=$(echo "$DS" | xargs)  # trim whitespace
    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  Processing: ${DS}  →  model=${MODEL}"
    echo "════════════════════════════════════════════════════"

    ${PYTHON} -u "${SCRIPT_DIR}/unified_data_process.py" \
        --dataset "$DS" \
        --dataset_path "$DATASET_PATH" \
        --model "$MODEL" \
        --output "$OUTPUT" \
        --sample_mode "$SAMPLE_MODE" \
        $EXTRA_ARGS
done

echo ""
echo "All datasets processed. Output: ${OUTPUT}"
