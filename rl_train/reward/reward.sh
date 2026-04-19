#!/usr/bin/env bash
# reward.sh — Reward 计算入口脚本（单视频模式）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONFIG_FILE="${REPO_ROOT}/config/reward.sh"
if [[ -f "$CONFIG_FILE" ]]; then
    source "$CONFIG_FILE"
fi

VIDEO_PATH="${VIDEO_PATH:-}"
GT_CAMERA_TXT="${GT_CAMERA_TXT:-}"
WORK_DIR="${WORK_DIR:-}"
REWARDS="${REWARDS:-all}"
GPU="${GPU:-0}"
PROMPT="${PROMPT:-}"
METADATA_JSON="${METADATA_JSON:-}"
NO_SKIP="${NO_SKIP:-}"
CONF_THRESHOLD="${CONF_THRESHOLD:-0.0}"
GEO_COMPARE_MODE="${GEO_COMPARE_MODE:-first_frame}"
FEATURE_COMPARE_MODE="${FEATURE_COMPARE_MODE:-first_frame}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --video_path)            VIDEO_PATH="$2";            shift 2 ;;
        --gt_camera_txt)         GT_CAMERA_TXT="$2";         shift 2 ;;
        --work_dir)              WORK_DIR="$2";              shift 2 ;;
        --rewards)               REWARDS="$2";               shift 2 ;;
        --gpu)                   GPU="$2";                   shift 2 ;;
        --prompt)                PROMPT="$2";                shift 2 ;;
        --metadata_json)         METADATA_JSON="$2";         shift 2 ;;
        --no_skip)               NO_SKIP="--no_skip";        shift ;;
        --conf_threshold)        CONF_THRESHOLD="$2";        shift 2 ;;
        --geo_compare_mode)      GEO_COMPARE_MODE="$2";      shift 2 ;;
        --feature_compare_mode)  FEATURE_COMPARE_MODE="$2";  shift 2 ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$VIDEO_PATH" ]] || [[ -z "$GT_CAMERA_TXT" ]] || [[ -z "$WORK_DIR" ]]; then
    echo "Usage: bash reward.sh --video_path <path> --gt_camera_txt <path> --work_dir <path> [--rewards all] [--gpu 0]"
    exit 1
fi

if [[ ! -f "$VIDEO_PATH" ]]; then
    echo "Error: video not found: $VIDEO_PATH"
    exit 1
fi

if [[ ! -f "$GT_CAMERA_TXT" ]]; then
    echo "Error: camera.txt not found: $GT_CAMERA_TXT"
    exit 1
fi

CONDA_ENV="rl_da3"
PYTHON_BIN=""
for candidate in \
    "/opt/conda/envs/${CONDA_ENV}/bin/python" \
    "${HOME}/miniconda3/envs/${CONDA_ENV}/bin/python" \
    "${HOME}/anaconda3/envs/${CONDA_ENV}/bin/python"; do
    if [[ -f "$candidate" ]]; then
        PYTHON_BIN="$candidate"
        break
    fi
done
if [[ -z "$PYTHON_BIN" ]]; then
    echo "Error: cannot find python for conda env '${CONDA_ENV}'"
    exit 1
fi

CMD=(
    "$PYTHON_BIN" -u "${SCRIPT_DIR}/reward_pipeline.py"
    --video_path "$VIDEO_PATH"
    --gt_camera_txt "$GT_CAMERA_TXT"
    --work_dir "$WORK_DIR"
    --rewards "$REWARDS"
    --gpu "$GPU"
    --conf_threshold "$CONF_THRESHOLD"
    --geo_compare_mode "$GEO_COMPARE_MODE"
    --feature_compare_mode "$FEATURE_COMPARE_MODE"
)

if [[ -n "$PROMPT" ]]; then
    CMD+=(--prompt "$PROMPT")
fi

if [[ -n "$METADATA_JSON" ]]; then
    CMD+=(--metadata_json "$METADATA_JSON")
fi

if [[ -n "$NO_SKIP" ]]; then
    CMD+=($NO_SKIP)
fi

echo "============================================================"
echo "[reward.sh] Reward Pipeline"
echo "[reward.sh] Video:    $VIDEO_PATH"
echo "[reward.sh] GT:       $GT_CAMERA_TXT"
echo "[reward.sh] Work:     $WORK_DIR"
echo "[reward.sh] Rewards:  $REWARDS"
echo "[reward.sh] GPU:      $GPU"
echo "[reward.sh] ConfThr:  $CONF_THRESHOLD"
echo "[reward.sh] GeoMode:  $GEO_COMPARE_MODE"
echo "[reward.sh] FeatMode: $FEATURE_COMPARE_MODE"
echo "============================================================"

mkdir -p "$WORK_DIR"

"${CMD[@]}"

echo ""
echo "[reward.sh] Done. Results in: $WORK_DIR/reward.json"
