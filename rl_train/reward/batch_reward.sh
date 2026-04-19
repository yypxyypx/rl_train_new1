#!/usr/bin/env bash
# batch_reward.sh — 批量 Reward 计算入口脚本
#
# 对一个推理输出目录（包含 infer_info.json + gen_0.mp4 ... gen_N.mp4）
# 的全部 rollout 跑完整 pipeline（DA3 / SAM3 / DINOv2 / VideoAlign + reward 计算）。
#
# 用法：
#   bash batch_reward.sh \
#       --sample_dir /path/to/dl3dv/0a6c01ac32127687 \
#       --output_dir /path/to/reward_output \
#       --rewards all \
#       --gpu 0
#
# --rewards 参数（同 reward.sh）：
#   all                     计算全部 reward
#   camera_traj             仅相机轨迹（自动只跑 DA3）
#   geo_semantic,camera_traj 指定组合
#
# 可选参数：
#   --prompt "..."          VideoAlign 用的视频描述（所有 rollout 共用）
#   --no_skip               强制重跑所有 steps（默认跳过已有输出）
#   --conf_threshold 0.0    DA3 深度置信度阈值
#   --geo_compare_mode first_frame   几何一致性比较模式
#   --feature_compare_mode first_frame  DINOv2 特征相似度比较模式

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ── 从中心配置文件加载默认值 ──────────────────────────────────
CONFIG_FILE="${REPO_ROOT}/config/reward.sh"
if [[ -f "$CONFIG_FILE" ]]; then
    source "$CONFIG_FILE"
fi

# ── 确保变量有初始值 ─────────────────────────────────────────
SAMPLE_DIR="${SAMPLE_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
REWARDS="${REWARDS:-all}"
GPU="${GPU:-0}"
PROMPT="${PROMPT:-}"
NO_SKIP="${NO_SKIP:-}"
CONF_THRESHOLD="${CONF_THRESHOLD:-0.0}"
GEO_COMPARE_MODE="${GEO_COMPARE_MODE:-first_frame}"
FEATURE_COMPARE_MODE="${FEATURE_COMPARE_MODE:-first_frame}"

# ── 命令行参数覆盖 config 值 ─────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sample_dir)            SAMPLE_DIR="$2";            shift 2 ;;
        --output_dir)            OUTPUT_DIR="$2";            shift 2 ;;
        --rewards)               REWARDS="$2";               shift 2 ;;
        --gpu)                   GPU="$2";                   shift 2 ;;
        --prompt)                PROMPT="$2";                shift 2 ;;
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

# ── 参数校验 ────────────────────────────────────────────────
if [[ -z "$SAMPLE_DIR" ]] || [[ -z "$OUTPUT_DIR" ]]; then
    echo "Usage: bash batch_reward.sh --sample_dir <path> --output_dir <path> [--rewards all] [--gpu 0]"
    exit 1
fi

if [[ ! -f "$SAMPLE_DIR/infer_info.json" ]]; then
    echo "Error: infer_info.json not found in: $SAMPLE_DIR"
    exit 1
fi

# ── 定位 Python（与 reward_pipeline.py 的 _env_python 逻辑一致）──
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

# ── 构建命令 ────────────────────────────────────────────────
CMD=(
    "$PYTHON_BIN" -u "${SCRIPT_DIR}/reward_pipeline.py"
    --mode batch
    --sample_dir "$SAMPLE_DIR"
    --output_dir "$OUTPUT_DIR"
    --rewards "$REWARDS"
    --gpu "$GPU"
    --conf_threshold "$CONF_THRESHOLD"
    --geo_compare_mode "$GEO_COMPARE_MODE"
    --feature_compare_mode "$FEATURE_COMPARE_MODE"
)

if [[ -n "$PROMPT" ]]; then
    CMD+=(--prompt "$PROMPT")
fi

if [[ -n "$NO_SKIP" ]]; then
    CMD+=($NO_SKIP)
fi

# ── 执行 ────────────────────────────────────────────────────
echo "============================================================"
echo "[batch_reward.sh] Batch Reward Pipeline"
echo "[batch_reward.sh] Sample:   $SAMPLE_DIR"
echo "[batch_reward.sh] Output:   $OUTPUT_DIR"
echo "[batch_reward.sh] Rewards:  $REWARDS"
echo "[batch_reward.sh] GPU:      $GPU"
echo "[batch_reward.sh] ConfThr:  $CONF_THRESHOLD"
echo "[batch_reward.sh] GeoMode:  $GEO_COMPARE_MODE"
echo "[batch_reward.sh] FeatMode: $FEATURE_COMPARE_MODE"
echo "============================================================"

mkdir -p "$OUTPUT_DIR"

"${CMD[@]}"

echo ""
echo "[batch_reward.sh] Done."
