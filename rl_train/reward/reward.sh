#!/usr/bin/env bash
# reward.sh — Reward 计算入口脚本
#
# 用法：
#   bash reward.sh \
#       --video_path /path/to/gen.mp4 \
#       --gt_camera_txt /path/to/camera.txt \
#       --work_dir /path/to/work/ \
#       --rewards all \
#       --gpu 0
#
# --rewards 参数：
#   all                     计算全部 reward
#   camera_traj             仅相机轨迹（自动只跑 DA3）
#   geo_semantic,camera_traj 指定组合（自动推断所需 steps）
#
# 可选参数：
#   --prompt "..."          VideoAlign 用的视频描述
#   --metadata_json /path   从 metadata.json 自动提取 prompt
#   --no_skip               强制重跑所有 steps（默认跳过已有输出）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ── 从中心配置文件加载默认值 ──────────────────────────────────
CONFIG_FILE="${REPO_ROOT}/config/reward.sh"
if [[ -f "$CONFIG_FILE" ]]; then
    source "$CONFIG_FILE"
fi

# ── 确保变量有初始值（config 未定义时兜底）───────────────────
VIDEO_PATH="${VIDEO_PATH:-}"
GT_CAMERA_TXT="${GT_CAMERA_TXT:-}"
WORK_DIR="${WORK_DIR:-}"
REWARDS="${REWARDS:-all}"
GPU="${GPU:-0}"
PROMPT="${PROMPT:-}"
METADATA_JSON="${METADATA_JSON:-}"
NO_SKIP="${NO_SKIP:-}"

# ── 命令行参数覆盖 config 值 ─────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --video_path)      VIDEO_PATH="$2";      shift 2 ;;
        --gt_camera_txt)   GT_CAMERA_TXT="$2";   shift 2 ;;
        --work_dir)        WORK_DIR="$2";         shift 2 ;;
        --rewards)         REWARDS="$2";          shift 2 ;;
        --gpu)             GPU="$2";              shift 2 ;;
        --prompt)          PROMPT="$2";           shift 2 ;;
        --metadata_json)   METADATA_JSON="$2";    shift 2 ;;
        --no_skip)         NO_SKIP="--no_skip";   shift ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# ── 参数校验 ────────────────────────────────────────────────
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

# ── 构建命令 ────────────────────────────────────────────────
CMD=(
    python -u "${SCRIPT_DIR}/reward_pipeline.py"
    --video_path "$VIDEO_PATH"
    --gt_camera_txt "$GT_CAMERA_TXT"
    --work_dir "$WORK_DIR"
    --rewards "$REWARDS"
    --gpu "$GPU"
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

# ── 执行 ────────────────────────────────────────────────────
echo "============================================================"
echo "[reward.sh] Reward Pipeline"
echo "[reward.sh] Video:   $VIDEO_PATH"
echo "[reward.sh] GT:      $GT_CAMERA_TXT"
echo "[reward.sh] Work:    $WORK_DIR"
echo "[reward.sh] Rewards: $REWARDS"
echo "[reward.sh] GPU:     $GPU"
echo "============================================================"

mkdir -p "$WORK_DIR"

"${CMD[@]}"

echo ""
echo "[reward.sh] Done. Results in: $WORK_DIR/reward.json"
