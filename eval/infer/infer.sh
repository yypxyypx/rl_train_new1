#!/bin/bash
# ============================================================
# infer.sh — 统一推理接口
# ============================================================
#
# 支持模型：gen3r、wan2.2
# 每条样本产出一个 gen_0.mp4（ODE 推理，无 rollout）
# 输出结构：<output>/<ckpt_tag>/<dataset>/<sample_id>/gen_0.mp4
#
# 用法示例：
#
#   # Gen3R 推理（指定数据目录）
#   bash infer.sh \
#       --model gen3r \
#       --checkpoint /path/to/gen3r_checkpoints \
#       --data /path/to/rl_data/gen3r \
#       --datasets dl3dv,scannet++ \
#       --output /path/to/output \
#       --ckpt_tag gen3r_baseline \
#       --num_gpus 4
#
#   # Wan2.2 推理（单样本调试）
#   bash infer.sh \
#       --model wan2.2 \
#       --checkpoint /path/to/Wan2.2-Fun-5B-Control-Camera \
#       --sample_dir /path/to/sample \
#       --output /tmp/debug
#
#   # 预构建 manifest
#   bash infer.sh \
#       --model gen3r \
#       --checkpoint /path/to/ckpt \
#       --manifest /path/to/manifest.jsonl \
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

# ── 变量初始默认值（config 未设置时兜底）─────────────────────
MODEL="${MODEL:-gen3r}"
CHECKPOINT="${CHECKPOINT:-}"
CKPT_TAG="${CKPT_TAG:-}"          # 输出目录中间层，默认用 MODEL 值
DATA="${DATA:-}"
DATASETS="${DATASETS:-}"
MANIFEST="${MANIFEST:-}"
SAMPLE_DIR="${SAMPLE_DIR:-}"
OUTPUT="${OUTPUT:-}"
NUM_GPUS="${NUM_GPUS:-1}"
MAX_PER_DATASET="${MAX_PER_DATASET:-0}"
SEED="${SEED:-42}"
SKIP_DONE="${SKIP_DONE:-}"
DEVICE_MODE="${DEVICE_MODE:-server}"
REQUIRE_GT="${REQUIRE_GT:-}"
MASTER_PORT="${MASTER_PORT:-29500}"

# ── 每条样本只生成 1 个视频（ODE 无 rollout）────────────────
# 以下参数根据模型有不同的默认值，在解析 --model 后设置

# ── 命令行参数覆盖 ────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)            MODEL="$2";            shift 2 ;;
        --checkpoint)       CHECKPOINT="$2";       shift 2 ;;
        --ckpt_tag)         CKPT_TAG="$2";         shift 2 ;;
        --data)             DATA="$2";             shift 2 ;;
        --datasets)         DATASETS="$2";         shift 2 ;;
        --manifest)         MANIFEST="$2";         shift 2 ;;
        --sample_dir)       SAMPLE_DIR="$2";       shift 2 ;;
        --output)           OUTPUT="$2";           shift 2 ;;
        --num_gpus)         NUM_GPUS="$2";         shift 2 ;;
        --max_per_dataset)  MAX_PER_DATASET="$2";  shift 2 ;;
        --seed)             SEED="$2";             shift 2 ;;
        --skip_done)        SKIP_DONE="--skip_done"; shift ;;
        --device_mode)      DEVICE_MODE="$2";      shift 2 ;;
        --require_gt)       REQUIRE_GT="--require_gt_video"; shift ;;
        --master_port)      MASTER_PORT="$2";      shift 2 ;;
        # 向下兼容：忽略旧版 SDE 参数
        --num_rollouts|--eta|--steps|--guidance|--shift|--num_frames|--target_size) shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# ── 必填检查 ──────────────────────────────────────────────────
if [[ -z "$CHECKPOINT" || -z "$OUTPUT" ]]; then
    echo "错误: --checkpoint 和 --output 是必填项"
    exit 1
fi

# ── CKPT_TAG 默认等于模型名 ──────────────────────────────────
if [[ -z "$CKPT_TAG" ]]; then
    CKPT_TAG="$MODEL"
fi

# 实际输出目录 = <OUTPUT>/<CKPT_TAG>
EFFECTIVE_OUTPUT="${OUTPUT}/${CKPT_TAG}"

# ── 推理脚本路径 ──────────────────────────────────────────────
INFER_SCRIPT="${SCRIPT_DIR}/${MODEL}/infer_${MODEL//./_}.py"
MANIFEST_SCRIPT="${SCRIPT_DIR}/${MODEL}/build_manifest.py"
if [[ ! -f "$INFER_SCRIPT" ]]; then
    echo "错误: 找不到推理脚本 ${INFER_SCRIPT}"
    exit 1
fi

# ── Build manifest（如果提供了 --data）────────────────────────
if [[ -n "$DATA" && -z "$MANIFEST" && -z "$SAMPLE_DIR" ]]; then
    DS_ARGS=""
    if [[ -n "$DATASETS" ]]; then
        IFS=',' read -ra DS_LIST <<< "$DATASETS"
        DS_ARGS="--datasets ${DS_LIST[*]}"
    fi
    MAX_ARG=""
    if [[ "${MAX_PER_DATASET}" -gt 0 ]]; then
        MAX_ARG="--max_per_dataset ${MAX_PER_DATASET}"
    fi
    MANIFEST="${EFFECTIVE_OUTPUT}/manifest.jsonl"
    mkdir -p "${EFFECTIVE_OUTPUT}"
    echo "Building manifest → ${MANIFEST}"
    ${PYTHON} -u "${MANIFEST_SCRIPT}" \
        --data_root "$DATA" \
        $DS_ARGS \
        $MAX_ARG \
        --output "$MANIFEST" \
        ${REQUIRE_GT}
    echo ""
fi

# ── 解析输入参数 ──────────────────────────────────────────────
INPUT_ARG=""
if [[ -n "$MANIFEST" ]]; then
    INPUT_ARG="--manifest ${MANIFEST}"
elif [[ -n "$SAMPLE_DIR" ]]; then
    INPUT_ARG="--sample_dir ${SAMPLE_DIR}"
else
    echo "错误: 需要 --data、--manifest 或 --sample_dir 之一"
    exit 1
fi

# ── 打印配置 ──────────────────────────────────────────────────
echo "════════════════════════════════════════════════════"
echo "  Model:      ${MODEL}"
echo "  Checkpoint: ${CHECKPOINT}"
echo "  Ckpt tag:   ${CKPT_TAG}"
echo "  GPUs:       ${NUM_GPUS}"
echo "  Output:     ${EFFECTIVE_OUTPUT}"
echo "════════════════════════════════════════════════════"

# ── 启动推理 ──────────────────────────────────────────────────
COMMON_ARGS=(
    ${INPUT_ARG}
    --checkpoint "${CHECKPOINT}"
    --output_root "${EFFECTIVE_OUTPUT}"
    --seed ${SEED}
    ${SKIP_DONE}
)

if [[ ${NUM_GPUS} -gt 1 ]]; then
    torchrun --nproc_per_node=${NUM_GPUS} \
        --master_port=${MASTER_PORT} \
        "${INFER_SCRIPT}" \
        "${COMMON_ARGS[@]}"
else
    ${PYTHON} -u "${INFER_SCRIPT}" \
        "${COMMON_ARGS[@]}"
fi

echo ""
echo "推理完成。输出: ${EFFECTIVE_OUTPUT}"
