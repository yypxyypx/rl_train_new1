#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# run_benchmark.sh — Benchmark 一键测评 bash 入口
#
# 用法：
#   # 全部评测
#   bash run_benchmark.sh --output_root /path/to/output --metrics all
#
#   # 按类别
#   bash run_benchmark.sh --output_root /path/to/output --metrics video_quality
#   bash run_benchmark.sh --output_root /path/to/output --metrics reward
#   bash run_benchmark.sh --output_root /path/to/output --metrics reconstruction
#
#   # 按子指标
#   bash run_benchmark.sh --output_root /path/to/output --metrics video_quality.psnr
#   bash run_benchmark.sh --output_root /path/to/output --metrics reward.camera_pose
#   bash run_benchmark.sh --output_root /path/to/output --metrics reconstruction.global
#
#   # 按细粒度接口
#   bash run_benchmark.sh --output_root /path/to/output --metrics reward.depth_reprojection.object
#
#   # 逗号分隔组合
#   bash run_benchmark.sh --output_root /path/to/output \
#       --metrics video_quality.psnr,reward.camera_pose,reconstruction.global
#
#   # 重建对齐方式
#   bash run_benchmark.sh --output_root /path/to/output \
#       --metrics reconstruction --align first_frame
#
#   # 指定 GPU
#   bash run_benchmark.sh --output_root /path/to/output --metrics all --gpu 0
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 从 config.sh 读取默认值
CONFIG_FILE="${SCRIPT_DIR}/config.sh"
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
fi

# 确保变量有初始值（config.sh 未定义时兜底）
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
METRICS="${METRICS:-all}"
GPU="${GPU:-0}"
ALIGN="${ALIGN:-both_align}"
VBENCH_CACHE="${VBENCH_CACHE:-}"
SKIP_INTERMEDIATES="${SKIP_INTERMEDIATES:-false}"
AGGREGATE_ONLY="${AGGREGATE_ONLY:-false}"
EXTRA_ARGS=""

# 命令行参数覆盖 config.sh
while [[ $# -gt 0 ]]; do
    case $1 in
        --output_root)
            OUTPUT_ROOT="$2"; shift 2 ;;
        --metrics)
            METRICS="$2"; shift 2 ;;
        --gpu)
            GPU="$2"; shift 2 ;;
        --align)
            ALIGN="$2"; shift 2 ;;
        --vbench_cache)
            VBENCH_CACHE="$2"; shift 2 ;;
        --skip_intermediates)
            SKIP_INTERMEDIATES=true; shift ;;
        --aggregate_only)
            AGGREGATE_ONLY=true; shift ;;
        *)
            echo "未知参数: $1"; exit 1 ;;
    esac
done

if [ "$SKIP_INTERMEDIATES" = true ]; then
    EXTRA_ARGS="$EXTRA_ARGS --skip_intermediates"
fi
if [ "$AGGREGATE_ONLY" = true ]; then
    EXTRA_ARGS="$EXTRA_ARGS --aggregate_only"
fi

if [ -z "$OUTPUT_ROOT" ] || [ "$OUTPUT_ROOT" = "/path/to/output" ]; then
    echo "错误: 请在 config.sh 中设置 OUTPUT_ROOT，或通过 --output_root 指定"
    echo "用法: bash run_benchmark.sh [--output_root /path/to/output] [--metrics all] [--gpu 0]"
    exit 1
fi

echo "════════════════════════════════════════════════"
echo "  Benchmark 评测系统"
echo "  output_root: $OUTPUT_ROOT"
echo "  metrics:     $METRICS"
echo "  gpu:         $GPU"
echo "  align:       $ALIGN"
echo "════════════════════════════════════════════════"

CMD="python -u ${SCRIPT_DIR}/run_benchmark.py \
    --output_root ${OUTPUT_ROOT} \
    --metrics ${METRICS} \
    --gpu ${GPU} \
    --align ${ALIGN}"

if [ -n "$VBENCH_CACHE" ]; then
    CMD="$CMD --vbench_cache ${VBENCH_CACHE}"
fi

if [ -n "$EXTRA_ARGS" ]; then
    CMD="$CMD $EXTRA_ARGS"
fi

export CUDA_VISIBLE_DEVICES=$GPU

echo "执行: $CMD"
eval $CMD

echo "════════════════════════════════════════════════"
echo "  Benchmark 完成"
echo "════════════════════════════════════════════════"
