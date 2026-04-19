#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# run_benchmark.sh — Benchmark 一键测评 bash 入口
#
# 用法：
#   # 全部评测
#   bash run_benchmark.sh --output_root /path/to/output --metrics all
#
#   # 按类别
#   bash run_benchmark.sh --metrics video_quality
#   bash run_benchmark.sh --metrics reward
#   bash run_benchmark.sh --metrics reconstruction
#
#   # 按子指标
#   bash run_benchmark.sh --metrics reward.camera_pose
#   bash run_benchmark.sh --metrics reconstruction.global
#
#   # 逗号分隔组合
#   bash run_benchmark.sh --metrics video_quality.psnr,reward.camera_pose,reconstruction.global
#
#   # 重建对齐方式（仅影响 reconstruction.*）
#   bash run_benchmark.sh --metrics reconstruction --align camera       # 单模式
#   bash run_benchmark.sh --metrics reconstruction --align first_frame  # 单模式
#   bash run_benchmark.sh --metrics reconstruction --align umeyama      # 单模式
#   bash run_benchmark.sh --metrics reconstruction --align icp          # 单模式，最精确
#   bash run_benchmark.sh --metrics reconstruction --align both_align   # camera+first_frame
#   bash run_benchmark.sh --metrics reconstruction --align all_align    # 全部四种模式
#
#   # 调整点云参数
#   bash run_benchmark.sh --metrics reconstruction --n_fps 50000 --conf_thresh 0.5
#
#   # 指定 GPU
#   bash run_benchmark.sh --gpu 1
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# 读取中心配置
CONFIG="${REPO_ROOT}/config/benchmark.sh"
if [ -f "$CONFIG" ]; then
    source "$CONFIG"
fi

# 确保变量有初始值（config 未定义时兜底）
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
METRICS="${METRICS:-all}"
GPU="${GPU:-0}"
DEVICE="${DEVICE:-}"
ALIGN="${ALIGN:-both_align}"
N_FPS="${N_FPS:-20000}"
CONF_THRESH="${CONF_THRESH:-0.0}"
VBENCH_CACHE="${VBENCH_CACHE:-}"
SKIP_INTERMEDIATES="${SKIP_INTERMEDIATES:-false}"
AGGREGATE_ONLY="${AGGREGATE_ONLY:-false}"

# 命令行参数覆盖 config
while [[ $# -gt 0 ]]; do
    case $1 in
        --output_root)    OUTPUT_ROOT="$2"; shift 2 ;;
        --metrics)        METRICS="$2"; shift 2 ;;
        --gpu)            GPU="$2"; shift 2 ;;
        --device)         DEVICE="$2"; shift 2 ;;
        --align)
            case "$2" in
                camera|first_frame|umeyama|icp|both_align|all_align)
                    ALIGN="$2"; shift 2 ;;
                *)
                    echo "错误: --align 须为 camera/first_frame/umeyama/icp/both_align/all_align"
                    exit 1 ;;
            esac ;;
        --n_fps)          N_FPS="$2"; shift 2 ;;
        --conf_thresh)    CONF_THRESH="$2"; shift 2 ;;
        --vbench_cache)   VBENCH_CACHE="$2"; shift 2 ;;
        --skip_intermediates)  SKIP_INTERMEDIATES=true; shift ;;
        --aggregate_only)      AGGREGATE_ONLY=true; shift ;;
        *)
            echo "未知参数: $1"; exit 1 ;;
    esac
done

if [ -z "$OUTPUT_ROOT" ] || [ "$OUTPUT_ROOT" = "/path/to/output" ]; then
    echo "错误: 请在 config/benchmark.sh 中设置 OUTPUT_ROOT，或通过 --output_root 指定"
    echo "用法: bash run_benchmark.sh [--output_root /path/to/output] [--metrics all] [--gpu 0]"
    exit 1
fi

# 构造 Python 命令
EXTRA_ARGS=""
if [ "$SKIP_INTERMEDIATES" = true ]; then
    EXTRA_ARGS="$EXTRA_ARGS --skip_intermediates"
fi
if [ "$AGGREGATE_ONLY" = true ]; then
    EXTRA_ARGS="$EXTRA_ARGS --aggregate_only"
fi

# device: 优先使用显式指定，否则从 GPU 号推导
if [ -z "$DEVICE" ]; then
    DEVICE="cuda:${GPU}"
fi

echo "════════════════════════════════════════════════"
echo "  Benchmark 评测系统"
echo "  output_root:  $OUTPUT_ROOT"
echo "  metrics:      $METRICS"
echo "  gpu:          $GPU"
echo "  device:       $DEVICE"
echo "  align:        $ALIGN  (仅 reconstruction)"
echo "  n_fps:        $N_FPS"
echo "  conf_thresh:  $CONF_THRESH"
echo "════════════════════════════════════════════════"

CMD="python -u ${SCRIPT_DIR}/run_benchmark.py \
    --output_root ${OUTPUT_ROOT} \
    --metrics ${METRICS} \
    --gpu ${GPU} \
    --device ${DEVICE} \
    --align ${ALIGN} \
    --n_fps ${N_FPS} \
    --conf_thresh ${CONF_THRESH}"

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
