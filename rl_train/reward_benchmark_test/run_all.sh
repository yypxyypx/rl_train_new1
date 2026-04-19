#!/usr/bin/env bash
# ==============================================================
# run_all.sh — 一键运行全部四个阶段
#
# 用法:
#   bash run_all.sh                          # 运行 Phase 1-4（默认 GPU 0,1,2,3）
#   bash run_all.sh --phase 4                # 仅运行 Phase 4（分析+可视化）
#   bash run_all.sh --phase 2,3,4            # 运行 Phase 2,3,4
#   bash run_all.sh --gpu_ids 1,2,3,5        # 指定 GPU
#
# Phase 2 使用 run_reward_multimode.py（三种 geo 帧对比模式）。
# Phase 3 只跑选择性指标（视频质量 + AUC + 点云，4种对齐）。
# Phase 4 读 reward_multimode.json，输出三份 Pearson 热力图 + 对比图。
#
# 所有阶段支持断点续跑（已完成的文件会自动跳过）。
# ==============================================================

OUTPUT_ROOT="/horizon-bucket/robot_lab/users/puxin.yan-labs/test_output1"
OUT_DIR="$(dirname "$0")/results"
PHASE="all"
GPU_IDS="0,1,2,3"
GPU="0"           # Phase 3 benchmark GPU
N_WORKERS="8"     # Phase 3 CPU 并行进程数

# ── 解析命令行 ─────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_root) OUTPUT_ROOT="$2"; shift 2 ;;
    --out_dir)     OUT_DIR="$2";     shift 2 ;;
    --phase)       PHASE="$2";       shift 2 ;;
    --gpu_ids)     GPU_IDS="$2";     shift 2 ;;
    --gpu)         GPU="$2";         shift 2 ;;
    --n_workers)   N_WORKERS="$2";   shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# ── 使用 rl_da3 环境运行（包含 cv2, torch, scipy, matplotlib） ──
PYTHON="/home/users/puxin.yan-labs/miniconda3/envs/rl_da3/bin/python"

THIS_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo "OUTPUT_ROOT: $OUTPUT_ROOT"
echo "OUT_DIR:     $OUT_DIR"
echo "PHASE:       $PHASE"
echo "GPU_IDS:     $GPU_IDS  (Phase 1 & 2 multi-GPU)"
echo "GPU:         $GPU      (Phase 3 benchmark)"
echo "N_WORKERS:   $N_WORKERS  (Phase 3 CPU 并行进程数)"
echo "============================================"

$PYTHON -u "$THIS_DIR/run_correlation_analysis.py" \
    --output_root "$OUTPUT_ROOT" \
    --out_dir     "$OUT_DIR" \
    --phase       "$PHASE" \
    --gpu_ids     "$GPU_IDS" \
    --gpu         "$GPU" \
    --align       all_align \
    --n_workers   "$N_WORKERS"
