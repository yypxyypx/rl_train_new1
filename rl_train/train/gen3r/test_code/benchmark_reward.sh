#!/bin/bash
# benchmark_reward.sh — 4×4090 reward 速度 benchmark 启动脚本
#
# 前提：已运行 Gen3R 推理生成视频，放在 INFER_DIR 下
# 目录结构：
#   INFER_DIR/
#     <dataset>/<sample_id>/
#       gen_0.mp4 ... gen_7.mp4
#       camera.txt
#       metadata.json
#
# 用法：
#   bash benchmark_reward.sh [INFER_DIR] [OUTPUT_DIR]

set -euo pipefail

# ── 可配置参数 ─────────────────────────────────────────────────────────────────
INFER_DIR="${1:-/path/to/generated_videos}"
OUTPUT_DIR="${2:-$(dirname "$0")/results/benchmark_$(date +%Y%m%d_%H%M%S)}"
ROLLOUT_IDX=0          # 每个样本用第 0 条 rollout

# 选 4 个样本（改成实际存在的路径）
SAMPLES=(
    "re10k/sample_0001"
    "re10k/sample_0002"
    "re10k/sample_0003"
    "re10k/sample_0004"
)

# ── 环境检查 ──────────────────────────────────────────────────────────────────
echo "========================================"
echo "Gen3R GRPO Reward Benchmark"
echo "========================================"
echo "INFER_DIR:  $INFER_DIR"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "ROLLOUT:    gen_${ROLLOUT_IDX}.mp4"
echo "SAMPLES:    ${SAMPLES[*]}"
echo ""

# GPU 检查
N_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Available GPUs: $N_GPUS"
if [ "$N_GPUS" -lt 4 ]; then
    echo "WARNING: Less than 4 GPUs detected. Parallel benchmark may share GPUs."
fi
echo ""

# ── 运行 benchmark ────────────────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="python"

echo "[Benchmark] Starting..."
$PYTHON_BIN "$SCRIPT_DIR/benchmark_reward.py" \
    --infer_dir "$INFER_DIR" \
    --samples "${SAMPLES[@]}" \
    --rollout_idx "$ROLLOUT_IDX" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "[Benchmark] Done. Results: $OUTPUT_DIR/benchmark_report.json"

# ── 打印报告摘要 ──────────────────────────────────────────────────────────────
if command -v python3 &> /dev/null; then
    python3 -c "
import json, sys
with open('$OUTPUT_DIR/benchmark_report.json') as f:
    r = json.load(f)
print('\\n--- BENCHMARK REPORT ---')
print(f'Parallel:   {r[\"parallel\"][\"total_s\"]:.1f}s')
if 'sequential_single_gpu' in r:
    print(f'Sequential: {r[\"sequential_single_gpu\"][\"total_s\"]:.1f}s')
    print(f'Speedup:    {r.get(\"speedup\", 0):.2f}x')
"
fi
