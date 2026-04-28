#!/usr/bin/env bash
# =============================================================================
# Wan2.2 框架端到端推理验证
#
# 用我们自己写的 wan2_2/infer_only.py 对两条样本各产出 4 条 rollout，
# 验证 model_loader / wan22_encode / grpo_core.run_sample_step 整条链路。
#
# 输出：./outputs/<dataset>/<sample_id>/{infer_info.json,camera.txt,gen_*.mp4}
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WAN22_DIR="$(cd "$SCRIPT_DIR/../wan2_2" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/outputs}"
mkdir -p "$OUTPUT_ROOT"

SAMPLES=(
  /mnt/afs/visitor16/RL_new/datasets_wan/dl3dv/1K/0a1b7c20a92c43c6
  /mnt/afs/visitor16/RL_new/datasets_wan/dl3dv/1K/0a78c25f77c1ba1d
)

NUM_ROLLOUTS="${NUM_ROLLOUTS:-4}"
NUM_FRAMES="${NUM_FRAMES:-49}"
SAMPLING_STEPS="${SAMPLING_STEPS:-50}"
SHIFT="${SHIFT:-5.0}"
ETA="${ETA:-0.0}"
CFG_INFER="${CFG_INFER:-6.0}"
RESOLUTION_H="${RESOLUTION_H:-704}"
RESOLUTION_W="${RESOLUTION_W:-1280}"
SEED_BASE="${SEED_BASE:-42}"
GPU="${GPU:-0}"

echo ""
echo "=========================================================="
echo "  Wan2.2 framework verification (infer_only.py)"
echo "=========================================================="
echo "  GPU:            $GPU"
echo "  Num rollouts:   $NUM_ROLLOUTS / sample"
echo "  Sampling steps: $SAMPLING_STEPS"
echo "  Resolution:     ${RESOLUTION_W}x${RESOLUTION_H} x ${NUM_FRAMES} frames"
echo "  Shift / ETA / CFG: $SHIFT / $ETA / $CFG_INFER"
echo "  Seed base:      $SEED_BASE  (rollout k uses seed_base + k*1000)"
echo "  Output dir:     $OUTPUT_ROOT"
echo "  Wan2_2 code:    $WAN22_DIR"
echo "=========================================================="
echo ""

cd "$WAN22_DIR"
export CUDA_VISIBLE_DEVICES="$GPU"
# 让 stdout/stderr 行缓冲，配合 tee 才能实时看到 [model_loader] / Sampling 进度，
# 避免误以为加载卡死（实际上 22GB 模型从 AFS 加载需 10–20 min）。
export PYTHONUNBUFFERED=1

# 关键：一次 python 调用处理所有 sample，模型只加载一次。
# 之前每个 sample 都重新 spawn python 会重复 load 22GB → 多用 15 min/sample。
echo ""
echo ">>> Processing ${#SAMPLES[@]} samples in a single python invocation"
echo ">>>   (loading 22 GB of weights from AFS once, please wait)"
python -u infer_only.py \
  --sample_dir "${SAMPLES[@]}" \
  --output_dir "$OUTPUT_ROOT" \
  --num_rollouts "$NUM_ROLLOUTS" \
  --num_frames "$NUM_FRAMES" \
  --frame_stride 1 \
  --resolution_h "$RESOLUTION_H" \
  --resolution_w "$RESOLUTION_W" \
  --sampling_steps "$SAMPLING_STEPS" \
  --shift "$SHIFT" \
  --eta "$ETA" \
  --cfg_infer "$CFG_INFER" \
  --seed_base "$SEED_BASE" \
  --no_skip_done 2>&1 | tee "$OUTPUT_ROOT/verify.log"

echo ""
echo "=========================================================="
echo "  Done. Inspect results:"
echo "    ls $OUTPUT_ROOT/dl3dv/*/gen_*.mp4"
echo "    cat $OUTPUT_ROOT/verify.log"
echo "=========================================================="
