#!/bin/bash
# =============================================================================
# run_t5_precompute_inplace.sh — 把 T5 embedding 直接预存到每条样本目录内
#
# 输出（每个 sample 各一份）：
#   {sample_dir}/prompt_embed.pt   ← 该样本 caption 的 T5 embedding
#   {sample_dir}/neg_embed.pt      ← 全局共享的 negative prompt embedding
#
# 之后训练时 model_loader.load_t5_embeds 会优先读这两个文件。
#
# 用法：
#   bash rl_train_new/rl_train/train/gen3r/run_t5_precompute_inplace.sh
#
# 选项（环境变量覆盖）：
#   GPU=0|1|...                单卡 GPU id
#   BATCH_SIZE=8               T5 真 batch 大小
#   DATASETS="dl3dv,scannet++" 处理哪些 dataset 子目录
#   OVERWRITE=1                覆盖已有 .pt
#   LIMIT=N                    只处理前 N 条 (debug)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=/mnt/afs/visitor16/rl_train_new

GPU=${GPU:-0}
BATCH_SIZE=${BATCH_SIZE:-8}
DATASETS=${DATASETS:-"dl3dv,scannet++"}
OVERWRITE_FLAG=""
if [ "${OVERWRITE:-0}" = "1" ]; then
    OVERWRITE_FLAG="--overwrite"
fi
LIMIT_ARG=""
if [ -n "${LIMIT:-}" ]; then
    LIMIT_ARG="--limit ${LIMIT}"
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

LOG_DIR="$ROOT/_test_runs/t5_precompute_inplace"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/t5_inplace_${TS}.log"

echo "[run_t5] CUDA_VISIBLE_DEVICES=$GPU  batch=$BATCH_SIZE  datasets=$DATASETS"
echo "[run_t5] log → $LOG_FILE"

/root/miniconda3/envs/rl_train/bin/python -u "$SCRIPT_DIR/t5_precompute_inplace.py" \
    --pretrained_model_path "$ROOT/model/Gen3R/checkpoints" \
    --config_path "$SCRIPT_DIR/Gen3R/gen3r/config/gen3r.yaml" \
    --data_root "$ROOT/hf_datasets/rl_data/gen3r" \
    --datasets "$DATASETS" \
    --batch_size "$BATCH_SIZE" \
    --device "cuda:0" \
    $OVERWRITE_FLAG $LIMIT_ARG 2>&1 | tee "$LOG_FILE"

echo "[run_t5] done. log: $LOG_FILE"
