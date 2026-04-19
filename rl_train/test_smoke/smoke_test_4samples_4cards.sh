#!/bin/bash
# smoke_test_4samples_4cards.sh
# ──────────────────────────────────────────────────────────────────────────────
# 端到端冒烟测试（真实 reward，不 dry_run）：
#   - 训练：cards 0-3（4 ranks DDP，每 rank 1 个样本）
#   - Reward：cards 4-7（DA3 / Qwen+SAM3 / DINOv2 / VideoAlign 各一卡，
#                       通过 RANK % len 让 4 个 rank 错开同一组里的不同卡）
#   - 4 个样本 × num_generations=2 = 8 个 rollouts
#   - max_train_steps=1，sampling_steps=10
#   - 校验整条链：dataloader → rollout → 5-reward → GRPO update → checkpoint

set -euo pipefail

SMOKE_ROOT="/home/users/puxin.yan-labs/RL_code/rl_train/test_smoke"
PRETRAINED_MODEL_PATH="/home/users/puxin.yan-labs/RL/gen3r/Gen3R/checkpoints"
CONFIG_PATH="${PRETRAINED_MODEL_PATH%/checkpoints}/gen3r/config/gen3r.yaml"
DATA_ROOT="${SMOKE_ROOT}/data"
T5_EMBED_DIR="${SMOKE_ROOT}/t5_cache"
OUTPUT_DIR="${SMOKE_ROOT}/run_$(date +%Y%m%d_%H%M%S)"
RL_MODEL_ROOT="/home/users/puxin.yan-labs/RL/model"

SCRIPT_DIR="/home/users/puxin.yan-labs/RL_code/rl_train/train/gen3r"

mkdir -p "$OUTPUT_DIR"

# ── 环境 ──────────────────────────────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${SCRIPT_DIR}/Gen3R:${PYTHONPATH:-}"
export RL_MODEL_ROOT

# 4 ranks for training on cards 0-3；让训练不要碰到 4-7（reward 用）
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# Reward GPU 分配（每组 4 张卡，dispatcher 内部按 RANK%4 错开）
REWARD_GPU_ASSIGNMENT='{"da3":[4,4,4,4],"qwen_sam3":[5,5,5,5],"dinov2_extract":[6,6,6,6],"videoalign":[7,7,7,7]}'

PY=/home/users/puxin.yan-labs/miniconda3/envs/rl_train/bin/python
TORCHRUN=/home/users/puxin.yan-labs/miniconda3/envs/rl_train/bin/torchrun

echo "================================================================"
echo "Smoke Test: 4 samples × 4 cards (DDP) + 4-card reward farm"
echo "================================================================"
echo "MODEL:  $PRETRAINED_MODEL_PATH"
echo "DATA:   $DATA_ROOT"
echo "T5:     $T5_EMBED_DIR"
echo "OUTPUT: $OUTPUT_DIR"
echo "Cards:  train=0-3   reward=4-7"
echo ""

"$TORCHRUN" \
    --standalone \
    --nproc_per_node=4 \
    "$SCRIPT_DIR/train_grpo_v2.py" \
    \
    --pretrained_model_path "$PRETRAINED_MODEL_PATH" \
    --config_path "$CONFIG_PATH" \
    --t5_embed_dir "$T5_EMBED_DIR" \
    \
    --data_root "$DATA_ROOT" \
    --datasets dl3dv \
    --num_frames 17 \
    --frame_stride 2 \
    --resolution 560 \
    --frame_mode video \
    --dataloader_num_workers 1 \
    \
    --max_train_steps 1 \
    --train_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 1 \
    --checkpointing_steps 1 \
    --max_grad_norm 1.0 \
    \
    --num_generations 2 \
    --sampling_steps 10 \
    --eta 0.2 \
    --shift 2.0 \
    --cfg_infer 5.0 \
    --train_timestep_strategy front \
    --sde_fraction 0.4 \
    --clip_range 1e-4 \
    --adv_clip_max 5.0 \
    --kl_coeff 0.0 \
    \
    --rewards "camera_traj,feature_sim,video_quality" \
    --reward_gpu_assignment "$REWARD_GPU_ASSIGNMENT" \
    --reward_model_root "$RL_MODEL_ROOT" \
    --keep_intermediates \
    --skip_done \
    \
    --gradient_checkpointing \
    \
    --output_dir "$OUTPUT_DIR" \
    --seed 42 \
    2>&1 | tee "$OUTPUT_DIR/train.log"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "SMOKE TEST PASSED"
    echo "Output dir: $OUTPUT_DIR"
    ls -la "$OUTPUT_DIR/"
    echo "--- reward log ---"
    cat "$OUTPUT_DIR/reward_log.jsonl" 2>/dev/null || echo "(none)"
    echo "--- train log tail ---"
    tail -30 "$OUTPUT_DIR/train.log"
else
    echo "SMOKE TEST FAILED (exit=$EXIT_CODE)"
    echo "Log tail:"
    tail -80 "$OUTPUT_DIR/train.log"
fi
echo "================================================================"
exit $EXIT_CODE
