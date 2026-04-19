#!/bin/bash
# smoke_test_4cards.sh
# ──────────────────────────────────────────────────────────────────────────────
# 4 张空闲卡（1, 2, 3, 5）端到端 smoke test：
#   - Cards 1, 2  : 训练 (DDP, world_size=2)
#   - Cards 3, 5  : Reward 子进程（DA3 + DINOv2 各占一张，rank 错开）
#
#   - train_batch_size=2 / rank, num_generations=2  ⇒  4 samples × 2 gens = 8 rollouts/step
#   - max_train_steps=1, sampling_steps=10
#   - rewards = camera_traj + feature_sim（只跑 DA3 + DINOv2 两组，避免 SAM3/VideoAlign 的额外环境）
#
# 校验链路：dataloader → rollout → reward(并行) → advantage → GRPO loss → backward → checkpoint

set -euo pipefail

SMOKE_ROOT="/home/users/puxin.yan-labs/RL_code/rl_train/test_smoke"
PRETRAINED_MODEL_PATH="/home/users/puxin.yan-labs/RL/gen3r/Gen3R/checkpoints"
CONFIG_PATH="/home/users/puxin.yan-labs/RL/gen3r/Gen3R/gen3r/config/gen3r.yaml"
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
export WANDB_MODE=disabled

# 训练用卡：本地编号 0,1 实际 -> 物理 1,2；reward 子进程独立指定（不受此变量约束，
# subprocess 用自己的 CUDA_VISIBLE_DEVICES 覆盖）。
export CUDA_VISIBLE_DEVICES="1,2"

# Reward GPU 分配：DA3 -> {3, 5}，DINOv2 -> {5, 3}，让 rank 0/1 错开
#   rank0: da3=3, dinov2=5
#   rank1: da3=5, dinov2=3
# 物理 GPU id 对 reward 子进程来说就是 nvidia-smi 看到的真实 id（subprocess 用
# 自己的 CUDA_VISIBLE_DEVICES，不继承父进程的限制）。
REWARD_GPU_ASSIGNMENT='{"da3":[3,5]}'

TORCHRUN=/home/users/puxin.yan-labs/miniconda3/envs/rl_train/bin/torchrun

echo "================================================================"
echo "Smoke Test: 4 cards total (train=1,2  reward=3,5)"
echo "================================================================"
echo "MODEL:  $PRETRAINED_MODEL_PATH"
echo "DATA:   $DATA_ROOT"
echo "T5:     $T5_EMBED_DIR"
echo "OUTPUT: $OUTPUT_DIR"
echo "Train ranks: 2 (DDP, train_batch_size=2 → 4 samples/step)"
echo "Rollouts:    2 generations × 4 samples = 8 rollouts/step"
echo "Reward:      camera_traj only (DA3)"
echo ""

"$TORCHRUN" \
    --standalone \
    --nproc_per_node=2 \
    "$SCRIPT_DIR/train_grpo.py" \
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
    --dataloader_num_workers 0 \
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
    --rewards "camera_traj" \
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
    echo "--- reward_log.jsonl ---"
    cat "$OUTPUT_DIR/reward_log.jsonl" 2>/dev/null | head -20 || echo "(none)"
    echo "--- training_log.jsonl ---"
    cat "$OUTPUT_DIR/training_log.jsonl" 2>/dev/null | head -10 || echo "(none)"
else
    echo "SMOKE TEST FAILED (exit=$EXIT_CODE)"
    echo "Log tail:"
    tail -120 "$OUTPUT_DIR/train.log"
fi
echo "================================================================"
exit $EXIT_CODE
