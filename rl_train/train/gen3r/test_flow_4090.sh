#!/usr/bin/env bash
# =============================================================================
# 4090 (24GB) 小显存 dry_run 测试脚本
#
# 测试数据：re10k/000000_5aca87f95a9412c6（49帧, 560x560）
# 目标：用随机 reward 跑通完整 GRPO 训练流程（rollout → reward → advantage → PPO）
#
# 显存优化策略：
#   1. --dry_run                      跳过 reward pipeline，用随机 reward
#   2. --resolution 280               分辨率减半，显存降约 4x
#   3. --num_frames 9                 减少帧数
#   4. --gradient_checkpointing       时间换显存
#   5. --trainable_modules camera_adapter  只训练小子模块
#   6. --num_generations 2            最小 rollout 组（2条可做组内归一化）
#   7. --train_timestep_strategy front + --sde_fraction 0.4
#      前 40% SDE（训练），后 60% ODE（只生成），省显存+省计算
#
# 用法:
#   bash test_flow_4090.sh \
#       --pretrained_model_path /path/to/gen3r_ckpts \
#       --vggt_path /path/to/vggt.pth \
#       --geo_adapter_path /path/to/geo_adapter
#
# 如果 280 分辨率仍 OOM，尝试追加: --resolution 168 --num_frames 5
# =============================================================================
set -euo pipefail

# 默认使用空闲 GPU（0 号常被占满时可 export CUDA_VISIBLE_DEVICES=1）
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 默认使用仓库内 eval 自带的 Gen3R checkpoint（与 HF 布局一致：tokenizer/、transformer/ 等）
DEFAULT_CKPT="/home/users/puxin.yan-labs/RL_code/eval/infer/gen3r/Gen3R/checkpoints"

PRETRAINED_MODEL_PATH=""
VGGT_PATH=""
GEO_ADAPTER_PATH=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pretrained_model_path) PRETRAINED_MODEL_PATH="$2"; shift 2 ;;
        --vggt_path) VGGT_PATH="$2"; shift 2 ;;
        --geo_adapter_path) GEO_ADAPTER_PATH="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-$DEFAULT_CKPT}"
VGGT_PATH="${VGGT_PATH:-${PRETRAINED_MODEL_PATH}/vggt}"
GEO_ADAPTER_PATH="${GEO_ADAPTER_PATH:-${PRETRAINED_MODEL_PATH}/geo_adapter}"

if [[ ! -d "$PRETRAINED_MODEL_PATH" ]]; then
    echo "[ERROR] pretrained_model_path not found: $PRETRAINED_MODEL_PATH"
    exit 1
fi

DATA_ROOT="/horizon-bucket/robot_lab/users/puxin.yan-labs/Datasets_chuli1"
CONFIG_PATH="${SCRIPT_DIR}/Gen3R/gen3r/config/gen3r.yaml"
OUTPUT_DIR="${SCRIPT_DIR}/outputs/test_4090_dryrun"

echo ""
echo "======================================================"
echo "  4090 Dry-Run Test (50 sampling steps, hybrid SDE/ODE)"
echo "======================================================"
echo "  Checkpoint:  ${PRETRAINED_MODEL_PATH}"
echo "  Data root:   ${DATA_ROOT}"
echo "  Test sample: re10k/000000_5aca87f95a9412c6"
echo "  Output:      ${OUTPUT_DIR}"
echo "  Strategy:    front (SDE 40% + ODE 60%)"
echo "======================================================"
echo ""

cd "$SCRIPT_DIR"

python train_grpo.py \
    --pretrained_model_path "$PRETRAINED_MODEL_PATH" \
    --vggt_path "$VGGT_PATH" \
    --geo_adapter_path "$GEO_ADAPTER_PATH" \
    --config_path "$CONFIG_PATH" \
    --data_root "$DATA_ROOT" \
    --datasets "re10k" \
    --frame_mode "video" \
    --num_frames 17 \
    --frame_stride 2 \
    --resolution 560 \
    --dataloader_num_workers 2 \
    --train_batch_size 1 \
    --learning_rate 1e-5 \
    --weight_decay 1e-4 \
    --max_grad_norm 1.0 \
    --lr_scheduler "constant_with_warmup" \
    --lr_warmup_steps 0 \
    --gradient_accumulation_steps 2 \
    --max_train_steps 2 \
    --seed 42 \
    --sampler_seed 1223627 \
    --output_dir "$OUTPUT_DIR" \
    --checkpointing_steps 100 \
    --sampling_steps 50 \
    --eta 0.2 \
    --shift 2.0 \
    --cfg_infer 5.0 \
    --tokenizer_max_length 512 \
    --num_generations 2 \
    --train_timestep_strategy "front" \
    --sde_fraction 0.4 \
    --timestep_fraction 0.5 \
    --clip_range 1e-4 \
    --adv_clip_max 5.0 \
    --rewards "all" \
    --gradient_checkpointing \
    --dry_run \
    --trainable_modules control_adapter \
    "${EXTRA_ARGS[@]}"

echo ""
echo "[test_flow_4090] Done. Check output at: ${OUTPUT_DIR}"
