#!/bin/bash
# =============================================================================
# run_smoke_h100_synthetic.sh — 单卡 H100 端到端 smoke
#
# 目的：在搬到 16 卡 H100 之前，先在本机单卡用 synthetic 模式（mock 所有
# Phase 1 / Phase 2 输出）验证：
#
#   1. 新参数 (--rollouts_per_rank / --rolling_ckpt_every / --permanent_ckpt_every)
#      正确解析、正确生效
#   2. KL=0 时 ref_transformer 不被加载（model_loader 路径）+ KL forward 整段跳过
#   3. P3 grpo_update 改造（auto GAS, no_sync ctx for non-DDP, 单 step 一次 step）
#      不报错，loss / grad_norm / kl 正常
#   4. Checkpoint 滚动逻辑：
#        每 2 步 rolling，每 4 步 permanent，跑 6 步：
#        预期产物：
#          rolling-6/                ← 最新滚动（rolling-2/4 已被覆盖删除）
#          permanent-4/              ← 永久节点
#          permanent-final/          ← 训练结束最终节点
#
# 单卡 H100 80GB：synthetic mode 不需要 dataset / reward worker，约 3-5min 跑完。
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=/mnt/afs/visitor16/rl_train_new
TS=$(date +%Y%m%d_%H%M%S)
OUT=$ROOT/_test_runs/smoke_h100_synthetic/output_${TS}
mkdir -p "$OUT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled

cat <<EOF
======================================================================
  Gen3R GRPO Single-H100 Synthetic Smoke
======================================================================
  Output:  $OUT
  GPU:     $CUDA_VISIBLE_DEVICES
  Steps:   6  (rolling every 2, permanent every 4)
  G=8, R=8, ranks/group=1 (single-GPU degenerate path)
  KL_COEFF=0  (验证 ref skip)
  num_frames=49 res=560 (匹配生产配置)
======================================================================
EOF

/root/miniconda3/envs/rl_train/bin/python "$SCRIPT_DIR/train_grpo_v2.py" \
    \
    `# ── 模型 ──` \
    --pretrained_model_path "$ROOT/model/Gen3R/checkpoints" \
    --config_path "$SCRIPT_DIR/Gen3R/gen3r/config/gen3r.yaml" \
    \
    `# ── 数据：synthetic 模式不读真实数据，但仍需占位参数 ──` \
    --data_root /tmp/dummy \
    --datasets dl3dv \
    --num_frames 49 \
    --frame_stride 1 \
    --resolution 560 \
    \
    `# ── synthetic 模式开关 ──` \
    --p3_synthetic \
    \
    `# ── GRPO：单卡退化路径 R=G=8（ranks_per_group=1） ──` \
    --num_generations 8 \
    --rollouts_per_rank 8 \
    --sampling_steps 50 \
    --eta 0.5 \
    --shift 2.0 \
    --cfg_rollout 1.0 \
    --cfg_train 1.0 \
    --train_timestep_strategy front \
    --sde_fraction 1.0 \
    --train_steps_count 4 \
    --clip_range 1e-4 \
    --adv_clip_max 5.0 \
    \
    `# ── KL=0：验证 ref_transformer 不被加载、KL forward 整段跳过 ──` \
    --kl_coeff 0.0 \
    --train_microbatch_size 1 \
    \
    `# ── 训练：6 步 ──` \
    --max_train_steps 6 \
    --train_batch_size 1 \
    --learning_rate 1e-5 \
    --weight_decay 1e-4 \
    --max_grad_norm 1.0 \
    --lr_scheduler constant \
    --lr_warmup_steps 0 \
    --seed 42 \
    \
    `# ── 新 ckpt 策略：rolling 覆盖最新 1 份, permanent 长期保留 ──` \
    --rolling_ckpt_every 2 \
    --permanent_ckpt_every 4 \
    --keep_last_n_permanent 0 \
    --checkpointing_steps 999999 \
    \
    `# ── 显存：H100 80GB 全开梯度 ckpt ──` \
    --gradient_checkpoint_layers -1 \
    \
    --output_dir "$OUT" \
    2>&1 | tee "$OUT/smoke.log"

echo ""
echo "======================================================================"
echo "Smoke finished. Validating ckpt artifacts under $OUT ..."
echo "======================================================================"
ls -1 "$OUT" | grep -E "^(rolling|permanent|checkpoint)" || echo "  (no ckpts found!)"

EXPECTED=("rolling-6" "permanent-4" "permanent-final")
ALL_OK=1
for d in "${EXPECTED[@]}"; do
    if [ -d "$OUT/$d" ]; then
        size=$(du -sh "$OUT/$d" | cut -f1)
        echo "  [OK] $d ($size)"
    else
        echo "  [MISSING] $d"
        ALL_OK=0
    fi
done

# 验证 rolling-2 / rolling-4 应已被删除
for d in "rolling-2" "rolling-4"; do
    if [ -d "$OUT/$d" ]; then
        echo "  [WARN] $d should have been deleted by rolling logic"
        ALL_OK=0
    fi
done

if [ "$ALL_OK" = "1" ]; then
    echo ""
    echo "[SMOKE PASS] All ckpt artifacts as expected."
else
    echo ""
    echo "[SMOKE FAIL] Some ckpt artifacts missing or unexpected."
    exit 1
fi
