#!/bin/bash
# =============================================================================
# run_train_h100_16gpu.sh — Gen3R GRPO 16×H100 正式训练入口
#
# 配置：
#   • 默认 config 文件：train_config_h100.env  (与本脚本同目录)
#   • 自定义 config：    CONFIG_ENV=/path/to/my_config.env  bash run_train_h100_16gpu.sh
#   • 单参数临时覆盖：   KL_COEFF=0.1  bash run_train_h100_16gpu.sh
#
# 优先级（高 → 低）：
#   命令行 env > CONFIG_ENV 内赋值 > 本脚本内置默认
#
# 单机 16 卡：
#   bash run_train_h100_16gpu.sh
#
# 2 节点 × 8 卡：
#   节点 0：NNODES=2 NODE_RANK=0 MASTER_ADDR=<ip> bash run_train_h100_16gpu.sh
#   节点 1：NNODES=2 NODE_RANK=1 MASTER_ADDR=<ip> bash run_train_h100_16gpu.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Source config 文件 ──────────────────────────────────────────────────────
CONFIG_ENV="${CONFIG_ENV:-$SCRIPT_DIR/train_config_h100.env}"
if [ -f "$CONFIG_ENV" ]; then
    echo "[run_train] sourcing config: $CONFIG_ENV"
    # 关闭 nounset 防止 config 里没赋值的变量触发 unbound 错误
    set +u
    # shellcheck source=/dev/null
    . "$CONFIG_ENV"
    set -u
else
    echo "[run_train] WARN: config not found ($CONFIG_ENV), using script defaults" >&2
fi

# ── 内置默认值（config 没设 / 没 source 时兜底）────────────────────────────
ROOT="${ROOT:-/mnt/afs/visitor16/rl_train_new}"
PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-$ROOT/model/Gen3R/checkpoints}"
CONFIG_PATH="${CONFIG_PATH:-$SCRIPT_DIR/Gen3R/gen3r/config/gen3r.yaml}"
DATA_ROOT="${DATA_ROOT:-$ROOT/hf_datasets/rl_data/gen3r}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-$DATA_ROOT/merged_train_manifest.jsonl}"
DATASETS="${DATASETS:-dl3dv,scannet++}"
OUTPUT_DIR_BASE="${OUTPUT_DIR_BASE:-$ROOT/_runs/grpo_h100_16gpu}"
RESUME_FROM="${RESUME_FROM:-}"

# OUTPUT_DIR 解析：
#   • RESUME_FROM = ""        → 新建 OUTPUT_DIR_BASE_<时间戳>
#   • RESUME_FROM = "auto"    → 找 OUTPUT_DIR_BASE_* 中最新的那个（mtime 最新）作为
#                                 OUTPUT_DIR，并把 --resume_from 也设为 "auto" 让
#                                 train_grpo_v2.py 自己在该目录里找 latest ckpt
#   • RESUME_FROM = 绝对路径   → OUTPUT_DIR = 该路径的父目录
if [ -z "${OUTPUT_DIR:-}" ]; then
    case "$RESUME_FROM" in
        "")
            TS=$(date +%Y%m%d_%H%M%S)
            OUTPUT_DIR="${OUTPUT_DIR_BASE}_${TS}"
            ;;
        auto)
            # 找最新的 OUTPUT_DIR_BASE_* 目录
            LATEST_RUN=$(ls -1dt "${OUTPUT_DIR_BASE}"_* 2>/dev/null | head -n1 || true)
            if [ -n "$LATEST_RUN" ] && [ -d "$LATEST_RUN" ]; then
                OUTPUT_DIR="$LATEST_RUN"
                echo "[run_train] RESUME_FROM=auto → 复用最近运行目录: $OUTPUT_DIR"
            else
                TS=$(date +%Y%m%d_%H%M%S)
                OUTPUT_DIR="${OUTPUT_DIR_BASE}_${TS}"
                echo "[run_train] RESUME_FROM=auto 但找不到旧 run，降级为新建: $OUTPUT_DIR"
                RESUME_FROM=""
            fi
            ;;
        *)
            OUTPUT_DIR="$(dirname "$RESUME_FROM")"
            echo "[run_train] RESUME_FROM=$RESUME_FROM → OUTPUT_DIR=$OUTPUT_DIR"
            ;;
    esac
fi

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi --list-gpus | wc -l)}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"

# ── 多机预检：catch 常见配置错误，避免到 NCCL 阶段才报错 ─────────────────
if [ "$NNODES" -gt 1 ]; then
    echo "[run_train] multi-node mode: NNODES=$NNODES NODE_RANK=$NODE_RANK"
    if [ "$MASTER_ADDR" = "localhost" ] || [ "$MASTER_ADDR" = "127.0.0.1" ]; then
        echo "[run_train] FATAL: 多机训练但 MASTER_ADDR=$MASTER_ADDR" >&2
        echo "             必须设为节点 0 的真实 IP（节点 1 能 ping 通的那个）" >&2
        exit 1
    fi
    if ! [[ "$NODE_RANK" =~ ^[0-9]+$ ]] || [ "$NODE_RANK" -ge "$NNODES" ]; then
        echo "[run_train] FATAL: NODE_RANK=$NODE_RANK 必须 ∈ [0, $NNODES)" >&2
        exit 1
    fi
    GPUS_DETECTED=$(nvidia-smi --list-gpus | wc -l)
    if [ "$NPROC_PER_NODE" -ne "$GPUS_DETECTED" ]; then
        echo "[run_train] WARN: NPROC_PER_NODE=$NPROC_PER_NODE 与本机检测到的 GPU 数 $GPUS_DETECTED 不一致。" >&2
        echo "             如果是有意为之（如调试时只用部分卡）请忽略。" >&2
    fi
    # 节点 0 上简单测一下 MASTER_PORT 是否被占用
    if [ "$NODE_RANK" = "0" ] && command -v ss >/dev/null 2>&1; then
        if ss -tln | awk '{print $4}' | grep -q ":$MASTER_PORT\$"; then
            echo "[run_train] WARN: 节点 0 上 MASTER_PORT=$MASTER_PORT 已被占用，可能是上次训练没清干净" >&2
        fi
    fi
    # 节点 1 简单测下能否 reach 节点 0
    if [ "$NODE_RANK" != "0" ] && command -v nc >/dev/null 2>&1; then
        if ! timeout 3 nc -zv "$MASTER_ADDR" "$MASTER_PORT" >/dev/null 2>&1; then
            echo "[run_train] NOTE: 当前 nc 探测 ${MASTER_ADDR}:${MASTER_PORT} 不通；" >&2
            echo "             如果节点 0 还没启动属正常（torchrun 会等），" >&2
            echo "             否则请检查防火墙 / IP / 端口。" >&2
        fi
    fi
fi

NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
ROLLOUTS_PER_RANK="${ROLLOUTS_PER_RANK:-4}"
SAMPLING_STEPS="${SAMPLING_STEPS:-50}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
TRAIN_MICROBATCH_SIZE="${TRAIN_MICROBATCH_SIZE:-1}"

KL_COEFF="${KL_COEFF:-0.01}"

ROLLING_CKPT_EVERY="${ROLLING_CKPT_EVERY:-4}"
PERMANENT_CKPT_EVERY="${PERMANENT_CKPT_EVERY:-50}"
KEEP_LAST_N_PERMANENT="${KEEP_LAST_N_PERMANENT:-0}"

MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-200}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
LR_SCHEDULER="${LR_SCHEDULER:-constant_with_warmup}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-5}"
SEED="${SEED:-42}"

NUM_FRAMES="${NUM_FRAMES:-49}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
RESOLUTION="${RESOLUTION:-560}"
FRAME_MODE="${FRAME_MODE:-video}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"

ETA="${ETA:-0.5}"
SHIFT="${SHIFT:-2.0}"
CFG_ROLLOUT="${CFG_ROLLOUT:-1.0}"
CFG_TRAIN="${CFG_TRAIN:-1.0}"
TRAIN_TIMESTEP_STRATEGY="${TRAIN_TIMESTEP_STRATEGY:-front}"
SDE_FRACTION="${SDE_FRACTION:-1.0}"
TRAIN_STEPS_COUNT="${TRAIN_STEPS_COUNT:-30}"
CLIP_RANGE="${CLIP_RANGE:-1e-4}"
ADV_CLIP_MAX="${ADV_CLIP_MAX:-5.0}"
VAE_DECODE_MICRO_BATCH="${VAE_DECODE_MICRO_BATCH:-4}"

GRADIENT_CHECKPOINT_LAYERS="${GRADIENT_CHECKPOINT_LAYERS:-22}"

REWARD_DISPATCH_MODE="${REWARD_DISPATCH_MODE:-workers}"
REWARDS="${REWARDS:-geo_global,feature_sim,camera_traj,video_quality}"
# 多卡默认 0 = VideoAlign 一次性真 batch（worker 独占 GPU，N=R=4 装得下）；
# 单卡 smoke 才需要 1（参见 run_h100_real_smoke.sh）
VA_MICRO_BATCH="${VA_MICRO_BATCH:-0}"

# 环境变量（若 config 没 export，这里补默认）
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

mkdir -p "$OUTPUT_DIR"
# 保留本次运行用到的 config 副本，方便复现
if [ -f "$CONFIG_ENV" ]; then
    cp "$CONFIG_ENV" "$OUTPUT_DIR/train_config_used.env"
fi

# ── 启动前预检：world_size 必须能被 ranks_per_group 整除（否则 build_sub_groups 会 raise）
WORLD_SIZE=$((NPROC_PER_NODE * NNODES))
if [ "$((NUM_GENERATIONS % ROLLOUTS_PER_RANK))" -ne 0 ]; then
    echo "[run_train] FATAL: NUM_GENERATIONS=$NUM_GENERATIONS 必须能被 ROLLOUTS_PER_RANK=$ROLLOUTS_PER_RANK 整除" >&2
    exit 1
fi
RANKS_PER_GROUP=$((NUM_GENERATIONS / ROLLOUTS_PER_RANK))
if [ "$((WORLD_SIZE % RANKS_PER_GROUP))" -ne 0 ]; then
    echo "[run_train] FATAL: world_size=$WORLD_SIZE 必须能被 ranks_per_group=$RANKS_PER_GROUP 整除" >&2
    echo "             (= NUM_GENERATIONS/ROLLOUTS_PER_RANK = $NUM_GENERATIONS/$ROLLOUTS_PER_RANK)" >&2
    exit 1
fi
N_GROUPS_TOTAL=$((WORLD_SIZE / RANKS_PER_GROUP))

# manifest 文件预检（避免到 P1 才发现路径错）
if [ -n "$TRAIN_MANIFEST" ] && [ ! -f "$TRAIN_MANIFEST" ]; then
    echo "[run_train] FATAL: TRAIN_MANIFEST not found: $TRAIN_MANIFEST" >&2
    exit 1
fi
# resume 路径预检
if [ -n "$RESUME_FROM" ] && [ "$RESUME_FROM" != "auto" ] && [ ! -d "$RESUME_FROM" ]; then
    echo "[run_train] FATAL: RESUME_FROM is not a directory: $RESUME_FROM" >&2
    exit 1
fi

# 推导每 step 视频数（P1 rollout）
VIDEOS_PER_STEP=$((ROLLOUTS_PER_RANK * WORLD_SIZE * TRAIN_BATCH_SIZE))
PROMPTS_PER_STEP=$((N_GROUPS_TOTAL * TRAIN_BATCH_SIZE))
SDE_STEPS_DERIVED=$(awk -v s="$SAMPLING_STEPS" -v f="$SDE_FRACTION" 'BEGIN{printf "%d", int(s*f)}')

cat <<EOF
========================================================================
 Gen3R GRPO Training — 16×H100   (SDE=$SDE_STEPS_DERIVED, CFG=$CFG_ROLLOUT)
========================================================================
  CONFIG_ENV:   $CONFIG_ENV
  MODEL:        $PRETRAINED_MODEL_PATH
  DATA:         $DATA_ROOT  [$DATASETS]
  MANIFEST:     ${TRAIN_MANIFEST:-<empty: 走目录扫描>}
  OUTPUT:       $OUTPUT_DIR
  Nodes:        $NNODES  NODE_RANK: $NODE_RANK  NPROC/node: $NPROC_PER_NODE
  Master:       ${MASTER_ADDR}:${MASTER_PORT}
  Total GPUs:   $WORLD_SIZE
  GRPO:         G=$NUM_GENERATIONS  R=$ROLLOUTS_PER_RANK  BS=$TRAIN_BATCH_SIZE
                ranks/group = $RANKS_PER_GROUP
                prompts/step = $PROMPTS_PER_STEP   (= n_groups_total × BS)
                effective rollouts/step = $VIDEOS_PER_STEP   (= R × world_size × BS)
  Sampling:     T=$SAMPLING_STEPS  shift=$SHIFT  eta=$ETA
                sde_fraction=$SDE_FRACTION → SDE=$SDE_STEPS_DERIVED + ODE=$((SAMPLING_STEPS - SDE_STEPS_DERIVED))
                train_steps_count=$TRAIN_STEPS_COUNT  (P3 实算梯度的 timestep 数)
  CFG:          rollout=$CFG_ROLLOUT  train=$CFG_TRAIN   $([ "$CFG_ROLLOUT" = "1.0" ] && echo "(disabled — single forward, ~2× faster)" || echo "(enabled — double forward)")
  KL_COEFF:     $KL_COEFF $([ "$KL_COEFF" = "0" ] || [ "$KL_COEFF" = "0.0" ] && echo "(disabled — ref_transformer not loaded)" || echo "(active)")
  Reward:       $REWARDS  (dispatch=$REWARD_DISPATCH_MODE  va_micro_batch=$VA_MICRO_BATCH)
  Checkpoint:   rolling every $ROLLING_CKPT_EVERY steps, permanent every $PERMANENT_CKPT_EVERY steps
  Resume:       ${RESUME_FROM:-<none>  (从头训练)}
========================================================================
EOF

torchrun \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    "$SCRIPT_DIR/train_grpo_v2.py" \
    \
    `# ── 模型 ──` \
    --pretrained_model_path "$PRETRAINED_MODEL_PATH" \
    --config_path "$CONFIG_PATH" \
    \
    `# ── 数据 ──` \
    --data_root "$DATA_ROOT" \
    --train_manifest "$TRAIN_MANIFEST" \
    --datasets "$DATASETS" \
    --num_frames "$NUM_FRAMES" \
    --frame_stride "$FRAME_STRIDE" \
    --resolution "$RESOLUTION" \
    --frame_mode "$FRAME_MODE" \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    \
    `# ── 训练 ──` \
    --max_train_steps "$MAX_TRAIN_STEPS" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --weight_decay "$WEIGHT_DECAY" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --lr_scheduler "$LR_SCHEDULER" \
    --lr_warmup_steps "$LR_WARMUP_STEPS" \
    --seed "$SEED" \
    \
    `# ── Checkpoint / Resume ──` \
    --rolling_ckpt_every "$ROLLING_CKPT_EVERY" \
    --permanent_ckpt_every "$PERMANENT_CKPT_EVERY" \
    --keep_last_n_permanent "$KEEP_LAST_N_PERMANENT" \
    --checkpointing_steps 999999 \
    --resume_from "$RESUME_FROM" \
    \
    `# ── GRPO ──` \
    --num_generations "$NUM_GENERATIONS" \
    --rollouts_per_rank "$ROLLOUTS_PER_RANK" \
    --sampling_steps "$SAMPLING_STEPS" \
    --eta "$ETA" \
    --shift "$SHIFT" \
    --cfg_rollout "$CFG_ROLLOUT" \
    --cfg_train "$CFG_TRAIN" \
    --train_timestep_strategy "$TRAIN_TIMESTEP_STRATEGY" \
    --sde_fraction "$SDE_FRACTION" \
    --train_steps_count "$TRAIN_STEPS_COUNT" \
    --clip_range "$CLIP_RANGE" \
    --adv_clip_max "$ADV_CLIP_MAX" \
    --kl_coeff "$KL_COEFF" \
    --train_microbatch_size "$TRAIN_MICROBATCH_SIZE" \
    --vae_decode_micro_batch "$VAE_DECODE_MICRO_BATCH" \
    \
    `# ── Reward ──` \
    --rewards "$REWARDS" \
    --reward_dispatch_mode "$REWARD_DISPATCH_MODE" \
    --geo_compare_mode all_pairs \
    --feature_compare_mode first_frame \
    --va_micro_batch "$VA_MICRO_BATCH" \
    --keep_intermediates \
    --skip_done \
    \
    `# ── 显存 ──` \
    --gradient_checkpoint_layers "$GRADIENT_CHECKPOINT_LAYERS" \
    \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/train.log"
