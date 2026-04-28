#!/bin/bash
# =============================================================================
# launch_platform.sh — 云平台 PyTorch DDP 任务的统一启动入口
#
# 适用场景：在「分布式训练任务」(PAI / TI-One / 商汤 AC2 / 启智 / 自建 K8s) 平台
# 选 PyTorch DDP 框架 + 多副本 (例如 2)，平台会在每个副本容器里执行同一条命令。
# 本脚本会自动从平台注入的环境变量里识别自己是哪个 NODE_RANK，以及 master 地址。
#
# 平台 → 任务配置参考：
#   • 框架：PyTorch DDP
#   • 副本数 (workers / replicas)：2     ← 这就是 NNODES
#   • 每副本 GPU 数：8                    ← 这就是 NPROC_PER_NODE
#   • 启动命令：bash /mnt/afs/visitor16/rl_train_new/rl_train/train/gen3r/launch_platform.sh
#   • 工作目录：随便，本脚本用绝对路径
#
# 平台变量自动适配（按平台命名习惯依次回退）：
#   NODE_RANK    ← $RANK / $NODE_RANK / $INDEX / $PADDLE_TRAINER_ID / $PMI_RANK
#   NNODES       ← $WORLD_SIZE / $NNODES / $HOST_NUM / $PADDLE_TRAINERS_NUM
#   MASTER_ADDR  ← $MASTER_ADDR / $CHIEF_IP / $MASTER_HOST / $POD_HOSTNAME_0
#   MASTER_PORT  ← $MASTER_PORT (默认 29500)
#
# 注意：torchrun 自己也会读 $WORLD_SIZE / $RANK，但语义不同：
#   • 平台注入的 RANK / WORLD_SIZE 是 NODE 级别 (=NODE_RANK / NNODES)
#   • torchrun 的 RANK / WORLD_SIZE 是 PROCESS 级别 (=GPU 进程编号)
#   所以本脚本读完平台变量后会 unset 它们，避免污染 torchrun。
#
# 本地运行（不在平台上）：
#   bash launch_platform.sh                     # 自动检测：单机
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1  bash launch_platform.sh   # 手动多机
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[launch_platform] === Platform env detection ==="
echo "[launch_platform] hostname = $(hostname)"
echo "[launch_platform] all rank-related env vars seen:"
env | grep -E "^(RANK|NODE_RANK|WORLD_SIZE|NNODES|INDEX|HOST_NUM|MASTER_ADDR|MASTER_PORT|CHIEF_IP|MASTER_HOST|POD_HOSTNAME|PADDLE_TRAINER|PMI_)" | sort | sed 's/^/    /' || true
echo "[launch_platform] ==============================="

# ── 1. NODE_RANK：本机在多副本里的编号 ──────────────────────────────────────
# 优先级：用户显式 NODE_RANK > 平台 RANK > INDEX > PADDLE_TRAINER_ID > PMI_RANK > 0
RESOLVED_NODE_RANK="${NODE_RANK:-${RANK:-${INDEX:-${PADDLE_TRAINER_ID:-${PMI_RANK:-0}}}}}"

# ── 2. NNODES：总副本数 ─────────────────────────────────────────────────────
# 优先级：用户显式 NNODES > 平台 WORLD_SIZE > HOST_NUM > PADDLE_TRAINERS_NUM > 1
RESOLVED_NNODES="${NNODES:-${WORLD_SIZE:-${HOST_NUM:-${PADDLE_TRAINERS_NUM:-1}}}}"

# ── 3. MASTER_ADDR ──────────────────────────────────────────────────────────
RESOLVED_MASTER_ADDR="${MASTER_ADDR:-${CHIEF_IP:-${MASTER_HOST:-${POD_HOSTNAME_0:-localhost}}}}"
RESOLVED_MASTER_PORT="${MASTER_PORT:-29500}"

# ── 4. NPROC_PER_NODE：每副本 GPU 数 ─────────────────────────────────────────
# 优先用平台/用户显式值，否则 nvidia-smi 自检
if [ -n "${NPROC_PER_NODE:-}" ]; then
    RESOLVED_NPROC="$NPROC_PER_NODE"
elif command -v nvidia-smi >/dev/null 2>&1; then
    RESOLVED_NPROC="$(nvidia-smi --list-gpus | wc -l)"
else
    RESOLVED_NPROC=1
fi

# ── 关键：unset 平台注入的 NODE 级 RANK/WORLD_SIZE，避免污染 torchrun ──────
# torchrun 内部用 $RANK/$WORLD_SIZE 表示 PROCESS 级，平台用的是 NODE 级
unset RANK WORLD_SIZE 2>/dev/null || true

# ── 边界检查 ────────────────────────────────────────────────────────────────
if ! [[ "$RESOLVED_NODE_RANK" =~ ^[0-9]+$ ]]; then
    echo "[launch_platform] FATAL: NODE_RANK 解析失败：'$RESOLVED_NODE_RANK'" >&2
    exit 1
fi
if ! [[ "$RESOLVED_NNODES" =~ ^[0-9]+$ ]] || [ "$RESOLVED_NNODES" -lt 1 ]; then
    echo "[launch_platform] FATAL: NNODES 解析失败：'$RESOLVED_NNODES'" >&2
    exit 1
fi
if [ "$RESOLVED_NODE_RANK" -ge "$RESOLVED_NNODES" ]; then
    echo "[launch_platform] FATAL: NODE_RANK=$RESOLVED_NODE_RANK ≥ NNODES=$RESOLVED_NNODES" >&2
    exit 1
fi

# 多机但 master 还是 localhost → 致命错误
if [ "$RESOLVED_NNODES" -gt 1 ] && \
   { [ "$RESOLVED_MASTER_ADDR" = "localhost" ] || [ "$RESOLVED_MASTER_ADDR" = "127.0.0.1" ]; }; then
    cat >&2 <<EOF
[launch_platform] FATAL: NNODES=$RESOLVED_NNODES > 1 但 MASTER_ADDR='$RESOLVED_MASTER_ADDR'。
                  请检查平台是否注入了 MASTER_ADDR / CHIEF_IP / MASTER_HOST 之一。
                  若没有，可以手动 export MASTER_ADDR=<节点0_IP> 后再启动。
EOF
    exit 1
fi

cat <<EOF
[launch_platform] === Resolved ===
[launch_platform]   NNODES         = $RESOLVED_NNODES
[launch_platform]   NODE_RANK      = $RESOLVED_NODE_RANK
[launch_platform]   NPROC_PER_NODE = $RESOLVED_NPROC
[launch_platform]   MASTER_ADDR    = $RESOLVED_MASTER_ADDR
[launch_platform]   MASTER_PORT    = $RESOLVED_MASTER_PORT
[launch_platform]   total GPUs     = $((RESOLVED_NNODES * RESOLVED_NPROC))
[launch_platform] ================
EOF

# ── 透传给 run_train_h100_16gpu.sh ──────────────────────────────────────────
# 用户在平台 UI 里把 KL_COEFF / LEARNING_RATE / RESUME_FROM 等加到环境变量里也会自动透传，
# 因为 run_train_h100_16gpu.sh 用的是 ${VAR:-default} 形式
NNODES="$RESOLVED_NNODES" \
NODE_RANK="$RESOLVED_NODE_RANK" \
NPROC_PER_NODE="$RESOLVED_NPROC" \
MASTER_ADDR="$RESOLVED_MASTER_ADDR" \
MASTER_PORT="$RESOLVED_MASTER_PORT" \
    exec bash "$SCRIPT_DIR/run_train_h100_16gpu.sh"
