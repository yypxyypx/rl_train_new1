#!/bin/bash
# =============================================================================
# launch_2node.sh — 2 节点 × 8 卡 H100 GRPO 训练一键启动脚手架
#
# 用法（两台机器各跑一次）：
#   节点 0：bash launch_2node.sh 0
#   节点 1：bash launch_2node.sh 1
#
# 必填环境变量（任选其一方式）：
#   1) 命令行 prefix：MASTER_IP=10.0.0.1 bash launch_2node.sh 0
#   2) 写到本脚本里 DEFAULT_MASTER_IP（推荐 — 改一次以后两台机器一致）
#
# 可选覆盖（直接放在命令前）：
#   KL_COEFF=0.1 LEARNING_RATE=2e-5 RESUME_FROM=auto bash launch_2node.sh 0
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ★ 两台机器都改这里，或者每次命令前 export MASTER_IP=...
DEFAULT_MASTER_IP=""    # 例如 "10.0.0.1"  ← 必填，改成节点 0 的 IP
DEFAULT_MASTER_PORT=29500

if [ "$#" -ne 1 ]; then
    echo "Usage: bash $0 <NODE_RANK>     # NODE_RANK = 0 or 1" >&2
    exit 1
fi
NODE_RANK="$1"

if [ "$NODE_RANK" != "0" ] && [ "$NODE_RANK" != "1" ]; then
    echo "ERROR: NODE_RANK must be 0 or 1, got '$NODE_RANK'" >&2
    exit 1
fi

MASTER_ADDR="${MASTER_IP:-$DEFAULT_MASTER_IP}"
MASTER_PORT="${MASTER_PORT:-$DEFAULT_MASTER_PORT}"

if [ -z "$MASTER_ADDR" ]; then
    cat >&2 <<EOF
ERROR: 没有指定 MASTER_IP。三种方式都行：
  方式 1：命令行       MASTER_IP=10.0.0.1 bash $0 $NODE_RANK
  方式 2：编辑本脚本    把 DEFAULT_MASTER_IP="" 改成节点 0 IP
  方式 3：写到 ~/.bashrc  export MASTER_IP=10.0.0.1
EOF
    exit 1
fi

# 自检：节点 0 上提示当前机器 IP，便于核对 MASTER_IP 是否对
if [ "$NODE_RANK" = "0" ]; then
    echo "[launch] 我是节点 0；本机可用 IP："
    ip -4 -o addr show | awk '{print "         " $2 "  " $4}' | grep -v "127.0.0.1" || true
    echo "         MASTER_ADDR 应为本机其中一个 IP：$MASTER_ADDR"
fi

# 透传所有可能用户在命令前 export 的覆盖项给 run_train 脚本
NNODES=2 \
NODE_RANK="$NODE_RANK" \
NPROC_PER_NODE="${NPROC_PER_NODE:-8}" \
MASTER_ADDR="$MASTER_ADDR" \
MASTER_PORT="$MASTER_PORT" \
    bash "$SCRIPT_DIR/run_train_h100_16gpu.sh"
