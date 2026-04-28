#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# 推理配置 — 对应 eval/infer/infer.sh
#
# 修改此文件后，直接运行：
#   bash eval/infer/infer.sh
# 命令行参数可覆盖此处的值，例如：
#   bash eval/infer/infer.sh --num_gpus 4 --output /other/path
#
# 注：${REPO_ROOT} 由 infer.sh 在 source 本文件前自动设置。
# ═══════════════════════════════════════════════════════════════════

# ── 模型名（对应 eval/infer/ 下的子文件夹）────────────────────────
# 可选：gen3r | wan2.2
MODEL="gen3r"

# ── [必填] Checkpoint 路径 ────────────────────────────────────────

# Gen3R 基础模型 checkpoint
CHECKPOINT="${REPO_ROOT}/eval/infer/gen3r/Gen3R/checkpoints"

# Wan2.2 基础模型 checkpoint（切换模型时取消注释）
# CHECKPOINT="${REPO_ROOT}/model/Wan2.2-Fun-5B-Control-Camera"

# ── Checkpoint 标识（用作输出目录中间层）─────────────────────────
# 默认等于 ${MODEL}，可以设置为 checkpoint 步数等，方便多次对比：
# CKPT_TAG="gen3r_baseline"
# CKPT_TAG="gen3r_step200"
CKPT_TAG=""   # 留空 = 用 ${MODEL} 值

# ── 输入数据（三选一）────────────────────────────────────────────

# 方式1：数据根目录（自动 build_manifest）
DATA="/mnt/afs/visitor16/rl_train_new/hf_datasets/rl_data/gen3r"
DATASETS="dl3dv,scannet++"

# 方式2：预构建 manifest.jsonl（与 DATA 二选一，留空则用 DATA）
MANIFEST=""

# 方式3：单个样本目录（调试用）
SAMPLE_DIR=""

# ── [必填] 输出根目录 ─────────────────────────────────────────────
# 实际写入：${OUTPUT}/${CKPT_TAG}/<dataset>/<sample_id>/
OUTPUT="/path/to/eval_output"

# ── GPU & 并行 ────────────────────────────────────────────────────
NUM_GPUS=1
MASTER_PORT=29500

# ── 每数据集最多取多少条（0=全部）───────────────────────────────
MAX_PER_DATASET=0

# ── 随机种子 ──────────────────────────────────────────────────────
SEED=42

# ── 标志位 ────────────────────────────────────────────────────────
# 跳过已完成样本：留空 或 "--skip_done"
SKIP_DONE="--skip_done"

# 设备模式（gen3r 用）：server（全量 GPU）| local（CPU offload）
DEVICE_MODE="server"

# 只处理含 gt.mp4 的样本：留空 或 "--require_gt_video"
REQUIRE_GT=""
