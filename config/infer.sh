#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# 推理配置 — 对应 eval/infer/infer.sh
# 修改此文件后，直接运行：
#   bash eval/infer/infer.sh
# 命令行参数可覆盖此处的值，例如：
#   bash eval/infer/infer.sh --num_gpus 4 --output /other/path
#
# 注：${REPO_ROOT} 由 infer.sh 在 source 本文件前自动设置，
#     代表 RL_code/ 项目根目录，可直接在此处引用。
# ═══════════════════════════════════════════════════════════════════

# ── 模型 ──────────────────────────────────────────────────────────

# 模型名，对应 eval/infer/ 下的子文件夹（如 gen3r）
MODEL="gen3r"

# ── [必填] Checkpoint 路径 ────────────────────────────────────────

# 使用仓库内自带的 Gen3R checkpoint（相对于项目根目录）
CHECKPOINT="${REPO_ROOT}/eval/infer/gen3r/Gen3R/checkpoints"

# 若使用 RL 训练后的 checkpoint，改为类似：
# CHECKPOINT="${REPO_ROOT}/rl_train/grpo_outputs/checkpoint-200"

# ── 输入数据（三选一） ────────────────────────────────────────────

# 方式1：处理后的数据根目录（脚本自动 build_manifest）
DATA="/path/to/processed"

# 配合 DATA 使用，指定数据集（逗号分隔）
DATASETS="re10k,dl3dv"

# 方式2：预构建的 manifest.jsonl（与 DATA 二选一，留空则用 DATA）
MANIFEST=""

# 方式3：单个样本目录（调试用，留空则用 DATA / MANIFEST）
SAMPLE_DIR=""

# ── [必填] 输出目录 ───────────────────────────────────────────────

OUTPUT="/path/to/output"

# ── GPU & 并行 ────────────────────────────────────────────────────

# GPU 数量（>1 时自动用 torchrun）
NUM_GPUS=1

# torchrun 的 master port（多卡时使用）
MASTER_PORT=29500

# ── 生成参数 ──────────────────────────────────────────────────────

# 每个样本生成多少条 rollout 视频
NUM_ROLLOUTS=8

# 生成帧数
NUM_FRAMES=49

# 输出分辨率
TARGET_SIZE=560

# SDE 噪声强度（η）
ETA=0.3

# 去噪步数
STEPS=50

# CFG scale
GUIDANCE=5.0

# sigma schedule shift
SHIFT=2.0

# 随机种子基值
SEED=42

# ── 标志位（留空=关闭，填写对应 flag 字符串=开启）─────────────────

# 跳过已完成样本：留空 或 "--skip_done"
SKIP_DONE="--skip_done"

# 设备模式：server（全量 GPU）| local（CPU offload）
DEVICE_MODE="server"

# 只处理含 gt.mp4 的样本：留空 或 "--require_gt_video"
REQUIRE_GT=""
