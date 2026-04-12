#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# Reward 计算配置 — 对应 rl_train/reward/reward.sh
# 修改此文件后，直接运行：
#   bash rl_train/reward/reward.sh
# 命令行参数可覆盖此处的值，例如：
#   bash rl_train/reward/reward.sh --gpu 1 --rewards camera_traj
# ═══════════════════════════════════════════════════════════════════

# ── [必填] 输入路径 ───────────────────────────────────────────────

# 生成视频路径（.mp4）
VIDEO_PATH="/path/to/gen.mp4"

# GT 相机轨迹文件（camera.txt）
GT_CAMERA_TXT="/path/to/camera.txt"

# ── [必填] 工作目录 ───────────────────────────────────────────────
# 所有中间产物和最终 reward.json 都写入此目录
WORK_DIR="/path/to/work_dir"

# ── Reward 计算范围 ───────────────────────────────────────────────
#
# 可选值（逗号分隔组合）：
#   all              — 计算全部 reward
#   camera_traj      — 仅相机轨迹（只跑 DA3）
#   geo_semantic     — 几何语义（深度重投影 + DINOv2）
#   videoalign       — VideoAlign 文本对齐
#   组合示例: "geo_semantic,camera_traj"
REWARDS="all"

# ── GPU ───────────────────────────────────────────────────────────

GPU=0

# ── VideoAlign 文本描述 ───────────────────────────────────────────
# VideoAlign 需要视频的文字描述，两种方式二选一：

# 方式1：直接填写 prompt
PROMPT=""

# 方式2：指向 metadata.json，脚本自动提取 caption（与 PROMPT 二选一）
METADATA_JSON=""

# ── 标志位（留空=关闭，填写对应 flag 字符串=开启）─────────────────

# 强制重跑所有 steps（默认跳过已有输出）：留空 或 "--no_skip"
NO_SKIP=""
