#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# Reward 计算配置
#
# 两种使用模式，修改对应变量后直接运行对应脚本：
#
#   【单视频模式】修改下方 "单视频模式" 区块，然后运行：
#     bash rl_train/reward/reward.sh
#
#   【批量模式】修改下方 "批量模式" 区块，然后运行：
#     bash rl_train/reward/batch_reward.sh
#
# 命令行参数可覆盖此文件中的值，例如：
#   bash rl_train/reward/batch_reward.sh --gpu 1 --rewards camera_traj
# ═══════════════════════════════════════════════════════════════════

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 【单视频模式】reward.sh 使用以下变量
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 生成视频路径（.mp4）
VIDEO_PATH="/path/to/gen.mp4"

# GT 相机轨迹文件（camera.txt）
GT_CAMERA_TXT="/path/to/camera.txt"

# 所有中间产物和最终 reward.json 都写入此目录
WORK_DIR="/path/to/work_dir"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 【批量模式】batch_reward.sh 使用以下变量
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 推理输出目录（含 infer_info.json + gen_0.mp4 ... gen_N.mp4）
SAMPLE_DIR="/path/to/dl3dv/0a6c01ac32127687"

# 结果输出根目录（自动在此下创建 <dataset>/<sample_id>/ 子目录）
OUTPUT_DIR="/path/to/reward_output"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 【共享参数】两种模式均适用
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Reward 计算范围，可选值（逗号分隔组合）：
#   all                      — 计算全部 reward
#   camera_traj              — 仅相机轨迹（只跑 DA3）
#   geo_semantic             — 几何+语义（DA3 + SAM3）
#   geo_global               — 仅全局几何（DA3）
#   feature_sim              — 特征相似度（DA3 + DINOv2/FeatUp）
#   video_quality            — 视频质量（VideoAlign）
#   组合示例: "geo_semantic,camera_traj"
REWARDS="all"

# GPU 编号
GPU=0

# VideoAlign 视频描述（两种方式二选一）：
# 方式1：直接填写 prompt
PROMPT=""
# 方式2：指向 metadata.json，脚本自动提取 caption
METADATA_JSON=""

# 强制重跑所有 steps（默认跳过已有中间产物）：留空 或 "--no_skip"
NO_SKIP=""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 【高级参数】调试与消融实验
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# DA3 深度置信度阈值（0.0 = 不过滤，0.5 = 中等过滤）
# geo_semantic 和 geo_global 共用同一个阈值
CONF_THRESHOLD=0.0

# 几何一致性比较模式（geo_semantic / geo_global 共用）
#   first_frame : 所有帧与首帧比较，取均值
#   adjacent    : 相邻帧两两比较，取均值
#   all_pairs   : 每帧与所有其他帧比较后取均值，再对所有帧取均值
GEO_COMPARE_MODE="first_frame"

# DINOv2 特征相似度比较模式
#   first_frame : 所有帧与首帧比较，取均值
#   adjacent    : 相邻帧两两比较，取均值
#   all_pairs   : 每帧与所有其他帧比较后取均值，再对所有帧取均值
FEATURE_COMPARE_MODE="first_frame"
