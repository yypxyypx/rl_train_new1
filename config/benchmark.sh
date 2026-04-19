#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# Benchmark 评测配置（唯一配置文件）
# 修改此文件后，直接运行：
#   bash eval/benchmark/run_benchmark.sh
# 命令行参数可覆盖此处的值，例如：
#   bash eval/benchmark/run_benchmark.sh --metrics video_quality --gpu 1
# ═══════════════════════════════════════════════════════════════════

# ── [必填] 推理输出根目录 ─────────────────────────────────────────

OUTPUT_ROOT="/path/to/output"

# ── 评测指标 ──────────────────────────────────────────────────────
#
# 可选粒度（逗号分隔组合）：
#   all                              — 全部
#   ────── video_quality ──────
#   video_quality                    — PSNR + VBench
#   video_quality.psnr               — 仅 PSNR/SSIM/LPIPS
#   video_quality.vbench             — 仅 VBench
#   ────── reward ──────
#   reward                           — 所有 reward
#   reward.camera_pose               — 旋转 AUC + 平移方向 AUC + Pose AUC + 平移距离
#   reward.depth_reprojection        — 物体级 + 全局深度重投影
#   reward.depth_reprojection.object — 仅物体级
#   reward.depth_reprojection.global — 仅全局
#   reward.videoalign                — VideoAlign
#   reward.feature_sim               — DINOv2 特征相似度
#   ────── reconstruction ──────
#   reconstruction                   — 全局 + 物体级点云
#   reconstruction.global            — 仅全局点云
#   reconstruction.object            — 仅物体级点云
#
#   组合示例: "video_quality.psnr,reward.camera_pose,reconstruction.global"
METRICS="all"

# ── GPU / 设备 ────────────────────────────────────────────────────

GPU=0
# 覆盖 CUDA 设备名，留空则自动设为 cuda:<GPU>
DEVICE=""

# ── 重建对齐方式（仅影响 reconstruction.* 指标）─────────────────
#
# 四种基础模式：
#   camera      — Umeyama 对齐相机位置求 scale → 用 pred 内参 + GT 外参重投影
#   first_frame — 首帧 pose 对齐 + 轨迹长度 scale → 用 pred 内参 + 对齐后外参重投影
#                 （最接近 RL 训练 reward 的对齐方式）
#   umeyama     — 各用自己相机参数重投影 → Umeyama 相似变换对齐点云
#   icp         — umeyama 基础上加 ICP 精化（最精确，耗时最长）
#
# 组合快捷：
#   both_align  — 同时运行 camera + first_frame（默认推荐）
#   all_align   — 同时运行全部四种模式（输出 JSON 包含四个子键）
ALIGN="both_align"

# ── 重建点云参数（仅影响 reconstruction.* 指标）─────────────────

# FPS 采样点数（全局点云），物体级按面积比例分配
N_FPS=20000
# 深度置信度阈值，低于此值的像素不参与点云生成（0.0 = 不过滤）
CONF_THRESH=0.0

# ── VBench 缓存目录 ───────────────────────────────────────────────
# 留空则自动检测
VBENCH_CACHE=""

# ── 标志位（true/false）──────────────────────────────────────────

# 跳过中间产物生成（假设已存在）
SKIP_INTERMEDIATES=false
# 仅汇总已有结果（不重新跑评测）
AGGREGATE_ONLY=false
