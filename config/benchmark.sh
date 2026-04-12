#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# Benchmark 评测配置 — 对应 eval/benchmark/run_benchmark.sh
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
#   video_quality                    — PSNR + VBench
#   video_quality.psnr               — 仅 PSNR/SSIM/LPIPS
#   video_quality.vbench             — 仅 VBench
#   reward                           — 所有 reward
#   reward.camera_pose               — 旋转 AUC + 平移
#   reward.depth_reprojection        — 物体级 + 全局深度重投影
#   reward.depth_reprojection.object — 仅物体级
#   reward.depth_reprojection.global — 仅全局
#   reward.videoalign                — VideoAlign
#   reward.feature_sim               — DINOv2 特征相似度
#   reconstruction                   — 全局 + 物体级点云
#   reconstruction.global            — 仅全局点云
#   reconstruction.object            — 仅物体级点云
#   组合示例: "video_quality.psnr,reward.camera_pose,reconstruction.global"
METRICS="all"

# ── GPU ───────────────────────────────────────────────────────────

GPU=0

# ── 重建对齐方式 ──────────────────────────────────────────────────
# 可选：camera | first_frame | both_align
ALIGN="both_align"

# ── VBench 缓存目录 ───────────────────────────────────────────────
# 留空则自动检测
VBENCH_CACHE=""

# ── 标志位（true/false）──────────────────────────────────────────

# 跳过中间产物生成（假设已存在）
SKIP_INTERMEDIATES=false

# 仅汇总已有结果（不重新跑评测）
AGGREGATE_ONLY=false
