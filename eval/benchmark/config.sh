#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# Benchmark 配置文件
# 修改下面的参数后，直接运行 bash run_benchmark.sh 即可
# ═══════════════════════════════════════════════════════════════════

# [必填] 推理输出根目录
OUTPUT_ROOT="/path/to/output"

# 要评测的指标，逗号分隔，可选粒度：
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

# GPU 编号
GPU=0

# 重建对齐方式: camera | first_frame | both_align
ALIGN="both_align"

# VBench 缓存目录（留空则自动检测）
VBENCH_CACHE=""

# 跳过中间产物生成（假设已存在）: true / false
SKIP_INTERMEDIATES=false

# 仅汇总已有结果（不跑评测）: true / false
AGGREGATE_ONLY=false
