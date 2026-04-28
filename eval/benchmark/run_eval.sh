#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# run_eval.sh — 一键推理 + 评测脚本
#
# 流程：
#   1. 调用 eval/infer/infer.sh 执行推理 → 产出 gen_0.mp4
#   2. 调用 eval/benchmark/run_benchmark.sh 执行评测
#      固定评测指标：
#        video_quality.psnr  (PSNR / SSIM / LPIPS)
#        video_quality.vbench (VBench: i2v_subject, i2v_background, imaging_quality)
#        reward.camera_pose  (rotation_auc30, translation_auc30, pose_auc30, translation_metric)
#        reconstruction.global (chamfer_distance, fscore; Umeyama 对齐)
#   3. 打印 summary.json 中的关键指标
#
# 用法：
#   bash eval/benchmark/run_eval.sh \
#       --model gen3r \
#       --checkpoint /path/to/gen3r_checkpoints \
#       --ckpt_tag gen3r_baseline \
#       --data /mnt/afs/visitor16/rl_train_new/hf_datasets/rl_data/gen3r \
#       --datasets dl3dv,scannet++ \
#       --output_root /path/to/eval_runs \
#       --num_gpus 1 \
#       [--max_per_dataset N] [--gpu 0]
#       [--skip_infer]   # 跳过推理步骤，只跑评测
#       [--skip_bench]   # 跳过评测步骤，只跑推理
#       [--vbench_cache /path/to/vbench_cache]
#
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
INFER_SH="${REPO_ROOT}/eval/infer/infer.sh"
BENCH_SH="${SCRIPT_DIR}/run_benchmark.sh"
PYTHON="${PYTHON:-python3}"

# ── 默认值 ────────────────────────────────────────────────────────
MODEL="gen3r"
CHECKPOINT=""
CKPT_TAG=""
DATA=""
DATASETS=""
MANIFEST=""
SAMPLE_DIR=""
OUTPUT_ROOT=""
NUM_GPUS=1
GPU=0
MAX_PER_DATASET=0
SEED=42
SKIP_DONE="--skip_done"
VBENCH_CACHE=""
SKIP_INFER=false
SKIP_BENCH=false
REQUIRE_GT=""
MASTER_PORT=29500

# ── 命令行解析 ────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)            MODEL="$2";            shift 2 ;;
        --checkpoint)       CHECKPOINT="$2";       shift 2 ;;
        --ckpt_tag)         CKPT_TAG="$2";         shift 2 ;;
        --data)             DATA="$2";             shift 2 ;;
        --datasets)         DATASETS="$2";         shift 2 ;;
        --manifest)         MANIFEST="$2";         shift 2 ;;
        --sample_dir)       SAMPLE_DIR="$2";       shift 2 ;;
        --output_root)      OUTPUT_ROOT="$2";      shift 2 ;;
        --num_gpus)         NUM_GPUS="$2";         shift 2 ;;
        --gpu)              GPU="$2";              shift 2 ;;
        --max_per_dataset)  MAX_PER_DATASET="$2";  shift 2 ;;
        --seed)             SEED="$2";             shift 2 ;;
        --vbench_cache)     VBENCH_CACHE="$2";     shift 2 ;;
        --skip_infer)       SKIP_INFER=true;       shift ;;
        --skip_bench)       SKIP_BENCH=true;       shift ;;
        --require_gt)       REQUIRE_GT="--require_gt"; shift ;;
        --master_port)      MASTER_PORT="$2";      shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# ── 必填检查 ──────────────────────────────────────────────────────
if [[ -z "$OUTPUT_ROOT" ]]; then
    echo "错误: --output_root 是必填项"
    exit 1
fi
if [[ "$SKIP_INFER" = false && -z "$CHECKPOINT" ]]; then
    echo "错误: 推理时 --checkpoint 是必填项（或使用 --skip_infer 跳过推理）"
    exit 1
fi

# CKPT_TAG 默认值
if [[ -z "$CKPT_TAG" ]]; then
    CKPT_TAG="$MODEL"
fi

EFFECTIVE_OUTPUT="${OUTPUT_ROOT}/${CKPT_TAG}"
FIXED_METRICS="video_quality.psnr,video_quality.vbench,reward.camera_pose,reconstruction.global"
FIXED_ALIGN="umeyama"

echo "════════════════════════════════════════════════════════"
echo "  run_eval.sh — 推理 + 评测一键脚本"
echo "  Model:        ${MODEL}"
echo "  Ckpt tag:     ${CKPT_TAG}"
echo "  Output:       ${EFFECTIVE_OUTPUT}"
echo "  Metrics:      ${FIXED_METRICS}"
echo "  Align:        ${FIXED_ALIGN}"
echo "════════════════════════════════════════════════════════"

# ══════════════════════════ Step 1: 推理 ═══════════════════════════
if [[ "$SKIP_INFER" = false ]]; then
    echo ""
    echo "── Step 1: 推理 ────────────────────────────────────────────"

    INFER_ARGS=(
        --model "$MODEL"
        --checkpoint "$CHECKPOINT"
        --ckpt_tag "$CKPT_TAG"
        --output "$OUTPUT_ROOT"
        --num_gpus "$NUM_GPUS"
        --seed "$SEED"
        --master_port "$MASTER_PORT"
    )
    [[ -n "$DATA" ]]            && INFER_ARGS+=(--data "$DATA")
    [[ -n "$DATASETS" ]]        && INFER_ARGS+=(--datasets "$DATASETS")
    [[ -n "$MANIFEST" ]]        && INFER_ARGS+=(--manifest "$MANIFEST")
    [[ -n "$SAMPLE_DIR" ]]      && INFER_ARGS+=(--sample_dir "$SAMPLE_DIR")
    [[ -n "$REQUIRE_GT" ]]      && INFER_ARGS+=("$REQUIRE_GT")
    [[ -n "$SKIP_DONE" ]]       && INFER_ARGS+=($SKIP_DONE)
    [[ "$MAX_PER_DATASET" -gt 0 ]] && INFER_ARGS+=(--max_per_dataset "$MAX_PER_DATASET")

    bash "$INFER_SH" "${INFER_ARGS[@]}"
    echo "── 推理完成 ────────────────────────────────────────────────"
else
    echo "── Step 1: 跳过推理（--skip_infer）────────────────────────"
fi

# ══════════════════════════ Step 2: 评测 ═══════════════════════════
if [[ "$SKIP_BENCH" = false ]]; then
    echo ""
    echo "── Step 2: 评测 ────────────────────────────────────────────"

    if [[ ! -d "$EFFECTIVE_OUTPUT" ]]; then
        echo "错误: 推理输出目录不存在: ${EFFECTIVE_OUTPUT}"
        exit 1
    fi

    BENCH_ARGS=(
        --output_root "$EFFECTIVE_OUTPUT"
        --metrics "$FIXED_METRICS"
        --align "$FIXED_ALIGN"
        --gpu "$GPU"
    )
    [[ -n "$VBENCH_CACHE" ]] && BENCH_ARGS+=(--vbench_cache "$VBENCH_CACHE")

    bash "$BENCH_SH" "${BENCH_ARGS[@]}"
    echo "── 评测完成 ────────────────────────────────────────────────"
else
    echo "── Step 2: 跳过评测（--skip_bench）────────────────────────"
fi

# ══════════════════════════ Step 3: 打印关键指标 ══════════════════════
SUMMARY_JSON="${EFFECTIVE_OUTPUT}/benchmark_results/summary.json"
if [[ -f "$SUMMARY_JSON" ]]; then
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  Summary: ${SUMMARY_JSON}"
    echo "════════════════════════════════════════════════════════"
    export _EVAL_SUMMARY_PATH="$SUMMARY_JSON"
    ${PYTHON} - "$SUMMARY_JSON" <<'PYEOF'
import json, sys

summary_path = sys.argv[1]
with open(summary_path) as f:
    data = json.load(f)

KEYS = [
    ("PSNR",             "psnr_ssim_lpips.psnr"),
    ("SSIM",             "psnr_ssim_lpips.ssim"),
    ("LPIPS",            "psnr_ssim_lpips.lpips"),
    ("VBench i2v_subj",  "vbench.i2v_subject"),
    ("VBench i2v_bg",    "vbench.i2v_background"),
    ("VBench img_qual",  "vbench.imaging_quality"),
    ("Rot AUC30",        "camera_pose.rotation_auc30"),
    ("Trans AUC30",      "camera_pose.translation_auc30"),
    ("Pose AUC30",       "camera_pose.pose_auc30"),
    ("Trans metric",     "camera_pose.translation_metric"),
]

for label, key in KEYS:
    val = data.get(key, {})
    if val and "mean" in val:
        print(f"  {label:<18s}: mean={val['mean']:.4f}  (n={val.get('n', '?')})")
    else:
        print(f"  {label:<18s}: N/A")

# Global point cloud（chamfer_distance, fscore）
for k, v in data.items():
    if "global_point_cloud" in k.lower() or "chamfer" in k.lower() or "fscore" in k.lower():
        if isinstance(v, dict) and "mean" in v:
            print(f"  {k:<18s}: mean={v['mean']:.4f}  (n={v.get('n', '?')})")
PYEOF
    echo "════════════════════════════════════════════════════════"
else
    echo "（summary.json 不存在，评测可能尚未完成）"
fi
