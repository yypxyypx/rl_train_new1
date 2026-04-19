#!/usr/bin/env python3
"""
run_sam3_recompute.py
=====================
Phase 1: 重跑 SAM3 视频模式（4 GPU 并行）。

文件系统约束：/horizon-bucket 挂载为只追加模式，无法删除/重命名已有文件。
解决方案：
  1. 将 manifest 中的输出路径指向 _v2 后缀的新文件（不存在 → worker 正常运行）
  2. Worker 完成后，将 _v2 文件 cp（覆盖写）到标准路径，下游代码路径不变
  3. _v2 文件无法删除，作为备份保留即可

同一 sample 的 GT + gen 视频必须在同一 shard（gen 依赖 GT 的 objects_v2.json）。
每个 worker 设 CUDA_VISIBLE_DEVICES=<gpu_id>，内部统一用 --sam3_gpu 0。

用法:
    python run_sam3_recompute.py \
        --output_root /horizon-bucket/robot_lab/users/puxin.yan-labs/test_output1 \
        [--n_gpus 4] [--gpu_ids 1,2,3,5] [--dry_run]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_RL_CODE_DIR = _THIS_DIR.parent.parent
_BENCHMARK_DIR = _RL_CODE_DIR / "eval" / "benchmark"
_WORKERS_DIR = _RL_CODE_DIR / "third_party" / "workers"

sys.path.insert(0, str(_BENCHMARK_DIR))
from common.scan import scan_output_root
from common.utils import log, env_python

# 标准路径 → _v2 路径 映射
GT_FILE_MAP = {
    "gt_masks.npz":      "gt_masks_v2.npz",
    "gt_objects.json":   "gt_objects_v2.json",
    "gt_label_maps.npz": "gt_label_maps_v2.npz",
}
PRED_FILE_MAP = {
    "pred_masks.npz":   "pred_masks_v2.npz",
    "pred_objects.json":"pred_objects_v2.json",
    "label_maps.npz":   "label_maps_v2.npz",
}


def detect_free_gpus(n: int = 4, min_free_mib: int = 10000) -> list:
    """通过 nvidia-smi 检测空闲 GPU，按可用显存排序，取 top-n。"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = line.strip().split(", ")
            if len(parts) == 2:
                idx, free = int(parts[0]), int(parts[1])
                if free >= min_free_mib:
                    gpus.append((free, idx))
        gpus.sort(reverse=True)
        selected = [idx for _, idx in gpus[:n]]
        log(f"Auto-detected free GPUs (top {n}): {selected}")
        return selected
    except Exception as e:
        log(f"nvidia-smi failed: {e}. Using GPUs 0,1,2,3 by default.")
        return list(range(n))


def build_manifest(entries: list) -> list:
    """
    构建 SAM3 batch manifest。
    输出路径使用 _v2 后缀（避免与已有文件冲突，且不需要删除旧文件）。
    GT 条目在前（需要 Qwen 识别），gen 条目在后（复用 GT objects_v2.json）。
    每个条目携带 _sample_id 用于后续分片。
    同时返回 copy_plan: [(src_v2, dst_standard), ...]，供 finalize 步骤使用。
    """
    batch = []
    copy_plan = []  # (v2_path, standard_path)

    for entry in entries:
        gt_inter = entry["sample_dir"] / "gt_intermediates"
        gt_inter.mkdir(parents=True, exist_ok=True)

        # GT 输出 → _v2 路径
        gt_masks_v2    = str(gt_inter / GT_FILE_MAP["gt_masks.npz"])
        gt_objects_v2  = str(gt_inter / GT_FILE_MAP["gt_objects.json"])
        gt_labels_v2   = str(gt_inter / GT_FILE_MAP["gt_label_maps.npz"])

        batch.append({
            "video_path":            entry["gt_video"],
            "output_masks_npz":      gt_masks_v2,
            "output_objects_json":   gt_objects_v2,
            "output_label_maps_npz": gt_labels_v2,
            "ref_objects_json":      None,
            "is_gt":                 True,
            "max_frames":            0,
            "_sample_id":            entry["sample_id"],
        })
        copy_plan.extend([
            (gt_masks_v2,   str(gt_inter / "gt_masks.npz")),
            (gt_objects_v2, str(gt_inter / "gt_objects.json")),
            (gt_labels_v2,  str(gt_inter / "gt_label_maps.npz")),
        ])

        # Gen 条目（复用 GT objects_v2.json）
        for gv in entry["gen_videos"]:
            inter_dir = gv["gen_dir"] / "intermediates"
            inter_dir.mkdir(parents=True, exist_ok=True)

            masks_v2   = str(inter_dir / PRED_FILE_MAP["pred_masks.npz"])
            objects_v2 = str(inter_dir / PRED_FILE_MAP["pred_objects.json"])
            labels_v2  = str(inter_dir / PRED_FILE_MAP["label_maps.npz"])

            batch.append({
                "video_path":            gv["video_path"],
                "output_masks_npz":      masks_v2,
                "output_objects_json":   objects_v2,
                "output_label_maps_npz": labels_v2,
                "ref_objects_json":      gt_objects_v2,
                "is_gt":                 False,
                "max_frames":            0,
                "_sample_id":            entry["sample_id"],
            })
            copy_plan.extend([
                (masks_v2,   str(inter_dir / "pred_masks.npz")),
                (objects_v2, str(inter_dir / "pred_objects.json")),
                (labels_v2,  str(inter_dir / "label_maps.npz")),
            ])

    return batch, copy_plan


def finalize_outputs(copy_plan: list) -> tuple:
    """
    将所有 _v2 文件 cp 覆盖到标准路径。
    返回 (n_ok, n_missing)。
    """
    n_ok = n_missing = 0
    for src, dst in copy_plan:
        src_p = Path(src)
        if not src_p.exists():
            n_missing += 1
            continue
        shutil.copy2(str(src_p), dst)
        n_ok += 1
    return n_ok, n_missing


def split_by_sample(batch: list, n_shards: int) -> list:
    """
    按 sample 为单位将 manifest 均分为 n_shards 份。
    保证同一 sample 的 GT + gen 全部在同一 shard。
    """
    from collections import OrderedDict

    groups = OrderedDict()
    for item in batch:
        sid = item["_sample_id"]
        groups.setdefault(sid, []).append(item)

    sample_ids = list(groups.keys())
    n = len(sample_ids)
    # 将 samples 均匀分配给 shards（循环分配，保证均衡）
    shards = [[] for _ in range(n_shards)]
    for i, sid in enumerate(sample_ids):
        shards[i % n_shards].extend(groups[sid])
    return shards


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: SAM3 recompute (_v2 suffix + finalize copy)")
    parser.add_argument("--output_root", required=True,
                        help="测试数据根目录，如 test_output1")
    parser.add_argument("--n_gpus", type=int, default=4,
                        help="并行 GPU 数量（auto-detect 时取 top-n）")
    parser.add_argument("--gpu_ids", type=str, default=None,
                        help="手动指定 GPU ID，逗号分隔，如 '1,2,3,5'")
    parser.add_argument("--min_free_mib", type=int, default=10000,
                        help="GPU 最低空闲显存 (MiB)")
    parser.add_argument("--skip_sam", action="store_true",
                        help="跳过 SAM3 运行（仅执行 finalize copy）")
    parser.add_argument("--dry_run", action="store_true",
                        help="仅打印将要执行的操作，不实际运行")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    log(f"[Phase 1] 扫描 {output_root} ...")
    entries = scan_output_root(str(output_root))
    log(f"找到 {len(entries)} 个样本")

    # ── Step 1: 构建 manifest（输出路径带 _v2 后缀）─────────────────
    log("[Phase 1] 构建 SAM3 batch manifest（_v2 输出路径）...")
    batch, copy_plan = build_manifest(entries)
    log(f"Manifest 总条目: {len(batch)}，Finalize 复制对: {len(copy_plan)}")

    if args.dry_run:
        log("[Phase 1] Dry run 模式，跳过实际执行")
        return

    if args.skip_sam:
        log("[Phase 1] --skip_sam: 跳过 SAM3 运行，直接执行 finalize ...")
        n_ok, n_miss = finalize_outputs(copy_plan)
        log(f"[Phase 1] Finalize 完成: ok={n_ok} missing={n_miss}")
        return

    # ── Step 2: 检测 GPU ──────────────────────────────────────────────
    if args.gpu_ids:
        gpu_ids = [int(g.strip()) for g in args.gpu_ids.split(",")]
    else:
        gpu_ids = detect_free_gpus(n=args.n_gpus,
                                   min_free_mib=args.min_free_mib)
    log(f"[Phase 1] 使用 GPU: {gpu_ids}")

    # ── Step 4: 分片 ─────────────────────────────────────────────────
    n_gpus = len(gpu_ids)
    shards = split_by_sample(batch, n_gpus)
    for i, shard in enumerate(shards):
        n_gt = sum(1 for x in shard if x.get("is_gt"))
        n_gen = len(shard) - n_gt
        log(f"  Shard {i} (GPU {gpu_ids[i]}): "
            f"{len(shard)} 条目 ({n_gt} GT + {n_gen} gen)")

    # ── Step 5: 写分片 manifest 并启动并行 worker ─────────────────────
    tmp_dir = output_root / "_benchmark_tmp" / "sam3_parallel"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    worker_script = str(_WORKERS_DIR / "worker_sam3.py")
    py_sam3 = env_python("SAM3")

    processes = []
    for i, (shard, gpu_idx) in enumerate(zip(shards, gpu_ids)):
        manifest_path = str(tmp_dir / f"sam3_shard_{i}.json")
        # 去掉内部 _sample_id 字段
        clean = [{k: v for k, v in item.items() if k != "_sample_id"}
                 for item in shard]
        with open(manifest_path, "w") as f:
            json.dump(clean, f, ensure_ascii=False, indent=2)

        log_file = str(tmp_dir / f"worker_{i}_gpu{gpu_idx}.log")
        cmd = [
            py_sam3, "-u", worker_script,
            "--batch_manifest", manifest_path,
            "--sam3_gpu", "0",
            "--qwen_gpu", "0",
        ]
        env_vars = dict(os.environ)
        env_vars["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
        log(f"  启动 shard {i} on GPU {gpu_idx} -> log: {log_file}")
        with open(log_file, "w") as lf:
            p = subprocess.Popen(cmd, env=env_vars, stdout=lf, stderr=lf)
        processes.append((i, gpu_idx, p, log_file))

    # ── Step 6: 等待所有 worker 完成 ──────────────────────────────────
    log(f"[Phase 1] 已启动 {len(processes)} 个 worker，等待完成 ...")
    failed = []
    for i, gpu_idx, p, log_file in processes:
        rc = p.wait()
        if rc != 0:
            log(f"  [ERROR] Shard {i} GPU {gpu_idx} 失败 "
                f"(rc={rc})，日志: {log_file}")
            failed.append(i)
        else:
            log(f"  [OK] Shard {i} GPU {gpu_idx} 完成")

    if failed:
        log(f"[Phase 1] 失败的 shard: {failed}")
        sys.exit(1)
    else:
        log("[Phase 1] 所有 SAM3 worker 完成！")

    # ── Step 7: Finalize：将 _v2 文件 cp 覆盖到标准路径 ──────────────
    log("[Phase 1] Finalize：将 _v2 输出复制到标准路径 ...")
    n_ok, n_miss = finalize_outputs(copy_plan)
    log(f"[Phase 1] Finalize 完成: ok={n_ok} missing={n_miss}")
    if n_miss > 0:
        log(f"[Phase 1] 警告：{n_miss} 个 _v2 文件缺失，对应 SAM3 可能失败")


if __name__ == "__main__":
    main()
