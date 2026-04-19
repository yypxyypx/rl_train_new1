#!/usr/bin/env python3
"""
run_benchmark_recompute.py
==========================
Phase 3: 只跑选择性 benchmark 指标（camera_pose / videoalign / 点云重建），
多进程并行加速（--n_workers 控制进程数，默认 8）。

PSNR / VBench 已由历史 run 写入 eval/，Phase 4 直接读取，不再重跑。

点云结果写入新文件名（默认 *_mm.json），避免 bucket 禁止 unlink。

用法:
    python run_benchmark_recompute.py \\
        --output_root /horizon-bucket/robot_lab/users/puxin.yan-labs/test_output1 \\
        [--gpu 0] [--align all_align] [--n_workers 8] [--recon_json_suffix mm]
"""

import argparse
import subprocess
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_RL_CODE_DIR = _THIS_DIR.parent.parent
_BENCHMARK_DIR = _RL_CODE_DIR / "eval" / "benchmark"

sys.path.insert(0, str(_BENCHMARK_DIR))
from common.scan import scan_output_root
from common.utils import log

# 与 Phase 4 run_correlation_analysis 中读取的后缀保持一致
DEFAULT_RECON_SUFFIX = "mm"

# PSNR / VBench 已有；camera_pose / videoalign / 点云 是本次需要重算的
SELECTIVE_METRICS = (
    "reward.camera_pose,"
    "reward.videoalign,"
    "reconstruction.both"
)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: Selective benchmark (bucket-safe, multi-process)")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--gpu", type=int, default=0,
                        help="单进程时的 GPU 编号；多进程时每个 worker 都用这个 GPU")
    parser.add_argument("--align", default="all_align",
                        choices=["camera", "first_frame", "umeyama", "icp",
                                 "both_align", "all_align"])
    parser.add_argument("--n_fps", type=int, default=20000)
    parser.add_argument("--conf_thresh", type=float, default=0.0)
    parser.add_argument("--recon_json_suffix", type=str,
                        default=DEFAULT_RECON_SUFFIX,
                        help="点云 eval 文件名后缀（默认 mm -> *_mm.json）")
    parser.add_argument("--no_force_recon", action="store_true",
                        help="不强制覆盖：若 *_mm.json 已存在则跳过")
    parser.add_argument("--n_workers", type=int, default=8,
                        help="并行进程数（默认 8），每进程处理约 1/N 的 entries")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    suf = (args.recon_json_suffix or "").strip()
    force_flag = [] if args.no_force_recon else ["--force_recon"]

    # ── 公共 benchmark 参数 ────────────────────────────────────────
    base_cmd = [
        sys.executable, "-u",
        str(_BENCHMARK_DIR / "run_benchmark.py"),
        "--output_root", str(output_root),
        "--metrics", SELECTIVE_METRICS,
        "--gpu", str(args.gpu),
        "--align", args.align,
        "--n_fps", str(args.n_fps),
        "--conf_thresh", str(args.conf_thresh),
        "--skip_intermediates",
        "--recon_json_suffix", suf,
    ] + force_flag

    n = args.n_workers
    log(f"[Phase 3] 启动 {n} 个并行 worker（camera_pose / videoalign / 点云）...")
    log(f"[Phase 3] 点云将写入 global_point_cloud_{suf}.json / "
        f"object_point_cloud_{suf}.json")

    if n == 1:
        cmd = base_cmd + ["--n_shards", "1", "--shard_idx", "0"]
        log(f"[Phase 3] 执行: {' '.join(cmd[-10:])}")
        rc = subprocess.run(cmd).returncode
    else:
        procs = []
        for i in range(n):
            cmd = base_cmd + [
                "--shard_idx", str(i),
                "--n_shards",  str(n),
            ]
            log(f"  启动 shard {i}/{n}")
            procs.append(subprocess.Popen(cmd))

        codes = [p.wait() for p in procs]
        if any(c != 0 for c in codes):
            log(f"[Phase 3] 部分 worker 失败: {codes}")
            sys.exit(1)
        rc = 0

    if rc != 0:
        log(f"[Phase 3] Benchmark 失败 (rc={rc})")
        sys.exit(rc)

    # ── 汇总（全量，单进程）────────────────────────────────────────
    log("[Phase 3] 所有 worker 完成，执行汇总 ...")
    agg_cmd = base_cmd + ["--aggregate_only"]
    rc2 = subprocess.run(agg_cmd).returncode
    if rc2 != 0:
        log(f"[Phase 3] 汇总失败 (rc={rc2})")
        sys.exit(rc2)

    log("[Phase 3] Benchmark 完成！")


if __name__ == "__main__":
    main()
