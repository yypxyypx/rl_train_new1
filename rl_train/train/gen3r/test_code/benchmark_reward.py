"""benchmark_reward.py — 4×4090 reward 速度 benchmark。

测试流程：
  1. 从数据集选 4 个样本
  2. 假设已有生成的视频（或从 --infer_dir 读取）
  3. 4 张 4090 分别负责一个 reward 步骤（完全并行）
  4. 记录每个步骤的耗时（模型加载 + 推理）
  5. 对比：顺序执行方案（单卡跑全部 reward）

用法：
    python benchmark_reward.py \\
        --infer_dir /path/to/generated_videos \\
        --output_dir ./results/benchmark_run_001 \\
        --samples re10k/sample_0001 re10k/sample_0002 re10k/sample_0003 re10k/sample_0004

    # 或者用 benchmark_reward.sh 运行（自动填好路径）
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent.parent
_REWARD_DIR = _HERE.parent.parent / "reward"
_STEPS_DIR = _REWARD_DIR / "steps"

STEP_CONDA_ENV = {
    "step_da3":            "rl_da3",
    "step_qwen_sam3":      "SAM3",
    "step_dinov2_extract": "rl_da3",
    "step_videoalign":     "Videoalign",
}

STEP_TO_GPU = {
    "step_da3":            0,
    "step_qwen_sam3":      1,
    "step_dinov2_extract": 2,
    "step_videoalign":     3,
}


def _env_python(env_name: str) -> str:
    candidates = [
        f"/opt/conda/envs/{env_name}/bin/python",
        os.path.expanduser(f"~/miniconda3/envs/{env_name}/bin/python"),
        os.path.expanduser(f"~/anaconda3/envs/{env_name}/bin/python"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return f"/opt/conda/envs/{env_name}/bin/python"


def _extract_frames_cv2(video_path: str, out_dir: str) -> list[str]:
    import cv2
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    existing = sorted(out.glob("frame_*.png"))
    if existing:
        print(f"  Frames already extracted: {len(existing)}")
        return [str(p) for p in existing]
    cap = cv2.VideoCapture(str(video_path))
    paths, idx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        p = out / f"frame_{idx:05d}.png"
        cv2.imwrite(str(p), frame)
        paths.append(str(p))
        idx += 1
    cap.release()
    print(f"  Extracted {len(paths)} frames -> {out_dir}")
    return paths


def run_step_timed(
    step_name: str,
    args_list: list[str],
    gpu_id: int,
    log_path: str,
) -> dict:
    """运行一个 reward step，返回 timing 信息。"""
    env_name = STEP_CONDA_ENV[step_name]
    py = _env_python(env_name)
    script = str(_STEPS_DIR / f"{step_name}.py")
    cmd = [py, "-u", script] + args_list

    env_vars = dict(os.environ)
    env_vars["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"  [{step_name}] GPU={gpu_id}  env={env_name}")
    t0 = time.time()

    with open(log_path, "w") as f:
        result = subprocess.run(cmd, env=env_vars, stdout=f, stderr=subprocess.STDOUT)

    elapsed = time.time() - t0
    ok = result.returncode == 0
    print(f"  [{step_name}] {'OK' if ok else 'FAILED'} in {elapsed:.1f}s")

    return {"step": step_name, "elapsed_s": elapsed, "ok": ok, "gpu": gpu_id}


def benchmark_parallel(
    video_paths: list[str],
    frames_dirs: list[str],
    gt_camera_txts: list[str],
    intermediates_dirs: list[str],
    prompts: list[str],
    logs_dir: str,
) -> dict:
    """4 步骤完全并行（各在不同 GPU），测量总时间。"""
    import threading

    os.makedirs(logs_dir, exist_ok=True)
    all_timings = []

    def _run_for_all_samples(step_name: str, gpu_id: int):
        for i, (frames_dir, inter_dir, vp, prompt) in enumerate(
                zip(frames_dirs, intermediates_dirs, video_paths, prompts)):
            Path(inter_dir).mkdir(parents=True, exist_ok=True)
            da3_npz = os.path.join(inter_dir, "da3_output.npz")
            label_maps_npz = os.path.join(inter_dir, "label_maps.npz")
            dinov2_npz = os.path.join(inter_dir, "dinov2_features.npz")
            videoalign_json = os.path.join(inter_dir, "videoalign.json")
            objects_json = os.path.join(inter_dir, "objects.json")

            log_path = os.path.join(logs_dir, f"{step_name}_sample{i}.log")

            if step_name == "step_da3":
                args_list = ["--video_frames_dir", frames_dir,
                             "--output", da3_npz, "--gpu", "0"]
            elif step_name == "step_qwen_sam3":
                args_list = ["--video_frames_dir", frames_dir,
                             "--output", label_maps_npz,
                             "--objects_output", objects_json, "--gpu", "0"]
            elif step_name == "step_dinov2_extract":
                args_list = ["--video_frames_dir", frames_dir,
                             "--output", dinov2_npz, "--gpu", "0"]
            elif step_name == "step_videoalign":
                args_list = ["--video_path", vp,
                             "--prompt", prompt,
                             "--output", videoalign_json, "--gpu", "0"]
            else:
                continue

            timing = run_step_timed(step_name, args_list, gpu_id, log_path)
            timing["sample_idx"] = i
            all_timings.append(timing)

    threads = []
    t_parallel_start = time.time()
    for step_name, gpu_id in STEP_TO_GPU.items():
        t = threading.Thread(target=_run_for_all_samples, args=(step_name, gpu_id), daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    total_parallel = time.time() - t_parallel_start
    return {"mode": "parallel", "total_s": total_parallel, "timings": all_timings}


def benchmark_sequential(
    video_paths: list[str],
    frames_dirs: list[str],
    gt_camera_txts: list[str],
    intermediates_dirs: list[str],
    prompts: list[str],
    logs_dir: str,
    gpu_id: int = 0,
) -> dict:
    """单卡顺序运行所有 reward 步骤，测量总时间（对照组）。"""
    os.makedirs(logs_dir, exist_ok=True)
    all_timings = []
    t_seq_start = time.time()

    for i, (frames_dir, inter_dir, vp, prompt) in enumerate(
            zip(frames_dirs, intermediates_dirs, video_paths, prompts)):
        Path(inter_dir).mkdir(parents=True, exist_ok=True)
        da3_npz = os.path.join(inter_dir, "da3_output_seq.npz")
        label_maps_npz = os.path.join(inter_dir, "label_maps_seq.npz")
        dinov2_npz = os.path.join(inter_dir, "dinov2_features_seq.npz")
        videoalign_json = os.path.join(inter_dir, "videoalign_seq.json")
        objects_json = os.path.join(inter_dir, "objects_seq.json")

        for step_name in ["step_da3", "step_qwen_sam3", "step_dinov2_extract", "step_videoalign"]:
            log_path = os.path.join(logs_dir, f"seq_{step_name}_sample{i}.log")

            if step_name == "step_da3":
                args_list = ["--video_frames_dir", frames_dir,
                             "--output", da3_npz, "--gpu", "0"]
            elif step_name == "step_qwen_sam3":
                args_list = ["--video_frames_dir", frames_dir,
                             "--output", label_maps_npz,
                             "--objects_output", objects_json, "--gpu", "0"]
            elif step_name == "step_dinov2_extract":
                args_list = ["--video_frames_dir", frames_dir,
                             "--output", dinov2_npz, "--gpu", "0"]
            elif step_name == "step_videoalign":
                args_list = ["--video_path", vp,
                             "--prompt", prompt,
                             "--output", videoalign_json, "--gpu", "0"]

            timing = run_step_timed(step_name, args_list, gpu_id, log_path)
            timing["sample_idx"] = i
            all_timings.append(timing)

    total_sequential = time.time() - t_seq_start
    return {"mode": "sequential_single_gpu", "total_s": total_sequential, "timings": all_timings}


def main():
    parser = argparse.ArgumentParser(description="Reward 速度 Benchmark")
    parser.add_argument("--infer_dir", type=str, required=True,
                        help="已生成的视频目录；每个样本一个子目录，含 gen_0.mp4...gen_7.mp4 + camera.txt")
    parser.add_argument("--samples", nargs="+", required=True,
                        help="相对于 infer_dir 的样本路径列表，如: re10k/sample_0001 re10k/sample_0002")
    parser.add_argument("--rollout_idx", type=int, default=0,
                        help="每个样本使用第 rollout_idx 条 rollout 做 benchmark")
    parser.add_argument("--output_dir", type=str, default="./results/benchmark_run",
                        help="benchmark 输出目录")
    parser.add_argument("--skip_sequential", action="store_true",
                        help="跳过顺序执行对照组（节省时间）")
    parser.add_argument("--default_prompt", type=str,
                        default="camera moving through a scene")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    infer_dir = Path(args.infer_dir)
    video_paths, frames_dirs, gt_camera_txts, intermediates_dirs, prompts = [], [], [], [], []

    for sample_rel in args.samples:
        sample_dir = infer_dir / sample_rel
        video_path = str(sample_dir / f"gen_{args.rollout_idx}.mp4")
        if not os.path.isfile(video_path):
            print(f"WARNING: {video_path} not found, skipping")
            continue
        gt_cam = str(sample_dir / "camera.txt")
        frames_dir = str(out_dir / "frames" / sample_rel.replace("/", "_"))
        inter_dir = str(out_dir / "intermediates" / sample_rel.replace("/", "_"))

        # 加载 prompt
        meta_path = sample_dir / "metadata.json"
        if meta_path.exists():
            with open(str(meta_path)) as f:
                meta = json.load(f)
            prompt = meta.get("caption", meta.get("prompt", args.default_prompt))
        else:
            prompt = args.default_prompt

        print(f"[Benchmark] Sample: {sample_rel}  video={video_path}")
        print(f"  Extracting frames...")
        _extract_frames_cv2(video_path, frames_dir)

        video_paths.append(video_path)
        frames_dirs.append(frames_dir)
        gt_camera_txts.append(gt_cam)
        intermediates_dirs.append(inter_dir)
        prompts.append(prompt)

    if not video_paths:
        print("No valid samples found. Exiting.")
        sys.exit(1)

    print(f"\n[Benchmark] Running PARALLEL mode ({len(video_paths)} samples)...")
    parallel_result = benchmark_parallel(
        video_paths, frames_dirs, gt_camera_txts, intermediates_dirs, prompts,
        logs_dir=str(out_dir / "logs_parallel"),
    )
    print(f"[Benchmark] PARALLEL total: {parallel_result['total_s']:.1f}s")

    sequential_result = None
    if not args.skip_sequential:
        print(f"\n[Benchmark] Running SEQUENTIAL mode (single GPU=0)...")
        sequential_result = benchmark_sequential(
            video_paths, frames_dirs, gt_camera_txts, intermediates_dirs, prompts,
            logs_dir=str(out_dir / "logs_sequential"),
            gpu_id=0,
        )
        print(f"[Benchmark] SEQUENTIAL total: {sequential_result['total_s']:.1f}s")

    # ── 输出报告 ──────────────────────────────────────────────────────────────
    report = {
        "num_samples": len(video_paths),
        "rollout_idx": args.rollout_idx,
        "parallel": {
            "total_s": parallel_result["total_s"],
            "per_step_breakdown": _summarize_timings(parallel_result["timings"]),
        },
    }
    if sequential_result:
        report["sequential_single_gpu"] = {
            "total_s": sequential_result["total_s"],
            "per_step_breakdown": _summarize_timings(sequential_result["timings"]),
        }
        speedup = sequential_result["total_s"] / max(parallel_result["total_s"], 0.001)
        report["speedup"] = speedup
        print(f"\n[Benchmark] Speedup: {speedup:.2f}x  "
              f"(parallel {parallel_result['total_s']:.1f}s vs "
              f"sequential {sequential_result['total_s']:.1f}s)")

    report_path = out_dir / "benchmark_report.json"
    with open(str(report_path), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[Benchmark] Report saved to {report_path}")
    _print_summary(report)


def _summarize_timings(timings: list[dict]) -> dict:
    from collections import defaultdict
    step_times: dict[str, list[float]] = defaultdict(list)
    for t in timings:
        step_times[t["step"]].append(t["elapsed_s"])
    return {
        step: {
            "mean_s": sum(v) / len(v),
            "total_s": sum(v),
            "n_samples": len(v),
        }
        for step, v in step_times.items()
    }


def _print_summary(report: dict):
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Samples: {report['num_samples']}")
    if "parallel" in report:
        print(f"\nParallel (4 GPUs): {report['parallel']['total_s']:.1f}s total")
        for step, info in report["parallel"]["per_step_breakdown"].items():
            print(f"  {step:<30s}: {info['mean_s']:.1f}s/sample  {info['total_s']:.1f}s total")
    if "sequential_single_gpu" in report:
        print(f"\nSequential (1 GPU): {report['sequential_single_gpu']['total_s']:.1f}s total")
    if "speedup" in report:
        print(f"\nSpeedup: {report['speedup']:.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()
