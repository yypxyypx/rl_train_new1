#!/usr/bin/env python3
"""
reward_pipeline.py
==================
Reward pipeline scheduler.

Automatically infers which model-inference steps are needed based on
--rewards, avoids calling unnecessary models, supports caching and
resume from intermediate results.

Dependencies:
  geo_semantic  -> step_da3, step_qwen_sam3
  geo_global    -> step_da3
  feature_sim   -> step_da3, step_dinov2_featup
  camera_traj   -> step_da3
  video_quality -> step_videoalign
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

_REWARD_DIR = Path(__file__).resolve().parent
_STEPS_DIR = _REWARD_DIR / "steps"
_RL_CODE_DIR = _REWARD_DIR.parent.parent

REWARD_TO_STEPS = {
    "geo_semantic":  ["step_da3", "step_qwen_sam3"],
    "geo_global":    ["step_da3"],
    # feature_sim 现在只依赖 step_dinov2_extract（不再依赖 DA3）。
    # warping 计算在 compute_all_rewards 聚合阶段完成，届时 da3_output.npz 也已就绪。
    "feature_sim":   ["step_dinov2_extract"],
    "camera_traj":   ["step_da3"],
    "video_quality": ["step_videoalign"],
}

STEP_CONDA_ENV = {
    "step_da3":              "rl_da3",
    "step_qwen_sam3":        "SAM3",
    "step_dinov2_extract":   "rl_da3",   # 新：只提取特征，不做 warping
    "step_dinov2_featup":    "rl_da3",   # 旧：保留作为独立工具
    "step_videoalign":       "Videoalign",
}

STEP_ORDER = [
    "step_da3", "step_qwen_sam3",
    "step_dinov2_extract", "step_dinov2_featup", "step_videoalign",
]
ALL_REWARDS = [
    "geo_semantic", "geo_global", "feature_sim",
    "camera_traj", "video_quality",
]


def resolve_steps(rewards: list) -> list:
    needed = set()
    for r in rewards:
        if r not in REWARD_TO_STEPS:
            raise ValueError(
                f"Unknown reward: {r}. "
                f"Valid: {list(REWARD_TO_STEPS.keys())}")
        needed.update(REWARD_TO_STEPS[r])
    return [s for s in STEP_ORDER if s in needed]


def _env_python(env: str) -> str:
    candidates = [
        f"/opt/conda/envs/{env}/bin/python",
        os.path.expanduser(f"~/miniconda3/envs/{env}/bin/python"),
        os.path.expanduser(f"~/anaconda3/envs/{env}/bin/python"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return f"/opt/conda/envs/{env}/bin/python"


def extract_frames(video_path: str, out_dir: str) -> list:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    existing = sorted(out.glob("frame_*.png"))
    if existing:
        return [str(p) for p in existing]
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
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
    return paths


def get_video_size(video_path: str) -> tuple:
    cap = cv2.VideoCapture(str(video_path))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    return H, W


def get_prompt(metadata_path: str) -> str:
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as f:
                meta = json.load(f)
            for key in ("prompt", "caption", "description", "text"):
                if key in meta:
                    return str(meta[key])
        except Exception:
            pass
    return "camera moving through a scene"


def run_step(step_name: str, args_list: list, gpu: int = 0):
    env_name = STEP_CONDA_ENV[step_name]
    py = _env_python(env_name)
    script = str(_STEPS_DIR / f"{step_name}.py")
    cmd = [py, "-u", script] + args_list
    print(f"\n{'='*60}")
    print(f"[Pipeline] Running {step_name} (env={env_name})")
    print(f"[Pipeline] CMD: {' '.join(cmd)}")
    print(f"{'='*60}")
    env_vars = dict(os.environ)
    env_vars["CUDA_VISIBLE_DEVICES"] = str(gpu)
    result = subprocess.run(cmd, env=env_vars)
    if result.returncode != 0:
        raise RuntimeError(
            f"{step_name} failed with code {result.returncode}")
    print(f"[Pipeline] {step_name} done")


def run_pipeline(
    video_path: str, gt_camera_txt: str, work_dir: str,
    rewards: list, gpu: int = 0, prompt: str = None,
    metadata_json: str = None, skip_done: bool = True,
    conf_threshold: float = 0.0,
    geo_compare_mode: str = "first_frame",
    feature_compare_mode: str = "first_frame",
):
    work = Path(work_dir)
    frames_dir = work / "frames"
    intermediates = work / "intermediates"
    intermediates.mkdir(parents=True, exist_ok=True)

    da3_npz = intermediates / "da3_output.npz"
    label_maps_npz = intermediates / "label_maps.npz"
    objects_json = intermediates / "objects.json"
    feature_sim_json = intermediates / "feature_sim_reward.json"   # 旧流程兼容
    dinov2_features_npz = intermediates / "dinov2_features.npz"    # 新流程
    videoalign_json = intermediates / "videoalign.json"

    steps = resolve_steps(rewards)
    print(f"[Pipeline] Rewards: {rewards}")
    print(f"[Pipeline] Steps needed: {steps}")
    print(f"[Pipeline] conf_threshold={conf_threshold}  "
          f"geo_mode={geo_compare_mode}  feat_mode={feature_compare_mode}")

    H, W = get_video_size(video_path)
    print(f"[Pipeline] Video: {video_path}  ({H}x{W})")

    need_frames = any(
        s in steps
        for s in ["step_da3", "step_qwen_sam3",
                  "step_dinov2_extract", "step_dinov2_featup"]
    )
    if need_frames:
        frame_paths = extract_frames(video_path, str(frames_dir))
        print(f"[Pipeline] Extracted {len(frame_paths)} frames")

    for step in steps:
        if step == "step_da3":
            if skip_done and da3_npz.exists():
                print(f"[Pipeline] Skip {step}: output exists")
                continue
            run_step(step, [
                "--video_frames_dir", str(frames_dir),
                "--output", str(da3_npz), "--gpu", "0",
            ], gpu)

        elif step == "step_qwen_sam3":
            if skip_done and label_maps_npz.exists():
                print(f"[Pipeline] Skip {step}: output exists")
                continue
            extra = [
                "--video_frames_dir", str(frames_dir),
                "--output", str(label_maps_npz),
                "--objects_output", str(objects_json), "--gpu", "0",
            ]
            if objects_json.exists():
                extra += ["--objects_json", str(objects_json)]
            run_step(step, extra, gpu)

        elif step == "step_dinov2_extract":
            # 新流程：只提取特征，不依赖 DA3（与 DA3 完全并行）
            if skip_done and dinov2_features_npz.exists():
                print(f"[Pipeline] Skip {step}: output exists")
                continue
            run_step(step, [
                "--video_frames_dir", str(frames_dir),
                "--output", str(dinov2_features_npz), "--gpu", "0",
            ], gpu)

        elif step == "step_dinov2_featup":
            # 旧流程：保留，输出 feature_sim_reward.json（包含 warping + 相似度）
            if skip_done and feature_sim_json.exists():
                print(f"[Pipeline] Skip {step}: output exists")
                continue
            run_step(step, [
                "--video_frames_dir", str(frames_dir),
                "--da3_output", str(da3_npz),
                "--output", str(feature_sim_json), "--gpu", "0",
                "--compare_mode", feature_compare_mode,
            ], gpu)

        elif step == "step_videoalign":
            if skip_done and videoalign_json.exists():
                print(f"[Pipeline] Skip {step}: output exists")
                continue
            vid_prompt = prompt
            if vid_prompt is None and metadata_json:
                vid_prompt = get_prompt(metadata_json)
            if vid_prompt is None:
                vid_prompt = "camera moving through a scene"
            run_step(step, [
                "--video_path", str(video_path),
                "--prompt", vid_prompt,
                "--output", str(videoalign_json), "--gpu", "0",
            ], gpu)

    print(f"\n{'='*60}")
    print(f"[Pipeline] Computing reward metrics...")
    print(f"{'='*60}")

    sys.path.insert(0, str(_REWARD_DIR))
    from reward_metrics import compute_all_rewards

    result = compute_all_rewards(
        da3_path=str(da3_npz),
        label_maps_path=str(label_maps_npz),
        feature_sim_path=str(feature_sim_json),
        dinov2_features_path=str(dinov2_features_npz),
        videoalign_path=str(videoalign_json),
        gt_camera_txt=gt_camera_txt,
        H_img=H, W_img=W,
        rewards_to_compute=rewards, device="cuda",
        conf_threshold=conf_threshold,
        geo_compare_mode=geo_compare_mode,
        feature_compare_mode=feature_compare_mode,
    )

    output_json = work / "reward.json"
    serializable = {k: v for k, v in result.items() if k != "details"}
    with open(str(output_json), "w") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2,
                  default=str)
    print(f"\n[Pipeline] Reward saved to {output_json}")

    details_json = work / "reward_details.json"
    with open(str(details_json), "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    return result


def run_batch_pipeline(
    sample_dir: str,
    output_dir: str,
    rewards: list,
    gpu: int = 0,
    prompt: str = None,
    skip_done: bool = True,
    conf_threshold: float = 0.0,
    geo_compare_mode: str = "first_frame",
    feature_compare_mode: str = "first_frame",
):
    """
    Run full reward pipeline on all rollouts in a sample directory.

    sample_dir structure:
        sample_dir/
            infer_info.json
            camera.txt
            gen_0.mp4 ... gen_N.mp4

    Per-rollout work_dir:
        output_dir/<dataset>/<sample_id>/gen_k/

    Summary:
        output_dir/<dataset>/<sample_id>/summary.json
    """
    sample = Path(sample_dir)
    infer_info_path = sample / "infer_info.json"
    if not infer_info_path.exists():
        raise FileNotFoundError(
            f"infer_info.json not found in {sample_dir}")

    with open(str(infer_info_path), "r") as f:
        infer_info = json.load(f)

    num_rollouts = infer_info.get("num_rollouts", 0)
    dataset = infer_info.get("dataset", "unknown")
    sample_id = infer_info.get("sample_id", sample.name)

    gt_camera_txt = str(sample / "camera.txt")
    if not os.path.isfile(gt_camera_txt):
        raise FileNotFoundError(
            f"camera.txt not found: {gt_camera_txt}")

    batch_out = Path(output_dir) / dataset / sample_id
    batch_out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[Batch] dataset={dataset}  sample_id={sample_id}")
    print(f"[Batch] num_rollouts={num_rollouts}  rewards={rewards}")
    print(f"[Batch] output_dir={batch_out}")
    print(f"[Batch] conf_threshold={conf_threshold}  "
          f"geo_mode={geo_compare_mode}  feat_mode={feature_compare_mode}")
    print(f"{'='*60}\n")

    rollout_results = []
    for k in range(num_rollouts):
        video_path = str(sample / f"gen_{k}.mp4")
        if not os.path.isfile(video_path):
            print(f"[Batch] WARNING: gen_{k}.mp4 not found, skipping")
            continue

        work_dir = str(batch_out / f"gen_{k}")
        print(f"\n[Batch] ===== Rollout gen_{k} "
              f"({k+1}/{num_rollouts}) =====")

        result = run_pipeline(
            video_path=video_path,
            gt_camera_txt=gt_camera_txt,
            work_dir=work_dir,
            rewards=rewards,
            gpu=gpu,
            prompt=prompt,
            metadata_json=None,
            skip_done=skip_done,
            conf_threshold=conf_threshold,
            geo_compare_mode=geo_compare_mode,
            feature_compare_mode=feature_compare_mode,
        )

        scalar = {k2: v for k2, v in result.items() if k2 != "details"}
        scalar["gen_id"] = k
        rollout_results.append(scalar)
        print(f"[Batch] gen_{k} reward_total="
              f"{result.get('reward_total', float('nan')):.4f}")

    summary = {
        "sample_id": sample_id,
        "dataset": dataset,
        "num_rollouts": num_rollouts,
        "rewards": rewards,
        "conf_threshold": conf_threshold,
        "geo_compare_mode": geo_compare_mode,
        "feature_compare_mode": feature_compare_mode,
        "rollouts": rollout_results,
    }
    summary_path = batch_out / "summary.json"
    with open(str(summary_path), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n[Batch] Done. Summary written to {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Reward Pipeline")
    parser.add_argument("--mode", type=str, default="single",
                        choices=["single", "batch"])

    # single mode args
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--gt_camera_txt", type=str, default=None)
    parser.add_argument("--work_dir", type=str, default=None)

    # batch mode args
    parser.add_argument("--sample_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

    # shared args
    parser.add_argument("--rewards", type=str, default="all")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--metadata_json", type=str, default=None)
    parser.add_argument("--no_skip", action="store_true")

    # advanced args
    parser.add_argument("--conf_threshold", type=float, default=0.0,
                        help="DA3 depth confidence threshold")
    parser.add_argument("--geo_compare_mode", type=str,
                        default="first_frame",
                        choices=["first_frame", "adjacent", "all_pairs"],
                        help="Comparison mode for geo rewards")
    parser.add_argument("--feature_compare_mode", type=str,
                        default="first_frame",
                        choices=["first_frame", "adjacent", "all_pairs"],
                        help="Comparison mode for DINOv2 feature sim")

    args = parser.parse_args()

    if args.rewards == "all":
        rewards = ALL_REWARDS
    else:
        rewards = [r.strip() for r in args.rewards.split(",")]

    if args.mode == "batch":
        if not args.sample_dir or not args.output_dir:
            parser.error(
                "--mode batch requires --sample_dir and --output_dir")
        run_batch_pipeline(
            sample_dir=args.sample_dir,
            output_dir=args.output_dir,
            rewards=rewards,
            gpu=args.gpu,
            prompt=args.prompt,
            skip_done=not args.no_skip,
            conf_threshold=args.conf_threshold,
            geo_compare_mode=args.geo_compare_mode,
            feature_compare_mode=args.feature_compare_mode,
        )
    else:
        if (not args.video_path or not args.gt_camera_txt
                or not args.work_dir):
            parser.error(
                "--mode single requires "
                "--video_path, --gt_camera_txt, --work_dir")
        run_pipeline(
            video_path=args.video_path,
            gt_camera_txt=args.gt_camera_txt,
            work_dir=args.work_dir,
            rewards=rewards,
            gpu=args.gpu,
            prompt=args.prompt,
            metadata_json=args.metadata_json,
            skip_done=not args.no_skip,
            conf_threshold=args.conf_threshold,
            geo_compare_mode=args.geo_compare_mode,
            feature_compare_mode=args.feature_compare_mode,
        )


if __name__ == "__main__":
    main()
