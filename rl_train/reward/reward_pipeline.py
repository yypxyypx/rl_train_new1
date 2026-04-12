#!/usr/bin/env python3
"""
reward_pipeline.py
==================
Reward 智能调度器。

根据 --rewards 参数自动推断需要执行的模型推理 steps，
避免调用不需要的模型，支持中间结果缓存和断点续算。

依赖关系：
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
    "feature_sim":   ["step_da3", "step_dinov2_featup"],
    "camera_traj":   ["step_da3"],
    "video_quality": ["step_videoalign"],
}

STEP_CONDA_ENV = {
    "step_da3":            "rl_da3",
    "step_qwen_sam3":      "SAM3",
    "step_dinov2_featup":  "rl_da3",
    "step_videoalign":     "Videoalign",
}

STEP_ORDER = ["step_da3", "step_qwen_sam3", "step_dinov2_featup", "step_videoalign"]
ALL_REWARDS = ["geo_semantic", "geo_global", "feature_sim", "camera_traj", "video_quality"]


def resolve_steps(rewards: list) -> list:
    """根据选定的 rewards 推断需要执行的 steps（去重、按顺序）。"""
    needed = set()
    for r in rewards:
        if r not in REWARD_TO_STEPS:
            raise ValueError(f"Unknown reward: {r}. Valid: {list(REWARD_TO_STEPS.keys())}")
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
        raise RuntimeError(f"{step_name} failed with code {result.returncode}")
    print(f"[Pipeline] {step_name} done")


def run_pipeline(
    video_path: str, gt_camera_txt: str, work_dir: str,
    rewards: list, gpu: int = 0, prompt: str = None,
    metadata_json: str = None, skip_done: bool = True,
):
    work = Path(work_dir)
    frames_dir = work / "frames"
    intermediates = work / "intermediates"
    intermediates.mkdir(parents=True, exist_ok=True)

    da3_npz = intermediates / "da3_output.npz"
    label_maps_npz = intermediates / "label_maps.npz"
    objects_json = intermediates / "objects.json"
    feature_sim_json = intermediates / "feature_sim_reward.json"
    videoalign_json = intermediates / "videoalign.json"

    steps = resolve_steps(rewards)
    print(f"[Pipeline] Rewards: {rewards}")
    print(f"[Pipeline] Steps needed: {steps}")

    H, W = get_video_size(video_path)
    print(f"[Pipeline] Video: {video_path}  ({H}x{W})")

    need_frames = any(s in steps for s in ["step_da3", "step_qwen_sam3", "step_dinov2_featup"])
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
            extra = ["--video_frames_dir", str(frames_dir),
                      "--output", str(label_maps_npz),
                      "--objects_output", str(objects_json), "--gpu", "0"]
            if objects_json.exists():
                extra += ["--objects_json", str(objects_json)]
            run_step(step, extra, gpu)

        elif step == "step_dinov2_featup":
            if skip_done and feature_sim_json.exists():
                print(f"[Pipeline] Skip {step}: output exists")
                continue
            run_step(step, [
                "--video_frames_dir", str(frames_dir),
                "--da3_output", str(da3_npz),
                "--output", str(feature_sim_json), "--gpu", "0",
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
                "--video_path", str(video_path), "--prompt", vid_prompt,
                "--output", str(videoalign_json), "--gpu", "0",
            ], gpu)

    print(f"\n{'='*60}")
    print(f"[Pipeline] Computing reward metrics...")
    print(f"{'='*60}")

    sys.path.insert(0, str(_REWARD_DIR))
    from reward_metrics import compute_all_rewards

    result = compute_all_rewards(
        da3_path=str(da3_npz), label_maps_path=str(label_maps_npz),
        feature_sim_path=str(feature_sim_json), videoalign_path=str(videoalign_json),
        gt_camera_txt=gt_camera_txt, H_img=H, W_img=W,
        rewards_to_compute=rewards, device="cuda",
    )

    output_json = work / "reward.json"
    serializable = {k: v for k, v in result.items() if k != "details"}
    with open(str(output_json), "w") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[Pipeline] Reward saved to {output_json}")

    details_json = work / "reward_details.json"
    with open(str(details_json), "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    return result


def main():
    parser = argparse.ArgumentParser(description="Reward Pipeline")
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--gt_camera_txt", required=True)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--rewards", type=str, default="all",
                        help="Comma-separated reward names or 'all'")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--metadata_json", type=str, default=None)
    parser.add_argument("--no_skip", action="store_true",
                        help="Re-run all steps even if outputs exist")
    args = parser.parse_args()

    if args.rewards == "all":
        rewards = ALL_REWARDS
    else:
        rewards = [r.strip() for r in args.rewards.split(",")]

    run_pipeline(
        video_path=args.video_path, gt_camera_txt=args.gt_camera_txt,
        work_dir=args.work_dir, rewards=rewards, gpu=args.gpu,
        prompt=args.prompt, metadata_json=args.metadata_json,
        skip_done=not args.no_skip,
    )


if __name__ == "__main__":
    main()
