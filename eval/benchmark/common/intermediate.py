#!/usr/bin/env python3
"""
intermediate.py — 中间产物调度器。

管理 DA3 / SAM3 / DINOv2 / VideoAlign / VBench 的中间产物生成与缓存，
按依赖顺序执行，跳过已存在的文件。
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from .utils import log, run_conda, env_python, save_json, extract_frames, get_prompt

_THIRD_PARTY_DIR = Path(__file__).resolve().parent.parent.parent.parent / "third_party"
_REWARD_CODE = _THIRD_PARTY_DIR / "reward_code"
_WORKERS_DIR = _THIRD_PARTY_DIR / "workers"

METRIC_TO_DEPS = {
    "video_quality.psnr":                [],
    "video_quality.vbench":              ["vbench"],
    "reward.camera_pose":                ["da3"],
    "reward.depth_reprojection.object":  ["da3", "sam3"],
    "reward.depth_reprojection.global":  ["da3"],
    "reward.depth_reprojection.both":    ["da3", "sam3"],
    "reward.videoalign":                 ["videoalign"],
    "reward.feature_sim":                ["da3", "dinov2"],
    "reconstruction.global":             ["da3"],
    "reconstruction.object":             ["da3", "sam3"],
    "reconstruction.both":               ["da3", "sam3"],
}

EXECUTION_ORDER = ["da3", "sam3", "dinov2", "videoalign", "vbench"]


class IntermediateManager:
    """按需生成中间产物，跳过已完成的。"""

    def __init__(self, gpu: int = 0, vbench_cache: str = None):
        self.gpu = gpu
        self.vbench_cache = vbench_cache or self._find_vbench_cache()

    @staticmethod
    def _find_vbench_cache() -> str:
        _rl_code_dir = _THIRD_PARTY_DIR.parent
        model_root = Path(os.environ.get("RL_MODEL_ROOT", str(_rl_code_dir / "model")))
        candidates = [
            model_root / "vbench_cache",
            _rl_code_dir / "model" / "vbench_cache",
            Path.home() / "WAN_TEST" / "model" / "vbench_cache",
        ]
        return str(next(
            (p for p in candidates if (p / "dreamsim_cache").exists()),
            candidates[0],
        ))

    def resolve_deps(self, metrics: list) -> set:
        """从指标列表解析需要的中间产物集合。"""
        needed = set()
        for m in metrics:
            deps = METRIC_TO_DEPS.get(m, [])
            needed.update(deps)
        return needed

    def prepare(self, needed_deps: set, entries: list, tmp_dir: Path):
        """按固定顺序生成所需中间产物。"""
        tmp_dir.mkdir(parents=True, exist_ok=True)
        for dep in EXECUTION_ORDER:
            if dep not in needed_deps:
                continue
            log(f"中间产物: {dep}")
            handler = getattr(self, f"_run_{dep}", None)
            if handler is None:
                log(f"  [跳过] 未实现: {dep}")
                continue
            handler(entries, tmp_dir)

    # ────────────────────── DA3 ──────────────────────────────────

    def _run_da3(self, entries: list, tmp_dir: Path):
        step_da3 = str(_REWARD_CODE / "step_da3.py")
        # DA3 依赖 moviepy/omegaconf 等，使用专用 rl_da3 环境
        py = env_python("rl_da3")

        for entry in entries:
            sample_id = entry["sample_id"]
            gt_inter = entry["sample_dir"] / "gt_intermediates"
            gt_frames_dir = gt_inter / "frames"
            gt_da3_npz = gt_inter / "da3_gt.npz"

            if not gt_da3_npz.exists():
                log(f"  DA3 GT: {sample_id}")
                gt_inter.mkdir(parents=True, exist_ok=True)
                extract_frames(entry["gt_video"], str(gt_frames_dir))
                cmd = [py, "-u", step_da3,
                       "--video_frames_dir", str(gt_frames_dir),
                       "--output", str(gt_da3_npz),
                       "--gpu", str(self.gpu)]
                subprocess.run(cmd, check=True)

            for gv in entry["gen_videos"]:
                inter_dir = gv["gen_dir"] / "intermediates"
                frames_dir = inter_dir / "frames"
                da3_npz = inter_dir / "da3_pred.npz"

                if not da3_npz.exists():
                    log(f"  DA3 pred: {sample_id}/{Path(gv['video_path']).name}")
                    inter_dir.mkdir(parents=True, exist_ok=True)
                    extract_frames(gv["video_path"], str(frames_dir))
                    cmd = [py, "-u", step_da3,
                           "--video_frames_dir", str(frames_dir),
                           "--output", str(da3_npz),
                           "--gpu", str(self.gpu)]
                    subprocess.run(cmd, check=True)

    # ────────────────────── SAM3 ─────────────────────────────────

    def _run_sam3(self, entries: list, tmp_dir: Path):
        batch = []
        for entry in entries:
            gt_inter = entry["sample_dir"] / "gt_intermediates"
            batch.append({
                "video_path":             entry["gt_video"],
                "output_masks_npz":       str(gt_inter / "gt_masks.npz"),
                "output_objects_json":    str(gt_inter / "gt_objects.json"),
                "output_label_maps_npz":  str(gt_inter / "gt_label_maps.npz"),
                "ref_objects_json":       None,
                "is_gt":                  True,
                "max_frames":             0,
            })

        for entry in entries:
            gt_objects = str(entry["sample_dir"] / "gt_intermediates" / "gt_objects.json")
            for gv in entry["gen_videos"]:
                inter_dir = gv["gen_dir"] / "intermediates"
                batch.append({
                    "video_path":             gv["video_path"],
                    "output_masks_npz":       str(inter_dir / "pred_masks.npz"),
                    "output_objects_json":    str(inter_dir / "pred_objects.json"),
                    "output_label_maps_npz":  str(inter_dir / "label_maps.npz"),
                    "ref_objects_json":       gt_objects,
                    "is_gt":                  False,
                    "max_frames":             0,
                })

        manifest_path = tmp_dir / "sam3_batch.json"
        with open(str(manifest_path), "w") as f:
            json.dump(batch, f, ensure_ascii=False, indent=2)

        run_conda(
            env="SAM3",
            script=str(_WORKERS_DIR / "worker_sam3.py"),
            extra_args=["--batch_manifest", str(manifest_path),
                        "--sam3_gpu", "0", "--qwen_gpu", "0"],
        )

    # ────────────────────── DINOv2 ───────────────────────────────

    def _run_dinov2(self, entries: list, tmp_dir: Path):
        sys.path.insert(0, str(_REWARD_CODE))
        from step_dinov2_featup import load_model, compute_feature_similarity
        import torch

        device = f"cuda:{self.gpu}"
        model, mode = load_model(device=device)

        for entry in entries:
            for gv in entry["gen_videos"]:
                inter_dir = gv["gen_dir"] / "intermediates"
                feat_json = inter_dir / "feature_sim_reward.json"
                da3_npz = inter_dir / "da3_pred.npz"
                frames_dir = inter_dir / "frames"

                if feat_json.exists() or not da3_npz.exists():
                    continue

                frame_paths = sorted(str(p) for p in frames_dir.glob("frame_*.png"))
                if not frame_paths:
                    continue

                try:
                    da3_data = dict(np.load(str(da3_npz), allow_pickle=True))
                    reward, details = compute_feature_similarity(
                        model, frame_paths, da3_data, device=device,
                    )
                    save_json(str(feat_json), {
                        "reward_feature_sim": reward,
                        "mode": mode,
                        "details": details,
                    })
                    log(f"  feat_sim={reward:.4f}  {Path(gv['video_path']).name}")
                except Exception as e:
                    log(f"  [错误] DINOv2 {Path(gv['video_path']).name}: {e}")

        del model
        torch.cuda.empty_cache()

    # ────────────────────── VideoAlign ────────────────────────────

    def _run_videoalign(self, entries: list, tmp_dir: Path):
        batch = []
        for entry in entries:
            prompt = get_prompt(entry)
            for gv in entry["gen_videos"]:
                inter_dir = gv["gen_dir"] / "intermediates"
                batch.append({
                    "video_path":  gv["video_path"],
                    "prompt":      prompt,
                    "output_json": str(inter_dir / "videoalign.json"),
                    "skip_done":   True,
                })

        manifest_path = tmp_dir / "videoalign_batch.json"
        with open(str(manifest_path), "w") as f:
            json.dump(batch, f, ensure_ascii=False, indent=2)

        run_conda(
            env="Videoalign",
            script=str(_WORKERS_DIR / "worker_videoalign.py"),
            extra_args=["--batch_manifest", str(manifest_path),
                        "--gpu", str(self.gpu)],
        )

    # ────────────────────── VBench ───────────────────────────────

    def _run_vbench(self, entries: list, tmp_dir: Path):
        device = f"cuda:{self.gpu}"
        vbench_tmp = tmp_dir / "vbench_inputs"

        vbench_entries = []
        for entry in entries:
            if not entry.get("start_png"):
                continue
            for gv in entry["gen_videos"]:
                sid = f"{entry['dataset']}__{entry['sample_id']}__gen_{gv['idx']}"
                vbench_entries.append({
                    "sample_id":   sid,
                    "img_path":    entry["start_png"],
                    "video_path":  gv["video_path"],
                    "output_json": str(gv["gen_dir"] / "intermediates" / "vbench.json"),
                    "skip_done":   True,
                })

        if not vbench_entries:
            log("  无有效 VBench 条目")
            return

        manifest = {
            "entries":      vbench_entries,
            "vbench_cache": self.vbench_cache,
            "device":       device,
            "tmp_dir":      str(vbench_tmp),
        }
        manifest_path = tmp_dir / "vbench_batch.json"
        with open(str(manifest_path), "w") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        run_conda(
            env="vbench",
            script=str(_WORKERS_DIR / "worker_vbench.py"),
            extra_args=["--batch_manifest", str(manifest_path)],
        )
