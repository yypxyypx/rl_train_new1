"""reward_dispatcher.py — 分布式 Reward 调度器。

将一批 rollout 视频的 reward 计算分配到不同 GPU 组，四组完全并行：
  GPU A: DA3            → da3_output.npz
  GPU B: Qwen+SAM3      → label_maps.npz
  GPU C: DINOv2 Extract → dinov2_features.npz
  GPU D: VideoAlign     → videoalign.json

各组结束后，CPU 端运行 compute_all_rewards() 聚合。

通信方式：共享文件系统（npz/json）+ 完成标记文件（.{step}_done）。

用法：
  from reward_dispatcher import RewardDispatcher
  dispatcher = RewardDispatcher(args)
  rewards = dispatcher.run(video_paths, gt_camera_paths, work_dirs, prompts)
  # rewards: list[float]
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
_REWARD_DIR = _HERE.parent.parent / "reward"
_STEPS_DIR = _REWARD_DIR / "steps"

STEP_CONDA_ENV = {
    "step_da3":            "rl_da3",
    "step_qwen_sam3":      "SAM3",
    "step_qwen":           "SAM3",
    "step_sam3":           "SAM3",
    "step_dinov2_extract": "rl_da3",
    "step_videoalign":     "Videoalign",
}

# centralized 模式默认每个 reward 组绑一张 GPU 的 local index（CUDA_VISIBLE_DEVICES 切片后）
CENTRALIZED_GPU_ASSIGNMENT = {
    "da3":            0,
    "dinov2_extract": 1,
    "qwen_sam3":      2,
    "videoalign":     3,
}

# 四组 GPU 分配（默认 4 卡方案）
DEFAULT_4GPU_ASSIGNMENT = {
    "da3":            [0],
    "qwen_sam3":      [1],
    "dinov2_extract": [2],
    "videoalign":     [3],
}

# 16 卡方案（每组 4 张卡，每张负责独立的 rollout，无 GPU 内部并行）
DEFAULT_16GPU_ASSIGNMENT = {
    "da3":            [0, 1, 2, 3],
    "qwen_sam3":      [4, 5, 6, 7],
    "dinov2_extract": [8, 9, 10, 11],
    "videoalign":     [12, 13, 14, 15],
}

ALL_REWARDS = ["geo_semantic", "geo_global", "feature_sim", "camera_traj", "video_quality"]

REWARD_TO_GROUPS = {
    "geo_semantic":  {"da3", "qwen_sam3"},
    "geo_global":    {"da3"},
    "feature_sim":   {"da3", "dinov2_extract"},
    "camera_traj":   {"da3"},
    "video_quality": {"videoalign"},
}


def _resolve_gpu_assignment(args) -> dict:
    """从 args.reward_gpu_assignment（JSON 字符串）或 GPU 数量自动选方案。"""
    if getattr(args, "reward_gpu_assignment", None):
        return json.loads(args.reward_gpu_assignment)

    import torch
    n_gpu = torch.cuda.device_count()
    if n_gpu >= 16:
        return DEFAULT_16GPU_ASSIGNMENT
    elif n_gpu >= 4:
        return DEFAULT_4GPU_ASSIGNMENT
    else:
        # 单卡回退：所有步骤跑在 GPU 0
        return {"da3": [0], "qwen_sam3": [0], "dinov2_extract": [0], "videoalign": [0]}


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


def _run_step_subprocess(
    step_name: str,
    args_list: list[str],
    gpu_id: int,
    done_marker: Path,
    log_file: Optional[str] = None,
) -> None:
    """在子进程中运行一个 reward step，完成后写标记文件。"""
    env_name = STEP_CONDA_ENV[step_name]
    py = _env_python(env_name)
    script = str(_STEPS_DIR / f"{step_name}.py")
    cmd = [py, "-u", script] + args_list

    env_vars = dict(os.environ)
    # 把 local gpu_id（CUDA_VISIBLE_DEVICES 切片后的 index）映射回物理 GPU id，
    # 避免训练用 CUDA_VISIBLE_DEVICES=1,2,3,5 时子进程跑到错误的物理卡上。
    parent_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if parent_cvd:
        visible = [d.strip() for d in parent_cvd.split(",") if d.strip()]
        if 0 <= gpu_id < len(visible):
            env_vars["CUDA_VISIBLE_DEVICES"] = visible[gpu_id]
        else:
            env_vars["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        env_vars["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[Dispatcher] [{step_name}] GPU(local={gpu_id} → physical={env_vars['CUDA_VISIBLE_DEVICES']})  "
          f"CMD={' '.join(cmd)}")

    stdout_fh = open(log_file, "w") if log_file else None
    try:
        result = subprocess.run(cmd, env=env_vars,
                                stdout=stdout_fh, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            print(f"[Dispatcher] [{step_name}] FAILED (code {result.returncode})")
            done_marker.write_text(f"FAILED:{result.returncode}")
        else:
            done_marker.write_text("OK")
            print(f"[Dispatcher] [{step_name}] Done.")
    finally:
        if stdout_fh:
            stdout_fh.close()


def _wait_for_markers(markers: list[Path], timeout: float = 3600.0, poll: float = 5.0) -> bool:
    """等待所有标记文件出现，返回 True 表示全部 OK，False 表示有失败或超时。

    修复：即使某个 step 失败，也继续等待其余 step 完成，避免提前返回导致
    compute_all_rewards 在 DA3/DINOv2 npz 文件未写完前就被调用（FUSE 竞态）。
    """
    start = time.time()
    pending = set(str(m) for m in markers)
    failed = set()
    while pending and (time.time() - start) < timeout:
        to_remove = set()
        for mp in list(pending):
            p = Path(mp)
            if p.exists():
                content = p.read_text().strip()
                if content.startswith("FAILED"):
                    print(f"[Dispatcher] Marker FAILED: {mp}  content={content}")
                    failed.add(mp)
                to_remove.add(mp)  # 无论成功失败都移出 pending，继续等其他
        pending -= to_remove
        if pending:
            time.sleep(poll)
    if pending:
        print(f"[Dispatcher] Timeout waiting for: {pending}")
        return False
    if failed:
        return False
    return True


def _extract_frames_cv2(video_path: str, out_dir: str) -> list[str]:
    import cv2
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
        import cv2 as _cv2
        _cv2.imwrite(str(p), frame)
        paths.append(str(p))
        idx += 1
    cap.release()
    return paths


# ══════════════════════════════════════════════════════════════════════════════
# 主类
# ══════════════════════════════════════════════════════════════════════════════

class RewardDispatcher:
    """分布式 reward 调度器。

    生命周期：
      1. __init__: 解析 GPU 分配方案、确定启用的 reward 组
      2. run():    逐个 rollout 并行启动各 GPU 组 → 等待所有组完成 → 聚合
    """

    def __init__(self, args):
        self.args = args
        self.gpu_assignment = _resolve_gpu_assignment(args)
        # Per-rank slice：当一组里给了多张卡（如 [4,5,6,7]）时，按 RANK 取一张，
        # 避免多 rank 并发把同一张卡打爆。仅缩到单卡 list，保留语义一致。
        rank = int(os.environ.get("RANK", 0))
        for k, lst in list(self.gpu_assignment.items()):
            if isinstance(lst, list) and len(lst) > 1:
                self.gpu_assignment[k] = [lst[rank % len(lst)]]
        self.reward_model_root = (
            getattr(args, "reward_model_root", None)
            or os.environ.get("RL_MODEL_ROOT", "")
        )
        if self.reward_model_root:
            os.environ["RL_MODEL_ROOT"] = self.reward_model_root

        # 解析启用的 reward 列表
        rewards_str = getattr(args, "rewards", "all")
        if rewards_str == "all":
            self.rewards = list(ALL_REWARDS)
        else:
            self.rewards = [r.strip() for r in rewards_str.split(",") if r.strip()]

        # 确定需要运行哪些 GPU 组
        self.active_groups: set[str] = set()
        for r in self.rewards:
            self.active_groups.update(REWARD_TO_GROUPS.get(r, set()))

        # 解析 reward 权重覆盖
        self.weights = self._parse_weights(getattr(args, "reward_weights", None))

        print(f"[Dispatcher] rewards={self.rewards}")
        print(f"[Dispatcher] active_groups={self.active_groups}")
        print(f"[Dispatcher] gpu_assignment={self.gpu_assignment}")

    def _parse_weights(self, weights_str: Optional[str]) -> Optional[dict]:
        if not weights_str:
            return None
        w = {}
        for item in weights_str.split(","):
            item = item.strip()
            if ":" in item:
                k, v = item.split(":", 1)
                w[k.strip()] = float(v.strip())
        return w if w else None

    def run_one(
        self,
        video_path: str,
        gt_camera_txt: str,
        work_dir: str,
        prompt: str = "camera moving through a scene",
        timeout: float = 3600.0,
    ) -> dict:
        """对单条 rollout 运行完整的 reward 计算，返回 reward dict。"""
        args = self.args
        dry_run = getattr(args, "dry_run", False)

        if dry_run:
            import random
            r = random.random()
            return {
                "reward_total": r,
                "reward_camera_traj": r,
                "reward_feature_sim": r,
                "reward_geo_semantic": r,
                "reward_video_quality": r,
            }

        work = Path(work_dir)
        frames_dir = work / "frames"
        intermediates = work / "intermediates"
        intermediates.mkdir(parents=True, exist_ok=True)

        # 预先提取帧（多组共享）
        if any(g in self.active_groups for g in ["da3", "qwen_sam3", "dinov2_extract"]):
            _extract_frames_cv2(str(video_path), str(frames_dir))

        # 获取视频尺寸
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()

        skip_done = getattr(args, "skip_done", True)
        da3_npz = intermediates / "da3_output.npz"
        label_maps_npz = intermediates / "label_maps.npz"
        objects_json = intermediates / "objects.json"
        dinov2_npz = intermediates / "dinov2_features.npz"
        videoalign_json = intermediates / "videoalign.json"
        logs_dir = work / "logs"
        logs_dir.mkdir(exist_ok=True)

        # ── 并行启动各 GPU 组 ────────────────────────────────────────────────
        threads: list[threading.Thread] = []
        markers: list[Path] = []

        def _launch(group_name: str, step_name: str, step_args: list, output_path: Path, gpu_id: int):
            marker = intermediates / f".{group_name}_done"
            if skip_done and output_path.exists():
                marker.write_text("OK")
                markers.append(marker)
                return
            marker.unlink(missing_ok=True)
            markers.append(marker)
            log_file = str(logs_dir / f"{group_name}.log")
            t = threading.Thread(
                target=_run_step_subprocess,
                args=(step_name, step_args, gpu_id, marker, log_file),
                daemon=True,
            )
            t.start()
            threads.append(t)

        if "da3" in self.active_groups:
            gpus = self.gpu_assignment.get("da3", [0])
            _launch("da3", "step_da3", [
                "--video_frames_dir", str(frames_dir),
                "--output", str(da3_npz), "--gpu", "0",
            ], da3_npz, gpus[0])

        if "qwen_sam3" in self.active_groups:
            gpus = self.gpu_assignment.get("qwen_sam3", [1])
            extra = [
                "--video_frames_dir", str(frames_dir),
                "--output", str(label_maps_npz),
                "--objects_output", str(objects_json), "--gpu", "0",
            ]
            if objects_json.exists():
                extra += ["--objects_json", str(objects_json)]
            _launch("qwen_sam3", "step_qwen_sam3", extra, label_maps_npz, gpus[0])

        if "dinov2_extract" in self.active_groups:
            gpus = self.gpu_assignment.get("dinov2_extract", [2])
            _launch("dinov2_extract", "step_dinov2_extract", [
                "--video_frames_dir", str(frames_dir),
                "--output", str(dinov2_npz), "--gpu", "0",
            ], dinov2_npz, gpus[0])

        if "videoalign" in self.active_groups:
            gpus = self.gpu_assignment.get("videoalign", [3])
            _launch("videoalign", "step_videoalign", [
                "--video_path", str(video_path),
                "--prompt", prompt,
                "--output", str(videoalign_json), "--gpu", "0",
            ], videoalign_json, gpus[0])

        # ── 等待所有组完成 ────────────────────────────────────────────────────
        ok = _wait_for_markers(markers, timeout=timeout)
        # join timeout 对齐 _wait_for_markers timeout，避免 DA3 未写完就聚合
        for t in threads:
            t.join(timeout=timeout)

        if not ok:
            print(f"[Dispatcher] WARNING: some steps failed/timed out for {video_path}")

        # ── 聚合 reward（GPU 加速，rank 本地 cuda:0）─────────────────────────
        sys.path.insert(0, str(_REWARD_DIR))
        from reward_metrics import compute_all_rewards
        import torch as _torch_mod

        _agg_device = "cuda:0" if _torch_mod.cuda.is_available() else "cpu"

        result = compute_all_rewards(
            da3_path=str(da3_npz),
            label_maps_path=str(label_maps_npz),
            feature_sim_path=str(intermediates / "feature_sim_reward.json"),
            dinov2_features_path=str(dinov2_npz),
            videoalign_path=str(videoalign_json),
            gt_camera_txt=gt_camera_txt,
            H_img=H, W_img=W,
            rewards_to_compute=self.rewards,
            weights=self.weights,
            device=_agg_device,
            conf_threshold=getattr(args, "conf_threshold", 0.0),
            geo_compare_mode=getattr(args, "geo_compare_mode", "all_pairs"),
            feature_compare_mode=getattr(args, "feature_compare_mode", "first_frame"),
        )
        if _agg_device.startswith("cuda"):
            _torch_mod.cuda.empty_cache()

        # 保存结果
        reward_json = work / "reward.json"
        serializable = {k: v for k, v in result.items() if k != "details"}
        with open(str(reward_json), "w") as f:
            json.dump(serializable, f, indent=2, default=str)

        details_json = work / "reward_details.json"
        with open(str(details_json), "w") as f:
            json.dump(result, f, indent=2, default=str)

        # 清理中间值（可选）
        if not getattr(args, "keep_intermediates", True):
            import shutil
            shutil.rmtree(str(intermediates), ignore_errors=True)
            shutil.rmtree(str(frames_dir), ignore_errors=True)

        return result

    # ──────────────────────────────────────────────────────────────────────
    # Centralized dispatch（rank0 起 4 线程，每线程绑定一张 GPU + 一种 step）
    # ──────────────────────────────────────────────────────────────────────

    def _build_step_args(
        self, group: str, video_path: str, intermediates: Path, prompt: str,
    ) -> tuple[Path, list[str]]:
        """返回 (期望产物路径, subprocess 参数列表)。仅用于 da3/dinov2_extract/videoalign。"""
        frames_dir = intermediates.parent / "frames"
        if group == "da3":
            out = intermediates / "da3_output.npz"
            return out, [
                "--video_frames_dir", str(frames_dir),
                "--output", str(out), "--gpu", "0",
            ]
        if group == "dinov2_extract":
            out = intermediates / "dinov2_features.npz"
            return out, [
                "--video_frames_dir", str(frames_dir),
                "--output", str(out), "--gpu", "0",
            ]
        if group == "videoalign":
            out = intermediates / "videoalign.json"
            return out, [
                "--video_path", str(video_path),
                "--prompt", prompt,
                "--output", str(out), "--gpu", "0",
            ]
        raise ValueError(f"_build_step_args: unsupported group {group!r}")

    def _run_group_loop(
        self,
        group: str,
        gpu_id: int,
        video_paths: list[str],
        work_dirs: list[str],
        prompts: list[str],
    ) -> None:
        """在一张 GPU 上串行处理 N 条 rollout 的某个 reward step。

        对 ``qwen_sam3`` 走两阶段顺序：先把全部 rollout 的 Qwen 跑完（每条
        subprocess 退出即释放 ~14GB Qwen 显存），再统一跑 SAM3（~6GB），从而
        把 GPU3 单卡峰值从 18-22GB 降到 14GB。
        """
        skip_done = getattr(self.args, "skip_done", True)

        if group == "qwen_sam3":
            # 阶段 1：32× Qwen
            for vp, wd, prompt in zip(video_paths, work_dirs, prompts):
                inter = Path(wd) / "intermediates"
                inter.mkdir(parents=True, exist_ok=True)
                logs = Path(wd) / "logs"
                logs.mkdir(exist_ok=True)
                obj_json = inter / "objects.json"
                marker = inter / ".qwen_done"
                if skip_done and obj_json.exists():
                    marker.write_text("OK")
                    continue
                marker.unlink(missing_ok=True)
                _run_step_subprocess(
                    "step_qwen",
                    [
                        "--video_frames_dir", str(Path(wd) / "frames"),
                        "--objects_output", str(obj_json),
                        "--gpu", "0",
                    ],
                    gpu_id, marker,
                    log_file=str(logs / "qwen.log"),
                )
            # 阶段 2：32× SAM3（Qwen subprocess 已经退出，显存自动释放）
            for vp, wd, prompt in zip(video_paths, work_dirs, prompts):
                inter = Path(wd) / "intermediates"
                logs = Path(wd) / "logs"
                obj_json = inter / "objects.json"
                label_npz = inter / "label_maps.npz"
                marker = inter / ".sam3_done"
                if skip_done and label_npz.exists():
                    marker.write_text("OK")
                    continue
                if not obj_json.exists():
                    print(f"[Dispatcher][centralized][sam3] objects.json missing for {wd}, "
                          f"skipping (Qwen 阶段失败?)")
                    marker.write_text("FAILED:no_objects")
                    continue
                marker.unlink(missing_ok=True)
                _run_step_subprocess(
                    "step_sam3",
                    [
                        "--video_frames_dir", str(Path(wd) / "frames"),
                        "--objects_json", str(obj_json),
                        "--output", str(label_npz),
                        "--gpu", "0",
                    ],
                    gpu_id, marker,
                    log_file=str(logs / "sam3.log"),
                )
            return

        step_name = {
            "da3":            "step_da3",
            "dinov2_extract": "step_dinov2_extract",
            "videoalign":     "step_videoalign",
        }[group]

        for vp, wd, prompt in zip(video_paths, work_dirs, prompts):
            inter = Path(wd) / "intermediates"
            inter.mkdir(parents=True, exist_ok=True)
            logs = Path(wd) / "logs"
            logs.mkdir(exist_ok=True)
            out_path, args_list = self._build_step_args(group, vp, inter, prompt)
            marker = inter / f".{group}_done"
            if skip_done and out_path.exists():
                marker.write_text("OK")
                continue
            marker.unlink(missing_ok=True)
            _run_step_subprocess(
                step_name, args_list, gpu_id, marker,
                log_file=str(logs / f"{group}.log"),
            )

    def run_centralized(
        self,
        video_paths: list[str],
        gt_camera_paths: list[str],
        work_dirs: list[str],
        prompts: Optional[list[str]] = None,
        timeout: float = 7200.0,
    ) -> list[float]:
        """Centralized dispatch：rank0 起最多 4 个工作线程，每个线程绑定一张
        GPU + 一种 reward step，串行处理全部 rollout（GPU3 内部 Qwen→SAM3 两阶段）。

        所有线程并发跑完后，CPU 端聚合每条 rollout 的 reward。
        """
        n = len(video_paths)
        if prompts is None:
            prompts = ["camera moving through a scene"] * n
        assert len(gt_camera_paths) == n and len(work_dirs) == n and len(prompts) == n

        dry_run = getattr(self.args, "dry_run", False)
        if dry_run:
            import random
            return [random.random() for _ in range(n)]

        # 1) 预先在 CPU 端把每条视频的 frames 全部抽出来（多组共享）
        if any(g in self.active_groups for g in ["da3", "qwen_sam3", "dinov2_extract"]):
            for vp, wd in zip(video_paths, work_dirs):
                Path(wd).mkdir(parents=True, exist_ok=True)
                _extract_frames_cv2(str(vp), str(Path(wd) / "frames"))

        # 2) 起线程，每个绑定一张 GPU + 一种 step
        threads: list[threading.Thread] = []
        for group, gpu_id in CENTRALIZED_GPU_ASSIGNMENT.items():
            if group not in self.active_groups:
                continue
            t = threading.Thread(
                target=self._run_group_loop,
                args=(group, gpu_id, list(video_paths), list(work_dirs), list(prompts)),
                daemon=True,
                name=f"reward-{group}",
            )
            t.start()
            threads.append(t)
            print(f"[Dispatcher][centralized] launched thread '{group}' on GPU {gpu_id}")

        # 等待所有线程结束
        deadline = time.time() + timeout
        for t in threads:
            remaining = max(0.0, deadline - time.time())
            t.join(timeout=remaining)
            if t.is_alive():
                print(f"[Dispatcher][centralized] WARNING: thread '{t.name}' still alive "
                      f"after {timeout}s timeout")

        # 3) GPU 端按条聚合 reward（rank0 的 local cuda:0 = physical GPU 1，
        #    Phase 2 时该卡已 offload 模型，VRAM 充足；CPU 跑 32 rollout 需 2-5 小时）
        sys.path.insert(0, str(_REWARD_DIR))
        from reward_metrics import compute_all_rewards
        import cv2
        import torch as _torch_mod

        agg_device = "cuda:0" if _torch_mod.cuda.is_available() else "cpu"
        print(f"[Dispatcher][centralized] reward aggregation device = {agg_device}")

        results: list[float] = []
        for i, (vp, gcp, wd) in enumerate(zip(video_paths, gt_camera_paths, work_dirs)):
            inter = Path(wd) / "intermediates"
            cap = cv2.VideoCapture(str(vp))
            H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap.release()

            try:
                result = compute_all_rewards(
                    da3_path=str(inter / "da3_output.npz"),
                    label_maps_path=str(inter / "label_maps.npz"),
                    feature_sim_path=str(inter / "feature_sim_reward.json"),
                    dinov2_features_path=str(inter / "dinov2_features.npz"),
                    videoalign_path=str(inter / "videoalign.json"),
                    gt_camera_txt=str(gcp),
                    H_img=H, W_img=W,
                    rewards_to_compute=self.rewards,
                    weights=self.weights,
                    device=agg_device,
                    conf_threshold=getattr(self.args, "conf_threshold", 0.0),
                    geo_compare_mode=getattr(self.args, "geo_compare_mode", "all_pairs"),
                    feature_compare_mode=getattr(self.args, "feature_compare_mode", "first_frame"),
                )
            except Exception as e:
                print(f"[Dispatcher][centralized] compute_all_rewards FAILED for {wd}: {e}")
                result = {"reward_total": float("nan")}
            finally:
                # 每条 rollout 后释放 GPU 端 features/depth 等张量，避免显存累计
                if agg_device.startswith("cuda"):
                    _torch_mod.cuda.empty_cache()

            scalar = result.get("reward_total", float("nan"))
            results.append(scalar)
            print(f"[Dispatcher][centralized] [{i+1}/{n}] reward_total={scalar:.4f}  ({wd})")

            # 持久化
            try:
                serializable = {k: v for k, v in result.items() if k != "details"}
                with open(str(Path(wd) / "reward.json"), "w") as f:
                    json.dump(serializable, f, indent=2, default=str)
                with open(str(Path(wd) / "reward_details.json"), "w") as f:
                    json.dump(result, f, indent=2, default=str)
            except Exception as e:
                print(f"[Dispatcher][centralized] save reward.json failed: {e}")

            if not getattr(self.args, "keep_intermediates", True):
                import shutil
                shutil.rmtree(str(inter), ignore_errors=True)
                shutil.rmtree(str(Path(wd) / "frames"), ignore_errors=True)

        return results

    def run(
        self,
        video_paths: list[str],
        gt_camera_paths: list[str],
        work_dirs: list[str],
        prompts: Optional[list[str]] = None,
        timeout: float = 3600.0,
    ) -> list[float]:
        """批量运行 reward 计算，返回每条 rollout 的 reward_total。

        对于多卡方案：当 GPU 组内有多张卡时（如 16 卡），将 rollouts 均匀分配到
        同一组内的不同 GPU，从而实现同组多条 rollout 并行。

        当前实现：每个 GPU 组并行执行不同类型任务，多条 rollout 顺序执行。
        """
        if prompts is None:
            prompts = ["camera moving through a scene"] * len(video_paths)

        results = []
        for i, (vp, gcp, wd, pt) in enumerate(
                zip(video_paths, gt_camera_paths, work_dirs, prompts)):
            print(f"\n[Dispatcher] ===== Rollout {i+1}/{len(video_paths)} =====")
            result = self.run_one(vp, gcp, wd, pt, timeout=timeout)
            scalar = result.get("reward_total", float("nan"))
            results.append(scalar)
            print(f"[Dispatcher] reward_total={scalar:.4f}")

        return results
