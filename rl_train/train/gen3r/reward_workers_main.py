"""reward_workers_main.py — 常驻 reward worker 子进程入口。

每个 worker 进程：
  1. 启动时根据 --worker {da3|dinov2|videoalign} 加载对应模型一次（导入路径
     来自 rl_train/reward/steps/step_*.py）
  2. 通过 stdin/stdout JSON line 协议接收任务、回写结果
  3. 收到 {"cmd": "exit"} 后退出，自动释放 GPU 显存

注意：本文件运行在子进程的 conda env 内（rl_da3 或 Videoalign），
不要在文件顶部 import 主 env 的依赖。所有 reward step 模块的 import
都延迟到具体 worker 启动函数里。
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_RL_TRAIN = _HERE.parent.parent
_REWARD = _RL_TRAIN / "reward"
_STEPS = _REWARD / "steps"

for p in (str(_REWARD), str(_STEPS)):
    if p not in sys.path:
        sys.path.insert(0, p)


# ════════════════════════════════════════════════════════════════════════════
# JSON line 协议工具（write 必须立即 flush，否则父进程读不到）
# ════════════════════════════════════════════════════════════════════════════

def _emit(msg: dict) -> None:
    """向 stdout 写一行 JSON 并 flush。"""
    sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _slog(tag: str, msg: str) -> None:
    """worker 的人读日志（→ stderr → 父进程 worker_<x>.log）。

    格式：HH:MM:SS [tag] msg
    """
    ts = time.strftime("%H:%M:%S")
    sys.stderr.write(f"{ts} [{tag}] {msg}\n")
    sys.stderr.flush()


def _read_loop():
    """逐行读 stdin，yield 解析后的 dict。EOF 时停止。"""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as e:
            sys.stderr.write(f"[worker] bad JSON on stdin: {e}: {line[:200]}\n")
            continue


# ════════════════════════════════════════════════════════════════════════════
# DA3 worker
# ════════════════════════════════════════════════════════════════════════════

def _main_da3() -> None:
    """常驻 DA3 worker：加载一次 DepthAnything3，循环处理 frames_dir → npz。

    支持两种 cmd：
      run        : 单个 sub-job（兼容旧路径）
      run_batch  : list of sub-jobs，worker 内部串行处理
                    （DA3 是单 scene 序列重建模型，不同 scene 不能 cat 成
                    一个 batch — 会破坏几何，所以串行是唯一安全做法）
    """
    import numpy as np
    import torch
    from PIL import Image

    # 把 DA3 源码路径加进来
    from step_da3 import DA3_SRC, DA3_WEIGHTS, _load_da3_local
    if DA3_SRC not in sys.path:
        sys.path.insert(0, DA3_SRC)

    _slog("da3", f"loading model from {DA3_WEIGHTS}")
    t0 = time.time()
    device = torch.device("cuda:0")
    model = _load_da3_local(DA3_WEIGHTS).to(device=device).eval()
    elapsed = time.time() - t0
    _slog("da3", f"model READY in {elapsed:.1f}s")
    _emit({"status": "ready", "elapsed": elapsed})

    def _run_one(frames_dir: str, output_path: str, label: str) -> float:
        """内部：跑一条 sub-job，返回耗时秒。失败抛异常。"""
        t_total0 = time.time()
        t_io0 = time.time()
        frame_paths = sorted([
            os.path.join(frames_dir, f)
            for f in os.listdir(frames_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        if not frame_paths:
            raise RuntimeError(f"no frames in {frames_dir}")
        images = [np.array(Image.open(p).convert("RGB")) for p in frame_paths]
        io_dt = time.time() - t_io0

        t_inf0 = time.time()
        with torch.no_grad():
            pred = model.inference(images)
        inf_dt = time.time() - t_inf0

        t_save0 = time.time()
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        np.savez_compressed(
            output_path,
            depth=pred.depth,
            extrinsics=pred.extrinsics,
            intrinsics=pred.intrinsics,
            conf=pred.conf if pred.conf is not None else np.array([]),
        )
        save_dt = time.time() - t_save0
        del pred
        torch.cuda.empty_cache()

        total_dt = time.time() - t_total0
        _slog("da3", f"  {label}: OK n={len(images)}  total={total_dt:.1f}s "
                     f"(io={io_dt:.1f} infer={inf_dt:.1f} save={save_dt:.1f}) "
                     f"→ {os.path.basename(output_path)}")
        return total_dt

    n_jobs = 0
    elapsed_sum = 0.0
    on_gpu = True  # 模型当前是否在 GPU
    for msg in _read_loop():
        cmd = msg.get("cmd")
        if cmd == "ping":
            _emit({"status": "ready" if on_gpu else "unloaded"})
            continue
        if cmd == "exit":
            _slog("da3", f"exit received (processed {n_jobs} jobs, avg {elapsed_sum / max(n_jobs,1):.1f}s)")
            break
        if cmd == "unload":
            if on_gpu:
                t0 = time.time()
                model.cpu()
                torch.cuda.empty_cache()
                on_gpu = False
                _slog("da3", f"UNLOADED to CPU in {time.time()-t0:.2f}s")
            _emit({"status": "unloaded"})
            continue
        if cmd == "reload":
            if not on_gpu:
                t0 = time.time()
                model.to(device=device).eval()
                on_gpu = True
                _slog("da3", f"RELOADED to GPU in {time.time()-t0:.2f}s")
            _emit({"status": "ready"})
            continue

        # ── 单 job 路径（兼容旧调用）──────────────────────────────────────
        if cmd == "run":
            job_id = msg.get("job_id")
            args = msg.get("args", {})
            frames_dir = args.get("frames_dir")
            output_path = args.get("output_path")
            short = (job_id or "?")[:6]
            if not on_gpu:
                _emit({"status": "FAILED", "job_id": job_id,
                       "error": "model is unloaded; reload before run"})
                continue
            try:
                dt = _run_one(frames_dir, output_path, label=f"job {short}")
                n_jobs += 1
                elapsed_sum += dt
                if n_jobs % 50 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                _emit({"job_id": job_id, "status": "ok",
                       "output_path": output_path,
                       "elapsed": dt})
            except Exception:
                _slog("da3", f"job {short}: FAILED")
                _emit({"job_id": job_id, "status": "FAILED",
                       "error": traceback.format_exc()})
                torch.cuda.empty_cache()
            continue

        # ── batch job 路径：list of sub-jobs，串行处理 ────────────────────
        if cmd == "run_batch":
            job_id = msg.get("job_id")
            sub_jobs = msg.get("args", {}).get("jobs", [])
            short = (job_id or "?")[:6]
            if not on_gpu:
                _emit({"status": "FAILED", "job_id": job_id,
                       "error": "model is unloaded; reload before run_batch"})
                continue
            t_batch0 = time.time()
            _slog("da3", f"batch {short}: {len(sub_jobs)} sub-jobs")
            results: list = []
            try:
                for k, sj in enumerate(sub_jobs):
                    dt = _run_one(sj["frames_dir"], sj["output_path"],
                                  label=f"batch {short}[{k+1}/{len(sub_jobs)}]")
                    results.append({"output_path": sj["output_path"],
                                    "elapsed": dt, "status": "ok"})
                    n_jobs += 1
                    elapsed_sum += dt
                gc.collect()
                torch.cuda.empty_cache()
                batch_dt = time.time() - t_batch0
                _slog("da3", f"batch {short}: ALL OK total={batch_dt:.1f}s "
                             f"({len(sub_jobs)} jobs, avg {batch_dt/len(sub_jobs):.1f}s)  "
                             f"[lifetime avg {elapsed_sum/max(n_jobs,1):.1f}s × {n_jobs}]")
                _emit({"job_id": job_id, "status": "ok",
                       "results": results, "elapsed": batch_dt})
            except Exception:
                _slog("da3", f"batch {short}: FAILED at sub-job {len(results)}")
                _emit({"job_id": job_id, "status": "FAILED",
                       "results": results,
                       "error": traceback.format_exc()})
                torch.cuda.empty_cache()
            continue

        _emit({"status": "FAILED", "job_id": msg.get("job_id"),
               "error": f"unknown cmd {cmd!r}"})


# ════════════════════════════════════════════════════════════════════════════
# DINOv2 worker
# ════════════════════════════════════════════════════════════════════════════

def _main_dinov2() -> None:
    """常驻 DINOv2 worker：加载一次 ViT-S/14，循环 extract patch tokens。

    支持两种 cmd：
      run        : 单 rollout（兼容旧路径，B=1 串行 49 帧）
      run_batch  : list of N rollouts，把所有 rollout 的 frames cat 到一起，
                   按 forward_batch（默认 8）做真 batch forward，按 rollout
                   切回去并写各自的 npz —— 用户要求的 "DINO 提取并行处理 8 个"。
    """
    import numpy as np
    import torch
    from PIL import Image
    import torchvision.transforms as T

    from step_dinov2_extract import (
        IMAGENET_MEAN, IMAGENET_STD, PATCH_SIZE,
        DINOv2PatchExtractor,
        extract_all_frames,
        load_extractor,
        make_divisible,
    )

    _slog("dinov2", "loading model")
    t0 = time.time()
    device = "cuda:0"
    model = load_extractor(device)
    elapsed = time.time() - t0
    _slog("dinov2", f"model READY in {elapsed:.1f}s")
    _emit({"status": "ready", "elapsed": elapsed})

    _normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    _to_tensor = T.ToTensor()

    def _load_and_normalize(p: str, H_model: int, W_model: int) -> torch.Tensor:
        img = Image.open(p).convert("RGB")
        if img.height != H_model or img.width != W_model:
            img = img.resize((W_model, H_model), Image.BILINEAR)
        return _normalize(_to_tensor(img))  # (3, H, W) float32 cpu

    @torch.no_grad()
    def _extract_batched(frame_paths: list, forward_batch: int):
        """对一个 rollout 的所有 frames 做 batch forward。返回
        (features fp16 numpy [N,C,H_feat,W_feat], H_feat, W_feat, H_model, W_model)."""
        ref_img = Image.open(frame_paths[0]).convert("RGB")
        H_orig, W_orig = ref_img.height, ref_img.width
        H_model = make_divisible(H_orig)
        W_model = make_divisible(W_orig)

        tensors = [_load_and_normalize(p, H_model, W_model) for p in frame_paths]
        x_all = torch.stack(tensors, dim=0)  # (N, 3, H, W) cpu
        N = x_all.shape[0]

        feat_chunks = []
        H_feat = W_feat = None
        for s in range(0, N, forward_batch):
            xb = x_all[s:s + forward_batch].to(device, non_blocking=True)
            fb = model(xb)  # (b, C, H_feat, W_feat)
            if H_feat is None:
                H_feat, W_feat = int(fb.shape[2]), int(fb.shape[3])
            feat_chunks.append(fb.to(torch.float16).cpu().numpy())
            del xb, fb
        del x_all
        feats = np.concatenate(feat_chunks, axis=0) if len(feat_chunks) > 1 else feat_chunks[0]
        return feats, H_feat, W_feat, H_model, W_model

    def _run_one(frames_dir: str, output_path: str, label: str,
                 forward_batch: int = 8) -> tuple[float, dict]:
        t_total0 = time.time()
        t_io0 = time.time()
        frame_paths = sorted([
            str(p) for p in Path(frames_dir).iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        ])
        if not frame_paths:
            raise RuntimeError(f"no frames in {frames_dir}")
        io_dt = time.time() - t_io0

        t_inf0 = time.time()
        features, H_feat, W_feat, H_model, W_model = _extract_batched(
            frame_paths, forward_batch=forward_batch,
        )
        inf_dt = time.time() - t_inf0

        t_save0 = time.time()
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        np.savez(
            output_path,
            features=features,
            H_feat=H_feat, W_feat=W_feat,
            H_model=H_model, W_model=W_model,
            mode="patch_tokens",
            frames_dir=str(frames_dir),
        )
        save_dt = time.time() - t_save0
        del features
        torch.cuda.empty_cache()
        total_dt = time.time() - t_total0
        meta = {"H_feat": H_feat, "W_feat": W_feat,
                "H_model": H_model, "W_model": W_model,
                "n_frames": len(frame_paths)}
        _slog("dinov2", f"  {label}: OK n={len(frame_paths)} total={total_dt:.1f}s "
                        f"(io={io_dt:.1f} extract={inf_dt:.1f} save={save_dt:.1f}) "
                        f"feat=[{H_feat}x{W_feat}] B={forward_batch}")
        return total_dt, meta

    n_jobs = 0
    elapsed_sum = 0.0
    on_gpu = True
    for msg in _read_loop():
        cmd = msg.get("cmd")
        if cmd == "ping":
            _emit({"status": "ready" if on_gpu else "unloaded"})
            continue
        if cmd == "exit":
            _slog("dinov2", f"exit received (processed {n_jobs} jobs, avg {elapsed_sum / max(n_jobs,1):.1f}s)")
            break
        if cmd == "unload":
            if on_gpu:
                t0 = time.time()
                model.cpu()
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
                gc.collect()
                torch.cuda.empty_cache()
                free_b, total_b = torch.cuda.mem_get_info(0)
                on_gpu = False
                _slog("dinov2", f"UNLOADED to CPU in {time.time()-t0:.2f}s "
                                f"(GPU free={free_b/1e9:.2f}/{total_b/1e9:.2f} GB)")
            _emit({"status": "unloaded"})
            continue
        if cmd == "reload":
            if not on_gpu:
                t0 = time.time()
                model.to(device).eval()
                on_gpu = True
                _slog("dinov2", f"RELOADED to GPU in {time.time()-t0:.2f}s")
            _emit({"status": "ready"})
            continue

        # ── 单 rollout 路径 ──────────────────────────────────────────────
        if cmd == "run":
            job_id = msg.get("job_id")
            args = msg.get("args", {})
            frames_dir = args.get("frames_dir")
            output_path = args.get("output_path")
            forward_batch = int(args.get("forward_batch", 8))
            short = (job_id or "?")[:6]
            if not on_gpu:
                _emit({"status": "FAILED", "job_id": job_id,
                       "error": "model is unloaded; reload before run"})
                continue
            try:
                dt, meta = _run_one(frames_dir, output_path,
                                    label=f"job {short}",
                                    forward_batch=forward_batch)
                n_jobs += 1
                elapsed_sum += dt
                if n_jobs % 50 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                _emit({"job_id": job_id, "status": "ok",
                       "output_path": output_path,
                       "elapsed": dt})
            except Exception:
                _slog("dinov2", f"job {short}: FAILED")
                _emit({"job_id": job_id, "status": "FAILED",
                       "error": traceback.format_exc()})
                torch.cuda.empty_cache()
            continue

        # ── batch path：N 个 rollout，每个内部 batched forward ─────────
        if cmd == "run_batch":
            job_id = msg.get("job_id")
            args_top = msg.get("args", {})
            sub_jobs = args_top.get("jobs", [])
            forward_batch = int(args_top.get("forward_batch", 8))
            short = (job_id or "?")[:6]
            if not on_gpu:
                _emit({"status": "FAILED", "job_id": job_id,
                       "error": "model is unloaded; reload before run_batch"})
                continue
            t_batch0 = time.time()
            _slog("dinov2", f"batch {short}: {len(sub_jobs)} rollouts (forward_batch={forward_batch})")
            results: list = []
            try:
                for k, sj in enumerate(sub_jobs):
                    dt, _meta = _run_one(
                        sj["frames_dir"], sj["output_path"],
                        label=f"batch {short}[{k+1}/{len(sub_jobs)}]",
                        forward_batch=forward_batch,
                    )
                    results.append({"output_path": sj["output_path"],
                                    "elapsed": dt, "status": "ok"})
                    n_jobs += 1
                    elapsed_sum += dt
                gc.collect()
                torch.cuda.empty_cache()
                batch_dt = time.time() - t_batch0
                _slog("dinov2", f"batch {short}: ALL OK total={batch_dt:.1f}s "
                                f"({len(sub_jobs)} rollouts, avg {batch_dt/len(sub_jobs):.1f}s)  "
                                f"[lifetime avg {elapsed_sum/max(n_jobs,1):.1f}s × {n_jobs}]")
                _emit({"job_id": job_id, "status": "ok",
                       "results": results, "elapsed": batch_dt})
            except Exception:
                _slog("dinov2", f"batch {short}: FAILED at sub-job {len(results)}")
                _emit({"job_id": job_id, "status": "FAILED",
                       "results": results,
                       "error": traceback.format_exc()})
                torch.cuda.empty_cache()
            continue

        _emit({"status": "FAILED", "job_id": msg.get("job_id"),
               "error": f"unknown cmd {cmd!r}"})


# ════════════════════════════════════════════════════════════════════════════
# VideoAlign worker
# ════════════════════════════════════════════════════════════════════════════

def _main_videoalign() -> None:
    """常驻 VideoAlign worker：加载一次 VideoVLMRewardInference，循环打分。"""
    import torch

    from step_videoalign import CHECKPOINT_PATH, VIDEOALIGN_ROOT
    if VIDEOALIGN_ROOT not in sys.path:
        sys.path.insert(0, VIDEOALIGN_ROOT)
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"

    from inference import VideoVLMRewardInference  # noqa: E402

    _slog("videoalign", f"loading model from {CHECKPOINT_PATH}")
    t0 = time.time()
    device = "cuda:0"
    inferencer = VideoVLMRewardInference(
        load_from_pretrained=CHECKPOINT_PATH,
        device=device,
        dtype=torch.bfloat16,
    )
    elapsed = time.time() - t0
    _slog("videoalign", f"model READY in {elapsed:.1f}s")
    _emit({"status": "ready", "elapsed": elapsed})

    n_jobs = 0
    elapsed_sum = 0.0
    for msg in _read_loop():
        cmd = msg.get("cmd")
        if cmd == "ping":
            _emit({"status": "ready"})
            continue
        if cmd == "exit":
            _slog("videoalign", f"exit received (processed {n_jobs} jobs, avg {elapsed_sum / max(n_jobs,1):.1f}s)")
            break
        if cmd in ("unload", "reload"):
            # VideoAlign 实测仅占 ~834MB GPU，省的不多但要保持协议一致；
            # Qwen2VL inferencer 内部 device_map 复杂，强行 .cpu() 风险大，
            # 这里直接 no-op，节省的 GPU 主要靠 DA3 + DINOv2。
            _emit({"status": "unloaded" if cmd == "unload" else "ready"})
            continue
        # ── 单 job 路径 ──────────────────────────────────────────────────
        if cmd == "run":
            job_id = msg.get("job_id")
            args = msg.get("args", {})
            video_path = args.get("video_path")
            prompt = args.get("prompt", "camera moving through a scene")
            output_path = args.get("output_path")
            short = (job_id or "?")[:6]
            try:
                t_total0 = time.time()
                _slog("videoalign", f"job {short}: scoring {os.path.basename(str(video_path))} ...")

                t_inf0 = time.time()
                with torch.no_grad():
                    rewards = inferencer.reward([video_path], [prompt], use_norm=True)
                inf_dt = time.time() - t_inf0
                result = rewards[0]
                serializable = {k: (float(v) if hasattr(v, "__float__") else v)
                                for k, v in result.items()}

                t_save0 = time.time()
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                with open(output_path, "w") as f:
                    json.dump(serializable, f, indent=2)
                save_dt = time.time() - t_save0

                n_jobs += 1
                torch.cuda.empty_cache()
                if n_jobs % 25 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                total_dt = time.time() - t_total0
                elapsed_sum += total_dt
                score_str = " ".join(f"{k}={v:.3f}" for k, v in serializable.items()
                                      if isinstance(v, float))
                _slog("videoalign", f"job {short}: OK total={total_dt:.1f}s "
                                    f"(infer={inf_dt:.1f} save={save_dt:.1f}) "
                                    f"{score_str}  [avg {elapsed_sum/n_jobs:.1f}s × {n_jobs} jobs]")

                _emit({"job_id": job_id, "status": "ok",
                       "output_path": output_path,
                       "elapsed": total_dt})
            except Exception:
                _slog("videoalign", f"job {short}: FAILED")
                _emit({"job_id": job_id, "status": "FAILED",
                       "error": traceback.format_exc()})
                torch.cuda.empty_cache()
            continue

        # ── batch 路径：N 个 (video, prompt) 一次 inferencer.reward 真 batch ──
        if cmd == "run_batch":
            job_id = msg.get("job_id")
            sub_jobs = msg.get("args", {}).get("jobs", [])
            short = (job_id or "?")[:6]
            t_batch0 = time.time()
            _slog("videoalign", f"batch {short}: {len(sub_jobs)} videos (TRUE batch)")
            try:
                video_paths = [sj["video_path"] for sj in sub_jobs]
                prompts = [sj.get("prompt", "camera moving through a scene")
                           for sj in sub_jobs]
                output_paths = [sj["output_path"] for sj in sub_jobs]

                t_inf0 = time.time()
                with torch.no_grad():
                    rewards = inferencer.reward(video_paths, prompts, use_norm=True)
                inf_dt = time.time() - t_inf0
                if len(rewards) != len(sub_jobs):
                    raise RuntimeError(
                        f"VideoAlign returned {len(rewards)} rewards for "
                        f"{len(sub_jobs)} sub-jobs"
                    )

                results: list = []
                t_save0 = time.time()
                for k, (r, op) in enumerate(zip(rewards, output_paths)):
                    serializable = {kk: (float(vv) if hasattr(vv, "__float__") else vv)
                                    for kk, vv in r.items()}
                    os.makedirs(os.path.dirname(os.path.abspath(op)), exist_ok=True)
                    with open(op, "w") as f:
                        json.dump(serializable, f, indent=2)
                    score_str = " ".join(f"{kk}={vv:.3f}" for kk, vv in serializable.items()
                                          if isinstance(vv, float))
                    _slog("videoalign", f"  batch {short}[{k+1}/{len(sub_jobs)}]: "
                                        f"{os.path.basename(str(video_paths[k]))} {score_str}")
                    results.append({"output_path": op, "status": "ok"})
                save_dt = time.time() - t_save0

                n_jobs += len(sub_jobs)
                torch.cuda.empty_cache()
                if n_jobs // max(1, len(sub_jobs)) % 5 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                batch_dt = time.time() - t_batch0
                elapsed_sum += batch_dt
                _slog("videoalign", f"batch {short}: ALL OK total={batch_dt:.1f}s "
                                    f"(infer={inf_dt:.1f} save={save_dt:.1f})  "
                                    f"avg/video={batch_dt/len(sub_jobs):.1f}s")
                _emit({"job_id": job_id, "status": "ok",
                       "results": results, "elapsed": batch_dt})
            except Exception:
                _slog("videoalign", f"batch {short}: FAILED")
                _emit({"job_id": job_id, "status": "FAILED",
                       "error": traceback.format_exc()})
                torch.cuda.empty_cache()
            continue

        _emit({"status": "FAILED", "job_id": msg.get("job_id"),
               "error": f"unknown cmd {cmd!r}"})


# ════════════════════════════════════════════════════════════════════════════
# 入口
# ════════════════════════════════════════════════════════════════════════════

WORKERS = {
    "da3":        _main_da3,
    "dinov2":     _main_dinov2,
    "videoalign": _main_videoalign,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True, choices=list(WORKERS.keys()))
    args = parser.parse_args()
    try:
        WORKERS[args.worker]()
    except Exception:
        # 启动期未发出 ready 就崩了 → 把异常发出去给父进程看
        try:
            _emit({"status": "FAILED", "error": traceback.format_exc()})
        except Exception:
            pass
        sys.stderr.write(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
