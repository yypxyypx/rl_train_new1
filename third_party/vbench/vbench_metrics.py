#!/usr/bin/env python3
"""
vbench_metrics.py
=================
VBench I2V 评测模块

迁移自 RL 仓库历史 run_vbench_i2v 逻辑（不修改源文件）。

计算三项指标：
  - i2v_subject    （DINO，衡量主体一致性）
  - i2v_background （DreamSim，衡量背景一致性）
  - imaging_quality（MUSIQ，衡量视频质量）

依赖（模型权重）
----------------
  ${VBENCH_CACHE_DIR}/dreamsim_cache/hub/facebookresearch_dino_main/
  ${VBENCH_CACHE_DIR}/dreamsim_cache/dino_vitb16_pretrain.pth
  ${VBENCH_CACHE_DIR}/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth

环境变量
--------
  VBENCH_CACHE_DIR : VBench 模型缓存根目录（默认 <RL 根>/model/vbench_cache）

用法（CLI）
----------
    python vbench_metrics.py \
        --outputs_root  path/to/outputs \
        --dataset       re10k \
        --device        cuda \
        --output_json   results/vbench.json

用法（函数接口）
----------------
    from vbench_metrics import run_vbench_eval
    results = run_vbench_eval(
        sample_ids=[...],
        img_dir=Path("..."),
        vid_dir=Path("..."),
        vbench_cache="...",
        device="cuda",
    )
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_RL_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_VBENCH_CACHE = str(_RL_ROOT / "model" / "vbench_cache")


# ─────────────────────────── 数据准备 ────────────────────────────


def prepare_vbench_inputs(
    outputs_root: Path,
    dataset: str,
    vbench_input_dir: Path,
    pred_filename: str = "pred.mp4",
    ref_filename: str = "start.png",
) -> Tuple[List[str], Path, Path]:
    """
    将 outputs_root/<dataset>/<sample_id>/ 下的预测视频和参考图像
    整理到 vbench_inputs 目录（images/ 和 videos/）。

    参数
    ----
    outputs_root    : 推理输出根目录
    dataset         : 数据集名（如 re10k）
    vbench_input_dir: VBench 输入目录（将自动创建 images/ videos/）
    pred_filename   : 预测视频文件名（默认 pred.mp4）
    ref_filename    : 参考图像文件名（默认 start.png）

    返回
    ----
    (sample_ids, img_dir, vid_dir)
    """
    img_dir = vbench_input_dir / "images"
    vid_dir = vbench_input_dir / "videos"
    img_dir.mkdir(parents=True, exist_ok=True)
    vid_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = outputs_root / dataset
    if not dataset_dir.exists():
        raise FileNotFoundError(f"推理输出目录不存在: {dataset_dir}")

    sample_ids = []
    for sample_dir in sorted(dataset_dir.iterdir()):
        if not sample_dir.is_dir():
            continue
        pred_mp4 = sample_dir / pred_filename
        start_png = sample_dir / ref_filename
        if not pred_mp4.exists() or not start_png.exists():
            print(f"  [跳过] {sample_dir.name}：缺少 {pred_filename} 或 {ref_filename}")
            continue

        sample_id = sample_dir.name
        dst_img = img_dir / f"{sample_id}.jpg"
        dst_vid = vid_dir / f"{sample_id}.mp4"
        if not dst_img.exists():
            shutil.copy2(start_png, dst_img)
        if not dst_vid.exists():
            shutil.copy2(pred_mp4, dst_vid)

        sample_ids.append(sample_id)
        print(f"  准备: {sample_id}")

    print(f"共准备 {len(sample_ids)} 条样本")
    return sample_ids, img_dir, vid_dir


# ─────────────────────────── 模型加载 ────────────────────────────


def load_dino_model(vbench_cache: str, device: str):
    """从本地缓存加载 DINO ViT-B/16（i2v_subject 用）。"""
    import torch

    dino_repo = os.path.join(
        vbench_cache, "dreamsim_cache", "hub", "facebookresearch_dino_main"
    )
    candidates = [
        os.path.join(vbench_cache, "dreamsim_cache", "checkpoints", "dino_vitbase16_pretrain.pth"),
        os.path.join(vbench_cache, "dino_model", "dino_vitbase16_pretrain.pth"),
        os.path.join(vbench_cache, "dreamsim_cache", "dino_vitb16_pretrain.pth"),
    ]
    dino_weights = next((p for p in candidates if os.path.isfile(p)), None)
    if dino_weights is None:
        raise FileNotFoundError(f"找不到 DINO 权重，尝试路径: {candidates}")
    if not os.path.isdir(dino_repo):
        raise FileNotFoundError(f"DINO repo 不存在: {dino_repo}")

    model = torch.hub.load(dino_repo, "dino_vitb16", pretrained=False, source="local")
    ckpt = torch.load(dino_weights, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "teacher" in ckpt:
        raw = ckpt["teacher"]
        state_dict = {k.replace("backbone.", ""): v for k, v in raw.items()
                      if not k.startswith("head.")}
    elif isinstance(ckpt, dict) and "student" in ckpt:
        raw = ckpt["student"]
        state_dict = {k.replace("backbone.", ""): v for k, v in raw.items()
                      if not k.startswith("head.")}
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    print(f"  DINO 模型加载成功（权重: {os.path.basename(dino_weights)}）")
    return model


def load_dreamsim_model(vbench_cache: str, device: str):
    """从本地缓存加载 DreamSim（i2v_background 用）。
    
    DreamSim ensemble 需要 ~1.2GB 远程权重，若未缓存则跳过。
    通过检测 ensemble_lora 目录是否存在来避免触发下载。
    """
    from dreamsim import dreamsim
    dreamsim_cache_dir = os.path.join(vbench_cache, "dreamsim_cache")
    # 只在权重已完整缓存时才运行，避免触发网络下载
    ensemble_lora_dir = os.path.join(dreamsim_cache_dir, "ensemble_lora")
    if not os.path.isdir(ensemble_lora_dir):
        raise FileNotFoundError(
            f"DreamSim ensemble 权重未缓存: {ensemble_lora_dir}\n"
            "请先手动下载: https://github.com/ssundaram21/dreamsim#usage"
        )
    result = dreamsim(pretrained=True, cache_dir=dreamsim_cache_dir, device=device)
    # 不同版本的 dreamsim() 返回值格式不同：(model, preprocess) 元组或单个模型
    if isinstance(result, tuple):
        model, _ = result
    else:
        model = result
    model.eval()
    print("  DreamSim 模型加载成功")
    return model


def load_musiq_model(vbench_cache: str, device: str):
    """从本地缓存加载 MUSIQ（imaging_quality 用）。"""
    from pyiqa.archs.musiq_arch import MUSIQ
    musiq_path = os.path.join(vbench_cache, "pyiqa_model", "musiq_spaq_ckpt-358bb6af.pth")
    if not os.path.isfile(musiq_path):
        raise FileNotFoundError(f"MUSIQ 权重不存在: {musiq_path}")
    model = MUSIQ(pretrained_model_path=musiq_path)
    model.to(device)
    model.training = False
    print("  MUSIQ 模型加载成功")
    return model


# ─────────────────────────── 指标计算 ────────────────────────────


def run_vbench_eval(
    sample_ids: List[str],
    img_dir: Path,
    vid_dir: Path,
    vbench_cache: str,
    device: str = "cuda",
) -> Dict:
    """
    对已准备好的 images/ videos/ 目录运行三项 VBench 评测。

    参数
    ----
    sample_ids   : 样本 ID 列表（与 img_dir/vid_dir 中文件名对应）
    img_dir      : 参考图像目录（.jpg 或 .png）
    vid_dir      : 预测视频目录（.mp4）
    vbench_cache : VBench 模型缓存目录
    device       : 推理设备

    返回
    ----
    dict: 每项指标的结构为::

        {
            "i2v_subject": {
                "mean": float,
                "per_video": {"gen_0": float, "gen_1": float, ...}
            },
            ...
        }
    """
    # 按 sample_ids 顺序构建输入列表（记录哪些 sid 实际有效）
    video_pair_list: List[Tuple[str, str]] = []
    video_list: List[str] = []
    valid_sids: List[str] = []

    for sid in sample_ids:
        img_path_jpg = str(img_dir / f"{sid}.jpg")
        img_path_png = str(img_dir / f"{sid}.png")
        if os.path.exists(img_path_jpg):
            img_path = img_path_jpg
        elif os.path.exists(img_path_png):
            img_path = img_path_png
        else:
            img_path = None
        vid_path = str(vid_dir / f"{sid}.mp4")
        if img_path and os.path.exists(vid_path):
            video_pair_list.append((img_path, vid_path))
            video_list.append(vid_path)
            valid_sids.append(sid)
        else:
            print(f"  [跳过] {sid}：找不到图像或视频")

    if not video_pair_list:
        print("[错误] 没有找到有效的 image-video pair")
        return {}

    def _extract_per_video(vbench_results: list, sids: List[str]) -> Dict[str, float]:
        """将 VBench per_video list 转为 {sample_id: score} dict。"""
        out = {}
        for i, item in enumerate(vbench_results):
            if i >= len(sids):
                break
            score = item["video_results"] if isinstance(item, dict) else float(item)
            out[sids[i]] = float(score)
        return out

    results: Dict = {}

    # ── i2v_subject ──
    print("\n计算 i2v_subject (DINO)...")
    try:
        dino_model = load_dino_model(vbench_cache, device)
        from vbench2_beta_i2v.i2v_subject import i2v_subject
        mean_score, per_video_raw = i2v_subject(dino_model, video_pair_list, device)
        per_video_dict = _extract_per_video(per_video_raw, valid_sids)
        results["i2v_subject"] = {
            "mean": float(mean_score),
            "per_video": per_video_dict,
        }
        print(f"  i2v_subject mean: {mean_score:.6f}")
        for sid, s in per_video_dict.items():
            print(f"    {sid}: {s:.6f}")
    except Exception as e:
        print(f"  [警告] i2v_subject 失败: {e}")

    # ── i2v_background ──
    print("\n计算 i2v_background (DreamSim)...")
    try:
        dream_model = load_dreamsim_model(vbench_cache, device)
        from vbench2_beta_i2v.i2v_background import i2v_background
        mean_score, per_video_raw = i2v_background(dream_model, video_pair_list, device)
        per_video_dict = _extract_per_video(per_video_raw, valid_sids)
        results["i2v_background"] = {
            "mean": float(mean_score),
            "per_video": per_video_dict,
        }
        print(f"  i2v_background mean: {mean_score:.6f}")
        for sid, s in per_video_dict.items():
            print(f"    {sid}: {s:.6f}")
    except Exception as e:
        print(f"  [警告] i2v_background 失败: {e}")

    # ── imaging_quality ──
    print("\n计算 imaging_quality (MUSIQ)...")
    try:
        musiq_model = load_musiq_model(vbench_cache, device)
        from vbench.imaging_quality import technical_quality
        mean_score, per_video_raw = technical_quality(
            musiq_model, video_list, device,
            imaging_quality_preprocessing_mode="longer",
        )
        per_video_dict = _extract_per_video(per_video_raw, valid_sids)
        results["imaging_quality"] = {
            "mean": float(mean_score),
            "per_video": per_video_dict,
        }
        print(f"  imaging_quality mean: {mean_score:.6f}")
        for sid, s in per_video_dict.items():
            print(f"    {sid}: {s:.6f}")
    except Exception as e:
        print(f"  [警告] imaging_quality 失败: {e}")

    return results


def get_vbench_summary(results: Dict) -> Dict:
    """提取 {指标名: 均值}（兼容旧格式 [mean, list] 和新格式 {"mean": ..., "per_video": ...}）。"""
    summary = {}
    for dim, val in results.items():
        if isinstance(val, dict) and "mean" in val:
            summary[dim] = float(val["mean"])
        elif isinstance(val, list) and len(val) >= 1:
            try:
                summary[dim] = float(val[0])
            except (TypeError, ValueError):
                pass
    return summary


def get_vbench_per_video(results: Dict) -> Dict[str, Dict[str, float]]:
    """
    从 run_vbench_eval 结果中提取逐视频分数。

    返回
    ----
    dict: {sample_id: {"i2v_subject": float, "i2v_background": float, "imaging_quality": float}}
    """
    per_video: Dict[str, Dict[str, float]] = {}
    for dim, val in results.items():
        if isinstance(val, dict) and "per_video" in val:
            for sid, score in val["per_video"].items():
                per_video.setdefault(sid, {})[dim] = float(score)
    return per_video


# ─────────────────────────── 结果保存 ────────────────────────────


def save_results(
    results: Dict,
    sample_ids: List[str],
    output_json: Path,
) -> None:
    """保存 VBench 评测结果到 JSON 文件。"""
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"\nVBench 结果已保存: {output_json}")

    print("\n── VBench 指标汇总 ──")
    for dim, val in results.items():
        mean_val = val[0] if isinstance(val, list) else val
        print(f"  {dim:<25s}: {mean_val:.6f}")


# ─────────────────────────── CLI ─────────────────────────────────


def _parse_args():
    parser = argparse.ArgumentParser(description="VBench I2V 评测")
    # ── 模式 A：传入已准备好的 img/vid 目录（供子进程调用）──
    parser.add_argument("--img_dir",    type=Path, default=None,
                        help="已准备好的参考图片目录（各 <sample_id>.png）")
    parser.add_argument("--vid_dir",    type=Path, default=None,
                        help="已准备好的预测视频目录（各 <sample_id>.mp4）")
    parser.add_argument("--sample_ids", type=str, nargs="+", default=None,
                        help="sample_id 列表（与 --img_dir/--vid_dir 配合使用）")
    # ── 模式 B：传入原始输出目录（旧接口）──
    parser.add_argument("--outputs_root", type=Path, default=None,
                        help="推理输出根目录（含各数据集子目录）")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["re10k", "dl3dv10k"],
                        help="数据集名称")
    parser.add_argument("--vbench_input_dir", type=Path, default=None,
                        help="VBench 输入目录（自动创建）")
    parser.add_argument("--pred_filename", type=str, default="pred.mp4")
    parser.add_argument("--ref_filename",  type=str, default="start.png")
    # ── 公共 ──
    parser.add_argument("--vbench_cache", type=str, default=None,
                        help="VBench 模型缓存目录（默认读 VBENCH_CACHE_DIR 环境变量）")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_json", type=Path, default=None,
                        help="保存评测结果的 JSON 路径（结构含 summary 和 per_video）")
    return parser.parse_args()


def main():
    args = _parse_args()

    vbench_cache = args.vbench_cache or os.environ.get(
        "VBENCH_CACHE_DIR", _DEFAULT_VBENCH_CACHE
    )

    # ── 模式 A：img_dir + vid_dir 直接传入 ──────────────────────
    if args.img_dir is not None and args.vid_dir is not None:
        img_dir    = args.img_dir
        vid_dir    = args.vid_dir
        sample_ids = args.sample_ids or [
            p.stem for p in sorted(vid_dir.glob("*.mp4"))
        ]
        if not sample_ids:
            print("[错误] vid_dir 中没有找到任何 .mp4 文件")
            return
        print(f"[VBench] 模式A: img_dir={img_dir}  vid_dir={vid_dir}")
        print(f"         sample_ids: {sample_ids}")

    # ── 模式 B：outputs_root + dataset ────────────────────────────
    else:
        if args.outputs_root is None or args.dataset is None:
            print("[错误] 需要指定 (--img_dir + --vid_dir) 或 (--outputs_root + --dataset)")
            return
        dataset = args.dataset
        vbench_input_dir = args.vbench_input_dir or Path(f"vbench_inputs/{dataset}")
        output_json_path = args.output_json or Path(f"results/{dataset}_vbench.json")

        print(f"outputs_root:     {args.outputs_root}")
        print(f"dataset:          {dataset}")
        print(f"VBENCH_CACHE_DIR: {vbench_cache}")

        print("\n── 准备 VBench 输入 ──")
        sample_ids, img_dir, vid_dir = prepare_vbench_inputs(
            args.outputs_root, dataset, vbench_input_dir,
            pred_filename=args.pred_filename,
            ref_filename=args.ref_filename,
        )
        if not sample_ids:
            print("[错误] 没有找到有效的推理结果")
            return

    # ── 运行评测 ──────────────────────────────────────────────────
    print("\n── 运行 VBench 评测 ──")
    results = run_vbench_eval(sample_ids, img_dir, vid_dir, vbench_cache, args.device)

    if not results:
        print("[错误] 所有指标计算均失败")
        return

    # ── 保存结果 ──────────────────────────────────────────────────
    output_json_path = args.output_json or Path("vbench_result.json")
    out_data = {
        "summary":   get_vbench_summary(results),
        "per_video": get_vbench_per_video(results),
        "raw":       results,
    }
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    print(f"\nVBench 结果已保存: {output_json_path}")

    print("\n── VBench 指标汇总 ──")
    for dim, val in out_data["summary"].items():
        print(f"  {dim:<25s}: {val:.6f}")


if __name__ == "__main__":
    main()
