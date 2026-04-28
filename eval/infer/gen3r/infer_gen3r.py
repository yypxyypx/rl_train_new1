#!/usr/bin/env python3
"""Gen3R 官方 pipeline 推理（ODE）。

使用 Gen3RPipeline.__call__ 的官方 ODE 推理路径，不依赖任何 GRPO 训练代码。
每条样本产出一个 gen_0.mp4，跳过 T5 文本编码器（从样本目录读取预编码的
prompt_embed.pt / neg_embed.pt）。

多卡支持：按样本分片，torchrun --nproc_per_node=N 启动。

输出目录结构：
  <output_root>/<dataset>/<sample_id>/
    gen_0.mp4
    start.png, gt.mp4, camera.txt, metadata.json  (从输入目录复制)
    infer_info.json

用法：
  # 单卡
  python infer_gen3r.py \\
      --manifest /path/to/manifest.jsonl \\
      --checkpoint /path/to/gen3r_checkpoints \\
      --output_root /path/to/output

  # 多卡
  torchrun --nproc_per_node=4 infer_gen3r.py \\
      --manifest /path/to/manifest.jsonl \\
      --checkpoint /path/to/gen3r_checkpoints \\
      --output_root /path/to/output

  # 单样本调试
  python infer_gen3r.py \\
      --sample_dir /path/to/sample \\
      --checkpoint /path/to/gen3r_checkpoints \\
      --output_root /tmp/debug
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import imageio
import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange
from torchvision.transforms.functional import resize

# ── Gen3R 路径 ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_GEN3R_ROOT = Path(os.environ.get("GEN3R_ROOT", str(_HERE / "Gen3R")))
if str(_GEN3R_ROOT) not in sys.path:
    sys.path.insert(0, str(_GEN3R_ROOT))

from gen3r.pipeline import Gen3RPipeline                    # noqa: E402
from gen3r.utils.data_utils import (                        # noqa: E402
    center_crop, compute_rays, preprocess_poses, get_K,
)
from gen3r.utils.common_utils import save_videos_grid       # noqa: E402


# ── decode_latents 设备对齐补丁 ───────────────────────────────────────────────
_orig_decode = Gen3RPipeline.decode_latents


def _decode_aligned(self, latents, min_max_depth_mask=False):
    dev, dt = latents.device, torch.bfloat16
    for mod in [self.geo_adapter, self.vggt, self.wan_vae]:
        if mod is not None:
            mod.to(device=dev, dtype=dt)
    return _orig_decode(self, latents, min_max_depth_mask=min_max_depth_mask)


Gen3RPipeline.decode_latents = _decode_aligned


# ═══════════════════════ 相机解析 ══════════════════════════════════════════════


def parse_camera_txt(camera_file, img_w, img_h, target_w, target_h):
    """camera.txt → OpenCV w2c 外参 (F,4,4) 与内参 (F,3,3)。

    每行格式：idx fx_n fy_n cx_n cy_n 0 0 r00..r22 tx ty tz
    fx_n/fy_n/cx_n/cy_n 为已按原始图像尺寸归一化的值。
    """
    ext_list, int_list = [], []
    with open(camera_file, "r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        vals = list(map(float, line.split()))
        fl_x = vals[1] * img_w
        fl_y = vals[2] * img_h
        cx   = vals[3] * img_w
        cy   = vals[4] * img_h
        mat34 = np.array(vals[7:19]).reshape(3, 4)
        mat44 = np.eye(4, dtype=np.float64)
        mat44[:3, :] = mat34
        ext_list.append(mat44)
        K = get_K(img_w, img_h, fl_x, fl_y, cx, cy, target_w, target_h)
        int_list.append(K.numpy())
    return np.array(ext_list), np.array(int_list)


# ═══════════════════════ 数据准备 ══════════════════════════════════════════════


def prepare_inputs(sample_dir: Path, device, dtype, num_frames: int,
                   target_h: int, target_w: int):
    """读取样本目录，返回 (ctrl, plucker, prompt_embeds, neg_embeds)。

    ctrl           : [1, 1, 3, H, W]   条件帧（首帧图像）
    plucker        : [1, F, 6, H, W]   Plücker 射线嵌入
    prompt_embeds  : list of [1, L, 4096] tensor
    neg_embeds     : list of [1, L, 4096] tensor
    """
    meta_path = sample_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    img_w = meta["img_w"]
    img_h = meta["img_h"]

    # ── 条件图像 ──────────────────────────────────────────────────────────────
    start_arr = imageio.v2.imread(str(sample_dir / "start.png"))[..., :3]
    ctrl = (torch.from_numpy(start_arr)
            .unsqueeze(0).permute(0, 3, 1, 2)   # [1, 3, H, W]
            .unsqueeze(0).float() / 255.0)        # [1, 1, 3, H, W]
    fh, fw = ctrl.shape[3], ctrl.shape[4]
    scale = target_h / min(fh, fw)
    ctrl = resize(ctrl[0], [round(fh * scale), round(fw * scale)])
    ctrl = center_crop(ctrl, (target_h, target_w)).unsqueeze(0).to(device, dtype)

    # ── 相机参数 ──────────────────────────────────────────────────────────────
    exts_np, ints_np = parse_camera_txt(
        sample_dir / "camera.txt", img_w, img_h, target_w, target_h)

    if len(exts_np) < num_frames:
        pad = num_frames - len(exts_np)
        exts_np = np.concatenate([exts_np, np.tile(exts_np[-1:], (pad, 1, 1))], axis=0)
        ints_np = np.concatenate([ints_np, np.tile(ints_np[-1:], (pad, 1, 1))], axis=0)
    else:
        exts_np = exts_np[:num_frames]
        ints_np = ints_np[:num_frames]

    Ks    = torch.from_numpy(ints_np).float()
    ext_t = torch.from_numpy(exts_np).float()
    # camera.txt 是 OpenCV w2c → 求逆得 OpenCV c2w
    c2w_abs = torch.linalg.inv(ext_t)
    c2ws = preprocess_poses(c2w_abs).unsqueeze(0).to(device, dtype)  # [1, F, 4, 4]

    rays_o, rays_d = compute_rays(c2ws[0].cpu().float(), Ks,
                                  h=target_h, w=target_w, device="cpu")
    plucker = torch.cat([torch.cross(rays_o, rays_d, dim=1), rays_d],
                        dim=1).unsqueeze(0).to(device, dtype)         # [1, F, 6, H, W]

    # ── 预编码 T5 embedding ───────────────────────────────────────────────────
    # transformer 期望 list of [L_i, 4096] (每个批次元素一个)，
    # CFG 用 list + list 拼接：[neg] + [pos] = [neg, pos]，stack 后得到 [2, text_len, 4096]
    prompt_embed = torch.load(sample_dir / "prompt_embed.pt",
                              map_location="cpu", weights_only=True)
    neg_embed = torch.load(sample_dir / "neg_embed.pt",
                           map_location="cpu", weights_only=True)
    # 统一到 2D [L, 4096]
    if prompt_embed.dim() == 3:
        prompt_embed = prompt_embed.squeeze(0)
    if neg_embed.dim() == 3:
        neg_embed = neg_embed.squeeze(0)

    prompt_embeds = [prompt_embed.to(device, dtype)]   # list of [L_pos, 4096]
    neg_embeds    = [neg_embed.to(device, dtype)]       # list of [L_neg, 4096]

    return ctrl, plucker, prompt_embeds, neg_embeds


# ═══════════════════════ 文件工具 ══════════════════════════════════════════════


def copy_input_files(sample_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for fname in ["start.png", "gt.mp4", "camera.txt", "metadata.json", "gt_depth.npz"]:
        src = sample_dir / fname
        dst = output_dir / fname
        if src.exists() and not dst.exists():
            shutil.copy2(str(src), str(dst))


def load_manifest(manifest_path: str) -> list:
    return [
        json.loads(line)
        for line in Path(manifest_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_manifest_from_dir(sample_dir: str) -> list:
    p = Path(sample_dir)
    meta_path = p / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return [{
        "dataset":   meta.get("dataset", p.parent.name),
        "sample_id": meta.get("sample_id", p.name),
        "sample_dir": str(p),
    }]


# ═══════════════════════ 主流程 ═══════════════════════════════════════════════


def main(args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank       = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dtype  = torch.bfloat16

    # ── 构建任务列表 ──────────────────────────────────────────────────────────
    if args.manifest:
        all_entries = load_manifest(args.manifest)
    elif args.sample_dir:
        all_entries = build_manifest_from_dir(args.sample_dir)
    else:
        raise ValueError("需要 --manifest 或 --sample_dir")

    # 本进程负责的样本（按 rank 轮询分片）
    my_entries = all_entries[rank::world_size]

    if rank == 0:
        print(f"\n[Gen3R Infer] 共 {len(all_entries)} 条样本，本进程处理 {len(my_entries)} 条")
        print(f"  checkpoint:      {args.checkpoint}")
        print(f"  num_frames:      {args.num_frames}")
        print(f"  target_size:     {args.target_size}")
        print(f"  num_steps:       {args.num_inference_steps}")
        print(f"  guidance_scale:  {args.guidance_scale}")
        print(f"  shift:           {args.shift}")
        print(f"  seed:            {args.seed}")
        print(f"  output_root:     {args.output_root}\n")

    # ── 加载官方 pipeline ─────────────────────────────────────────────────────
    pipeline = Gen3RPipeline.from_pretrained(args.checkpoint)
    pipeline = pipeline.to(device).to(dtype)
    # 把 text_encoder 移回 CPU（必须在 .to(device) 之后操作）以节省显存；
    # 因为我们直接传 prompt_embeds，encode_prompt 不会实际调用 T5 forward。
    if pipeline.text_encoder is not None:
        pipeline.text_encoder = pipeline.text_encoder.cpu()
        torch.cuda.empty_cache()

    # pipeline.check_inputs 会对 list 类型的 prompt_embeds 调用 .shape 而报错；
    # 我们传 prompt="" + prompt_embeds=list 并存，所以只保留必要检查。
    def _check_inputs_patched(prompt, height, width, negative_prompt,
                              cbs_ti, prompt_embeds=None, negative_prompt_embeds=None):
        if height % 14 != 0 or width % 14 != 0:
            raise ValueError(f"`height` and `width` must be divisible by 14.")
    pipeline.check_inputs = _check_inputs_patched

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    for entry_idx, entry in enumerate(my_entries):
        dataset   = entry.get("dataset", "unknown")
        sample_id = entry.get("sample_id", f"sample_{entry_idx}")
        sample_dir = Path(entry["sample_dir"])
        output_dir = Path(args.output_root) / dataset / sample_id

        out_video = output_dir / "gen_0.mp4"
        if args.skip_done and out_video.exists():
            print(f"[{rank}] skip (done): {dataset}/{sample_id}")
            continue

        print(f"[{rank}] [{entry_idx+1}/{len(my_entries)}] {dataset}/{sample_id}")
        copy_input_files(sample_dir, output_dir)

        try:
            ctrl, plucker, prompt_embeds, neg_embeds = prepare_inputs(
                sample_dir, device, dtype,
                num_frames=args.num_frames,
                target_h=args.target_size,
                target_w=args.target_size,
            )
        except Exception as e:
            print(f"  [rank {rank}] 输入准备失败: {e}")
            continue

        try:
            with torch.no_grad():
                sample = pipeline(
                    prompt="",                # 非 None 使 batch_size=1，绕过 prompt_embeds.shape 检查
                    negative_prompt="bad detailed",  # 与官方推理一致
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=neg_embeds,
                    control_cameras=plucker,
                    control_images=ctrl,
                    num_frames=args.num_frames,
                    height=args.target_size,
                    width=args.target_size,
                    guidance_scale=args.guidance_scale,
                    shift=args.shift,
                    num_inference_steps=args.num_inference_steps,
                    return_dict=True,
                    min_max_depth_mask=True,  # 与官方推理一致
                )

            rgbs = sample.rgbs  # [B, F, H, W, 3]
            rgb_grid = rearrange(rgbs, "b f h w c -> b c f h w").float().cpu()
            save_videos_grid(rgb_grid, str(out_video), rescale=False)
            print(f"  [rank {rank}] gen_0.mp4 保存成功")

        except Exception as e:
            import traceback
            print(f"  [rank {rank}] 推理失败: {e}")
            traceback.print_exc()
            continue

        # 写推理参数记录
        info = {
            "checkpoint":          args.checkpoint,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale":      args.guidance_scale,
            "shift":               args.shift,
            "num_frames":          args.num_frames,
            "target_size":         args.target_size,
            "seed":                args.seed,
            "sampler":             "ODE (FlowMatch)",
        }
        (output_dir / "infer_info.json").write_text(
            json.dumps(info, indent=2), encoding="utf-8")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

    if rank == 0:
        print(f"\n[Gen3R Infer] 完成。输出: {args.output_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gen3R 官方 pipeline ODE 推理")

    input_grp = parser.add_mutually_exclusive_group(required=True)
    input_grp.add_argument("--manifest",   type=str, help="manifest.jsonl 路径")
    input_grp.add_argument("--sample_dir", type=str, help="单条样本目录（调试用）")

    parser.add_argument("--checkpoint",          type=str, required=True,
                        help="Gen3R checkpoint 目录")
    parser.add_argument("--output_root",         type=str, required=True)
    parser.add_argument("--num_frames",          type=int,   default=49)
    parser.add_argument("--target_size",         type=int,   default=560)
    parser.add_argument("--num_inference_steps", type=int,   default=50)
    parser.add_argument("--guidance_scale",      type=float, default=5.0)
    parser.add_argument("--shift",               type=float, default=5.0)
    parser.add_argument("--seed",                type=int,   default=42)
    parser.add_argument("--skip_done",           action="store_true")

    main(parser.parse_args())
