"""gen3r_encode.py — Gen3R 条件编码工具。

严格复用 RL/gen3r/train_grpo_gen3r.py 的逻辑，只修改 import 路径。
包含：
    encode_text              — T5 文本编码
    build_plucker_embeds     — Plücker ray embedding（时序压缩 + width 复制）
    encode_control_latents   — 控制图像 VAE + CLIP 编码
    transformer_forward      — Transformer 单步前向（支持 CFG）
    decode_rgb_video         — 解码 RGB 视频并保存
    write_camera_txt         — c2w + pixel K → camera.txt（w2c 归一化格式）
    save_gt_video            — GT 帧序列 → MP4
"""

from __future__ import annotations

import os
import random

import cv2
import numpy as np
import torch
from diffusers.utils import export_to_video
from einops import rearrange

from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
_GEN3R_PKG = _HERE / "Gen3R"
if str(_GEN3R_PKG) not in sys.path:
    sys.path.insert(0, str(_GEN3R_PKG))

from gen3r.utils.data_utils import compute_rays  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# 文本编码
# ══════════════════════════════════════════════════════════════════════════════

def encode_text(
    text: str,
    tokenizer,
    text_encoder,
    device,
    max_length: int = 512,
    dropout_prob: float = 0.0,
) -> torch.Tensor:
    """将文本编码为变长 T5 embedding。

    Args:
        dropout_prob : 若 > 0，以此概率将文本替换为空字符串（训练时 prompt dropout）

    Returns:
        prompt_embed : Tensor [actual_len, 4096]（不含 padding）
    """
    if dropout_prob > 0 and random.random() < dropout_prob:
        text = ""

    text_inputs = tokenizer(
        text,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids
    attention_mask = text_inputs.attention_mask
    seq_len = attention_mask.gt(0).sum(dim=1).long()

    with torch.no_grad():
        embeds = text_encoder(
            input_ids.to(device),
            attention_mask=attention_mask.to(device),
        )[0]  # [1, max_length, 4096]

    prompt_embed = embeds[0, :seq_len[0]]  # [actual_len, 4096]
    return prompt_embed


# ══════════════════════════════════════════════════════════════════════════════
# Plücker ray embedding
# ══════════════════════════════════════════════════════════════════════════════

def build_plucker_embeds(
    c2ws: torch.Tensor,
    Ks: torch.Tensor,
    h: int,
    w: int,
    num_frames: int,
    device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """从相机参数计算 Plücker ray embedding，并做时序压缩和 width 复制。

    Args:
        c2ws        : [F, 4, 4] camera-to-world
        Ks          : [F, 3, 3] 像素坐标内参
        h, w        : 图像分辨率（Gen3R 为 560）
        num_frames  : F

    Returns:
        plucker_embeds : [1, 24, f_latent, H, W*2]
    """
    rays_o, rays_d = compute_rays(
        c2ws.to(device),
        Ks.to(device),
        h=h, w=w,
        device=device,
    )  # [F, 3, H, W]

    o_cross_d = torch.cross(rays_o, rays_d, dim=1)
    plucker = torch.cat([o_cross_d, rays_d], dim=1).to(dtype)  # [F, 6, H, W]
    plucker = plucker.unsqueeze(0).transpose(1, 2)              # [1, 6, F, H, W]

    # 时序压缩：首帧 repeat 4 次 + 剩余帧，再 transpose
    plucker_padded = torch.cat([
        torch.repeat_interleave(plucker[:, :, 0:1], repeats=4, dim=2),
        plucker[:, :, 1:],
    ], dim=2).transpose(1, 2)  # [1, F+3, 6, H, W]

    B, f_padded, c, hh, ww = plucker_padded.shape
    f_latent = f_padded // 4
    plucker_embeds = (
        plucker_padded
        .contiguous()
        .view(B, f_latent, 4, c, hh, ww)
        .transpose(2, 3)                  # [B, f_latent, 6, 4, H, W]
        .contiguous()
        .view(B, f_latent, c * 4, hh, ww)  # [B, f_latent, 24, H, W]
        .transpose(1, 2)                   # [B, 24, f_latent, H, W]
    )

    # 沿 width 复制，匹配联合 latent 的 w*2
    plucker_embeds = torch.cat([plucker_embeds, plucker_embeds], dim=-1)  # [B,24,f,H,W*2]
    return plucker_embeds


# ══════════════════════════════════════════════════════════════════════════════
# 控制图像编码
# ══════════════════════════════════════════════════════════════════════════════

def encode_control_latents(
    control_images: torch.Tensor,
    wan_vae,
    geo_adapter,
    clip_image_encoder,
    control_index: list,
    num_frames: int,
    device,
    dtype: torch.dtype,
):
    """将控制图像编码为 control_latents 和 CLIP 特征。

    复用 Gen3R train_dit.py batch_encode_control_latents 的逻辑。

    Args:
        control_images : [1, F, C, H, W] float32 [0, 1]，非控制帧已置零
        control_index  : list of int，哪些帧是控制帧

    Returns:
        control_latents : [1, 20, f_latent, h_lat, w_lat*2]
        clip_context    : [1, 257, 1280]
    """
    B = 1

    # WAN VAE encode
    control_img_bc = rearrange(
        (control_images * 2 - 1).clamp(-1, 1),
        "b f c h w -> b c f h w"
    ).to(device, dtype)

    with torch.no_grad():
        wan_latents = wan_vae.encode(control_img_bc).latent_dist.sample()  # [1,16,f,h,w]

    # 3D geo 部分使用全零
    geo_latents = torch.zeros_like(wan_latents)
    combined = torch.cat([wan_latents, geo_latents], dim=-1)  # [1,16,f,h,w*2]

    # 构建 mask：控制帧的 RGB 部分（前半 width）为 1
    h_lat, w_lat2 = combined.shape[-2], combined.shape[-1]
    w_lat = w_lat2 // 2
    f_latent = combined.shape[2]

    masks_spatial = torch.zeros(B, num_frames, h_lat, w_lat2, device=device, dtype=dtype)
    masks_spatial[:, control_index, :, :w_lat] = 1.0

    # 时序压缩 mask
    masks_padded = torch.cat([
        torch.repeat_interleave(masks_spatial[:, :1], repeats=4, dim=1),
        masks_spatial[:, 1:],
    ], dim=1)  # [1, F+3, h, w*2]

    masks = (
        masks_padded
        .view(B, (num_frames + 3) // 4, 4, h_lat, w_lat2)
        .contiguous()
        .transpose(1, 2)  # [1, 4, f_latent, h, w*2]
    )

    control_latents = torch.cat([combined, masks], dim=1)  # [1,20,f,h,w*2]

    # CLIP encode：取第一个控制帧
    first_ctrl_idx = control_index[0]
    clip_img = control_images[0, first_ctrl_idx]  # [C, H, W] float32 [0,1]
    clip_img_t = (clip_img * 2 - 1).clamp(-1, 1).to(device, dtype)
    with torch.no_grad():
        clip_context = clip_image_encoder([clip_img_t[:, None, :, :]])  # [1,257,1280]

    return control_latents, clip_context


# ══════════════════════════════════════════════════════════════════════════════
# Transformer 单步前向
# ══════════════════════════════════════════════════════════════════════════════

def transformer_forward(
    transformer,
    z: torch.Tensor,
    timesteps: torch.Tensor,
    prompt_embeds: list,
    neg_embeds: list,
    seq_len: int,
    control_latents: torch.Tensor,
    plucker_embeds: torch.Tensor,
    clip_context: torch.Tensor,
    cfg_scale: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Gen3R Transformer 单步前向，支持 CFG。

    Transformer 接口：(x, context, t, seq_len, y, y_camera, clip_fea)
    """
    if cfg_scale > 1.0:
        z_in = torch.cat([z, z], dim=0)
        t_in = torch.cat([timesteps, timesteps], dim=0)
        ctx_in = neg_embeds + prompt_embeds  # list concat（CFG: uncond first）
        ctrl_in = torch.cat([control_latents, control_latents], dim=0)
        plk_in = torch.cat([plucker_embeds, plucker_embeds], dim=0)
        clip_in = torch.cat([clip_context, clip_context], dim=0)

        with torch.autocast("cuda", dtype):
            pred = transformer(
                x=z_in, context=ctx_in, t=t_in, seq_len=seq_len,
                y=ctrl_in, y_camera=plk_in, clip_fea=clip_in,
            )
        pred_uncond, pred_cond = pred.chunk(2)
        pred = pred_uncond.to(torch.float32) + cfg_scale * (
            pred_cond.to(torch.float32) - pred_uncond.to(torch.float32)
        )
    else:
        with torch.autocast("cuda", dtype):
            pred = transformer(
                x=z, context=prompt_embeds, t=timesteps, seq_len=seq_len,
                y=control_latents, y_camera=plucker_embeds, clip_fea=clip_context,
            )
        pred = pred.to(torch.float32)
    return pred


# ══════════════════════════════════════════════════════════════════════════════
# 解码 RGB 视频
# ══════════════════════════════════════════════════════════════════════════════

def decode_rgb_video(
    final_latent: torch.Tensor,
    wan_vae,
    video_path: str,
    fps: int = 16,
) -> str:
    """从联合 latent 解码 RGB 部分并保存视频。

    Args:
        final_latent : [1, 16, f, h, w*2]
        wan_vae      : WAN VAE（frozen）
        video_path   : 保存路径（.mp4）

    Returns:
        video_path
    """
    latent_w = final_latent.shape[-1] // 2
    wan_latent = final_latent[..., :latent_w]  # [1, 16, f, h, w]

    # diffusers 的 AutoencoderKL 常有 enable_tiling() 做大图分块解码；Gen3R 的 AutoencoderKLWan
    # 若未实现该方法则跳过（不影响 decode，只是少一种省显存手段）
    if hasattr(wan_vae, "enable_tiling"):
        wan_vae.enable_tiling()
    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            video = wan_vae.decode(wan_latent).sample  # [1, 3, F, H, W], [-1, 1]

    video_01 = (video / 2 + 0.5).clamp(0, 1)
    frames_np = (
        video_01[0].permute(1, 2, 3, 0).float().cpu().numpy() * 255
    ).astype(np.uint8)  # [F, H, W, 3]
    frames = list(frames_np)

    os.makedirs(os.path.dirname(os.path.abspath(video_path)), exist_ok=True)
    export_to_video(frames, video_path, fps=fps)
    return video_path


# ══════════════════════════════════════════════════════════════════════════════
# 相机 txt 写入
# ══════════════════════════════════════════════════════════════════════════════

def write_camera_txt(
    c2ws: torch.Tensor,
    Ks: torch.Tensor,
    H: int,
    W: int,
    out_path: str,
) -> None:
    """将 c2w 矩阵和像素内参写成 camera.txt（w2c 归一化格式，供 reward 使用）。

    每行格式：
        frame_idx  fx/W  fy/H  cx/W  cy/H  0  0  <w2c 3x4 row-major>
    """
    c2ws_np = c2ws.float().cpu().numpy()  # [F, 4, 4]
    Ks_np = Ks.float().cpu().numpy()      # [F, 3, 3]
    lines = []
    for i, (K, c2w) in enumerate(zip(Ks_np, c2ws_np)):
        fx_n = K[0, 0] / W
        fy_n = K[1, 1] / H
        cx_n = K[0, 2] / W
        cy_n = K[1, 2] / H
        w2c = np.linalg.inv(c2w)          # [4, 4]
        w2c_flat = w2c[:3, :].flatten()   # 12 values
        vals = (
            f"{i} {fx_n:.10f} {fy_n:.10f} {cx_n:.10f} {cy_n:.10f} 0 0 "
            + " ".join(f"{v:.10f}" for v in w2c_flat)
        )
        lines.append(vals)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# GT 视频保存
# ══════════════════════════════════════════════════════════════════════════════

def save_gt_video(
    pixel_values: torch.Tensor,
    out_path: str,
    fps: int = 16,
) -> None:
    """将 [F, C, H, W] float32 [0,1] 的 GT 帧序列保存为 MP4。"""
    frames = pixel_values.float().cpu().permute(0, 2, 3, 1).numpy()
    frames = (frames * 255).clip(0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    export_to_video(list(frames), out_path, fps=fps)
