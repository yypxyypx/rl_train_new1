"""wan22_encode.py — Wan2.2-Fun-5B-Control-Camera 条件编码工具。

提供 GRPO rollout / training 所需的全部条件编码：

    encode_text                 — T5 文本编码（与 gen3r 同构）
    build_camera_control        — c2ws + Ks → Plücker [1,6,F,H,W]
    chunk_camera_control        — Plücker → y_camera [1,24,F_lat,H,W]  (4-frame chunk)
    encode_inpaint_conditions   — start_image → y (52ch) + mask (1ch latent space)
    transformer_forward         — 单步 transformer 调用（含 CFG + 时间步 broadcasting）
    apply_mask_clamp            — 把首帧 latent 强制对齐到 masked_video_latents
    decode_rgb_video            — VAE3_8 decode + tiling，存 MP4
    write_camera_txt            — 与 gen3r 同名工具，复制相机 txt
    save_gt_video               — 与 gen3r 同名工具，存 GT mp4

设计要点：
- in_dim=100：transformer 期望 `cat([latent(48), y(52)], dim=channel)` 共 100 通道，
  其中 y = cat([mask_latents(4), masked_video_latents(48)], dim=channel)。
- spatial_compression_ratio=16 + I2V：每个 timestep 输入是 per-token 的 `[B, seq_len]`
  shape，取自 `mask[..., ::patch_h, ::patch_w] * t`，首帧 token 对应 ts=0（无噪声）。
- 首帧 latent clamp：`apply_mask_clamp` 在初始化和每步去噪后调用一次，让首帧
  latent 始终等于 `VAE.encode(start_image)`，确保 I2V 一致性。
"""

from __future__ import annotations

import math
import os
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

# ─── videox_fun import 路径 ──────────────────────────────────────────────────
from paths import videox_fun_root

_VIDEOX_ROOT = videox_fun_root()
if str(_VIDEOX_ROOT) not in sys.path:
    sys.path.insert(0, str(_VIDEOX_ROOT))


# ══════════════════════════════════════════════════════════════════════════════
# 文本编码（与 gen3r 同构）
# ══════════════════════════════════════════════════════════════════════════════

def encode_text(
    text: str,
    tokenizer,
    text_encoder,
    device,
    max_length: int = 512,
    dropout_prob: float = 0.0,
) -> torch.Tensor:
    """将文本编码为变长 T5 embedding（去掉 padding）。

    Returns:
        prompt_embed : Tensor [actual_len, dim]
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
        )[0]

    return embeds[0, : seq_len[0]]


# ══════════════════════════════════════════════════════════════════════════════
# Plücker 相机控制
# ══════════════════════════════════════════════════════════════════════════════

def _custom_meshgrid(*args):
    return torch.meshgrid(*args, indexing="ij")


def _ray_condition(K_pix: torch.Tensor, c2w: torch.Tensor, H: int, W: int, device) -> torch.Tensor:
    """复刻官方 [ray_condition](VideoX-Fun/videox_fun/data/utils.py:243)。

    Args:
        K_pix : [B, V, 4]   像素坐标 (fx, fy, cx, cy)
        c2w   : [B, V, 4, 4]
    Returns:
        plucker : [B, V, H, W, 6]
    """
    B = K_pix.shape[0]
    j, i = _custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5
    j = j.reshape([1, 1, H * W]).expand([B, 1, H * W]) + 0.5

    fx, fy, cx, cy = K_pix.chunk(4, dim=-1)
    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)
    rays_o = c2w[..., :3, 3]
    rays_o = rays_o[:, :, None].expand_as(rays_d)
    rays_dxo = torch.cross(rays_o, rays_d, dim=-1)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)
    return plucker


def _get_relative_pose_first_frame_id(c2ws: torch.Tensor) -> torch.Tensor:
    """复刻官方 [get_relative_pose](VideoX-Fun/videox_fun/data/utils.py:226)：
    将所有相机 pose 表达为以第 0 帧为 identity 的相对 pose。

    Args:
        c2ws : [F, 4, 4] OpenCV c2w
    Returns:
        rel_c2w : [F, 4, 4] 首帧 == identity
    """
    F = c2ws.shape[0]
    abs_w2cs = torch.linalg.inv(c2ws)  # [F, 4, 4]
    target_cam_c2w = torch.eye(4, device=c2ws.device, dtype=c2ws.dtype)
    abs2rel = target_cam_c2w @ abs_w2cs[0]  # [4, 4]

    out = [target_cam_c2w]
    for i in range(1, F):
        out.append(abs2rel @ c2ws[i])
    return torch.stack(out, dim=0)


def build_camera_control(
    c2ws: torch.Tensor,
    Ks: torch.Tensor,
    H: int,
    W: int,
    device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """从 [F,4,4] c2w + [F,3,3] K 构造像素分辨率的 Plücker tensor。

    与官方 [process_pose_file](VideoX-Fun/videox_fun/data/utils.py:278)
    完全等价（只是输入是已解析的张量而不是文本文件，规避跳第一行 / IO 重复）。

    Args:
        c2ws : [F, 4, 4]   OpenCV c2w，**未归一化**
        Ks   : [F, 3, 3]   像素内参（fx_pixel, fy_pixel, cx_pixel, cy_pixel）

    Returns:
        plucker : [F, 6, H, W]   像素分辨率 Plücker（与官方 process_pose_file 同 layout）
    """
    c2ws = c2ws.to(device=device, dtype=torch.float32)
    Ks = Ks.to(device=device, dtype=torch.float32)
    Frames = c2ws.shape[0]

    # 首帧对齐 → identity
    rel_c2w = _get_relative_pose_first_frame_id(c2ws)  # [F,4,4]

    # 抽取 fx,fy,cx,cy（像素单位）
    fx = Ks[:, 0, 0]
    fy = Ks[:, 1, 1]
    cx = Ks[:, 0, 2]
    cy = Ks[:, 1, 2]
    K_pix = torch.stack([fx, fy, cx, cy], dim=-1)  # [F, 4]

    K_pix = K_pix.unsqueeze(0)        # [1, F, 4]
    rel_c2w = rel_c2w.unsqueeze(0)    # [1, F, 4, 4]

    plucker = _ray_condition(K_pix, rel_c2w, H, W, device=device)  # [1,F,H,W,6]
    plucker = plucker[0].permute(0, 3, 1, 2).contiguous()           # [F, 6, H, W]
    return plucker.to(dtype)


def chunk_camera_control(
    plucker_fchw: torch.Tensor,
    num_frames: int,
) -> torch.Tensor:
    """把 [F,6,H,W] plücker 折叠成 [1, 24, F_lat, H, W]，供 transformer.y_camera 使用。

    复刻官方 pipeline 中 `control_camera_video → control_camera_latents` 的流程：

        control_camera_video = plucker.permute([3,0,1,2]).unsqueeze(0)  # [1,6,F,H,W]
        repeat first frame x4 along temporal:                            # [1,6,F+3,H,W]
            -> transpose to [1, F+3, 6, H, W]
            -> view  [1, (F+3)/4, 4, 6, H, W]
            -> transpose(2,3) [1, F_lat, 6, 4, H, W]
            -> view  [1, F_lat, 24, H, W]
            -> transpose(1,2) [1, 24, F_lat, H, W]

    Args:
        plucker_fchw : [F, 6, H, W] (float, on device)
        num_frames   : F（用来做内部断言）

    Returns:
        y_camera : [1, 24, F_lat, H, W]
    """
    assert plucker_fchw.shape[0] == num_frames, \
        f"plucker frames {plucker_fchw.shape[0]} != num_frames {num_frames}"
    assert (num_frames - 1) % 4 == 0 or num_frames == 1, \
        f"num_frames={num_frames} 必须满足 (F-1)%4==0（wan2.2 时序压缩约束）"

    plucker = plucker_fchw.permute(1, 0, 2, 3).unsqueeze(0)  # [1,6,F,H,W]
    plucker = torch.cat([
        torch.repeat_interleave(plucker[:, :, 0:1], repeats=4, dim=2),
        plucker[:, :, 1:],
    ], dim=2).transpose(1, 2)  # [1, F+3, 6, H, W]

    b, f, c, h, w = plucker.shape
    assert f % 4 == 0
    plucker = plucker.contiguous().view(b, f // 4, 4, c, h, w).transpose(2, 3)
    plucker = plucker.contiguous().view(b, f // 4, c * 4, h, w).transpose(1, 2)
    return plucker  # [1, 24, F_lat, H, W]


# ══════════════════════════════════════════════════════════════════════════════
# I2V 条件（mask_latents + masked_video_latents）
# ══════════════════════════════════════════════════════════════════════════════

def _resize_mask_for_latent(
    mask_chunked: torch.Tensor, latent: torch.Tensor, process_first_frame_only: bool = True
) -> torch.Tensor:
    """复刻官方 [resize_mask](VideoX-Fun/videox_fun/pipeline/pipeline_wan2_2_fun_control.py:100)。

    Args:
        mask_chunked : [B, 4, F_lat, H, W]   像素分辨率（chunked）
        latent       : [B, C, F_lat, H_lat, W_lat]  目标 latent 形状
    Returns:
        mask_latent  : [B, 4, F_lat, H_lat, W_lat]
    """
    latent_size = latent.size()
    target_size = list(latent_size[2:])

    if process_first_frame_only:
        target_size[0] = 1
        first_resized = F.interpolate(
            mask_chunked[:, :, 0:1], size=target_size, mode="trilinear", align_corners=False
        )
        target_size = list(latent_size[2:])
        target_size[0] = target_size[0] - 1
        if target_size[0] != 0:
            rest_resized = F.interpolate(
                mask_chunked[:, :, 1:], size=target_size, mode="trilinear", align_corners=False
            )
            return torch.cat([first_resized, rest_resized], dim=2)
        return first_resized
    else:
        return F.interpolate(mask_chunked, size=target_size, mode="trilinear", align_corners=False)


def encode_inpaint_conditions(
    start_image_pixels: torch.Tensor,
    vae,
    num_frames: int,
    H: int,
    W: int,
    device,
    dtype: torch.dtype,
):
    """构造 wan2.2 I2V (start-image → 49 frames) 所需的全部 latent 级条件。

    输入：
        start_image_pixels : [3, H, W] float32 [0, 1]   (RGB 首帧)
    返回：
        y                  : [1, 52, F_lat, H_lat, W_lat]  bf16
                             = cat([mask_latents(4), masked_video_latents(48)], dim=1)
        masked_video_lat   : [1, 48, F_lat, H_lat, W_lat]  bf16  (后续 latent clamp 用)
        latent_mask        : [1, 1, F_lat, H_lat, W_lat]   bf16  (用于 latent clamp + temp_ts)

    严格遵循 [pipeline_wan2_2_fun_control.py L619-L685](VideoX-Fun/videox_fun/pipeline/pipeline_wan2_2_fun_control.py:619)
    的 init_video / mask_video 流程。
    """
    spatial = vae.spatial_compression_ratio
    temporal = vae.temporal_compression_ratio
    F_lat = (num_frames - 1) // temporal + 1
    H_lat = H // spatial
    W_lat = W // spatial

    # ── init_video（tile 首帧 → F 帧）和 mask_condition ─────────────────────
    # ⚠️ 重要：必须传 float [0,1] 给 VaeImageProcessor.preprocess。
    # 若传 uint8 [0,255]，preprocess 内部 normalize=2*x-1 会得到 [-1, 509]，
    # 把 VAE 输入放大 255×，编码器 tanh 饱和后整段视频几乎全黑。
    # （官方 get_image_to_video_latent + pipeline 的 uint8 路径其实也有这个问题，
    #  只是 5B Control-Camera 这条链路平时极少有人完整跑过。）
    start_cpu = start_image_pixels.detach().to(device="cpu").float().clamp(0, 1)  # [3,H,W] in [0,1]
    init_video = start_cpu.unsqueeze(1).unsqueeze(0).repeat(1, 1, num_frames, 1, 1)  # [1,3,F,H,W]
    mask_video = torch.zeros((1, 1, num_frames, H, W), dtype=torch.float32)
    mask_video[:, :, 1:] = 1.0  # 与官方 mask_processor(do_binarize=True) 阈值化结果一致

    # ── VaeImageProcessor 预处理（init_video 走 normalize 到 [-1,1]，mask 走 binarize）
    from diffusers.image_processor import VaeImageProcessor
    image_processor = VaeImageProcessor(vae_scale_factor=spatial)
    mask_processor = VaeImageProcessor(
        vae_scale_factor=spatial, do_normalize=False, do_binarize=True, do_convert_grayscale=True
    )

    # init_video → preprocess → [1, 3, F, H, W] in [-1, 1]（CPU，float32）
    init_video_2d = rearrange(init_video, "b c f h w -> (b f) c h w")
    init_video_2d = image_processor.preprocess(init_video_2d, height=H, width=W).to(dtype=torch.float32)
    init_video = rearrange(init_video_2d, "(b f) c h w -> b c f h w", f=num_frames)

    # mask_video → preprocess → [1, 1, F, H, W] in {0, 1}（CPU，float32）
    mask_2d = rearrange(mask_video, "b c f h w -> (b f) c h w")
    mask_condition = mask_processor.preprocess(mask_2d, height=H, width=W).to(dtype=torch.float32)
    mask_condition = rearrange(mask_condition, "(b f) c h w -> b c f h w", f=num_frames)

    # ── masked_video = init_video * (mask < 0.5)，只保留首帧（CPU 上做避免 OOM） ─
    masked_video = init_video * (torch.tile(mask_condition, [1, 3, 1, 1, 1]) < 0.5)

    # ── VAE encode masked_video ──────────────────────────────────────────────
    # ⚠️ Bug fix：时序 VAE 编码 49 帧 masked_video 时，第 1-48 帧全为 0.0（灰色），
    # 灰色的时序上下文会污染第 0 帧（start image）的 latent 编码，
    # 导致 masked_video_lat[:, :, 0] 解码出来呈青色/teal 而非正确颜色。
    # 修复：单独编码首帧（1 帧上下文），得到干净的 start_lat，
    # 再替换 masked_video_lat 的第 0 帧 latent，使 apply_mask_clamp 锚点颜色正确。
    masked_video = masked_video.to(device=device, dtype=vae.dtype)
    with torch.no_grad():
        masked_video_lat = vae.encode(masked_video)[0].mode()  # [1, 48, F_lat, H_lat, W_lat]

        # 单独编码首帧以消除灰色帧时序污染
        start_frame_1f = init_video[:, :, 0:1].to(device=device, dtype=vae.dtype)  # [1, 3, 1, H, W]
        start_lat_1f = vae.encode(start_frame_1f)[0].mode()  # [1, 48, 1, H_lat, W_lat]
        # 用干净的单帧 latent 替换第 0 帧，消除时序污染
        masked_video_lat = masked_video_lat.clone()
        masked_video_lat[:, :, 0:1] = start_lat_1f

    masked_video_lat = masked_video_lat.to(dtype=dtype)

    # ── mask_condition chunk 折叠 → [1, 4, F_lat, H, W] ──────────────────────
    mask_condition = torch.cat([
        torch.repeat_interleave(mask_condition[:, :, 0:1], repeats=4, dim=2),
        mask_condition[:, :, 1:],
    ], dim=2)  # [1, 1, F+3, H, W]
    bs = mask_condition.shape[0]
    mask_condition = mask_condition.view(
        bs, mask_condition.shape[2] // 4, 4, H, W
    ).transpose(1, 2)  # [1, 4, F_lat, H, W]

    mask_condition = mask_condition.to(device=device)
    mask_latents_4ch = _resize_mask_for_latent(
        1 - mask_condition, masked_video_lat, process_first_frame_only=True
    ).to(dtype=dtype)  # [1, 4, F_lat, H_lat, W_lat]

    # ── y = cat([mask_latents(4), masked_video_latents(48)], dim=1) ─────────
    y = torch.cat([mask_latents_4ch, masked_video_lat], dim=1)  # [1, 52, F_lat, H_lat, W_lat]

    # ── latent_mask（spatial_compression_ratio>=16 路径用）──────────────────
    # 取 mask_condition 第一通道（4 通道里都一样），插值到 latent shape
    latent_mask = F.interpolate(
        mask_condition[:, :1], size=masked_video_lat.size()[-3:],
        mode="trilinear", align_corners=True,
    ).to(device=device, dtype=dtype)  # [1, 1, F_lat, H_lat, W_lat]

    # 官方逻辑：若首帧 latent_mask 全 0（始终成立，因为我们 mask=0 at f0），
    # 则把后续帧 mask 置 1，确保 clamp 只锁首帧
    if not latent_mask[:, :, 0, :, :].any():
        latent_mask = latent_mask.clone()
        latent_mask[:, :, 1:, :, :] = 1.0

    return y, masked_video_lat, latent_mask


def apply_mask_clamp(
    latent: torch.Tensor,
    masked_video_lat: torch.Tensor,
    latent_mask: torch.Tensor,
) -> torch.Tensor:
    """官方 I2V 关键 trick：每步去噪后把首帧 latent 强制对齐到 masked_video_lat。

    `latent = (1 - latent_mask) * masked_video_lat + latent_mask * latent`

    其中 latent_mask 在首帧 == 0（强制等于 GT），其他帧 == 1（保留去噪结果）。
    """
    return (1 - latent_mask) * masked_video_lat + latent_mask * latent


# ══════════════════════════════════════════════════════════════════════════════
# Transformer 单步前向
# ══════════════════════════════════════════════════════════════════════════════

def _make_per_token_timestep(
    t_value: float,
    latent_mask: torch.Tensor,
    seq_len: int,
    patch_size: tuple,
    dtype: torch.dtype,
    device,
):
    """复刻 [pipeline_wan2_2_fun_control.py L832-L843](VideoX-Fun/videox_fun/pipeline/pipeline_wan2_2_fun_control.py:832)。

    spatial_compression_ratio>=16 + I2V 时，timestep 不是标量，而是 per-token 张量：
        temp_ts = (latent_mask[0,0,:,::patch_h,::patch_w] * t).flatten()
        若长度 > seq_len 截断；不足则补 t
        最终 [B, seq_len]
    """
    patch_h, patch_w = patch_size[1], patch_size[2]
    temp_ts = (latent_mask[0, 0, :, ::patch_h, ::patch_w] * t_value).flatten().to(device=device)
    if temp_ts.size(0) >= seq_len:
        temp_ts = temp_ts[:seq_len]
    else:
        pad = temp_ts.new_ones(seq_len - temp_ts.size(0)) * t_value
        temp_ts = torch.cat([temp_ts, pad])
    return temp_ts.unsqueeze(0)  # [1, seq_len]


def transformer_forward(
    transformer,
    z: torch.Tensor,
    t_value: float,
    prompt_embeds: list,
    neg_embeds: list,
    seq_len: int,
    control_latents: torch.Tensor,
    control_camera_latents: torch.Tensor,
    latent_mask: torch.Tensor,
    cfg_scale: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Wan2.2-Fun-5B-Control-Camera 单步前向，支持 CFG。

    Args:
        z                       : [1, 48, F_lat, H_lat, W_lat]  当前噪声 latent (bf16)
        t_value                 : float / int   sigma * 1000，标量
        prompt_embeds, neg_embeds : list of [L, 4096] embed
        seq_len                 : int  (固定)
        control_latents         : [1, 52, F_lat, H_lat, W_lat]  bf16
        control_camera_latents  : [1, 24, F_lat, H, W]          bf16
        latent_mask             : [1, 1, F_lat, H_lat, W_lat]   bf16
        cfg_scale               : float (>1 时启用 CFG)

    Returns:
        pred : [1, 48, F_lat, H_lat, W_lat]  fp32
    """
    device = z.device
    patch_size = transformer.config.patch_size if hasattr(transformer, "config") \
        else getattr(transformer, "patch_size", (1, 2, 2))

    do_cfg = cfg_scale > 1.0
    if do_cfg:
        z_in = torch.cat([z, z], dim=0)
        ctx_in = neg_embeds + prompt_embeds
        ctrl_in = torch.cat([control_latents, control_latents], dim=0)
        cam_in = torch.cat([control_camera_latents, control_camera_latents], dim=0)
    else:
        z_in = z
        ctx_in = prompt_embeds
        ctrl_in = control_latents
        cam_in = control_camera_latents

    # per-token timestep (因为 5B spatial_compression_ratio == 16)
    t_per_token = _make_per_token_timestep(
        float(t_value), latent_mask, seq_len, patch_size, dtype, device
    )  # [1, seq_len]
    t_in = t_per_token.expand(z_in.shape[0], t_per_token.size(1))

    with torch.autocast("cuda", dtype):
        pred = transformer(
            x=z_in,
            t=t_in,
            context=ctx_in,
            seq_len=seq_len,
            y=ctrl_in,
            y_camera=cam_in,
            full_ref=None,
            clip_fea=None,
        )

    if do_cfg:
        pred_uncond, pred_cond = pred.chunk(2)
        pred = pred_uncond.to(torch.float32) + cfg_scale * (
            pred_cond.to(torch.float32) - pred_uncond.to(torch.float32)
        )
    else:
        pred = pred.to(torch.float32)
    return pred


# ══════════════════════════════════════════════════════════════════════════════
# VAE Decode → MP4
# ══════════════════════════════════════════════════════════════════════════════

def decode_rgb_video(
    final_latent: torch.Tensor,
    vae,
    video_path: str,
    fps: int = 16,
) -> str:
    """从 [1, 48, F_lat, H_lat, W_lat] latent 解码 RGB 视频并保存。

    decode 前 enable_tiling 防 OOM（1280×704×49 比 gen3r 的 560×560×17 大约 9 倍）。
    """
    if hasattr(vae, "enable_tiling"):
        try:
            vae.enable_tiling()
        except Exception:
            pass

    final_latent = final_latent.to(dtype=vae.dtype)
    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            video = vae.decode(final_latent).sample  # [1, 3, F, H, W] in [-1, 1]

    video_01 = (video / 2 + 0.5).clamp(0, 1)
    frames_np = (
        video_01[0].permute(1, 2, 3, 0).float().cpu().numpy() * 255
    ).astype(np.uint8)  # [F, H, W, 3]
    frames = list(frames_np)

    os.makedirs(os.path.dirname(os.path.abspath(video_path)), exist_ok=True)
    from diffusers.utils import export_to_video
    export_to_video(frames, video_path, fps=fps)
    return video_path


# ══════════════════════════════════════════════════════════════════════════════
# 工具：camera txt 写入 / GT 视频保存（供 infer_only 直接复制）
# ══════════════════════════════════════════════════════════════════════════════

def write_camera_txt(
    c2ws: torch.Tensor,
    Ks: torch.Tensor,
    H: int,
    W: int,
    out_path: str,
) -> None:
    """与 gen3r [write_camera_txt](rl_train_new/rl_train/train/gen3r/gen3r_encode.py:307) 完全一致。"""
    c2ws_np = c2ws.float().cpu().numpy()
    Ks_np = Ks.float().cpu().numpy()
    lines = []
    for i, (K, c2w) in enumerate(zip(Ks_np, c2ws_np)):
        fx_n = K[0, 0] / W
        fy_n = K[1, 1] / H
        cx_n = K[0, 2] / W
        cy_n = K[1, 2] / H
        w2c = np.linalg.inv(c2w)
        w2c_flat = w2c[:3, :].flatten()
        vals = (
            f"{i} {fx_n:.10f} {fy_n:.10f} {cx_n:.10f} {cy_n:.10f} 0 0 "
            + " ".join(f"{v:.10f}" for v in w2c_flat)
        )
        lines.append(vals)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def save_gt_video(
    pixel_values: torch.Tensor,
    out_path: str,
    fps: int = 16,
) -> None:
    """[F, C, H, W] float32 [0,1] → MP4。"""
    from diffusers.utils import export_to_video
    frames = pixel_values.float().cpu().permute(0, 2, 3, 1).numpy()
    frames = (frames * 255).clip(0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    export_to_video(list(frames), out_path, fps=fps)
