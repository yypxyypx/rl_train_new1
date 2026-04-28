"""model_loader.py — Wan2.2-Fun-5B-Control-Camera 模型加载。

加载五件套：
    1. Wan2_2Transformer3DModel        — 主去噪 transformer（含 control_adapter）
    2. AutoencoderKLWan3_8             — 16x spatial / 4x temporal VAE
    3. WanT5EncoderModel               — umt5-xxl 文本编码器（可选 skip 用预算 cache）
    4. AutoTokenizer                   — google/umt5-xxl 子目录
    5. FlowMatchEulerDiscreteScheduler — 去噪 scheduler

完全照搬 bundled VideoX-Fun 内
``examples/wan2.2_fun/predict_v2v_control_camera_5b.py`` 的加载流程，
只是把硬编码 path 改成参数化，并按 args.t5_embed_dir 决定是否加载 T5。

`pretrained_model_path` / ``config_path`` 默认值见 ``config.py``；
``videox_fun`` 包默认从仓库内 ``eval/infer/wan2.2/VideoX-Fun`` 加载（可用 ``VIDEOX_FUN_ROOT`` 覆盖）。

5B 是 single-stage（`transformer_combination_type=single`），不加载 transformer_2；
camera control 模型不需要 CLIP image encoder，clip_image_encoder 始终为 None。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import torch
from omegaconf import OmegaConf

from paths import videox_fun_root

# ─── 把 videox_fun 加进 path ─────────────────────────────────────────────────
_VIDEOX_ROOT = videox_fun_root()
if str(_VIDEOX_ROOT) not in sys.path:
    sys.path.insert(0, str(_VIDEOX_ROOT))

from diffusers import FlowMatchEulerDiscreteScheduler  # noqa: E402

from videox_fun.models import (AutoencoderKLWan3_8, AutoTokenizer,  # noqa: E402
                               Wan2_2Transformer3DModel, WanT5EncoderModel)
from videox_fun.utils.utils import filter_kwargs  # noqa: E402


WEIGHT_DTYPE = torch.bfloat16


def _maybe_load_extra_state_dict(model, ckpt_path: Optional[str]) -> None:
    """若 ckpt_path 不为 None，从 safetensors / pt 文件加载到 model。"""
    if not ckpt_path:
        return
    print(f"[model_loader] Loading extra checkpoint from: {ckpt_path}")
    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(ckpt_path)
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu")
    state_dict = state_dict.get("state_dict", state_dict)
    m, u = model.load_state_dict(state_dict, strict=False)
    print(f"[model_loader]   missing={len(m)}, unexpected={len(u)}")


def load_all_models(
    args,
    config,
    device: torch.device,
    weight_dtype: torch.dtype = WEIGHT_DTYPE,
) -> dict:
    """加载 Wan2.2-Fun-5B-Control-Camera 全部组件。

    Returns:
        dict 含 keys:
            transformer    : Wan2_2Transformer3DModel
            vae            : AutoencoderKLWan3_8
            text_encoder   : WanT5EncoderModel | None  (若 args.t5_embed_dir 设置则 None)
            tokenizer      : AutoTokenizer | None
            scheduler      : FlowMatchEulerDiscreteScheduler
            config         : OmegaConf
    """
    model_root = args.pretrained_model_path

    # ── Transformer ──────────────────────────────────────────────────────────
    transformer_subpath = config["transformer_additional_kwargs"].get(
        "transformer_low_noise_model_subpath", "./"
    )
    transformer_dir = os.path.join(model_root, transformer_subpath)
    print(f"[model_loader] Loading Wan2_2Transformer3DModel from: {transformer_dir}")
    transformer = Wan2_2Transformer3DModel.from_pretrained(
        transformer_dir,
        transformer_additional_kwargs=OmegaConf.to_container(
            config["transformer_additional_kwargs"]
        ),
        torch_dtype=weight_dtype,
    )
    _maybe_load_extra_state_dict(transformer, getattr(args, "transformer_path", None))
    transformer = transformer.to(dtype=weight_dtype)

    # ── VAE3_8 ───────────────────────────────────────────────────────────────
    vae_subpath = config["vae_kwargs"].get("vae_subpath", "Wan2.2_VAE.pth")
    vae_path = os.path.join(model_root, vae_subpath)
    print(f"[model_loader] Loading AutoencoderKLWan3_8 from: {vae_path}")
    vae = AutoencoderKLWan3_8.from_pretrained(
        vae_path,
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    ).to(dtype=weight_dtype)

    # ── T5 / Tokenizer（按需）───────────────────────────────────────────────
    text_encoder = None
    tokenizer = None
    if not getattr(args, "t5_embed_dir", None):
        tok_subpath = config["text_encoder_kwargs"].get("tokenizer_subpath", "google/umt5-xxl")
        tok_path = os.path.join(model_root, tok_subpath)
        print(f"[model_loader] Loading tokenizer from: {tok_path}")
        tokenizer = AutoTokenizer.from_pretrained(tok_path)

        te_subpath = config["text_encoder_kwargs"].get(
            "text_encoder_subpath", "models_t5_umt5-xxl-enc-bf16.pth"
        )
        te_path = os.path.join(model_root, te_subpath)
        print(f"[model_loader] Loading WanT5EncoderModel from: {te_path}")
        text_encoder = WanT5EncoderModel.from_pretrained(
            te_path,
            additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        ).eval()
        text_encoder.requires_grad_(False)
    else:
        print(f"[model_loader] --t5_embed_dir is set ({args.t5_embed_dir}); "
              "skipping T5 load, will use precomputed embeddings.")

    # ── Scheduler ────────────────────────────────────────────────────────────
    scheduler_kwargs = OmegaConf.to_container(config["scheduler_kwargs"])
    print(f"[model_loader] Building FlowMatchEulerDiscreteScheduler with kwargs: {scheduler_kwargs}")
    scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, scheduler_kwargs)
    )

    # 冻结 VAE，transformer 由训练侧决定是否冻结
    vae.requires_grad_(False)
    vae.eval()

    return {
        "transformer": transformer,
        "vae": vae,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "scheduler": scheduler,
        "config": config,
    }


# ══════════════════════════════════════════════════════════════════════════════
# T5 缓存加载（grpo_engine / infer_only 都会调用）
# ══════════════════════════════════════════════════════════════════════════════

def load_t5_embeds(args, sample_id: str, dataset_name: str):
    """从 args.t5_embed_dir 加载预计算的 prompt + neg embedding。

    与 gen3r 同名函数接口完全一致：
        Returns: (prompt_embed_cpu [L, 4096], neg_embed_cpu [L_neg, 4096]) bf16
    """
    embed_dir = Path(args.t5_embed_dir)
    prompt_path = embed_dir / dataset_name / f"{sample_id}.pt"
    neg_path = embed_dir / "neg_embed.pt"

    if not prompt_path.exists():
        raise FileNotFoundError(f"T5 prompt embed not found: {prompt_path}")
    if not neg_path.exists():
        raise FileNotFoundError(f"T5 neg embed not found: {neg_path}")

    prompt_embed = torch.load(str(prompt_path), map_location="cpu")
    neg_embed = torch.load(str(neg_path), map_location="cpu")
    return prompt_embed, neg_embed
