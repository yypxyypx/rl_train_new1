"""model_loader_v2.py — Gen3R GRPO 精简模型加载器。

相比 model_loader.py 的优化：
  - 不加载 T5（用预计算 embedding 替代）
  - 不加载 geo_adapter（训练流程中从未调用）
  - 不加载 VGGT（训练流程中从未调用）
  - 额外加载 reference_transformer（用于 KL 散度计算），eval 模式无梯度
  - 支持 T5 embedding 缓存目录（t5_embed_dir）

显存占用：
  - wan_vae       ~0.3 GB
  - clip_image_encoder  ~1.2 GB
  - transformer   ~2.6 GB (可训练)
  - ref_transformer ~2.6 GB (frozen，常驻 GPU)
  合计 ~6.7 GB，加上激活约 10-15 GB，32 GB 显存完全够用
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import torch
from diffusers.optimization import get_scheduler
from omegaconf import OmegaConf

_HERE = Path(__file__).resolve().parent
_GEN3R_PKG = _HERE / "Gen3R"
if str(_GEN3R_PKG) not in sys.path:
    sys.path.insert(0, str(_GEN3R_PKG))

from gen3r.models import (  # noqa: E402
    AutoencoderKLWan,
    CLIPModel,
    WanTransformer3DModel,
)


def _resolve_ckpt_subpath(
    root: str,
    preferred: str,
    fallbacks: list[str],
    *,
    need_config_json: bool = False,
) -> tuple[str, str]:
    candidates = [preferred] + [f for f in fallbacks if f and f != preferred]
    for sub in candidates:
        if not sub:
            continue
        p = os.path.normpath(os.path.join(root, sub))
        if need_config_json:
            if os.path.isfile(os.path.join(p, "config.json")):
                return p, sub
        else:
            if os.path.isdir(p) or os.path.isfile(p):
                return p, sub
    raise FileNotFoundError(
        f"Cannot resolve checkpoint under {root!r}. Tried: {candidates!r}"
    )


def load_models(args, config: OmegaConf, device, weight_dtype: torch.dtype) -> dict:
    """加载 GRPO 训练所需的精简模型集合。

    返回 dict，keys：
        wan_vae             — 冻结，Rollout encode/decode 用
        clip_image_encoder  — 冻结，控制图像编码用
        transformer         — 可训练
        ref_transformer     — 冻结，KL 散度参考策略

    不包含：tokenizer, text_encoder, geo_adapter, vggt
    """
    rank = int(os.environ.get("RANK", 0))

    def _log(msg: str):
        if rank == 0:
            print(msg)

    # ── WAN VAE（冻结） ───────────────────────────────────────────────────────
    vae_yaml = config["vae_kwargs"].get("vae_subpath", "wan_vae")
    vae_path, vae_used = _resolve_ckpt_subpath(
        args.pretrained_model_path, vae_yaml, ["wan_vae"], need_config_json=True
    )
    wan_vae = AutoencoderKLWan.from_pretrained(
        vae_path,
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    ).to(device=device, dtype=weight_dtype).eval()
    wan_vae.requires_grad_(False)
    _log(f"[ModelV2] WAN VAE loaded from {vae_used}")

    # ── CLIP 图像编码器（冻结） ───────────────────────────────────────────────
    clip_yaml = config["image_encoder_kwargs"].get(
        "image_encoder_subpath",
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    )
    clip_path, clip_used = _resolve_ckpt_subpath(
        args.pretrained_model_path, clip_yaml, ["clip_image_encoder"], need_config_json=True
    )
    clip_image_encoder = CLIPModel.from_pretrained(clip_path).to(
        device=device, dtype=weight_dtype
    ).eval()
    clip_image_encoder.requires_grad_(False)
    _log(f"[ModelV2] CLIP loaded from {clip_used}")

    # ── Transformer（可训练） ─────────────────────────────────────────────────
    tr_yaml = config["transformer_additional_kwargs"].get("transformer_subpath", "transformer")
    tr_path, tr_used = _resolve_ckpt_subpath(
        args.pretrained_model_path, tr_yaml, ["transformer", "."], need_config_json=True
    )
    transformer = WanTransformer3DModel.from_pretrained(
        tr_path,
        transformer_additional_kwargs=OmegaConf.to_container(
            config["transformer_additional_kwargs"]
        ),
        torch_dtype=weight_dtype,
    ).to(device=device)
    _log(f"[ModelV2] Transformer loaded from {tr_used}")

    # 可选：加载额外 transformer 权重
    if getattr(args, "transformer_path", None):
        _load_extra_weights(transformer, args.transformer_path)
        _log(f"[ModelV2] Extra transformer weights loaded from {args.transformer_path}")

    # ── Reference Transformer（冻结，KL 计算用） ───────────────────────────────
    ref_transformer = WanTransformer3DModel.from_pretrained(
        tr_path,
        transformer_additional_kwargs=OmegaConf.to_container(
            config["transformer_additional_kwargs"]
        ),
        torch_dtype=weight_dtype,
    ).to(device=device).eval()
    ref_transformer.requires_grad_(False)
    # 如果加载了额外权重，reference 也同步加载（保持参考策略与训练起点一致）
    if getattr(args, "transformer_path", None):
        _load_extra_weights(ref_transformer, args.transformer_path)
    _log("[ModelV2] Reference Transformer loaded (frozen, for KL divergence)")

    if rank == 0:
        trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
        total = sum(p.numel() for p in transformer.parameters())
        _log(f"[ModelV2] Trainable params: {trainable / 1e9:.3f}B / {total / 1e9:.3f}B total")

    return {
        "wan_vae": wan_vae,
        "clip_image_encoder": clip_image_encoder,
        "transformer": transformer,
        "ref_transformer": ref_transformer,
    }


def _load_extra_weights(model: torch.nn.Module, path: str) -> None:
    """从 safetensors 或 .pt 加载额外权重（strict=False）。"""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(path)
    else:
        state = torch.load(path, map_location="cpu")
    state = state.get("state_dict", state)
    model.load_state_dict(state, strict=False)


def load_t5_embeds(args, sample_id: str, dataset_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """从缓存目录加载预计算的 T5 embedding。

    Returns:
        prompt_embed : [L, 4096]
        neg_embed    : [L_neg, 4096]

    若 t5_embed_dir 未设置，抛出 RuntimeError 提示用户先运行 t5_precompute.py。
    """
    if not getattr(args, "t5_embed_dir", None):
        raise RuntimeError(
            "args.t5_embed_dir is not set. "
            "Please run t5_precompute.py first, then pass --t5_embed_dir."
        )

    embed_dir = Path(args.t5_embed_dir)
    prompt_path = embed_dir / dataset_name / f"{sample_id}.pt"
    neg_path = embed_dir / "neg_embed.pt"

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"T5 embedding not found: {prompt_path}. "
            f"Run t5_precompute.py for dataset '{dataset_name}'."
        )
    if not neg_path.exists():
        raise FileNotFoundError(
            f"Negative T5 embedding not found: {neg_path}. "
            f"Run t5_precompute.py to generate it."
        )

    prompt_embed = torch.load(str(prompt_path), map_location="cpu")
    neg_embed = torch.load(str(neg_path), map_location="cpu")
    return prompt_embed, neg_embed


# ══════════════════════════════════════════════════════════════════════════════
# 可训练参数设置
# ══════════════════════════════════════════════════════════════════════════════

def setup_trainable_params(transformer, trainable_modules: Optional[list[str]]) -> None:
    """设置 transformer 的可训练参数。

    Args:
        trainable_modules : None → 全部可训练；
                            list[str] → 只训练名称中包含这些子串的参数
    """
    transformer.train()
    if trainable_modules:
        transformer.requires_grad_(False)
        for name, param in transformer.named_parameters():
            if any(mod in name for mod in trainable_modules):
                param.requires_grad_(True)
    else:
        transformer.requires_grad_(True)

    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
        total = sum(p.numel() for p in transformer.parameters())
        print(f"[ModelV2] Setup trainable: {trainable / 1e9:.3f}B / {total / 1e9:.3f}B")


# ══════════════════════════════════════════════════════════════════════════════
# 优化器 & LR Scheduler
# ══════════════════════════════════════════════════════════════════════════════

def create_optimizer(
    transformer,
    lr: float,
    weight_decay: float,
    use_8bit: bool = False,
) -> torch.optim.Optimizer:
    """Create AdamW optimizer.

    Args:
        use_8bit: True 时用 bitsandbytes.optim.AdamW8bit，把 optimizer state
                  从 4B/param 压到 1B/param（省 ~4.8GB on 1.6B model）。
                  4090 24GB 必须开；5090 32GB 可关以保数值稳定。
                  bitsandbytes 缺失时打 warning 并 fallback 到 torch.optim.AdamW。
    """
    params = [p for p in transformer.parameters() if p.requires_grad]
    if use_8bit:
        try:
            import bitsandbytes as bnb
            n_params = sum(p.numel() for p in params)
            print(f"[ModelV2] Using bitsandbytes AdamW8bit "
                  f"(saves ~{n_params * 3 / 1e9:.2f} GB optim state on "
                  f"{n_params / 1e9:.2f}B params)")
            return bnb.optim.AdamW8bit(params, lr=lr, betas=(0.9, 0.999),
                                       weight_decay=weight_decay, eps=1e-8)
        except ImportError:
            print("[ModelV2] WARNING: bitsandbytes not installed, "
                  "falling back to torch.optim.AdamW (will likely OOM on 24GB). "
                  "Run: pip install bitsandbytes")
    return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999),
                             weight_decay=weight_decay, eps=1e-8)


def create_lr_scheduler(optimizer, scheduler_type: str, warmup_steps: int):
    return get_scheduler(
        scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=1_000_000,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint 保存
# ══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(transformer, output_dir: str, step: int | str, rank: int) -> None:
    if rank != 0:
        return
    import json
    from safetensors.torch import save_file

    ckpt_dir = Path(output_dir) / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cpu_state = {k: v.cpu() for k, v in transformer.state_dict().items()}
    save_file(cpu_state, str(ckpt_dir / "diffusion_pytorch_model.safetensors"))
    cfg_dict = {k: v for k, v in dict(transformer.config).items() if k != "dtype"}
    with open(str(ckpt_dir / "config.json"), "w") as f:
        json.dump(cfg_dict, f, indent=2)
    print(f"[ModelV2] Checkpoint saved to {ckpt_dir}")
