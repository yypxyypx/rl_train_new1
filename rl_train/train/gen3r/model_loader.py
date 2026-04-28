"""model_loader.py — Gen3R GRPO 精简模型加载器。

相比原版的优化：
  - 不加载 T5（用预计算 embedding 替代）
  - 不加载 geo_adapter / VGGT（训练流程中从未调用）
  - kl_coeff=0 时完全跳过 ref_transformer 加载（省 ~6GB 网络 IO）
  - kl_coeff>0 时直接加载 ref_transformer 到 GPU 常驻（reward worker 架构
    下 Phase 2 不再 offload 模型，ref 留在 GPU 省 Phase 3 拷贝时间）

显存占用（kl_coeff=0）：
  - wan_vae              ~0.3 GB
  - clip_image_encoder   ~1.2 GB
  - transformer          ~2.6 GB (可训练)
  合计 ~4.1 GB；kl_coeff>0 再加 ~2.6 GB 的 ref_transformer。
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
        ref_transformer     — 冻结，KL 散度参考策略（kl_coeff=0 时为 None）

    不包含：tokenizer, text_encoder, geo_adapter, vggt
    """
    rank = int(os.environ.get("RANK", 0))
    kl_coeff = getattr(args, "kl_coeff", 0.01)

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

    if rank == 0:
        trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
        total = sum(p.numel() for p in transformer.parameters())
        _log(f"[ModelV2] Trainable params: {trainable / 1e9:.3f}B / {total / 1e9:.3f}B total")

    # ── Reference Transformer（冻结，KL 计算用） ───────────────────────────────
    # kl_coeff=0：完全跳过，节省 ~6GB 网络 IO（FUSE 存储约 6 分钟）
    # kl_coeff>0：直接加载到 GPU 常驻。新的 reward worker 架构下不再做
    #             Phase 2 模型 offload，所以 ref_transformer 留在 GPU 即可，
    #             避免 Phase 3 每步都 CPU↔GPU 拷贝。
    if kl_coeff == 0.0:
        _log("[ModelV2] kl_coeff=0: skipping ref_transformer load (~6GB saved)")
        ref_transformer = None
    else:
        _log("[ModelV2] Loading ref_transformer directly to GPU ...")
        ref_transformer = WanTransformer3DModel.from_pretrained(
            tr_path,
            transformer_additional_kwargs=OmegaConf.to_container(
                config["transformer_additional_kwargs"]
            ),
            torch_dtype=weight_dtype,
        ).to(device=device, dtype=weight_dtype).eval()
        ref_transformer.requires_grad_(False)
        if getattr(args, "transformer_path", None):
            _load_extra_weights(ref_transformer, args.transformer_path)
        _log("[ModelV2] ref_transformer ready on GPU")

    return {
        "wan_vae": wan_vae,
        "clip_image_encoder": clip_image_encoder,
        "transformer": transformer,
        "ref_transformer": ref_transformer,
    }


def join_ref_transformer(models: dict, device, weight_dtype: torch.dtype) -> None:
    """兼容老代码的空操作。新架构下 ref_transformer 已在 load_models 时直接上 GPU。"""
    return


def _load_extra_weights(model: torch.nn.Module, path: str) -> None:
    """从 safetensors 或 .pt 加载额外权重（strict=False）。"""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(path)
    else:
        state = torch.load(path, map_location="cpu")
    state = state.get("state_dict", state)
    model.load_state_dict(state, strict=False)


def load_t5_embeds(
    args, sample_id: str, dataset_name: str, sample_dir: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """加载预计算的 T5 embedding。

    查找顺序：
      1) **In-place（首选）**：{sample_dir}/prompt_embed.pt + {sample_dir}/neg_embed.pt
         由 t5_precompute_inplace.py 生成。每条样本独立两个 .pt。
      2) **Fallback（旧路径）**：{args.t5_embed_dir}/{dataset_name}/{sample_id}.pt
                                  + {args.t5_embed_dir}/neg_embed.pt
         由旧版 t5_precompute.py 生成。

    Returns:
        prompt_embed : [L, 4096]
        neg_embed    : [L_neg, 4096]
    """
    # ── 1) 优先 in-place ───────────────────────────────────────────────
    if sample_dir:
        sd = Path(sample_dir)
        in_prompt = sd / "prompt_embed.pt"
        in_neg = sd / "neg_embed.pt"
        if in_prompt.exists() and in_neg.exists():
            prompt_embed = torch.load(str(in_prompt), map_location="cpu")
            neg_embed = torch.load(str(in_neg), map_location="cpu")
            return prompt_embed, neg_embed

    # ── 2) Fallback 老路径 ─────────────────────────────────────────────
    if not getattr(args, "t5_embed_dir", None):
        hint = (f"sample_dir={sample_dir!r}" if sample_dir else "sample_dir=None")
        raise RuntimeError(
            f"T5 embedding not found in-place ({hint}) and "
            "args.t5_embed_dir is not set. Either run "
            "t5_precompute_inplace.py to write per-sample embeddings, "
            "or run t5_precompute.py and pass --t5_embed_dir."
        )

    embed_dir = Path(args.t5_embed_dir)
    prompt_path = embed_dir / dataset_name / f"{sample_id}.pt"
    neg_path = embed_dir / "neg_embed.pt"

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"T5 embedding not found in-place ({sample_dir}/prompt_embed.pt) "
            f"nor at fallback path ({prompt_path}). "
            f"Run t5_precompute_inplace.py or t5_precompute.py."
        )
    if not neg_path.exists():
        raise FileNotFoundError(
            f"Negative T5 embedding not found at {neg_path}. "
            f"Run t5_precompute_inplace.py or t5_precompute.py."
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

def save_checkpoint(
    transformer,
    output_dir: str,
    step: int | str,
    rank: int,
    *,
    tag_prefix: str = "checkpoint",
    training_state: Optional[dict] = None,
) -> Optional[Path]:
    """落盘一份 transformer 权重 + 配置 (+ 可选 training_state.json)。
    仅 rank0 写盘，其它 rank no-op。

    Args:
        tag_prefix     : 目录前缀。"checkpoint" / "rolling" / "permanent" 区分用途。
        training_state : 可序列化 dict（global_step / sampler_seed 等）；用于 resume。
    Returns:
        保存目录的 Path（仅 rank0），其它 rank 返回 None。
    """
    if rank != 0:
        return None
    import json
    from safetensors.torch import save_file

    ckpt_dir = Path(output_dir) / f"{tag_prefix}-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cpu_state = {k: v.cpu() for k, v in transformer.state_dict().items()}
    save_file(cpu_state, str(ckpt_dir / "diffusion_pytorch_model.safetensors"))
    cfg_dict = {k: v for k, v in dict(transformer.config).items() if k != "dtype"}
    with open(str(ckpt_dir / "config.json"), "w") as f:
        json.dump(cfg_dict, f, indent=2)
    if training_state is not None:
        with open(str(ckpt_dir / "training_state.json"), "w") as f:
            json.dump(training_state, f, indent=2, default=str)
    print(f"[ModelV2] {tag_prefix} saved to {ckpt_dir}"
          f"{' (+training_state)' if training_state is not None else ''}",
          flush=True)
    return ckpt_dir


# 模块级状态：记录上一份滚动 ckpt 路径，便于覆盖时删除
_LAST_ROLLING_CKPT: Optional[Path] = None


def save_rolling_checkpoint(
    transformer,
    output_dir: str,
    step: int,
    rank: int,
    *,
    training_state: Optional[dict] = None,
) -> Optional[Path]:
    """滚动 ckpt：每次保存新的同时删掉上一份。仅保留最新 1 份。

    用于断电/抢占恢复，不长期占盘。仅 rank0 写盘。
    """
    global _LAST_ROLLING_CKPT
    if rank != 0:
        return None
    new_dir = save_checkpoint(
        transformer, output_dir, step, rank,
        tag_prefix="rolling", training_state=training_state,
    )
    # 删除上一份滚动 ckpt（如果存在且不是当前这个）
    if _LAST_ROLLING_CKPT is not None and _LAST_ROLLING_CKPT.exists():
        if _LAST_ROLLING_CKPT.resolve() != (new_dir.resolve() if new_dir else None):
            try:
                import shutil
                shutil.rmtree(_LAST_ROLLING_CKPT)
                print(f"[ModelV2] removed previous rolling ckpt {_LAST_ROLLING_CKPT}", flush=True)
            except Exception as e:
                print(f"[ModelV2] WARN: failed to remove old rolling ckpt {_LAST_ROLLING_CKPT}: {e}",
                      flush=True)
    _LAST_ROLLING_CKPT = new_dir
    return new_dir


def save_permanent_checkpoint(
    transformer,
    output_dir: str,
    step: int,
    rank: int,
    *,
    keep_last_n: int = 0,
    training_state: Optional[dict] = None,
) -> Optional[Path]:
    """永久 ckpt：保存且不删除（除非启用 keep_last_n 回收策略）。"""
    if rank != 0:
        return None
    new_dir = save_checkpoint(
        transformer, output_dir, step, rank,
        tag_prefix="permanent", training_state=training_state,
    )
    if keep_last_n > 0:
        _gc_old_permanent(Path(output_dir), keep_last_n)
    return new_dir


# ══════════════════════════════════════════════════════════════════════════════
# Resume：自动找最新 ckpt + 加载权重 / 状态
# ══════════════════════════════════════════════════════════════════════════════

def find_latest_resume_checkpoint(output_dir: str) -> Optional[Path]:
    """在 output_dir 下找 step 最大的 resume 候选 ckpt。

    优先级（同 step 时）：rolling > permanent > checkpoint
    （rolling 是最近一次写的，但 permanent / checkpoint 也兼容。）
    """
    import re
    out = Path(output_dir)
    if not out.exists():
        return None
    pattern = re.compile(r"^(rolling|permanent|checkpoint)-(\d+)$")
    candidates: list[tuple[int, int, Path]] = []
    prio = {"rolling": 2, "permanent": 1, "checkpoint": 0}
    for p in out.iterdir():
        if not p.is_dir():
            continue
        m = pattern.match(p.name)
        if not m:
            continue
        # 必须真有权重才算有效 ckpt
        if not (p / "diffusion_pytorch_model.safetensors").exists():
            continue
        step = int(m.group(2))
        candidates.append((step, prio[m.group(1)], p))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][2]


def load_transformer_weights(transformer, ckpt_dir: Path, device, *, strict: bool = True) -> None:
    """从 ckpt_dir/diffusion_pytorch_model.safetensors 加载到 transformer。
    用 in-place state_dict 替换，调用方负责把 transformer 已经放在正确 device。
    """
    from safetensors.torch import load_file
    weights_path = ckpt_dir / "diffusion_pytorch_model.safetensors"
    state = load_file(str(weights_path), device="cpu")
    missing, unexpected = transformer.load_state_dict(state, strict=strict)
    if missing:
        print(f"[ModelV2] resume WARN: missing keys ({len(missing)}): "
              f"{missing[:3]}{'...' if len(missing) > 3 else ''}", flush=True)
    if unexpected:
        print(f"[ModelV2] resume WARN: unexpected keys ({len(unexpected)}): "
              f"{unexpected[:3]}{'...' if len(unexpected) > 3 else ''}", flush=True)
    # 加载完后再搬上 device，避免 cpu->gpu 占用 2× 临时显存
    transformer.to(device)


def load_training_state(ckpt_dir: Path) -> Optional[dict]:
    """读 ckpt_dir/training_state.json；不存在返回 None。"""
    import json
    state_path = ckpt_dir / "training_state.json"
    if not state_path.exists():
        return None
    with open(str(state_path), "r") as f:
        return json.load(f)


def register_existing_rolling_ckpt(ckpt_path: Optional[Path]) -> None:
    """resume 时把已存在的 rolling ckpt 注册进模块状态，
    这样下一次 save_rolling_checkpoint 时能正确把它清掉、不留孤儿。"""
    global _LAST_ROLLING_CKPT
    _LAST_ROLLING_CKPT = ckpt_path


def _gc_old_permanent(output_dir: Path, keep_last_n: int) -> None:
    """只保留最新 N 个 permanent ckpt，更老的删掉。"""
    import re
    import shutil
    pattern = re.compile(r"^permanent-(\d+)$")
    candidates: list[tuple[int, Path]] = []
    for p in output_dir.iterdir():
        if not p.is_dir():
            continue
        m = pattern.match(p.name)
        if m:
            candidates.append((int(m.group(1)), p))
    candidates.sort(key=lambda x: x[0])
    if len(candidates) <= keep_last_n:
        return
    for _, p in candidates[: -keep_last_n]:
        try:
            shutil.rmtree(p)
            print(f"[ModelV2] GC permanent ckpt {p.name}", flush=True)
        except Exception as e:
            print(f"[ModelV2] WARN: GC failed for {p}: {e}", flush=True)
