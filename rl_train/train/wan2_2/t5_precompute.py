"""t5_precompute.py — Wan2.2 离线预计算 T5 (umt5-xxl) text embedding 缓存。

输出格式（与 gen3r 完全一致）：
    {embed_dir}/{dataset}/{sample_id}.pt   — Tensor [L, 4096]（去 padding）
    {embed_dir}/neg_embed.pt               — 全局共享的 negative prompt embed

negative prompt 默认使用 Wan2.2-Fun-5B-Control-Camera 官方推理脚本中的长中文版本。

用法：
    python t5_precompute.py \\
        --pretrained_model_path <repo>/model/Wan2.2-Fun-5B-Control-Camera \\
        --data_root <your_rl_dataset_root> \\
        --datasets dl3dv \\
        --embed_dir <repo>/data/t5_cache_wan22 \\
        --device cuda:0

    （``--config_path`` 默认为仓库内 ``eval/infer/wan2.2/VideoX-Fun/config/...``；也可用
    环境变量 ``WAN22_CONFIG_PATH`` / ``VIDEOX_FUN_ROOT`` 覆盖。）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from paths import default_wan22_config_path, videox_fun_root  # noqa: E402

# 让 videox_fun 可以 import
_VIDEOX_ROOT = videox_fun_root()
if str(_VIDEOX_ROOT) not in sys.path:
    sys.path.insert(0, str(_VIDEOX_ROOT))

from transformers import AutoTokenizer  # noqa: E402

from videox_fun.models import WanT5EncoderModel  # noqa: E402

from config import WAN22_DEFAULT_NEG_PROMPT  # noqa: E402


def encode_one(text: str, tokenizer, text_encoder, device: str, max_length: int = 512) -> torch.Tensor:
    """Encode 一条文本，返回 [actual_len, 4096] bf16 tensor (cpu)。"""
    inputs = tokenizer(
        text,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    seq_len = attention_mask.gt(0).sum(dim=1).long()

    with torch.no_grad():
        embeds = text_encoder(input_ids, attention_mask=attention_mask)[0]

    return embeds[0, : seq_len[0]].cpu()


def main():
    parser = argparse.ArgumentParser(description="Wan2.2 T5 embedding 离线预计算")
    parser.add_argument("--pretrained_model_path", type=str, required=True,
                        help="Wan2.2-Fun-5B-Control-Camera 根目录")
    parser.add_argument("--config_path", type=str,
                        default=default_wan22_config_path())
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--datasets", type=str, default="dl3dv")
    parser.add_argument("--embed_dir", type=str, required=True,
                        help="embedding 缓存输出目录")
    parser.add_argument("--negative_prompt", type=str, default=WAN22_DEFAULT_NEG_PROMPT,
                        help="CFG 负向 prompt（默认 Wan2.2 5B 官方长中文 neg prompt）")
    parser.add_argument("--tokenizer_max_length", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = args.device
    embed_dir = Path(args.embed_dir)
    embed_dir.mkdir(parents=True, exist_ok=True)

    # ── 加载 tokenizer + T5 ─────────────────────────────────────────────────
    config = OmegaConf.load(args.config_path)
    tok_sub = config["text_encoder_kwargs"].get("tokenizer_subpath", "google/umt5-xxl")
    te_sub = config["text_encoder_kwargs"].get("text_encoder_subpath",
                                               "models_t5_umt5-xxl-enc-bf16.pth")

    tok_path = os.path.join(args.pretrained_model_path, tok_sub)
    te_path = os.path.join(args.pretrained_model_path, te_sub)

    print(f"[T5Precompute] Loading tokenizer from {tok_path}")
    tokenizer = AutoTokenizer.from_pretrained(tok_path)

    print(f"[T5Precompute] Loading T5 encoder from {te_path}")
    text_encoder = WanT5EncoderModel.from_pretrained(
        te_path,
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    ).to(device=device).eval()
    text_encoder.requires_grad_(False)
    print("[T5Precompute] T5 loaded.")

    # ── neg prompt（全局共享） ──────────────────────────────────────────────
    neg_cache_path = embed_dir / "neg_embed.pt"
    if args.overwrite or not neg_cache_path.exists():
        neg_embed = encode_one(args.negative_prompt, tokenizer, text_encoder,
                               device, args.tokenizer_max_length)
        torch.save(neg_embed, str(neg_cache_path))
        print(f"[T5Precompute] Neg embed saved: {neg_cache_path}  shape={tuple(neg_embed.shape)}")
    else:
        print(f"[T5Precompute] Neg embed already exists: {neg_cache_path}")

    # ── 遍历样本 ────────────────────────────────────────────────────────────
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    total_ok = total_skip = total_err = 0

    for ds_name in datasets:
        ds_dir = Path(args.data_root) / ds_name
        if not ds_dir.exists():
            print(f"[T5Precompute] WARNING: {ds_dir} not found, skipping")
            continue

        out_ds_dir = embed_dir / ds_name
        out_ds_dir.mkdir(parents=True, exist_ok=True)

        sample_dirs = sorted([d for d in ds_dir.iterdir() if d.is_dir()])
        print(f"\n[T5Precompute] Dataset {ds_name}: {len(sample_dirs)} samples")

        for sample_dir in sample_dirs:
            meta_path = sample_dir / "metadata.json"
            if not meta_path.exists():
                continue

            out_path = out_ds_dir / f"{sample_dir.name}.pt"
            if not args.overwrite and out_path.exists():
                total_skip += 1
                continue

            try:
                with open(str(meta_path), "r") as f:
                    meta = json.load(f)
                caption = (meta.get("caption", meta.get("prompt", "")) or "").strip()
                if not caption:
                    caption = "camera moving through a scene"

                prompt_embed = encode_one(caption, tokenizer, text_encoder,
                                          device, args.tokenizer_max_length)
                torch.save(prompt_embed, str(out_path))
                total_ok += 1

                if total_ok % 100 == 0:
                    print(f"[T5Precompute] {total_ok} done, shape={tuple(prompt_embed.shape)}, "
                          f"caption={caption[:60]!r}")
            except Exception as e:
                print(f"[T5Precompute] ERROR {sample_dir.name}: {e}")
                total_err += 1

    print(f"\n[T5Precompute] Done. ok={total_ok}, skip={total_skip}, err={total_err}")
    print(f"[T5Precompute] Embeddings saved to: {embed_dir}")


if __name__ == "__main__":
    main()
