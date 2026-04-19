"""t5_precompute.py — 离线预计算 T5 text embedding 缓存。

对训练集所有样本的 caption 运行 T5 编码，输出 .pt 文件。
训练时直接加载 tensor，无需加载 26GB T5 模型。

输出格式（每个 sample_id 一个文件）：
    {embed_dir}/{dataset}/{sample_id}.pt
    内容：{"prompt_embed": Tensor[L, 4096], "neg_embed": Tensor[L_neg, 4096]}

用法：
    python t5_precompute.py \\
        --pretrained_model_path /path/to/gen3r_ckpts \\
        --data_root /path/to/data \\
        --datasets re10k,dl3dv \\
        --embed_dir /path/to/embed_cache \\
        --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_GEN3R_PKG = _HERE / "Gen3R"
if str(_GEN3R_PKG) not in sys.path:
    sys.path.insert(0, str(_GEN3R_PKG))

from gen3r.models import WanT5EncoderModel  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def _resolve_ckpt_subpath(root: str, preferred: str, fallbacks: list[str], *, need_config_json: bool = False) -> str:
    candidates = [preferred] + [f for f in fallbacks if f and f != preferred]
    for sub in candidates:
        if not sub:
            continue
        p = os.path.normpath(os.path.join(root, sub))
        if need_config_json:
            if os.path.isfile(os.path.join(p, "config.json")):
                return p
        else:
            if os.path.isdir(p) or os.path.isfile(p):
                return p
    raise FileNotFoundError(f"Cannot find checkpoint under {root!r}. Tried: {candidates!r}")


def encode_one(text: str, tokenizer, text_encoder, device: str, max_length: int = 512) -> torch.Tensor:
    """返回 [actual_len, 4096] bf16 tensor（去掉 padding）。"""
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
        embeds = text_encoder(input_ids, attention_mask=attention_mask)[0]  # [1, max_len, 4096]

    return embeds[0, : seq_len[0]].cpu()  # [actual_len, 4096]


def main():
    parser = argparse.ArgumentParser(description="离线预计算 T5 embedding")
    parser.add_argument("--pretrained_model_path", type=str, required=True,
                        help="Gen3R checkpoint 根目录（含 tokenizer/ text_encoder/ 子目录）")
    parser.add_argument("--config_path", type=str,
                        default=str(_HERE / "Gen3R" / "gen3r" / "config" / "gen3r.yaml"))
    parser.add_argument("--data_root", type=str, required=True, help="数据根目录")
    parser.add_argument("--datasets", type=str, default="re10k,dl3dv", help="逗号分隔的数据集名称")
    parser.add_argument("--embed_dir", type=str, required=True, help="embedding 缓存输出目录")
    parser.add_argument("--negative_prompt", type=str, default="bad detailed",
                        help="CFG 负向 prompt（全局共享，只算一次）")
    parser.add_argument("--tokenizer_max_length", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有的 .pt 文件")
    args = parser.parse_args()

    device = args.device
    embed_dir = Path(args.embed_dir)
    embed_dir.mkdir(parents=True, exist_ok=True)

    # ── 加载 tokenizer 和 T5 ────────────────────────────────────────────────
    config = OmegaConf.load(args.config_path)
    tok_sub = config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer")
    te_sub = config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder")

    tok_path = _resolve_ckpt_subpath(args.pretrained_model_path, tok_sub, ["tokenizer"])
    te_path = _resolve_ckpt_subpath(args.pretrained_model_path, te_sub, ["text_encoder"], need_config_json=True)

    print(f"[T5Precompute] Loading tokenizer from {tok_path}")
    tokenizer = AutoTokenizer.from_pretrained(tok_path)

    print(f"[T5Precompute] Loading T5 encoder from {te_path}")
    text_encoder = WanT5EncoderModel.from_pretrained(
        te_path,
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=True,
    ).to(device=device, dtype=torch.bfloat16).eval()
    text_encoder.requires_grad_(False)
    print("[T5Precompute] T5 loaded.")

    # ── 预计算负向 prompt（全局共享，存一个文件）──────────────────────────────
    neg_cache_path = embed_dir / "neg_embed.pt"
    if args.overwrite or not neg_cache_path.exists():
        neg_embed = encode_one(args.negative_prompt, tokenizer, text_encoder, device, args.tokenizer_max_length)
        torch.save(neg_embed, str(neg_cache_path))
        print(f"[T5Precompute] Neg embed saved: {neg_cache_path}  shape={tuple(neg_embed.shape)}")
    else:
        print(f"[T5Precompute] Neg embed already exists: {neg_cache_path}")

    # ── 遍历数据集 ───────────────────────────────────────────────────────────
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    total_ok = total_skip = total_err = 0

    for ds_name in datasets:
        ds_dir = Path(args.data_root) / ds_name
        if not ds_dir.exists():
            print(f"[T5Precompute] WARNING: {ds_dir} not found, skipping dataset {ds_name}")
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
                caption = meta.get("caption", meta.get("prompt", "")).strip()

                prompt_embed = encode_one(caption, tokenizer, text_encoder, device, args.tokenizer_max_length)
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
    print(f"[T5Precompute] Neg embed: {neg_cache_path}")


if __name__ == "__main__":
    main()
