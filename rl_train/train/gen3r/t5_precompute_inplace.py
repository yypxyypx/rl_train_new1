"""t5_precompute_inplace.py — 把 T5 embedding 直接预存到每条样本目录内。

与原 `t5_precompute.py` 的区别：
  - 旧版: {embed_dir}/{dataset}/{sample_id}.pt + {embed_dir}/neg_embed.pt
  - 新版: {sample_dir}/prompt_embed.pt + {sample_dir}/neg_embed.pt（每条样本各存一份）

样本发现策略：递归扫描 {data_root}/{dataset}/**/metadata.json
  - 兼容 scannet++ 平铺 ({split}/{sample_id})
  - 兼容 dl3dv 嵌套 ({split}/{subset}/{sample_id})

性能：
  - 单 H100 + bf16 + batch=8: ~12 sample/s（caption 都是短文本，T5 forward < 0.5s/batch）
  - 3600 样本预计 5-10 分钟完成；不需要分布式

用法：
    python t5_precompute_inplace.py \\
        --pretrained_model_path /path/to/gen3r_ckpts \\
        --data_root /mnt/afs/visitor16/rl_train_new/hf_datasets/rl_data/gen3r \\
        --datasets dl3dv,scannet++ \\
        --batch_size 8 \\
        --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch

_HERE = Path(__file__).resolve().parent
_GEN3R_PKG = _HERE / "Gen3R"
if str(_GEN3R_PKG) not in sys.path:
    sys.path.insert(0, str(_GEN3R_PKG))

from gen3r.models import WanT5EncoderModel  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# 共用工具
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_ckpt_subpath(
    root: str, preferred: str, fallbacks: List[str], *,
    need_config_json: bool = False,
) -> str:
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


@torch.no_grad()
def encode_batch(
    texts: List[str], tokenizer, text_encoder, device: str,
    max_length: int = 512,
) -> List[torch.Tensor]:
    """B 条文本一次 forward，返回 list of [actual_len_i, 4096] cpu fp32 tensors。

    注意：变长 — 每个样本的 actual_len 不同（attention_mask 决定），所以返回 list
    而不是 stacked tensor。
    """
    inputs = tokenizer(
        texts,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    seq_lens = attention_mask.gt(0).sum(dim=1).long()  # [B]

    embeds = text_encoder(input_ids, attention_mask=attention_mask)[0]  # [B, max_len, 4096]
    out: List[torch.Tensor] = []
    for i in range(embeds.shape[0]):
        actual = int(seq_lens[i].item())
        out.append(embeds[i, :actual].detach().to("cpu"))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 样本扫描
# ──────────────────────────────────────────────────────────────────────────────

def discover_samples(
    data_root: Path, datasets: List[str],
) -> List[Tuple[str, Path, dict]]:
    """递归扫 {data_root}/{ds}/**/metadata.json。

    返回 list of (dataset_name, sample_dir, metadata_dict)。
    """
    out: List[Tuple[str, Path, dict]] = []
    for ds in datasets:
        ds_dir = data_root / ds
        if not ds_dir.exists():
            print(f"[T5InplaceScan] WARNING: {ds_dir} not exists, skip", flush=True)
            continue
        n_before = len(out)
        for meta_path in ds_dir.rglob("metadata.json"):
            sample_dir = meta_path.parent
            # 必须有 camera.txt 才算训练样本
            if not (sample_dir / "camera.txt").exists():
                continue
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
            except Exception as e:
                print(f"[T5InplaceScan] bad metadata {meta_path}: {e}", flush=True)
                continue
            out.append((ds, sample_dir, meta))
        print(f"[T5InplaceScan] {ds}: {len(out) - n_before} samples", flush=True)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="In-place T5 embedding 预处理")
    parser.add_argument("--pretrained_model_path", type=str, required=True,
                        help="Gen3R checkpoint 根目录（含 tokenizer/ text_encoder/ 子目录）")
    parser.add_argument("--config_path", type=str,
                        default=str(_HERE / "Gen3R" / "gen3r" / "config" / "gen3r.yaml"))
    parser.add_argument("--data_root", type=str, required=True,
                        help="数据集根目录，例如 .../hf_datasets/rl_data/gen3r")
    parser.add_argument("--datasets", type=str, default="dl3dv,scannet++",
                        help="逗号分隔 dataset 子目录名")
    parser.add_argument("--negative_prompt", type=str, default="bad detailed",
                        help="CFG 负向 prompt（每个样本都写一份）")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="T5 真 batch 大小（默认 8）")
    parser.add_argument("--tokenizer_max_length", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--overwrite", action="store_true",
                        help="覆盖已有的 prompt_embed.pt / neg_embed.pt")
    parser.add_argument("--limit", type=int, default=0,
                        help="只处理前 N 条（debug 用，0=全部）")
    parser.add_argument("--prompt_filename", type=str, default="prompt_embed.pt")
    parser.add_argument("--neg_filename", type=str, default="neg_embed.pt")
    parser.add_argument("--also_write_global", type=str, default="",
                        help="若不为空，把 neg_embed 同步写一份到这个全局路径"
                             "（兼容旧 args.t5_embed_dir 路径）")
    args = parser.parse_args()

    device = args.device
    data_root = Path(args.data_root).resolve()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    # ── 1. 加载 tokenizer + T5 ───────────────────────────────────────────
    config = OmegaConf.load(args.config_path)
    tok_sub = config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer")
    te_sub = config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder")

    tok_path = _resolve_ckpt_subpath(args.pretrained_model_path, tok_sub, ["tokenizer"])
    te_path = _resolve_ckpt_subpath(
        args.pretrained_model_path, te_sub, ["text_encoder"], need_config_json=True
    )

    print(f"[T5Inplace] tokenizer: {tok_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(tok_path)

    print(f"[T5Inplace] T5 encoder: {te_path}", flush=True)
    t_load0 = time.time()
    text_encoder = WanT5EncoderModel.from_pretrained(
        te_path,
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=True,
    ).to(device=device, dtype=torch.bfloat16).eval()
    text_encoder.requires_grad_(False)
    print(f"[T5Inplace] T5 loaded in {time.time()-t_load0:.1f}s on {device}", flush=True)

    # ── 2. 编码 negative prompt（一次，复制写入每个样本）──────────────────
    print(f"[T5Inplace] encoding neg prompt: {args.negative_prompt!r}", flush=True)
    neg_embed_cpu = encode_batch(
        [args.negative_prompt], tokenizer, text_encoder, device,
        max_length=args.tokenizer_max_length,
    )[0]
    print(f"[T5Inplace] neg_embed shape={tuple(neg_embed_cpu.shape)} "
          f"dtype={neg_embed_cpu.dtype}", flush=True)

    if args.also_write_global:
        gp = Path(args.also_write_global)
        gp.parent.mkdir(parents=True, exist_ok=True)
        torch.save(neg_embed_cpu, str(gp))
        print(f"[T5Inplace] neg_embed mirrored to {gp}", flush=True)

    # ── 3. 扫所有样本 ───────────────────────────────────────────────────
    samples = discover_samples(data_root, datasets)
    if args.limit > 0:
        samples = samples[:args.limit]
        print(f"[T5Inplace] limit={args.limit} → 只处理 {len(samples)} 条", flush=True)
    print(f"[T5Inplace] total {len(samples)} samples to process "
          f"(batch_size={args.batch_size})", flush=True)

    # ── 4. 过滤已存在（除非 overwrite）──────────────────────────────────
    pending: List[Tuple[str, Path, dict]] = []
    n_skip = 0
    for ds, sd, meta in samples:
        prompt_path = sd / args.prompt_filename
        neg_path = sd / args.neg_filename
        if (not args.overwrite) and prompt_path.exists() and neg_path.exists():
            n_skip += 1
            continue
        pending.append((ds, sd, meta))
    print(f"[T5Inplace] skip {n_skip} (already exist), pending {len(pending)}",
          flush=True)

    # ── 5. 真 batch encode + 写盘 ───────────────────────────────────────
    n_ok = 0
    n_err = 0
    t_run0 = time.time()
    last_log_t = t_run0
    last_log_n = 0
    bs = max(1, int(args.batch_size))

    for s in range(0, len(pending), bs):
        e = min(s + bs, len(pending))
        chunk = pending[s:e]
        captions = []
        for ds, sd, meta in chunk:
            cap = (meta.get("caption", meta.get("prompt", "")) or "").strip()
            if not cap:
                cap = "camera moving through a scene"
            captions.append(cap)

        try:
            embeds_cpu = encode_batch(
                captions, tokenizer, text_encoder, device,
                max_length=args.tokenizer_max_length,
            )
        except Exception as ex:
            print(f"[T5Inplace] batch [{s}:{e}] T5 FAILED: {ex}", flush=True)
            n_err += len(chunk)
            continue

        # 写盘（每个样本独立两个文件）
        for (ds, sd, _meta), pe in zip(chunk, embeds_cpu):
            try:
                torch.save(pe, str(sd / args.prompt_filename))
                # neg_embed 全部一致，但每个样本各写一份（用户语义）
                torch.save(neg_embed_cpu, str(sd / args.neg_filename))
                n_ok += 1
            except Exception as ex:
                print(f"[T5Inplace] save FAILED {sd}: {ex}", flush=True)
                n_err += 1

        # 定期吐日志（每 ~5s 或每 200 个）
        now = time.time()
        if (n_ok + n_err) - last_log_n >= 200 or (now - last_log_t) >= 5.0:
            done = n_ok + n_err
            rate = (done - last_log_n) / max(now - last_log_t, 1e-3)
            eta = (len(pending) - done) / max(rate, 1e-3)
            sample_lens = [int(pe.shape[0]) for pe in embeds_cpu]
            print(f"[T5Inplace] {done}/{len(pending)}  ok={n_ok} err={n_err}  "
                  f"rate={rate:.1f}/s  eta={eta:.0f}s  "
                  f"last_lens={sample_lens}",
                  flush=True)
            last_log_t = now
            last_log_n = done

    dt = time.time() - t_run0
    print(f"[T5Inplace] DONE in {dt:.1f}s  ok={n_ok}  err={n_err}  skip={n_skip}",
          flush=True)
    print(f"[T5Inplace] avg {n_ok/max(dt,1e-3):.1f} samples/s  "
          f"per sample written: {args.prompt_filename} + {args.neg_filename}",
          flush=True)


if __name__ == "__main__":
    main()
