"""probe_activation.py — 单卡激活显存扫描。

加载 WanTransformer3DModel 全参可训，造合成 latent / plucker / clip / context，
对 Phase 3 backward 跑一次 forward+backward+optimizer.step，记录峰值显存。

通过 monkey-patch torch.utils.checkpoint.checkpoint，按调用顺序让前 N 个 block
跳过 ckpt（直接 inline 调用，不存 checkpoint 边界，激活全保留），其余 block
继续走 ckpt。可同时扫 cfg_infer ∈ {1, 5}、num_frames ∈ {25, 33, 49}。

输出每个 (N, cfg, num_frames) 组合的 max_memory_allocated 与是否 OOM。

Usage:
    python probe_activation.py \
        --pretrained_model_path /path/to/gen3r_ckpts \
        --config_path Gen3R/gen3r/config/gen3r.yaml \
        --use_8bit_adam \
        --num_frames_list 25,33,49 \
        --cfg_list 1,5 \
        --N_list 0,1,2 \
        --resolution 560
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

_HERE = Path(__file__).resolve().parent
_GEN3R_DIR = _HERE.parent
if str(_GEN3R_DIR) not in sys.path:
    sys.path.insert(0, str(_GEN3R_DIR))

from model_loader import (  # noqa: E402
    create_optimizer,
    setup_trainable_params,
)

_GEN3R_PKG = _GEN3R_DIR / "Gen3R"
if str(_GEN3R_PKG) not in sys.path:
    sys.path.insert(0, str(_GEN3R_PKG))

from gen3r.models import WanTransformer3DModel  # noqa: E402

WEIGHT_DTYPE = torch.bfloat16
SPATIAL_DS = 8
TEMPORAL_DS = 4
IN_CHANNELS = 16


def _resolve_transformer_path(root: str) -> str:
    for sub in ("transformer", "."):
        p = os.path.normpath(os.path.join(root, sub))
        if os.path.isfile(os.path.join(p, "config.json")):
            return p
    raise FileNotFoundError(f"Cannot find transformer config.json under {root!r}")


def install_skip_first_n_ckpt(n_skip: int) -> None:
    """Monkey-patch torch.utils.checkpoint.checkpoint：前 n_skip 次调用直接 inline
    执行（关闭 ckpt），其余正常走 ckpt。计数器在每次调用 reset_ckpt_counter() 后归零。
    """
    import torch.utils.checkpoint as _ckpt
    if not hasattr(_ckpt, "_orig_checkpoint"):
        _ckpt._orig_checkpoint = _ckpt.checkpoint
    state = {"count": 0, "n_skip": n_skip}
    _ckpt._probe_state = state

    def patched(function, *args, use_reentrant=None, **kwargs):
        if state["count"] < state["n_skip"]:
            state["count"] += 1
            # inline call, ignore ckpt
            return function(*args)
        # delegate to original
        if use_reentrant is None:
            return _ckpt._orig_checkpoint(function, *args, **kwargs)
        return _ckpt._orig_checkpoint(function, *args, use_reentrant=use_reentrant, **kwargs)

    _ckpt.checkpoint = patched


def reset_ckpt_counter() -> None:
    import torch.utils.checkpoint as _ckpt
    if hasattr(_ckpt, "_probe_state"):
        _ckpt._probe_state["count"] = 0


def synth_inputs(
    num_frames: int,
    resolution: int,
    cfg: float,
    device,
    dtype,
):
    """合成 transformer 一次前向需要的所有张量（CFG=cfg 时复制为 batch=2）。"""
    latent_t = ((num_frames - 1) // TEMPORAL_DS) + 1
    latent_h = resolution // SPATIAL_DS
    latent_w = resolution // SPATIAL_DS
    patch_h = patch_w = 2
    seq_len = math.ceil((latent_w * 2 * latent_h) / (patch_h * patch_w) * latent_t)

    z = torch.randn(
        (1, IN_CHANNELS, latent_t, latent_h, latent_w * 2),
        device=device, dtype=dtype, requires_grad=False,
    )
    timesteps = torch.tensor([500], device=device, dtype=torch.long)
    # T5 prompt embed: list of [L, 4096]
    prompt_embed = torch.randn((64, 4096), device=device, dtype=dtype)
    neg_embed = torch.randn((64, 4096), device=device, dtype=dtype)
    control_latents = torch.randn(
        (1, 20, latent_t, latent_h, latent_w * 2),
        device=device, dtype=dtype,
    )
    # plucker_embeds 在像素分辨率（H, W*2），control_adapter 内部做 pixel_unshuffle/8
    H_pix = resolution
    W_pix2 = resolution * 2
    plucker_embeds = torch.randn(
        (1, 24, latent_t, H_pix, W_pix2),
        device=device, dtype=dtype,
    )
    clip_context = torch.randn((1, 257, 1280), device=device, dtype=dtype)

    if cfg > 1.0:
        z_in = torch.cat([z, z], dim=0)
        t_in = torch.cat([timesteps, timesteps], dim=0)
        ctx_in = [neg_embed, prompt_embed]
        ctrl_in = torch.cat([control_latents, control_latents], dim=0)
        plk_in = torch.cat([plucker_embeds, plucker_embeds], dim=0)
        clip_in = torch.cat([clip_context, clip_context], dim=0)
    else:
        z_in, t_in = z, timesteps
        ctx_in = [prompt_embed]
        ctrl_in = control_latents
        plk_in = plucker_embeds
        clip_in = clip_context

    return dict(
        x=z_in, t=t_in, context=ctx_in, seq_len=seq_len,
        y=ctrl_in, y_camera=plk_in, clip_fea=clip_in,
    )


def load_transformer(args, device) -> WanTransformer3DModel:
    config = OmegaConf.load(args.config_path)
    tr_path = _resolve_transformer_path(args.pretrained_model_path)
    transformer = WanTransformer3DModel.from_pretrained(
        tr_path,
        transformer_additional_kwargs=OmegaConf.to_container(
            config["transformer_additional_kwargs"]
        ),
        torch_dtype=WEIGHT_DTYPE,
    ).to(device=device)
    transformer.enable_gradient_checkpointing()
    setup_trainable_params(transformer, None)  # 全参可训
    transformer.train()
    return transformer


def run_probe_once(
    transformer,
    optimizer,
    inputs: dict,
    n_skip: int,
    cfg: float,
    device,
) -> dict:
    """用 n_skip 控制前 N 个 block 关 ckpt，跑一次 forward+backward+step，返回峰值显存。"""
    install_skip_first_n_ckpt(n_skip)
    reset_ckpt_counter()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    mem_before = torch.cuda.memory_allocated(device) / 1e9

    try:
        with torch.autocast("cuda", WEIGHT_DTYPE):
            pred = transformer(**inputs)
        if cfg > 1.0:
            pred_uncond, pred_cond = pred.chunk(2)
            pred = pred_uncond.float() + cfg * (pred_cond.float() - pred_uncond.float())
        # 合成 target，造一个标量 loss
        target = torch.zeros_like(pred)
        loss = torch.nn.functional.mse_loss(pred.float(), target.float())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        peak = torch.cuda.max_memory_allocated(device) / 1e9
        return {"ok": True, "peak_gb": peak, "before_gb": mem_before,
                "loss": float(loss.detach().item())}
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        return {"ok": False, "error": "OOM", "peak_gb": peak,
                "before_gb": mem_before, "msg": str(e)[:200]}
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            peak = torch.cuda.max_memory_allocated(device) / 1e9
            return {"ok": False, "error": "OOM", "peak_gb": peak,
                    "before_gb": mem_before, "msg": str(e)[:200]}
        raise


def parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def parse_float_list(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Single-GPU activation memory probe")
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--config_path", type=str,
                        default=str(_GEN3R_PKG / "gen3r" / "config" / "gen3r.yaml"))
    parser.add_argument("--use_8bit_adam", action="store_true",
                        help="Match 4090 训练配置（默认开）")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--resolution", type=int, default=560)
    parser.add_argument("--num_frames_list", type=str, default="49",
                        help="逗号分隔，如 '25,33,49'")
    parser.add_argument("--cfg_list", type=str, default="5",
                        help="逗号分隔，如 '1,5'")
    parser.add_argument("--N_list", type=str, default="0,1,2",
                        help="逗号分隔的「跳过 ckpt 的前 N 个 block」")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_json", type=str,
                        default=str(_HERE / "results" / "probe_activation.json"))
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    print(f"[Probe] device={device}  use_8bit_adam={args.use_8bit_adam}")
    print(f"[Probe] N_list={args.N_list}  cfg_list={args.cfg_list}  "
          f"num_frames_list={args.num_frames_list}")

    transformer = load_transformer(args, device)
    optimizer = create_optimizer(
        transformer, args.learning_rate, args.weight_decay,
        use_8bit=args.use_8bit_adam,
    )
    n_params = sum(p.numel() for p in transformer.parameters())
    print(f"[Probe] Model loaded: {n_params / 1e9:.2f}B params")
    print(f"[Probe] Memory after model+optim init: "
          f"{torch.cuda.memory_allocated(device) / 1e9:.2f} GB")

    results = []
    for nf in parse_int_list(args.num_frames_list):
        for cfg in parse_float_list(args.cfg_list):
            for n_skip in parse_int_list(args.N_list):
                tag = f"nf={nf}  cfg={cfg}  N_skip={n_skip}"
                print(f"\n[Probe] === {tag} ===")
                inputs = synth_inputs(nf, args.resolution, cfg, device, WEIGHT_DTYPE)
                out = run_probe_once(transformer, optimizer, inputs, n_skip, cfg, device)
                out.update({"num_frames": nf, "cfg_infer": cfg, "N_skip": n_skip})
                if out["ok"]:
                    print(f"[Probe]   OK  peak={out['peak_gb']:.2f} GB  "
                          f"baseline={out['before_gb']:.2f} GB  loss={out['loss']:.4f}")
                else:
                    print(f"[Probe]   {out['error']}  peak={out['peak_gb']:.2f} GB  "
                          f"msg={out.get('msg','')}")
                results.append(out)
                # 释放 inputs 与图，避免下一轮被影响
                del inputs
                gc.collect()
                torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"\n[Probe] Saved results to {args.output_json}")

    # Summary table
    print("\n[Probe] Summary:")
    print(f"  {'num_frames':>10} {'cfg':>5} {'N_skip':>7} {'peak_GB':>10} {'status':>8}")
    for r in results:
        status = "OK" if r["ok"] else r["error"]
        print(f"  {r['num_frames']:>10} {r['cfg_infer']:>5} {r['N_skip']:>7} "
              f"{r['peak_gb']:>10.2f} {status:>8}")


if __name__ == "__main__":
    main()
