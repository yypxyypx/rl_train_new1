#!/usr/bin/env python3
"""
worker_vbench.py
================
VBench 批量 worker —— 在 vbench conda 环境中运行。

调用 third_party/vbench/vbench_metrics.py 中的接口：
    run_vbench_eval(sample_ids, img_dir, vid_dir, vbench_cache, device)
    get_vbench_per_video(results)

运行方式（由 run_benchmark.py 通过 conda run 调用）：
    conda run -n vbench python worker_vbench.py --batch_manifest /path/to/batch.json

batch_manifest.json 格式：
{
  "entries": [
    {
      "sample_id":   "re10k__abc123__gen_0",
      "img_path":    "/path/to/start.png",
      "video_path":  "/path/to/gen_0.mp4",
      "output_json": "/path/to/gen_0/vbench.json",
      "skip_done":   true
    },
    ...
  ],
  "vbench_cache":  "/path/to/model/vbench_cache",
  "device":        "cuda:0",
  "tmp_dir":       "/tmp/vbench_worker_inputs"
}
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# ── 路径注入：使用 third_party/vbench/vbench_metrics.py ──────────────────────
_THIRD_PARTY_DIR = Path(__file__).resolve().parent.parent
_VBENCH_DIR = _THIRD_PARTY_DIR / "vbench"
if str(_VBENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_VBENCH_DIR))

# 自动探测 vbench_cache（支持 RL_MODEL_ROOT 环境变量）
_WORKSPACE = _THIRD_PARTY_DIR.parent.parent
_ENV_MODEL = Path(os.environ.get("RL_MODEL_ROOT", str(_WORKSPACE / "RL" / "model")))
_VBENCH_CANDIDATES = [
    _ENV_MODEL / "vbench_cache",
    _WORKSPACE / "RL" / "model" / "vbench_cache",
    Path.home() / "WAN_TEST" / "model" / "vbench_cache",
]
_VBENCH_CACHE_DEFAULT = str(next(
    (p for p in _VBENCH_CANDIDATES if (p / "dreamsim_cache").exists()),
    _ENV_MODEL / "vbench_cache",
))


def main():
    parser = argparse.ArgumentParser(description="VBench 批量 worker")
    parser.add_argument("--batch_manifest", required=True)
    args = parser.parse_args()

    with open(args.batch_manifest, "r") as f:
        manifest = json.load(f)

    entries      = manifest["entries"]
    vbench_cache = manifest.get("vbench_cache", _VBENCH_CACHE_DEFAULT)
    device       = manifest.get("device", "cuda:0")
    tmp_dir      = Path(manifest.get("tmp_dir", "/tmp/vbench_worker_inputs"))

    print(f"[worker_vbench] 共 {len(entries)} 条视频  device={device}")
    print(f"[worker_vbench] vbench_cache={vbench_cache}")

    # ── 过滤已完成 ─────────────────────────────────────────────────────────────
    todo_entries = []
    for e in entries:
        out = Path(e["output_json"])
        is_done = (e.get("skip_done", True) and out.exists() and
                   out.stat().st_size > 100 and
                   all(k in (json.load(open(out)) if out.stat().st_size < 10240 else {})
                       for k in ("i2v_subject", "i2v_background", "imaging_quality")))
        if is_done:
            print(f"  [跳过] {e['sample_id']}")
        else:
            todo_entries.append(e)

    if not todo_entries:
        print("[worker_vbench] 所有条目已完成，退出")
        return

    # ── 拷贝视频 / 图像到统一 tmp 目录
    img_dir = tmp_dir / "images"
    vid_dir = tmp_dir / "videos"
    img_dir.mkdir(parents=True, exist_ok=True)
    vid_dir.mkdir(parents=True, exist_ok=True)

    valid_entries = []
    for e in todo_entries:
        sid     = e["sample_id"]
        img_src = Path(e["img_path"])
        vid_src = Path(e["video_path"])
        if not img_src.exists() or not vid_src.exists():
            print(f"  [跳过] {sid}：文件不存在 ({img_src.name} / {vid_src.name})")
            continue

        dst_img = img_dir / f"{sid}.png"
        dst_vid = vid_dir / f"{sid}.mp4"
        if not dst_img.exists():
            shutil.copy2(str(img_src), str(dst_img))
        if not dst_vid.exists():
            shutil.copy2(str(vid_src), str(dst_vid))
        valid_entries.append(e)

    if not valid_entries:
        print("[worker_vbench] 没有有效条目，退出")
        return

    sample_ids = [e["sample_id"] for e in valid_entries]

    # ── 调用 vbench_metrics ──────────────────────────────────────────────────
    from vbench_metrics import run_vbench_eval, get_vbench_per_video

    results = run_vbench_eval(
        sample_ids=sample_ids,
        img_dir=img_dir,
        vid_dir=vid_dir,
        vbench_cache=vbench_cache,
        device=device,
    )

    per_video = get_vbench_per_video(results)

    # ── 写出各条目 JSON ──────────────────────────────────────────────────────
    n_ok, n_err = 0, 0
    for e in valid_entries:
        sid      = e["sample_id"]
        out_json = Path(e["output_json"])
        out_json.parent.mkdir(parents=True, exist_ok=True)
        scores = per_video.get(sid, {})
        scores["video"]     = e["video_path"]
        scores["sample_id"] = sid
        try:
            with open(str(out_json), "w") as f:
                json.dump(scores, f, ensure_ascii=False, indent=2)
            print(f"  [✓] {sid}: i2v_subject={scores.get('i2v_subject', float('nan')):.4f}  "
                  f"i2v_bg={scores.get('i2v_background', float('nan')):.4f}  "
                  f"iq={scores.get('imaging_quality', float('nan')):.4f}")
            n_ok += 1
        except Exception as ex:
            print(f"  [错误] 写 JSON {out_json}: {ex}")
            n_err += 1

    print(f"\n[worker_vbench] 完成: 成功 {n_ok}  失败 {n_err}")


if __name__ == "__main__":
    main()
