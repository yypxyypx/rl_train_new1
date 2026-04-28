#!/usr/bin/env python3
"""扫描 Gen3R 处理后的数据目录，生成 manifest.jsonl。

目录结构支持：
  data_root/
    <dataset>/
      test/
        <sample_id>/          (scannet++ 平铺，如 02a980c994_clip0/)
          start.png
          camera.txt
          ...
        <subset>/<sample_id>/ (dl3dv 嵌套，如 1K/00b1ad87c296635c/)
          start.png
          camera.txt
          ...

sample_id 取相对 <dataset>/test/ 的路径，并把 '/' 替换成 '_'
（如 1K_00b1ad87c296635c）。

用法：
    python build_manifest.py \\
        --data_root /mnt/afs/visitor16/rl_train_new/hf_datasets/rl_data/gen3r \\
        --datasets dl3dv scannet++ \\
        --output manifest.jsonl \\
        [--require_gt_video] [--max_per_dataset 50]
"""

import argparse
import json
from pathlib import Path

REQUIRED_FILES = ["start.png", "camera.txt"]


def scan_dataset(ds_test_root: Path, dataset_name: str,
                 max_samples: int = 0,
                 require_gt_video: bool = False) -> list:
    """递归扫描 ds_test_root/**，找出所有含 start.png + camera.txt 的叶目录。"""
    if not ds_test_root.exists():
        print(f"  [warn] 目录不存在: {ds_test_root}")
        return []

    entries = []
    # rglob('start.png') 找到所有叶目录
    for start_png in sorted(ds_test_root.rglob("start.png")):
        sample_dir = start_png.parent
        if not all((sample_dir / f).exists() for f in REQUIRED_FILES):
            continue
        if require_gt_video and not (sample_dir / "gt.mp4").exists():
            continue

        # sample_id = 相对 ds_test_root 的路径，'/' → '_'
        rel = sample_dir.relative_to(ds_test_root)
        sample_id = str(rel).replace("/", "_").replace("\\", "_")

        meta_path = sample_dir / "metadata.json"
        prompt = ""
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                prompt = meta.get("caption", meta.get("prompt", "")).strip()
            except Exception:
                pass

        entries.append({
            "dataset":     dataset_name,
            "sample_id":   sample_id,
            "sample_dir":  str(sample_dir.resolve()),
            "prompt":      prompt,
            "has_gt_video": (sample_dir / "gt.mp4").exists(),
            "has_gt_depth": (sample_dir / "gt_depth.npz").exists(),
        })

        if max_samples > 0 and len(entries) >= max_samples:
            break

    return entries


def main():
    parser = argparse.ArgumentParser(description="Gen3R 推理 manifest 生成")
    parser.add_argument("--data_root",       type=Path, required=True,
                        help="数据根目录（包含 dl3dv/、scannet++/ 等子目录）")
    parser.add_argument("--datasets",        nargs="+", default=["dl3dv", "scannet++"],
                        help="数据集名称列表")
    parser.add_argument("--output",          type=Path, default=Path("manifest.jsonl"))
    parser.add_argument("--max_per_dataset", type=int,  default=0,
                        help="每个数据集最多取多少条（0=不限）")
    parser.add_argument("--require_gt_video", action="store_true",
                        help="只保留有 gt.mp4 的样本")
    args = parser.parse_args()

    all_entries = []
    for ds in args.datasets:
        # 扫描 data_root/<ds>/test/
        ds_test_root = args.data_root / ds / "test"
        print(f"[{ds}] 扫描 {ds_test_root}")
        entries = scan_dataset(ds_test_root, ds,
                               args.max_per_dataset, args.require_gt_video)
        all_entries.extend(entries)
        print(f"  找到 {len(entries)} 条样本")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for e in all_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"\nManifest: {args.output}  (共 {len(all_entries)} 条)")


if __name__ == "__main__":
    main()
