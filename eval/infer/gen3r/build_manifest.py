#!/usr/bin/env python3
"""Scan processed data directories and generate manifest.jsonl for inference.

Usage:
    python build_manifest.py \\
        --data_root /path/to/processed \\
        --datasets re10k dl3dv \\
        --output manifest.jsonl \\
        [--require_gt_video] [--max_per_dataset 50]
"""

import argparse
import json
from pathlib import Path

REQUIRED_FILES = ["start.png", "camera.txt"]


def scan_dataset(dataset_root: Path, dataset_name: str,
                 max_samples: int = 0, require_gt_video: bool = False) -> list:
    if not dataset_root.exists():
        print(f"  [warn] not found: {dataset_root}")
        return []

    entries = []
    for sample_dir in sorted(d for d in dataset_root.iterdir() if d.is_dir()):
        missing = [f for f in REQUIRED_FILES if not (sample_dir / f).exists()]
        if missing:
            continue
        if require_gt_video and not (sample_dir / "gt.mp4").exists():
            continue

        meta_path = sample_dir / "metadata.json"
        prompt = ""
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                prompt = meta.get("caption", meta.get("prompt", "")).strip()
            except Exception:
                pass

        entries.append({
            "dataset": dataset_name,
            "sample_id": sample_dir.name,
            "sample_dir": str(sample_dir.resolve()),
            "prompt": prompt,
            "has_gt_video": (sample_dir / "gt.mp4").exists(),
            "has_gt_depth": (sample_dir / "gt_depth.npz").exists(),
        })

        if max_samples > 0 and len(entries) >= max_samples:
            break

    return entries


def main():
    parser = argparse.ArgumentParser(description="Build inference manifest.jsonl")
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["re10k", "dl3dv"])
    parser.add_argument("--output", type=Path, default=Path("manifest.jsonl"))
    parser.add_argument("--max_per_dataset", type=int, default=0)
    parser.add_argument("--require_gt_video", action="store_true")
    args = parser.parse_args()

    all_entries = []
    for ds in args.datasets:
        ds_root = args.data_root / ds
        print(f"[{ds}] scanning {ds_root}")
        entries = scan_dataset(ds_root, ds, args.max_per_dataset, args.require_gt_video)
        all_entries.extend(entries)
        print(f"  found {len(entries)} samples")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for e in all_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"\nManifest: {args.output}  ({len(all_entries)} total)")


if __name__ == "__main__":
    main()
