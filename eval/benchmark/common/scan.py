#!/usr/bin/env python3
"""
scan.py — 扫描推理输出目录，自动检测 rollout 数量 N。

匹配 infer_gen3r.py 的输出格式：
  <output_root>/<dataset>/<sample_id>/
    gen_0.mp4 ~ gen_{N-1}.mp4
    start.png, gt.mp4, camera.txt, gt_depth.npz, metadata.json
"""

from pathlib import Path


def scan_output_root(output_root: str | Path) -> list:
    """
    扫描 output_root，收集所有推理视频条目。

    返回 list of dict：
    {
        "dataset": str,
        "sample_id": str,
        "sample_dir": Path,
        "gt_video": str,
        "start_png": str | None,
        "camera_txt": str | None,
        "gt_depth_npz": str | None,
        "gen_videos": [
            {"idx": int, "video_path": str, "gen_dir": Path},
            ...
        ]
    }
    """
    output_root = Path(output_root)
    entries = []

    for dataset_dir in sorted(output_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        # skip benchmark internal dirs
        if dataset_dir.name.startswith("_") or dataset_dir.name == "benchmark_results":
            continue
        dataset = dataset_dir.name

        for sample_dir in sorted(dataset_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            sample_id = sample_dir.name
            gt_video = sample_dir / "gt.mp4"
            start_png = sample_dir / "start.png"
            camera_txt = sample_dir / "camera.txt"

            if not gt_video.exists():
                continue

            gen_videos = []
            for gen_mp4 in sorted(sample_dir.glob("gen_*.mp4")):
                idx = int(gen_mp4.stem.split("_")[1])
                gen_dir = sample_dir / gen_mp4.stem
                gen_dir.mkdir(exist_ok=True)
                gen_videos.append({
                    "idx": idx,
                    "video_path": str(gen_mp4),
                    "gen_dir": gen_dir,
                })

            if not gen_videos:
                continue

            entries.append({
                "dataset": dataset,
                "sample_id": sample_id,
                "sample_dir": sample_dir,
                "gt_video": str(gt_video),
                "start_png": str(start_png) if start_png.exists() else None,
                "camera_txt": str(camera_txt) if camera_txt.exists() else None,
                "gt_depth_npz": str(sample_dir / "gt_depth.npz")
                               if (sample_dir / "gt_depth.npz").exists() else None,
                "gen_videos": gen_videos,
            })

    total_gen = sum(len(e["gen_videos"]) for e in entries)
    print(f"[scan] 扫描完成: {len(entries)} 个样本, {total_gen} 条生成视频 "
          f"(rollouts per sample: "
          f"{[len(e['gen_videos']) for e in entries[:3]]}{'...' if len(entries) > 3 else ''})")
    return entries
