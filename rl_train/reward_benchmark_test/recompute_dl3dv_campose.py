#!/usr/bin/env python3
"""
recompute_dl3dv_campose.py
==========================
Re-run reward.camera_pose benchmark ONLY for DL3DV samples, after the
GL → CV camera.txt conversion. Overwrites existing camera_pose.json in place.

Multi-threaded (pure CPU).
"""
from __future__ import annotations
import argparse
import concurrent.futures as cf
import json
import sys
import traceback
from pathlib import Path
import numpy as np

_THIS = Path(__file__).resolve()
_BENCH = _THIS.parents[2] / "eval" / "benchmark"
sys.path.insert(0, str(_BENCH))

from common.scan import scan_output_root  # noqa: E402
from common.utils import parse_camera_txt, save_json  # noqa: E402
from reward.camera_pose import evaluate_camera_pose  # noqa: E402


def _size(video_path: str):
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, 0
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    return H, W


def process_entry(entry) -> tuple[str, int, int]:
    sid = Path(entry["gt_video"]).parent.name
    if not entry.get("camera_txt"):
        return (sid, 0, 0)
    H, W = _size(entry["gt_video"])
    if H == 0 or W == 0:
        return (sid, 0, 0)
    _, gt_c2w = parse_camera_txt(entry["camera_txt"], H, W)
    ok = 0
    err = 0
    for gv in entry["gen_videos"]:
        out = gv["gen_dir"] / "eval" / "camera_pose.json"
        da3_npz = gv["gen_dir"] / "intermediates" / "da3_pred.npz"
        if not da3_npz.exists():
            continue
        try:
            da3 = dict(np.load(str(da3_npz), allow_pickle=True))
            result = evaluate_camera_pose(da3, gt_c2w)
            save_json(str(out), result)
            ok += 1
        except Exception as e:
            err += 1
            sys.stderr.write(f"[ERR] {sid}/{gv['gen_dir'].name}: {e}\n")
            traceback.print_exc()
    return (sid, ok, err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", required=True,
                    help="Must contain only dl3dv (or just limit via scan)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    entries = scan_output_root(args.output_root)
    print(f"Scanned {len(entries)} entries", flush=True)

    total_ok = 0
    total_err = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (sid, ok, err) in enumerate(ex.map(process_entry, entries)):
            total_ok += ok
            total_err += err
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(entries)}] done (running ok={total_ok}, err={total_err})",
                      flush=True)
    print(f"Finished: ok={total_ok}, err={total_err}", flush=True)


if __name__ == "__main__":
    main()
