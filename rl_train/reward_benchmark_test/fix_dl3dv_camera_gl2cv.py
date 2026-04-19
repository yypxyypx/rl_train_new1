#!/usr/bin/env python3
"""
fix_dl3dv_camera_gl2cv.py — Convert DL3DV camera.txt from OpenGL w2c to OpenCV w2c.

Derivation:
    stored  = w2c_gl          (legacy "Datasets_chuli1" pipeline)
    c2w_cv  = c2w_gl @ diag(1, -1, -1, 1)
    w2c_cv  = inv(diag) @ inv(c2w_gl) = diag(1, -1, -1, 1) @ w2c_gl
    (diag(1,-1,-1,1) is self-inverse)

Each sample's camera.txt is:
  - backed up to camera.txt.gl (idempotent: skip if .gl already exists)
  - overwritten in-place with the OpenCV w2c version
"""
from __future__ import annotations
import argparse
import concurrent.futures as cf
import shutil
from pathlib import Path
import numpy as np

D = np.diag([1.0, -1.0, -1.0, 1.0])


def rewrite_one(sample_dir: Path) -> tuple[str, str]:
    cam = sample_dir / "camera.txt"
    if not cam.exists():
        return (sample_dir.name, "no-camera.txt")
    backup = sample_dir / "camera.txt.gl"
    if not backup.exists():
        shutil.copy2(cam, backup)

    lines_in = backup.read_text().strip().splitlines()
    out_lines = []
    for ln in lines_in:
        v = ln.split()
        if len(v) != 19:
            return (sample_dir.name, f"bad-format len={len(v)}")
        idx = int(float(v[0]))
        intr5 = v[1:7]
        w2c_flat = np.array(v[7:19], dtype=np.float64).reshape(3, 4)
        w2c = np.eye(4)
        w2c[:3, :] = w2c_flat
        w2c_cv = D @ w2c
        new12 = w2c_cv[:3, :].flatten()
        out_lines.append(
            f"{idx} {' '.join(intr5)} " + " ".join(f"{x:.10f}" for x in new12)
        )
    cam.write_text("\n".join(out_lines) + "\n")
    return (sample_dir.name, "ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="e.g. /horizon-bucket/.../test_output1/dl3dv")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    root = Path(args.root)
    samples = sorted(d for d in root.iterdir() if d.is_dir())
    print(f"Found {len(samples)} samples under {root}")

    ok = 0
    bad = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for name, status in ex.map(rewrite_one, samples):
            if status == "ok":
                ok += 1
            else:
                bad.append((name, status))
    print(f"Rewrote: {ok}/{len(samples)}")
    if bad:
        print(f"Bad/skipped: {len(bad)}")
        for n, s in bad[:20]:
            print(f"  {n}: {s}")


if __name__ == "__main__":
    main()
