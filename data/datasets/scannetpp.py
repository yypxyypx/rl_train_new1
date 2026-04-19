"""ScanNet++ iPhone parser -> RawSample (OpenCV c2w).

Design (v2 -- old-driver-style, undistorted)
--------------------------------------------
- **Frame source**: the raw ``iphone/rgb/`` directory (60 fps, full frames).
  Some scenes are shipped with only ``rgb.mkv`` (no extracted JPGs); for those
  we fall back to extracting only the frames we need with ``ffmpeg select``
  into a per-scene cache directory.
- **Sampling**: per scene we split the available 60 fps frame range into
  ``num_clips_per_scene`` non-overlapping windows.  Inside each window we
  pick a deterministic random start offset and take ``num_frames`` frames
  with ``stride`` (so each clip spans ``(num_frames - 1) * stride`` original
  60 fps frames, e.g. 49 frames @ stride=2 -> 1.6 s real time).
- **Intrinsics + distortion**: from COLMAP ``cameras.txt`` (OPENCV model;
  fx, fy, cx, cy + k1, k2, p1, p2).  Each frame is undistorted in-place via
  ``cv2.undistort`` so downstream code can treat the intrinsics as pinhole.
  K is constant per scene (one COLMAP camera per iPhone capture).
- **Poses**: ARKit ``pose`` field from ``pose_intrinsic_imu.json`` (every
  60 fps frame has one).  ScanNet++ documents this as a 4x4 c2w matrix in a
  right-handed, +Z-camera-forward convention -- i.e. **already OpenCV** --
  so we use it directly with no axis flip.

Reproducibility
---------------
Per-clip start offset uses a deterministic FNV-1a-style 64-bit mix of
``(master_seed, scene_id, clip_k)``, so re-running with the same master seed
yields bit-identical clips.  Every metadata blob records the master seed,
the per-clip seed, the window bounds, the chosen offset, and the *exact*
list of original 60 fps frame indices used.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path
from random import Random
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .base import RawSample


# ═══════════════════════ COLMAP camera parser ═══════════════════════


def _load_colmap_camera(cameras_txt: Path) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Parse the (single-camera) ``cameras.txt`` produced by ScanNet++.

    Expects the OPENCV model::

        CAMERA_ID OPENCV WIDTH HEIGHT  fx fy cx cy k1 k2 p1 p2

    Returns
    -------
    K : (3, 3) float64 pixel intrinsics
    dist : (5,) float64 OpenCV distortion vector ``[k1, k2, p1, p2, 0]``
    width, height : int
    """
    with open(cameras_txt) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            tok = line.split()
            model = tok[1]
            if model != "OPENCV":
                raise ValueError(
                    f"{cameras_txt}: expected OPENCV model, got {model}"
                )
            width = int(tok[2])
            height = int(tok[3])
            fx, fy, cx, cy = (float(x) for x in tok[4:8])
            k1, k2, p1, p2 = (float(x) for x in tok[8:12])
            K = np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            dist = np.array([k1, k2, p1, p2, 0.0], dtype=np.float64)
            return K, dist, width, height
    raise ValueError(f"{cameras_txt}: no camera entry found")


# ═══════════════════════ ARKit pose JSON loader ═══════════════════════


def _load_arkit_poses(pose_json_path: Path) -> Dict[int, np.ndarray]:
    """Load ARKit per-frame ``pose`` (4x4 c2w) keyed by frame index.

    The JSON is keyed ``"frame_NNNNNN"``; we strip the prefix to int.
    Value layout per frame is::

        {"timestamp": ..., "pose": 4x4, "intrinsic": 3x3,
         "imu": ..., "aligned_pose": 4x4}
    """
    with open(pose_json_path) as f:
        data = json.load(f)
    out: Dict[int, np.ndarray] = {}
    for k, v in data.items():
        try:
            fid = int(k.split("_")[-1])
        except (ValueError, IndexError):
            continue
        if "pose" not in v:
            continue
        out[fid] = np.asarray(v["pose"], dtype=np.float64)
    return out


# ═══════════════════════ window selection / RNG ═══════════════════════


def _window_bounds(num_total: int, num_clips: int) -> List[Tuple[int, int]]:
    """Return ``num_clips`` non-overlapping ``[lo, hi)`` half-open windows."""
    base = num_total // num_clips
    out: List[Tuple[int, int]] = []
    for k in range(num_clips):
        lo = k * base
        hi = lo + base if k < num_clips - 1 else num_total
        out.append((lo, hi))
    return out


def _clip_rng_seed(master_seed: int, scene_id: str, clip_k: int) -> int:
    """Deterministic 32-bit seed via 64-bit FNV-1a-style mix."""
    h = 0xCBF29CE484222325
    prime = 0x100000001B3
    mask = (1 << 64) - 1
    for b in f"{master_seed}|{scene_id}|{clip_k}".encode("utf-8"):
        h = ((h ^ b) * prime) & mask
    return int(h & 0xFFFFFFFF)


# ═══════════════════════ JPG access (with mkv fallback) ═══════════════════════


def _ensure_frames_via_ffmpeg(
    mkv_path: Path,
    cache_dir: Path,
    frame_indices: Sequence[int],
) -> Dict[int, Path]:
    """Extract selected frames from ``rgb.mkv`` into ``cache_dir``.

    Mirrors the official ScanNet++ toolkit's ``extract_rgb_in_colmap`` logic:
    ffmpeg's ``select`` filter writes outputs with sequential numbering
    starting at ``frame_000001.jpg``; we sort the desired frame indices and
    rename outputs to ``frame_<idx>.jpg`` so callers can address them by
    original index.

    Returns a dict mapping ``original_frame_id -> path_to_jpg``.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    sorted_ids = sorted(set(frame_indices))
    final_paths: Dict[int, Path] = {}

    # Skip already-cached frames (allows resume).
    missing_ids: List[int] = []
    for fid in sorted_ids:
        target = cache_dir / f"frame_{fid:06d}.jpg"
        if target.exists() and target.stat().st_size > 0:
            final_paths[fid] = target
        else:
            missing_ids.append(fid)

    if not missing_ids:
        return final_paths

    # Build ffmpeg select expression and run into a tmp dir to avoid name
    # clashes with already-cached files.
    expr = "+".join(f"eq(n,{fid})" for fid in missing_ids)
    filter_str = f"select='{expr}'"

    with tempfile.TemporaryDirectory(dir=str(cache_dir.parent)) as tdir_str:
        tdir = Path(tdir_str)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, dir=str(tdir)
        ) as tmpf:
            tmpf.write(filter_str)
            filter_path = tmpf.name

        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(mkv_path),
            "-filter_script:v", filter_path,
            "-vsync", "vfr",
            "-q:v", "1",
            str(tdir / "frame_%06d.jpg"),
        ]
        subprocess.run(cmd, check=True)

        extracted = sorted(tdir.glob("frame_*.jpg"))
        if len(extracted) != len(missing_ids):
            raise RuntimeError(
                f"ffmpeg returned {len(extracted)} frames but {len(missing_ids)} "
                f"were requested from {mkv_path}"
            )
        for src_path, fid in zip(extracted, missing_ids):
            dst_path = cache_dir / f"frame_{fid:06d}.jpg"
            shutil.move(str(src_path), str(dst_path))
            final_paths[fid] = dst_path

    return final_paths


def _resolve_frame_paths(
    rgb_dir: Path,
    mkv_path: Optional[Path],
    cache_dir: Optional[Path],
    frame_indices: Sequence[int],
) -> Dict[int, Path]:
    """Locate JPG paths for ``frame_indices``, extracting from mkv if needed."""
    needed: List[int] = []
    out: Dict[int, Path] = {}
    rgb_dir_exists = rgb_dir.exists()
    for fid in frame_indices:
        p = rgb_dir / f"frame_{fid:06d}.jpg"
        if rgb_dir_exists and p.exists():
            out[fid] = p
        else:
            needed.append(fid)

    if not needed:
        return out

    if mkv_path is None or not mkv_path.exists():
        raise FileNotFoundError(
            f"frames missing from {rgb_dir} and no rgb.mkv to fall back on; "
            f"first missing: frame_{needed[0]:06d}.jpg"
        )
    if cache_dir is None:
        raise ValueError("cache_dir must be provided when rgb.mkv fallback is needed")

    extracted = _ensure_frames_via_ffmpeg(mkv_path, cache_dir, needed)
    out.update(extracted)
    return out


# ═══════════════════════ image loading + undistort ═══════════════════════


def _load_and_undistort(
    rgb_path: Path,
    K: np.ndarray,
    dist: np.ndarray,
) -> np.ndarray:
    """Load a JPEG and remove lens distortion (output size unchanged)."""
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(f"failed to read image: {rgb_path}")
    und = cv2.undistort(bgr, K, dist, newCameraMatrix=K)
    return cv2.cvtColor(und, cv2.COLOR_BGR2RGB)


# ═══════════════════════ public parser ═══════════════════════


def parse(
    data_root: Path,
    max_samples: int = 0,
    verbose: bool = True,
    *,
    scene_ids: Optional[List[str]] = None,
    master_seed: int = 0,
    num_clips_per_scene: int = 2,
    num_frames: int = 49,
    stride: int = 2,
    cache_root: Optional[Path] = None,
) -> Iterator[RawSample]:
    """Yield ``RawSample`` clips from ScanNet++ iPhone captures.

    Parameters
    ----------
    data_root : Path
        ScanNet++ root containing ``data/`` and ``splits/``.
    scene_ids : list[str], optional
        If given, restrict to these scene ids (no shuffling here).
    master_seed : int
        Seed used to derive each clip's start-offset RNG.
    num_clips_per_scene : int
        Number of non-overlapping clips taken from each scene.
    num_frames : int
        Frames per clip.
    stride : int
        Step (in original 60 fps frame indices) between successive frames in
        a clip.  The clip therefore covers ``(num_frames - 1) * stride + 1``
        original frames.
    cache_root : Path, optional
        Directory under which per-scene RGB caches are placed when frames
        must be extracted from ``rgb.mkv``.  Required only for scenes that
        ship without an extracted ``iphone/rgb/`` directory.
    max_samples : int
        Stop after this many clips (0 = unlimited).
    """
    data_root = Path(data_root)
    scenes_dir = data_root / "data"
    if not scenes_dir.exists():
        warnings.warn(f"scannetpp: not found at {scenes_dir}")
        return

    if scene_ids is None:
        scene_ids = sorted(d.name for d in scenes_dir.iterdir() if d.is_dir())

    yielded = 0
    clip_span = (num_frames - 1) * stride + 1   # original frames covered

    for scene_id in scene_ids:
        if max_samples > 0 and yielded >= max_samples:
            return

        iphone_dir = scenes_dir / scene_id / "iphone"
        rgb_dir = iphone_dir / "rgb"
        mkv_path = iphone_dir / "rgb.mkv"
        cam_txt = iphone_dir / "colmap" / "cameras.txt"
        pose_json = iphone_dir / "pose_intrinsic_imu.json"

        if not (cam_txt.exists() and pose_json.exists()):
            if verbose:
                print(f"  [scannetpp] skip {scene_id}: missing colmap or pose JSON")
            continue
        if not (rgb_dir.exists() or mkv_path.exists()):
            if verbose:
                print(f"  [scannetpp] skip {scene_id}: no rgb/ or rgb.mkv")
            continue

        try:
            K, dist, _w, _h = _load_colmap_camera(cam_txt)
        except Exception as e:
            if verbose:
                print(f"  [scannetpp] skip {scene_id}: cameras.txt parse failed ({e})")
            continue

        try:
            poses = _load_arkit_poses(pose_json)
        except Exception as e:
            if verbose:
                print(f"  [scannetpp] skip {scene_id}: pose JSON parse failed ({e})")
            continue

        if not poses:
            if verbose:
                print(f"  [scannetpp] skip {scene_id}: empty pose JSON")
            continue

        # Total available 60 fps frames: max contiguous index covered by ARKit poses.
        max_fid = max(poses)
        total_frames = max_fid + 1

        if total_frames < num_clips_per_scene * clip_span:
            if verbose:
                print(
                    f"  [scannetpp] skip {scene_id}: "
                    f"only {total_frames} fps60 frames < "
                    f"{num_clips_per_scene * clip_span}"
                )
            continue

        windows = _window_bounds(total_frames, num_clips_per_scene)
        scene_cache_dir = (cache_root / scene_id) if cache_root is not None else None

        # Track whether we extracted any frames via ffmpeg so we can flag it.
        used_mkv_fallback = not rgb_dir.exists()

        for clip_k, (lo, hi) in enumerate(windows):
            if max_samples > 0 and yielded >= max_samples:
                return

            window_size = hi - lo
            max_offset = window_size - clip_span
            if max_offset < 0:
                continue

            rng_seed = _clip_rng_seed(master_seed, scene_id, clip_k)
            offset = Random(rng_seed).randint(0, max_offset)
            start_fid = lo + offset
            frame_indices = [start_fid + i * stride for i in range(num_frames)]

            # All requested frames must have ARKit poses.
            missing_pose = [fid for fid in frame_indices if fid not in poses]
            if missing_pose:
                if verbose:
                    print(
                        f"  [scannetpp] skip {scene_id} clip {clip_k}: "
                        f"missing ARKit pose for {len(missing_pose)} frame(s); "
                        f"first={missing_pose[0]}"
                    )
                continue

            try:
                paths = _resolve_frame_paths(
                    rgb_dir, mkv_path, scene_cache_dir, frame_indices
                )
            except Exception as e:
                if verbose:
                    print(f"  [scannetpp] skip {scene_id} clip {clip_k}: {e}")
                continue

            frames: List[np.ndarray] = []
            c2ws: List[np.ndarray] = []
            Ks: List[np.ndarray] = []

            ok = True
            orig_h_first = orig_w_first = None
            for fid in frame_indices:
                try:
                    img = _load_and_undistort(paths[fid], K, dist)
                except Exception as e:
                    if verbose:
                        print(f"  [scannetpp] skip {scene_id} clip {clip_k}: {e}")
                    ok = False
                    break
                if orig_h_first is None:
                    orig_h_first, orig_w_first = img.shape[:2]
                frames.append(img)
                c2ws.append(poses[fid])
                Ks.append(K.copy())

            if not ok or len(frames) != num_frames:
                continue

            extra: Dict = {
                "master_seed": int(master_seed),
                "clip_k": int(clip_k),
                "clip_rng_seed": int(rng_seed),
                "stride": int(stride),
                "real_duration_sec": round((num_frames - 1) * stride / 60.0, 4),
                "fps60_total_frames": int(total_frames),
                "clip_window_in_fps60": [int(lo), int(hi)],
                "clip_start_fid": int(start_fid),
                "frame_indices": [int(f) for f in frame_indices],
                "frame_filenames": [f"frame_{f:06d}.jpg" for f in frame_indices],
                "pose_source": "arkit_pose",
                "intrinsic_source": "colmap_OPENCV_undistorted",
                "colmap_distortion_original": dist.tolist(),
                "rgb_source": "rgb_dir" if not used_mkv_fallback else "rgb_mkv_ffmpeg",
            }

            yield RawSample(
                sample_id=f"{scene_id}_clip{clip_k}",
                frames=frames,
                c2ws=np.stack(c2ws),
                Ks=np.stack(Ks),
                caption="A 3D indoor scene",
                orig_w=orig_w_first,
                orig_h=orig_h_first,
                dataset="scannetpp",
                orig_id=scene_id,
                depths=None,
                extra=extra,
            )
            yielded += 1
