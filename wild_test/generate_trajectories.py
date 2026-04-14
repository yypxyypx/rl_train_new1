"""Wild Test 轨迹生成脚本 v2

基于 look-at 相机模型的高级轨迹系统：
  - 所有轨迹都包含真实的位移（禁止纯旋转）
  - 支持 orbit、dolly、crane、tracking 及其复合组合
  - 非线性速度曲线（ease_inout 等）
  - random 模式：随机采样合理轨迹参数，用于大规模 RL 训练

运行：
    cd /home/users/puxin.yan-labs/wild_test
    conda activate gen3r
    python generate_trajectories.py [--mode preset|random] [--force]
"""

import argparse
import json
import math
import os
import random as pyrandom
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms.functional import resize

# ─── 路径配置 ─────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_GEN3R_ROOT = _HERE.parent / "RL" / "gen3r" / "Gen3R"
for _p in [str(_GEN3R_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gen3r.pipeline import Gen3RPipeline
from gen3r.utils.data_utils import center_crop, preprocess_poses
from gen3r.models.vggt.utils.pose_enc import pose_encoding_to_extri_intri

# ─── 常量 ─────────────────────────────────────────────────────────────────────
CHECKPOINTS = str(_GEN3R_ROOT / "checkpoints")
DATA_DIR = _HERE / "wild_test_data"
OUTPUT_BASE = _HERE / "results"
TARGET_H = TARGET_W = 560
NUM_FRAMES = 49


# ═══════════════════════════════════════════════════════════════════════════════
# 速度曲线
# ═══════════════════════════════════════════════════════════════════════════════

def make_progress(num_frames, speed="random_smooth", rng=None):
    """返回 [F] 的进度值 0→1，严格单调递增。

    speed 参数：
      random_smooth  有界随机游走（默认）：速度每帧小幅随机偏移，
                     限制在均值 65%~145% 范围内，无系统性减速，无停顿。
      random_uniform 帧间独立随机（更粗粒度），轻微平滑后使用。
      linear         匀速（仅调试用）。

    设计保证：
      - 任意帧步长 ≥ 平均步长 × 0.65，即最慢帧也有均值 65% 的运动量。
      - 无 cos/sin/t² 等解析函数，每次结果不同（种子不同即不同）。
    """
    if rng is None:
        rng = pyrandom

    F = int(num_frames)

    if speed == "random_smooth":
        # 有界随机游走：v[i+1] = clip(v[i] + uniform(-step, step), v_min, v_max)
        v_base = 1.0
        v_min = 0.65        # 不低于均值的 65%
        v_max = 1.45        # 不超过均值的 145%
        max_delta = 0.18    # 每帧最大变化幅度，控制平滑度
        v = np.empty(F)
        v[0] = rng.uniform(0.8, 1.2)
        for i in range(1, F):
            delta = rng.uniform(-max_delta, max_delta)
            v[i] = np.clip(v[i - 1] + delta, v_min, v_max)

    elif speed == "random_uniform":
        # 独立采样 + 平滑
        v = np.array([rng.uniform(0.5, 1.5) for _ in range(F)])
        kernel = np.ones(7) / 7
        v = np.convolve(v, kernel, mode='same')
        v = np.clip(v, 0.3, None)

    else:
        v = np.ones(F)

    cumsum = np.cumsum(v)
    progress = (cumsum - cumsum[0]) / (cumsum[-1] - cumsum[0])
    return torch.tensor(progress, dtype=torch.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# 旋转辅助
# ═══════════════════════════════════════════════════════════════════════════════

def _rot_y(theta):
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=torch.float32)


def _rot_x(theta):
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=torch.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Look-at 相机模型
# ═══════════════════════════════════════════════════════════════════════════════

def look_at_c2w(eye, target):
    """计算 c2w 矩阵：相机在 eye，朝向 target。

    坐标系约定（OpenCV / Gen3R）：+x 右，+y 下，+z 前。
    """
    forward = target - eye
    norm = forward.norm()
    if norm < 1e-8:
        return torch.eye(4, dtype=torch.float32)
    forward = forward / norm

    world_up = torch.tensor([0.0, -1.0, 0.0])
    right = torch.cross(forward, world_up)
    if right.norm() < 1e-6:
        world_up = torch.tensor([0.0, 0.0, -1.0])
        right = torch.cross(forward, world_up)
    right = right / right.norm()

    down = torch.cross(forward, right)
    down = down / down.norm()

    c2w = torch.eye(4, dtype=torch.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = eye
    return c2w


# ═══════════════════════════════════════════════════════════════════════════════
# 核心轨迹生成：统一的 orbit + translation 框架
# ═══════════════════════════════════════════════════════════════════════════════

def compute_trajectory(num_frames, scene_scale,
                       orbit_y_deg=0.0, orbit_x_deg=0.0,
                       dolly_z=0.0, dolly_y=0.0, dolly_x=0.0,
                       speed="random_smooth", look_dist_factor=3.0, rng=None):
    """生成 c2w 轨迹 [F, 4, 4]。

    speed="random_smooth"（默认）：每次生成随机速度曲线，模拟真实摄影的不确定性。
    rng 传入同一个 random.Random 对象，保证整个 pipeline 的随机可复现。
    """
    F = int(num_frames)
    d = float(scene_scale)
    orbit_pivot = torch.tensor([0.0, 0.0, d])
    look_target = torch.tensor([0.0, 0.0, d * look_dist_factor])
    progress = make_progress(F, speed, rng=rng)

    poses = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(F, 1, 1)

    for i in range(F):
        t = progress[i].item()

        theta_y = math.radians(orbit_y_deg * t)
        theta_x = math.radians(orbit_x_deg * t)

        approach = dolly_z * d * t
        r = max(d - approach, 0.1 * d)

        Ry = _rot_y(theta_y)
        Rx = _rot_x(theta_x)
        R_orbit = Ry @ Rx
        offset = R_orbit @ torch.tensor([0.0, 0.0, -r])
        position = orbit_pivot + offset

        position[1] -= dolly_y * d * t
        position[0] += dolly_x * d * t

        poses[i] = look_at_c2w(position, look_target)

    return poses


# ═══════════════════════════════════════════════════════════════════════════════
# 命名轨迹配置表（20+ 种类型）
# ═══════════════════════════════════════════════════════════════════════════════

TRAJECTORY_CONFIGS = {
    # ── 环绕（orbit）──
    "orbit_right":          {"orbit_y_deg":  60},
    "orbit_left":           {"orbit_y_deg": -60},
    "orbit_up":             {"orbit_x_deg": -40},
    "orbit_down":           {"orbit_x_deg":  35},

    # ── 推拉（dolly）──
    "push_in":              {"dolly_z":  2.2, "orbit_y_deg":  6},
    "pull_out":             {"dolly_z": -1.8, "orbit_y_deg": -6},

    # ── 升降（crane）──
    "crane_up":             {"dolly_y":  1.0, "orbit_x_deg":  12},
    "crane_down":           {"dolly_y": -1.0, "orbit_x_deg": -12},

    # ── 横移（tracking）──
    "tracking_right":       {"dolly_x":  1.3},
    "tracking_left":        {"dolly_x": -1.3},

    # ── 螺旋推进/后退（orbit + dolly_z）──
    "spiral_in_right":      {"orbit_y_deg":  55, "dolly_z":  1.4},
    "spiral_in_left":       {"orbit_y_deg": -55, "dolly_z":  1.4},
    "spiral_out_right":     {"orbit_y_deg":  50, "dolly_z": -1.1},
    "spiral_out_left":      {"orbit_y_deg": -50, "dolly_z": -1.1},

    # ── 升轨道（orbit + 升高）──
    "rise_orbit_right":     {"orbit_y_deg":  55, "dolly_y":  0.65},
    "rise_orbit_left":      {"orbit_y_deg": -55, "dolly_y":  0.65},

    # ── 俯冲轨道（orbit + 降低）──
    "dive_orbit_right":     {"orbit_y_deg":  50, "dolly_y": -0.60},
    "dive_orbit_left":      {"orbit_y_deg": -50, "dolly_y": -0.60},

    # ── 推进 + 升降 ──
    "push_crane_up":        {"dolly_z":  1.5, "dolly_y":  0.65},
    "push_crane_down":      {"dolly_z":  1.5, "dolly_y": -0.60},

    # ── 三轴复合（orbit + dolly_z + dolly_y）──
    "spiral_rise_right":    {"orbit_y_deg":  50, "dolly_z": 1.1, "dolly_y":  0.50},
    "spiral_rise_left":     {"orbit_y_deg": -50, "dolly_z": 1.1, "dolly_y":  0.50},
    "spiral_dive_right":    {"orbit_y_deg":  50, "dolly_z": 1.1, "dolly_y": -0.45},
    "spiral_dive_left":     {"orbit_y_deg": -50, "dolly_z": 1.1, "dolly_y": -0.45},

    # ── 横移 + 推进（斜向运动）──
    "track_push_right":     {"dolly_x":  1.0, "dolly_z": 1.3},
    "track_push_left":      {"dolly_x": -1.0, "dolly_z": 1.3},
}
# 所有轨迹均使用 random_smooth 速度曲线（在 compute_trajectory 中默认）


# ═══════════════════════════════════════════════════════════════════════════════
# 随机轨迹采样（用于大规模 RL 训练数据生成）
# ═══════════════════════════════════════════════════════════════════════════════

def sample_random_trajectory_config(rng=None):
    """随机采样一组合理的轨迹参数。

    设计原则：
      1. 至少一个自由度有显著位移（禁止纯旋转）
      2. 所有参数从 0 单调到目标值（无回弹）
      3. 各参数方向协调，不出现矛盾运动
    """
    if rng is None:
        rng = pyrandom

    speed = rng.choice(["random_smooth", "random_smooth", "random_smooth",
                         "random_uniform", "linear"])

    # 决定主运动类型及其参数范围
    primary = rng.choice(["orbit", "dolly", "crane", "tracking"])

    orbit_y = 0.0
    orbit_x = 0.0
    dz = 0.0
    dy = 0.0
    dx = 0.0

    if primary == "orbit":
        orbit_y = rng.uniform(30, 60) * rng.choice([-1, 1])
        if rng.random() < 0.5:
            dz = rng.uniform(0.5, 1.2) * rng.choice([-1, 1])
        if rng.random() < 0.4:
            dy = rng.uniform(0.25, 0.60) * rng.choice([-1, 1])
        if rng.random() < 0.3:
            orbit_x = rng.uniform(-20, 20)

    elif primary == "dolly":
        dz = rng.uniform(1.0, 2.0) * rng.choice([-1, 1])
        if rng.random() < 0.6:
            orbit_y = rng.uniform(5, 20) * rng.choice([-1, 1])
        if rng.random() < 0.4:
            dy = rng.uniform(0.2, 0.55) * rng.choice([-1, 1])

    elif primary == "crane":
        dy = rng.uniform(0.5, 0.9) * rng.choice([-1, 1])
        orbit_x = rng.uniform(5, 15) * (1 if dy > 0 else -1)
        if rng.random() < 0.5:
            dz = rng.uniform(0.5, 1.0)
        if rng.random() < 0.4:
            orbit_y = rng.uniform(15, 35) * rng.choice([-1, 1])

    elif primary == "tracking":
        dx = rng.uniform(0.6, 1.2) * rng.choice([-1, 1])
        if rng.random() < 0.5:
            dz = rng.uniform(0.5, 1.0)
        if rng.random() < 0.3:
            dy = rng.uniform(0.15, 0.4) * rng.choice([-1, 1])

    return {
        "orbit_y_deg": orbit_y,
        "orbit_x_deg": orbit_x,
        "dolly_z": dz,
        "dolly_y": dy,
        "dolly_x": dx,
        "speed": speed,
    }


def config_to_str(cfg):
    """将配置转为可读字符串。"""
    parts = []
    if abs(cfg.get("orbit_y_deg", 0)) > 0.5:
        parts.append(f"orbit_y={cfg['orbit_y_deg']:.1f}°")
    if abs(cfg.get("orbit_x_deg", 0)) > 0.5:
        parts.append(f"orbit_x={cfg['orbit_x_deg']:.1f}°")
    if abs(cfg.get("dolly_z", 0)) > 0.01:
        parts.append(f"dolly_z={cfg['dolly_z']:.2f}")
    if abs(cfg.get("dolly_y", 0)) > 0.01:
        parts.append(f"dolly_y={cfg['dolly_y']:.2f}")
    if abs(cfg.get("dolly_x", 0)) > 0.01:
        parts.append(f"dolly_x={cfg['dolly_x']:.2f}")
    parts.append(f"speed={cfg.get('speed', 'ease_inout')}")
    return ", ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# 12 张测试图：每张分配多样化的复合轨迹
# ═══════════════════════════════════════════════════════════════════════════════

IMAGE_CONFIGS = {
    "shinei1": {
        "file": "shinei1.png",
        "traj_type": "spiral_in_right",
        "caption": "A video exploring a modern duplex living room with teal sofas, a glass staircase, and an open kitchen from multiple camera angles.",
    },
    "shinei2": {
        "file": "shinei2.png",
        "traj_type": "push_crane_up",
        "caption": "A video moving through an elegant study room with a wooden desk, bookshelves, and golden pendant lights.",
    },
    "shinei3": {
        "file": "shinei3.jpg",
        "traj_type": "rise_orbit_left",
        "caption": "A video touring a bohemian-style living room with a gray sofa, colorful rug, and decorative plants from multiple angles.",
    },
    "shinei4": {
        "file": "shinei4.jpg",
        "traj_type": "spiral_out_left",
        "caption": "A video panning across a mid-century modern living room with an Eames lounge chair and warm sunlight streaming through the window.",
    },
    "shinei5": {
        "file": "shinei5.png",
        "traj_type": "push_in",
        "caption": "A video walking down a grand corridor with a red carpet, wall-mounted candles, and classical paintings on the walls.",
    },
    "shinei6": {
        "file": "shinei6.png",
        "traj_type": "dive_orbit_right",
        "caption": "A video exploring a luxurious neoclassical bedroom with a chandelier, upholstered bed, and silk curtains.",
    },
    "shiwai1": {
        "file": "shiwai1.jpeg",
        "traj_type": "spiral_rise_right",
        "caption": "A video following a winding garden path with white deer sculptures, green lawns, and city buildings in the background.",
    },
    "shiwai2": {
        "file": "shiwai2.webp",
        "traj_type": "orbit_left",
        "caption": "A video capturing an outdoor scene from multiple camera angles.",
    },
    "shiwai3": {
        "file": "shiwai3.png",
        "traj_type": "crane_down",
        "caption": "A video flying over a vast green valley with braided rivers, rolling hills, and snow-capped mountains in the distance.",
    },
    "shiwai4": {
        "file": "shiwai4.webp",
        "traj_type": "track_push_right",
        "caption": "A video capturing an outdoor scene from multiple camera angles.",
    },
    "shiwai5": {
        "file": "shiwai5.png",
        "traj_type": "push_crane_down",
        "caption": "A video moving along a running track through a sunny park with lush green trees and open grass fields.",
    },
    "shiwai6": {
        "file": "shiwai6.png",
        "traj_type": "spiral_dive_left",
        "caption": "A video following a curved path in an autumn park with golden sunlight streaming through the trees.",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# 图片预处理
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_preprocess(image_path, target_h=TARGET_H, target_w=TARGET_W):
    img = Image.open(str(image_path)).convert("RGB")
    img_tensor = TF.to_tensor(img)
    fh, fw = img_tensor.shape[1], img_tensor.shape[2]
    scale = target_h / min(fh, fw)
    img_resized = resize(img_tensor, [round(fh * scale), round(fw * scale)])
    img_cropped = center_crop(img_resized.unsqueeze(0), (target_h, target_w)).squeeze(0)
    return img_cropped


# ═══════════════════════════════════════════════════════════════════════════════
# VGGT 深度 + 内参估计
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_depth_and_intrinsics(pipeline, control_images, device):
    with torch.no_grad():
        agg_tokens, ps_idx = pipeline.vggt.aggregator(control_images)
        agg_tokens_depth = [
            agg_tokens[i] for i in pipeline.vggt.depth_head.intermediate_layer_idx
        ]
        pose_enc = pipeline.vggt.camera_head(agg_tokens_depth)[-1]
        intrinsic = pose_encoding_to_extri_intri(
            pose_enc, control_images.shape[-2:]
        )[1]
        depth_maps, _ = pipeline.vggt.depth_head(
            agg_tokens_depth, control_images, ps_idx
        )
    scene_scale = 0.8 * torch.median(depth_maps).item()
    return scene_scale, intrinsic


# ═══════════════════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["preset", "random"], default="preset",
                        help="preset: 每张图用预定义轨迹; random: 随机采样")
    parser.add_argument("--force", action="store_true",
                        help="覆盖已有的 cameras.json")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（random 模式使用）")
    args = parser.parse_args()

    rng = pyrandom.Random(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print(f"Loading Gen3R pipeline from: {CHECKPOINTS}")
    pipeline = Gen3RPipeline.from_pretrained(CHECKPOINTS)
    pipeline.vggt.to(device=device, dtype=dtype).eval()
    print("VGGT loaded.")

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    print(f"\nMode: {args.mode}")
    print(f"Available named trajectory types: {len(TRAJECTORY_CONFIGS)}")

    for name, cfg in IMAGE_CONFIGS.items():
        out_dir = OUTPUT_BASE / name
        cameras_path = out_dir / "cameras.json"

        if cameras_path.exists() and not args.force:
            print(f"[{name}] cameras.json exists, skipping (use --force to overwrite).")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        img_path = DATA_DIR / cfg["file"]
        caption = cfg["caption"]

        print(f"\n[{name}] Loading image: {img_path.name}")
        img_tensor = load_and_preprocess(img_path)

        from torchvision.utils import save_image
        save_image(img_tensor, str(out_dir / "input.png"))

        ctrl = img_tensor.unsqueeze(0).unsqueeze(0).to(device, dtype)
        print(f"[{name}] Estimating depth and intrinsics with VGGT ...")
        scene_scale, intrinsic = estimate_depth_and_intrinsics(pipeline, ctrl, device)

        # 选择轨迹配置
        if args.mode == "random":
            traj_config = sample_random_trajectory_config(rng)
            traj_type_name = "random"
        else:
            traj_type_name = cfg["traj_type"]
            traj_config = dict(TRAJECTORY_CONFIGS[traj_type_name])

        # speed 由 compute_trajectory 默认为 random_smooth，也可在 traj_config 中覆盖
        speed = traj_config.pop("speed", "random_smooth")
        print(f"[{name}] scene_scale={scene_scale:.4f}, traj={traj_type_name}")
        print(f"         params: {config_to_str({**traj_config, 'speed': speed})}")

        # 生成轨迹（传入 rng 保证 random_smooth 速度曲线的可复现性）
        c2ws = compute_trajectory(
            num_frames=NUM_FRAMES,
            scene_scale=scene_scale,
            speed=speed,
            rng=rng,
            **traj_config,
        )
        c2ws = preprocess_poses(c2ws)
        extrinsics = torch.linalg.inv(c2ws)

        # 内参
        K_single = intrinsic[0, 0].float().cpu()
        Ks = K_single.unsqueeze(0).repeat(NUM_FRAMES, 1, 1)

        # 保存 cameras.json
        cameras_data = {
            "extrinsics": extrinsics.float().numpy().tolist(),
            "intrinsics": Ks.float().numpy().tolist(),
        }
        with open(cameras_path, "w") as f:
            json.dump(cameras_data, f, indent=4)

        # 保存 prompt.txt
        with open(out_dir / "prompt.txt", "w") as f:
            f.write(caption)

        # 保存 meta.json
        meta = {
            "image_name": name,
            "source_file": cfg["file"],
            "trajectory_type": traj_type_name,
            "trajectory_config": {**traj_config, "speed": speed},
            "scene_scale": float(scene_scale),
            "num_frames": NUM_FRAMES,
            "resolution": [TARGET_H, TARGET_W],
        }
        with open(out_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=4)
        print(f"[{name}] Done -> {out_dir}")

    print("\nAll trajectories generated.")
    print("Results summary:")
    for name in IMAGE_CONFIGS:
        out_dir = OUTPUT_BASE / name
        ok = (out_dir / "cameras.json").exists()
        info = ""
        if ok:
            with open(out_dir / "meta.json") as f:
                m = json.load(f)
            info = f"scale={m['scene_scale']:.3f}, type={m['trajectory_type']}"
            tc = m.get("trajectory_config", {})
            if tc:
                info += f", {config_to_str(tc)}"
        print(f"  {name}: {'OK' if ok else 'MISSING'} {info}")


if __name__ == "__main__":
    main()
