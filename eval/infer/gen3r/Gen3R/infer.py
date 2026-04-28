import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import torch
import math
from datetime import datetime
import argparse
from accelerate.utils import set_seed
import numpy as np
import imageio
import json
from einops import rearrange
from torchvision.transforms.functional import resize

from gen3r.models.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from gen3r.utils.data_utils import center_crop, compute_rays, preprocess_poses
from gen3r.utils.common_utils import colorize_depth_map, downsample_and_save_pointcloud, save_videos_grid
from gen3r.pipeline import Gen3RPipeline

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['NCCL_DEBUG'] = 'ERROR'


BUILTIN_CAMERA_TRAJECTORIES = [
    "zoom_in",
    "zoom_out",
    "arc_left",
    "arc_right",
    "translate_up",
    "translate_down",
    "free"
]


def get_poses(cam_type='', num_frames=49, scene_scale=1.0):

    F = int(num_frames)
    if F <= 0:
        raise ValueError("num_frames must be > 0")

    poses = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(F, 1, 1)

    def rot_y(theta):
        c = math.cos(theta)
        s = math.sin(theta)
        return torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=torch.float32)

    def rot_x(theta):
        c = math.cos(theta)
        s = math.sin(theta)
        return torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=torch.float32)

    if cam_type == 'zoom_in' or cam_type == 'zoom_out':
        end = 1.0 * float(scene_scale)
        if cam_type == 'zoom_out':
            end = -end
        zs = torch.linspace(0.0, end, steps=F, dtype=torch.float32)
        for i in range(F):
            poses[i, 2, 3] = zs[i]

    elif cam_type == 'arc_right' or cam_type == 'arc_left':
        end_angle = math.radians(15)
        if cam_type == 'arc_left':
            end_angle = -end_angle
        angles = torch.linspace(0.0, end_angle, steps=F, dtype=torch.float32)
        for i in range(F):
            R = rot_y(float(angles[i]))
            poses[i, :3, :3] = R

    elif cam_type == 'translate_up' or cam_type == 'translate_down':
        end_angle = math.radians(15.0)
        if cam_type == 'translate_down':
            end_angle = -end_angle
        angles = torch.linspace(0.0, end_angle, steps=F, dtype=torch.float32)
        for i in range(F):
            R = rot_x(float(angles[i]))
            poses[i, :3, :3] = R

    elif cam_type == 'free':
        pass

    else:
        raise NotImplementedError(f"Camera type {cam_type} not implemented.")

    return poses


def save_results(args, results, prompts):
    for key, value in results.items():
        
        if key == "rgbs" and value is not None:  
            rgb = rearrange(value, "b f h w c -> b c f h w").float().cpu()
            filename = os.path.join(args.output_dir, f"rgb.mp4")
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            save_videos_grid(rgb, filename, rescale=False)

        # elif key == "depth_maps":
        #     # colorize the depth map
        #     depth = colorize_depth_map(value[0])[None, ...]  # [F, H, W, 1] -> [B, F, H, W, 3]
        #     depth = rearrange(depth, "b f h w c -> b c f h w").float().cpu()
        #     filename = os.path.join(args.output_dir, f"depth.mp4")
        #     os.makedirs(os.path.dirname(filename), exist_ok=True)
        #     save_videos_grid(depth, filename, rescale=False)

        elif key == "pcds":
            downsample_and_save_pointcloud(
                os.path.join(args.output_dir, f"pcds.ply"),
                value[0],
                results['rgbs'][0],
                voxel_size=0.005 if not args.remove_far_points else 0.003,
                filter_outliers=True, 
                depth_percentile=0,
                remove_far_points=args.remove_far_points,
            )
            
        elif key == 'cameras':
            filename = os.path.join(args.output_dir, f"cameras.json")
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            extrinsics, Ks = value  # [B, F, 3, 4], [B, F, 3, 3]
            extrinsics = torch.cat(
                [extrinsics, 
                 torch.tensor([0, 0, 0, 1], device=extrinsics.device).view(1, 1, 1, 4).repeat(extrinsics.shape[0], extrinsics.shape[1], 1, 1)], dim=2)  # [B, F, 4, 4]
            
            cameras = {
                'extrinsics': extrinsics[0].float().cpu().numpy().tolist(),
                'intrinsics': Ks[0].float().cpu().numpy().tolist(),
            }
            with open(filename, "w") as f:
                json.dump(cameras, f, indent=4)

    # prompts
    filename = os.path.join(args.output_dir, f"prompts.txt")
    with open(filename, "w") as f:
        f.write(prompts[0])


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--pretrained_model_name_or_path", 
        type=str, 
        default='./checkpoints'
    )
    args.add_argument(
        "--task", 
        type=str, 
        default='1view', 
        choices=['1view', '2view', 'allview'],
        help=(
            "Task type: "
            "1view: First Frame to 3D, "
            "2view: First-last Frames to 3D, "
            "allview: 3D Reconstruction."
        )
    )
    args.add_argument(
        "--frame_path", 
        nargs='+', 
        required=True,
        help=(
            "Path to the conditional images or video. "
            "For the allview task, this can be either a directory containing all frames "
            "or a path to the conditional video. "
            "For the other tasks, this should be the path to the conditional image(s)."
        )
    )
    args.add_argument(
        "--cameras", 
        type=str, 
        default='free',
        help=(
            "Path to the conditional camera extrinsics and intrinsics, "
            "or one of the predefined trajectories: "
            + ", ".join(BUILTIN_CAMERA_TRAJECTORIES)
        )
    )
    args.add_argument(
        "--prompts", 
        type=str, 
        required=True,
        help=(
            "The text prompt string or the path to the text prompt file"
        )
    )
    args.add_argument(
        "--output_dir", 
        type=str, 
        default='./results'
    )
    args.add_argument(
        "--remove_far_points", 
        action='store_true', 
        help="Whether to remove far points in the point cloud."
    )
    args = args.parse_args()

    if len(args.frame_path) == 1:
        assert args.task == '1view' or args.task == 'allview'
        if args.task == 'allview':
            assert args.frame_path[0].split('/')[-1].split('.')[-1] == 'mp4' or os.path.isdir(args.frame_path[0])
    elif len(args.frame_path) == 2:
        assert args.task == '2view'

    args.output_dir = os.path.join(args.output_dir, f'{args.task}', f'{datetime.now().strftime("%Y-%m-%d-%H-%M-00")}')
    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipeline = Gen3RPipeline.from_pretrained(args.pretrained_model_name_or_path)
    pipeline.to(device).to(torch.bfloat16)

    # load control images
    control_images = []
    if not args.task == 'allview' or args.task == 'allview' and os.path.isdir(args.frame_path[0]):
        if args.task == 'allview':
            args.frame_path = sorted([os.path.join(args.frame_path[0], f) for f in os.listdir(args.frame_path[0]) if f.endswith('.png') or f.endswith('.jpg')])
        for frame_path in args.frame_path:
            frame = torch.from_numpy(imageio.v2.imread(frame_path))[..., :3]  # [H, W, 3]
            control_images.append(frame)  
        control_images = torch.stack(control_images).to(device, torch.bfloat16)  # [num_frames, H, W, 3]
    else:
        control_images = torch.from_numpy(imageio.v3.imread(args.frame_path[0]))  # [num_frames, H, W, 3]
    
    if args.task == '1view':
        control_images = control_images[:1, ...]  # [1, 3, H, W]
    elif args.task == '2view':
        control_images = control_images[[0, -1], ...]  # [2, 3, H, W]
    else:
        control_images = control_images[:49, ...]  # [num_frames, 3, H, W]
    
    control_images = control_images.permute(0, 3, 1, 2).unsqueeze(0).float() / 255.0  # [1, num_frames, 3, H, W]
    frame_height, frame_width = control_images.shape[3], control_images.shape[4]
    scale = 560 / min(frame_height, frame_width)
    new_height, new_width = round(frame_height * scale), round(frame_width * scale)
    control_images = resize(control_images[0], [new_height, new_width])
    control_images = center_crop(control_images, (560, 560))[None, ...].to(device, torch.bfloat16)  # [1, num_frames, 3, 560, 560]
    
    if args.cameras in BUILTIN_CAMERA_TRAJECTORIES:
        # use vggt to obtain cameras intrinsics and scene scale
        print("Using built-in camera trajectory:", args.cameras)
        with torch.no_grad():
            aggregated_token_list, ps_idx = pipeline.vggt.aggregator(control_images)
            aggregated_token_list = [aggregated_token_list[i] for i in pipeline.vggt.depth_head.intermediate_layer_idx]
            pose_enc = pipeline.vggt.camera_head(aggregated_token_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, control_images.shape[-2:])  # [B, F, 3, 4], [B, F, 3, 3]
            depth_maps, _ = pipeline.vggt.depth_head(aggregated_token_list, control_images, ps_idx)  # [B, F, H, W, 1]

        Ks = intrinsic[:, :1].repeat(1, 49, 1, 1)  # [B, F, 3, 3]
        scene_scale = 0.8 * torch.median(depth_maps).item()
        c2ws = get_poses(cam_type=args.cameras, num_frames=49, scene_scale=scene_scale).to(device)  # [F, 4, 4]
    else:
        with open(args.cameras, "r") as f:
            cameras = json.load(f)
        extrinsics = torch.from_numpy(np.array(cameras['extrinsics']))[:49].to(device)  # [F, 4, 4]
        Ks = torch.from_numpy(np.array(cameras['intrinsics']))[None, :49].to(device)  # [B, F, 3, 3]
        c2ws = torch.linalg.inv(extrinsics)  # [F, 4, 4]
    c2ws = preprocess_poses(c2ws)[None, ...]  # [B, F, 4, 4]

    plucker_embeddings_list = []
    for i in range(len(c2ws)):
        # compute plucker ray embeddings
        rays_o, rays_d = compute_rays(
            c2ws[i], Ks[i], h=560, w=560, device=device
        )  # [F, 3, H, W]
        o_cross_d = torch.cross(rays_o, rays_d, dim=1)
        plucker_embeddings = torch.cat([o_cross_d, rays_d], dim=1)  # [F, 6, H, W]
        plucker_embeddings_list.append(plucker_embeddings)
    plucker_embeddings = torch.stack(plucker_embeddings_list, dim=0)  # [B, F, 6, H, W]
    if args.cameras == 'free' or args.task == 'allview':
        plucker_embeddings = torch.zeros_like(plucker_embeddings)

    # load text prompts
    if os.path.isfile(args.prompts):
        with open(args.prompts, "r", encoding="utf-8") as file:
            prompts = file.readlines()[0]
    else:
        prompts = args.prompts
    
    sample = pipeline(
        prompt = prompts,
        control_cameras = plucker_embeddings,  # [B, F, 6, H, W]
        control_images = control_images,
        num_frames = 49,
        negative_prompt = "bad detailed",
        height      = 560,
        width       = 560,
        guidance_scale = 5,
        return_dict = True,
        min_max_depth_mask = True,
    )
    artifacts = {
        'rgbs': sample.rgbs,  # [B, F, H, W, 3]
        'depth_maps': sample.depth_maps,  # [B, F, H, W, 1]
        "pcds": sample.pcds,  # [B, F, H, W, 3]
        'point_masks': sample.point_masks,  # [B, F, H, W]
        'cameras': sample.cameras,  # [B, F, 3, 4], [B, F, 3, 3]
    }

    save_results(args, artifacts, [prompts])