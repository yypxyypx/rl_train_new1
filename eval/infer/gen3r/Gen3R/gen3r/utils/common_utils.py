import os
import imageio
import inspect
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import cv2
import open3d as o3d
from einops import rearrange
from PIL import Image


def filter_kwargs(cls, kwargs):
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs


def color_transfer(sc, dc):
    """
    Transfer color distribution from of sc, referred to dc.

    Args:
        sc (numpy.ndarray): input image to be transfered.
        dc (numpy.ndarray): reference image

    Returns:
        numpy.ndarray: Transferred color distribution on the sc.
    """
    def get_mean_and_std(img):
        x_mean, x_std = cv2.meanStdDev(img)
        x_mean = np.hstack(np.around(x_mean, 2))
        x_std = np.hstack(np.around(x_std, 2))
        return x_mean, x_std

    sc = cv2.cvtColor(sc, cv2.COLOR_RGB2LAB)
    s_mean, s_std = get_mean_and_std(sc)
    dc = cv2.cvtColor(dc, cv2.COLOR_RGB2LAB)
    t_mean, t_std = get_mean_and_std(dc)
    img_n = ((sc - s_mean) * (t_std / s_std)) + t_mean
    np.putmask(img_n, img_n > 255, 255)
    np.putmask(img_n, img_n < 0, 0)
    dst = cv2.cvtColor(cv2.convertScaleAbs(img_n), cv2.COLOR_LAB2RGB)
    return dst


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=12, imageio_backend=True, color_transfer_post_process=False):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(Image.fromarray(x))

    if color_transfer_post_process:
        for i in range(1, len(outputs)):
            outputs[i] = Image.fromarray(color_transfer(np.uint8(outputs[i]), np.uint8(outputs[0])))
    
    if os.path.dirname(path) != '':
        os.makedirs(os.path.dirname(path), exist_ok=True)

    if imageio_backend:
        if path.endswith("mp4"):
            imageio.mimsave(path, outputs, fps=fps)
        else:
            imageio.mimsave(path, outputs, duration=(1000 * 1/fps), loop=0)
    else:
        if path.endswith("mp4"):
            path = path.replace('.mp4', '.gif')
        outputs[0].save(path, format='GIF', append_images=outputs, save_all=True, duration=100, loop=0)


def string_to_filename(s: str) -> str:
    if len(s) > 0:
        return (
            s.replace(" ", "-")
            .replace("/", "-")
            .replace(":", "-")
            .replace(".", "-")
            .replace(",", "-")
            .replace(";", "-")
            .replace("!", "-")
            .replace("?", "-")
        )
    else:
        return 'no_prompt'


def spectral_cmap(x: torch.Tensor):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    x_rgb = torch.zeros((x.shape[0], x.shape[1]), dtype=torch.float32, device=x.device)
    x_max, x_min = x.max(), x.min()
    x_normalize = (x - x_min) / (x_max - x_min)
    x_rgb = torch.clamp(x_normalize, 0, 1)
    x_rgb = plt.get_cmap('Spectral')(x_rgb.float().cpu().numpy(), bytes=True)[..., :3] / 255.
    return torch.from_numpy(x_rgb)


def colorize_depth_map(depth_maps: torch.Tensor) -> torch.Tensor:
    if depth_maps.ndim > 3:
        depth_maps = depth_maps.squeeze(-1)   
    colorized_depth_maps = torch.zeros((depth_maps.shape[0], depth_maps.shape[1], depth_maps.shape[2], 3), device=depth_maps.device)
    
    for i in range(depth_maps.shape[0]):
        colorized_depth_maps[i] = spectral_cmap(depth_maps[i]).to(depth_maps.device)
    
    return colorized_depth_maps


def convert_to_token_list(latents: torch.Tensor, patch_size: int) -> torch.Tensor:

    B, T, H, W, D = latents.shape
    D = D // 5

    frames = torch.zeros(
        (B, T, 3, H*patch_size, W*patch_size), device=latents.device, dtype=latents.dtype)
    cam_register_tokens = latents[..., -D:].mean(dim=(2, 3)).unsqueeze(2).repeat(1, 1, 5, 1)  # [B, T, 5, 2048]
    aggregated_token_list = [
        torch.cat([cam_register_tokens, latents[..., :D].reshape(B, T, -1, D)], dim=2),
        torch.cat([cam_register_tokens, latents[..., D:2*D].reshape(B, T, -1, D)], dim=2),
        torch.cat([cam_register_tokens, latents[..., 2*D:3*D].reshape(B, T, -1, D)], dim=2),
        torch.cat([cam_register_tokens, latents[..., 3*D:4*D].reshape(B, T, -1, D)], dim=2),
    ]

    return aggregated_token_list, frames


def downsample_and_save_pointcloud(filename, point_maps, rgbs, voxel_size=0.005, filter_outliers=True, depth_percentile=0, remove_far_points=False):
    # downsample the point map using open3d
    points = point_maps.reshape(-1, 3).contiguous().float()
    colors = torch.clip(rgbs.reshape(-1, 3).float(), 0.0, 1.0)

    if remove_far_points:
        depths = points[:, 2]
        thresh = float(np.percentile(depths.cpu().numpy(), 80))
        keep_mask = depths <= thresh
        points = points[keep_mask]
        colors = colors[keep_mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(colors.cpu().numpy())

    if filter_outliers:
        # statistical outlier removal
        filtered, ind = pcd.remove_statistical_outlier(nb_neighbors=25, std_ratio=1.25)

        # radius outlier removal (optional)
        pts_all = np.asarray(filtered.points)
        depths = pts_all[:, 2]
        thresh = float(np.percentile(depths, depth_percentile))

        near_idx = np.where(depths <= thresh)[0].tolist()
        far_idx = np.where(depths > thresh)[0].tolist()

        near_pcd = filtered.select_by_index(near_idx) if near_idx else o3d.geometry.PointCloud()
        far_pcd = filtered.select_by_index(far_idx) if far_idx else o3d.geometry.PointCloud()

        if len(near_pcd.points) == 0 or np.all(np.asarray(near_pcd.points)[:, 2] == thresh):
            filtered_near = near_pcd
        else:
            pts = np.asarray(near_pcd.points)
            m = len(pts)
            if m < 2:
                filtered_near = near_pcd
            else:
                sample_n = min(2000, m)
                idxs = np.random.choice(m, sample_n, replace=False)
                kdtree = o3d.geometry.KDTreeFlann(near_pcd)
                nn_dists = []
                for i in idxs:
                    _, idx_neigh, dist2 = kdtree.search_knn_vector_3d(near_pcd.points[i], 2)
                    if len(dist2) >= 2:
                        nn_dists.append(np.sqrt(dist2[1]))
                if nn_dists:
                    median_nn = float(np.median(nn_dists))
                    r_used = 4.0 * median_nn
                        
                filtered_near, ind2 = near_pcd.remove_radius_outlier(nb_points=10, radius=r_used)

        def concat_pcd(a: o3d.geometry.PointCloud, b: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
            pa = np.asarray(a.points) if len(a.points) > 0 else np.zeros((0, 3))
            pb = np.asarray(b.points) if len(b.points) > 0 else np.zeros((0, 3))
            new = o3d.geometry.PointCloud()
            if pa.size == 0 and pb.size == 0:
                return new
            if pa.size == 0:
                new.points = o3d.utility.Vector3dVector(pb)
            elif pb.size == 0:
                new.points = o3d.utility.Vector3dVector(pa)
            else:
                new.points = o3d.utility.Vector3dVector(np.vstack([pa, pb]))

            # colors
            ca = np.asarray(a.colors) if len(a.colors) > 0 else None
            cb = np.asarray(b.colors) if len(b.colors) > 0 else None
            if ca is not None or cb is not None:
                if ca is None:
                    new.colors = o3d.utility.Vector3dVector(cb)
                elif cb is None:
                    new.colors = o3d.utility.Vector3dVector(ca)
                else:
                    new.colors = o3d.utility.Vector3dVector(np.vstack([ca, cb]))

            return new

        pcd = concat_pcd(filtered_near, far_pcd)

    if voxel_size is not None:
        pcd_downsampled = pcd.voxel_down_sample_and_trace(
            voxel_size=voxel_size, min_bound=pcd.get_min_bound(), max_bound=pcd.get_max_bound())[0]
    else:
        pcd_downsampled = pcd
    
    # print(f"Saving pointmaps to {filename}")
    o3d.io.write_point_cloud(filename, pcd_downsampled)