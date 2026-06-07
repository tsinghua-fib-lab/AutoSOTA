"""Visualization helpers for volumetric renders."""

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from pytorch3d.transforms import so3_exp_map
from pytorch3d.renderer import FoVPerspectiveCameras, look_at_view_transform
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes


try:
    from pytorch3d.vis.plotly_vis import plot_scene
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("Warning: plotly is missing; install via `pip install plotly` for 3D widgets.")


def generate_rotating_volume(volume_model, output_dir, n_frames=50, 
                             camera_distance=1.5, camera_elevation=0.0,
                             device="cuda", threshold=0.5, max_points=100000,
                             illusion_type='parallel_arrows'):

    os.makedirs(output_dir, exist_ok=True)
    
    # Select camera paths according to the requested illusion preset
    if illusion_type == 'parallel_arrows':
        azims = torch.linspace(0, 360, n_frames, device=device)
        Rs_1 = []
        Ts_1 = []
        Rs_2 = []
        Ts_2 = []
        
        for azim in azims:
            R1, T1 = look_at_view_transform(
                dist=camera_distance,
                elev=camera_elevation,
                azim=azim,
                device=device
            )
            Rs_1.append(R1)
            Ts_1.append(T1)
            
            R2, T2 = look_at_view_transform(
                dist=camera_distance,
                elev=camera_elevation,
                azim=azim + 90.0,  # rotated to emphasize illusion
                device=device
            )
            Rs_2.append(R2)
            Ts_2.append(T2)
        
        Rs_1 = torch.cat(Rs_1)
        Ts_1 = torch.cat(Ts_1)
        Rs_2 = torch.cat(Rs_2)
        Ts_2 = torch.cat(Ts_2)
        
    elif illusion_type == 'impossible_object':
        key_angles = [0, 120, 240]  # evenly spaced viewpoints
        azims = torch.linspace(0, 360, n_frames, device=device)
        Rs_1 = []
        Ts_1 = []
        Rs_2 = []
        Ts_2 = []
        Rs_3 = []
        Ts_3 = []
        
        for azim in azims:
            for i, key_angle in enumerate(key_angles):
                R, T = look_at_view_transform(
                    dist=camera_distance,
                    elev=camera_elevation + i * 5,  # slight elevation offsets
                    azim=azim + key_angle,
                    device=device
                )
                if i == 0:
                    Rs_1.append(R)
                    Ts_1.append(T)
                elif i == 1:
                    Rs_2.append(R)
                    Ts_2.append(T)
                else:
                    Rs_3.append(R)
                    Ts_3.append(T)
        
        Rs_1 = torch.cat(Rs_1)
        Ts_1 = torch.cat(Ts_1)
        Rs_2 = torch.cat(Rs_2)
        Ts_2 = torch.cat(Ts_2)
        Rs_3 = torch.cat(Rs_3)
        Ts_3 = torch.cat(Ts_3)
        
    else:
        azims = torch.linspace(0, 360, n_frames, device=device)
        Rs_1 = []
        Ts_1 = []
        Rs_2 = []
        Ts_2 = []
        
        for azim in azims:
            R1, T1 = look_at_view_transform(
                dist=camera_distance,
                elev=camera_elevation,
                azim=azim,
                device=device
            )
            Rs_1.append(R1)
            Ts_1.append(T1)
            

            R2, T2 = look_at_view_transform(
                dist=camera_distance,
                elev=camera_elevation,
                azim=azim + 90.0,
                device=device
            )
            Rs_2.append(R2)
            Ts_2.append(T2)
        
        Rs_1 = torch.cat(Rs_1)
        Ts_1 = torch.cat(Ts_1)
        Rs_2 = torch.cat(Rs_2)
        Ts_2 = torch.cat(Ts_2)
    
    print('Generating rotating volume frames...')
    for i in tqdm(range(n_frames)):
        camera_1 = FoVPerspectiveCameras(
            R=Rs_1[i:i+1], 
            T=Ts_1[i:i+1],
            fov=60.0,
            znear=0.1,
            zfar=10.0,
            device=device,
        )

        camera_2 = FoVPerspectiveCameras(
            R=Rs_2[i:i+1], 
            T=Ts_2[i:i+1],
            fov=60.0,
            znear=0.1,
            zfar=10.0,
            device=device,
        )

        with torch.no_grad():
            render_result_1 = volume_model(camera_1)
            rgb_1 = render_result_1[..., :3].clamp(0.0, 1.0)

            render_result_2 = volume_model(camera_2)
            rgb_2 = render_result_2[..., :3].clamp(0.0, 1.0)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(rgb_1[0].cpu().numpy())
        axes[0].set_title(f'View1 (Front) - Azimuth {azims[i]:.1f} deg')
        axes[0].axis('off')
        

        axes[1].imshow(rgb_2[0].cpu().numpy())
        axes[1].set_title(f'View2 (Side) - Azimuth {azims[i] + 90:.1f} deg')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"frame_{i:03d}.png"), bbox_inches='tight')
        plt.close(fig)
    
    print(f'Frames saved to {output_dir}')
    

    final_frame_idx = n_frames // 4  
    camera_1 = FoVPerspectiveCameras(
        R=Rs_1[final_frame_idx:final_frame_idx+1], 
        T=Ts_1[final_frame_idx:final_frame_idx+1],
        fov=60.0,
        znear=0.1,
        zfar=10.0,
        device=device,
    )
    camera_2 = FoVPerspectiveCameras(
        R=Rs_2[final_frame_idx:final_frame_idx+1], 
        T=Ts_2[final_frame_idx:final_frame_idx+1],
        fov=60.0,
        znear=0.1,
        zfar=10.0,
        device=device,
    )
    
    with torch.no_grad():
        render_result_1 = volume_model(camera_1)
        rgb_1 = render_result_1[..., :3].clamp(0.0, 1.0)
        render_result_2 = volume_model(camera_2)
        rgb_2 = render_result_2[..., :3].clamp(0.0, 1.0)
    

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(np.hstack([rgb_1[0].cpu().numpy(), rgb_2[0].cpu().numpy()]))
    ax.set_title('Visual Illusion Effect - Two Perpendicular Views')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "illusion_effect.png"), bbox_inches='tight')
    plt.close(fig)
    
    return


def generate_volume_with_cameras_visualization(volume_model, cameras, output_path, device, 
                                               threshold=0.05, renderer_kwargs={"resolution": 512}):

    if not HAS_PLOTLY:
        print("Warning: plotly unavailable; falling back to 2D projections.")
        
        with torch.no_grad():
            cam_indices = [0, 1] if len(cameras.R) > 1 else [0, 0]
            views = []
            
            for i in cam_indices:
                camera = FoVPerspectiveCameras(
                    R=cameras.R[i:i+1], 
                    T=cameras.T[i:i+1],
                    znear=cameras.znear[i:i+1],
                    zfar=cameras.zfar[i:i+1],
                    aspect_ratio=cameras.aspect_ratio[i:i+1] if hasattr(cameras, 'aspect_ratio') else None,
                    fov=cameras.fov[i:i+1],
                    device=device,
                )
                rendered = volume_model(camera)
                views.append(rendered[..., :3].clamp(0.0, 1.0))
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            axes[0].imshow(views[0][0].cpu().numpy())
            axes[0].set_title(f'Camera1 View')
            axes[0].axis('off')
            
            axes[1].imshow(views[1][0].cpu().numpy())
            axes[1].set_title(f'Camera2 View')
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.savefig(output_path, bbox_inches='tight')
            plt.close(fig)
            
        return
    
    with torch.no_grad():
        densities = torch.sigmoid(volume_model.log_densities)
        colors = torch.sigmoid(volume_model.log_colors)

        valid_mask = densities[0] > threshold
        
        if valid_mask.sum() == 0:
            print("Warning: no voxels exceed the threshold; consider lowering it.")
            return

        volume_size = densities.shape[1:]
        voxel_size = volume_model._voxel_size
        
        x = torch.arange(volume_size[0], device=device)
        y = torch.arange(volume_size[1], device=device)
        z = torch.arange(volume_size[2], device=device)
        
        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')

        points_x = (grid_x - volume_size[0] / 2) * voxel_size
        points_y = (grid_y - volume_size[1] / 2) * voxel_size
        points_z = (grid_z - volume_size[2] / 2) * voxel_size
        
        coords = torch.stack([points_x, points_y, points_z], dim=-1)

        valid_coords = coords[valid_mask]
        valid_colors = colors[:, valid_mask].permute(1, 0)  # reshape to [N, 3]

        point_cloud = Pointclouds(
            points=[valid_coords],
            features=[valid_colors]
        )

        fig = plot_scene({
            "Volume Points": point_cloud,
        }, camera_scale=0.3, **renderer_kwargs)

        fig.write_image(output_path)


def image_grid(images, rows=None, cols=None, fill=True, rgb=True):

    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()
        
    if rows is None != cols is None:
        raise ValueError("rows and cols must both be specified or both omitted")
        
    if rows is None:
        rows = int(np.sqrt(len(images)))
        cols = (len(images) + rows - 1) // rows
    
    if fill:

        n_miss = rows * cols - len(images)
        if n_miss > 0:
            if rgb:
                blank = np.ones((images.shape[1], images.shape[2], images.shape[3]))
            else:
                blank = np.ones((images.shape[1], images.shape[2]))
            images = np.concatenate([images, np.tile(blank, (n_miss, 1, 1, 1))])

    if rgb:
        grid = np.concatenate([images[i*cols:(i+1)*cols] for i in range(rows)], axis=1)

        grid = np.transpose(grid, (1, 0, 2, 3))

        grid = np.reshape(grid, (grid.shape[0], grid.shape[1] * grid.shape[2], grid.shape[3]))
    else:
        grid = np.concatenate([images[i*cols:(i+1)*cols] for i in range(rows)], axis=1)

        grid = np.transpose(grid, (1, 0, 2))

        grid = np.reshape(grid, (grid.shape[0], grid.shape[1] * grid.shape[2]))
    
    return grid 