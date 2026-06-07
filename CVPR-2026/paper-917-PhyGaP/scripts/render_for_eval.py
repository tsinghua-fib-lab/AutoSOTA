import os
import sys
import time
import torch
from tqdm import tqdm
from torchvision.utils import save_image
from torchvision.io import read_image
# Load model and scene (refer to viz_results for details)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene import Scene, GaussianModel
from gaussian_renderer import render_surfel, render_initial, render_volume, render_surfel_nodefer

from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from datetime import datetime
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
from utils.graphics_utils import linear_to_srgb
from utils.mesh_utils import GaussianExtractor, post_process_mesh

from utils.general_utils import safe_normalize
from utils.refl_utils import sample_camera_rays, reflection
from utils.gridmap_envmap_utils import render_cubemap, build_cubemap,build_surf_cubemap
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# from sklearn_extra.cluster import KMedoids
from utils.stokes_io_utils import load_aolp_dop
from matplotlib import cm
from utils.time_logger import append_time_log

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def viz_targets(targets):
    if len(targets) > 0:
        all_targets = targets.cpu().numpy()
        kmeans = KMeans(n_clusters=4, random_state=0).fit(all_targets)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        # # 使用KMedoids替代KMeans进行聚类
        # kmedoids = KMedoids(n_clusters=4, random_state=0, method='pam').fit(all_targets)
        # labels = kmedoids.labels_
        # centers = kmedoids.cluster_centers_
        fig = plt.figure(figsize=(16, 8))
        # 标记kmeans中心
    
        # 原视角
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        scatter1 = ax1.scatter(
        all_targets[:, 0], all_targets[:, 2], all_targets[:, 1],
        s=1, c=labels, cmap='tab10'
        )
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('Original View')

        # 沿z轴旋转180度
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        scatter2 = ax2.scatter(
        -all_targets[:, 0], -all_targets[:, 2], all_targets[:, 1],
        s=1, c=labels, cmap='tab10'
        )
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_title('Rotated 180° around Z')

        # 调整center为同label target中距该center最近的点
        new_centers = []
        for i in range(centers.shape[0]):
            cluster_points = all_targets[labels == i]
            if len(cluster_points) == 0:
                new_centers.append(centers[i])
                continue
            dists = np.linalg.norm(cluster_points - centers[i], axis=1)
            min_idx = np.argmin(dists)
            new_centers.append(cluster_points[min_idx])
        new_centers = np.stack(new_centers, axis=0)
        centers = new_centers

        # 标记kmeans中心（黑色圆点）
        ax1.scatter(
            centers[:, 0], centers[:, 2], centers[:, 1],
            s=50, c='black', marker='o', edgecolors='k', linewidths=1, label='Centers'
        )
        ax2.scatter(
            -centers[:, 0], -centers[:, 2], centers[:, 1],
            s=50, c='black', marker='o', edgecolors='k', linewidths=1, label='Centers'
        )

        plt.tight_layout()
        plt.savefig(os.path.join("./", "target_points_kmeans_dualview.png"))
        plt.close(fig)
    else:
        print("No target points to visualize.")

def set_gaussian_para(gaussians, opt, vol=False):
    gaussians.enlarge_scale = opt.enlarge_scale
    gaussians.rough_msk_thr = opt.rough_msk_thr 
    gaussians.init_roughness_value = opt.init_roughness_value
    gaussians.init_refl_value = opt.init_refl_value
    gaussians.refl_msk_thr = opt.refl_msk_thr

def reset_gaussian_para(gaussians, opt):
    gaussians.reset_ori_color()
    gaussians.reset_refl_strength(opt.init_refl_value)
    gaussians.reset_roughness(opt.init_roughness_value)
    gaussians.refl_msk_thr = opt.refl_msk_thr
    gaussians.rough_msk_thr = opt.rough_msk_thr
def render_and_save_eval(dataset, opt, pipe, checkpoint, mask_folder, output_root="eval", final_itr=30000, relight=False, relight_env_path=None, subset=None):
    # Setup output dirs
    # If subset is specified, write into subfolder (train/test)
    if subset in ("train", "test"):
        out_root = os.path.join(output_root, subset)
    else:
        out_root = output_root

    pred_dir = os.path.join(out_root, "pred")
    gt_dir = os.path.join(out_root, "gt")
    mask_dir = os.path.join(out_root, "mask")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    render_start_time = time.time()
    camera_count = 0

    with torch.no_grad():
        gaussians = GaussianModel(dataset.albedo_sh_degree, dataset.sh_degree)
        set_gaussian_para(gaussians, opt, vol=(opt.volume_render_until_iter > opt.init_until_iter))
        scene = Scene(dataset, gaussians, load_iteration=final_itr, shuffle=False)
        # import pdb; pdb.set_trace()
        # gaussians.load_ply(os.path.join(os.path.dirname(checkpoint),f"point_cloud/iteration_{final_itr}/point_cloud.ply"),relight=relight,\
        #                             args=dataset,relight_env_path = relight_env_path,env_rotate_degree = [0,0],relight_scale = 3.0)
        # if checkpoint:
        #     (model_params, _) = torch.load(checkpoint)
        #     gaussians.restore(model_params, opt)
        if final_itr >= opt.indirect_from_iter:
            opt.indirect = 1
            bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
            gaussians.load_mesh_from_ply(os.path.dirname(checkpoint), final_itr)
        

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        train_cameras = scene.getTrainCameras()
        test_cameras = scene.getTestCameras()
        if subset == "train":
            cameras = train_cameras
        elif subset == "test":
            cameras = test_cameras
        else:
            cameras = train_cameras + test_cameras
        camera_count = len(cameras)

        if pipe.indirect_type in ['obj_env','surf_env']:
            gaussians.reset_gridmap_envmap()
            if pipe.indirect_type == 'obj_env':
                gridmap_cubemaps = build_cubemap(cameras, gaussians, pipe, background, dataset, opt)
            elif pipe.indirect_type == 'surf_env':
                gridmap_cubemaps = build_surf_cubemap(cameras, gaussians, pipe, background, dataset, opt)
            gaussians.update_gridmap_envmap(gridmap_cubemaps,dataset)

            # import pdb; pdb.set_trace()
            gridmap_env_dir = os.path.join(out_root, "gridmap_env")
            os.makedirs(gridmap_env_dir, exist_ok=True)
            for index, o_env in enumerate(gaussians.gridmap_envmap):
                o_env_dict = gaussians.render_gridmap_envmap(index)
                diff_o_env_dict = gaussians.render_gridmap_envmap(index,mode="diffuse")
                grid = [
                    o_env_dict["env1"].permute(2, 0, 1),
                    o_env_dict["env2"].permute(2, 0, 1),
                ]
                grid = make_grid(grid, nrow=1, padding=10)
                save_image(linear_to_srgb(grid), os.path.join(gridmap_env_dir, f"gridmap_env_{index}.png"))
                diff_grid = [
                    diff_o_env_dict["env1"].permute(2, 0, 1),
                    diff_o_env_dict["env2"].permute(2, 0, 1),
                ]
                diff_grid = make_grid(diff_grid, nrow=1, padding=10)
                save_image(linear_to_srgb(diff_grid), os.path.join(gridmap_env_dir, f"gridmap_env_{index}_diffuse.png"))
        #

        # import pdb; pdb.set_trace()
        env_dict = gaussians.render_env_map()
        diffuse_env_dict = gaussians.render_env_map_diffuse()

        grid = [
            env_dict["env1"].permute(2, 0, 1),
            env_dict["env2"].permute(2, 0, 1),
        ]
        grid = make_grid(grid, nrow=1, padding=10)
        save_image(linear_to_srgb(grid), os.path.join(out_root, f"env.png"))

        diff_grid = [ 
            diffuse_env_dict["env1"].permute(2, 0, 1),
            diffuse_env_dict["env2"].permute(2, 0, 1),
        ]
        diff_grid = make_grid(diff_grid, nrow=1, padding=10)
        # import pdb; pdb.set_trace()
        save_image(linear_to_srgb(diff_grid), os.path.join(out_root, f"env_diffuse.png"))
        import re
        for _, camera in enumerate(tqdm(cameras, desc="Rendering for eval")):
            _name = camera.image_name
            stem = os.path.splitext(_name)[0]
            m = re.search(r"(\d+)", stem)
            if m:
                cam_idx = int(m.group(1))
            else:
                raise ValueError(f"Could not extract integer index from image_name '{_name}'")
            # Render final rgb
            render = render_surfel
            render_pkg = render(camera, gaussians, pipe, background, opt=opt)
            # import pdb; pdb.set_trace()
            rgb = render_pkg['render'].clamp(0,1)
            save_image(linear_to_srgb(rgb), os.path.join(pred_dir, f"{cam_idx:02d}.png"))

            # Save Stokes S1/S2, AoLP, DoP if available
            if 'stokes_combined' in render_pkg:
                stokes = render_pkg['stokes_combined']  # expected shape: (3, H, W, 3) where last dim is [S0,S1,S2]
                # Fallback handling for possible alternative shapes
                # Ensure we have tensors for S0/S1/S2 as CHW for saving and as HW for AoLP/DoP computation
                if stokes.dim() == 4 and stokes.shape[-1] == 3:
                    # CHW + Stokes-last
                    s0_chw = stokes[..., 0]
                    s1_chw = stokes[..., 1]
                    s2_chw = stokes[..., 2]
                elif stokes.dim() == 4 and stokes.shape[0] == 3:
                    # Stokes-first + CHW
                    s0_chw = stokes[0]
                    s1_chw = stokes[1]
                    s2_chw = stokes[2]
                else:
                    # Unknown layout; skip gracefully
                    s0_chw = s1_chw = s2_chw = None

                if s0_chw is not None:
                    # Create output dirs
                    s1_dir = os.path.join(out_root, "s1")
                    s2_dir = os.path.join(out_root, "s2")
                    aolp_dir = os.path.join(out_root, "aolp")
                    dop_dir = os.path.join(out_root, "dop")
                    os.makedirs(s1_dir, exist_ok=True)
                    os.makedirs(s2_dir, exist_ok=True)
                    os.makedirs(aolp_dir, exist_ok=True)
                    os.makedirs(dop_dir, exist_ok=True)

                    eps = 1e-6
                    # Normalize S1/S2 by S0 for visualization, map [-1,1] -> [0,1]
                    s1_vis = 0.5 * (s1_chw / (s0_chw.abs().clamp_min(eps)) + 1.0)
                    s2_vis = 0.5 * (s2_chw / (s0_chw.abs().clamp_min(eps)) + 1.0)
                    s1_vis = s1_vis.clamp(0, 1)
                    s2_vis = s2_vis.clamp(0, 1)

                    # Tonemap pred s1/s2 to twilight colormap (average color channels to grayscale first)
                    s1_gray = s1_vis.mean(dim=0).cpu().numpy()
                    s2_gray = s2_vis.mean(dim=0).cpu().numpy()
                    s1_col = cm.get_cmap('twilight')(s1_gray)[..., :3]
                    s2_col = cm.get_cmap('twilight')(s2_gray)[..., :3]
                    s1_col = torch.from_numpy(s1_col).permute(2, 0, 1).float()
                    s2_col = torch.from_numpy(s2_col).permute(2, 0, 1).float()
                    save_image(s1_col, os.path.join(s1_dir, f"{cam_idx:02d}.png"))
                    save_image(s2_col, os.path.join(s2_dir, f"{cam_idx:02d}.png"))

                    # Compute AoLP and DoP from grayscale-averaged Stokes
                    # Convert CHW -> HW by averaging across color channel
                    s0_hw = s0_chw.mean(dim=0)
                    s1_hw = s1_chw.mean(dim=0)

                    # Also process stokes_spec -> save original (S0) + AoLP + DoP
                    if 'stokes_spec' in render_pkg:
                        st_spec = render_pkg['stokes_spec']
                        if st_spec.dim() == 4 and st_spec.shape[-1] == 3:
                            s0s_chw = st_spec[..., 0]
                            s1s_chw = st_spec[..., 1]
                            s2s_chw = st_spec[..., 2]
                        elif st_spec.dim() == 4 and st_spec.shape[0] == 3:
                            s0s_chw = st_spec[0]
                            s1s_chw = st_spec[1]
                            s2s_chw = st_spec[2]
                        else:
                            s0s_chw = s1s_chw = s2s_chw = None

                        if s0s_chw is not None:
                            spec_orig_dir = os.path.join(out_root, "spec_orig")
                            spec_aolp_dir = os.path.join(out_root, "spec_aolp")
                            spec_dop_dir = os.path.join(out_root, "spec_dop")
                            spec_s1_dir = os.path.join(out_root, "spec_s1")
                            spec_s2_dir = os.path.join(out_root, "spec_s2")
                            os.makedirs(spec_orig_dir, exist_ok=True)
                            os.makedirs(spec_aolp_dir, exist_ok=True)
                            os.makedirs(spec_dop_dir, exist_ok=True)
                            os.makedirs(spec_s1_dir, exist_ok=True)
                            os.makedirs(spec_s2_dir, exist_ok=True)

                            s0s_hw = s0s_chw.mean(dim=0)
                            s1s_hw = s1s_chw.mean(dim=0)
                            s2s_hw = s2s_chw.mean(dim=0)
                            st_spec_hw3 = torch.stack([s0s_hw, s1s_hw, s2s_hw], dim=-1)
                            aolp_s, dop_s = load_aolp_dop(st_spec_hw3)

                            aolp_norm_s = (aolp_s / 180.0).clamp(0, 1)
                            dop_norm_s = dop_s.clamp(0, 1)
                            aolp_col_s = cm.get_cmap('twilight')(aolp_norm_s.cpu().numpy())[..., :3]
                            dop_col_s = cm.get_cmap('viridis')(dop_norm_s.cpu().numpy())[..., :3]
                            aolp_col_s = torch.from_numpy(aolp_col_s).permute(2, 0, 1).float()
                            dop_col_s = torch.from_numpy(dop_col_s).permute(2, 0, 1).float()

                            # Save original intensity (S0) as a 3-channel image
                            s0s_vis = s0s_hw.unsqueeze(0).repeat(3, 1, 1)
                            save_image(linear_to_srgb(s0s_vis), os.path.join(spec_orig_dir, f"{cam_idx:02d}.png"))
                            save_image(aolp_col_s, os.path.join(spec_aolp_dir, f"{cam_idx:02d}.png"))
                            save_image(dop_col_s, os.path.join(spec_dop_dir, f"{cam_idx:02d}.png"))

                            # Normalize per-channel then average to grayscale before applying colormap to avoid 4D colormap output
                            spec_s1_norm = 0.5 * (s1s_chw / (s0s_chw.abs().clamp_min(eps)) + 1.0)
                            spec_s2_norm = 0.5 * (s2s_chw / (s0s_chw.abs().clamp_min(eps)) + 1.0)
                            spec_s1_gray = spec_s1_norm.mean(dim=0).cpu().numpy()
                            spec_s2_gray = spec_s2_norm.mean(dim=0).cpu().numpy()
                            spec_s1_col = cm.get_cmap('twilight')(spec_s1_gray)[..., :3]
                            spec_s2_col = cm.get_cmap('twilight')(spec_s2_gray)[..., :3]
                            spec_s1_col = torch.from_numpy(spec_s1_col).permute(2, 0, 1).float()
                            spec_s2_col = torch.from_numpy(spec_s2_col).permute(2, 0, 1).float()
                            save_image(spec_s1_col, os.path.join(spec_s1_dir, f"{cam_idx:02d}.png"))
                            save_image(spec_s2_col, os.path.join(spec_s2_dir, f"{cam_idx:02d}.png"))

                    # Also process stokes_diff -> save original (S0) + AoLP + DoP
                    if 'stokes_diff' in render_pkg:
                        st_diff = render_pkg['stokes_diff']
                        if st_diff.dim() == 4 and st_diff.shape[-1] == 3:
                            s0d_chw = st_diff[..., 0]
                            s1d_chw = st_diff[..., 1]
                            s2d_chw = st_diff[..., 2]
                        elif st_diff.dim() == 4 and st_diff.shape[0] == 3:
                            s0d_chw = st_diff[0]
                            s1d_chw = st_diff[1]
                            s2d_chw = st_diff[2]
                        else:
                            s0d_chw = s1d_chw = s2d_chw = None

                        if s0d_chw is not None:
                            diff_orig_dir = os.path.join(out_root, "diff_orig")
                            diff_aolp_dir = os.path.join(out_root, "diff_aolp")
                            diff_dop_dir = os.path.join(out_root, "diff_dop")
                            diff_s1_dir = os.path.join(out_root, "diff_s1")
                            diff_s2_dir = os.path.join(out_root, "diff_s2")
                            os.makedirs(diff_orig_dir, exist_ok=True)
                            os.makedirs(diff_aolp_dir, exist_ok=True)
                            os.makedirs(diff_dop_dir, exist_ok=True)
                            os.makedirs(diff_s1_dir, exist_ok=True)
                            os.makedirs(diff_s2_dir, exist_ok=True)

                            s0d_hw = s0d_chw.mean(dim=0)
                            s1d_hw = s1d_chw.mean(dim=0)
                            s2d_hw = s2d_chw.mean(dim=0)
                            st_diff_hw3 = torch.stack([s0d_hw, s1d_hw, s2d_hw], dim=-1)
                            aolp_d, dop_d = load_aolp_dop(st_diff_hw3)

                            aolp_norm_d = (aolp_d / 180.0).clamp(0, 1)
                            dop_norm_d = dop_d.clamp(0, 1)
                            aolp_col_d = cm.get_cmap('twilight')(aolp_norm_d.cpu().numpy())[..., :3]
                            dop_col_d = cm.get_cmap('viridis')(dop_norm_d.cpu().numpy())[..., :3]
                            aolp_col_d = torch.from_numpy(aolp_col_d).permute(2, 0, 1).float()
                            dop_col_d = torch.from_numpy(dop_col_d).permute(2, 0, 1).float()

                            s0d_vis = s0d_hw.unsqueeze(0).repeat(3, 1, 1)
                            save_image(linear_to_srgb(s0d_vis), os.path.join(diff_orig_dir, f"{cam_idx:02d}.png"))
                            save_image(aolp_col_d, os.path.join(diff_aolp_dir, f"{cam_idx:02d}.png"))
                            save_image(dop_col_d, os.path.join(diff_dop_dir, f"{cam_idx:02d}.png"))

                            # Normalize and tonemap diff s1/s2
                            diff_s1_norm = 0.5 * (s1d_chw / (s0d_chw.abs().clamp_min(eps)) + 1.0)
                            diff_s2_norm = 0.5 * (s2d_chw / (s0d_chw.abs().clamp_min(eps)) + 1.0)
                            diff_s1_gray = diff_s1_norm.mean(dim=0).cpu().numpy()
                            diff_s2_gray = diff_s2_norm.mean(dim=0).cpu().numpy()
                            diff_s1_col = cm.get_cmap('twilight')(diff_s1_gray)[..., :3]
                            diff_s2_col = cm.get_cmap('twilight')(diff_s2_gray)[..., :3]
                            diff_s1_col = torch.from_numpy(diff_s1_col).permute(2, 0, 1).float()
                            diff_s2_col = torch.from_numpy(diff_s2_col).permute(2, 0, 1).float()
                            save_image(diff_s1_col, os.path.join(diff_s1_dir, f"{cam_idx:02d}.png"))
                            save_image(diff_s2_col, os.path.join(diff_s2_dir, f"{cam_idx:02d}.png"))
                    s2_hw = s2_chw.mean(dim=0)
                    st_hw3 = torch.stack([s0_hw, s1_hw, s2_hw], dim=-1)
                    aolp, dop = load_aolp_dop(st_hw3)
                    # Color map like save_training_vis: twilight for AoLP, viridis for DoP
                    aolp_norm = (aolp / 180.0).clamp(0, 1)
                    dop_norm = dop.clamp(0, 1)
                    aolp_colored = cm.get_cmap('twilight')(aolp_norm.cpu().numpy())[..., :3]
                    dop_colored = cm.get_cmap('viridis')(dop_norm.cpu().numpy())[..., :3]
                    aolp_colored = torch.from_numpy(aolp_colored).permute(2, 0, 1).float()
                    dop_colored = torch.from_numpy(dop_colored).permute(2, 0, 1).float()
                    save_image(aolp_colored, os.path.join(aolp_dir, f"{cam_idx:02d}.png"))
                    save_image(dop_colored, os.path.join(dop_dir, f"{cam_idx:02d}.png"))
                gt_stokes = camera.original_stokes if camera.original_stokes is not None else None
                if gt_stokes is not None:
                    gt_s1_chw = gt_stokes[..., 1]
                    gt_s2_chw = gt_stokes[..., 2]
                    if gt_s1_chw is not None and gt_s2_chw is not None:
                        # Create output dirs
                        gt_s1_dir = os.path.join(out_root, "gt_s1")
                        gt_s2_dir = os.path.join(out_root, "gt_s2")
                        os.makedirs(gt_s1_dir, exist_ok=True)
                        os.makedirs(gt_s2_dir, exist_ok=True)

                        eps = 1e-6
                        # Normalize S1/S2 by S0 for visualization, map [-1,1] -> [0,1]
                        gt_s0_chw = gt_stokes[..., 0]
                        gt_s1_vis = 0.5 * (gt_s1_chw / (gt_s0_chw.abs().clamp_min(eps)) + 1.0)
                        gt_s2_vis = 0.5 * (gt_s2_chw / (gt_s0_chw.abs().clamp_min(eps)) + 1.0)
                        gt_s1_vis = gt_s1_vis.clamp(0, 1)
                        gt_s2_vis = gt_s2_vis.clamp(0, 1)
                        save_image(gt_s1_vis, os.path.join(gt_s1_dir, f"{cam_idx:02d}.png"))
                        save_image(gt_s2_vis, os.path.join(gt_s2_dir, f"{cam_idx:02d}.png"))

                        # Tonemap gt_s1 and gt_s2 to twilight colormap (average color channels to grayscale first)
                        gt_s1_gray = gt_s1_vis.mean(dim=0).cpu().numpy()
                        gt_s2_gray = gt_s2_vis.mean(dim=0).cpu().numpy()
                        gt_s1_col = cm.get_cmap('twilight')(gt_s1_gray)[..., :3]
                        gt_s2_col = cm.get_cmap('twilight')(gt_s2_gray)[..., :3]
                        gt_s1_col = torch.from_numpy(gt_s1_col).permute(2, 0, 1).float()
                        gt_s2_col = torch.from_numpy(gt_s2_col).permute(2, 0, 1).float()
                        save_image(gt_s1_col, os.path.join(gt_s1_dir, f"{cam_idx:02d}.png"))
                        save_image(gt_s2_col, os.path.join(gt_s2_dir, f"{cam_idx:02d}.png"))
            # Save base_color_map (albedo)
            albedo_dir = os.path.join(out_root, "albedo")
            os.makedirs(albedo_dir, exist_ok=True)
            if 'base_color_map' in render_pkg:
                base_color = render_pkg['base_color_map'].clamp(0, 1)
                save_image(linear_to_srgb(base_color), os.path.join(albedo_dir, f"{cam_idx:02d}.png"))

            # Save normal_map
            normal_dir = os.path.join(out_root, "normal")
            dep_normal_dir = os.path.join(out_root, "depth_normal")
            diff_env_dir = os.path.join(out_root, "diffuse_env")
            os.makedirs(diff_env_dir, exist_ok=True)
            os.makedirs(dep_normal_dir, exist_ok=True)
            os.makedirs(normal_dir, exist_ok=True)
            if 'rend_normal' in render_pkg:
                normal_map = render_pkg['rend_normal']
                normal_map_vis = (normal_map * 0.5 + 0.5).clamp(0, 1)  # Normalize to [0, 1] for visualization
                save_image(normal_map_vis, os.path.join(normal_dir, f"{cam_idx:02d}.png"))
                render_alpha = render_pkg['rend_alpha']
                diff_env = gaussians.get_envmap(normal_map.permute(1,2,0)/render_alpha.permute(1,2,0).clamp_min(1e-6), mode="diffuse")
                save_image(linear_to_srgb(diff_env.permute(2,0,1)), os.path.join(diff_env_dir, f"{cam_idx:02d}.png"))
            if 'surf_normal' in render_pkg:
                dep_normal_map = render_pkg['surf_normal']
                dep_normal_map_vis = (dep_normal_map * 0.5 + 0.5).clamp(0, 1)  # Normalize to [0, 1] for visualization
                save_image(dep_normal_map_vis, os.path.join(dep_normal_dir, f"{cam_idx:02d}.png"))
            # Save roughness
            roughness_dir = os.path.join(out_root, "roughness")
            os.makedirs(roughness_dir, exist_ok=True)
            if 'roughness_map' in render_pkg:
                roughness_map = render_pkg['roughness_map'].clamp(0, 1)
                save_image(roughness_map, os.path.join(roughness_dir, f"{cam_idx:02d}.png"))

            # Save specular
            specular_dir = os.path.join(out_root, "specular")
            os.makedirs(specular_dir, exist_ok=True)
            if 'specular_map' in render_pkg:
                specular_map = render_pkg['specular_map'].clamp(0, 1)
                save_image(linear_to_srgb(specular_map), os.path.join(specular_dir, f"{cam_idx:02d}.png"))

            # Save diffuse
            diffuse_dir = os.path.join(out_root, "diffuse")
            os.makedirs(diffuse_dir, exist_ok=True)
            if 'diffuse_map' in render_pkg:
                diffuse_map = render_pkg['diffuse_map'].clamp(0, 1)
                save_image(linear_to_srgb(diffuse_map), os.path.join(diffuse_dir, f"{cam_idx:02d}.png"))
            # Save half_eta_map (grayscale visualization)
            half_eta_dir = os.path.join(out_root, "half_eta")
            os.makedirs(half_eta_dir, exist_ok=True)
            if 'half_eta_map' in render_pkg:
                half_eta = render_pkg['half_eta_map']
                # Move to CPU and clamp for saving
                half_eta = half_eta.detach().clamp(0, 1).cpu()
                # Accept CHW or HW formats
                if half_eta.dim() == 2:
                    half_eta = half_eta.unsqueeze(0)
                if half_eta.dim() == 3:
                    # If single-channel, repeat to 3 channels for visualization
                    if half_eta.shape[0] == 1:
                        save_image(half_eta.repeat(3, 1, 1), os.path.join(half_eta_dir, f"{cam_idx:02d}.png"))
                    elif half_eta.shape[0] == 3:
                        save_image(half_eta, os.path.join(half_eta_dir, f"{cam_idx:02d}.png"))
                    else:
                        # Unexpected channel layout: convert to HWC then pick first 3 channels
                        try:
                            arr = half_eta.permute(1, 2, 0).numpy()
                            arr = arr[..., :3]
                            save_image(torch.from_numpy(arr).permute(2, 0, 1).float(), os.path.join(half_eta_dir, f"{cam_idx:02d}.png"))
                        except Exception:
                            # Fallback: save the mean over channels
                            save_image(half_eta.mean(dim=0, keepdim=True).repeat(3, 1, 1), os.path.join(half_eta_dir, f"{cam_idx:02d}.png"))
            # Save gt_stokes[...,0]
            # import pdb; pdb.set_trace()
            gt_stokes = camera.original_stokes if camera.original_stokes is not None else None
            if gt_stokes is not None:
                gt_s0 = gt_stokes[...,0].cpu()
                save_image(linear_to_srgb(gt_s0), os.path.join(gt_dir, f"{cam_idx:02d}.png"))
            else:
                # If no gt_stokes, save original image as gt
                gt_img = camera.original_image
                save_image(linear_to_srgb(gt_img), os.path.join(gt_dir, f"{cam_idx:02d}.png"))
            
            visibility_dir = os.path.join(out_root,"visibility")
            os.makedirs(visibility_dir,exist_ok=True)
            if 'ray_trace_visibility' in render_pkg:
                visibility = render_pkg['ray_trace_visibility'].clamp(0, 1)
                save_image(visibility.repeat(3,1,1),os.path.join(visibility_dir, f"{cam_idx:02d}.png"))

            indirect_dir = os.path.join(out_root,"indirect")
            os.makedirs(indirect_dir,exist_ok = True)
            if "indirect" in render_pkg:
                indirect = render_pkg['indirect'].clamp(0,1)
                save_image(indirect, os.path.join(indirect_dir,f"{cam_idx:02d}.png"))
                
            # import pdb; pdb.set_trace()
            # Copy mask
            mask_img = camera.gt_alpha_mask
            save_image(mask_img, os.path.join(mask_dir, f"{cam_idx:02d}.png"))

    duration = time.time() - render_start_time
    subset_label = subset if subset is not None else "all"
    # append_time_log(
    #     f"RENDER subset={subset_label} output={out_root} duration={duration:.2f}s cameras={camera_count} checkpoint={checkpoint}"
    # )

# 用法示例（需根据实际参数替换）
# render_and_save(dataset, opt, pipe, checkpoint, mask_folder="/path/to/mask", output_root="eval")
if __name__ == "__main__":
    parser = ArgumentParser(description="Render for eval with optional cfg_args bootstrap from checkpoint")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10000,20000,30000,40000,50000,60000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[10000,20000,30000,40000,50000])
    parser.add_argument("--checkpoint", type=str, default = None)
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint .pth (used to locate cfg_args)")
    parser.add_argument("--output_dir", type=str, default="debug/test", help="Directory to save the output images")
    parser.add_argument("--final_iterations",type = int, default = 50000, help="Final iterations to run the evaluation")
    parser.add_argument("--mask_folder", type=str, default=None, help="Folder containing the masks for the images")
    parser.add_argument("--render_relight",  action ="store_true",default=False, help="if relight")
    parser.add_argument("--relight_env_path", type=str, default=None, help="the path to relight env map")
    parser.add_argument("--subset", type=str, choices=["train", "test", "both"], default="both", help="Which camera subset to render")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.test_iterations = args.test_iterations + [i for i in range(10000, args.iterations+1, 5000)]
    args.test_iterations.append(args.volume_render_until_iter)

    script_start_time = time.time()
    # append_time_log(
    #     f"RENDER script start output_dir={args.output_dir} subset={args.subset}"
    # )

    # Bootstrap dataset/opt/pipe from cfg_args next to checkpoint (like render_envmap_rotation)
    ckpt_path = args.ckpt if args.ckpt is not None else args.checkpoint
    if ckpt_path is None:
        raise FileNotFoundError("Please provide --ckpt or --checkpoint to locate cfg_args")
    ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))
    cfg_path = os.path.join(ckpt_dir, 'cfg_args')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"cfg_args not found in {ckpt_dir}")
    with open(cfg_path, 'r') as f:
        cfg_text = f.read()
    # Safe eval: only allow argparse.Namespace
    cfg_ns = eval(cfg_text, {"__builtins__": {}}, {"Namespace": Namespace})

    # Instantiate parameter builders with fresh parsers and extract from cfg_ns
    lp_boot = ModelParams(ArgumentParser())
    op_boot = OptimizationParams(ArgumentParser())
    pp_boot = PipelineParams(ArgumentParser())
    dataset = lp_boot.extract(cfg_ns)
    opt = op_boot.extract(cfg_ns)
    pipe = pp_boot.extract(cfg_ns)
    opt.voxel_size = 1
    if dataset.double_view:
        dataset.double_view = False  # Disable double view loading for eval rendering
        orig_source_path = dataset.source_path
        for cam in ["camera_0", "camera_1"]:
            # import pdb; pdb.set_trace()
            dataset.source_path = os.path.join(orig_source_path, cam)
            # import pdb; pdb.set_trace()
            output_subdir = os.path.join(args.output_dir, cam)
            os.makedirs(output_subdir, exist_ok=True)
            if args.subset == "both":
                # Render train
                render_and_save_eval(dataset, opt, pipe, ckpt_path, mask_folder=args.mask_folder, output_root=output_subdir,
                                    final_itr=args.final_iterations, relight=args.render_relight, relight_env_path=args.relight_env_path, subset="train")
                # Render test
                render_and_save_eval(dataset, opt, pipe, ckpt_path, mask_folder=args.mask_folder, output_root=output_subdir,
                                    final_itr=args.final_iterations, relight=args.render_relight, relight_env_path=args.relight_env_path, subset="test")
            else:
                render_and_save_eval(dataset, opt, pipe, ckpt_path, mask_folder=args.mask_folder, output_root=output_subdir,
                                    final_itr=args.final_iterations, relight=args.render_relight, relight_env_path=args.relight_env_path, subset=args.subset)
            # import pdb; pdb.set_trace()
    else:
        # import pdb; pdb.set_trace()
        if args.subset == "both":
            # Render train
            render_and_save_eval(dataset, opt, pipe, ckpt_path, mask_folder=args.mask_folder, output_root=args.output_dir,
                                final_itr=args.final_iterations, relight=args.render_relight, relight_env_path=args.relight_env_path, subset="train")
            # Render test
            render_and_save_eval(dataset, opt, pipe, ckpt_path, mask_folder=args.mask_folder, output_root=args.output_dir,
                                final_itr=args.final_iterations, relight=args.render_relight, relight_env_path=args.relight_env_path, subset="test")
        else:
            render_and_save_eval(dataset, opt, pipe, ckpt_path, mask_folder=args.mask_folder, output_root=args.output_dir,
                                final_itr=args.final_iterations, relight=args.render_relight, relight_env_path=args.relight_env_path, subset=args.subset)

    total_script_duration = time.time() - script_start_time
    # append_time_log(
    #     f"RENDER script complete output_dir={args.output_dir} subset={args.subset} duration={total_script_duration:.2f}s"
    # )