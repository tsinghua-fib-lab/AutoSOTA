import ast
import os
import sys
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

from argparse import ArgumentParser
import argparse

def _get_HWK_from_camera(cam):
    """Return (H, W, K) for the given camera. If camera.HWK exists, use it; otherwise build K from FoV and image size."""
    if hasattr(cam, 'HWK') and cam.HWK is not None:
        return cam.HWK
    # Build intrinsics from FoV and resolution
    H = int(cam.image_height)
    W = int(cam.image_width)
    # FoVx/FoVy are radians; utils.graphics_utils.fov2focal expects radians
    from utils.graphics_utils import fov2focal
    fx = fov2focal(cam.FoVx, W)
    fy = fov2focal(cam.FoVy, H)
    cx = W / 2.0
    cy = H / 2.0
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    return (H, W, K)

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
def main():

    parser = ArgumentParser(description="Render a single view with envmap rotation sweep.")
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--env_map', type=str, required=True, help='Path to envmap (.hdr/.exr)')
    #../mitsuba3/scenes/envmaps/museum.hdr
    parser.add_argument('--idx', type=int, required=True, help='Camera index to render')
    parser.add_argument('--output_dir', type=str, default='envmap_rot_sweep', help='Output directory')
    parser.add_argument('--final_itr', type=int, default=20000, help='Final iteration number to load the model')
    parser.add_argument('--env_view_mode', type=str, default='direct', choices=['direct','scale','hfov','native'],
                        help='Mode to generate env_view: direct=build rays with FoV; scale=shrink original focal; hfov=recompute intrinsics from HFOV; native=use original camera K with sample_camera_rays.')
    parser.add_argument('--env_view_hfov_deg', type=float, default=90.0, help='Horizontal FOV (deg) for env_view when mode=direct or mode=hfov (default 60).')
    parser.add_argument('--env_view_focal_scale', type=float, default=0.25, help='Scale factor to shrink camera focal for env_view when mode=scale (e.g., 0.25≈4x wider).')
    args = parser.parse_args()
    with torch.no_grad():
        # 自动读取cfg_args
        ckpt_dir = os.path.dirname(os.path.abspath(args.ckpt))
        cfg_path = os.path.join(ckpt_dir, 'cfg_args')
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"cfg_args not found in {ckpt_dir}")
        with open(cfg_path, 'r') as f:
            cfg_text = f.read()
        # 安全地通过 argparse.Namespace 解析 cfg_args
        cfg_ns = eval(cfg_text, {"__builtins__": {}}, {"Namespace": argparse.Namespace})
        # Generate rotation angles with 5-degree intervals
        angles = [[yaw, 0] for yaw in range(0, 360, 30)]
        # angles = [[0,0],[45,0],[90,0],[135,0],[180,0],[225,0],[270,0],[315,0]]#,[30,0],[60,0],[90,0],[120,0],[150,0],[180,0],[210,0],[240,0],[270,0],[300,0],[330,0]]
        #[0,0],[45,0],[90,0],[135,0],[180,0],[225,0],[270,0],[315,0]
        assert isinstance(angles, list) and all(isinstance(x, (list, tuple)) and len(x)==2 for x in angles)
        angles = angles[::-1]  # Reverse order for convenience
        os.makedirs(args.output_dir, exist_ok=True)

        # Load model and scene
        # (model_params, _) = torch.load(args.ckpt)
        from arguments import ModelParams, PipelineParams, OptimizationParams
        # 这些类需要 parser 作为构造参数
        lp = ModelParams(ArgumentParser())
        op = OptimizationParams(ArgumentParser())
        pp = PipelineParams(ArgumentParser())
        dataset = lp.extract(cfg_ns)
        opt = op.extract(cfg_ns)
        pipe = pp.extract(cfg_ns)
        dataset.envmap_max_res = 256
        gaussians = GaussianModel(dataset.albedo_sh_degree, dataset.sh_degree)
        set_gaussian_para(gaussians, opt, vol=(opt.volume_render_until_iter > opt.init_until_iter))
        scene = Scene(dataset, gaussians, shuffle = False)
        gaussians.load_ply(os.path.join(ckpt_dir,f"point_cloud/iteration_{args.final_itr}/point_cloud.ply"),relight=True,\
                                        args=dataset,relight_env_path = args.env_map,env_rotate_degree = [0,0],relight_scale = 1.0)
        if args.final_itr >= opt.indirect_from_iter:
            opt.indirect = 1
            bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
            gaussians.load_mesh_from_ply(ckpt_dir, args.final_itr)
        

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        # Get camera
        cameras = scene.getTrainCameras() + scene.getTestCameras()
        idx_strs = [str(args.idx)] + [f"{args.idx:0{n}d}" for n in range(2, 7)]
        camera = None
        # 1) exact match on common zero-pads
        for cam in cameras:
            name = getattr(cam, 'image_name', '')
            if name in idx_strs:
                camera = cam
                break
        # 2) numeric name equality
        if camera is None:
            for cam in cameras:
                name = getattr(cam, 'image_name', '')
                if name.isdigit() and int(name) == args.idx:
                    camera = cam
                    break
        # 3) trailing digits match
        import re
        if camera is None:
            for cam in cameras:
                name = getattr(cam, 'image_name', '')
                m = re.search(r'(\d+)$', name)
                if m and int(m.group(1)) == args.idx:
                    camera = cam
                    break

        if camera is None:
            samples = [getattr(c, 'image_name', '') for c in cameras[:10]]
            raise ValueError(f"No camera found with image_name matching idx={args.idx}. First names: {samples} (total {len(cameras)}).")
        # import pdb;pdb.set_trace()
        for i, (yaw, pitch) in enumerate(angles):
            # if i == len(angles_reverse)-1:
            #     import pdb; pdb.set_trace() 
            gaussians.fix_env_map(args.env_map, dataset, trainable_env=False, env_rotate_degree=[yaw, pitch])
            bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            cameras = scene.getTrainCameras().copy() + scene.getTestCameras().copy()
            gaussians.reset_gridmap_envmap()
            if pipe.indirect_type == 'obj_env':
                gridmap_cubemaps = build_cubemap(cameras, gaussians, pipe, background, dataset, opt)
                gaussians.update_gridmap_envmap(gridmap_cubemaps,dataset)
            elif pipe.indirect_type == 'surf_env':
                gridmap_cubemaps = build_surf_cubemap(cameras, gaussians, pipe, background, dataset, opt)
                gaussians.update_gridmap_envmap(gridmap_cubemaps,dataset)
            else:
                pipe.indirect_type = 'obj_env'
                gridmap_cubemaps = build_cubemap(cameras, gaussians, pipe, background, dataset, opt)
                gaussians.update_gridmap_envmap(gridmap_cubemaps,dataset)
            
            render_pkg = render_surfel(camera, gaussians, pipe, torch.tensor([0,0,0], dtype=torch.float32, device='cuda'), opt=opt)
            img = render_pkg['render']
            albedo = render_pkg["base_color_map"].clamp(0,1)
            specular = render_pkg["specular_map"]
            diffuse = render_pkg["diffuse_map"]
            render_path = os.path.join(args.output_dir, f"rendered_images")
            os.makedirs(render_path, exist_ok=True)
            albedo_path = os.path.join(args.output_dir, f"albedo_images")
            os.makedirs(albedo_path, exist_ok=True)
            diffuse_path = os.path.join(args.output_dir, f"diffuse_images")
            os.makedirs(diffuse_path, exist_ok=True)
            specular_path = os.path.join(args.output_dir, f"specular_images")
            os.makedirs(specular_path, exist_ok=True)
            env_path = os.path.join(args.output_dir, f"envmaps")
            os.makedirs(env_path, exist_ok=True)
            f_name = os.path.join(render_path, f"img_idx{args.idx}_yaw{yaw:03d}_pitch{pitch:03d}.png")
            # fname = os.path.join(args.output_dir, f"img_idx{args.idx}_yaw{yaw:03d}_pitch{pitch:03d}.png")
            # import pdb; pdb.set_trace()
            mask = render_pkg['rend_alpha']
        # Combine mask with img, albedo, specular, and diffuse to create RGBA images
            img_rgba = torch.cat([linear_to_srgb(img), mask], dim=0)
            albedo_rgba = torch.cat([linear_to_srgb(albedo), mask], dim=0)
            specular_rgba = torch.cat([linear_to_srgb(specular), mask], dim=0)
            diffuse_rgba = torch.cat([linear_to_srgb(diffuse), mask], dim=0)
            save_image(img_rgba,os.path.join(render_path, f"img_idx{args.idx}_yaw{yaw:03d}_pitch{pitch:03d}.png"))
            save_image(albedo_rgba, os.path.join(albedo_path, f"img_idx{args.idx}_yaw{yaw:03d}_pitch{pitch:03d}.png"))
            save_image(diffuse_rgba, os.path.join(diffuse_path, f"img_idx{args.idx}_yaw{yaw:03d}_pitch{pitch:03d}.png"))
            save_image(specular_rgba, os.path.join(specular_path, f"img_idx{args.idx}_yaw{yaw:03d}_pitch{pitch:03d}.png"))

            # Also export environment-only image sampled along camera view directions
            try:
                if getattr(gaussians, 'env_map', None) is not None:
                    H, W, K_old = _get_HWK_from_camera(camera)
                    H = 1080
                    W = 1920
                    cx, cy = K_old[0, 2], K_old[1, 2]
                    mode = getattr(args, 'env_view_mode', 'direct')

                    if mode == 'direct':
                        # Build rays directly with specified HFOV using camera R (ignore sample_camera_rays).
                        hfov = np.deg2rad(float(args.env_view_hfov_deg))
                        fx = 0.5 * W / np.tan(0.5 * hfov)
                        vfov = 2.0 * np.arctan((H / W) * np.tan(0.5 * hfov))
                        fy = 0.5 * H / np.tan(0.5 * vfov)
                        cx = W/2.0; cy = H/2.0
                        i = torch.arange(W, device="cuda", dtype=torch.float32)
                        j = torch.arange(H, device="cuda", dtype=torch.float32)
                        ii, jj = torch.meshgrid(i, j, indexing='xy')
                        x = (ii - cx) / fx
                        y = (jj - cy) / fy
                        z = torch.ones_like(x)
                        d_cam = torch.stack([x, y, z], dim=-1)
                        d_cam = F.normalize(d_cam, dim=-1)

                        R_use = camera.R.transpose(0, 1).contiguous()
                        rays_cam = torch.matmul(d_cam, R_use)
                        rays_cam = F.normalize(rays_cam, dim=-1)

                    elif mode == 'scale':
                        # Shrink the original focal lengths by a scale to widen the view.
                        scale = float(getattr(args, 'env_view_focal_scale', 0.25))
                        scale = max(scale, 1e-6)
                        fx_new = float(K_old[0, 0]) * scale
                        fy_new = float(K_old[1, 1]) * scale
                        K_new = np.array([[fx_new, 0.0, cx],
                                          [0.0,   fy_new, cy],
                                          [0.0,   0.0,   1.0]], dtype=np.float32)
                        HWK = (H, W, K_new)
                        rays_cam, _ = sample_camera_rays(HWK, camera.R, camera.T)

                    elif mode == 'hfov':
                        # Recompute intrinsics from a target HFOV and sample using sample_camera_rays.
                        target_hfov = np.deg2rad(float(args.env_view_hfov_deg))
                        fx_new = 0.5 * W / np.tan(0.5 * target_hfov)
                        target_vfov = 2.0 * np.arctan((H / W) * np.tan(0.5 * target_hfov))
                        fy_new = 0.5 * H / np.tan(0.5 * target_vfov)
                        K_new = np.array([[fx_new, 0.0, cx],
                                          [0.0,   fy_new, cy],
                                          [0.0,   0.0,   1.0]], dtype=np.float32)
                        HWK = (H, W, K_new)
                        rays_cam, _ = sample_camera_rays(HWK, camera.R, camera.T)
                    elif mode == 'native':
                        # Use original camera intrinsics as-is and call sample_camera_rays.
                        HWK = _get_HWK_from_camera(camera)
                        rays_cam, _ = sample_camera_rays(HWK, camera.R, camera.T)
                    else:
                        raise ValueError(f"Unknown env_view_mode: {mode}")

                    env_view = gaussians.env_map(rays_cam, mode="pure_env")  # HxWx3
                    env_view = env_view.permute(2, 0, 1).contiguous()
                    save_image(linear_to_srgb(env_view), os.path.join(env_path, f"envview_idx{args.idx}_yaw{yaw:03d}_pitch{pitch:03d}.png"))
            except Exception as e:
                print(f"[warn] envview export failed for yaw={yaw}, pitch={pitch}: {e}")
            
            # NOTE: The code below is for visualizing the gridmap envmap(GridMap)
            # for index, o_env in enumerate(gaussians.gridmap_envmap):
            #     output_root = os.path.join(args.output_dir, f"gridmap_envmaps_idx{args.idx}_yaw{yaw}_pitch{pitch}")
            #     os.makedirs(output_root, exist_ok=True)
            #     o_env_dict = gaussians.render_gridmap_envmap(index)
            #     diff_o_env_dict = gaussians.render_gridmap_envmap(index,mode="diffuse")
            #     grid = [
            #         o_env_dict["env1"].permute(2, 0, 1),
            #         # o_env_dict["env2"].permute(2, 0, 1),
            #     ]
            #     grid = make_grid(grid, nrow=1, padding=10)
            #     save_image(linear_to_srgb(grid), os.path.join(output_root, f"gridmap_env_{index}.png"))
            #     diff_grid = [
            #         diff_o_env_dict["env1"].permute(2, 0, 1),
            #         # diff_o_env_dict["env2"].permute(2, 0, 1),
            #     ]
            #     diff_grid = make_grid(diff_grid, nrow=1, padding=10)
            #     save_image(linear_to_srgb(diff_grid), os.path.join(output_root, f"gridmap_env_{index}_diffuse.png"))
            #     os.makedirs(os.path.join(output_root, f"gridmap_env_{index}"), exist_ok=True)
            #     for j in range(6):
            #         g_env = gaussians.env_map.base[j]
            #         o_env_fig = linear_to_srgb(torch.where(o_env.base[j].permute(2,0,1) > 0, torch.relu(o_env.base[j].permute(2,0,1)+0.5), torch.sigmoid(o_env.base[j].permute(2,0,1))))
            #         save_image(o_env_fig, os.path.join(output_root, f"gridmap_env_{index}/face{j}.png"))
            #         g_env_fig = linear_to_srgb(torch.where(g_env.permute(2,0,1) > 0, torch.relu(g_env.permute(2,0,1)+0.5), torch.sigmoid(g_env.permute(2,0,1))))
            #         save_image(torch.abs(g_env_fig-o_env_fig), os.path.join(output_root, f"gridmap_env_{index}/delta_face{j}.png"))
            #         if index == 0:
            #             save_image(g_env_fig, os.path.join(args.output_dir, f"global_env_face{j}.png"))
            #             g_grid = gaussians.render_env_map()["env1"].permute(2,0,1)
            #             save_image(linear_to_srgb(g_grid), os.path.join(args.output_dir, f"global_env.png"))
            #             g_grid_diffuse = gaussians.render_env_map_diffuse()["env1"].permute(2,0,1)
            #             save_image(linear_to_srgb(g_grid_diffuse), os.path.join(args.output_dir, f"global_env_diffuse.png"))
            #             # import pdb; pdb.set_trace()
            # save_image(linear_to_srgb(diff_grid), os.path.join(args.output_dir, f"env_diffuse.png"))

            print(f"Saved: {f_name}")

if __name__ == "__main__":
    main()
