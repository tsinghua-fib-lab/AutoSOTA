#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import open3d as o3d
from random import randint
from utils.loss_utils import calculate_loss, l1_loss
from gaussian_renderer import render_surfel, render_initial, render_volume, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.stokes_utils import calc_aolp_dop
import numpy as np
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from datetime import datetime
import time
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
import matplotlib.cm as cm
from utils.image_utils import visualize_depth
from utils.graphics_utils import linear_to_srgb
from utils.mesh_utils import GaussianExtractor, post_process_mesh
from scene.linear_polarizer import LinearPolarizer
from utils.gridmap_envmap_utils import  build_cubemap,  build_surf_cubemap, build_mesh_cubemap
from utils.time_logger import append_time_log
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False






def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations,\
             checkpoint, model_path, debug_from=None, fix_env_map =None):
    first_iter = 0
    tb_writer = prepare_output_and_logger()

    # Set up parameters 
    TOT_ITER = opt.iterations + 1
    TEST_INTERVAL = 1000
    MESH_EXTRACT_INTERVAL = 1000

    # For real scenes
    USE_ENV_SCOPE = opt.use_env_scope  # False
    if USE_ENV_SCOPE:
        center = [float(c) for c in opt.env_scope_center]
        ENV_CENTER = torch.tensor(center, device='cuda')
        ENV_RADIUS = opt.env_scope_radius
        REFL_MSK_LOSS_W = 0.4

    if  opt.lambda_stokes <=0:
        opt.use_stokes = False
    gaussians = GaussianModel(dataset.albedo_sh_degree, dataset.sh_degree)
    set_gaussian_para(gaussians, opt, vol=(opt.volume_render_until_iter > opt.init_until_iter)) # #
    LP = None
    
    if opt.use_stokes and opt.use_LP:
        LP = []
        LP.append(LinearPolarizer(init_value=torch.deg2rad(torch.tensor(45.0)),opt = opt)) 
        LP.append(LinearPolarizer(init_value=torch.deg2rad(torch.tensor(135.0)),opt = opt))
    scene = Scene(dataset, gaussians,LP = LP)  # init all parameters(pos, scale, rot...) from pcds
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        gaussExtractor = GaussianExtractor(gaussians, render_initial, pipe, bg_color=bg_color) 
        gaussExtractor.reconstruction(scene.getTrainCameras())
        if dataset.real_dataset == True:
            mesh = gaussExtractor.extract_mesh_unbounded(resolution=opt.mesh_res)
        else:
            depth_trunc = (gaussExtractor.radius * 2.0) if opt.depth_trunc < 0  else opt.depth_trunc
            voxel_size = (depth_trunc / opt.mesh_res) if opt.voxel_size < 0 else opt.voxel_size
            sdf_trunc = 5.0 * voxel_size if opt.sdf_trunc < 0 else opt.sdf_trunc
            mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
        mesh = post_process_mesh(mesh, cluster_to_keep=opt.num_cluster)
        # import pdb;pdb.set_trace()
        gaussians.update_mesh(mesh)

    # import pdb;pdb.set_trace()
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    gaussExtractor = GaussianExtractor(gaussians, render_initial, pipe, bg_color=bg_color) 
    # linearPolarizer = LinearPolarizer(init_value=torch.deg2rad(torch.tensor(0.0)),opt = opt) if opt.use_stokes else None
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_normal_smooth_for_log = 0.0
    ema_depth_smooth_for_log = 0.0
    ema_psnr_for_log = 0.0
    psnr_test = 0

    progress_bar = tqdm(range(first_iter, TOT_ITER), desc="Training progress")
    first_iter += 1
    iteration = first_iter

    print(f'Propagation until: {opt.normal_prop_until_iter }')
    print(f'Densify until: {opt.densify_until_iter}')
    print(f'Total iterations: {TOT_ITER}')

    timing_interval = 3000
    initial_iteration = iteration
    run_start_time = time.time()
    chunk_start_time = run_start_time
    chunk_start_iter = iteration
    next_timing_iter = ((max(iteration, 1) - 1) // timing_interval + 1) * timing_interval


    initial_stage = opt.initial
    if not initial_stage:
        opt.init_until_iter = 0

    
    # Training loop
    build_cubemap_duration = None
    mesh_duration = None
    while iteration < TOT_ITER:
        iter_start.record()
        train_start_time = time.time()
        
        gaussians.update_learning_rate(iteration)


        # Increase SH levels every 1000 iterations
        if iteration > opt.feature_rest_from_iter and iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Control the init stage
        if iteration > opt.init_until_iter:
            initial_stage = False
        
        # Control the indirect stage
        if iteration >= opt.indirect_from_iter + 1:
            opt.indirect = 1



        if iteration == (opt.volume_render_until_iter + 1) and opt.volume_render_until_iter > opt.init_until_iter:
            reset_gaussian_para(gaussians, opt)


        # Initialize envmap
        if not initial_stage:
            if iteration <= opt.volume_render_until_iter:
                envmap2 = gaussians.get_envmap_2 
                envmap2.build_mips()
            else:
                envmap = gaussians.get_envmap 
                envmap.build_mips()
            if fix_env_map is not None:
                gaussians.fix_env_map(fix_env_map, dataset)
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        linearPolarizer = viewpoint_cam.LP if viewpoint_cam.LP else None

        # Set render
        render = select_render_method(iteration, opt, initial_stage)
        render_start = time.time()
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, opt=opt)
        render_time = time.time() - render_start

        use_stokes = opt.use_stokes
        if render == render_surfel:
            stokes = {}
            stokes["spec"] = render_pkg["stokes_spec"]
            stokes["diff"] = render_pkg["stokes_diff"]
            stokes["combine"] = render_pkg["stokes_combined"]
        else:
            use_stokes=False
            stokes= None
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        gt_image = viewpoint_cam.original_stokes[...,0].cuda() if viewpoint_cam.original_stokes is not None else viewpoint_cam.original_image
        gt_stokes = viewpoint_cam.original_stokes.cuda() if viewpoint_cam.original_stokes is not None else None

        total_loss, tb_dict = calculate_loss(viewpoint_cam, gaussians, render_pkg, opt, iteration)
        dist_loss, normal_loss, loss, Ll1, normal_smooth_loss, depth_smooth_loss,stokes_loss = \
        tb_dict["loss_dist"], tb_dict["loss_normal_render_depth"], tb_dict["loss0"], tb_dict["loss_l1"], tb_dict["loss_normal_smooth"], tb_dict["loss_depth_smooth"],tb_dict["loss_stokes"]

        def get_outside_msk():
            return None if not USE_ENV_SCOPE else torch.sum((gaussians.get_xyz - ENV_CENTER[None])**2, dim=-1) > ENV_RADIUS**2
        
        if USE_ENV_SCOPE and 'refl_strength_map' in render_pkg:
            refls = gaussians.get_refl
            refl_msk_loss = refls[get_outside_msk()].mean()
            total_loss += REFL_MSK_LOSS_W * refl_msk_loss
        
        total_loss.backward()

        iter_end.record()


        with torch.no_grad():
            
            if iteration % TEST_INTERVAL == 0 or iteration == first_iter + 1 or iteration == opt.volume_render_until_iter + 1:
                
                save_training_vis(viewpoint_cam, gaussians, background, render, pipe, opt, iteration, initial_stage,use_stokes=use_stokes,stokes=stokes,gt_stokes=gt_stokes)

            ema_loss_for_log = 0.4 * loss + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss + 0.6 * ema_normal_for_log
            ema_normal_smooth_for_log = 0.4 * normal_smooth_loss + 0.6 * ema_normal_smooth_for_log
            ema_depth_smooth_for_log = 0.4 * depth_smooth_loss + 0.6 * ema_depth_smooth_for_log
            ema_psnr_for_log = 0.4 * psnr(image, gt_image).mean().double().item() + 0.6 * ema_psnr_for_log
            if iteration % TEST_INTERVAL == 0:
                psnr_test = evaluate_psnr(scene, render, {"pipe": pipe, "bg_color": background, "opt": opt})
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "Distort": f"{ema_dist_for_log:.{5}f}",
                    "Normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}",
                    "PSNR-train": f"{ema_psnr_for_log:.{4}f}",
                    "PSNR-test": f"{psnr_test:.{4}f}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration % 100 == 0:
                if LP is not None:
                    print(f"[ITER {iteration}],Linear Polarizer phi: {torch.rad2deg(LP[0].phi).item():.2f},{torch.rad2deg(LP[1].phi).item():.2f}")
            if iteration == TOT_ITER:
                progress_bar.close()

            if tb_writer:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render, {"pipe": pipe, "bg_color": background, "opt":opt})

            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter and iteration != opt.volume_render_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration <= opt.init_until_iter:
                    opacity_reset_intval = 3000
                    densification_interval = 100
                elif iteration <= opt.normal_prop_until_iter :
                    opacity_reset_intval = 3000
                    densification_interval = opt.densification_interval_when_prop
                else:
                    opacity_reset_intval = 3000
                    densification_interval = 100

                if iteration > opt.densify_from_iter and iteration % densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.prune_opacity_threshold, scene.cameras_extent,
                                                size_threshold)

                HAS_RESET0 = False
                if (iteration % opacity_reset_intval == 0 or (dataset.white_background and iteration == opt.densify_from_iter)) \
                    and (iteration not in checkpoint_iterations):
                    HAS_RESET0 = True
                    outside_msk = get_outside_msk()
                    gaussians.reset_opacity0()
                    gaussians.reset_refl(exclusive_msk=outside_msk)
                if opt.opac_lr0_interval > 0 and (
                        opt.init_until_iter < iteration <= opt.normal_prop_until_iter ) and iteration % opt.opac_lr0_interval == 0:
                    gaussians.set_opacity_lr(opt.opacity_lr)
                if (opt.init_until_iter < iteration <= opt.normal_prop_until_iter ) and iteration % opt.normal_prop_interval == 0:
                    if not HAS_RESET0:
                        outside_msk = get_outside_msk()
                        gaussians.reset_opacity1(exclusive_msk=outside_msk)
                        if iteration > opt.volume_render_until_iter and opt.volume_render_until_iter > opt.init_until_iter:
                            gaussians.dist_color(exclusive_msk=outside_msk)
                            # gaussians.dist_albedo(exclusive_msk=outside_msk)

                        gaussians.reset_scale(exclusive_msk=outside_msk)
                        if opt.opac_lr0_interval > 0 and iteration != opt.normal_prop_until_iter :
                            gaussians.set_opacity_lr(0.0)
                
            if (iteration > opt.indirect_from_iter and iteration % MESH_EXTRACT_INTERVAL == 0) \
                    or iteration == (opt.indirect_from_iter-1):
                if not HAS_RESET0:
                    mesh_start = time.time()
                    gaussExtractor.reconstruction(scene.getTrainCameras())
                    if dataset.real_dataset == True:
                        mesh = gaussExtractor.extract_mesh_unbounded(resolution=opt.mesh_res)
                    else:
                        depth_trunc = (gaussExtractor.radius * 2.0) if opt.depth_trunc < 0  else opt.depth_trunc
                        voxel_size = (depth_trunc / opt.mesh_res) if opt.voxel_size < 0 else opt.voxel_size
                        sdf_trunc = 5.0 * voxel_size if opt.sdf_trunc < 0 else opt.sdf_trunc
                        mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
                    mesh = post_process_mesh(mesh, cluster_to_keep=opt.num_cluster)
                    ply_path = os.path.join(model_path,f'test_{iteration:06d}.ply')
                    o3d.io.write_triangle_mesh(ply_path, mesh)
                    # import pdb;pdb.set_trace()
                    gaussians.update_mesh(mesh)
                    mesh_duration = time.time() - mesh_start

                    if (pipe.indirect_type in ['mesh_env']) and iteration >= opt.subenv_from_iter :
                        gridmap_cubemaps = build_mesh_cubemap(mesh, gaussians, pipe, background, dataset, opt)
                        gaussians.update_gridmap_envmap(gridmap_cubemaps,dataset)
            
            if (iteration >= opt.indirect_from_iter)\
                and (iteration % opt.indirect_gridmap_env_interval == 0 or iteration % MESH_EXTRACT_INTERVAL == 0)  and (not initial_stage) and (pipe.indirect_type in ['obj_env','surf_env']):
                bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
                background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
                cameras = scene.getTrainCameras().copy() + scene.getTestCameras().copy()
                gaussians.reset_gridmap_envmap()

                build_start = time.time()
                if pipe.indirect_type == 'obj_env':
                    gridmap_cubemaps = build_cubemap(cameras, gaussians, pipe, background, dataset, opt)
                elif pipe.indirect_type == 'surf_env':
                    gridmap_cubemaps = build_surf_cubemap(cameras, gaussians, pipe, background, dataset, opt)
                build_cubemap_duration = time.time() - build_start
                gaussians.update_gridmap_envmap(gridmap_cubemaps,dataset)

            if iteration < TOT_ITER:
                gaussians.optimizer.step()
                if use_stokes and linearPolarizer is not None:
                    linearPolarizer.optimizer.step()
                    linearPolarizer.scheduler.step()
                    if linearPolarizer.phi > np.pi or linearPolarizer.phi < 0:
                        linearPolarizer.set_phi(torch.remainder(linearPolarizer.phi, torch.pi))
                gaussians.optimizer.zero_grad(set_to_none=True)
                
                    

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration), scene.model_path + f"/chkpnt{iteration}.pth")
                log_timing("checkpoint", iteration,time.time()-train_start_time, render_time, build_cubemap_duration)
            
            elif iteration % 4998 == 0:
                log_timing("train_iter", iteration,time.time()-train_start_time, render_time, build_cubemap_duration,mesh_duration)
            

        should_log_chunk = (iteration >= next_timing_iter) or (iteration == TOT_ITER - 1)
        if should_log_chunk:
            chunk_elapsed = time.time() - chunk_start_time
            chunk_start_time = time.time()
            chunk_start_iter = iteration + 1
            next_timing_iter += timing_interval

        iteration += 1

    total_duration = time.time() - run_start_time
    last_iteration = min(iteration - 1, TOT_ITER - 1)






# ============================================================
# Utils for training

def log_timing(event, iteration, training_elapsed=None, render_time=None, build_cubemap_time=None, mesh_duration=None):
    
    if not hasattr(args, "visualize_log_path"):
        log_time_path = os.path.join(args.model_path, "visualize_log.csv")
    else:
        log_time_path = args.visualize_log_path
    row = [
        event,
        str(iteration),
        "" if training_elapsed is None else f"{training_elapsed:.6f}",
        "" if render_time is None else f"{render_time:.6f}",
        "" if build_cubemap_time is None else f"{build_cubemap_time:.6f}",
        "" if mesh_duration is None else f"{mesh_duration:.6f}",
    ]
    with open(log_time_path, "a") as log_file:
        log_file.write(",".join(row) + "\n")


def select_render_method(iteration, opt, initial_stage):

    if initial_stage:
        render = render_initial
    elif iteration <= opt.volume_render_until_iter:
        render = render_volume
    else:   
        render = render_surfel

    return render


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




def save_training_vis(viewpoint_cam, gaussians, background, render_fn, pipe, opt, iteration, initial_stage,\
                        use_stokes=False, stokes=None, gt_stokes=None):
    with torch.no_grad():
        render_pkg = render_fn(viewpoint_cam, gaussians, pipe, background, opt=opt)
        # import pdb;pdb.set_trace()
        gt_image = viewpoint_cam.original_stokes[...,0].cuda() if viewpoint_cam.original_stokes is not None else viewpoint_cam.original_image.cuda()
        error_map = torch.abs(gt_image.cuda() - render_pkg["render"])

        if initial_stage:
            visualization_list = [
                gt_image.cuda(),
                render_pkg["render"].clamp(0,1), 
                render_pkg["rend_alpha"].repeat(3, 1, 1),
                visualize_depth(render_pkg["surf_depth"]),  
                render_pkg["rend_normal"] * 0.5 + 0.5, 
                render_pkg["surf_normal"] * 0.5 + 0.5, 
                error_map 
            ]

        elif iteration <= opt.volume_render_until_iter:
            visualization_list = [
                gt_image.cuda(),  
                render_pkg["render"].clamp(0,1), 
                render_pkg["base_color_map"].clamp(0,1), 
                render_pkg["diffuse_map"].clamp(0,1),      
                render_pkg["specular_map"].clamp(0,1),  
                render_pkg["refl_strength_map"].repeat(3, 1, 1),  
                render_pkg["roughness_map"].repeat(3, 1, 1),
                render_pkg["rend_alpha"].repeat(3, 1, 1),  
                visualize_depth(render_pkg["surf_depth"]), 
                render_pkg["rend_normal"] * 0.5 + 0.5,  
                render_pkg["surf_normal"] * 0.5 + 0.5, 
                error_map
            ]
            if opt.indirect:
                visualization_list += [
                    render_pkg["visibility"].repeat(3, 1, 1),
                    render_pkg["direct_light"],
                    render_pkg["indirect_light"],
                ]

        else:
            visualization_list = [
                gt_image.cuda(),  
                render_pkg["render"].clamp(0,1),  
                render_pkg["base_color_map"].clamp(0,1),  
                render_pkg["diffuse_map"].clamp(0,1),
                render_pkg["specular_map"].clamp(0,1),
                render_pkg["half_eta_map"],  
                render_pkg["roughness_map"].repeat(3, 1, 1),
                render_pkg["rend_alpha"].repeat(3, 1, 1),  
                visualize_depth(render_pkg["surf_depth"]),  
                render_pkg["rend_normal"] * 0.5 + 0.5,  
                render_pkg["surf_normal"] * 0.5 + 0.5,  
                error_map, 
            ]
            if opt.indirect:
                visualization_list += [
                    render_pkg["visibility"].repeat(3, 1, 1),
                    render_pkg["direct_light"].clamp(0,1),
                    render_pkg["indirect_light"].clamp(0,1),
                    render_pkg["specular_weight"].clamp(0,1),
                ]
            else:
                visualization_list += [render_pkg["direct_light"].clamp(0,1),render_pkg["specular_weight"].clamp(0,1)]

        for i in range(len(visualization_list)):
            if visualization_list[i].shape[0] == 3 and i<6:
                visualization_list[i] = linear_to_srgb(visualization_list[i])
        grid = torch.stack(visualization_list, dim=0)
        grid = make_grid(grid, nrow=4)
        scale = grid.shape[-2] / 800
        grid = F.interpolate(grid[None], (int(grid.shape[-2] / scale), int(grid.shape[-1] / scale)))[0]
        save_image(grid, os.path.join(args.visualize_path, f"{iteration:06d}.png"))

        if not initial_stage:
            if opt.volume_render_until_iter > opt.init_until_iter and iteration <= opt.volume_render_until_iter:
                env_dict = gaussians.render_env_map_2() 
                diffuse_env_dict = gaussians.render_env_map_2_diffuse()
            else:
                env_dict = gaussians.render_env_map()
                diffuse_env_dict = gaussians.render_env_map_diffuse()

            grid = [
                env_dict["env1"].permute(2, 0, 1),
                env_dict["env2"].permute(2, 0, 1),
            ]
            grid = make_grid(grid, nrow=1, padding=10)
            save_image(grid, os.path.join(args.visualize_path, f"{iteration:06d}_env.png"))

            diffuse_grid = [
                diffuse_env_dict["env1"].permute(2, 0, 1),
                diffuse_env_dict["env2"].permute(2, 0, 1),
            ]
            diffuse_grid = make_grid(diffuse_grid, nrow=1, padding=10)
            save_image(diffuse_grid, os.path.join(args.visualize_path, f"{iteration:06d}_env_diffuse.png"))

            # 可视化 gridmap_envmap（仅前三张）
            # Save gridmap_envmap images to args.visualize_path/gridmap_envs/$iteration folder
            gridmap_envs_dir = os.path.join(args.visualize_path, "gridmap_envs", f"{iteration}")
            os.makedirs(gridmap_envs_dir, exist_ok=True)
            if hasattr(gaussians, 'gridmap_envmap') and gaussians.gridmap_envmap is not None:
                for obj_idx, gridmap_env in enumerate(gaussians.gridmap_envmap[:4]):
                    gridmap_env_dict = gaussians.render_gridmap_envmap(obj_idx)
                    diff_gridmap_env_dict = gaussians.render_gridmap_envmap(obj_idx, mode="diffuse")
                    obj_grid = [
                        gridmap_env_dict["env1"].permute(2, 0, 1),
                        gridmap_env_dict["env2"].permute(2, 0, 1),
                    ]
                    obj_grid = make_grid(obj_grid, nrow=1, padding=10)
                    save_image(obj_grid, os.path.join(gridmap_envs_dir, f"gridmap_env_{obj_idx}.png"))
                    diff_obj_grid = [
                        diff_gridmap_env_dict["env1"].permute(2, 0, 1),
                        diff_gridmap_env_dict["env2"].permute(2, 0, 1),
                    ]
                    diff_obj_grid = make_grid(diff_obj_grid, nrow=1, padding=10)
                    save_image(diff_obj_grid, os.path.join(gridmap_envs_dir, f"gridmap_env_{obj_idx}_diffuse.png"))
                # visualization_list_stokes = []
        if use_stokes:
            def color_and_norm_dop_aolp(aolp,dop):
                vmin, vmax = 0, 180
                aolp_norm = (aolp - vmin) / (vmax - vmin)
                # aolp, dop are H, W, 3; take mean over channel
                aolp_norm = aolp_norm.mean(dim=-1).clamp(0, 1)
                dop = dop.mean(dim=-1)
                cmap = cm.get_cmap('twilight')
                aolp_colored = cmap(aolp_norm.cpu().numpy())[..., :3]
                cmap = cm.get_cmap('viridis')
                dop_colored = cmap(dop.cpu().numpy())[..., :3]
                aolp_colored = torch.from_numpy(aolp_colored).permute(2, 0, 1).float()
                dop_colored = torch.from_numpy(dop_colored).permute(2, 0, 1).float()
                return aolp_colored, dop_colored
            
            def calculate_delta_and_norm(s,gt_s):
                delta = torch.abs(s - gt_s)
                norm = torch.max(torch.abs(gt_s))
                if norm > 0:
                    delta = delta / norm
                return delta
                
            stokes_viz_path = os.path.join(args.visualize_path, "stokes")
            if not os.path.exists(stokes_viz_path):
                os.makedirs(stokes_viz_path)
            stokes_viz_list = []
            if gt_stokes is not None:
                gt_aolp, gt_dop = calc_aolp_dop(gt_stokes)
                gt_aolp, gt_dop = gt_aolp.permute(1,2,0) , gt_dop.permute(1,2,0) 
                gt_aolp_colored, gt_dop_colored = color_and_norm_dop_aolp(gt_aolp, gt_dop)
                gt_s0 = gt_stokes[...,0].cpu()
                gt_s1 = gt_stokes[...,1].cpu()
                gt_s2 = gt_stokes[...,2].cpu()
                stokes_viz_list  += [
                    gt_s0, gt_s1, gt_s2, 
                    gt_aolp_colored, gt_dop_colored,
                ]
            for stokes_part in ["combine","diff","spec"]:
            # Calculate the angle of linear polarization (AOLP) and degree of polarization (DOP)
                aolp, dop = calc_aolp_dop(stokes[stokes_part])
                aolp, dop = aolp.permute(1,2,0) , dop.permute(1,2,0) 
                # import pdb;pdb.set_trace()
                aolp_colored, dop_colored = color_and_norm_dop_aolp(aolp, dop)
                # import pdb;pdb.set_trace()
                stokes_viz_list += [
                    stokes[stokes_part][...,0].clamp(0,1).cpu(),
                    stokes[stokes_part][...,1].clamp(0,1).cpu(),
                    stokes[stokes_part][...,2].clamp(0,1).cpu(),
                    aolp_colored,
                    dop_colored
                ]
            if opt.use_LP:
                aolp_linear, dop_linear = calc_aolp_dop(render_pkg["linear_stokes"])
                aolp_linear, dop_linear = aolp_linear.permute(1,2,0) , dop_linear.permute(1,2,0) 
                aolp_linear_colored, dop_linear_colored = color_and_norm_dop_aolp(aolp_linear, dop_linear)
                stokes_viz_list += [render_pkg["linear_stokes"][...,0].cpu().clamp(0,1),
                                    render_pkg["linear_stokes"][...,1].cpu().clamp(0,1),
                                    render_pkg["linear_stokes"][...,2].cpu().clamp(0,1),
                                    aolp_linear_colored, dop_linear_colored,viewpoint_cam.original_image.cpu()
                                    ]
            for i in range(len(stokes_viz_list)):
                if i % 5 <= 2: # s0, s1, s2
                    stokes_viz_list[i] = linear_to_srgb(stokes_viz_list[i])
            stokes_grid = torch.stack(stokes_viz_list, dim=0)
            stokes_grid = make_grid(stokes_grid, nrow=5)
            scale = stokes_grid.shape[-2] / 1600
            stokes_grid = F.interpolate(stokes_grid[None], (int(stokes_grid.shape[-2] / scale), int(stokes_grid.shape[-1] / scale)))[0]
            save_image(stokes_grid, os.path.join(stokes_viz_path, f"{iteration:06d}.png"))

      
NORM_CONDITION_OUTSIDE = False
def prepare_output_and_logger():    
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    args.visualize_path = os.path.join(args.model_path, "visualize")
    
    os.makedirs(args.visualize_path, exist_ok=True)
    print("Visualization folder: {}".format(args.visualize_path))
    
    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderkwargs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1, iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss, iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(tqdm(config['cameras'])):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, **renderkwargs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = viewpoint.original_stokes[...,0].cuda() if viewpoint.original_stokes is not None else viewpoint.original_image
                    gt_image = torch.clamp(gt_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

@torch.no_grad()
def evaluate_psnr(scene, renderFunc, renderkwargs):
    psnr_test = 0.0
    torch.cuda.empty_cache()
    if len(scene.getTestCameras()):
        for viewpoint in scene.getTestCameras():
            render_pkg = renderFunc(viewpoint, scene.gaussians, **renderkwargs)
            image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            gt_image = viewpoint.original_stokes[...,0].cuda() if viewpoint.original_stokes is not None else viewpoint.original_image
            gt_image = torch.clamp(gt_image.to("cuda"), 0.0, 1.0)
            psnr_test += psnr(image, gt_image).mean().double()

        psnr_test /= len(scene.getTestCameras())
        
    torch.cuda.empty_cache()
    return psnr_test





# ============================================================================
# Main function


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10000,12000,15000,20000,30000,40000,50000,60000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[10000,15000,20000,30000,40000,50000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--fix_env_map",type=str,default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.test_iterations = args.test_iterations + [i for i in range(10000, args.iterations+1, 5000)]
    args.test_iterations.append(args.volume_render_until_iter)

    
    if not args.model_path:
        # get timestamp
        current_time = datetime.now().strftime('%m%d_%H%M')
        # get the last subfolder name of args.source_path
        last_subdir = os.path.basename(os.path.normpath(args.source_path))
        # generate model path with last subfolder name and timestamp
        args.model_path = os.path.join(
            "./output/", f"{last_subdir}/",
            f"{last_subdir}-{current_time}-st-{args.lambda_stokes}-ind-{args.indirect_type}"
        )

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations,\
     args.start_checkpoint, args.model_path,fix_env_map = args.fix_env_map)

    # All done
    print("\nTraining complete.")