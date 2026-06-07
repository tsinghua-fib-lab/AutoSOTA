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


import torch
import torch.nn.functional as F
import math
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.point_utils import depth_to_normal
from utils.refl_utils import  get_full_color_volume, get_full_color_volume_indirect,get_full_color_surfel_direct,get_full_color_surfel_indirect
from scene.linear_polarizer import LinearPolarizer
import numpy as np



def compute_2dgs_normal_and_regularizations(allmap, viewpoint_camera, pipe):
    # 2DGS normal and regularizations
    # additional regularizations
    render_alpha = allmap[1:2]
    
    # get normal map
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    
    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)
    
    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    
    # get depth distortion map
    render_dist = allmap[6:7]
    
    # pseudo surface attributes
    surf_depth = render_depth_expected * (1 - pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
    
    # assume the depth points form the 'surface' and generate pseudo surface normal for regularizations.
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * render_alpha.detach()
    
    return {
        'render_alpha': render_alpha,
        'render_normal': render_normal,
        'render_depth_median': render_depth_median,
        'render_depth_expected': render_depth_expected,
        'render_dist': render_dist,
        'surf_depth': surf_depth,
        'surf_normal': surf_normal,
        'orig_render_normal': allmap[2:5]
    }



def render_initial(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, 
            scaling_modifier = 1.0, override_color = None, opt=None):

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    imH = int(viewpoint_camera.image_height)
    imW = int(viewpoint_camera.image_width)

    raster_settings = GaussianRasterizationSettings(
        image_height=imH,
        image_width=imW,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg = torch.zeros_like(bg_color),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # currently don't support normal consistency loss if use precomputed covariance
        splat2world = pc.get_covariance(scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, far-near, near],
            [0, 0, 0, 1]]).float().cuda().T
        world2pix =  viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9) # column major
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    pipe.convert_SHs_python = False
    shs = None
    colors_precomp = None


    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
        
    contrib, rendered_image, rendered_features, radii, allmap = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp
    )

    regularizations = compute_2dgs_normal_and_regularizations(allmap, viewpoint_camera, pipe)
    render_alpha = regularizations['render_alpha']
    render_normal = regularizations['render_normal']
    render_depth_median = regularizations['render_depth_median']
    render_depth_expected = regularizations['render_depth_expected']
    render_dist = regularizations['render_dist']
    surf_depth = regularizations['surf_depth']
    surf_normal = regularizations['surf_normal']

    final_image = rendered_image + bg_color[:, None, None] * (1 - render_alpha)

    rets =  {"render": final_image,
        "viewspace_points": means2D,
        "visibility_filter" : radii > 0,
        "radii": radii,
        'rend_alpha': render_alpha,
        'rend_normal': render_normal,
        'rend_dist': render_dist,
        'surf_depth': surf_depth,
        'surf_normal': surf_normal,
        'orig_normal': regularizations['orig_render_normal']
    }

    return rets




def render_surfel(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, opt=None):

 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    imH = int(viewpoint_camera.image_height)
    imW = int(viewpoint_camera.image_width)

    raster_settings = GaussianRasterizationSettings(
        image_height=imH,
        image_width=imW,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg = torch.zeros_like(bg_color),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    ## reflection strength 定义（即refl ratio）
    refl = pc.get_refl
    ori_color = pc.get_ori_color
    roughness = pc.get_rough
    half_eta = pc.get_half_eta

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # currently don't support normal consistency loss if use precomputed covariance
        splat2world = pc.get_covariance(scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, far-near, near],
            [0, 0, 0, 1]]).float().cuda().T
        world2pix =  viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9) # column major
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    pipe.convert_SHs_python = True
    shs = None
    colors_precomp = None


    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            # sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            # colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            colors_precomp = torch.sigmoid(shs_view[..., 0])
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # indirect light sh
    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center)
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    normals = pc.get_normal(scaling_modifier, dir_pp_normalized)
    w_o = -dir_pp_normalized
    reflection = 2 * torch.sum(normals * w_o, dim=1, keepdim=True) * normals - w_o
    shs_indirect = pc.get_indirect.transpose(1, 2).view(-1, 3, (pc.indirect_sh_degree+1)**2)
    sh2indirect = eval_sh(3, shs_indirect, reflection)
    indirect = torch.clamp_min(sh2indirect, 0.0)
    sh2indirect_diffuse = eval_sh(0, shs_indirect, normals)
    indirect_diffuse = torch.clamp_min(sh2indirect_diffuse, 0.0)
    
    

    contrib, rendered_image, rendered_features, radii, allmap = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        features = torch.cat((half_eta, roughness, ori_color, indirect,refl, indirect_diffuse), dim=-1),
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
    )


    albedo = rendered_image
    
    half_eta = rendered_features[0:1].repeat(3,1,1)
    roughness = rendered_features[1:2]
    # albedo = rendered_features[4:7]
    indirect_light = rendered_features[5:8]
    refl_strength = rendered_features[8:9]
    indirect_light_diffuse = rendered_features[9:12]


    # 2DGS normal and regularizations
    regularizations = compute_2dgs_normal_and_regularizations(allmap, viewpoint_camera, pipe)
    render_alpha = regularizations['render_alpha']
    render_normal = regularizations['render_normal']
    render_dist = regularizations['render_dist']
    surf_depth = regularizations['surf_depth']
    surf_normal = regularizations['surf_normal']

    # import pdb;pdb.set_trace()
    # Use normal map computed in 2DGS pipeline to perform reflection query
    normal_map = render_normal.permute(1,2,0)
    normal_map = normal_map / render_alpha.permute(1,2,0).clamp_min(1e-6)
    # import pdb;pdb.set_trace()
    LP = viewpoint_camera.LP
    if opt.indirect:
        specular, diffuse, extra_dict = get_full_color_surfel_indirect(pc.get_envmap, albedo.permute(1,2,0), viewpoint_camera.HWK, 
                                            viewpoint_camera.R, viewpoint_camera.T, normal_map, render_alpha.permute(1,2,0), 
                                            half_eta = half_eta.permute(1,2,0), refl_strength=refl_strength.permute(1,2,0), roughness=roughness.permute(1,2,0), 
                                            pc=pc, surf_depth=surf_depth, indirect_light_sh=indirect_light.permute(1,2,0),
                                            use_stokes=opt.use_stokes,LP=LP,indirect_diffuse_sh=indirect_light_diffuse.permute(1,2,0),
                                            indirect_type=pipe.indirect_type)
    else:
        specular, diffuse, extra_dict = get_full_color_surfel_direct(pc.get_envmap, albedo.permute(1,2,0), viewpoint_camera.HWK, 
                                            viewpoint_camera.R, viewpoint_camera.T, normal_map, render_alpha.permute(1,2,0), 
                                            half_eta = half_eta.permute(1,2,0), refl_strength=refl_strength.permute(1,2,0), roughness=roughness.permute(1,2,0), 
                                            pc=pc, surf_depth=surf_depth, use_stokes=opt.use_stokes, LP=LP)
        
    
    if opt.use_stokes:
        stokes_spec = torch.stack([specular[0,:]* extra_dict["stokes_spec_fac"][0].permute(2,0,1),# h,w,s -> s,h,w
                                   specular[1,:]* extra_dict["stokes_spec_fac"][1].permute(2,0,1),
                                   specular[2,:]* extra_dict["stokes_spec_fac"][2].permute(2,0,1)]).permute(0,2,3,1) #C,S,H,W-> C,H,W,S
        stokes_diff = torch.stack([diffuse[0,:]* extra_dict["stokes_diff_fac"][0].permute(2,0,1),
                                   diffuse[1,:]* extra_dict["stokes_diff_fac"][1].permute(2,0,1),
                                   diffuse[2,:]* extra_dict["stokes_diff_fac"][2].permute(2,0,1)]).permute(0,2,3,1)
        stokes_combined = stokes_spec + stokes_diff
        diffuse = stokes_diff[...,0]
        specular = stokes_spec[...,0]
    else:
        stokes_spec = None
        stokes_diff = None
        stokes_combined = None

    # Integrate the final image
    final_image = diffuse + specular 
    


    final_image = final_image + bg_color[:, None, None] * (1 - render_alpha)
    if opt.indirect:
        indirect_color = albedo + extra_dict['indirect_color']
        indirect_color = indirect_color + bg_color[:, None, None] * (1 - render_alpha)
        extra_dict['indirect_color'] = indirect_color.clamp(min=0.0, max=1.0)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    results =  {"render": final_image,
            "refl_strength_map": refl_strength,
            "half_eta_map": half_eta,
            "diffuse_map": diffuse,
            "specular_map": specular,
            "base_color_map": albedo,
            "roughness_map": roughness,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "stokes_spec": stokes_spec,
            "stokes_diff": stokes_diff,
            "stokes_combined": stokes_combined,
            ## normal, accum alpha, dist, depth map
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_dist': render_dist,
            'surf_depth': surf_depth,
            'surf_normal': surf_normal,
            'orig_normal': regularizations['orig_render_normal'],
            "linear_stokes": LP(stokes_combined) if LP is not None else None,
            
    }
    if extra_dict is not None and opt.indirect:
        results['indirect'] = extra_dict['indirect_light']
        results['ray_trace_visibility'] = extra_dict['visibility']
    # {"ray_trace_visibility":extra_dict["visibility"],
    # "indirect": extra_dict['indirect_light']}
    if extra_dict is not None:
        results.update(extra_dict)



    return results

def render_surfel_nodefer(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, opt=None):
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    imH = int(viewpoint_camera.image_height)
    imW = int(viewpoint_camera.image_width)

    raster_settings = GaussianRasterizationSettings(
        image_height=imH,
        image_width=imW,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg = torch.zeros_like(bg_color),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    ## reflection strength 定义（即refl ratio）
    refl = pc.get_refl
    ori_color = pc.get_ori_color
    roughness = pc.get_rough
    half_eta = pc.get_half_eta

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # currently don't support normal consistency loss if use precomputed covariance
        splat2world = pc.get_covariance(scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, far-near, near],
            [0, 0, 0, 1]]).float().cuda().T
        world2pix =  viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9) # column major
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    pipe.convert_SHs_python = True
    shs = None
    colors_precomp = None


    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            # sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            # colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            colors_precomp = torch.sigmoid(shs_view[..., 0])
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center)
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    normals = pc.get_normal(scaling_modifier, dir_pp_normalized)
    w_o = -dir_pp_normalized
    reflection = 2 * torch.sum(normals * w_o, dim=1, keepdim=True) * normals - w_o
    # import pdb;pdb.set_trace()
    shs_indirect = pc.get_indirect.transpose(1, 2).view(-1, 3, (pc.indirect_sh_degree+1)**2)
    sh2indirect = eval_sh(3, shs_indirect, reflection)
    indirect = torch.clamp_min(sh2indirect, 0.0)
    sh2indirect_diffuse = eval_sh(0, shs_indirect, normals)
    indirect_diffuse = torch.clamp_min(sh2indirect_diffuse, 0.0)
        
    

    contrib, rendered_image, rendered_features, radii, allmap = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        features = torch.cat((half_eta, roughness, ori_color, indirect,refl, indirect_diffuse), dim=-1),
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
    )


    albedo = rendered_image
    
    half_eta = rendered_features[0:1].repeat(3,1,1)
    roughness = rendered_features[1:2]
    # albedo = rendered_features[4:7]
    indirect_light = rendered_features[5:8]
    refl_strength = rendered_features[8:9]
    indirect_light_diffuse = rendered_features[9:12]


    # 2DGS normal and regularizations
    regularizations = compute_2dgs_normal_and_regularizations(allmap, viewpoint_camera, pipe)
    render_alpha = regularizations['render_alpha']
    render_normal = regularizations['render_normal']
    render_dist = regularizations['render_dist']
    surf_depth = regularizations['surf_depth']
    surf_normal = regularizations['surf_normal']

    # import pdb;pdb.set_trace()
    # Use normal map computed in 2DGS pipeline to perform reflection query
    normal_map = render_normal.permute(1,2,0)
    normal_map = normal_map / render_alpha.permute(1,2,0).clamp_min(1e-6)
    results = {"albedo_map": albedo.permute(1, 2, 0),
            "refl_strength_map": refl_strength.permute(1, 2, 0),
            "half_eta_map": half_eta.permute(1, 2, 0),
            "diffuse_map": None,
            "specular_map": None,
            "roughness_map": roughness.permute(1, 2, 0),
            "rend_normal": normal_map,
            "rend_alpha" : render_alpha.permute(1, 2, 0),
            "depth_map": surf_depth.permute(1, 2, 0)}
    return results

def render_volume(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, 
    scaling_modifier = 1.0, override_color = None,  opt = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    imH = int(viewpoint_camera.image_height)
    imW = int(viewpoint_camera.image_width)

    raster_settings = GaussianRasterizationSettings(
        image_height=imH,
        image_width=imW,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg = torch.zeros_like(bg_color),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    refl = pc.get_refl
    ori_color = pc.get_ori_color
    roughness = pc.get_rough

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # currently don't support normal consistency loss if use precomputed covariance
        splat2world = pc.get_covariance(scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, far-near, near],
            [0, 0, 0, 1]]).float().cuda().T
        world2pix =  viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9) # column major
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    pipe.convert_SHs_python = False
    shs = None
    colors_precomp = None


    NORMAL_RES = False

    dir_pp = (means3D - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)

    normals = pc.get_normal(scaling_modifier, dir_pp_normalized, return_delta=NORMAL_RES)
    delta_normal_norm = None

    # indirect light
    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center)
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    normals = pc.get_normal(scaling_modifier, dir_pp_normalized)
    w_o = -dir_pp_normalized
    reflection = 2 * torch.sum(normals * w_o, dim=1, keepdim=True) * normals - w_o
    # import pdb;pdb.set_trace()
    shs_indirect = pc.get_indirect.transpose(1, 2).view(-1, 3, (pc.indirect_sh_degree+1)**2)
    # import pdb;pdb.set_trace()
    sh2indirect = eval_sh(3, shs_indirect, reflection)
    indirect = torch.clamp_min(sh2indirect, 0.0)


    if opt.indirect:
        diffuse, specular, extra = get_full_color_volume_indirect(pc.get_envmap_2, means3D, ori_color, viewpoint_camera.HWK, viewpoint_camera.R, viewpoint_camera.T, normals.contiguous(), opacity, refl_strength=refl, roughness=roughness, pc=pc, indirect_light=indirect)
        visibility = extra['visibility']
        direct_light = extra["direct_light"]
    else: 
        diffuse, specular = get_full_color_volume(pc.get_envmap_2, means3D, ori_color, viewpoint_camera.HWK, viewpoint_camera.R, viewpoint_camera.T, normals.contiguous(), opacity, refl_strength=refl, roughness=roughness)
    colors_precomp = specular + diffuse




    if opt.indirect:
        features = torch.cat((roughness, refl, diffuse, specular, ori_color, visibility, indirect, direct_light), dim=-1)
    else:
        features = torch.cat((roughness, refl, diffuse, specular, ori_color), dim=-1)

    contrib, rendered_image, rendered_features, radii, allmap = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        features = features,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
    )


    # get rendered diffuse color and other paras
    full_color = rendered_image     # (3,H,W)
    render_roughness = rendered_features[:1]   # (1,H,W)
    render_refl_strength = rendered_features[1:2]   # (1,H,W)
    render_diffuse_color = rendered_features[2:5]
    render_specular_color = rendered_features[5:8]
    render_ori_color = rendered_features[8:11]  #
    if opt.indirect:
        render_visibility = rendered_features[11:12]
        render_indirect = rendered_features[12:15] 
        render_direct = rendered_features[15:18]     





    # 2DGS normal and regularizations
    regularizations = compute_2dgs_normal_and_regularizations(allmap, viewpoint_camera, pipe)
    render_alpha = regularizations['render_alpha']
    render_normal = regularizations['render_normal']
    render_dist = regularizations['render_dist']
    surf_depth = regularizations['surf_depth']
    surf_normal = regularizations['surf_normal']


    final_image = full_color + bg_color[:, None, None] * (1 - render_alpha)
    



    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    results =  {"render": final_image,
            "refl_strength_map": render_refl_strength,
            "diffuse_map": render_diffuse_color,
            "specular_map": render_specular_color,
            "base_color_map": render_ori_color,
            "roughness_map": render_roughness,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
            ## normal, accum alpha, dist, depth map
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_dist': render_dist,
            'surf_depth': surf_depth,
            'surf_normal': surf_normal,
            'orig_normal': regularizations['orig_render_normal']
    }

    if delta_normal_norm is not None:
        results.update({"delta_normal_norm": delta_normal_norm.repeat(3,1,1)})

    if opt.indirect:
        results.update(
            {
                "visibility": render_visibility,
                "indirect_light": render_indirect,
                "direct_light": render_direct
            }
        )

    return results
