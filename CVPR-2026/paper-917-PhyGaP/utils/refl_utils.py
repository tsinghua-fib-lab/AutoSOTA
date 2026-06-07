import torch
import numpy as np
import nvdiffrast.torch as dr
from .general_utils import safe_normalize, flip_align_view
from utils.sh_utils import eval_sh
from utils.stokes_utils import stokes_fac_from_normal
import kornia

env_rayd1 = None
FG_LUT = torch.from_numpy(np.fromfile("assets/bsdf_256_256.bin", dtype=np.float32).reshape(1, 256, 256, 2)).cuda()
def init_envrayd1(H,W):
    i, j = np.meshgrid(
        np.linspace(-np.pi, np.pi, W, dtype=np.float32),
        np.linspace(0, np.pi, H, dtype=np.float32),
        indexing='xy'
    )
    xy1 = np.stack([i, j], axis=2)
    z = np.cos(xy1[..., 1])
    x = np.sin(xy1[..., 1])*np.cos(xy1[...,0])
    y = np.sin(xy1[..., 1])*np.sin(xy1[...,0])
    global env_rayd1
    env_rayd1 = torch.tensor(np.stack([x,y,z], axis=-1)).cuda()

def get_env_rayd1(H,W):
    if env_rayd1 is None:
        init_envrayd1(H,W)
    return env_rayd1

env_rayd2 = None
def init_envrayd2(H,W):
    gy, gx = torch.meshgrid(torch.linspace( 0.0 + 1.0 / H, 1.0 - 1.0 / H, H, device='cuda'), 
                            torch.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W, device='cuda'),
                            # indexing='ij')
                            )
    
    sintheta, costheta = torch.sin(gy*np.pi), torch.cos(gy*np.pi)
    sinphi, cosphi     = torch.sin(gx*np.pi), torch.cos(gx*np.pi)
    
    reflvec = torch.stack((
        sintheta*sinphi, 
        costheta, 
        -sintheta*cosphi
        ), dim=-1)
    global env_rayd2
    env_rayd2 = reflvec

def get_env_rayd2(H,W):
    if env_rayd2 is None:
        init_envrayd2(H,W)
    return env_rayd2



pixel_camera = None
def sample_camera_rays(HWK, R, T):
    H,W,K = HWK
    R = R.T # NOTE!!! the R rot matrix is transposed save in 3DGS
    
    global pixel_camera
    if pixel_camera is None or pixel_camera.shape[0] != H:
        K = K.astype(np.float32)
        i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                        np.arange(H, dtype=np.float32),
                        indexing='xy')
        xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
        pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
        pixel_camera = torch.tensor(pixel_camera).cuda()

    rays_o = (-R.T @ T.unsqueeze(-1)).flatten()
    pixel_world = (pixel_camera - T[None, None]).reshape(-1, 3) @ R
    rays_d = pixel_world - rays_o[None]
    rays_d = rays_d / torch.norm(rays_d, dim=1, keepdim=True)
    rays_d = rays_d.reshape(H,W,3)
    return rays_d, rays_o

def sample_camera_rays_unnormalize(HWK, R, T):
    H,W,K = HWK
    R = R.T # NOTE!!! the R rot matrix is transposed save in 3DGS
    
    global pixel_camera
    if pixel_camera is None or pixel_camera.shape[0] != H:
        K = K.astype(np.float32)
        i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                        np.arange(H, dtype=np.float32),
                        indexing='xy')
        xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
        pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
        pixel_camera = torch.tensor(pixel_camera).cuda()

    rays_o = (-R.T @ T.unsqueeze(-1)).flatten()
    pixel_world = (pixel_camera - T[None, None]).reshape(-1, 3) @ R
    rays_d = pixel_world - rays_o[None]
    rays_d = rays_d.reshape(H,W,3)
    return rays_d, rays_o

def reflection(w_o, normal):
    NdotV = torch.sum(w_o*normal, dim=-1, keepdim=True)
    w_k = 2*normal*NdotV - w_o
    return w_k, NdotV


def get_full_color_surfel_direct(envmap: torch.Tensor, albedo, HWK, R, T, normal_map, render_alpha, scaling_modifier = 1.0, 
                              half_eta = None, refl_strength = None, roughness = None, pc=None, surf_depth=None, 
                              use_stokes=True, LP=None): #RT W2C
    global FG_LUT
    H,W,K = HWK
    rays_cam, rays_o = sample_camera_rays(HWK, R, T)
    w_o = -rays_cam
    rays_refl, NdotV = reflection(w_o, normal_map)
    rays_refl = safe_normalize(rays_refl)
    eta = 1.3 + half_eta if half_eta is not None else 1.5 * torch.ones_like(albedo)
    rays_d_blender = torch.cat((rays_cam[..., 0:1], rays_cam[..., 2:3], -rays_cam[..., 1:2]), dim=-1)
    normal_map_blender = torch.cat((normal_map[...,0:1], normal_map[..., 2:3], -normal_map[..., 1:2]), dim=-1)

    if use_stokes:
        stokes_diff_fac, stokes_spec_fac = stokes_fac_from_normal(rays_o,rays_d_blender,normal_map_blender,eta = eta)
        
        stokes_diff_fac = stokes_diff_fac.squeeze(-3)
        stokes_spec_fac = stokes_spec_fac.squeeze(-3)

    else:
        stokes_diff_fac = None
        stokes_spec_fac = None
    # import pdb;pdb.set_trace()
    # Query BSDF
    fg_uv = torch.cat([NdotV, roughness], -1).clamp(0, 1) 
    fg = dr.texture(FG_LUT, fg_uv.reshape(1, -1, 1, 2).contiguous(), filter_mode="linear", boundary_mode="clamp").reshape(1, H, W, 2) 
    # Compute direct light
    direct_light = envmap(rays_refl, roughness=roughness)
    if torch.isnan(direct_light).any():
        print("NaN detected in direct_light computation!")
        import pdb; pdb.set_trace()
    
    F_0 = ((1-eta)**2/(1+eta)**2).cuda()
    specular_weight = ( F_0 * fg[0][..., 0:1] + fg[0][..., 1:2])
    

    NdotV_clamp = NdotV.clamp(0,1)
    F = F_0 + (1 - F_0) * (1 - NdotV_clamp).pow(5)# ~= (1-NdotV)/(9*NdotV + 1 )
    
    
    direct_diffuse_light = envmap(normal_map, mode="diffuse")

    specular_light = direct_light
    diff_env_light = direct_diffuse_light
    

    # Compute specular color
    specular_raw = specular_light * render_alpha
    specular = specular_raw * specular_weight 
    diffuse =  diff_env_light * (1-F) * albedo * render_alpha #(1-refl_strength) * albedo * render_alpha
    

    if torch.isnan(specular).any():
        print("NaN detected in specular color computation!")
        import pdb; pdb.set_trace()

    if use_stokes == True:
        extra_dict = {
            "direct_light": direct_light.permute(2,0,1),
            "specular_weight": specular_weight.permute(2,0,1),
            "stokes_diff_fac": stokes_diff_fac.permute(2,0,1,3), # H, W, Color, Stokes -> C, Stokes, H, W
            "stokes_spec_fac": stokes_spec_fac.permute(2,0,1,3),
            "F_0": F_0.permute(2,0,1)
        }
    else:
        extra_dict = {
             "direct_light": direct_light.permute(2,0,1),
            "specular_weight": specular_weight.permute(2,0,1),
            "F_0": F_0.permute(2,0,1)
        }
        
    return specular.permute(2,0,1), diffuse.permute(2,0,1), extra_dict


def get_full_color_surfel_indirect(envmap: torch.Tensor, albedo, HWK, R, T, normal_map, render_alpha, scaling_modifier = 1.0, 
                              half_eta = None, refl_strength = None, roughness = None, pc=None, surf_depth=None, 
                              use_stokes=True, LP=None,indirect_light_sh=None, indirect_diffuse_sh = None, indirect_type = "obj_env"): #RT W2C
    global FG_LUT
    H,W,K = HWK
    rays_cam, rays_o = sample_camera_rays(HWK, R, T)
    w_o = -rays_cam
    rays_refl, NdotV = reflection(w_o, normal_map)
    rays_refl = safe_normalize(rays_refl)

    visibility = torch.ones_like(render_alpha)
    if pc.ray_tracer is not None:
        mask = (render_alpha>0)[..., 0]
        sd_b = (surf_depth.squeeze(0)>0).bool()
        mask = mask & sd_b
        intersections = rays_o + surf_depth.permute(1, 2, 0) * rays_cam
        # import pdb;pdb.set_trace()
        _, _, depth = pc.ray_tracer.trace(intersections[mask], rays_refl[mask])
        visibility[mask] = (depth >= 10).float().unsqueeze(-1)
        # import pdb;pdb.set_trace()
        if indirect_type in ["obj_env", "surf_env"] and pc.gridmap_envmap is not None:   # Build a full-sized depth map where entries corresponding to mask are filled
            # assert pc.gridmap_envmap is not None, "gridmap_envmap must be provided in pc when indirect_type is 'obj_env'"
            assert len(pc.gridmap_envmap) > 0, "gridmap_envmap is empty"
            
            # Pass camera world origin (rays_o) so we can filter centers by camera visibility when needed
            indirect_light, indirect_diffuse , _ = get_gridmap_envmaps_indirect_light(
                pc,
                render_alpha,
                intersections,
                depth,
                normal_map,
                rays_refl,
                roughness,
                indirect_type=indirect_type,
                cam_o=rays_o
            )       
        else:
            indirect_light, indirect_diffuse = indirect_light_sh, indirect_diffuse_sh


    eta = 1.3 + half_eta if half_eta is not None else 1.5 * torch.ones_like(albedo)

    rays_d_blender = torch.cat((rays_cam[..., 0:1], rays_cam[..., 2:3], -rays_cam[..., 1:2]), dim=-1)
    normal_map_blender = torch.cat((normal_map[...,0:1], normal_map[..., 2:3], -normal_map[..., 1:2]), dim=-1)
    stokes_diff_fac, stokes_spec_fac = stokes_fac_from_normal(rays_o,rays_d_blender,normal_map_blender,eta = eta) # We follow the stokes fac from normal implementation in PANDORA, which assumes blender coordinate system
    stokes_diff_fac = stokes_diff_fac.squeeze(-3)
    stokes_spec_fac = stokes_spec_fac.squeeze(-3)

    fg_uv = torch.cat([NdotV, roughness], -1).clamp(0, 1) 
    fg = dr.texture(FG_LUT, fg_uv.reshape(1, -1, 1, 2).contiguous(), filter_mode="linear", boundary_mode="clamp").reshape(1, H, W, 2) 
    # Compute direct light
    direct_light = envmap(rays_refl, roughness=roughness)
    if torch.isnan(direct_light).any():
        print("NaN detected in direct_light computation!")
        import pdb; pdb.set_trace()
    
    F_0 = ((1-eta)**2/(1+eta)**2).cuda()
    specular_weight = ( F_0 * fg[0][..., 0:1] + fg[0][..., 1:2])

    NdotV_clamp = NdotV.clamp(0,1)
    F = F_0 + (1 - F_0) * (1 - NdotV_clamp).pow(5)# ~= (1-NdotV)/(9*NdotV + 1 )
    
    
    direct_diffuse_light = envmap(normal_map, mode="diffuse")
    if pc.ray_tracer is not None:
        # indirect light
        specular_light = direct_light * visibility + (1 - visibility) * indirect_light
        if indirect_type in ["obj_env", "surf_env"] and pc.gridmap_envmap is not None:
            diff_env_light = indirect_diffuse # use GridMap interpolated diffuse for better quality
        else:
            diff_env_light = direct_diffuse_light * visibility + (1 - visibility) * indirect_diffuse
        indirect_color = (1 - visibility) * indirect_light * render_alpha * specular_weight
    else:
        specular_light = direct_light
        diff_env_light = direct_diffuse_light
    

    # Compute specular color
    specular_raw = specular_light * render_alpha
    specular = specular_raw * specular_weight 
    diffuse =  diff_env_light * (1-F) * albedo * render_alpha #(1-refl_strength) * albedo * render_alpha
    

    if torch.isnan(specular).any():
        print("NaN detected in specular color computation!")
        import pdb; pdb.set_trace()
    if use_stokes ==False:
        # if stokes is not used, no need to compute stokes facs
        stokes_diff_fac = torch.zeros((H,W,3,4), device=direct_light.device)
        stokes_spec_fac = torch.zeros((H,W,3,4), device=direct_light.device)
    extra_dict = {
        "visibility": visibility.permute(2,0,1),
        "indirect_light": indirect_light.permute(2,0,1),
        "direct_light": direct_light.permute(2,0,1),
        "indirect_color": indirect_color.permute(2,0,1),
        "specular_weight": specular_weight.permute(2,0,1),
        "stokes_diff_fac": stokes_diff_fac.permute(2,0,1,3),
        "stokes_spec_fac": stokes_spec_fac.permute(2,0,1,3),
        "F_0": F_0.permute(2,0,1)
    } 
        
    return specular.permute(2,0,1), diffuse.permute(2,0,1), extra_dict

def get_full_color_volume(envmap: torch.Tensor, xyz, albedo, HWK, R, T, normal_map, render_alpha, scaling_modifier = 1.0, refl_strength = None, roughness = None): #RT W2C
    global FG_LUT
    _, rays_o = sample_camera_rays(HWK, R, T)
    N, _ = normal_map.shape
    rays_o = rays_o.expand(N, -1)
    w_o = safe_normalize(rays_o - xyz)
    rays_refl, NdotV = reflection(w_o, normal_map)
    rays_refl = safe_normalize(rays_refl)

    # Query BSDF
    fg_uv = torch.cat([NdotV, roughness], -1).clamp(0, 1) 
    fg_uv = fg_uv.unsqueeze(0).unsqueeze(2)  # [1, N, 1, 2]
    fg = dr.texture(FG_LUT, fg_uv, filter_mode="linear", boundary_mode="clamp").squeeze(2).squeeze(0)  # [N, 2]
    # Compute diffuse
    diffuse = envmap(normal_map, mode="diffuse") * (1-refl_strength) * albedo
    # Compute specular
    specular = envmap(rays_refl, roughness=roughness) * ((0.04 * (1 - refl_strength) + albedo * refl_strength) * fg[0][..., 0:1] + fg[0][..., 1:2]) 

    return diffuse, specular




def get_full_color_volume_indirect(envmap: torch.Tensor, xyz, albedo, HWK, R, T, normal_map, render_alpha, scaling_modifier = 1.0, refl_strength = None, roughness = None, pc=None, indirect_light=None): #RT W2C
    global FG_LUT
    _, rays_o = sample_camera_rays(HWK, R, T)
    N, _ = normal_map.shape
    rays_o = rays_o.expand(N, -1)
    w_o = safe_normalize(rays_o - xyz)
    rays_refl, NdotV = reflection(w_o, normal_map)
    rays_refl = safe_normalize(rays_refl)

    # visibility
    visibility = torch.ones_like(render_alpha)
    if pc.ray_tracer is not None:
        mask = (render_alpha>0).squeeze()
        intersections = xyz
        _, _, depth = pc.ray_tracer.trace(intersections[mask], rays_refl[mask])
        visibility[mask] = (depth >= 10).unsqueeze(1).float()

    # Query BSDF
    fg_uv = torch.cat([NdotV, roughness], -1).clamp(0, 1) 
    fg_uv = fg_uv.unsqueeze(0).unsqueeze(2)  # [1, N, 1, 2]
    fg = dr.texture(FG_LUT, fg_uv, filter_mode="linear", boundary_mode="clamp").squeeze(2).squeeze(0)  # [N, 2]
    # Compute diffuse
    diffuse = envmap(normal_map, mode="diffuse") * (1-refl_strength) * albedo
    # Compute specular
    direct_light = envmap(rays_refl, roughness=roughness) 
    specular_weight = ((0.04 * (1 - refl_strength) + albedo * refl_strength) * fg[0][..., 0:1] + fg[0][..., 1:2]) 
    specular_light = direct_light * visibility + (1 - visibility) * indirect_light
    specular = specular_light * specular_weight

    extra_dict = {
        "visibility": visibility,
        "direct_light": direct_light,
    }

    return diffuse, specular, extra_dict



import time
def get_gridmap_envmaps_indirect_light(pc, render_alpha, intersections, depth, normal_map, rays_refl, roughness, indirect_type='obj_env', cam_o=None):
    """
    Use grid map envmaps to compute indirect lighting for pixels that are not directly visible to the global envmap.
    Input：
        pc: GaussianModel
        pipe: rendering pipeline params
        render_alpha: (H,W,1) float tensor
        intersections: (H,W,3) float tensor
        depth: (H,W) float tensor, ray tracing depth
        normal_map: (H,W,3) float tensor
        rays_refl: (H,W,3) float tensor, reflection direction of camera rays
        roughness: (H,W,1) float tensor
    """
    start = time.time()
    nearest_type_list = ['mesh_env']
    mask = (render_alpha>0)[..., 0]
    H,W = render_alpha.shape[0:2]
    visibility = torch.ones_like(render_alpha)
    visibility[mask] = (depth >= 10).float().unsqueeze(-1)
    indirect_light = torch.zeros_like(normal_map)
    indirect_diffuse = torch.zeros_like(normal_map)
    # Build a full-sized depth map where entries corresponding to mask are filled
    depth_full = torch.zeros_like(visibility[..., 0])
    # import pdb;pdb.set_trace()
    depth_full[mask] = depth
    # keep gridmap_envmap_centers as before
    gridmap_envmap_centers = pc.gridmap_envmap_xyz # k,3
    
    
    # ensure centers is a torch tensor on same device/dtype as mid_points_unvis
    centers = gridmap_envmap_centers.to(intersections.device).to(intersections.dtype)
    # Work on a local copy of envmaps to avoid mutating pc.gridmap_envmap
    envmaps = pc.gridmap_envmap if hasattr(pc, 'gridmap_envmap') and pc.gridmap_envmap is not None else []

    k = centers.shape[0]
    # initialize empty masks (k, H, W)
    gridmap_env_masks = torch.zeros((k, H, W), dtype=torch.bool, device=intersections.device) 
    
    intersections_1d = intersections[mask] 
    if intersections_1d.numel() > 0 and k > 0:

        dif = intersections_1d.unsqueeze(1) - centers.unsqueeze(0)
        d2 = torch.sum(dif * dif, dim=-1)  # (M, k)

        assign = torch.argmin(d2, dim=1)   # (M,)

        # map local masked positions to global HxW indices
        mask_inds = torch.nonzero(mask, as_tuple=False)  # (M,2)
        # for each center, set corresponding global positions True
        
        # Vectorized (parallel) version of the loop above:
        # assign: (M,), mask_inds: (M,2)
        # For each element m, set gridmap_env_masks[assign[m], y, x] = True
        gridmap_env_masks[assign.long(), mask_inds[:, 0], mask_inds[:, 1]] = True

        M = intersections_1d.shape[0]
        eps = 1e-6
        inv_d2 = 1.0 / (d2 + eps)          # (M,k)
        weights = inv_d2 / inv_d2.sum(dim=1, keepdim=True)  # 归一化 (M,k)
        # print("Weight computation time:", time.time() - start)
        # start=time.time()
        
        if indirect_type not in nearest_type_list:

            if k > 0:
                rough_mask = roughness[mask]
                normals_mask = safe_normalize(normal_map[mask])
                rays_refl_mask = rays_refl[mask]
                # 并行获得所有 envmap 的结果
                # import pdb;pdb.set_trace()
                diffuse_stack = batch_query(
                    envmaps,
                    normals_mask,
                    mode="diffuse"
                ).permute(1, 0, 2).contiguous()                                     # (M, k, 3)
                indirect_stack = batch_query(
                    envmaps,
                    rays_refl_mask,
                    roughness=rough_mask
                ).permute(1, 0, 2).contiguous()                                     # (M, k, 3)

                w = weights.unsqueeze(-1)             # (M, k, 1)
                indirect_diffuse[mask] = (diffuse_stack * w).sum(dim=1)
                indirect_light[mask] = (indirect_stack * w).sum(dim=1)
                
        elif indirect_type in nearest_type_list:
            center_has_pixels = gridmap_env_masks.view(k, -1).any(dim=1)
            if not center_has_pixels.any():
                # no pixels assigned to this center, skip to avoid empty queries
                pass
            else:
                normals_mask = safe_normalize(normal_map[mask])
                rough_mask = roughness[mask]
                rays_refl_mask = rays_refl[mask]
                assign_idx = assign.to(device=normals_mask.device, dtype=torch.long)
                diffuse_stack = batch_query(envmaps, normals_mask, mode="diffuse").permute(1, 0, 2).contiguous()
                indirect_stack = batch_query(envmaps, rays_refl_mask, roughness=rough_mask).permute(1, 0, 2).contiguous()
                gather_idx = assign_idx.view(-1, 1, 1).expand(-1, 1, diffuse_stack.shape[-1])
                indirect_diffuse_selected = diffuse_stack.gather(dim=1, index=gather_idx).squeeze(1)
                indirect_light_selected = indirect_stack.gather(dim=1, index=gather_idx).squeeze(1)
                indirect_diffuse[mask] += indirect_diffuse_selected
                indirect_light[mask] += indirect_light_selected
        # import pdb;pdb.set_trace()
    return indirect_light, indirect_diffuse, gridmap_env_masks   

def batch_query(envmaps, directions, roughness=None, mode=None):
    """
    Batch query a sequence of GridMap envmaps for speed considerations.
    NOTE: This part should be fused into the build-up process of Gridmap envmaps for better efficiency.
    For the simplicity of implementation, we keep it as a separate function for now, but still it accelerates the querying process.
    """
    if len(envmaps) == 0:
        raise ValueError("envmaps must be a non-empty sequence.")
    device = envmaps[0].device
    dtype = envmaps[0].base.dtype

    dirs = directions.to(device=device, dtype=dtype)
    prefix = dirs.shape[:-1]
    flat_dirs = dirs.reshape(1, 1, -1, 3)

    num_envs = len(envmaps)

    def _flatten_cubemaps(stack):
        faces, height, width, channels = stack.shape[1:]
        return stack.permute(1, 2, 3, 4, 0).reshape(6, height, width, channels * num_envs).contiguous()

    def _reshape_output(light_tensor, channels):
        light_tensor = light_tensor.view(1, *prefix, channels * num_envs)
        light_tensor = light_tensor.view(1, *prefix, channels, num_envs)
        return torch.movedim(light_tensor, -1, 0).squeeze(1)

    rough = None
    if roughness is not None:
        rough = roughness.to(device=device, dtype=dtype).reshape(1, 1, -1, 1)

    if mode == "diffuse":
        tex = torch.stack([env.diffuse for env in envmaps], dim=0)
        tex_flat = _flatten_cubemaps(tex).unsqueeze(0)
        light = dr.texture(tex_flat, flat_dirs, filter_mode='linear', boundary_mode='cube')
        light = torch.where(light > 0, torch.relu(light + 0.5), torch.sigmoid(light))
        light = _reshape_output(light, tex.shape[-1])
    elif mode == "pure_env":
        tex = torch.stack([env.base for env in envmaps], dim=0)
        tex_flat = _flatten_cubemaps(tex).unsqueeze(0)
        light = dr.texture(tex_flat, flat_dirs, filter_mode='linear', boundary_mode='cube')
        light = torch.where(light > 0, torch.relu(light + 0.5), torch.sigmoid(light))
        light = _reshape_output(light, tex.shape[-1])
    else:
        if rough is None:
            raise ValueError("roughness must be provided for specular queries.")
        spec_len = len(envmaps[0].specular)
        if any(len(env.specular) != spec_len for env in envmaps):
            raise ValueError("All envmaps must share identical mip chain length.")
        base_tex = torch.stack([env.specular[0] for env in envmaps], dim=0)
        base_flat = _flatten_cubemaps(base_tex).unsqueeze(0)
        mip_tex_flat = []
        for level in range(1, spec_len):
            level_stack = torch.stack([env.specular[level] for env in envmaps], dim=0)
            mip_tex_flat.append(_flatten_cubemaps(level_stack).unsqueeze(0))
        mip_level = envmaps[0].get_mip(rough)[..., 0]
        light = dr.texture(
            base_flat,
            flat_dirs,
            mip=mip_tex_flat,
            mip_level_bias=mip_level,
            filter_mode='linear-mipmap-linear',
            boundary_mode='cube'
        )
        light = torch.where(light > 0, torch.relu(light + 0.5), torch.sigmoid(light))
        light = _reshape_output(light, base_tex.shape[-1])
    return light