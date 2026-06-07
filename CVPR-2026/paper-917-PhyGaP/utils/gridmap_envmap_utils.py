import math
import torch
import numpy as np
import nvdiffrast.torch as dr
from scene.cameras import Camera
from utils.refl_utils import sample_camera_rays
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from gaussian_renderer import compute_2dgs_normal_and_regularizations, render_surfel_nodefer
from utils.sh_utils import eval_sh
from tqdm import tqdm
from sklearn.cluster import KMeans
from utils.general_utils import safe_normalize
from utils.refl_utils import sample_camera_rays, reflection
import matplotlib.pyplot as plt

def farthest_point_sampling(xyz, n_samples):
        """
        xyz: (N, 3) tensor
        n_samples: int
        return: (n_samples,) long tensor of selected indices
        """
        # Robustify input: accept (..., 3) and flatten to (N,3)
        if xyz.ndim != 2 or xyz.shape[-1] != 3:
            xyz = xyz.reshape(-1, 3)
        N = xyz.shape[0]
        centroids = torch.zeros(n_samples, dtype=torch.long, device=xyz.device)
        distance = torch.full((N,), float('inf'), device=xyz.device)
        farthest = torch.randint(0, N, (1,), device=xyz.device).item()
        for i in range(n_samples):
            centroids[i] = farthest
            centroid = xyz[farthest].unsqueeze(0)  # (1,3)
            dist = torch.norm(xyz - centroid, dim=1)
            distance = torch.minimum(distance, dist)
            farthest = torch.argmax(distance).item()
        return centroids

def _batched_knn_interpolate(query_pts, points, colors, k=3, batch_size=4096):
    """Robust KNN color interpolation with defensive checks.

    Args:
        query_pts: (M,3) float tensor (device: cuda/cpu)
        points:    (P,3) float tensor
        colors:    (P,3) float tensor (one color per point)
        k:         number of neighbors (int)
        batch_size: chunk size for cdist/topk processing

    Returns:
        (M,3) interpolated colors. If P == 0 returns zeros.
    """
    # Basic validation / early exits
    if points is None or colors is None:
        return torch.zeros((query_pts.shape[0], 3), device=query_pts.device, dtype=query_pts.dtype)
    if points.shape[0] == 0 or colors.shape[0] == 0:
        return torch.zeros((query_pts.shape[0], 3), device=query_pts.device, dtype=query_pts.dtype)
    if points.shape[0] != colors.shape[0]:
        raise ValueError(f"points ({points.shape}) and colors ({colors.shape}) length mismatch")
    if query_pts.numel() == 0:
        return torch.zeros((0, 3), device=query_pts.device, dtype=query_pts.dtype)

    device = query_pts.device
    P = points.shape[0]
    k_eff = min(max(1, k), P)

    # Torch fallback path (chunked cdist + topk) ---------------------------------
    M = query_pts.shape[0]
    out = torch.zeros((M, 3), dtype=colors.dtype, device=device)
    # Pre-normalize to reduce risk of huge distances (optional)
    # (Not normalizing here because we want absolute distances.)

    for s in range(0, M, batch_size):
        e = min(s + batch_size, M)
        q = query_pts[s:e]  # (B,3)
        # Compute squared distances using (q - p)^2 = q^2 + p^2 -2 q·p
        q_sq = (q * q).sum(dim=1, keepdim=True)  # (B,1)
        p_sq = (points * points).sum(dim=1).unsqueeze(0)  # (1,P)
        prod = q @ points.t()  # (B,P)
        d2 = (q_sq + p_sq - 2 * prod).clamp_min_(0.0)
        # Top-k
        vals, idx = torch.topk(d2, k_eff, largest=False, sorted=False)  # (B,k)
        # Defensive clamp (should not be necessary but avoids device assert if something upstream changed sizes)
        if idx.numel() and (idx.max() >= P or idx.min() < 0):
            idx = idx.clamp(0, P - 1)
        d = vals.clamp_min_(1e-20).sqrt_()  # (B,k)
        neighbor_cols = colors[idx]  # (B,k,3)
        w = 1.0 / (d + 1e-8)
        w = w / (w.sum(dim=1, keepdim=True) + 1e-12)
        out[s:e] = (neighbor_cols * w.unsqueeze(-1)).sum(dim=1)

    return out


def cubemap_views():
    # 顺序严格对应envmap_base的六面
    # 0: +X (right)
    # 1: -X (left)
    # 2: +Y (top)
    # 3: -Y (bottom)
    # 4: +Z (front)
    # 5: -Z (back)
    return [
        (np.array([-1,0,0]), np.array([0,1,0])),  # -X, up +Y
        (np.array([1,0,0]), np.array([0,1,0])),   # +X, up +Y
        (np.array([0,1,0]), np.array([0,0,-1])),    # +Y, up -Z
        (np.array([0,-1,0]), np.array([0,0,1])),  # -Y, up +Z
        (np.array([0,0,1]), np.array([0,1,0])),   # +Z, up +Y
        (np.array([0,0,-1]), np.array([0,1,0])),  # -Z, up +Y
    ]

def render_cubemap(pc, center, pipe, model_params, device="cuda",fov_deg = 90):
    """
    Render the 6 faces of cubemap, centered at 'center'.
    Input:
        pc: GaussianModel
        center: (3,) array-like，center of cubemap
        pipe: rendering pipeline params
        model_params: model params
        device: cuda/cpu
    Output: list of 6 faces, each (3,H,W) tensor

    """
    H,W = model_params.envmap_max_res, model_params.envmap_max_res
    FoV = math.radians(fov_deg)
    envmap_base = pc.env_map.get_cube_map().permute(0, 3, 1, 2).contiguous()  # (6, 3, H, W)
    diffuse_light_src = pc.env_map

    cubemap = []

    # prepare point cloud for knn
    points = pc.get_xyz  # (P,3)
    # Evaluate SH -> RGB per point using pc normals (so colors come from SH features)
    shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2).contiguous()

    # compute per-point normals to use as SH evaluation directions
    # dir_pp_normalized = vector from point to camera center (use current face cam)
    # import pdb; pdb.set_trace() 
    dir_pp = (points - torch.from_numpy(center).to(device=device, dtype=points.dtype))
    dir_pp_norm = dir_pp / (dir_pp.norm(dim=-1, keepdim=True) + 1e-12)
    normals = pc.get_normal(1.0, dir_pp_norm)
    dirs_for_sh = normals / (normals.norm(dim=-1, keepdim=True) + 1e-12)

    # evaluate SH to get colors (will keep gradient path)
    colors_precomp = eval_sh(pc.active_sh_degree, shs_view, dirs_for_sh)
    # match existing pipeline convention: clamp and shift
    colors = torch.clamp_min(colors_precomp + 0.5, 0.0).to(device)

    # precompute camera intrinsics inverse for our simple pinhole model
    focal = W / (2 * math.tan(FoV / 2))
    K = np.array([[focal, 0.0, W / 2.0], [0.0, focal, H / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    # batch sizes
    # Reduce trace batch size to avoid launching extremely large GPU kernels that may trigger device-side asserts
    trace_batch = 1<<16 #min(1 << 16, max(1024, (H * W) // 4))  # conservative default
    knn_batch = 2048
    # import pdb; pdb.set_trace()
    for idx, (look_dir, up) in enumerate(cubemap_views()):
        eye = np.array(center)
        # target = eye + look_dir
        z_axis = (look_dir) / np.linalg.norm(look_dir)
        x_axis = np.cross(z_axis,up)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        R = np.stack([x_axis, y_axis, z_axis], axis=1)  # 3x3
        T = -R.T @ eye

        # Build ray directions in world for every pixel without calling sample_camera_rays
        # Prepare rotation as camera-to-world; camera origin is at 'eye'
        R_torch = torch.tensor(R, dtype=torch.float32, device=device)
        rays_o_cam = torch.tensor(eye, dtype=torch.float32, device=device)

        # Create pixel grid and compute pinhole directions in camera coords
        cx = W / 2.0
        cy = H / 2.0
        # pixel centers
        u = torch.arange(W, device=device, dtype=torch.float32) + 0.5
        v = torch.arange(H, device=device, dtype=torch.float32) + 0.5
        uu = u.view(1, W).expand(H, W)
        vv = v.view(H, 1).expand(H, W)
        x_cam = (uu - cx) / focal
        y_cam = (vv - cy) / focal
        z_cam = torch.ones((H, W), device=device, dtype=torch.float32)
        dirs_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)
        dirs_cam = dirs_cam / (dirs_cam.norm(dim=-1, keepdim=True) + 1e-12)
        # Transform to world: v_world = R @ v_cam; with row-vector, multiply by R^T
        dirs_world = torch.matmul(dirs_cam, R_torch.t()).reshape(-1, 3)
        Npix = dirs_world.shape[0]
        rays_o = rays_o_cam.to(device=device).unsqueeze(0).repeat(Npix, 1)

        colors_out = torch.zeros((Npix, 3), dtype=torch.float32, device=device)
        env_colors = None

        # try to use raytracer if available
        rt = pc.ray_tracer
        # store per-pixel depths so we can strictly apply depth>10 rule
        depths_all = torch.full((Npix,), float(1e9), device=device, dtype=torch.float32)
        # iterate in chunks to avoid memory blowup
        for s in range(0, Npix, trace_batch):
            e = min(s + trace_batch, Npix)
            ro = rays_o[s:e]
            rd = dirs_world[s:e]
            # ensure input tensors are contiguous and on correct device/dtype
            ro = ro.contiguous().float()
            rd = rd.contiguous().float()

            positions, face_normals, depth = rt.trace(ro, rd)

            # positions (B,3), depth (B,)
            # store depths
            depths_all[s:e] = depth

            # valid hits are those with 0 < depth <= 10 (strict rule)
            valid_mask = (depth > 0) & (depth < 10.0)
            # ensure mask is on the same device as colors_out and is boolean
            try:
                if valid_mask.device != colors_out.device:
                    valid_mask = valid_mask.to(device=colors_out.device)
            except Exception:
                # depth/valid_mask might be numpy array; convert to tensor on device
                valid_mask = torch.as_tensor(valid_mask, device=colors_out.device)
            valid_mask = valid_mask.bool()

            if valid_mask.any():
                hit_pos = positions[valid_mask]
                hit_normal = face_normals[valid_mask]
                interpolated = _batched_knn_interpolate(hit_pos, points, colors, k=3, batch_size=knn_batch)
                diffuse_env = diffuse_light_src(hit_normal, mode="diffuse").detach()
                diffuse = diffuse_env * interpolated
                del diffuse_env
                del interpolated

                # defensive shape check before assignment to avoid device-side asserts
                num_sel = int(valid_mask.sum().item())
            
                colors_out[s:e][valid_mask] = diffuse

            # for other pixels (miss or depth>10), leave zeros for now; will sample envmap later

        # use envmap base directly per-face where depth<=0 or depth>10 (strict rule)
        # import pdb; pdb.set_trace()
        if env_colors is None:
            # envmap_base has shape (6, 3, H, W)
            # pick the current face and flatten to (H*W, 3)
            env_face = envmap_base[idx]  # (3, H, W)
            # convert to (H, W, 3) then flatten to (H*W, 3)
            env_colors = env_face.permute(1, 2, 0).contiguous().view(-1, 3)

        # build mask where we should use envmap: depth<=0 (miss) OR depth>10
        use_env_mask = (depths_all <= 0) | (depths_all >= 10.0) | torch.isnan(depths_all)
        # import pdb; pdb.set_trace() 
        colors_out[use_env_mask] = env_colors[use_env_mask]

        # reshape colors_out to (3,H,W)
        face = colors_out.t().view(3, H, W)
        cubemap.append(face)

    return torch.stack(cubemap, dim=0), diffuse_light_src  # (6, 3, H, W)


def build_cubemap(cameras, gaussians, pipe, background, dataset, opt,fix_center = False,max_env_num=96):
    gridmap_cubemaps = []
    # Add 52 cubemaps at the bounding box corners of gaussians
    pts = gaussians.get_xyz
    if isinstance(pts, torch.Tensor) and pts.numel() > 0:
        mins = pts.min(dim=0).values
        maxs = pts.max(dim=0).values
        x0, y0, z0 = [float(v) for v in mins.tolist()]
        x1, y1, z1 = [float(v) for v in maxs.tolist()]
        corners = [
            np.array([x0, y0, z0], dtype=np.float32),
            np.array([x0, y0, z1], dtype=np.float32),
            np.array([x0, y1, z0], dtype=np.float32),
            np.array([x0, y1, z1], dtype=np.float32),
            np.array([x1, y0, z0], dtype=np.float32),
            np.array([x1, y0, z1], dtype=np.float32),
            np.array([x1, y1, z0], dtype=np.float32),
            np.array([x1, y1, z1], dtype=np.float32),
        ]

        # Add points on edges (2 per edge at 1/3 and 2/3) and faces (4 per face at 1/3 offsets)
        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
        x1 = x1+0.1*dx # expand the box a bit to get more coverage
        y1 = y1+0.1*dy
        z1 = z1+0.1*dz
        x0 = x0-0.1*dx
        y0 = y0-0.1*dx
        z0 = z0-0.1*dx
        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
        samples = list(corners)
        def add_point(x, y, z):
            samples.append(np.array([x, y, z], dtype=np.float32))

        # Edges along X (y in {y0,y1}, z in {z0,z1})
        for y in (y0, y1):
            for z in (z0, z1):
                add_point(x0 + dx / 3.0, y, z)
                add_point(x0 + 2.0 * dx / 3.0, y, z)
        # Edges along Y (x in {x0,x1}, z in {z0,z1})
        for x in (x0, x1):
            for z in (z0, z1):
                add_point(x, y0 + dy / 3.0, z)
                add_point(x, y0 + 2.0 * dy / 3.0, z)
        # Edges along Z (x in {x0,x1}, y in {y0,y1})
        for x in (x0, x1):
            for y in (y0, y1):
                add_point(x, y, z0 + dz / 3.0)
                add_point(x, y, z0 + 2.0 * dz / 3.0)

        # Faces x = const (vary y, z)
        for x in (x0, x1):
            for y in (y0 + dy / 3.0, y0 + 2.0 * dy / 3.0):
                for z in (z0 + dz / 3.0, z0 + 2.0 * dz / 3.0):
                    add_point(x, y, z)

        # Faces y = const (vary x, z)
        for y in (y0, y1):
            for x in (x0 + dx / 3.0, x0 + 2.0 * dx / 3.0):
                for z in (z0 + dz / 3.0, z0 + 2.0 * dz / 3.0):
                    add_point(x, y, z)

        # # Faces z = const (vary x, y)
        
        for x in (x0 + dx / 3.0, x0 + 2.0 * dx / 3.0):
            for y in (y0 + dy / 3.0, y0 + 2.0 * dy / 3.0):
                add_point(x, y, z1)

        # Deduplicate (handles degenerate boxes where dx/dy/dz may be 0)
        unique = []
        seen = set()
        for p in samples:
            key = tuple(np.round(p, 6))
            if key not in seen:
                seen.add(key)
                unique.append(p)
        corners = unique
        for corner in corners:
            # import pdb; pdb.set_trace()
            cubemap, _ = render_cubemap(gaussians, corner, pipe, dataset, device="cuda")
            gridmap_cubemaps.append((cubemap, corner))
    return gridmap_cubemaps

def build_surf_cubemap(cameras, gaussians, pipe, background, dataset, opt, fix_center = False, max_env_num = 64):
    if fix_center and gaussians.gridmap_envmap_xyz is not None and len(gaussians.gridmap_envmap_xyz)>0:
        assert False, "fix_center=True not supported for surf_env"
        centers = gaussians.gridmap_envmap_xyz.cpu().numpy()
        cluster_num = len(centers)
    else:
        targets = []
        target_normal = []
        target_albedo = []
        print("Building GridMaps...")
        for cam_idx, camera in enumerate(tqdm(cameras)):
            first_stage_result = render_surfel_nodefer(camera, gaussians, pipe, background, opt=opt)
            normal_map = first_stage_result['rend_normal']
            render_alpha = first_stage_result['rend_alpha']
            surf_depth = first_stage_result['depth_map']
            albedo = first_stage_result['albedo_map']

            H,W,K = camera.HWK
            rays_cam, rays_o = sample_camera_rays(camera.HWK, camera.R, camera.T)
            w_o = safe_normalize(-rays_cam)
            rays_refl, NdotV = reflection(w_o, normal_map)
            rays_refl = safe_normalize(rays_refl)
            visibility = torch.ones_like(render_alpha)
            target = torch.zeros_like(rays_refl)
            normal_t = torch.zeros_like(normal_map)
            albedo_t = torch.zeros_like(albedo)
            if gaussians.ray_tracer is not None:
                mask = camera.gt_alpha_mask.cuda().bool() if camera.gt_alpha_mask is not None else (render_alpha>0)[..., 0]
                # import pdb; pdb.set_trace()
                if mask.dim() == 4:
                    mask = mask.squeeze(0).squeeze(0)
                elif mask.dim() == 3:
                    mask = mask.squeeze(0)
                else:
                    mask = mask
                sd_b = (surf_depth.squeeze(-1)>0).bool()
                mask = mask & sd_b & (render_alpha>0)[..., 0]
                intersections = rays_o + surf_depth * rays_cam
                # import pdb;pdb.set_trace()
                _, _, depth = gaussians.ray_tracer.trace(intersections[mask], rays_refl[mask])
                visibility[mask] = (depth >= 10).float().unsqueeze(-1)
                depth = torch.clamp(depth, max=1.0)
                target[mask] = (intersections[mask] + depth.unsqueeze(-1) * rays_refl[mask]*0.01) #ensure outside the surface
                # target[mask] = intersections[mask] #+  depth.unsqueeze(-1) * rays_refl[mask] * 0.01 #close to the surface
                normal_t[mask] = normal_map[mask].detach()
                albedo_t[mask] = albedo[mask].detach()
            # import pdb; pdb.set_trace() 
            # 只收集visibility为0的target点（每个点3个维度）
            # mask_indices = (visibility[..., 0] == 0)
            # import pdb; pdb.set_trace()
            # selected_targets = target[mask_indices]  # shape: (N, 3)
            # selected_target_normal = normal_t[mask_indices]
            # selected_target_albedo = albedo_t[mask_indices]
            # Reshape to (N, 3) for KMeans (flatten all pixels)
            selected_targets = target[mask]
            selected_target_normal = normal_t[mask]
            selected_target_albedo = albedo_t[mask]
            if selected_targets.numel() > 0:
                targets.append(selected_targets)
                target_normal.append(selected_target_normal)
                target_albedo.append(selected_target_albedo)
            else:
                import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace() 
        # if len(targets) == 0:
        #     import pdb;pdb.set_trace()
        targets = torch.cat(targets, dim=0)
        target_normal = torch.cat(target_normal, dim=0)
        target_albedo = torch.clamp_min(torch.cat(target_albedo, dim=0), 0.0)
        # 随机取1/20的点
        # import pdb; pdb.set_trace() 
        num_points = targets.shape[0]
        sample_size = max(1, num_points // 200)
        indices = torch.randperm(num_points)[:sample_size]
        targets = targets[indices]
        target_normal = target_normal[indices]
        target_albedo = target_albedo[indices]

        # 使用 FPS 替代 KMeans，从子采样后的 targets 中挑选代表点
        num_pts = targets.shape[0]
        cluster_num = min(max_env_num, sample_size)
        if num_pts == 0:
            # 无可用目标点，返回空结果
            return []
        if num_pts <= cluster_num:
            fps_indices = torch.arange(num_pts, device=targets.device)
        else:
            fps_indices = farthest_point_sampling(targets, cluster_num)

        # 根据 FPS 选中的索引提取中心、法向和反照率
        centers_t = targets[fps_indices]
        centers_normal_t = target_normal[fps_indices]
        centers_albedo_t = target_albedo[fps_indices]

        # 转为 numpy 以与后续渲染接口保持一致
        centers = centers_t.detach().cpu().numpy()
        centers_normal = centers_normal_t.detach().cpu().numpy()
        centers_albedo = centers_albedo_t.detach().cpu().numpy()

    gridmap_cubemaps = []
    # import pdb; pdb.set_trace()
    for j in range(min(opt.num_gridmap_envmaps, cluster_num)):
        # import pdb; pdb.set_trace() 
        cubemap,_ = render_cubemap(gaussians, centers[j], pipe, dataset, device="cuda")
        gridmap_cubemaps.append((cubemap,centers[j],centers_normal[j]))
    # import pdb; pdb.set_trace()
    return gridmap_cubemaps

def build_mesh_cubemap(mesh,gaussians, pipe, background, dataset, opt, max_env_num = 96):
    vertices_np = np.array(mesh.vertices)
    targets = torch.from_numpy(vertices_np).to(device="cuda", dtype=torch.float32)
    num_points = targets.shape[0]
    sample_size = max(1, num_points // 50)
    # 最远点采样（FPS）
    

    if num_points <= sample_size:
        fps_indices = torch.arange(num_points, device=targets.device)
    else:
        fps_indices = farthest_point_sampling(targets, sample_size)
    targets = targets[fps_indices]
    all_targets = targets.cpu().numpy()
    cluster_num = min(max_env_num, sample_size)
    # Perform KMeans clustering to find a representative center
    kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(all_targets)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    # Sort centers by their cluster size (number of points in each label), descending
    counts = np.bincount(labels, minlength=centers.shape[0])
    order = np.argsort(-counts)
    centers_target = []
    for i in range(centers.shape[0]):
        center = centers[i]
        dists = np.linalg.norm(all_targets - center, axis=1)
        nearest_idx = np.argmin(dists)
        centers_target.append(all_targets[nearest_idx])

    centers_target = np.array(centers_target)
    centers = centers_target[order]
    gridmap_cubemaps = []
    for j in range(min(opt.num_gridmap_envmaps, cluster_num)):
        cubemap,_ = render_cubemap(gaussians, centers_target[j], pipe, dataset, device="cuda")
        gridmap_cubemaps.append((cubemap,centers[j]))
    return gridmap_cubemaps

