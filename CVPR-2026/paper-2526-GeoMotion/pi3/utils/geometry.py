import numpy as np
import torch
import torch.nn.functional as F
import cv2
import os
from typing import List, Dict, Tuple
from collections import deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def se3_inverse(T):
    """
    Computes the inverse of a batch of SE(3) matrices.
    T: Tensor of shape (B, 4, 4)
    """
    if len(T.shape) == 2:
        T = T[None]
        unseq_flag = True
    else:
        unseq_flag = False

    if torch.is_tensor(T):
        R = T[:, :3, :3]
        t = T[:, :3, 3].unsqueeze(-1)
        R_inv = R.transpose(-2, -1)
        t_inv = -torch.matmul(R_inv, t)
        T_inv = torch.cat([
            torch.cat([R_inv, t_inv], dim=-1),
            torch.tensor([0, 0, 0, 1], device=T.device, dtype=T.dtype).repeat(T.shape[0], 1, 1)
        ], dim=1)
    else:
        R = T[:, :3, :3]
        t = T[:, :3, 3, np.newaxis]

        R_inv = np.swapaxes(R, -2, -1)
        t_inv = -R_inv @ t

        bottom_row = np.zeros((T.shape[0], 1, 4), dtype=T.dtype)
        bottom_row[:, :, 3] = 1

        top_part = np.concatenate([R_inv, t_inv], axis=-1)
        T_inv = np.concatenate([top_part, bottom_row], axis=1)

    if unseq_flag:
        T_inv = T_inv[0]
    return T_inv

def get_pixel(H, W):
    # get 2D pixels (u, v) for image_a in cam_a pixel space
    u_a, v_a = np.meshgrid(np.arange(W), np.arange(H))
    # u_a = np.flip(u_a, axis=1)
    # v_a = np.flip(v_a, axis=0)
    pixels_a = np.stack([
        u_a.flatten() + 0.5, 
        v_a.flatten() + 0.5, 
        np.ones_like(u_a.flatten())
    ], axis=0)
    
    return pixels_a

def depthmap_to_absolute_camera_coordinates(depthmap, camera_intrinsics, camera_pose, z_far=0, **kw):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
        - camera_pose: a 4x3 or 4x4 cam2world matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels."""
    X_cam, valid_mask = depthmap_to_camera_coordinates(depthmap, camera_intrinsics)
    if z_far > 0:
        valid_mask = valid_mask & (depthmap < z_far)

    X_world = X_cam # default
    if camera_pose is not None:
        # R_cam2world = np.float32(camera_params["R_cam2world"])
        # t_cam2world = np.float32(camera_params["t_cam2world"]).squeeze()
        R_cam2world = camera_pose[:3, :3]
        t_cam2world = camera_pose[:3, 3]

        # Express in absolute coordinates (invalid depth values)
        X_world = np.einsum("ik, vuk -> vui", R_cam2world, X_cam) + t_cam2world[None, None, :]

    return X_world, valid_mask


def depthmap_to_camera_coordinates(depthmap, camera_intrinsics, pseudo_focal=None):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels.
    """
    camera_intrinsics = np.float32(camera_intrinsics)
    H, W = depthmap.shape

    # Compute 3D ray associated with each pixel
    # Strong assumption: there are no skew terms
    # assert camera_intrinsics[0, 1] == 0.0
    # assert camera_intrinsics[1, 0] == 0.0
    if pseudo_focal is None:
        fu = camera_intrinsics[0, 0]
        fv = camera_intrinsics[1, 1]
    else:
        assert pseudo_focal.shape == (H, W)
        fu = fv = pseudo_focal
    cu = camera_intrinsics[0, 2]
    cv = camera_intrinsics[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z_cam = depthmap
    x_cam = (u - cu) * z_cam / fu
    y_cam = (v - cv) * z_cam / fv
    X_cam = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    # Mask for valid coordinates
    valid_mask = (depthmap > 0.0)
    # Invalid any depth > 80m
    valid_mask = valid_mask
    return X_cam, valid_mask

def homogenize_points(
    points,
):
    """Convert batched points (xyz) to (xyz1)."""
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def get_gt_warp(depth1, depth2, T_1to2, K1, K2, depth_interpolation_mode = 'bilinear', relative_depth_error_threshold = 0.05, H = None, W = None):
    
    if H is None:
        B,H,W = depth1.shape
    else:
        B = depth1.shape[0]
    with torch.no_grad():
        x1_n = torch.meshgrid(
            *[
                torch.linspace(
                    -1 + 1 / n, 1 - 1 / n, n, device=depth1.device
                )
                for n in (B, H, W)
            ],
            indexing = 'ij'
        )
        x1_n = torch.stack((x1_n[2], x1_n[1]), dim=-1).reshape(B, H * W, 2)
        mask, x2 = warp_kpts(
            x1_n.double(),
            depth1.double(),
            depth2.double(),
            T_1to2.double(),
            K1.double(),
            K2.double(),
            depth_interpolation_mode = depth_interpolation_mode,
            relative_depth_error_threshold = relative_depth_error_threshold,
        )
        prob = mask.float().reshape(B, H, W)
        x2 = x2.reshape(B, H, W, 2)
        return x2, prob

@torch.no_grad()
def warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1, smooth_mask = False, return_relative_depth_error = False, depth_interpolation_mode = "bilinear", relative_depth_error_threshold = 0.05):
    """Warp kpts0 from I0 to I1 with depth, K and Rt
    Also check covisibility and depth consistency.
    Depth is consistent if relative error < 0.2 (hard-coded).
    # https://github.com/zju3dv/LoFTR/blob/94e98b695be18acb43d5d3250f52226a8e36f839/src/loftr/utils/geometry.py adapted from here
    Args:
        kpts0 (torch.Tensor): [N, L, 2] - <x, y>, should be normalized in (-1,1)
        depth0 (torch.Tensor): [N, H, W],
        depth1 (torch.Tensor): [N, H, W],
        T_0to1 (torch.Tensor): [N, 3, 4],
        K0 (torch.Tensor): [N, 3, 3],
        K1 (torch.Tensor): [N, 3, 3],
    Returns:
        calculable_mask (torch.Tensor): [N, L]
        warped_keypoints0 (torch.Tensor): [N, L, 2] <x0_hat, y1_hat>
    """
    (
        n,
        h,
        w,
    ) = depth0.shape
    if depth_interpolation_mode == "combined":
        # Inspired by approach in inloc, try to fill holes from bilinear interpolation by nearest neighbour interpolation
        if smooth_mask:
            raise NotImplementedError("Combined bilinear and NN warp not implemented")
        valid_bilinear, warp_bilinear = warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1, 
                  smooth_mask = smooth_mask, 
                  return_relative_depth_error = return_relative_depth_error, 
                  depth_interpolation_mode = "bilinear",
                  relative_depth_error_threshold = relative_depth_error_threshold)
        valid_nearest, warp_nearest = warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1, 
                  smooth_mask = smooth_mask, 
                  return_relative_depth_error = return_relative_depth_error, 
                  depth_interpolation_mode = "nearest-exact",
                  relative_depth_error_threshold = relative_depth_error_threshold)
        nearest_valid_bilinear_invalid = (~valid_bilinear).logical_and(valid_nearest) 
        warp = warp_bilinear.clone()
        warp[nearest_valid_bilinear_invalid] = warp_nearest[nearest_valid_bilinear_invalid]
        valid = valid_bilinear | valid_nearest
        return valid, warp
        
        
    kpts0_depth = F.grid_sample(depth0[:, None], kpts0[:, :, None], mode = depth_interpolation_mode, align_corners=False)[
        :, 0, :, 0
    ]
    kpts0 = torch.stack(
        (w * (kpts0[..., 0] + 1) / 2, h * (kpts0[..., 1] + 1) / 2), dim=-1
    )  # [-1+1/h, 1-1/h] -> [0.5, h-0.5]
    # Sample depth, get calculable_mask on depth != 0
    # nonzero_mask = kpts0_depth != 0
    # Sample depth, get calculable_mask on depth > 0
    nonzero_mask = kpts0_depth > 0

    # Unproject
    kpts0_h = (
        torch.cat([kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1)
        * kpts0_depth[..., None]
    )  # (N, L, 3)
    kpts0_n = K0.inverse() @ kpts0_h.transpose(2, 1)  # (N, 3, L)
    kpts0_cam = kpts0_n

    # Rigid Transform
    w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]  # (N, 3, L)
    w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]

    # Project
    w_kpts0_h = (K1 @ w_kpts0_cam).transpose(2, 1)  # (N, L, 3)
    w_kpts0 = w_kpts0_h[:, :, :2] / (
        w_kpts0_h[:, :, [2]] + 1e-4
    )  # (N, L, 2), +1e-4 to avoid zero depth

    # Covisible Check
    h, w = depth1.shape[1:3]
    covisible_mask = (
        (w_kpts0[:, :, 0] > 0)
        * (w_kpts0[:, :, 0] < w - 1)
        * (w_kpts0[:, :, 1] > 0)
        * (w_kpts0[:, :, 1] < h - 1)
    )
    w_kpts0 = torch.stack(
        (2 * w_kpts0[..., 0] / w - 1, 2 * w_kpts0[..., 1] / h - 1), dim=-1
    )  # from [0.5,h-0.5] -> [-1+1/h, 1-1/h]
    # w_kpts0[~covisible_mask, :] = -5 # xd

    w_kpts0_depth = F.grid_sample(
        depth1[:, None], w_kpts0[:, :, None], mode=depth_interpolation_mode, align_corners=False
    )[:, 0, :, 0]
    
    relative_depth_error = (
        (w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth
    ).abs()
    if not smooth_mask:
        consistent_mask = relative_depth_error < relative_depth_error_threshold
    else:
        consistent_mask = (-relative_depth_error/smooth_mask).exp()
    valid_mask = nonzero_mask * covisible_mask * consistent_mask
    if return_relative_depth_error:
        return relative_depth_error, w_kpts0
    else:
        return valid_mask, w_kpts0


def geotrf(Trf, pts, ncol=None, norm=False):
    """ Apply a geometric transformation to a list of 3-D points.

    H: 3x3 or 4x4 projection matrix (typically a Homography)
    p: numpy/torch/tuple of coordinates. Shape must be (...,2) or (...,3)

    ncol: int. number of columns of the result (2 or 3)
    norm: float. if != 0, the resut is projected on the z=norm plane.

    Returns an array of projected 2d points.
    """
    assert Trf.ndim >= 2
    if isinstance(Trf, np.ndarray):
        pts = np.asarray(pts)
    elif isinstance(Trf, torch.Tensor):
        pts = torch.as_tensor(pts, dtype=Trf.dtype)

    # adapt shape if necessary
    output_reshape = pts.shape[:-1]
    ncol = ncol or pts.shape[-1]

    # optimized code
    if (isinstance(Trf, torch.Tensor) and isinstance(pts, torch.Tensor) and
            Trf.ndim == 3 and pts.ndim == 4):
        d = pts.shape[3]
        if Trf.shape[-1] == d:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf, pts)
        elif Trf.shape[-1] == d + 1:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf[:, :d, :d], pts) + Trf[:, None, None, :d, d]
        else:
            raise ValueError(f'bad shape, not ending with 3 or 4, for {pts.shape=}')
    else:
        if Trf.ndim >= 3:
            n = Trf.ndim - 2
            assert Trf.shape[:n] == pts.shape[:n], 'batch size does not match'
            Trf = Trf.reshape(-1, Trf.shape[-2], Trf.shape[-1])

            if pts.ndim > Trf.ndim:
                # Trf == (B,d,d) & pts == (B,H,W,d) --> (B, H*W, d)
                pts = pts.reshape(Trf.shape[0], -1, pts.shape[-1])
            elif pts.ndim == 2:
                # Trf == (B,d,d) & pts == (B,d) --> (B, 1, d)
                pts = pts[:, None, :]

        if pts.shape[-1] + 1 == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf[..., :-1, :] + Trf[..., -1:, :]
        elif pts.shape[-1] == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf
        else:
            pts = Trf @ pts.T
            if pts.ndim >= 2:
                pts = pts.swapaxes(-1, -2)

    if norm:
        pts = pts / pts[..., -1:]  # DONT DO /= BECAUSE OF WEIRD PYTORCH BUG
        if norm != 1:
            pts *= norm

    res = pts[..., :ncol].reshape(*output_reshape, ncol)
    return res


def inv(mat):
    """ Invert a torch or numpy matrix
    """
    if isinstance(mat, torch.Tensor):
        return torch.linalg.inv(mat)
    if isinstance(mat, np.ndarray):
        return np.linalg.inv(mat)
    raise ValueError(f'bad matrix type = {type(mat)}')

def opencv_camera_to_plucker(poses, K, H, W):
    device = poses.device
    B = poses.shape[0]

    pixel = torch.from_numpy(get_pixel(H, W).astype(np.float32)).to(device).T.reshape(H, W, 3)[None].repeat(B, 1, 1, 1)         # (3, H, W)
    pixel = torch.einsum('bij, bhwj -> bhwi', torch.inverse(K), pixel)
    ray_directions = torch.einsum('bij, bhwj -> bhwi', poses[..., :3, :3], pixel)

    ray_origins = poses[..., :3, 3][:, None, None].repeat(1, H, W, 1)

    ray_directions = ray_directions / ray_directions.norm(dim=-1, keepdim=True)
    plucker_normal = torch.cross(ray_origins, ray_directions, dim=-1)
    plucker_ray = torch.cat([ray_directions, plucker_normal], dim=-1)

    return plucker_ray


def depth_edge(depth: torch.Tensor, atol: float = None, rtol: float = None, kernel_size: int = 3, mask: torch.Tensor = None) -> torch.BoolTensor:
    """
    Compute the edge mask of a depth map. The edge is defined as the pixels whose neighbors have a large difference in depth.
    
    Args:
        depth (torch.Tensor): shape (..., height, width), linear depth map
        atol (float): absolute tolerance
        rtol (float): relative tolerance

    Returns:
        edge (torch.Tensor): shape (..., height, width) of dtype torch.bool
    """
    shape = depth.shape
    depth = depth.reshape(-1, 1, *shape[-2:])
    if mask is not None:
        mask = mask.reshape(-1, 1, *shape[-2:])

    if mask is None:
        diff = (F.max_pool2d(depth, kernel_size, stride=1, padding=kernel_size // 2) + F.max_pool2d(-depth, kernel_size, stride=1, padding=kernel_size // 2))
    else:
        diff = (F.max_pool2d(torch.where(mask, depth, -torch.inf), kernel_size, stride=1, padding=kernel_size // 2) + F.max_pool2d(torch.where(mask, -depth, -torch.inf), kernel_size, stride=1, padding=kernel_size // 2))

    edge = torch.zeros_like(depth, dtype=torch.bool)
    if atol is not None:
        edge |= diff > atol
    if rtol is not None:
        edge |= (diff / depth).nan_to_num_() > rtol
    edge = edge.reshape(*shape)
    return edge

def compute_relative_depth_weights(depth_map: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    """
    计算基于相对深度的权重，近处权重高，远处权重低
    
    Args:
        depth_map: 深度图 (H, W)
        alpha: 权重衰减参数，越大则远处权重衰减越快
    
    Returns:
        weights: 深度权重图 (H, W)
    """
    # 使用相对深度排序而不是绝对值
    depth_flat = depth_map.flatten()
    depth_percentiles = torch.quantile(depth_flat, torch.tensor([0.1, 0.5, 0.9]).to(depth_map.device))
    
    # 归一化到 [0, 1]，近处为0，远处为1
    depth_normalized = torch.clamp(
        (depth_map - depth_percentiles[0]) / (depth_percentiles[2] - depth_percentiles[0] + 1e-6),
        0, 1
    )
    
    # 计算权重：近处权重为1，远处权重接近0
    weights = torch.exp(-alpha * depth_normalized)
    
    return weights

def compute_motion_consistency_filter(
    motion_vectors: torch.Tensor,
    confidence: torch.Tensor,
    kernel_size: int = 5) -> torch.Tensor:
    """
    基于运动一致性过滤噪声，而不仅仅依赖置信度
    
    Args:
        motion_vectors: 运动向量 (H, W, 3)
        confidence: 置信度 (H, W)  
        kernel_size: 邻域大小
        
    Returns:
        consistency_mask: 运动一致性蒙版 (H, W)
    """
    H, W = motion_vectors.shape[:2]
    motion_magnitude = torch.norm(motion_vectors, dim=-1)
    
    # 使用空间滤波检测一致性运动区域
    motion_magnitude_padded = F.pad(motion_magnitude.unsqueeze(0).unsqueeze(0), 
                                  (kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2), 
                                  mode='reflect')
    
    # 计算局部方差 - 一致运动区域方差小
    unfold = F.unfold(motion_magnitude_padded, kernel_size, stride=1)
    local_patches = unfold.view(1, kernel_size*kernel_size, H, W)
    local_mean = local_patches.mean(dim=1)
    local_var = ((local_patches - local_mean.unsqueeze(1))**2).mean(dim=1)
    
    # 计算运动方向一致性
    motion_direction = F.normalize(motion_vectors, p=2, dim=-1)
    direction_padded = F.pad(motion_direction.permute(2, 0, 1).unsqueeze(0), 
                           (kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2), 
                           mode='reflect')
    
    direction_consistency = []
    for i in range(3):  # x, y, z components
        unfold_dir = F.unfold(direction_padded[:, i:i+1], kernel_size, stride=1)
        patches_dir = unfold_dir.view(1, kernel_size*kernel_size, H, W)
        dir_var = ((patches_dir - patches_dir.mean(dim=1, keepdim=True))**2).mean(dim=1)
        direction_consistency.append(dir_var)
    
    direction_consistency = torch.stack(direction_consistency, dim=0).mean(dim=0).squeeze()
    
    # 综合一致性评分：运动幅度方差小 + 方向一致性好 + 置信度适中
    magnitude_consistency = torch.exp(-local_var.squeeze() * 10)  # 方差越小一致性越好
    direction_consistency_score = torch.exp(-direction_consistency * 5)
    
    # 不过度依赖置信度，给运动区域更多机会
    confidence_adjusted = torch.sigmoid(confidence - 0.3)  # 降低置信度阈值
    
    # 综合评分
    consistency_score = (magnitude_consistency * 0.4 + 
                        direction_consistency_score * 0.4 + 
                        confidence_adjusted * 0.2)
    
    return consistency_score > 0.5

def compute_adaptive_motion_threshold(
    motion_magnitude: torch.Tensor,
    depth_weights: torch.Tensor,
    confidence_mask: torch.Tensor,
    percentile_low: float = 20,
    percentile_high: float = 80) -> float:
    """
    基于深度权重的自适应阈值计算
    """
    # 只考虑有效区域
    valid_mask = confidence_mask & (motion_magnitude > 0)
    if not valid_mask.any():
        return 0.01
    
    # 加权运动幅度
    weighted_motion = motion_magnitude[valid_mask] * depth_weights[valid_mask]
    
    if len(weighted_motion) < 10:
        return 0.01
    
    # 使用分位数确定背景运动和前景运动
    low_motion = torch.quantile(weighted_motion, percentile_low / 100.0)
    high_motion = torch.quantile(weighted_motion, percentile_high / 100.0)
    
    # 自适应阈值设在两者之间
    adaptive_threshold = low_motion + 0.3 * (high_motion - low_motion)
    
    return max(adaptive_threshold.item(), 0.005)  # 最小阈值保护

def apply_transform_to_points(points: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    """使用4x4位姿矩阵变换三维点"""
    ones = torch.ones(*points.shape[:-1], 1, device=points.device, dtype=points.dtype)
    points_homo = torch.cat([points, ones], dim=-1)
    transformed_homo = torch.matmul(pose, points_homo.unsqueeze(-1)).squeeze(-1)
    return transformed_homo[..., :3] / (transformed_homo[..., 3:4] + 1e-8)

def _compute_single_frame_motion_improved(
    ref_output: Dict[str, torch.Tensor],
    curr_output: Dict[str, torch.Tensor],
    motion_threshold: float,
    confidence_threshold: float,
    adaptive_threshold_factor: float,
    erosion_kernel_size: int,
    dilation_kernel_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    改进的单帧运动计算，解决相对深度和远处噪声问题
    """
    # 提取数据
    points1_world = ref_output['points'].squeeze(0).squeeze(0)
    points2_world = curr_output['points'].squeeze(0).squeeze(0)
    pose1_c2w = ref_output['camera_poses'].squeeze(0).squeeze(0)
    pose2_c2w = curr_output['camera_poses'].squeeze(0).squeeze(0)
    conf1 = ref_output['conf'].squeeze(0).squeeze(0).squeeze(-1)
    conf2 = curr_output['conf'].squeeze(0).squeeze(0).squeeze(-1)

    # 转换到相机坐标系
    pose1_w2c = torch.inverse(pose1_c2w)
    pose2_w2c = torch.inverse(pose2_c2w)
    points1_cam1 = apply_transform_to_points(points1_world, pose1_w2c)
    points2_cam2 = apply_transform_to_points(points2_world, pose2_w2c)

    # 计算相对变换和运动
    relative_pose = pose2_w2c @ pose1_c2w
    points1_in_cam2_rigid = apply_transform_to_points(points1_cam1, relative_pose)
    motion_vectors = points2_cam2 - points1_in_cam2_rigid
    motion_magnitude = torch.norm(motion_vectors, dim=-1)

    # 1. 计算相对深度权重（解决远处噪声问题）
    depth_map = points2_cam2[..., 2]
    depth_weights = compute_relative_depth_weights(depth_map, alpha=2.0)
    # import pdb;pdb.set_trace()
    
    # 2. 运动一致性过滤（不过度依赖置信度）
    confidence_mean = (torch.sigmoid(conf1) + torch.sigmoid(conf2)) / 2
    consistency_mask = compute_motion_consistency_filter(motion_vectors, confidence_mean)
    
    # 3. 基础置信度过滤（降低阈值，给运动区域机会）
    basic_confidence_mask = confidence_mean > 0.01  # 大幅降低置信度要求
    
    # 4. 综合有效区域
    valid_mask = basic_confidence_mask & consistency_mask
    
    # 5. 深度加权的自适应阈值
    adaptive_threshold = compute_adaptive_motion_threshold(
        motion_magnitude, depth_weights, valid_mask
    )
    
    final_threshold = max(adaptive_threshold * adaptive_threshold_factor, motion_threshold)
    
    # 6. 深度加权的运动检测
    weighted_motion = motion_magnitude * depth_weights
    motion_mask_raw = (weighted_motion > final_threshold) & valid_mask
    
    # 7. 近距离增强：对近处物体给予更多关注
    near_mask = depth_map < torch.quantile(depth_map[valid_mask], 0.3)  # 前30%近的区域
    near_motion_mask = (motion_magnitude > final_threshold * 0.5) & near_mask & basic_confidence_mask
    
    # 8. 合并近距离和常规检测结果
    motion_mask_raw = motion_mask_raw | near_motion_mask

     # --- 新增：按深度分位数把最远的 (1 - keep_percentile) 部分点直接屏蔽掉 ---
    keep_percentile = 0.8  # 保留最近的80%点
    try:
        if valid_mask.any():
            depth_thresh = torch.quantile(depth_map[valid_mask], keep_percentile)
        else:
            depth_thresh = torch.quantile(depth_map, keep_percentile)
    except Exception:
        # 以防 quantile 失败（例如样本太少），退回到全图分位数
        depth_thresh = torch.quantile(depth_map, keep_percentile)

    far_mask = depth_map > depth_thresh

    # 把远处点从最终运动 mask 中移除（无论之前是否被判为动）
    motion_mask_raw = motion_mask_raw & (~far_mask)
    motion_magnitude = motion_magnitude * (~far_mask).float()
    weighted_motion = weighted_motion * (~far_mask).float()
    
    # 9. 形态学后处理
    motion_mask_np = motion_mask_raw.cpu().numpy().astype(np.uint8)
    if erosion_kernel_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_kernel_size, erosion_kernel_size))
        motion_mask_np = cv2.erode(motion_mask_np, kernel, iterations=1)
    if dilation_kernel_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size))
        motion_mask_np = cv2.dilate(motion_mask_np, kernel, iterations=1)

    motion_mask = torch.from_numpy(motion_mask_np).bool().to(points1_world.device)
    
    # 输出深度加权的运动幅度用于可视化
    motion_magnitude_weighted = weighted_motion * valid_mask.float()

    return motion_mask, motion_magnitude_weighted

def _fuse_motion_results_improved(valid_comparisons: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    改进的运动结果融合，更注重近期帧的结果
    """
    if not valid_comparisons:
        return None, None
        
    device = valid_comparisons[0]['mask'].device
    output_shape = valid_comparisons[0]['mask'].shape
    
    # 加权投票融合
    vote_map = torch.zeros(output_shape, device=device, dtype=torch.float32)
    magnitude_sum = torch.zeros(output_shape, device=device, dtype=torch.float32)
    weight_sum = torch.zeros(output_shape, device=device, dtype=torch.float32)
    
    for comp in valid_comparisons:
        mask = comp['mask'].float()
        magnitude = comp['magnitude']
        weight = comp['weight']
        
        vote_map += mask * weight
        magnitude_sum += magnitude * weight  
        weight_sum += weight
    
    # 归一化
    weight_sum = torch.clamp(weight_sum, min=1e-8)
    final_vote = vote_map / weight_sum
    final_magnitude = magnitude_sum / weight_sum
    
    # 降低投票阈值，更容易检测到运动
    final_mask = final_vote > 0.3  # 从0.5降低到0.3
    
    return final_mask, final_magnitude

def compute_motion_mask_sliding_window_improved(
    pi3_outputs_window: List[Dict[str, torch.Tensor]],
    current_frame_idx: int,
    window_size: int = 5,
    motion_threshold: float = 0.01,  # 降低基础阈值
    confidence_threshold: float = 0.01,  # 大幅降低置信度要求
    overlap_threshold: float = 0.2,  # 降低重叠度要求
    adaptive_threshold_factor: float = 2.0,  # 降低自适应因子
    erosion_kernel_size: int = 2,  # 减小腐蚀核
    dilation_kernel_size: int = 4) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    改进的滑动窗口运动检测
    """
    
    if len(pi3_outputs_window) < 2:
        current_output = pi3_outputs_window[-1]
        points_shape = current_output['points'].squeeze().shape[:-1]
        empty_mask = torch.zeros(points_shape, dtype=torch.bool, device=current_output['points'].device)
        empty_magnitude = torch.zeros(points_shape, device=current_output['points'].device)
        return empty_mask, empty_magnitude, {'method': 'insufficient_frames'}
    
    current_output = pi3_outputs_window[-1]
    valid_comparisons = []
    debug_info = {'comparisons': [], 'method': 'sliding_window_improved'}
    
    # 与窗口内的每一帧进行比较
    for i, ref_output in enumerate(pi3_outputs_window[:-1]):
        ref_frame_idx = current_frame_idx - len(pi3_outputs_window) + 1 + i
        
        # 使用改进的运动计算
        mask, magnitude = _compute_single_frame_motion_improved(
            ref_output, current_output,
            motion_threshold, confidence_threshold,
            adaptive_threshold_factor,
            erosion_kernel_size, dilation_kernel_size
        )
        
        # 时间权重：优先考虑相邻帧
        time_weight = 1.0 / (len(pi3_outputs_window) - 1 - i)
        
        valid_comparisons.append({
            'mask': mask,
            'magnitude': magnitude,
            'weight': time_weight,
            'ref_frame_idx': ref_frame_idx
        })
        
        debug_info['comparisons'].append({
            'ref_frame_idx': ref_frame_idx,
            'weight': time_weight,
            'used': True
        })
    
    if not valid_comparisons:
        points_curr = current_output['points'].squeeze(0).squeeze(0)
        points_shape = points_curr.shape[:-1]
        empty_mask = torch.zeros(points_shape, dtype=torch.bool, device=points_curr.device)
        empty_magnitude = torch.zeros(points_shape, device=points_curr.device)
        return empty_mask, empty_magnitude, {'method': 'no_valid_comparisons'}
    
    # 融合多个比较结果
    final_mask, final_magnitude = _fuse_motion_results_improved(valid_comparisons)
    
    debug_info['num_valid_comparisons'] = len(valid_comparisons)
    
    return final_mask, final_magnitude, debug_info

# 改进的运动累积器
class ImprovedMotionAccumulator:
    """
    改进的运动累积器，考虑深度权重和运动一致性
    """
    def __init__(self, decay_factor: float = 0.9, min_accumulated_score: float = 0.1, 
                 depth_aware: bool = True):
        self.decay_factor = decay_factor
        self.min_accumulated_score = min_accumulated_score
        self.depth_aware = depth_aware
        self.accumulated_motion = None
        self.accumulated_weights = None
        self.frame_count = 0
    
    def update(self, motion_mask: torch.Tensor, motion_magnitude: torch.Tensor, 
               depth_weights: torch.Tensor = None) -> torch.Tensor:
        """
        更新累积运动蒙版，考虑深度权重
        """
        # 基础运动评分
        current_score = motion_mask.float() * torch.clamp(motion_magnitude, 0, 1)
        
        # 如果提供深度权重，进行加权
        if self.depth_aware and depth_weights is not None:
            current_score = current_score * depth_weights
            current_weight = depth_weights
        else:
            current_weight = torch.ones_like(current_score)
        
        if self.accumulated_motion is None:
            self.accumulated_motion = current_score.clone()
            self.accumulated_weights = current_weight.clone()
        else:
            # 加权累积
            self.accumulated_motion = (self.accumulated_motion * self.decay_factor + 
                                     current_score * current_weight)
            self.accumulated_weights = (self.accumulated_weights * self.decay_factor + 
                                      current_weight)
        
        self.frame_count += 1
        
        # 归一化累积分数
        normalized_score = self.accumulated_motion / (self.accumulated_weights + 1e-8)
        accumulated_mask = normalized_score > self.min_accumulated_score
        
        return accumulated_mask

def process_video_with_improved_sliding_window(
    pi3_outputs: Dict[str, torch.Tensor],
    window_size: int = 5,
    **motion_params) -> List[Dict]:
    """
    使用改进的滑动窗口策略处理整个视频
    """
    num_frames = pi3_outputs['points'].shape[0]
    motion_accumulator = ImprovedMotionAccumulator(
        decay_factor=motion_params.get('decay_factor', 0.9),
        min_accumulated_score=motion_params.get('min_accumulated_score', 0.1)
    )
    results = []
    
    # 滑动窗口
    window = deque(maxlen=window_size)
    
    for i in range(num_frames):
        # 构建当前帧的输出字典
        current_frame_output = {
            k: v[i].unsqueeze(0).unsqueeze(0) for k, v in pi3_outputs.items()
        }
        window.append(current_frame_output)
        
        # 计算运动蒙版
        motion_mask, motion_magnitude, debug_info = compute_motion_mask_sliding_window_improved(
            list(window), i, window_size, **{k: v for k, v in motion_params.items() 
                                           if k not in ['decay_factor', 'min_accumulated_score']}
        )
        
        # 计算当前帧的深度权重
        points_curr_world = current_frame_output['points'].squeeze(0).squeeze(0)
        pose_curr_c2w = current_frame_output['camera_poses'].squeeze(0).squeeze(0)

        # 转换到相机坐标系
        pose_curr_w2c = torch.inverse(pose_curr_c2w)
        points_curr_cam = apply_transform_to_points(points_curr_world, pose_curr_w2c)

        # 使用相机坐标系的Z值计算深度权重
        camera_depth_map = points_curr_cam[..., 2]
        depth_weights = compute_relative_depth_weights(camera_depth_map, alpha=2.0)

        # import pdb;pdb.set_trace()
        
        # 更新累积器
        accumulated_mask = motion_accumulator.update(motion_mask, motion_magnitude, depth_weights)
        
        results.append({
            'frame_idx': i,
            'motion_mask': motion_mask,
            'motion_magnitude': motion_magnitude,
            'accumulated_mask': accumulated_mask,
            'depth_weights': depth_weights,
            'debug_info': debug_info
        })
    
    return results

def visualize_motion_results_improved(
    image: np.ndarray,
    motion_mask: torch.Tensor,
    accumulated_mask: torch.Tensor,
    motion_magnitude: torch.Tensor,
    depth_weights: torch.Tensor,
    debug_info: Dict,
    save_path: str):
    """改进的可视化函数，包含深度权重信息"""
    if image.dtype != np.uint8:
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

    motion_mask_np = motion_mask.cpu().numpy().astype(np.uint8) * 255
    accumulated_mask_np = accumulated_mask.cpu().numpy().astype(np.uint8) * 255
    motion_magnitude_np = motion_magnitude.cpu().numpy()
    depth_weights_np = depth_weights.cpu().numpy()

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    frame_name = os.path.basename(save_path).split('.')[0]
    
    debug_text = f"Method: {debug_info.get('method', 'unknown')}"
    if 'num_valid_comparisons' in debug_info:
        debug_text += f", Comparisons: {debug_info['num_valid_comparisons']}"
    
    fig.suptitle(f"{frame_name} - {debug_text}", fontsize=14)

    # 第一行：原图、瞬时运动、累积运动、深度权重
    axes[0,0].imshow(image)
    axes[0,0].set_title('Original Image')
    axes[0,0].axis('off')

    # 瞬时运动蒙版叠加
    overlay_instant = np.zeros_like(image)
    overlay_instant[:, :, 0] = motion_mask_np
    instant_result = cv2.addWeighted(image, 0.7, overlay_instant, 0.6, 0)
    axes[0,1].imshow(instant_result)
    axes[0,1].set_title('Instant Motion (Red)')
    axes[0,1].axis('off')

    # 累积运动蒙版叠加
    overlay_accumulated = np.zeros_like(image)
    overlay_accumulated[:, :, 1] = accumulated_mask_np
    accumulated_result = cv2.addWeighted(image, 0.7, overlay_accumulated, 0.6, 0)
    axes[0,2].imshow(accumulated_result)
    axes[0,2].set_title('Accumulated Motion (Green)')
    axes[0,2].axis('off')

    # 深度权重可视化
    im_depth = axes[0,3].imshow(depth_weights_np, cmap='plasma', vmin=0, vmax=1)
    axes[0,3].set_title('Depth Weights')
    axes[0,3].axis('off')
    plt.colorbar(im_depth, ax=axes[0,3], shrink=0.7, label='Weight')

    # 'Motion Magnitude
    vmax = np.percentile(motion_magnitude_np[motion_magnitude_np > 0], 95) if np.any(motion_magnitude_np > 0) else 0.1
    im1 = axes[1,0].imshow(motion_magnitude_np, cmap='hot', vmin=0, vmax=vmax)
    axes[1,0].set_title('Motion Magnitude')
    axes[1,0].axis('off')
    plt.colorbar(im1, ax=axes[1,0], shrink=0.7, label='Motion')

    # 加权运动幅度
    weighted_motion = motion_magnitude_np * depth_weights_np
    vmax_weighted = np.percentile(weighted_motion[weighted_motion > 0], 95) if np.any(weighted_motion > 0) else 0.1
    im2 = axes[1,1].imshow(weighted_motion, cmap='hot', vmin=0, vmax=vmax_weighted)
    axes[1,1].set_title('Depth-Weighted Motion')
    axes[1,1].axis('off')
    plt.colorbar(im2, ax=axes[1,1], shrink=0.7, label='Weighted Motion')

    # 对比：运动区域 vs 深度权重
    overlay_comparison = image.copy()
    motion_areas = motion_mask_np > 0
    high_weight_areas = depth_weights_np > 0.5
    
    overlay_comparison[motion_areas] = [255, 0, 0]  # 红色：运动区域
    # overlay_comparison[high_weight_areas & ~motion_areas] = [0, 0, 255]  # 蓝色：高权重但无运动
    overlay_comparison[motion_areas & high_weight_areas] = [255, 255, 0]  # 黄色：运动+高权重
    
    axes[1,2].imshow(overlay_comparison)
    axes[1,2].set_title('Motion vs Depth\n(Red:Motion, Blue:HighWeight, Yellow:Both)')
    axes[1,2].axis('off')

    # 统计信息
    instant_motion_pixels = motion_mask.sum().item()
    accumulated_motion_pixels = accumulated_mask.sum().item()
    total_pixels = motion_mask.numel()
    avg_depth_weight = depth_weights.mean().item()
    
    stats_text = f"Instant Motion: {instant_motion_pixels} px ({instant_motion_pixels/total_pixels*100:.1f}%)\n"
    stats_text += f"Accumulated: {accumulated_motion_pixels} px ({accumulated_motion_pixels/total_pixels*100:.1f}%)\n"
    stats_text += f"Avg Motion: {motion_magnitude_np.mean():.4f}\n"
    stats_text += f"Avg Depth Weight: {avg_depth_weight:.3f}\n"
    stats_text += f"Motion in Near Field: {((motion_mask & (depth_weights > 0.5)).sum().item() / max(motion_mask.sum().item(), 1) * 100):.1f}%"
    
    axes[1,3].text(0.1, 0.7, stats_text, transform=axes[1,3].transAxes, 
                   fontsize=9, verticalalignment='top', fontfamily='monospace')
    axes[1,3].set_xlim(0, 1)
    axes[1,3].set_ylim(0, 1)
    axes[1,3].set_title('Statistics')
    axes[1,3].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def save_improved_summary_report(results: List[Dict], save_dir: str, motion_params: Dict):
    """保存改进算法的处理总结报告"""
    report_path = os.path.join(save_dir, "improved_processing_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("Improved Motion Detection Processing Report\n")
        f.write("="*60 + "\n\n")
        
        f.write("Algorithm Improvements:\n")
        f.write("- Relative depth-based weighting (reduces far-field noise)\n")
        f.write("- Motion consistency filtering (less reliance on confidence)\n")
        f.write("- Adaptive thresholding based on depth-weighted motion\n")
        f.write("- Near-field motion enhancement\n")
        f.write("- Improved motion accumulation with depth awareness\n\n")
        
        f.write("Parameters Used:\n")
        for key, value in motion_params.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        total_frames = len(results)
        frames_with_motion = sum(1 for r in results if r['motion_mask'].any())
        frames_with_accumulated = sum(1 for r in results if r['accumulated_mask'].any())
        
        # 深度权重统计
        avg_depth_weights = [r['depth_weights'].mean().item() for r in results]
        avg_depth_weight = np.mean(avg_depth_weights)
        
        # 近场运动统计
        near_field_motion_ratios = []
        for r in results:
            if r['motion_mask'].any():
                near_motion = (r['motion_mask'] & (r['depth_weights'] > 0.5)).sum().item()
                total_motion = r['motion_mask'].sum().item()
                ratio = near_motion / max(total_motion, 1)
                near_field_motion_ratios.append(ratio)
        
        avg_near_field_ratio = np.mean(near_field_motion_ratios) if near_field_motion_ratios else 0
        
        f.write(f"Processing Results:\n")
        f.write(f"  Total frames processed: {total_frames}\n")
        f.write(f"  Frames with instant motion: {frames_with_motion} ({frames_with_motion/total_frames*100:.1f}%)\n")
        f.write(f"  Frames with accumulated motion: {frames_with_accumulated} ({frames_with_accumulated/total_frames*100:.1f}%)\n")
        f.write(f"  Average depth weight: {avg_depth_weight:.3f}\n")
        f.write(f"  Average near-field motion ratio: {avg_near_field_ratio:.3f}\n\n")
        
        f.write("Quality Metrics:\n")
        motion_detection_rate = frames_with_motion / total_frames
        if motion_detection_rate < 0.1:
            f.write("  - Very sparse motion detected (may indicate static scene or high thresholds)\n")
        elif motion_detection_rate > 0.8:
            f.write("  - Very frequent motion detected (may indicate noise or low thresholds)\n")
        else:
            f.write("  - Reasonable motion detection rate\n")
            
        if avg_near_field_ratio > 0.6:
            f.write("  - Good near-field focus (motion primarily detected in foreground)\n")
        else:
            f.write("  - Motion spread across depth ranges (check for far-field noise)\n")
        
        f.write("\nPer-frame details:\n")
        f.write("-" * 80 + "\n")
        f.write("Frame | Method              | Comps | Instant | Accum   | DepthW | NearRatio\n")
        f.write("-" * 80 + "\n")
        
        for i, result in enumerate(results):
            debug = result['debug_info']
            instant_pixels = result['motion_mask'].sum().item()
            accum_pixels = result['accumulated_mask'].sum().item()
            depth_w = result['depth_weights'].mean().item()
            
            # 计算近场比例
            if instant_pixels > 0:
                near_pixels = (result['motion_mask'] & (result['depth_weights'] > 0.5)).sum().item()
                near_ratio = near_pixels / instant_pixels
            else:
                near_ratio = 0
            
            f.write(f"{i:5d} | {debug.get('method', 'unknown')[:18]:18s} | "
                   f"{debug.get('num_valid_comparisons', 0):5d} | "
                   f"{instant_pixels:7d} | {accum_pixels:7d} | "
                   f"{depth_w:.3f}  | {near_ratio:.3f}\n")


import torch
import torch.nn.functional as F
from typing import Tuple
import torch.nn as nn
import torch.optim as optim

def project_trajectory(xyz, K, T_cw):
    """
    Project world trajectory to 2D using K and Pose (T_cw).
    Robustly handles various input shapes.
    
    Args:
        xyz: [B, T, N, 3] or [B, T, N, 3, 1]
        K:   [B, T, 3, 3] or [B, T, 1, 3, 3]
        T_cw:[B, T, 4, 4]
        
    Returns:
        uv:  [B, T, N, 2]
    """
    # 1. 维度标准化
    # 如果 xyz 已经是 [..., 3, 1] (由诊断脚本传入)，则不需要再 unsqueeze
    if xyz.shape[-1] == 1 and xyz.shape[-2] == 3:
        xyz_exp = xyz
        B, T, N = xyz.shape[:3]
    else:
        # [B, T, N, 3] -> [B, T, N, 3, 1]
        B, T, N = xyz.shape[:3]
        xyz_exp = xyz.unsqueeze(-1)

    # K: 确保是 [B, T, 1, 3, 3]
    if K.dim() == 4: # [B, T, 3, 3]
        K_exp = K.unsqueeze(2)
    elif K.dim() == 3: # [B, 3, 3]
        K_exp = K.unsqueeze(1).unsqueeze(1)
    else:
        K_exp = K # Assume already [B, T, 1, 3, 3]

    # Pose: [B, T, 4, 4]
    if T_cw.dim() == 3: # [T, 4, 4] -> [1, T, 4, 4]
        T_cw = T_cw.unsqueeze(0)
        
    # 2. Pose Inversion (T_cw -> T_wc)
    R_cw = T_cw[..., :3, :3] # [B, T, 3, 3]
    t_cw = T_cw[..., :3, 3]  # [B, T, 3]
    
    R_wc = R_cw.transpose(-1, -2) # [B, T, 3, 3]
    t_wc = -torch.matmul(R_wc, t_cw.unsqueeze(-1)) # [B, T, 3, 1]
    
    # 3. World -> Camera
    # X_cam = R_wc * X_world + t_wc
    
    # Broadcast R_wc over N: [B, T, 3, 3] -> [B, T, 1, 3, 3]
    R_exp = R_wc.unsqueeze(2)
    # Broadcast t_wc over N: [B, T, 3, 1] -> [B, T, 1, 3, 1]
    t_exp = t_wc.unsqueeze(2)
    
    # [B, T, 1, 3, 3] @ [B, T, N, 3, 1] -> [B, T, N, 3, 1]
    xyz_cam = torch.matmul(R_exp, xyz_exp) + t_exp
    
    # 4. Camera -> Image
    # uvz = K * X_cam
    # [B, T, 1, 3, 3] @ [B, T, N, 3, 1] -> [B, T, N, 3, 1]
    uvz = torch.matmul(K_exp, xyz_cam)
    uvz = uvz.squeeze(-1) # [B, T, N, 3]
    
    z = uvz[..., 2:3]
    # Avoid div by zero
    z_safe = torch.where(z.abs() < 1e-4, torch.ones_like(z) * 1e-4, z)
    
    uv = uvz[..., :2] / z_safe
    return uv


def solve_focal_only(points: torch.Tensor,
                     mask: torch.Tensor = None) -> torch.Tensor:
    """
    Robustly estimate focal length (in pixels) assuming Shift=0,
    then enforce *temporal sharing* of intrinsics within each sequence.

    基本假设：
      - points: [B, T, H, W, 3] 是相机坐标系下的 3D 点 (X, Y, Z)
      - T 维是时间，我们假设同一序列在时间上共享同一个 focal
      - 返回 focals: [B, T]，但对于每个 batch，时间维度上的值是常数

    数学模型：
      u = f * X / Z
      v = f * Y / Z
      => 对所有点做最小二乘，求出标量 f

    Args:
        points: [B, T, H, W, 3] in Camera Coordinate
        mask:   [B, T, H, W]，bool，True 表示该像素有效；如果为 None，则全为 True

    Returns:
        focals: [B, T]，每个 batch 在时间维度上共享同一个 f
    """
    B, T, H, W, _ = points.shape
    device = points.device
    dtype = points.dtype

    # 1. 构建以图像中心为原点的 UV grid
    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij'
    )
    # 像素中心 + 居中
    u_grid = (x + 0.5) - W / 2.0
    v_grid = (y + 0.5) - H / 2.0

    # 展平成向量，后面用 valid_idx 索引
    u_flat = u_grid.flatten()  # [H*W]
    v_flat = v_grid.flatten()  # [H*W]

    # 2. mask 处理
    if mask is None:
        mask = torch.ones((B, T, H, W), dtype=torch.bool, device=device)

    focals_per_frame = []  # 存储每个 batch 的 [T] 向量

    # 3. 对每个 batch & 每个时间帧做一次最小二乘估计 f
    for b in range(int(B)):
        f_t_list = []
        for t in range(int(T)):
            b_idx = int(b)
            t_idx = int(t)

            # 当前帧的 mask 和点
            m = mask[b_idx, t_idx].reshape(-1)       # [H*W]
            pts = points[b_idx, t_idx].reshape(-1, 3)  # [H*W, 3]

            # 3.1 只保留 mask 内有效像素
            valid_idx = torch.nonzero(m, as_tuple=False).squeeze(1)  # [N_valid]
            
            X = pts[valid_idx, 0]
            Y = pts[valid_idx, 1]
            Z = pts[valid_idx, 2]

            # 与点对应的 UV
            U = u_flat[valid_idx]
            V = v_flat[valid_idx]

            valid_z = Z.abs() > 1e-3

            X, Y, Z, U, V = X[valid_z], Y[valid_z], Z[valid_z], U[valid_z], V[valid_z]

            # 3.4 最小二乘：f * [X; Y] ≈ [U*Z; V*Z]
            # numerator = Σ (X * U * Z + Y * V * Z)
            # denominator = Σ (X^2 + Y^2)
            numerator = (X * U * Z).sum() + (Y * V * Z).sum()
            denominator = (X * X).sum() + (Y * Y).sum()

            f = numerator / (denominator + 1e-6)

            f_t_list.append(f)

        focals_per_frame.append(torch.stack(f_t_list, dim=0))  # [T]

    # 4. 得到 per-frame 的估计结果：focals_raw [B, T]
    focals_raw = torch.stack(focals_per_frame, dim=0)  # [B, T]

    # 5. 在时间维度上做鲁棒聚合：每个 batch 共享一个 f（时间不变）
    #    使用 median + MAD 做离群值过滤，然后对 inlier 做平均
    with torch.no_grad():
        # 5.1 时间上的中位数
        med = focals_raw.median(dim=1, keepdim=True).values  # [B, 1]

        # 5.2 绝对偏差
        abs_dev = (focals_raw - med).abs()  # [B, T]

        # 5.3 MAD（median absolute deviation）
        mad = abs_dev.median(dim=1, keepdim=True).values + 1e-6  # [B, 1]

        # 5.4 设定一个阈值系数 k，用于判断离群值
        k = 3.0
        inlier_mask = abs_dev <= (k * mad)  # [B, T]，True 表示 inlier

        # 如果某个 batch 全被判成 outlier，就退回全 True
        all_outlier = ~inlier_mask.any(dim=1, keepdim=True)  # [B, 1]
        if all_outlier.any():
            inlier_mask = torch.where(
                all_outlier,
                torch.ones_like(inlier_mask, dtype=torch.bool),
                inlier_mask
            )

        # 5.5 只在 inlier 上做平均，得到每个 batch 的全局 f
        focals_masked = focals_raw * inlier_mask  # [B, T]
        num_inliers = inlier_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
        f_global = focals_masked.sum(dim=1, keepdim=True) / num_inliers  # [B, 1]

    # 6. 将每个 batch 的 f_global 拓展到时间维度，得到最终 [B, T]
    focals_shared = f_global.expand(-1, T)  # [B, T]

    return focals_shared


def intrinsics_from_focal_center(fx, fy, cx_norm, cy_norm, H, W):
    """
    Construct [B, T, 3, 3] K matrix.
    Args:
        fx, fy: [B, T] Pixel focal length (e.g. 518.0)
        cx_norm, cy_norm: [B, T] or scalar, Relative center (usually 0.5)
        H, W: Image dimensions (scalars)
    """
    B, T = fx.shape
    device = fx.device
    
    K = torch.zeros((B, T, 3, 3), dtype=torch.float32, device=device)
    
    # === FIX: Do not multiply by W/H again if fx/fy are already in pixels ===

    K[:, :, 0, 0] = fx 
    K[:, :, 1, 1] = fy 
    
    # Principal point is typically defined relative to image size
    K[:, :, 0, 2] = cx_norm * W
    K[:, :, 1, 2] = cy_norm * H
    K[:, :, 2, 2] = 1.0
    
    return K



def recover_intrinsics_and_shift_global(points: torch.Tensor, mask: torch.Tensor = None, num_samples: int = 8192) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Global Robust Solver for Focal Length AND Z-Shift.
    Replaces scipy.optimize.least_squares with Batch Linear Least Squares on GPU.
    
    Solves the linear equation system derived from:
        x = (u - cx) * (z + s) / f_norm
    Rearranged to linear form A[f, s]^T = b:
        x * f_norm - u * s = u * z
    
    Args:
        points: [B, T, H, W, 3] Camera-frame points (x, y, z)
        mask:   [B, T, H, W] Boolean valid mask
        num_samples: Subsample count for speed
        
    Returns:
        focal_pixels: [B] Focal length in PIXELS
        shift:        [B] Z-shift value
    """
    B, T, H, W, _ = points.shape
    device = points.device
    
    # 1. Coordinate Grid (Centered, Normalized by Width)
    # We use Width-normalization so f_norm = f_pix / W
    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=points.dtype),
        torch.arange(W, device=device, dtype=points.dtype),
        indexing='ij'
    )
    # u, v range approx [-0.5, 0.5]
    u = (x + 0.5) / W - 0.5
    v = (y + 0.5) / H - 0.5 # We normalize v by H here? No, let's normalize by W to keep f consistent.
    # To keep single f_norm, we typically normalize both by W (assuming square pixels)
    v = (y + 0.5) / W - 0.5 * (H / W) 

    # Expand to T
    u = u.unsqueeze(0).expand(T, -1, -1)
    v = v.unsqueeze(0).expand(T, -1, -1)
    
    if mask is None:
        mask = torch.ones((B, T, H, W), dtype=torch.bool, device=device)
        
    focals_pixels = []
    shifts = []
    
    for b in range(B):
        # Flatten current batch
        m_b = mask[b]
        
        # Valid points check
        if m_b.sum() < 100:
            focals_pixels.append(torch.tensor(float(W), device=device)) 
            shifts.append(torch.tensor(0.0, device=device))
            continue
            
        # Sampling
        valid_indices = torch.nonzero(m_b, as_tuple=False)
        if valid_indices.shape[0] > num_samples:
            perm = torch.randperm(valid_indices.shape[0], device=device)[:num_samples]
            valid_indices = valid_indices[perm]
            
        ts, hs, ws = valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]
        
        X = points[b, ts, hs, ws, 0]
        Y = points[b, ts, hs, ws, 1]
        Z = points[b, ts, hs, ws, 2]
        u_val = u[ts, hs, ws]
        v_val = v[ts, hs, ws]
        
        # Filter Z close to 0
        valid_z = Z.abs() > 1e-4
        X, Y, Z, u_val, v_val = X[valid_z], Y[valid_z], Z[valid_z], u_val[valid_z], v_val[valid_z]
        
        # Linear System Construction:
        # Eq 1: X * f - u * s = u * Z
        # Eq 2: Y * f - v * s = v * Z
        
        # A matrix: [2N, 2]
        # Col 0: Coeff for f_norm (X or Y)
        # Col 1: Coeff for s      (-u or -v)
        A_x = torch.stack([X, -u_val], dim=1)
        A_y = torch.stack([Y, -v_val], dim=1)
        A = torch.cat([A_x, A_y], dim=0) 
        
        # b vector: [2N]
        b_x = u_val * Z
        b_y = v_val * Z
        b_vec = torch.cat([b_x, b_y], dim=0)
        
        # Solve Ax = b using Least Squares
        # solution = [f_norm, s]
        try:
            solution = torch.linalg.lstsq(A, b_vec).solution
            f_norm, s = solution[0], solution[1]
        except:
            f_norm = torch.tensor(1.0, device=device)
            s = torch.tensor(0.0, device=device)
        
        # Sanity Checks
        # Focal length cannot be negative or absurdly small/large
        # FOV 10 deg ~ 160 deg
        f_norm = torch.clamp(f_norm, 0.3, 5.0) 
        
        focals_pixels.append(f_norm * W) # Convert back to pixels
        shifts.append(s)
        
    return torch.stack(focals_pixels), torch.stack(shifts)


def recover_focal_shift(points: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Recover focal length and depth shift from Affine-invariant Point Map.
    Based on the equation: x = (u - cx) * (z + s) / f
    Rearranged: x * f - u * s = u * z
    
    Args:
        points: [B, H, W, 3] (x, y, z)
        mask:   [B, H, W] (bool) valid points
    Returns:
        focal: [B] (Relative to image height/width/diagonal, depending on formulation)
        shift: [B]
    """
    B, H, W, _ = points.shape
    device = points.device
    
    # 1. Generate Normalized UV Coordinates (centered at 0)
    # range [-1, 1] roughly, but strictly: u = (col + 0.5) / W - 0.5
    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=points.dtype),
        torch.arange(W, device=device, dtype=points.dtype),
        indexing='ij'
    )
    
    # Center relative to image center
    u = (x + 0.5) / W - 0.5
    v = (y + 0.5) / H - 0.5
    
    # Aspect ratio handling
    # Pi3 usually normalizes such that the larger dimension is 1 or similar.
    # Let's follow the MoGe convention implicitly by solving the linear system.
    
    # Expand batch
    u = u.unsqueeze(0).expand(B, -1, -1)
    v = v.unsqueeze(0).expand(B, -1, -1)
    
    X = points[..., 0]
    Y = points[..., 1]
    Z = points[..., 2]
    
    if mask is None:
        mask = torch.ones_like(X, dtype=torch.bool)
        
    # 2. Construct Linear System Ax = b
    # We want to solve for [f, s]
    # Eq 1: X*f - u*s = u*Z
    # Eq 2: Y*f - v*s = v*Z
    
    # Stack equations for valid pixels
    focals = []
    shifts = []
    
    for b in range(B):
        m = mask[b]
        if m.sum() < 10: # Fallback if too few valid points
            focals.append(torch.tensor(1.0, device=device))
            shifts.append(torch.tensor(0.0, device=device))
            continue
            
        # Flatten valid points
        X_valid = X[b][m]
        Y_valid = Y[b][m]
        Z_valid = Z[b][m]
        u_valid = u[b][m]
        v_valid = v[b][m]
        
        # A matrix: [N*2, 2]
        # col 0: coeff for f (X or Y)
        # col 1: coeff for s (-u or -v)
        A1 = torch.stack([X_valid, -u_valid], dim=1)
        A2 = torch.stack([Y_valid, -v_valid], dim=1)
        A = torch.cat([A1, A2], dim=0)
        
        # b vector: [N*2]
        # val: u*Z or v*Z
        b1 = u_valid * Z_valid
        b2 = v_valid * Z_valid
        b_vec = torch.cat([b1, b2], dim=0)
        
        # Solve: A^T A x = A^T b
        # [2, 2] @ [2] = [2]
        try:
            solution = torch.linalg.lstsq(A, b_vec).solution
            f = solution[0]
            s = solution[1]
        except:
            f = torch.tensor(1.0, device=device)
            s = torch.tensor(0.0, device=device)
            
        focals.append(f)
        shifts.append(s)
        
    return torch.stack(focals), torch.stack(shifts)

def solve_intrinsics_optim(
    points: torch.Tensor, 
    mask: torch.Tensor = None, 
    num_steps: int = 100, 
    lr: float = 0.1
):
    """
    Optimize camera intrinsic parameters (fx, fy, and shift) using gradient descent.
    This function estimates the intrinsic matrix K (with potentially different fx and fy)
    and a depth shift parameter for each batch element, given 3D points projected onto an image grid.
    It supports masked points and uses Adam optimizer with SmoothL1 loss.
    Args:
        points (torch.Tensor): Input 3D points of shape (B, T, H, W, 3), where
            B = batch size,
            T = time steps,
            H = image height,
            W = image width,
            3 = (X, Y, Z) coordinates.
        mask (torch.Tensor, optional): Boolean mask of shape (B, T, H, W) indicating valid points.
            If None, all points are considered valid. Default: None.
        num_steps (int, optional): Number of optimization steps for each batch element. Default: 100.
        lr (float, optional): Learning rate for the Adam optimizer. Default: 0.1.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - Ks: Intrinsic matrices of shape (B, T, 3, 3) for each batch and time step.
            - shifts: Depth shift values of shape (B,) for each batch element.
    Notes:
        - The function assumes the principal point is at the image center.
        - If there are too few valid points (<100), a fallback K is used.
        - For efficiency, at most 4096 valid points are sampled per batch.
        - The optimization is performed independently for each batch element.
    """
    B, T, H, W, _ = points.shape
    device = points.device

    # 1. 预计算 UV 网格 (扩展到 Time 维度)
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    u_target = ((x + 0.5) / W - 0.5).float()
    v_target = ((y + 0.5) / W - 0.5 * (H/W)).float()
    u_target = u_target.unsqueeze(0).expand(T, -1, -1).reshape(-1)
    v_target = v_target.unsqueeze(0).expand(T, -1, -1).reshape(-1)

    Ks = []
    shifts = []

    for b in range(B):
        # Flatten current batch
        pts = points[b].reshape(-1, 3) 
        m = mask[b].reshape(-1) if mask is not None else torch.ones(pts.shape[0], dtype=torch.bool, device=device)
        
        # 生成有效索引
        valid_idx = torch.nonzero(m).squeeze()
        if valid_idx.numel() < 100:
            # Fallback
            K = torch.eye(3, device=device)
            K[0,0]=W; K[1,1]=W; K[0,2]=W/2; K[1,2]=H/2
            Ks.append(K.unsqueeze(0).expand(T, -1, -1))
            shifts.append(torch.tensor(0.0, device=device))
            continue
            
        if valid_idx.numel() > 4096:
            perm = torch.randperm(valid_idx.numel(), device=device)[:4096]
            indices = valid_idx[perm]
        else:
            indices = valid_idx
            
        pts_sub = pts[indices]
        u_gt = u_target[indices]
        v_gt = v_target[indices]
        
        # Filter Z near 0
        valid_z = pts_sub[:, 2].abs() > 1e-4
        pts_sub = pts_sub[valid_z]
        u_gt = u_gt[valid_z]
        v_gt = v_gt[valid_z]
        
        # 把数据从计算图中分离，当做常量
        X, Y, Z = pts_sub[:, 0].detach(), pts_sub[:, 1].detach(), pts_sub[:, 2].detach()

        # --- Optimization ---
        # 使用 enable_grad() 覆盖外部的 no_grad()
        with torch.enable_grad():
            # Init params
            f_x = torch.tensor(1.0, device=device, requires_grad=True)
            f_y = torch.tensor(1.0, device=device, requires_grad=True)
            s = torch.tensor(0.0, device=device, requires_grad=True)
            
            optimizer = optim.Adam([f_x, f_y, s], lr=lr)
            criterion = nn.SmoothL1Loss(beta=0.02)

            for _ in range(num_steps):
                optimizer.zero_grad()
                denom = Z + s
                denom = torch.where(denom.abs() < 1e-4, torch.sign(denom)*1e-4, denom)
                
                u_pred = f_x * X / denom
                v_pred = f_y * Y / denom
                
                loss = criterion(u_pred, u_gt) + criterion(v_pred, v_gt)
                
                reg = 0.0
                if f_x < 0.1: reg += (0.1 - f_x)**2
                if f_y < 0.1: reg += (0.1 - f_y)**2
                
                (loss + reg).backward()
                optimizer.step()
            
        # Collect results
        with torch.no_grad():
            fx_val = f_x.item() * W
            fy_val = f_y.item() * W
            s_val = s.item()
            
        K_mat = torch.eye(3, device=device)
        K_mat[0, 0] = fx_val
        K_mat[1, 1] = fy_val
        K_mat[0, 2] = W / 2.0
        K_mat[1, 2] = H / 2.0
        
        Ks.append(K_mat.unsqueeze(0).expand(T, -1, -1))
        shifts.append(torch.tensor(s_val, device=device))
        
    return torch.stack(Ks), torch.stack(shifts)

def recover_intrinsics_robust_ransac(points, mask=None, num_samples=8192, iterations=50, threshold=0.01):
    """
    RANSAC 鲁棒求解器 (支持视频序列维度 [B, T, ...])
    解决 Linear Least Squares 被动态物体干扰导致内参估计错误的问题。
    
    假设整个视频共享同一个焦距 f 和 偏移 s。
    
    Args:
        points: [B, T, H, W, 3] Local coordinates
        mask:   [B, T, H, W] Valid mask
        num_samples: 采样池大小 (从 T*H*W 个点中随机选这么多个点做 RANSAC)
        iterations: RANSAC 迭代次数
        threshold: Inlier 判定阈值 (Residual error)
        
    Returns:
        focal_pixels: [B]
        shift:        [B]
    """
    B, T, H, W, _ = points.shape
    device = points.device
    
    # 1. 预计算 UV 网格 (扩展到 Time 维度)
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    
    # 归一化 UV: 为了保持 f 统一，我们通常用宽度 W 归一化
    u_grid = (x + 0.5) / W - 0.5
    v_grid = (y + 0.5) / W - 0.5 * (H / W) # 注意这里除以 W 保持宽高比
    
    # [H, W] -> [T, H, W]
    u_grid = u_grid.unsqueeze(0).expand(T, -1, -1)
    v_grid = v_grid.unsqueeze(0).expand(T, -1, -1)
    
    if mask is None:
        mask = torch.ones((B, T, H, W), dtype=torch.bool, device=device)

    focals = []
    shifts = []

    for b in range(B):
        # 2. 展平当前 Batch 的所有帧
        # [T, H, W, 3] -> [N_total, 3]
        pts_flat = points[b].reshape(-1, 3)
        u_flat = u_grid.reshape(-1)
        v_flat = v_grid.reshape(-1)
        mask_flat = mask[b].reshape(-1)
        
        # 3. 筛选有效点并下采样 (Subsampling)
        # 我们不能对几百万个点跑 RANSAC，先随机采一个 pool (比如 8192 个)
        valid_indices = torch.nonzero(mask_flat).squeeze()
        
        if valid_indices.numel() < 100:
            # 兜底：点太少，返回默认值
            focals.append(torch.tensor(1.0 * W, device=device))
            shifts.append(torch.tensor(0.0, device=device))
            continue
            
        if valid_indices.numel() > num_samples:
            # 随机采样 num_samples 个点
            perm = torch.randperm(valid_indices.numel(), device=device)[:num_samples]
            indices = valid_indices[perm]
        else:
            indices = valid_indices
            
        # 提取采样池数据
        X = pts_flat[indices, 0]
        Y = pts_flat[indices, 1]
        Z = pts_flat[indices, 2]
        u = u_flat[indices]
        v = v_flat[indices]
        
        # 过滤 Z 接近 0 的点防止数值不稳定
        valid_z = Z.abs() > 1e-4
        X, Y, Z, u, v = X[valid_z], Y[valid_z], Z[valid_z], u[valid_z], v[valid_z]
        pool_size = X.shape[0]
        
        if pool_size < 10:
             focals.append(torch.tensor(1.0 * W, device=device))
             shifts.append(torch.tensor(0.0, device=device))
             continue

        # 4. 构建方程组数据 (A pool, b pool)
        # 方程: X*f - u*s = u*Z
        #       Y*f - v*s = v*Z
        # 为了高效，我们将 X 和 Y 方程拼在一起
        # A: [2*pool, 2]
        # b: [2*pool]
        
        A_x = torch.stack([X, -u], dim=1)
        A_y = torch.stack([Y, -v], dim=1)
        A_pool = torch.cat([A_x, A_y], dim=0) 
        b_pool = torch.cat([u * Z, v * Z], dim=0)
        
        best_inliers = -1
        # 默认值
        best_f = torch.tensor(1.0, device=device)
        best_s = torch.tensor(0.0, device=device)
        
        # --- RANSAC Loop ---
        for _ in range(iterations):
            # A. 随机选 4 个点 (Minimal Set)
            # indices 范围是 [0, pool_size)
            sample_idx = torch.randint(0, pool_size, (4,), device=device)
            
            # 从 pool 里取对应的 X方程和 Y方程
            # A_pool 的前 pool_size 行是 X方程，后 pool_size 行是 Y方程
            idx_rows = torch.cat([sample_idx, sample_idx + pool_size])
            
            A_curr = A_pool[idx_rows]
            b_curr = b_pool[idx_rows]
            
            # B. 最小二乘求解 f, s
            try:
                # lstsq or inverse
                sol = torch.linalg.lstsq(A_curr, b_curr).solution
                f_hyp, s_hyp = sol[0], sol[1]
            except RuntimeError:
                continue
            
            # 约束：焦距必须是正数且在合理范围内 (0.1 ~ 10.0 倍图宽)
            if f_hyp < 0.1 or f_hyp > 10.0:
                continue
                
            # C. 统计 Inliers
            # 计算所有点的残差 |Ax - b|
            pred_all = torch.matmul(A_pool, sol)
            diff = (pred_all - b_pool).abs()
            
            # 将 X 和 Y 的误差加起来 (对应同一个点)
            total_err = diff[:pool_size] + diff[pool_size:]
            
            inliers_count = (total_err < threshold).sum().item()
            
            if inliers_count > best_inliers:
                best_inliers = inliers_count
                best_f = f_hyp
                best_s = s_hyp
        
        # (可选) Refinement: 用所有 Inliers 再拟合一次
        # 这里直接用 RANSAC 最优解
        
        focals.append(best_f * W) # 恢复到像素单位
        shifts.append(best_s)
        
    return torch.stack(focals), torch.stack(shifts)


def project_points(xyz: torch.Tensor, K: torch.Tensor, T_cw: torch.Tensor) -> torch.Tensor:
    """
    Project world-space points to image UV using OpenCV pinhole model.

    Args:
        xyz:  [B, N, 3] or [B, T, N, 3]          (world coordinates)
        K:    [B, 3, 3] or [B, 1, 3, 3] or [B, T, 3, 3]  (intrinsics)
        T_cw: [B, 4, 4] or [B, T, 4, 4]          (Cam-to-World)

    Returns:
        uv:   [B, N, 2] or [B, T, N, 2]
    """

    # ---- 0) 清理数值 ----
    xyz  = torch.nan_to_num(xyz,  nan=0.0)
    K    = torch.nan_to_num(K,    nan=0.0)
    T_cw = torch.nan_to_num(T_cw, nan=0.0)

    # ---- 1) 时间维对齐（若有 T 维）----
    if xyz.ndim == 4 and T_cw.ndim == 4:
        T_xyz  = xyz.shape[1]
        T_pose = T_cw.shape[1]
        if T_pose != T_xyz:
            if T_pose > T_xyz:
                T_cw = T_cw[:, :T_xyz]
            else:
                raise ValueError(f"Pose length ({T_pose}) is shorter than point track length ({T_xyz}).")
    elif xyz.ndim == 3 and T_cw.ndim == 4:
        # 点是 [B,N,3]，外参是 [B,T,4,4] —— 这种组合通常不该出现
        # 如确有需要，可选择 T=1 的第一帧
        T_cw = T_cw[:, :1]  # -> [B,1,4,4]

    # ---- 2) Cam-to-World -> World-to-Cam ----
    try:
        T_wc = torch.linalg.inv(T_cw)
    except RuntimeError:
        T_wc = torch.linalg.pinv(T_cw)

    R = T_wc[..., :3, :3].contiguous()
    t = T_wc[..., :3,  3].contiguous()

    # ---- 3) 准备 K 的形状 ----
    if xyz.ndim == 3:
        # 目标：R [B,3,3], t [B,3], K [B,3,3]
        if R.ndim == 4:  # [B,1,3,3]
            R = R[:, 0]
            t = t[:, 0]
        if K.ndim == 4:
            # [B,1,3,3] 或 [B,T,3,3]（不建议）
            if K.shape[1] == 1:
                K = K[:, 0]
            else:
                # 若给了 [B,T,3,3]，取第 0 帧
                K = K[:, 0]
    else:
        # xyz: [B,T,N,3] -> 希望 R,K 都是 [B,T,3,3]，t 是 [B,T,3]
        if R.ndim == 3:  # [B,3,3] -> [B,1,3,3]
            R = R.unsqueeze(1)
            t = t.unsqueeze(1)
        if K.ndim == 3:              # [B,3,3] -> [B,1,3,3]
            K = K.unsqueeze(1)
        elif K.ndim == 4 and K.shape[1] > xyz.shape[1]:
            K = K[:, :xyz.shape[1]]  # 截齐时间长度

    # ---- 4) World -> Camera: X_c = X_w @ R^T + t ----
    if xyz.ndim == 3:
        # [B,N,3] @ [B,3,3]^T + [B,3]
        xyz_cam = torch.matmul(xyz, R.transpose(-1, -2)) + t.unsqueeze(1)
    else:
        # [B,T,N,3] @ [B,T,3,3]^T + [B,T,3]
        xyz_cam = torch.matmul(xyz, R.transpose(-1, -2)) + t.unsqueeze(2)

    # ---- 5) Camera -> Pixel (Homogeneous): u = (K @ X_c)_xy / z ----
    if xyz_cam.ndim == 3:
        # [B,N,3] @ [B,3,3]^T -> [B,N,3]
        uvz = torch.matmul(xyz_cam, K.transpose(-1, -2))
    else:
        # [B,T,N,3] @ [B,T,3,3]^T -> [B,T,N,3]
        uvz = torch.matmul(xyz_cam, K.transpose(-1, -2))

    z = uvz[..., 2:3]
    eps = 1e-3
    z_safe = torch.where(z.abs() < eps, torch.ones_like(z), z)
    uv = uvz[..., :2] / z_safe
    uv = torch.where(z.abs() < eps, torch.zeros_like(uv), uv)

    return uv


def apply_transform_to_cloud(xyz: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """
    Apply a rigid transform T (W2C) to world points:
      xyz: [B,T,N,3] or [B,N,3]
      T:   [B,1,4,4] or [B,T,4,4]
    Returns:
      transformed points: same shape as xyz
    """
    add_time = False
    if xyz.ndim == 3:  # [B,N,3] -> [B,1,N,3]
        xyz = xyz.unsqueeze(1)
        add_time = True

    B, Tt, N, _ = xyz.shape
    if T.ndim == 3:
        T = T.unsqueeze(1)  # [B,1,4,4]

    R = T[..., :3, :3]
    t = T[..., :3, 3]

    # x_cam = x_world @ R^T + t
    out = torch.matmul(xyz, R.transpose(-1, -2)) + t.unsqueeze(2)

    if add_time:
        out = out.squeeze(1)
    return out



def robust_umeyama_alignment(source, target, weights=None):
    """
    Weighted Sim3 Alignment: Source -> Target
    Find s, R, t such that: s * R * source + t ≈ target
    minimize sum(w * || target - (s * R * source + t) ||^2)
    
    Args:
        source: [B, N, 3] (Points to be transformed)
        target: [B, N, 3] (Reference/Ground Truth)
        weights:[B, N, 1]
    """
    B, N, C = source.shape
    device = source.device
    
    # 1. Weights
    if weights is None:
        weights = torch.ones(B, N, 1, device=device)
    w_sum = weights.sum(dim=1, keepdim=True) + 1e-8
    w_norm = weights / w_sum
    
    # 2. Centering
    mu_src = (source * w_norm).sum(dim=1, keepdim=True)
    mu_tgt = (target * w_norm).sum(dim=1, keepdim=True)
    
    src_c = source - mu_src
    tgt_c = target - mu_tgt
    
    # 3. Covariance
    # H = src_c^T * W * tgt_c
    # [B, 3, N] @ [B, N, 3]
    H = torch.matmul((src_c * w_norm).transpose(1, 2), tgt_c)
    
    # 4. SVD
    U, S, V = torch.svd(H.float()) # H = U S V.T
    
    # 5. Rotation R = V U.T
    # Note: Torch svd returns V not V.T
    R = torch.matmul(V, U.transpose(1, 2))
    
    # Reflection fix
    det = torch.det(R.float())
    mask = (det < 0).float().view(B, 1, 1)
    V_corr = V.clone()
    V_corr[:, :, 2] = V_corr[:, :, 2] * (1.0 - 2.0 * mask.squeeze(-1))
    R = torch.matmul(V_corr, U.transpose(1, 2))
    
    # 6. Scale s
    # s = tr(S) / sigma_src^2  (Simplified if R is handled)
    # Rigorous: s = sum(w * src_rot * tgt_c) / sum(w * src_c^2)
    
    src_rot = torch.matmul(src_c, R.transpose(1, 2)) # R @ src
    
    # Nom: sum(w * (R*src) * tgt)
    nom = (w_norm * src_rot * tgt_c).sum(dim=(1, 2))
    # Denom: sum(w * src^2) -- Source is in denominator!
    denom = (w_norm * src_c.pow(2)).sum(dim=(1, 2)) + 1e-8
    
    s = nom / denom
    s = s.view(B, 1, 1)
    
    # 7. Translation t
    # t = mu_tgt - s * R * mu_src
    t = mu_tgt - s * torch.matmul(mu_src, R.transpose(1, 2))
    
    # T matrix
    T_mat = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)
    T_mat[:, :3, :3] = R * s
    T_mat[:, :3, 3] = t.view(B, 3)
    
    return s, R, t, T_mat

def apply_sim3(points, s, R, t):
    """
    通用 Sim3 变换应用函数
    points: [B, ..., 3]
    """
    orig_shape = points.shape
    B = orig_shape[0]
    
    # Flatten non-batch dims
    pts_flat = points.view(B, -1, 3)
    
    # Rotate
    pts_rot = torch.matmul(pts_flat, R.transpose(1, 2))
    
    # Scale & Translate
    pts_out = s * pts_rot + t
    
    return pts_out.view(*orig_shape)

def sample_pi3_features(feature_map, uv, H, W):
    """
    从 Dense Map (Points or Conf) 中采样 UV 对应的值
    feature_map: [B, T, H, W, C]
    uv: [B, T, N, 2]
    Returns: [B, T, N, C]
    """
    B, T, H_map, W_map, C = feature_map.shape
    _, _, N, _ = uv.shape
    
    # Flatten Batch and Time
    feat_reshaped = feature_map.view(B*T, H_map, W_map, C).permute(0, 3, 1, 2) # [BT, C, H, W]
    uv_reshaped = uv.view(B*T, N, 2) # [BT, N, 2]
    
    # Normalize UV to [-1, 1]
    u = uv_reshaped[..., 0] / (W - 1) * 2 - 1
    v = uv_reshaped[..., 1] / (H - 1) * 2 - 1
    grid = torch.stack([u, v], dim=-1).unsqueeze(1) # [BT, 1, N, 2]
    
    # Grid Sample
    sampled = F.grid_sample(feat_reshaped, grid, align_corners=False, padding_mode='border')
    # [BT, C, 1, N] -> [BT, N, C]
    sampled = sampled.squeeze(2).transpose(1, 2)
    
    return sampled.view(B, T, N, C)


def robust_affine_alignment(pred_points, gt_points, weights=None, eps=1e-6):
    """
    计算加权仿射变换 (Batched & Regularized).
    Minimize: sum w * || (pred @ A.T + t) - gt ||^2
    
    Args:
        pred_points: [B, N, 3]
        gt_points:   [B, N, 3]
        weights:     [B, N, 1] or None. 
        eps:         正则化项，防止奇异矩阵导致 NaN
        
    Returns:
        A: [B, 3, 3] 线性变换矩阵 (包含旋转、缩放、剪切)
        t: [B, 1, 3] 平移向量
    """
    B, N, _ = pred_points.shape
    device = pred_points.device
    
    # 1. 权重处理
    if weights is None:
        weights = torch.ones(B, N, 1, device=device)
    
    # 归一化权重 (sum=1)
    w_sum = weights.sum(dim=1, keepdim=True) + 1e-8
    weights_norm = weights / w_sum
    
    # 2. 中心化 (Weighted Centering)
    # [B, 1, 3]
    mu_pred = (pred_points * weights_norm).sum(dim=1, keepdim=True)
    mu_gt   = (gt_points * weights_norm).sum(dim=1, keepdim=True)
    
    pred_c = pred_points - mu_pred
    gt_c   = gt_points - mu_gt
    
    # 3. 构建加权协方差矩阵
    # 目标: 求解 A，使得 pred_c @ A.T ≈ gt_c
    # 闭式解 (Ridge Regression): A.T = (X^T W X + eps*I)^-1 (X^T W Y)
    # 这里我们把 W 乘进 X 和 Y 里: X_w = sqrt(w) * X
    
    w_sqrt = torch.sqrt(weights_norm)
    X = pred_c * w_sqrt # [B, N, 3]
    Y = gt_c * w_sqrt   # [B, N, 3]
    
    # 计算协方差: X^T @ X -> [B, 3, 3]
    XTX = torch.matmul(X.transpose(1, 2), X)
    XTY = torch.matmul(X.transpose(1, 2), Y)
    
    # 添加正则化项 (Ridge) 保证可逆
    I = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1)
    XTX_reg = XTX + eps * I
    
    # 求解 A.T
    # 使用 solve 比 inv 更稳，但在 3x3 矩阵上 inv 也很快且支持 batch
    try:
        # A_T: [B, 3, 3]
        A_T = torch.matmul(torch.linalg.inv(XTX_reg), XTY)
    except RuntimeError:
        # Fallback (极少发生，因为加了 eps)
        A_T = torch.eye(3, device=device).expand(B, -1, -1)
        
    A = A_T.transpose(1, 2) # [B, 3, 3]
    
    # 4. 计算平移
    # t = mu_gt - mu_pred @ A.T
    # [B, 1, 3]
    t = mu_gt - torch.matmul(mu_pred, A_T)
    
    return A, t

def apply_affine(points, A, t):
    """
    Args:
        points: [B, N, 3] or [B, T, N, 3]
        A:      [B, 3, 3]
        t:      [B, 1, 3]
    """
    # 自动广播处理
    # 如果 points 是 [B, T, N, 3]，我们需要把 A, t 扩充维度
    if points.dim() == 4: 
        # points: [B, T, N, 3]
        # A: [B, 1, 3, 3]
        # t: [B, 1, 1, 3]
        out = torch.matmul(points, A.unsqueeze(1).transpose(-1, -2)) + t.unsqueeze(1)
    else:
        # points: [B, N, 3]
        out = torch.matmul(points, A.transpose(1, 2)) + t
        
    return out