"""
Data structures and utilities for Gaussian block management.

Defines the core data structures for representing Gaussian blocks,
their metadata, and scheduling information.
"""

import numpy as np
import torch
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class GaussianBlock:
    """
    Represents a block of Gaussian points.

    Attributes:
        block_id: Unique identifier for this block
        num_points: Number of Gaussian points in this block
        xyz: Positions (N, 3)
        scales: Scaling factors (N, 3)
        rotations: Quaternions (N, 4)
        opacity: Opacity values (N, 1)
        features_dc: DC component of spherical harmonics (N, 3)
        features_rest: Higher-order SH coefficients (N, 45) for degree 3
    """
    block_id: int
    num_points: int
    xyz: torch.Tensor
    scales: torch.Tensor
    rotations: torch.Tensor
    opacity: torch.Tensor
    features_dc: Optional[torch.Tensor] = None
    features_rest: Optional[torch.Tensor] = None

    def to_tensor(self) -> torch.Tensor:
        """
        Pack all attributes into a single flat tensor.

        Returns:
            Tensor of shape (num_points, point_dim) where point_dim = 59
            Layout: [xyz(3), scales(3), rotations(4), opacity(1), features_dc(3), features_rest(45)]
        """
        components = [
            self.xyz,           # 3
            self.scales,        # 3
            self.rotations,     # 4
            self.opacity,       # 1
        ]

        if self.features_dc is not None:
            components.append(self.features_dc)  # 3

        if self.features_rest is not None:
            components.append(self.features_rest)  # 45

        # Concatenate along feature dimension
        tensor = torch.cat(components, dim=-1)
        return tensor

    @staticmethod
    def from_tensor(block_id: int, tensor: torch.Tensor) -> 'GaussianBlock':
        """
        Unpack a flat tensor into GaussianBlock.

        Args:
            block_id: Block identifier
            tensor: Tensor of shape (num_points, point_dim)

        Returns:
            GaussianBlock instance
        """
        num_points = tensor.shape[0]

        # Parse tensor columns
        idx = 0
        xyz = tensor[:, idx:idx+3]
        idx += 3

        scales = tensor[:, idx:idx+3]
        idx += 3

        rotations = tensor[:, idx:idx+4]
        idx += 4

        opacity = tensor[:, idx:idx+1]
        idx += 1

        features_dc = None
        features_rest = None

        if tensor.shape[1] > idx:
            features_dc = tensor[:, idx:idx+3]
            idx += 3

        if tensor.shape[1] > idx:
            features_rest = tensor[:, idx:idx+45]

        return GaussianBlock(
            block_id=block_id,
            num_points=num_points,
            xyz=xyz,
            scales=scales,
            rotations=rotations,
            opacity=opacity,
            features_dc=features_dc,
            features_rest=features_rest
        )

    def to(self, device: torch.device) -> 'GaussianBlock':
        """Move all tensors to specified device."""
        return GaussianBlock(
            block_id=self.block_id,
            num_points=self.num_points,
            xyz=self.xyz.to(device),
            scales=self.scales.to(device),
            rotations=self.rotations.to(device),
            opacity=self.opacity.to(device),
            features_dc=self.features_dc.to(device) if self.features_dc is not None else None,
            features_rest=self.features_rest.to(device) if self.features_rest is not None else None,
        )


@dataclass
class BlockScheduleInfo:
    """
    Scheduling information for block loading order.

    Used by TSP scheduler to determine optimal loading sequence.
    """
    block_id: int
    camera_cluster_id: int  # Which camera cluster this block is associated with
    priority: float  # Higher = load earlier
    hotspot_score: float  # Visibility frequency across cameras
    morton_code: Optional[int] = None  # For spatial locality


# ==============================================================================
# Helper Functions for 6-Plane Frustum Culling
# ==============================================================================

def extract_frustum_planes(view_matrix: np.ndarray, proj_matrix: np.ndarray, 
                          coordinate_system: str = 'opengl', validate: bool = True) -> np.ndarray:
    """
    Extract 6 normalized frustum planes from View-Projection matrix.
    
    Uses the Gribb & Hartmann method to extract planes from the combined VP matrix.
    Each plane is normalized so that the distance calculation is Euclidean.
    
    Plane equation: Ax + By + Cz + D = 0
    Normalized: (A,B,C,D) / sqrt(A² + B² + C²)
    
    Args:
        view_matrix: 4x4 view matrix
        proj_matrix: 4x4 projection matrix
        coordinate_system: 'opengl' (default) or 'opencv'
        validate: If True, perform sanity checks on extracted planes
    
    Returns:
        (6, 4) array of plane coefficients [A, B, C, D]
        Order: Left, Right, Bottom, Top, Near, Far
    """
    # Handle coordinate system conversion if needed
    view_mat_converted = view_matrix.copy()
    if coordinate_system == 'opencv':
        # OpenCV → OpenGL: flip Y and Z axes
        flip = np.array([
            [1,  0,  0,  0],
            [0, -1,  0,  0],  # Y flip
            [0,  0, -1,  0],  # Z flip
            [0,  0,  0,  1]
        ], dtype=np.float32)
        view_mat_converted = flip @ view_matrix
    
    # Compute View-Projection matrix
    vp = proj_matrix @ view_mat_converted
    
    # Extract plane coefficients using Gribb & Hartmann method
    # Each plane is extracted by adding/subtracting rows of VP matrix
    planes = np.zeros((6, 4), dtype=np.float32)
    
    # Left plane: VP[3] + VP[0]
    planes[0] = vp[3, :] + vp[0, :]
    
    # Right plane: VP[3] - VP[0]
    planes[1] = vp[3, :] - vp[0, :]
    
    # Bottom plane: VP[3] + VP[1]
    planes[2] = vp[3, :] + vp[1, :]
    
    # Top plane: VP[3] - VP[1]
    planes[3] = vp[3, :] - vp[1, :]
    
    # Near plane: VP[3] + VP[2]
    planes[4] = vp[3, :] + vp[2, :]
    
    # Far plane: VP[3] - VP[2]
    planes[5] = vp[3, :] - vp[2, :]
    
    # Normalize each plane: divide by length of normal vector (A, B, C)
    for i in range(6):
        normal_length = np.sqrt(planes[i, 0]**2 + planes[i, 1]**2 + planes[i, 2]**2)
        if normal_length > 1e-8:
            planes[i] /= normal_length
        else:
            # Degenerate plane - should not happen with valid matrices
            raise ValueError(f"Degenerate frustum plane {i}: normal length = {normal_length}")
    
    # Validation: check plane orientations
    if validate:
        # Near plane normal should point into frustum (typically -Z direction in camera space)
        # Far plane normal should point out of frustum (typically +Z direction)
        near_plane_z = planes[4, 2]  # Near plane Z component
        far_plane_z = planes[5, 2]   # Far plane Z component
        
        # For OpenGL convention: near should be negative, far should be positive
        if coordinate_system == 'opengl':
            if near_plane_z > 0 or far_plane_z < 0:
                import warnings
                warnings.warn(
                    f"Suspicious plane orientations: near_z={near_plane_z:.3f}, far_z={far_plane_z:.3f}. "
                    "Expected near_z < 0 and far_z > 0 for OpenGL. Check projection matrix."
                )
    
    return planes


def compute_block_bounding_radius(block_size: int, point_extent: float = 1.0) -> float:
    """
    Compute bounding sphere radius for a block.
    
    For a cubic block, the radius is half the space diagonal.
    For non-uniform distributions, add a safety margin.
    
    Args:
        block_size: Number of points per block
        point_extent: Approximate spatial extent of one Gaussian point
    
    Returns:
        Bounding sphere radius
    """
    # Estimate block spatial size (assuming uniform distribution)
    # For MatrixCity: block_size=4096, scene ~50x60x5m
    # Conservative estimate: radius = block_size^(1/3) * point_extent * sqrt(3)/2
    
    # Simplified: assume cubic blocks with side length proportional to cube root
    side_length = (block_size ** (1.0/3.0)) * point_extent
    
    # Bounding sphere radius = half of space diagonal = side * sqrt(3) / 2
    radius = side_length * np.sqrt(3.0) / 2.0
    
    # Add 20% safety margin for non-uniform distributions
    radius *= 1.2
    
    return radius


# ==============================================================================
# 1. CPU 版本 (已优化向量化，比原版 for 循环快 100 倍)
# ==============================================================================
class FrustumCuller:
    """
    CPU-based Frustum culling with precise 6-plane test and bounding spheres.
    """

    def __init__(
        self,
        block_bounds: np.ndarray,
        scene_radius: float = None,
        block_size: int = 4096,
        verbose: bool = True,
    ):
        self.block_bounds = np.asarray(block_bounds, dtype=np.float32).copy()
        self.num_blocks = len(self.block_bounds)
        self.scene_radius = scene_radius
        self.block_size = block_size
        self.verbose = bool(verbose)
        
        # 预计算中心点 (N, 3)，避免每帧重复计算
        # 利用 Numpy 广播机制
        self.block_centers = (self.block_bounds[:, :3] + self.block_bounds[:, 3:]) / 2.0
        
        # ================================================================
        # [FIX] Compute PER-BLOCK bounding sphere radius
        # Using a uniform radius causes:
        #   - Large blocks: radius too small → miss visible blocks
        #   - Small blocks: radius too large → include invisible blocks
        # ================================================================
        diagonals = np.linalg.norm(self.block_bounds[:, 3:] - self.block_bounds[:, :3], axis=1)
        
        # Per-block radius with 15% safety margin
        self.block_radii = (diagonals / 2.0) * 1.15
        
        # Keep scalar radius for legacy compatibility (use max for safety)
        self.block_radius = float(np.max(self.block_radii))
        
        # Store statistics for debugging
        self.block_radius_min = float(np.min(diagonals) / 2.0)
        self.block_radius_max = float(np.max(diagonals) / 2.0)
        self.block_radius_median = float(np.median(diagonals) / 2.0)

        self._log(f"[FrustumCuller-CPU] Initialized with {self.num_blocks} blocks")
        self._log("[FrustumCuller-CPU] Using PER-BLOCK radius (more precise culling)")
        self._log(
            f"[FrustumCuller-CPU] Radius range: "
            f"[{self.block_radius_min:.3f}, {self.block_radius_max:.3f}], "
            f"median: {self.block_radius_median:.3f}"
        )

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def update_block_bounds(self, block_ids, block_bounds: np.ndarray) -> int:
        ids = np.asarray(block_ids, dtype=np.int64)
        if ids.size == 0:
            return 0

        bounds = np.asarray(block_bounds, dtype=np.float32)
        if bounds.shape != (ids.size, 6):
            raise ValueError(
                f"Expected block_bounds shape {(ids.size, 6)}, got {bounds.shape}"
            )

        mins = bounds[:, :3]
        maxs = bounds[:, 3:]
        centers = (mins + maxs) / 2.0
        radii = (np.linalg.norm(maxs - mins, axis=1) / 2.0) * 1.15

        self.block_bounds[ids] = bounds
        self.block_centers[ids] = centers
        self.block_radii[ids] = radii
        if radii.size:
            self.block_radius = max(float(self.block_radius), float(np.max(radii)))
        return int(ids.size)

    def cull(
        self,
        camera_position: np.ndarray,
        view_matrix: np.ndarray,
        projection_matrix: np.ndarray,
        margin: float = 1.0,
        use_6plane: bool = True
    ) -> List[int]:
        """
        Perform vectorized frustum culling.
        
        Args:
            camera_position: Camera position in world space (3,)
            view_matrix: 4x4 view matrix
            projection_matrix: 4x4 projection matrix
            margin: Conservative margin (not used in 6-plane mode)
            use_6plane: If True, use precise 6-plane test; else use legacy cone method
        
        Returns:
            List of visible block indices
        """
        if use_6plane:
            # ================================================================
            # NEW: Precise 6-Plane Frustum Culling with Bounding Spheres
            # ================================================================
            
            # Extract normalized frustum planes
            planes = extract_frustum_planes(view_matrix, projection_matrix)
            
            # Vectorized plane-sphere test for all blocks
            # For each plane, compute signed distance from all block centers
            # Shape: (N, 6) - signed distances to each plane
            
            # Expand planes for broadcasting: (6, 4) -> (1, 6, 4)
            planes_expanded = planes[np.newaxis, :, :]  # (1, 6, 4)
            
            # Expand block centers: (N, 3) -> (N, 1, 3) and add homogeneous coordinate
            centers_homo = np.concatenate([
                self.block_centers,
                np.ones((self.num_blocks, 1), dtype=np.float32)
            ], axis=1)  # (N, 4)
            centers_homo = centers_homo[:, np.newaxis, :]  # (N, 1, 4)
            
            # Compute signed distances: (N, 6)
            # signed_dist[i, j] = dot(planes[j], centers_homo[i])
            signed_distances = np.sum(planes_expanded * centers_homo, axis=2)  # (N, 6)
            
            # ================================================================
            # [FIX] Use per-block radius for precise culling
            # ================================================================
            # Culling condition: block is visible if ALL signed distances > -radius[i]
            # Compare each block's distances against its own radius
            visible_mask = np.all(signed_distances >= -self.block_radii[:, np.newaxis], axis=1)  # (N,)
            
            visible_blocks = np.where(visible_mask)[0].tolist()
            
            return visible_blocks
        
        else:
            # ================================================================
            # LEGACY: Approximate Cone + Distance Culling
            # ================================================================
            # 1. 设置阈值
            if self.scene_radius is not None:
                max_view_distance = min(self.scene_radius * 0.5, 100.0)
            else:
                max_view_distance = 50.0
            
            threshold_dist = max_view_distance * margin

            # 2. 获取相机方向 (-Z)
            camera_z_axis = view_matrix[2, :3]
            camera_forward = camera_z_axis
            camera_forward = camera_forward / (np.linalg.norm(camera_forward) + 1e-8)

            # -----------------------------------------------------------
            # Vectorized Calculation (不再使用 for 循环)
            # -----------------------------------------------------------
            
            # A. 计算向量 (N, 3)
            to_blocks = self.block_centers - camera_position

            # B. 计算距离 (N,)
            distances = np.linalg.norm(to_blocks, axis=1)

            # C. 计算点积 (N,)
            # 避免除以零
            with np.errstate(divide='ignore', invalid='ignore'):
                to_blocks_norm = to_blocks / (distances[:, np.newaxis] + 1e-8)
            
            dots = np.dot(to_blocks_norm, camera_forward)

            # -----------------------------------------------------------
            # Logic Masking
            # -----------------------------------------------------------
            
            # 距离剔除 & 角度剔除
            # dots > -0.3 意味着 FOV 大约 107度 (非常保守)
            mask = (distances <= threshold_dist) & ((dots > -0.3) | (distances < 1e-6))

            # 获取 True 的索引
            visible_blocks = np.where(mask)[0].tolist()

            return visible_blocks

    def get_visibility_order(self, visible_blocks: List[int], camera_position: np.ndarray) -> List[int]:
        """Sort visible blocks by distance (front-to-back)."""
        if not visible_blocks:
            return []
            
        # 向量化排序
        # 1. 提取可见块的中心点
        centers = self.block_centers[visible_blocks]
        
        # 2. 计算距离平方 (比开根号快，排序结果一样)
        diff = centers - camera_position
        dist_sq = np.sum(diff**2, axis=1)
        
        # 3. 获取排序后的索引
        # argsort 返回的是局部索引，需要映射回 visible_blocks
        sorted_local_indices = np.argsort(dist_sq)
        
        return [visible_blocks[i] for i in sorted_local_indices]


def compute_morton_code(position: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray) -> int:
    """
    Compute 3D Morton code (Z-order curve) for spatial locality.

    Args:
        position: 3D position (x, y, z)
        bbox_min: Minimum bounds of space
        bbox_max: Maximum bounds of space

    Returns:
        Morton code as integer
    """
    # Normalize to [0, 1]
    normalized = (position - bbox_min) / (bbox_max - bbox_min + 1e-8)
    normalized = np.clip(normalized, 0, 1)

    # Quantize to 10-bit integers (0-1023)
    quantized = (normalized * 1023).astype(np.uint32)

    # Interleave bits
    def split_by_3(x: int) -> int:
        """Spread bits by inserting two zeros between each bit."""
        x = x & 0x3FF  # Keep only 10 bits
        x = (x | x << 16) & 0x30000FF
        x = (x | x << 8) & 0x300F00F
        x = (x | x << 4) & 0x30C30C3
        x = (x | x << 2) & 0x9249249
        return x

    xx = split_by_3(quantized[0])
    yy = split_by_3(quantized[1])
    zz = split_by_3(quantized[2])

    return xx | (yy << 1) | (zz << 2)


def compute_block_bounds(xyz, block_size: int) -> np.ndarray:
    """
    Compute bounding boxes for each block.
    兼容 Tensor (GPU/CPU) 和 Numpy Array。
    """
    # [FIX] 统一转换为 Numpy 数组
    # 如果是 Tensor，就 .detach().cpu().numpy()
    # 如果已经是 Numpy，就直接用
    if isinstance(xyz, torch.Tensor):
        if xyz.is_cuda:
            xyz_cpu = xyz.detach().cpu().numpy()
        else:
            xyz_cpu = xyz.detach().numpy()
    elif isinstance(xyz, np.ndarray):
        xyz_cpu = xyz
    else:
        raise TypeError(f"Unsupported type for xyz: {type(xyz)}")

    total_points = xyz_cpu.shape[0]
    num_blocks = (total_points + block_size - 1) // block_size

    bounds = np.zeros((num_blocks, 6), dtype=np.float32)

    for block_id in range(num_blocks):
        start_idx = block_id * block_size
        end_idx = min(start_idx + block_size, total_points)

        # [FIX] 直接切片 Numpy 数组，不再调用 .cpu()
        block_xyz = xyz_cpu[start_idx:end_idx]

        bounds[block_id, :3] = block_xyz.min(axis=0)  # min
        bounds[block_id, 3:] = block_xyz.max(axis=0)  # max

    return bounds

def get_morton_code_torch(xyz: torch.Tensor, 
                          global_min: Optional[torch.Tensor] = None,
                          global_max: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    [GPU Optimized] 计算 3D Morton Code
    
    Args:
        xyz: 点云坐标 (N, 3)
        global_min: 全局最小值 (3,), 用于分块计算时保持一致的归一化
        global_max: 全局最大值 (3,), 用于分块计算时保持一致的归一化
    
    Returns:
        Morton codes (N,)
    """
    # 1. 归一化到 [0, 1]
    if global_min is None or global_max is None:
        _min = xyz.min(dim=0).values
        _max = xyz.max(dim=0).values
    else:
        _min = global_min
        _max = global_max
    
    normalized = (xyz - _min) / (_max - _min + 1e-8)
    
    # 2. 量化到 10-bit (0-1023)
    quantized = (normalized * 1023).long()
    
    # 3. 位交织 (Bit Interleaving)
    def spread_bits(x):
        x = (x | (x << 16)) & 0x030000FF
        x = (x | (x <<  8)) & 0x0300F00F
        x = (x | (x <<  4)) & 0x030C30C3
        x = (x | (x <<  2)) & 0x09249249
        return x

    xx = spread_bits(quantized[:, 0])
    yy = spread_bits(quantized[:, 1])
    zz = spread_bits(quantized[:, 2])
    
    return xx | (yy << 1) | (zz << 2)


def get_hierarchical_morton_code_torch(xyz: torch.Tensor, 
                                       global_min: Optional[torch.Tensor] = None,
                                       global_max: Optional[torch.Tensor] = None,
                                       grid_resolution: int = 32) -> torch.Tensor:
    """
    [IMPROVED] 分层 Morton Code - 解决稀疏场景的空间跳跃问题
    
    策略:
    1. 将场景划分为粗粒度网格 (如 32x32x32)
    2. 先按网格的 Morton Code 排序 (粗排序)
    3. 在每个网格内部再按精细 Morton Code 排序 (细排序)
    
    效果:
    - 避免跨越大片空白区域
    - 同一个 block 的点在空间上更加聚集
    
    Args:
        xyz: 点云坐标 (N, 3)
        global_min: 全局最小值 (3,), 用于分块计算时保持一致的归一化
        global_max: 全局最大值 (3,), 用于分块计算时保持一致的归一化
        grid_resolution: 粗网格分辨率 (默认 32, 即将场景分成 32x32x32 个格子)
    
    Returns:
        分层 Morton Code (N,)
    """
    # 1. 归一化到 [0, 1]
    if global_min is None or global_max is None:
        _min = xyz.min(dim=0).values
        _max = xyz.max(dim=0).values
    else:
        _min = global_min
        _max = global_max
    
    normalized = (xyz - _min) / (_max - _min + 1e-8)
    
    # 2. 粗网格索引 (grid_resolution bit)
    grid_quantized = (normalized * (grid_resolution - 1)).long().clamp(0, grid_resolution - 1)
    
    # 3. 细网格内的归一化坐标 (10 bit)
    grid_size = 1.0 / grid_resolution
    grid_min = grid_quantized.float() * grid_size
    fine_normalized = ((normalized - grid_min) / grid_size).clamp(0, 1)
    fine_quantized = (fine_normalized * 1023).long()
    
    # 4. 位交织函数
    def spread_bits_coarse(x, bits=5):  # 32 = 2^5, 需要 5 bits
        """粗网格 Morton (5 bits per dimension)"""
        x = x & 0x1F  # 保留 5 bits
        x = (x | (x << 8)) & 0x100F
        x = (x | (x << 4)) & 0x10C3
        x = (x | (x << 2)) & 0x1249
        return x
    
    def spread_bits_fine(x):  # 1024 = 2^10, 需要 10 bits
        """细网格 Morton (10 bits per dimension)"""
        x = (x | (x << 16)) & 0x030000FF
        x = (x | (x <<  8)) & 0x0300F00F
        x = (x | (x <<  4)) & 0x030C30C3
        x = (x | (x <<  2)) & 0x09249249
        return x
    
    # 5. 计算粗网格 Morton Code (高位)
    xx_coarse = spread_bits_coarse(grid_quantized[:, 0])
    yy_coarse = spread_bits_coarse(grid_quantized[:, 1])
    zz_coarse = spread_bits_coarse(grid_quantized[:, 2])
    coarse_morton = xx_coarse | (yy_coarse << 1) | (zz_coarse << 2)
    
    # 6. 计算细网格 Morton Code (低位)
    xx_fine = spread_bits_fine(fine_quantized[:, 0])
    yy_fine = spread_bits_fine(fine_quantized[:, 1])
    zz_fine = spread_bits_fine(fine_quantized[:, 2])
    fine_morton = xx_fine | (yy_fine << 1) | (zz_fine << 2)
    
    # 7. 组合: 高位 = 粗网格, 低位 = 细网格
    # coarse_morton: 15 bits (5*3), fine_morton: 30 bits (10*3)
    hierarchical_morton = (coarse_morton.long() << 30) | fine_morton.long()
    
    return hierarchical_morton

def compute_block_bounds_cpu(xyz_cpu: np.ndarray, block_size: int) -> np.ndarray:
    """
    [CPU Optimized] 基于 CPU 缓存计算 Block Bounds，零 PCIe 开销
    """
    total_points = xyz_cpu.shape[0]
    num_blocks = (total_points + block_size - 1) // block_size
    bounds = np.zeros((num_blocks, 6), dtype=np.float32)

    # 这里的 xyz_cpu 已经是 Morton 排序过的，所以切片后的点在空间上是紧凑的
    for block_id in range(num_blocks):
        start_idx = block_id * block_size
        end_idx = min(start_idx + block_size, total_points)

        # 纯内存操作，极快
        block_data = xyz_cpu[start_idx:end_idx]
        
        bounds[block_id, :3] = block_data.min(axis=0)
        bounds[block_id, 3:] = block_data.max(axis=0)

    return bounds
