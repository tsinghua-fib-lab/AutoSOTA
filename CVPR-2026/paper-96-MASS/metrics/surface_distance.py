"""Surface distance metrics for 3D segmentation masks.

This module computes average surface distance and robust Hausdorff distance
from binary masks and voxel spacing, using local lookup tables for surface
area estimation.
"""

import numpy as np
import scipy.ndimage as ndimage
from .look_up_table import _NEIGHBOUR_CODE_TO_NORMALS
from typing import Dict, Any, List, Tuple, Union, Optional


# Lookup tables for surface distance calculations
def _create_kernel_3d():
    """Create a 3D kernel for neighborhood encoding."""
    return np.array([[[128, 64], [32, 16]], [[8, 4], [2, 1]]])


def _create_neighborhood_code_to_surface_area(spacing_mm: List[float]):
    """
    Create a lookup table mapping neighborhood codes to surface areas.
    
    Args:
        spacing_mm: Voxel spacing in array-axis order, matching mask axes
        
    Returns:
        Array mapping neighborhood codes to surface areas
    """
    # Precompute surface areas for all 256 possible surface elements
    # (given a 2x2x2 neighborhood) according to the spacing_mm
    neighbor_code_to_surface_area = np.zeros([256])
    for code in range(256):
        normals = np.array(_NEIGHBOUR_CODE_TO_NORMALS[code])
        sum_area = 0
        for normal_idx in range(normals.shape[0]):
            # Normal vector
            n = np.zeros([3])
            n[0] = normals[normal_idx, 0] * spacing_mm[1] * spacing_mm[2]
            n[1] = normals[normal_idx, 1] * spacing_mm[0] * spacing_mm[2]
            n[2] = normals[normal_idx, 2] * spacing_mm[0] * spacing_mm[1]
            area = np.linalg.norm(n)
            sum_area += area
        neighbor_code_to_surface_area[code] = sum_area
    
    return neighbor_code_to_surface_area



def _compute_bounding_box(mask: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Compute the bounding box of a binary mask.
    
    Args:
        mask: Binary mask as numpy array
        
    Returns:
        Tuple of (min_coords, max_coords) or None if mask is empty
    """
    ndim = len(mask.shape)
    bbox_min = np.zeros(ndim, np.int64)
    bbox_max = np.zeros(ndim, np.int64)
    
    # Project along each axis to find non-zero indices
    for axis in range(ndim):
        nonzero = None
        axes_to_project = list(range(ndim))
        axes_to_project.remove(axis)
        projection = np.any(mask, axis=tuple(axes_to_project))
        nonzero = np.nonzero(projection)[0]
        
        if len(nonzero) == 0:
            # Empty mask
            return None
        
        bbox_min[axis] = np.min(nonzero)
        bbox_max[axis] = np.max(nonzero)
    
    return bbox_min, bbox_max


def _crop_to_bounding_box(
    mask: np.ndarray, 
    bbox_min: np.ndarray, 
    bbox_max: np.ndarray
) -> np.ndarray:
    """
    Crop a mask to its bounding box with padding.
    
    Args:
        mask: Binary mask as numpy array
        bbox_min: Minimum coordinates of bounding box
        bbox_max: Maximum coordinates of bounding box
        
    Returns:
        Cropped mask with padding
    """
    ndim = len(mask.shape)
    cropmask = np.zeros((bbox_max - bbox_min) + 2, np.uint8)
    
    region_slices = tuple(slice(bbox_min[i], bbox_max[i] + 1) for i in range(ndim))
    cropmask_slices = tuple(slice(0, -1) for _ in range(ndim))
    
    # Copy the data
    cropmask[cropmask_slices] = mask[region_slices]
    
    return cropmask


def calculate_surface_distance(
    mask_gt: np.ndarray,
    mask_pred: np.ndarray,
    spacing_mm: List[float]
) -> Dict[str, float]:
    """
    Calculate surface distance metrics between ground truth and predicted masks.
    
    Args:
        mask_gt: Ground truth binary mask as numpy array
        mask_pred: Predicted binary mask as numpy array
        spacing_mm: Voxel spacing in array-axis order, matching mask axes
        
    Returns:
        Dictionary with surface distance metrics:
        - mean_surface_distance: Average symmetric surface distance (mm)
        - median_surface_distance: Median symmetric surface distance (mm)
        - max_surface_distance: Maximum symmetric surface distance (mm)
        - hausdorff_distance_95: 95th percentile Hausdorff distance (mm)
        - hausdorff_distance: Hausdorff distance (mm)
    """
    if not isinstance(mask_gt, np.ndarray) or not isinstance(mask_pred, np.ndarray):
        raise ValueError("Masks must be numpy arrays")
    
    if mask_gt.shape != mask_pred.shape:
        raise ValueError(f"Mask shapes must match: {mask_gt.shape} != {mask_pred.shape}")
    
    if mask_gt.dtype != bool:
        mask_gt = mask_gt.astype(bool)
    
    if mask_pred.dtype != bool:
        mask_pred = mask_pred.astype(bool)
    
    if not np.any(mask_gt) or not np.any(mask_pred):
        return {
            'mean_surface_distance': float('inf'),
            'median_surface_distance': float('inf'),
            'max_surface_distance': float('inf'),
            'hausdorff_distance_95': float('inf'),
            'hausdorff_distance': float('inf')
        }
    
    bbox_gt = _compute_bounding_box(mask_gt)
    bbox_pred = _compute_bounding_box(mask_pred)
    
    if bbox_gt is None or bbox_pred is None:
        return {
            'mean_surface_distance': float('inf'),
            'median_surface_distance': float('inf'),
            'max_surface_distance': float('inf'),
            'hausdorff_distance_95': float('inf'),
            'hausdorff_distance': float('inf')
        }
    
    # Combine bounding boxes to get a region containing both masks
    ndim = len(mask_gt.shape)
    bbox_min = np.zeros(ndim, np.int64)
    bbox_max = np.zeros(ndim, np.int64)
    
    for i in range(ndim):
        bbox_min[i] = min(bbox_gt[0][i], bbox_pred[0][i])
        bbox_max[i] = max(bbox_gt[1][i], bbox_pred[1][i])
    
    # Crop masks to bounding box
    cropmask_gt = _crop_to_bounding_box(mask_gt, bbox_min, bbox_max)
    cropmask_pred = _crop_to_bounding_box(mask_pred, bbox_min, bbox_max)
    
    kernel = _create_kernel_3d()
    neighbour_code_map_gt = ndimage.correlate(
        cropmask_gt.astype(np.uint8), kernel, mode="constant", cval=0
    )
    neighbour_code_map_pred = ndimage.correlate(
        cropmask_pred.astype(np.uint8), kernel, mode="constant", cval=0
    )
    
    borders_gt = ((neighbour_code_map_gt != 0) & (neighbour_code_map_gt != 255))
    borders_pred = ((neighbour_code_map_pred != 0) & (neighbour_code_map_pred != 255))
    
    distance_gt = ndimage.distance_transform_edt(~borders_gt, sampling=spacing_mm)
    distance_pred = ndimage.distance_transform_edt(~borders_pred, sampling=spacing_mm)
    
    area_map = _create_neighborhood_code_to_surface_area(spacing_mm)
    surface_area_gt = area_map[neighbour_code_map_gt]
    surface_area_pred = area_map[neighbour_code_map_pred]
    
    distances_gt_to_pred = distance_pred[borders_gt]
    distances_pred_to_gt = distance_gt[borders_pred]
    surfel_areas_gt = surface_area_gt[borders_gt]
    surfel_areas_pred = surface_area_pred[borders_pred]
    
    sorted_distances_gt_to_pred = np.sort(distances_gt_to_pred)
    sorted_distances_pred_to_gt = np.sort(distances_pred_to_gt)
    
    dist_gt_to_pred = np.sum(distances_gt_to_pred * surfel_areas_gt) / np.sum(surfel_areas_gt)
    dist_pred_to_gt = np.sum(distances_pred_to_gt * surfel_areas_pred) / np.sum(surfel_areas_pred)
    mean_surface_distance = (dist_gt_to_pred + dist_pred_to_gt) / 2
    
    # Median distance
    if len(sorted_distances_gt_to_pred) > 0 and len(sorted_distances_pred_to_gt) > 0:
        median_surface_distance = (
            np.median(sorted_distances_gt_to_pred) + 
            np.median(sorted_distances_pred_to_gt)
        ) / 2
    else:
        median_surface_distance = float('inf')
    
    # Maximum distance (Hausdorff)
    max_distance = max(
        np.max(sorted_distances_gt_to_pred) if len(sorted_distances_gt_to_pred) > 0 else 0,
        np.max(sorted_distances_pred_to_gt) if len(sorted_distances_pred_to_gt) > 0 else 0
    )
    
    # 95th percentile Hausdorff distance
    hd95_gt_to_pred = np.percentile(sorted_distances_gt_to_pred, 95) if len(sorted_distances_gt_to_pred) > 0 else 0
    hd95_pred_to_gt = np.percentile(sorted_distances_pred_to_gt, 95) if len(sorted_distances_pred_to_gt) > 0 else 0
    hausdorff_distance_95 = max(hd95_gt_to_pred, hd95_pred_to_gt)
    
    return {
        'mean_surface_distance': mean_surface_distance,
        'median_surface_distance': median_surface_distance,
        'max_surface_distance': max_distance,
        'hausdorff_distance_95': hausdorff_distance_95,
        'hausdorff_distance': max_distance
    }


def calculate_surface_dice_at_tolerance(
    mask_gt: np.ndarray,
    mask_pred: np.ndarray,
    spacing_mm: List[float],
    tolerance_mm: float = 1.0
) -> float:
    """
    Calculate surface Dice at a specified tolerance.
    
    Args:
        mask_gt: Ground truth binary mask as numpy array
        mask_pred: Predicted binary mask as numpy array
        spacing_mm: Voxel spacing in array-axis order, matching mask axes
        tolerance_mm: Distance tolerance in mm
        
    Returns:
        Surface Dice score at the specified tolerance
    """
    if not isinstance(mask_gt, np.ndarray) or not isinstance(mask_pred, np.ndarray):
        raise ValueError("Masks must be numpy arrays")
    
    if mask_gt.shape != mask_pred.shape:
        raise ValueError(f"Mask shapes must match: {mask_gt.shape} != {mask_pred.shape}")
    
    if mask_gt.dtype != bool:
        mask_gt = mask_gt.astype(bool)
    
    if mask_pred.dtype != bool:
        mask_pred = mask_pred.astype(bool)
    
    if not np.any(mask_gt) and not np.any(mask_pred):
        return 1.0  # Both masks empty = perfect match
    
    if not np.any(mask_gt) or not np.any(mask_pred):
        return 0.0  # One mask empty, other not = no match
    
    bbox_gt = _compute_bounding_box(mask_gt)
    bbox_pred = _compute_bounding_box(mask_pred)
    
    if bbox_gt is None or bbox_pred is None:
        return 0.0
    
    # Combine bounding boxes to get a region containing both masks
    ndim = len(mask_gt.shape)
    bbox_min = np.zeros(ndim, np.int64)
    bbox_max = np.zeros(ndim, np.int64)
    
    for i in range(ndim):
        bbox_min[i] = min(bbox_gt[0][i], bbox_pred[0][i])
        bbox_max[i] = max(bbox_gt[1][i], bbox_pred[1][i])
    
    # Crop masks to bounding box
    cropmask_gt = _crop_to_bounding_box(mask_gt, bbox_min, bbox_max)
    cropmask_pred = _crop_to_bounding_box(mask_pred, bbox_min, bbox_max)
    
    kernel = _create_kernel_3d()
    neighbour_code_map_gt = ndimage.correlate(
        cropmask_gt.astype(np.uint8), kernel, mode="constant", cval=0
    )
    neighbour_code_map_pred = ndimage.correlate(
        cropmask_pred.astype(np.uint8), kernel, mode="constant", cval=0
    )
    
    borders_gt = ((neighbour_code_map_gt != 0) & (neighbour_code_map_gt != 255))
    borders_pred = ((neighbour_code_map_pred != 0) & (neighbour_code_map_pred != 255))
    
    distance_gt = ndimage.distance_transform_edt(~borders_gt, sampling=spacing_mm)
    distance_pred = ndimage.distance_transform_edt(~borders_pred, sampling=spacing_mm)
    
    area_map = _create_neighborhood_code_to_surface_area(spacing_mm)
    surface_area_gt = area_map[neighbour_code_map_gt]
    surface_area_pred = area_map[neighbour_code_map_pred]
    
    distances_gt_to_pred = distance_pred[borders_gt]
    distances_pred_to_gt = distance_gt[borders_pred]
    surfel_areas_gt = surface_area_gt[borders_gt]
    surfel_areas_pred = surface_area_pred[borders_pred]
    
    overlap_gt = np.sum(surfel_areas_gt[distances_gt_to_pred <= tolerance_mm])
    overlap_pred = np.sum(surfel_areas_pred[distances_pred_to_gt <= tolerance_mm])
    
    surface_dice = (overlap_gt + overlap_pred) / (np.sum(surfel_areas_gt) + np.sum(surfel_areas_pred))
    
    return surface_dice
