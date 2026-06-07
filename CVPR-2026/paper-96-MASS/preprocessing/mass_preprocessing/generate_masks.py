#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SAM2-based auto-mask generation for MASS preprocessing.

This script converts a standardized 3D medical image into SAM2-friendly 2D
slices, runs automatic mask generation on sampled slices, propagates masks
through the volume with SAM2 video tracking, and writes class-agnostic 3D auto
masks for MASS self-supervised pretraining.
"""

import os
import numpy as np
import torch
torch.set_float32_matmul_precision('high')
import torch.nn.functional as F
import SimpleITK as sitk
import nibabel as nib
import cv2
import shutil
import tempfile
import logging
from pathlib import Path
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import json
import time
import traceback
import glob
import multiprocessing
from multiprocessing import Manager, Process
from functools import partial
import gc
from concurrent.futures import ThreadPoolExecutor

# Default only; users can override this before launch to match their node.
os.environ.setdefault('NUMEXPR_MAX_THREADS', '112')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()


# Allow unsupported torch.compile paths to fall back to eager mode.
import torch._dynamo
torch._dynamo.config.suppress_errors = True



# SAM2, enhancement, sampling, and mask-size parameters are grouped here so
# dataset wrapper scripts can override them consistently.

DATASET_CONFIGS = {
    'abdomen_ct': {
        'modality': 'CT',
        'sam_params': {
            'points_per_side': 32,
            'points_per_batch': 128,
            'crop_n_layers': 2,
            'crop_n_points_downscale_factor': 1,
            'pred_iou_thresh': 0.4,
            'stability_score_thresh': 0.7,
            'stability_score_offset': 1.0,
            'mask_threshold': 0.0,
            'crop_overlap_ratio': 0.5,
            'box_nms_thresh': 0.5,
            'crop_nms_thresh': 0.5,
            'output_mode': 'binary_mask',
            'multimask_output': True,
            'use_m2m': False,
        },
        'enhancement': {
            'window_ranges': [
                {'center': 60, 'width': 350},
                {'center': 15, 'width': 250},
                {'center': 150, 'width': 1200}
            ],
            'enable_clahe': False,
            'clahe_clip_limit': 2.0,
            'clahe_grid_size': (8, 8)
        },
        'min_pixel_count': 500,
        'min_voxel_count': 1000
    },
    'abdomen_mr': {
        'modality': 'MRI',
        'sam_params': {
            'points_per_side': 32,
            'points_per_batch': 128,
            'crop_n_layers': 2,
            'crop_n_points_downscale_factor': 1,
            'pred_iou_thresh': 0.4,
            'stability_score_thresh': 0.7,
            'stability_score_offset': 1.0,
            'mask_threshold': 0.0,
            'crop_overlap_ratio': 0.5,
            'box_nms_thresh': 0.5,
            'crop_nms_thresh': 0.5,
            'output_mode': 'binary_mask',
            'multimask_output': True,
            'use_m2m': False,
        },
        'enhancement': {
            'quantile_ranges': [
                {'lower': 5, 'upper': 95},
                {'lower': 15, 'upper': 85},
                {'lower': 1, 'upper': 99}
            ],
            'enable_clahe': False,
            'clahe_clip_limit': 2.0,
            'clahe_grid_size': (8, 8)
        },
        'min_pixel_count': 500,
        'min_voxel_count': 1000
    },
    'brain_mr': {
        'modality': 'MRI',
        'sam_params': {
            'points_per_side': 32,
            'points_per_batch': 128,
            'crop_n_layers': 2,
            'crop_n_points_downscale_factor': 1,
            'pred_iou_thresh': 0.4,
            'stability_score_thresh': 0.7,
            'stability_score_offset': 1.0,
            'mask_threshold': 0.0,
            'crop_overlap_ratio': 0.5,
            'box_nms_thresh': 0.5,
            'crop_nms_thresh': 0.5,
            'output_mode': 'binary_mask',
            'multimask_output': True,
            'use_m2m': False,
        },
        'enhancement': {
            'quantile_ranges': [
                {'lower': 1, 'upper': 99},
                {'lower': 5, 'upper': 95},
                {'lower': 10, 'upper': 90}
            ],
            'enable_clahe': False,
            'clahe_clip_limit': 2.0,
            'clahe_grid_size': (8, 8)
        },
        'min_pixel_count': 200,
        'min_voxel_count': 500
    },
    'cardiac_mr': {
        'modality': 'MRI',
        'sam_params': {
            'points_per_side': 32,
            'points_per_batch': 128,
            'crop_n_layers': 2,
            'crop_n_points_downscale_factor': 1,
            'pred_iou_thresh': 0.4,
            'stability_score_thresh': 0.7,
            'stability_score_offset': 1.0,
            'mask_threshold': 0.0,
            'crop_overlap_ratio': 0.5,
            'box_nms_thresh': 0.5,
            'crop_nms_thresh': 0.5,
            'output_mode': 'binary_mask',
            'multimask_output': True,
            'use_m2m': False,
        },
        'enhancement': {
            'quantile_ranges': [
                {'lower': 5, 'upper': 95},
                {'lower': 15, 'upper': 85},
                {'lower': 1, 'upper': 99}
            ],
            'enable_clahe': False,
            'clahe_clip_limit': 2.0,
            'clahe_grid_size': (8, 8)
        },
        'min_pixel_count': 100,
        'min_voxel_count': 200
    },
    'chest_ct': {
        'modality': 'CT',
        'sam_params': {
            'points_per_side': 32,
            'points_per_batch': 128,
            'crop_n_layers': 2,
            'crop_n_points_downscale_factor': 1,
            'pred_iou_thresh': 0.4,
            'stability_score_thresh': 0.7,
            'stability_score_offset': 1.0,
            'mask_threshold': 0.0,
            'crop_overlap_ratio': 0.5,
            'box_nms_thresh': 0.5,
            'crop_nms_thresh': 0.5,
            'output_mode': 'binary_mask',
            'multimask_output': True,
            'use_m2m': False,
        },
        'enhancement': {
            'window_ranges': [
                {'center': -600, 'width': 1500},  # Lung window
                {'center': 50, 'width': 350},     # Mediastinum window
                {'center': 500, 'width': 2000}    # Bone window
            ],
            'enable_clahe': False,
            'clahe_clip_limit': 2.0,
            'clahe_grid_size': (8, 8)
        },
        'min_pixel_count': 500,
        'min_voxel_count': 3000
    },
    'totalseg_ct': {
        'modality': 'CT',
        'sam_params': {
            'points_per_side': 32,
            'points_per_batch': 128,
            'crop_n_layers': 2,
            'crop_n_points_downscale_factor': 1,
            'pred_iou_thresh': 0.4,
            'stability_score_thresh': 0.7,
            'stability_score_offset': 1.0,
            'mask_threshold': 0.0,
            'crop_overlap_ratio': 0.5,
            'box_nms_thresh': 0.5,
            'crop_nms_thresh': 0.5,
            'output_mode': 'binary_mask',
            'multimask_output': True,
            'use_m2m': False,
        },
        'enhancement': {
            'window_ranges': [
                {'center': 60, 'width': 350},
                {'center': 15, 'width': 250},
                {'center': 150, 'width': 1200}
            ],
            'enable_clahe': False,
            'clahe_clip_limit': 2.0,
            'clahe_grid_size': (8, 8)
        },
        'min_pixel_count': 400,
        'min_voxel_count': 800
    },
    'totalseg_mr': {
        'modality': 'MRI',
        'sam_params': {
            'points_per_side': 32,
            'points_per_batch': 128,
            'crop_n_layers': 2,
            'crop_n_points_downscale_factor': 1,
            'pred_iou_thresh': 0.3,
            'stability_score_thresh': 0.7,
            'stability_score_offset': 1.0,
            'mask_threshold': 0.0,
            'crop_overlap_ratio': 0.5,
            'box_nms_thresh': 0.5,
            'crop_nms_thresh': 0.5,
            'output_mode': 'binary_mask',
            'multimask_output': True,
            'use_m2m': False,
        },
        'enhancement': {
            'quantile_ranges': [
                {'lower': 5, 'upper': 95},
                {'lower': 15, 'upper': 85},
                {'lower': 1, 'upper': 99}
            ],
            'enable_clahe': False,
            'clahe_clip_limit': 2.0,
            'clahe_grid_size': (8, 8)
        },
        'min_pixel_count': 300,
        'min_voxel_count': 700
    },
    'lits': {
        'modality': 'CT',
        'sam_params': {
            'points_per_side': 32,
            'points_per_batch': 128,
            'crop_n_layers': 2,
            'crop_n_points_downscale_factor': 1,
            'pred_iou_thresh': 0.4,
            'stability_score_thresh': 0.7,
            'stability_score_offset': 1.0,
            'mask_threshold': 0.0,
            'crop_overlap_ratio': 0.5,
            'box_nms_thresh': 0.5,
            'crop_nms_thresh': 0.5,
            'output_mode': 'binary_mask',
            'multimask_output': True,
            'use_m2m': False,
        },
        'enhancement': {
            'window_ranges': [
                {'center': 60, 'width': 350},
                {'center': 15, 'width': 250},
                {'center': 150, 'width': 1200}
            ],
            'enable_clahe': False,
            'clahe_clip_limit': 2.0,
            'clahe_grid_size': (8, 8)
        },
        'min_pixel_count': 400,
        'min_voxel_count': 8000
    },
    'autopet_ct': {
        'modality': 'CT',
        'sam_params': {
            'points_per_side': 32,
            'points_per_batch': 128,
            'crop_n_layers': 2,
            'crop_n_points_downscale_factor': 1,
            'pred_iou_thresh': 0.4,
            'stability_score_thresh': 0.7,
            'stability_score_offset': 1.0,
            'mask_threshold': 0.0,
            'crop_overlap_ratio': 0.5,
            'box_nms_thresh': 0.5,
            'crop_nms_thresh': 0.5,
            'output_mode': 'binary_mask',
            'multimask_output': True,
            'use_m2m': False,
        },
        'enhancement': {
            'window_ranges': [
                {'center': -600, 'width': 1500},
                {'center': 50, 'width': 350},
                {'center': 700, 'width': 2500}
            ],
            'enable_clahe': False,
            'clahe_clip_limit': 2.0,
            'clahe_grid_size': (8, 8)
        },
        'min_pixel_count': 700,
        'min_voxel_count': 1500
    },
    'autopet_suv': {
        'modality': 'PET',
        'sam_params': {
            'points_per_side': 32,
            'points_per_batch': 128,
            'crop_n_layers': 2,
            'crop_n_points_downscale_factor': 1,
            'pred_iou_thresh': 0.4,
            'stability_score_thresh': 0.7,
            'stability_score_offset': 1.0,
            'mask_threshold': 0.0,
            'crop_overlap_ratio': 0.5,
            'box_nms_thresh': 0.5,
            'crop_nms_thresh': 0.5,
            'output_mode': 'binary_mask',
            'multimask_output': True,
            'use_m2m': False,
        },
        'enhancement': {
            'window_ranges': [
                {'center': 1.5, 'width': 3.0},
                {'center': 5.0, 'width': 10.0},
                {'center': 15.0, 'width': 30.0}
            ],
            'enable_clahe': False,
            'clahe_clip_limit': 2.0,
            'clahe_grid_size': (8, 8)
        },
        'min_pixel_count': 500,
        'min_voxel_count': 1000
    },
    'structseg_head_oar': {
        'modality': 'CT',
        'sam_params': {
            'points_per_side': 32,
            'points_per_batch': 128,
            'crop_n_layers': 2,
            'crop_n_points_downscale_factor': 1,
            'pred_iou_thresh': 0.3,
            'stability_score_thresh': 0.5,
            'stability_score_offset': 1.0,
            'mask_threshold': 0.0,
            'crop_overlap_ratio': 0.5,
            'box_nms_thresh': 0.5,
            'crop_nms_thresh': 0.5,
            'output_mode': 'binary_mask',
            'multimask_output': True,
            'use_m2m': False,
        },
        'enhancement': {
            'window_ranges': [
                {'center': 90, 'width': 300},    # Vascular head window
                {'center': 55, 'width': 120},    # Head narrow window
                {'center': 600, 'width': 3000}   # Temporal bone window
            ],
            'enable_clahe': False,
            'clahe_clip_limit': 2.5,
            'clahe_grid_size': (8, 8)
        },
        'min_pixel_count': 10,
        'min_voxel_count': 30
    },
}


def determine_best_axes(spacing, isotropic_threshold=1.3, process_all_axes_if_isotropic=False):
    """
    Determine the best axes to process based on image spacing.
    """
    x_spacing, y_spacing, z_spacing = spacing
    is_exactly_isotropic = (x_spacing == y_spacing == z_spacing)
    if is_exactly_isotropic:
        logger.info(f"Image has exactly isotropic spacing: {spacing}mm")
        if process_all_axes_if_isotropic:
            logger.info("Processing all axes as requested for isotropic data")
            return [0, 1, 2]
        else:
            logger.info("Using default axis 0 (axial) for exactly isotropic data")
            return [0]
    xy_ratio = max(x_spacing, y_spacing) / min(x_spacing, y_spacing)
    xz_ratio = max(x_spacing, z_spacing) / min(x_spacing, z_spacing)
    yz_ratio = max(y_spacing, z_spacing) / min(y_spacing, z_spacing)
    is_approximately_isotropic = (xy_ratio <= isotropic_threshold and
                                  xz_ratio <= isotropic_threshold and
                                  yz_ratio <= isotropic_threshold)
    if is_approximately_isotropic:
        logger.info(f"Image has approximately isotropic spacing: {spacing}mm")
        logger.info(f"Spacing ratios - xy: {xy_ratio:.2f}, xz: {xz_ratio:.2f}, yz: {yz_ratio:.2f}")
        if process_all_axes_if_isotropic:
            logger.info("Processing all axes as requested for approximately isotropic data")
            return [0, 1, 2]
    # Pick the slicing direction with the best in-plane resolution.
    axial_resolution = (x_spacing + y_spacing) / 2
    sagittal_resolution = (x_spacing + z_spacing) / 2
    coronal_resolution = (y_spacing + z_spacing) / 2
    resolutions = [
        (0, axial_resolution, "axial"),
        (1, sagittal_resolution, "sagittal"),
        (2, coronal_resolution, "coronal")
    ]
    resolutions.sort(key=lambda x: x[1])
    best_axis = resolutions[0][0]
    best_axis_name = resolutions[0][2]
    logger.info(f"Image spacing: {spacing}mm")
    logger.info(f"In-plane resolutions - Axial: {axial_resolution:.2f}mm, "
                f"Sagittal: {sagittal_resolution:.2f}mm, Coronal: {coronal_resolution:.2f}mm")
    logger.info(f"Selected best axis: {best_axis_name} (axis {best_axis})")
    return [best_axis]


def get_axes_for_image(image_path, auto_select=False, specified_axes=None, isotropic_threshold=1.3, process_all_axes_if_isotropic=False):
    """
    Determine which axes to process for a given image.
    """
    if not auto_select:
        return specified_axes
    sitk_image = sitk.ReadImage(str(image_path))
    spacing = sitk_image.GetSpacing()
    return determine_best_axes(spacing, isotropic_threshold, process_all_axes_if_isotropic)



def load_checkpoint(checkpoint_path):
    """Load checkpoint from file."""
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return {}
    return {}


def save_checkpoint(checkpoint, checkpoint_path):
    """Save checkpoint to file."""
    try:
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving checkpoint: {e}")


def update_checkpoint(checkpoint_path, task_id, status, error=None, lock=None):
    """Update checkpoint for a specific task (thread-safe if lock provided)."""
    if lock:
        with lock:
            checkpoint = load_checkpoint(checkpoint_path)
            checkpoint[task_id] = {
                "status": status,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "error": error
            }
            save_checkpoint(checkpoint, checkpoint_path)
    else:
        checkpoint = load_checkpoint(checkpoint_path)
        checkpoint[task_id] = {
            "status": status,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "error": error
        }
        save_checkpoint(checkpoint, checkpoint_path)


def reset_running_tasks_for_resume(checkpoint_path):
    """
    Mark tasks left in ``running`` state by a previous interrupted run as retryable.
    """
    checkpoint = load_checkpoint(checkpoint_path)
    reset_count = 0

    for task_id, info in checkpoint.items():
        if info.get("status") == "running":
            checkpoint[task_id] = {
                "status": "partial",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "error": "Reset stale running status on restart",
            }
            reset_count += 1

    if reset_count:
        save_checkpoint(checkpoint, checkpoint_path)
        logger.info(f"Reset {reset_count} stale running task(s) for resume")

    return checkpoint


def should_process_task(checkpoint_path, task_id, checkpoint_lock=None):
    """
    Check if a task should be processed (not already running or completed).
    Returns True if task should be processed, False otherwise.
    """
    if checkpoint_lock:
        with checkpoint_lock:
            checkpoint = load_checkpoint(checkpoint_path)
    else:
        checkpoint = load_checkpoint(checkpoint_path)

    if task_id in checkpoint:
        status = checkpoint[task_id].get("status", "")
        if status in ["running", "completed"]:
            return False
    return True

def create_task_list(image_paths, axes=None, auto_select_axes=False, isotropic_threshold=1.3, process_all_axes_if_isotropic=False):
    """
    Create list of tasks for multiple images and axes.
    Images are processed at their original resolution.
    """
    tasks = []
    for image_path in image_paths:
        base_filename = Path(image_path).stem.replace('.nii', '')
        axes_to_process = get_axes_for_image(
            image_path,
            auto_select=auto_select_axes,
            specified_axes=axes,
            isotropic_threshold=isotropic_threshold,
            process_all_axes_if_isotropic=process_all_axes_if_isotropic
        )
        for axis in axes_to_process:
            axis_name = ['axial', 'sagittal', 'coronal'][axis]
            task_id = f"{base_filename}_axis{axis}"
            tasks.append({
                'task_id': task_id,
                'image_path': str(image_path),
                'axis': axis,
                'axis_name': axis_name,
                'base_filename': base_filename
            })
    logger.info(f"Created {len(tasks)} total tasks for {len(image_paths)} images")
    return tasks



def is_slice_processed(output_dir, base_filename, axis, slice_index):
    """
    Check if a specific slice has already been processed by looking for the mapping file.
    """
    mapping_filename = f"{base_filename}_axis{axis}_slice{slice_index}_mapping.json"
    mapping_path = os.path.join(output_dir, mapping_filename)
    return os.path.exists(mapping_path)


def get_unprocessed_slices(output_dir, base_filename, axis, slice_indices):
    """
    Filter slice indices to only include unprocessed slices.
    """
    unprocessed_indices = []
    for slice_index in slice_indices:
        if not is_slice_processed(output_dir, base_filename, axis, slice_index):
            unprocessed_indices.append(slice_index)
    return np.array(unprocessed_indices)



def apply_ct_window(volume_data, center, width):
    """Apply CT windowing to volume data."""
    volume_min = center - width / 2
    volume_max = center + width / 2
    windowed = np.clip(volume_data, volume_min, volume_max)
    normalized = ((windowed - volume_min) / (volume_max - volume_min) * 255)
    return normalized.astype(np.uint8)



def create_3channel_windows(volume_data, window_ranges):
    """
    Create 3-channel representation using 3 CT windows.
    Each window becomes a separate channel, preserving full information.
    """
    if len(window_ranges) != 3:
        raise ValueError(f"Exactly 3 windows required for 3-channel input, got {len(window_ranges)}")
    channel_arrays = []
    for window in window_ranges:
        windowed = apply_ct_window(volume_data, window['center'], window['width'])
        channel_arrays.append(windowed)
    three_channel_array = np.stack(channel_arrays, axis=-1)
    return three_channel_array

_QUANTILE_BODY_MASK = None
def set_quantile_body_mask(mask):
    """
    Set or clear the global body mask for MRI quantile normalization.
    Pass a boolean numpy array shaped like the volume (z, y, x), or None to clear.
    """
    global _QUANTILE_BODY_MASK
    _QUANTILE_BODY_MASK = mask

def apply_quantile_normalization(volume_data, lower_percentile, upper_percentile):
    """Apply quantile-based normalization for MRI images.

    NOTE (updated): Percentiles are computed over the current body region if a
    global mask is set via `set_quantile_body_mask`; otherwise the whole volume.
    """
    global _QUANTILE_BODY_MASK

    # Choose values to compute percentiles from
    if _QUANTILE_BODY_MASK is not None and _QUANTILE_BODY_MASK.shape == volume_data.shape:
        values = volume_data[_QUANTILE_BODY_MASK.astype(bool)]
        if values.size == 0:
            logger.warning("Body mask contains no voxels; falling back to full-volume quantiles.")
            values = volume_data.ravel()
    else:
        if _QUANTILE_BODY_MASK is not None and _QUANTILE_BODY_MASK.shape != volume_data.shape:
            logger.warning("Body mask shape mismatch; falling back to full-volume quantiles.")
        values = volume_data.ravel()

    volume_min = np.percentile(values, lower_percentile)
    volume_max = np.percentile(values, upper_percentile)
    # Guard against degenerate ranges
    if not np.isfinite(volume_min):
        volume_min = float(np.nanmin(values)) if values.size else 0.0
    if not np.isfinite(volume_max):
        volume_max = float(np.nanmax(values)) if values.size else 1.0
    if volume_max <= volume_min:
        volume_max = volume_min + 1e-6

    volume_data = np.clip(volume_data, volume_min, volume_max)
    normalized = ((volume_data - volume_min) / (volume_max - volume_min) * 255)
    return normalized.astype(np.uint8)

def create_3channel_quantiles(volume_data, quantile_ranges):
    """
    Create 3-channel representation using 3 different quantile ranges for MRI.
    Each quantile range becomes a separate channel.

    Percentiles are computed over the current body region if a global mask has
    been set via `set_quantile_body_mask`; otherwise, over the whole volume.
    """
    if len(quantile_ranges) != 3:
        raise ValueError(f"Exactly 3 quantile ranges required for 3-channel input, got {len(quantile_ranges)}")
    channel_arrays = []
    for qrange in quantile_ranges:
        normalized = apply_quantile_normalization(
            volume_data,
            qrange['lower'],
            qrange['upper']
        )
        channel_arrays.append(normalized)
    three_channel_array = np.stack(channel_arrays, axis=-1)
    return three_channel_array


def apply_clahe_2d(image_2d, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Apply CLAHE to a 2D image slice."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tile_gridSize=tile_grid_size)
    return clahe.apply(image_2d)



def body_segment_with_totalsegmentator(image_path, modality='CT'):
    """
    Use TotalSegmentator to segment the body from the 3D image.
    """
    try:
        from totalsegmentator.python_api import totalsegmentator
        if modality in ['MRI', 'MR']:
            task = 'body_mr'
            logger.info(f"Using TotalSegmentator with task='body_mr' for MRI image")
        elif modality in ['CT']:
            task = 'body'
            logger.info(f"Using TotalSegmentator with task='body' for CT image")
        else:
            logger.info(f"Not CT or MRI, using full volume as body mask")
            sitk_image = sitk.ReadImage(image_path)
            image_array = sitk.GetArrayFromImage(sitk_image)
            return np.ones_like(image_array, dtype=np.uint8)

        input_img = nib.load(image_path)
        output_seg = totalsegmentator(input_img, ml=True, task=task)
        np_data = output_seg.get_fdata()
        np_data[np_data > 0] = 1
        return np.transpose(np_data.astype(np.uint8), (2, 1, 0))
    except ImportError:
        logger.warning("TotalSegmentator not found. Using full volume as body mask.")
        sitk_image = sitk.ReadImage(image_path)
        image_array = sitk.GetArrayFromImage(sitk_image)
        return np.ones_like(image_array, dtype=np.uint8)



def find_gt_file(image_path):
    """
    Find corresponding GT label file for an image.
    """
    image_path = Path(image_path)
    base_name = image_path.stem.replace('.nii', '')
    gt_patterns = [
        f"{base_name}_gt.nii.gz",
        f"{base_name}_gt.nii",
        f"{base_name}gt.nii.gz",
        f"{base_name}gt.nii",
    ]
    for pattern in gt_patterns:
        gt_path = image_path.parent / pattern
        if gt_path.exists():
            return str(gt_path)
    return None


def analyze_gt_for_dense_sampling(gt_path, axis, dense_sampling_classes=None):
    """
    Analyze GT labels to determine which slice ranges need dense sampling.

    If `dense_sampling_classes` is None or empty, dense sampling is disabled and
    this function returns an empty list.
    """
    if not dense_sampling_classes:
        logger.info("Dense sampling disabled (dense_sampling_classes is None or empty).")
        return []

    DENSE_SAMPLING_CLASSES = dense_sampling_classes
    MIN_REGION_SIZE = 5
    PADDING_SLICES = 1
    try:
        sitk_gt = sitk.ReadImage(gt_path)
        gt_array = sitk.GetArrayFromImage(sitk_gt)

        target_mask = np.zeros_like(gt_array, dtype=bool)
        for class_id in DENSE_SAMPLING_CLASSES:
            target_mask |= (gt_array == class_id)

        if axis == 0:
            projection = np.any(target_mask, axis=(1, 2))
        elif axis == 1:
            projection = np.any(target_mask, axis=(0, 2))
        elif axis == 2:
            projection = np.any(target_mask, axis=(0, 1))
        else:
            logger.warning(f"Invalid axis {axis} for dense sampling analysis; expected 0, 1, or 2.")
            return []

        dense_regions = []
        in_region = False
        start = 0
        for i, has_target in enumerate(projection):
            if has_target and not in_region:
                start = i
                in_region = True
            elif not has_target and in_region:
                if i - start >= MIN_REGION_SIZE:
                    region_start = max(0, start - PADDING_SLICES)
                    region_end = min(len(projection) - 1, i + PADDING_SLICES)
                    dense_regions.append((region_start, region_end))
                in_region = False

        if in_region and len(projection) - start >= MIN_REGION_SIZE:
            region_start = max(0, start - PADDING_SLICES)
            region_end = len(projection) - 1
            dense_regions.append((region_start, region_end))

        return dense_regions

    except Exception as e:
        logger.warning(f"Failed to analyze GT for dense sampling: {e}")
        return []



def sample_slices_with_interval_gt_aware(image_array, body_mask, physical_interval_mm=30.0,
                                         axis=0, spacing=None, image_path=None, dense_sampling_classes=None,
                                         min_slices_per_sample=10):
    """
    Sample slices with GT-aware variable density.
    """
    if spacing is None:
        raise ValueError("Image spacing must be provided to calculate slice intervals")
    if len(image_array.shape) == 4:
        depth = image_array.shape[axis]
    else:
        depth = image_array.shape[axis]
    if axis == 0:
        projection = np.any(body_mask, axis=(1, 2))
    elif axis == 1:
        projection = np.any(body_mask, axis=(0, 2))
    elif axis == 2:
        projection = np.any(body_mask, axis=(0, 1))
    valid_slices = np.where(projection)[0]
    if valid_slices.size == 0:
        logger.warning(f"No body slices found along axis {axis}. Using regular sampling.")
        indices = np.arange(0, depth, max(1, depth // min_slices_per_sample))
        return indices
    bbox_min = valid_slices.min()
    bbox_max = valid_slices.max()
    body_range = bbox_max - bbox_min + 1
    axis_spacing_map = {0: spacing[2], 1: spacing[1], 2: spacing[0]}
    axis_spacing = axis_spacing_map[axis]
    sparse_interval = max(1, int(round(physical_interval_mm / axis_spacing)))
    dense_interval = max(1, sparse_interval // 3)  # Dense interval is 1/3 of the sparse interval
    dense_regions = []
    if image_path:
        gt_path = find_gt_file(image_path)
        if gt_path:
            logger.info(f"Found GT file: {gt_path}")
            dense_regions = analyze_gt_for_dense_sampling(gt_path, axis, dense_sampling_classes)
            if dense_regions:
                logger.info(f"Identified {len(dense_regions)} regions for dense sampling: {dense_regions}")
        else:
            logger.info("No GT file found, using uniform sampling")
    indices = []
    current = bbox_min
    while current <= bbox_max:
        indices.append(current)
        in_dense_region = False
        for start, end in dense_regions:
            if start <= current <= end:
                in_dense_region = True
                break
        if in_dense_region:
            current += dense_interval
        else:
            current += sparse_interval
    indices = np.array(indices)
    if len(indices) < min_slices_per_sample:
        if body_range < min_slices_per_sample:
            indices = np.arange(bbox_min, bbox_max + 1)
            logger.info(f"Body region has only {body_range} slices, sampling all of them")
        else:
            indices = np.linspace(bbox_min, bbox_max, min_slices_per_sample, dtype=int)
            indices = np.unique(indices)
            logger.info(f"Physical-based sampling yielded only {len(indices)} slices, "
                       f"switching to uniform sampling with {len(indices)} slices")
    else:
        physical_extent = (bbox_max - bbox_min) * axis_spacing
        logger.info(f"Sampled {len(indices)} slices from body region [{bbox_min}, {bbox_max}] "
                    f"(physical extent: {physical_extent:.1f}mm)")
        if dense_regions:
            logger.info(f"Used dense sampling in {len(dense_regions)} regions")
    return indices


def extract_slice(image_array, axis, index):
    """Extract a 2D slice from a 3D/4D volume along the specified axis."""
    if len(image_array.shape) == 4:
        if axis == 0:
            return image_array[index, :, :, :]
        elif axis == 1:
            return image_array[:, index, :, :]
        elif axis == 2:
            return image_array[:, :, index, :]
    else:
        if axis == 0:
            return image_array[index, :, :]
        elif axis == 1:
            return image_array[:, index, :]
        elif axis == 2:
            return image_array[:, :, index]
    raise ValueError("Axis must be 0, 1, or 2.")


def insert_slice(volume, slice_data, axis, index):
    """Insert a 2D slice into a 3D volume along the specified axis."""
    if axis == 0:
        volume[index, :, :] = slice_data
    elif axis == 1:
        volume[:, index, :] = slice_data
    elif axis == 2:
        volume[:, :, index] = slice_data
    else:
        raise ValueError("Axis must be 0, 1, or 2.")


def prepare_video_directory(image_array_3ch, axis, output_dir, enhancement_config):
    """
    Prepare a directory with image slices for video propagation.

    Uses RAM disk (/dev/shm) or $SAM2_TMPFS when available for faster I/O,
    falling back to output_dir. Slices are written as JPEG (quality 90) using
    a thread pool to minimise I/O latency.
    """
    # Choose a fast temp root if available
    candidates = []
    env_tmp = os.environ.get("SAM2_TMPFS")
    if env_tmp:
        candidates.append(env_tmp)
    candidates.append("/dev/shm")
    candidates.append(output_dir)

    temp_dir = None
    for root in candidates:
        try:
            if root and os.path.isdir(root) and os.access(root, os.W_OK):
                temp_dir = tempfile.mkdtemp(dir=root)
                break
        except Exception:
            continue
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(dir=output_dir)

    enable_clahe = enhancement_config.get('enable_clahe', False)

    num_slices = image_array_3ch.shape[axis]
    logger.info(f"Preparing video directory with {num_slices} slices in {temp_dir}")
    if enable_clahe:
        logger.info(f"Applying CLAHE to all slices (clip_limit={enhancement_config['clahe_clip_limit']}, "
                    f"grid_size={enhancement_config['clahe_grid_size']})")

    def save_one(i: int):
        slice_rgb = extract_slice(image_array_3ch, axis, i)
        if enable_clahe:
            slice_rgb_clahe = np.zeros_like(slice_rgb)
            for ch in range(3):
                slice_rgb_clahe[:, :, ch] = apply_clahe_2d(
                    slice_rgb[:, :, ch],
                    clip_limit=enhancement_config['clahe_clip_limit'],
                    tile_gridSize=enhancement_config['clahe_grid_size']
                )
            slice_rgb = slice_rgb_clahe
        image_path = os.path.join(temp_dir, f"{i:05d}.jpg")
        ok = cv2.imwrite(image_path, slice_rgb, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return ok

    max_workers = max(2, min(8, (os.cpu_count() or 8) // 2))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(save_one, range(num_slices)))

    saved_images = sorted(glob.glob(os.path.join(temp_dir, "*.jpg")))
    logger.info(f"Saved {len(saved_images)} JPG images to {temp_dir}")
    if len(saved_images) == 0:
        raise RuntimeError(f"Failed to save images to {temp_dir}")
    return temp_dir


def visualize_and_save_masks(slice_data, masks, output_path, title=None):
    """Visualize and save masks overlaid on the original slice."""
    h, w = slice_data.shape[:2]
    aspect_ratio = w / h
    fig_w = 12
    fig_h = fig_w / aspect_ratio
    plt.figure(figsize=(fig_w, fig_h))
    plt.imshow(slice_data)
    cmap = plt.get_cmap('tab10')
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        color = np.array(cmap(i % 10)[:3])
        h, w = mask.shape
        colored_mask = np.zeros((h, w, 4))
        colored_mask[mask, :3] = color
        colored_mask[mask, 3] = 0.5
        plt.imshow(colored_mask, alpha=0.5)
        contours, _ = cv2.findContours(mask.astype(np.uint8),
                                      cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour_reshaped = contour.reshape(-1, 2)
            plt.plot(contour_reshaped[:, 0], contour_reshaped[:, 1],
                    color=color, linewidth=1)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def pack_masks_efficiently(mask_volumes, original_shape):
    """
    Pack overlapping masks into as few volumes as possible.
    Uses graph coloring approach to minimize memory usage.
    """
    num_masks = len(mask_volumes)
    if num_masks == 0:
        return [], {}
    overlap_matrix = np.zeros((num_masks, num_masks), dtype=bool)
    mask_list = list(mask_volumes.values())
    obj_ids = list(mask_volumes.keys())
    for i in range(num_masks):
        for j in range(i+1, num_masks):
            overlap = np.any(mask_list[i] & mask_list[j])
            overlap_matrix[i, j] = overlap
            overlap_matrix[j, i] = overlap
    colors = [-1] * num_masks
    for i in range(num_masks):
        neighbor_colors = set()
        for j in range(num_masks):
            if overlap_matrix[i, j] and colors[j] != -1:
                neighbor_colors.add(colors[j])
        color = 0
        while color in neighbor_colors:
            color += 1
        colors[i] = color
    num_colors = max(colors) + 1
    packed_volumes = []
    obj_id_mapping = {}
    for color in range(num_colors):
        packed_volume = np.zeros(original_shape, dtype=np.uint16)
        label = 1
        for i, c in enumerate(colors):
            if c == color:
                obj_id = obj_ids[i]
                packed_volume[mask_list[i] > 0] = label
                obj_id_mapping[obj_id] = (color, label)
                label += 1
        packed_volumes.append(packed_volume)
    logger.info(f"Packed {num_masks} masks into {len(packed_volumes)} volumes")
    return packed_volumes, obj_id_mapping



def load_sam2_models(sam2_checkpoint, model_cfg, device, dataset_config, torch_compile=False):
    """
    Load SAM2 models for automatic mask generation and video prediction.
    """
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2_video_predictor
    logger.info(f"Loading SAM2 models on device {device} (This should happen ONCE per GPU)")
    start_time = time.time()
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    sam_config = dataset_config['sam_params'].copy()
    sam_config['min_mask_region_area'] = dataset_config['min_pixel_count']
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        **sam_config
    )
    logger.info(f"Loading SAM2 video predictor (torch_compile={torch_compile})")
    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device, vos_optimized=torch_compile)
    load_time = time.time() - start_time
    logger.info(f"Model loading completed in {load_time:.2f} seconds")
    return mask_generator, video_predictor


def _env_int(name, default):
    try:
        v = os.environ.get(name, None)
        if v is None:
            return default
        return int(v)
    except Exception:
        return default

def _infer_default_chunk_size():
    """
    Heuristic: choose a per-propagation object limit based on GPU VRAM.
    Overridable via env SAM2_MAX_OBJECTS_PER_CHUNK.
    """
    override = _env_int("SAM2_MAX_OBJECTS_PER_CHUNK", None)
    if override is not None and override > 0:
        return override

    try:
        dev = torch.cuda.current_device()
        total_gb = torch.cuda.get_device_properties(dev).total_memory / (1024**3)
    except Exception:
        total_gb = 16.0

    # Use conservative chunk sizes to keep propagation memory bounded.
    if total_gb <= 8:
        return 16
    elif total_gb <= 12:
        return 24
    elif total_gb <= 16:
        return 32
    elif total_gb <= 24:
        return 48
    elif total_gb <= 40:
        return 64
    else:
        return 80

def _is_cuda_oom(e: Exception) -> bool:
    msg = str(e).lower()
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in msg or "cuda error: out of memory" in msg


def _inplane_pixel_area_mm2(spacing, axis:int) -> float:
    """
    Compute the physical area (mm^2) covered by one pixel in the current 2D slice plane.
    spacing: (sx, sy, sz) in mm from SimpleITK (x,y,z order)
    axis: slicing axis in numpy array terms used in this script (0=z,1=y,2=x)
    Returns: mm^2 per pixel in the current plane.
    """
    sx, sy, sz = float(spacing[0]), float(spacing[1]), float(spacing[2])
    if axis == 0:       # axial plane is (y, x)
        return sx * sy
    elif axis == 1:     # sagittal plane is (z, x)
        return sx * sz
    elif axis == 2:     # coronal plane is (z, y)
        return sy * sz
    else:
        raise ValueError("Axis must be 0, 1, or 2.")

def compute_adaptive_min_mask_region_area(spacing, axis, dataset_config, plane_shape=None):
    """
    Convert a dataset-level target minimum region (in mm^2) into pixel units for the current image.

    - Uses dataset_config['min_pixel_count'] as the canonical target at 1mm x 1mm in-plane pixels,
      i.e. target_mm2 = min_pixel_count * 1.0 mm^2.
    - Scales by 1 / (inplane_pixel_area_mm2) to get pixels for this image/axis.
    - Caps to avoid being too small or too big:
        * lower bound: max(4, int(0.25 * dataset_config['min_pixel_count']))
        * upper bound: min(GLOBAL_MAX, int(4.0 * dataset_config['min_pixel_count']))
          where GLOBAL_MAX ~= 250k pixels (below 512x512 area).
      (These caps are conservative and keep behavior stable across datasets.)
    """
    # 1) target physical area in mm^2, derived from the dataset baseline at 1x1 mm pixels
    baseline_px = max(1, int(dataset_config.get('min_pixel_count', 100)))
    target_mm2 = float(baseline_px)  # treat "min_pixel_count" as mm^2 at 1 mm/px

    # 2) compute per-pixel area (mm^2) in the current plane
    try:
        px_area_mm2 = _inplane_pixel_area_mm2(spacing, axis)
        if not np.isfinite(px_area_mm2) or px_area_mm2 <= 0:
            raise ValueError("Invalid spacing -> nonpositive pixel area")
    except Exception as e:
        logger.warning(f"Adaptive area: invalid spacing {spacing} for axis {axis}: {e}. "
                       f"Falling back to dataset baseline={baseline_px}")
        return baseline_px

    # 3) scale to pixels for this image/axis
    raw_px = int(round(target_mm2 / px_area_mm2))
    raw_px = max(1, raw_px)

    # 4) cap the value (dataset-relative + global)
    LOWER_CAP = max(4, int(0.25 * baseline_px))
    UPPER_CAP = max(LOWER_CAP + 1, int(4.0 * baseline_px))
    GLOBAL_MAX = 250_000  # ~ below 512x512 area; keeps guardrails for very tiny spacing
    cap_min = LOWER_CAP
    cap_max = min(GLOBAL_MAX, UPPER_CAP)

    # Optional: never exceed the plane area if provided
    if plane_shape is not None and len(plane_shape) == 2:
        plane_area = int(plane_shape[0]) * int(plane_shape[1])
        if plane_area > 0:
            cap_max = min(cap_max, plane_area)

    adapted_px = int(np.clip(raw_px, cap_min, cap_max))

    logger.info(
        f"Adaptive min_mask_region_area: target_mm2={target_mm2:.2f}, "
        f"px_area_mm2={px_area_mm2:.4f}, raw_px={raw_px}, "
        f"clamped_px={adapted_px} (caps: {cap_min}-{cap_max})"
    )
    return adapted_px



def process_single_task(task, output_dir, mask_generator, video_predictor,
                       physical_interval_mm, max_masks_per_slice, dataset_config,
                       device, checkpoint_path, checkpoint_lock, min_slices_per_sample=10):
    """
    Process a single (image, axis) task using pre-loaded SAM2 models.

    One inference state is created per (image, axis). All seeds from selected
    slices are added at once, then propagated bidirectionally (forward + backward)
    to fill the full volume. If the number of objects exceeds the GPU memory
    budget, propagation is performed in chunks (controlled via the env var
    SAM2_MAX_OBJECTS_PER_CHUNK or auto-sized from available VRAM).
    min_mask_region_area is set adaptively from image spacing and the current
    slicing axis.
    """
    task_id = task['task_id']
    temp_dir = None
    enhancement_config = dataset_config['enhancement']
    min_pixel_count = dataset_config['min_pixel_count']
    min_voxel_count = dataset_config['min_voxel_count']
    enable_clahe = enhancement_config.get('enable_clahe', False)

    if not should_process_task(checkpoint_path, task_id, checkpoint_lock):
        logger.info(f"Task {task_id} already running or completed by another node, skipping")
        return task_id, True


    try:
        update_checkpoint(checkpoint_path, task_id, "running", lock=checkpoint_lock)
        logger.info(f"Processing task: {task_id} (using pre-loaded models)")

        sitk_image = sitk.ReadImage(task['image_path'])
        spacing = sitk_image.GetSpacing()
        logger.info(f"Original image size: {sitk_image.GetSize()}, spacing: {spacing} mm")

        modality = dataset_config.get('modality', 'CT')
        logger.info("Generating body mask using TotalSegmentator")
        body_mask_array = body_segment_with_totalsegmentator(task['image_path'], modality=modality)

        image_array = sitk.GetArrayFromImage(sitk_image)
        body_mask = body_mask_array
        original_shape = image_array.shape

        # Adaptive min area based on spacing & axis; set on mask_generator
        if task['axis'] == 0:
            plane_shape = (original_shape[1], original_shape[2])
        elif task['axis'] == 1:
            plane_shape = (original_shape[0], original_shape[2])
        elif task['axis'] == 2:
            plane_shape = (original_shape[0], original_shape[1])
        else:
            plane_shape = None

        adaptive_min_area_px = compute_adaptive_min_mask_region_area(
            spacing=spacing,
            axis=task['axis'],
            dataset_config=dataset_config,
            plane_shape=plane_shape
        )
        try:
            prev_val = getattr(mask_generator, "min_mask_region_area", None)
            mask_generator.min_mask_region_area = int(adaptive_min_area_px)
            logger.info(f"Set SAM min_mask_region_area to {adaptive_min_area_px} (was {prev_val})")
        except Exception as e:
            logger.warning(f"Failed to set adaptive min_mask_region_area; using previous value. Error: {e}")
        min_pixel_count = int(adaptive_min_area_px)

        modality = dataset_config['modality']
        if modality in ['MRI', 'MR']:
            logger.info(f"Creating 3-channel MRI representation with quantile ranges: {enhancement_config['quantile_ranges']}")
            logger.info("Using body-mask-based quantile normalization for MRI")
            try:
                set_quantile_body_mask(body_mask.astype(bool))
            except Exception as _e:
                logger.warning(f"Failed to set body mask for quantile normalization; falling back to full-volume quantiles: {_e}")
                set_quantile_body_mask(None)
            image_array_3ch = create_3channel_quantiles(image_array, enhancement_config['quantile_ranges'])
            set_quantile_body_mask(None)
        else:
            logger.info(f"Creating 3-channel CT representation with windows: {enhancement_config['window_ranges']}")
            image_array_3ch = create_3channel_windows(image_array, enhancement_config['window_ranges'])

        debug_dir = os.path.join(output_dir, "debug_images", f"axis{task['axis']}_{task['axis_name']}")
        os.makedirs(debug_dir, exist_ok=True)

        all_slice_indices = sample_slices_with_interval_gt_aware(
            image_array_3ch, body_mask, physical_interval_mm, task['axis'], spacing,
            image_path=task['image_path'],
            dense_sampling_classes=dataset_config.get('dense_sampling_classes'),
            min_slices_per_sample=min_slices_per_sample
        )

        init_slice_indices = get_unprocessed_slices(
            output_dir, task['base_filename'], task['axis'], all_slice_indices
        )

        if len(init_slice_indices) == 0:
            logger.info(f"All slices already processed for task {task_id}, marking as completed")
            update_checkpoint(checkpoint_path, task_id, "completed", lock=checkpoint_lock)
            return task_id, True

        logger.info(f"Processing {len(init_slice_indices)} unprocessed slices out of {len(all_slice_indices)} total slices")

        temp_dir = prepare_video_directory(
            image_array_3ch, task['axis'], output_dir,
            enhancement_config=enhancement_config
        )

        # Global containers
        mask_3d_volumes_global = {}          # global_obj_id -> 3D uint8 volume
        global_obj_to_seedidx = {}           # global_obj_id -> seed slice index
        per_slice_info = {}                  # slice_idx -> {"global_ids": [gids in local order], "meta": {local_id: {...}}}
        global_obj_meta = {}                 # global_obj_id -> {"predicted_iou":..., "stability_score":...}
        global_obj_counter = 0

        seed_entries = []  # list of (gid, init_slice_index, seed_mask_2d)

        for slice_counter, init_slice_index in enumerate(init_slice_indices):
            logger.info(f"Seeding slice {slice_counter+1}/{len(init_slice_indices)} (index {init_slice_index})")

            slice_data_rgb = extract_slice(image_array_3ch, task['axis'], init_slice_index)
            if enable_clahe:
                slice_data_rgb_clahe = np.zeros_like(slice_data_rgb)
                for ch in range(3):
                    slice_rgb_ch = slice_data_rgb[:, :, ch]
                    slice_data_rgb_clahe[:, :, ch] = apply_clahe_2d(
                        slice_rgb_ch,
                        clip_limit=enhancement_config['clahe_clip_limit'],
                        tile_gridSize=enhancement_config['clahe_grid_size']
                    )
                slice_data_rgb = slice_data_rgb_clahe

            body_mask_slice = extract_slice(body_mask, task['axis'], init_slice_index)

            logger.info("Generating masks with SAM2")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                raw_masks = mask_generator.generate(slice_data_rgb)

            filtered_masks = []
            for md in raw_masks:
                m = md["segmentation"]
                overlap = np.logical_and(m, body_mask_slice)
                overlap_ratio = np.sum(overlap) / (np.sum(m) + 1e-6)
                if overlap_ratio >= 0.9:
                    filtered_masks.append(md)

            num_masks = min(max_masks_per_slice, len(filtered_masks))
            masks = filtered_masks[:num_masks]

            vis_path = os.path.join(
                debug_dir,
                f"{task['base_filename']}_slice{init_slice_index}_masks.png"
            )
            visualize_and_save_masks(
                slice_data_rgb, masks, vis_path,
                title=f"{task['axis_name']} - Slice {init_slice_index}"
            )

            if num_masks == 0:
                logger.warning(f"No masks found for slice {init_slice_index}, skipping")
                continue

            if init_slice_index not in per_slice_info:
                per_slice_info[init_slice_index] = {"global_ids": [], "meta": {}}

            for local_idx, md in enumerate(masks, start=1):
                global_obj_counter += 1
                gid = int(global_obj_counter)
                seed_mask_2d = md["segmentation"].astype(bool)

                # Allocate a global 3D volume and insert seed slice (CPU)
                mask_3d_volumes_global[gid] = np.zeros(original_shape, dtype=np.uint8)
                insert_slice(mask_3d_volumes_global[gid], seed_mask_2d.astype(np.uint8),
                             task['axis'], int(init_slice_index))

                global_obj_to_seedidx[gid] = int(init_slice_index)
                per_slice_info[init_slice_index]["global_ids"].append(gid)
                per_slice_info[init_slice_index]["meta"][local_idx] = {
                    "predicted_iou": float(md.get("predicted_iou", 0.0)),
                    "stability_score": float(md.get("stability_score", 0.0))
                }
                global_obj_meta[gid] = {
                    "predicted_iou": float(md.get("predicted_iou", 0.0)),
                    "stability_score": float(md.get("stability_score", 0.0)),
                    "local_id": local_idx,
                    "seed_slice": int(init_slice_index)
                }

                seed_entries.append((gid, int(init_slice_index), seed_mask_2d))

        # If no seeds were added across all slices, mark partial and exit
        if len(seed_entries) == 0:
            logger.warning("No masks were added for any unprocessed slice; skipping propagation")
            # Keep this task resumable instead of marking empty slices as done.
            update_checkpoint(checkpoint_path, task_id, "partial",
                              f"{len(init_slice_indices)} slices remaining", lock=checkpoint_lock)
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return task_id, True

        # Propagation in chunks to avoid GPU OOM on large volumes
        target_chunk = _infer_default_chunk_size()
        min_chunk = max(1, _env_int("SAM2_MIN_CHUNK_SIZE", 1))
        logger.info(f"Total objects to propagate: {len(seed_entries)}; target chunk size: {target_chunk}")

        processed_gids = set()

        offset = 0
        while offset < len(seed_entries):
            # start with target chunk size and shrink on OOM
            chunk_size = min(target_chunk, len(seed_entries) - offset)
            oom_retry = 0
            while True:
                chunk = seed_entries[offset:offset + chunk_size]
                gids_in_chunk = [g for (g, _, _) in chunk]
                start_seed = int(min(s for (_, s, _) in chunk))
                logger.info(f"Propagating chunk at offset {offset}: {len(chunk)} objects (start frame {start_seed})")

                inference_state = None
                try:
                    # init state and add only this chunk's objects
                    inference_state = video_predictor.init_state(video_path=temp_dir)
                    for gid, seed_idx, seed_mask_2d in chunk:
                        mask_tensor = torch.from_numpy(seed_mask_2d)
                        _ = video_predictor.add_new_mask(
                            inference_state=inference_state,
                            frame_idx=int(seed_idx),
                            obj_id=int(gid),
                            mask=mask_tensor,
                        )

                    # forward propagation
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(
                            inference_state, start_frame_idx=start_seed, reverse=False
                        ):
                            masks_np = (out_mask_logits > 0.0).cpu().numpy()
                            for i, gid in enumerate(out_obj_ids):
                                if gid in mask_3d_volumes_global:
                                    insert_slice(mask_3d_volumes_global[gid],
                                                 masks_np[i].astype(np.uint8),
                                                 task['axis'], int(out_frame_idx))

                    # backward propagation if needed
                    if start_seed > 0:
                        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                            for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(
                                inference_state, start_frame_idx=start_seed, reverse=True
                            ):
                                masks_np = (out_mask_logits > 0.0).cpu().numpy()
                                for i, gid in enumerate(out_obj_ids):
                                    if gid in mask_3d_volumes_global:
                                        insert_slice(mask_3d_volumes_global[gid],
                                                     masks_np[i].astype(np.uint8),
                                                     task['axis'], int(out_frame_idx))

                    # success for this chunk
                    processed_gids.update(gids_in_chunk)
                    break  # exit OOM retry loop

                except Exception as e:
                    if _is_cuda_oom(e):
                        logger.warning(f"CUDA OOM during propagation of {len(chunk)} objects; shrinking chunk. Error: {e}")
                        # free state and caches
                        try:
                            if inference_state is not None:
                                video_predictor.reset_state(inference_state)
                        except Exception:
                            pass
                        inference_state = None
                        gc.collect()
                        torch.cuda.empty_cache()

                        if chunk_size <= min_chunk:
                            # give up on these objects; drop them to continue others
                            logger.error(f"Chunk size already at minimum ({min_chunk}) but still OOM. Skipping {len(chunk)} objects.")
                            # remove their preallocated volumes so they don't end up in outputs
                            for gid in gids_in_chunk:
                                if gid in mask_3d_volumes_global:
                                    del mask_3d_volumes_global[gid]
                            # do not add to processed_gids; they'll be absent downstream
                            break  # give up and move on
                        else:
                            # halve chunk and retry
                            chunk_size = max(min_chunk, chunk_size // 2)
                            oom_retry += 1
                            logger.info(f"Retrying with smaller chunk size: {chunk_size} (retry #{oom_retry})")
                            continue
                    else:
                        # non-OOM error -> log and skip these objects
                        logger.error(f"Error during propagation: {e}\n{traceback.format_exc()}")
                        try:
                            if inference_state is not None:
                                video_predictor.reset_state(inference_state)
                        except Exception:
                            pass
                        del inference_state
                        gc.collect()
                        torch.cuda.empty_cache()
                        # remove their preallocated volumes so they don't end up in outputs
                        for gid in gids_in_chunk:
                            if gid in mask_3d_volumes_global:
                                del mask_3d_volumes_global[gid]
                        break  # move on

                finally:
                    # Safely reset and clear 'inference_state' if it exists
                    try:
                        st = locals().get("inference_state", None)
                        if st is not None:
                            video_predictor.reset_state(st)
                    except Exception:
                        pass
                    inference_state = None
                    gc.collect()
                    torch.cuda.empty_cache()

            # advance to next chunk
            offset += len(chunk)

        # Post-process each object: keep largest component connected to its seed slice; apply min_voxel_count
        for gid in list(mask_3d_volumes_global.keys()):
            vol = mask_3d_volumes_global[gid]
            # skip volumes that were never propagated due to OOM skip
            if gid not in processed_gids:
                del mask_3d_volumes_global[gid]
                continue

            labeled, num_features = ndimage.label(vol)
            if num_features > 1:
                seed_idx = global_obj_to_seedidx[gid]
                original_mask_slice = extract_slice(vol, task['axis'], seed_idx)
                intersecting_components = []
                for lab in range(1, num_features + 1):
                    component_mask = (labeled == lab)
                    component_slice = extract_slice(component_mask, task['axis'], seed_idx)
                    if np.any(component_slice & original_mask_slice):
                        size = int(np.sum(component_mask))
                        intersecting_components.append((lab, size))
                if intersecting_components:
                    largest_label = max(intersecting_components, key=lambda x: x[1])[0]
                    mask_3d_volumes_global[gid] = (labeled == largest_label).astype(np.uint8)
                else:
                    logger.warning(f"No component intersects seed for gid={gid}; removing")
                    del mask_3d_volumes_global[gid]
                    continue
            # Min-voxel filter
            if np.sum(mask_3d_volumes_global.get(gid, 0)) < min_voxel_count:
                del mask_3d_volumes_global[gid]

        for init_slice_index in init_slice_indices:
            info = per_slice_info.get(init_slice_index, None)
            if not info:
                # No objects were seeded for this slice.
                continue

            slice_mask_volumes = {}
            # info["global_ids"] is in local order; local_id is position (starting at 1)
            for local_pos, gid in enumerate(info["global_ids"], start=1):
                if gid in mask_3d_volumes_global:
                    slice_mask_volumes[local_pos] = mask_3d_volumes_global[gid]

            if not slice_mask_volumes:
                logger.warning(f"All objects filtered or skipped for slice {init_slice_index}; skipping save")
                continue

            # Pack local object masks into the smallest number of label volumes.
            packed_volumes, obj_id_mapping = pack_masks_efficiently(
                slice_mask_volumes, original_shape
            )

            for vol_idx, packed_vol in enumerate(packed_volumes):
                packed_sitk = sitk.GetImageFromArray(packed_vol)
                packed_sitk.SetSpacing(sitk_image.GetSpacing())
                packed_sitk.SetOrigin(sitk_image.GetOrigin())
                packed_sitk.SetDirection(sitk_image.GetDirection())
                output_filename = (f"{task['base_filename']}_"
                                   f"axis{task['axis']}_slice{init_slice_index}_vol{vol_idx}.nii.gz")
                output_path = os.path.join(output_dir, output_filename)
                sitk.WriteImage(packed_sitk, output_path)
                logger.info(f"Saved packed volume to {output_path}")

            mapping_filename = (f"{task['base_filename']}_"
                                f"axis{task['axis']}_slice{init_slice_index}_mapping.json")
            mapping_path = os.path.join(output_dir, mapping_filename)
            mapping_info = {
                "original_objects": {
                    str(local_id): {
                        "volume_idx": obj_id_mapping[local_id][0],
                        "label_value": int(obj_id_mapping[local_id][1]),
                        "predicted_iou": float(info["meta"].get(local_id, {}).get("predicted_iou", 0.0)),
                        "stability_score": float(info["meta"].get(local_id, {}).get("stability_score", 0.0))
                    }
                    for local_id in obj_id_mapping.keys()
                },
                "total_volumes": len(packed_volumes),
                "total_objects": len(slice_mask_volumes),
                "source_slice": int(init_slice_index),
                "axis": int(task['axis']),
                "original_spacing": list(sitk_image.GetSpacing()),
                "original_size": list(sitk_image.GetSize()),
                "physical_interval_mm": physical_interval_mm,
                "min_pixel_count": min_pixel_count,
                "min_voxel_count": min_voxel_count
            }
            with open(mapping_path, 'w') as f:
                json.dump(mapping_info, f, indent=2)

        del image_array
        del image_array_3ch
        del body_mask
        del body_mask_array

        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")

        final_slice_check = get_unprocessed_slices(
            output_dir, task['base_filename'], task['axis'], all_slice_indices
        )
        if len(final_slice_check) == 0:
            update_checkpoint(checkpoint_path, task_id, "completed", lock=checkpoint_lock)
            logger.info(f"Task {task_id} fully completed - all slices processed")
        else:
            update_checkpoint(checkpoint_path, task_id, "partial",
                              f"{len(final_slice_check)} slices remaining", lock=checkpoint_lock)
            logger.info(f"Task {task_id} partially completed - {len(final_slice_check)} slices remaining")

        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"Task {task_id} completed, memory cleaned")
        return task_id, True

    except Exception as e:
        error_msg = f"Error processing task {task_id}: {e}\n{traceback.format_exc()}"
        logger.error(error_msg)
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory after error: {temp_dir}")
            except Exception as cleanup_error:
                logger.error(f"Failed to clean up temp directory: {cleanup_error}")
        update_checkpoint(checkpoint_path, task_id, "failed", str(e), lock=checkpoint_lock)
        gc.collect()
        torch.cuda.empty_cache()
        return task_id, False


def gpu_worker(gpu_id, tasks, fixed_params):
    """
    Worker function for a specific GPU.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = 'cuda:0'
    torch.cuda.set_device(0)
    logger.info(f"GPU {gpu_id}: Starting worker with {len(tasks)} tasks on {torch.cuda.get_device_name(0)}")
    dataset_config = DATASET_CONFIGS[fixed_params['dataset_name']]
    if 'dense_sampling_classes' in fixed_params and fixed_params['dense_sampling_classes'] is not None:
        dataset_config['dense_sampling_classes'] = fixed_params['dense_sampling_classes']
    start_time = time.time()
    mask_generator, video_predictor = load_sam2_models(
        sam2_checkpoint=fixed_params['sam2_checkpoint'],
        model_cfg=fixed_params['model_cfg'],
        device=device,
        dataset_config=dataset_config,
        torch_compile=fixed_params.get('torch_compile', False)
    )
    load_time = time.time() - start_time
    logger.info(f"GPU {gpu_id}: Model loading completed in {load_time:.2f} seconds")
    for i, task in enumerate(tasks):
        if not should_process_task(fixed_params['checkpoint_path'], task['task_id'], fixed_params['checkpoint_lock']):
            logger.info(f"Task {task['task_id']} already running or completed by another node, skipping")
            continue

        logger.info(f"GPU {gpu_id}: Processing task {i+1}/{len(tasks)}: {task['task_id']}")
        task_id, success = process_single_task(
            task=task,
            output_dir=fixed_params['output_dir'],
            mask_generator=mask_generator,
            video_predictor=video_predictor,
            physical_interval_mm=fixed_params['physical_interval_mm'],
            max_masks_per_slice=fixed_params['max_masks_per_slice'],
            dataset_config=dataset_config,
            device=device,
            checkpoint_path=fixed_params['checkpoint_path'],
            checkpoint_lock=fixed_params['checkpoint_lock'],
            min_slices_per_sample=fixed_params['min_slices_per_sample']
        )
        status = "completed" if success else "failed"
        logger.info(f"GPU {gpu_id}: Task {task_id} {status}")
        gc.collect()
        torch.cuda.empty_cache()
    logger.info(f"GPU {gpu_id}: Finished processing all tasks")


def distribute_tasks_to_gpus(all_tasks, num_gpus):
    """
    Evenly distribute tasks among available GPUs.
    """
    gpu_tasks = {i: [] for i in range(num_gpus)}
    for i, task in enumerate(all_tasks):
        gpu_id = i % num_gpus
        gpu_tasks[gpu_id].append(task)
    for gpu_id, tasks in gpu_tasks.items():
        logger.info(f"GPU {gpu_id}: Assigned {len(tasks)} tasks")
    return gpu_tasks


def main():
    """
    Main function for SAM2-based annotation-free 3D mask generation.
    Works on original image resolution (no resampling).
    SAM2 models are loaded once per GPU and reused across all images.
    Supports slice-level resume: already-processed slices are skipped on restart.
    """
    import argparse
    parser = argparse.ArgumentParser(
        description="MASS SAM2-based annotation-free 3D medical image mask generation"
    )
    parser.add_argument("--image", help="Path to a single 3D medical image (.nii / .nii.gz)")
    parser.add_argument("--image_dir", default=None,
                       help="Directory containing 3D medical images (.nii.gz)")
    parser.add_argument("--images", type=str, nargs='+', default=None,
                       help="Optional image names to process after filename stripping")
    parser.add_argument("--output_dir", required=True,
                       help="Directory to save output masks")
    parser.add_argument("--dataset_name", type=str, default='abdomen_ct',
                       help="Dataset name for SAM2 configuration. Available: "
                            "abdomen_ct, abdomen_mr, brain_mr, cardiac_mr, chest_ct, "
                            "totalseg_ct, totalseg_mr, lits, autopet_ct, autopet_suv, "
                            "structseg_head_oar (default: abdomen_ct)")
    parser.add_argument("--sam2_checkpoint", required=True,
                       help="Path to the SAM2 checkpoint file (e.g. sam2.1_hiera_large.pt)")
    parser.add_argument("--model_cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml",
                       help="Path to the SAM2 model config YAML (default: configs/sam2.1/sam2.1_hiera_l.yaml)")
    parser.add_argument("--gpu_ids", type=int, nargs='+', required=True,
                       help="GPU IDs to use (e.g. 0 1 2 3)")
    parser.add_argument("--axes", type=int, nargs='+', default=None,
                       help="Axes to process (0: axial, 1: sagittal, 2: coronal). If not specified with --auto_select_axes, will process axial only.")
    parser.add_argument("--dense_sampling_classes", type=int, nargs='+', default=None,
                       help="Class IDs for dense sampling in GT-aware mode (e.g., 2 3 4 for pancreas, spleen, kidney)")
    parser.add_argument("--auto_select_axes", action="store_true",
                       help="Automatically select best axis/axes based on image spacing")
    parser.add_argument("--isotropic_threshold", type=float, default=1.3,
                       help="Ratio threshold to consider spacing as isotropic (default: 1.3)")
    parser.add_argument("--process_all_axes_if_isotropic", action="store_true",
                       help="Process all axes for isotropic data instead of just the best one")
    parser.add_argument("--min_slices_per_sample", type=int, default=10,
                   help="Minimum number of slices to sample. If physical-based sampling yields fewer slices, "
                        "uniform sampling will be used instead (default: 10)")
    parser.add_argument("--physical_interval_mm", type=float, default=30.0,
                       help="Physical distance between selected slices in mm (default: 30.0mm)")
    parser.add_argument("--max_masks_per_slice", type=int, default=70,
                       help="Maximum masks per slice (default: 70)")
    parser.add_argument("--torch_compile", action="store_true",
                       help="Use torch.compile for optimization")
    args = parser.parse_args()
    if args.dataset_name not in DATASET_CONFIGS:
        logger.error(f"Dataset '{args.dataset_name}' not found in configurations. Available datasets: {list(DATASET_CONFIGS.keys())}")
        return
    if args.axes is None and not args.auto_select_axes:
        args.axes = [0]
    if args.image:
        image_paths = [Path(args.image)]
    else:
        if args.image_dir is None:
            raise ValueError("Either --image or --image_dir must be provided")
        # Treat standardized *_gt files as labels, not images to segment.
        image_paths = []
        for ext in ['.nii.gz', '.nii']:
            all_files = list(Path(args.image_dir).glob(f"**/*{ext}"))
            image_files = [f for f in all_files if not 'gt' in f.stem]
            image_paths.extend(image_files)
        image_paths = sorted(image_paths)
    if args.images:
        requested = {str(name) for name in args.images}

        def _matches_requested_name(path: str) -> bool:
            image_path = Path(path)
            aliases = {image_path.name, image_path.stem, image_path.stem.replace('.nii', '')}
            return bool(aliases & requested)

        image_paths = [path for path in image_paths if _matches_requested_name(path)]
        logger.info(f"Processing only specified images: {sorted(requested)}")
    if not image_paths:
        raise ValueError(f"No medical images found in {args.image_dir}")
    logger.info(f"Found {len(image_paths)} images to process (excluding _gt files)")
    dataset_config = DATASET_CONFIGS[args.dataset_name]
    modality = dataset_config.get('modality', 'CT')
    logger.info(f"Using dataset configuration: {args.dataset_name} (Modality: {modality})")
    if modality in ['MRI', 'MR']:
        logger.info(f"  - MRI quantile ranges: {dataset_config['enhancement']['quantile_ranges']}")
    else:
        logger.info(f"  - CT window ranges: {dataset_config['enhancement']['window_ranges']}")
    logger.info(f"  - CLAHE enabled: {dataset_config['enhancement'].get('enable_clahe', False)}")
    if dataset_config['enhancement'].get('enable_clahe', False):
        logger.info(f"  - CLAHE settings: clip_limit={dataset_config['enhancement']['clahe_clip_limit']}, "
                    f"grid_size={dataset_config['enhancement']['clahe_grid_size']}")
    logger.info(f"  - Min pixel count (2D): {dataset_config['min_pixel_count']}")
    logger.info(f"  - Min voxel count (3D): {dataset_config['min_voxel_count']}")
    if args.dense_sampling_classes:
        logger.info(f"  - GT-guided dense sampling enabled for classes: {args.dense_sampling_classes}")
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, "processing_checkpoint.json")
    checkpoint = reset_running_tasks_for_resume(checkpoint_path)
    if checkpoint:
        completed = sum(1 for t in checkpoint.values() if t["status"] == "completed")
        partial = sum(1 for t in checkpoint.values() if t["status"] == "partial")
        total = len(checkpoint)
        logger.info(f"Resuming from checkpoint: {completed} completed, {partial} partial, {total} total tasks")
    all_tasks = create_task_list(
        image_paths,
        axes=args.axes,
        auto_select_axes=args.auto_select_axes,
        isotropic_threshold=args.isotropic_threshold,
        process_all_axes_if_isotropic=args.process_all_axes_if_isotropic
    )
    if args.auto_select_axes:
        logger.info(f"Using automatic axis selection with isotropic threshold: {args.isotropic_threshold}")
        if args.process_all_axes_if_isotropic:
            logger.info("Will process all axes for isotropic data")
    else:
        axes_names = [['axial', 'sagittal', 'coronal'][i] for i in args.axes]
        logger.info(f"Using manually specified axes: {axes_names}")
    logger.info(f"Created {len(all_tasks)} total tasks:")
    logger.info(f"  - Images: {len(image_paths)}")
    logger.info(f"  - Physical interval: {args.physical_interval_mm}mm between slices")
    logger.info(f"  - Processing at ORIGINAL RESOLUTION")
    pending_tasks = []
    for task in all_tasks:
        # Failed or partial tasks are retried; completed tasks are skipped.
        if task['task_id'] not in checkpoint:
            pending_tasks.append(task)
        elif checkpoint[task['task_id']]['status'] == 'completed':
            continue
        elif checkpoint[task['task_id']]['status'] in ['failed', 'partial', 'running']:
            pending_tasks.append(task)
        else:
            pending_tasks.append(task)
    if not pending_tasks:
        logger.info("All tasks already completed!")
        return
    if len(args.gpu_ids) == 1:
        gpu_id = args.gpu_ids[0]
        device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)
        logger.info(f"Using single GPU {gpu_id} ({torch.cuda.get_device_name(gpu_id)})")
        dataset_config = DATASET_CONFIGS[args.dataset_name]
        if args.dense_sampling_classes is not None:
            dataset_config['dense_sampling_classes'] = args.dense_sampling_classes
        logger.info("Loading SAM2 models for single GPU processing...")
        start_time = time.time()
        mask_generator, video_predictor = load_sam2_models(
            sam2_checkpoint=args.sam2_checkpoint,
            model_cfg=args.model_cfg,
            device=device,
            dataset_config=dataset_config,
            torch_compile=args.torch_compile
        )
        load_time = time.time() - start_time
        logger.info(f"Model loading completed in {load_time:.2f} seconds")
        for i, task in enumerate(pending_tasks):
            if not should_process_task(checkpoint_path, task['task_id'], None):
                logger.info(f"Task {task['task_id']} already handled by another node, skipping")
                continue

            logger.info(f"Processing task {i+1}/{len(pending_tasks)}: {task['task_id']}")
            process_single_task(
                task=task,
                output_dir=args.output_dir,
                mask_generator=mask_generator,
                video_predictor=video_predictor,
                physical_interval_mm=args.physical_interval_mm,
                max_masks_per_slice=args.max_masks_per_slice,
                dataset_config=dataset_config,
                device=device,
                checkpoint_path=checkpoint_path,
                checkpoint_lock=None,
                min_slices_per_sample=args.min_slices_per_sample
            )
            gc.collect()
            torch.cuda.empty_cache()
    else:
        logger.info(f"Using {len(args.gpu_ids)} GPUs for parallel processing")
        logger.info("Each GPU will load the model once and process its assigned tasks")
        gpu_tasks = distribute_tasks_to_gpus(pending_tasks, len(args.gpu_ids))
        multiprocessing.set_start_method('spawn', force=True)
        manager = Manager()
        checkpoint_lock = manager.Lock()
        fixed_params = {
            "output_dir": args.output_dir,
            "sam2_checkpoint": args.sam2_checkpoint,
            "model_cfg": args.model_cfg,
            "physical_interval_mm": args.physical_interval_mm,
            "max_masks_per_slice": args.max_masks_per_slice,
            "dataset_name": args.dataset_name,
            "dense_sampling_classes": args.dense_sampling_classes,
            "checkpoint_path": checkpoint_path,
            "checkpoint_lock": checkpoint_lock,
            "torch_compile": args.torch_compile,
            "min_slices_per_sample": args.min_slices_per_sample
        }
        processes = []
        for gpu_id in args.gpu_ids:
            tasks_for_gpu = gpu_tasks[args.gpu_ids.index(gpu_id)]
            if tasks_for_gpu:
                p = Process(target=gpu_worker, args=(gpu_id, tasks_for_gpu, fixed_params))
                p.start()
                processes.append(p)
        for p in processes:
            p.join()
    checkpoint = load_checkpoint(checkpoint_path)
    completed = sum(1 for t in checkpoint.values() if t["status"] == "completed")
    partial = sum(1 for t in checkpoint.values() if t["status"] == "partial")
    failed = sum(1 for t in checkpoint.values() if t["status"] == "failed")
    logger.info(f"\nProcessing Summary:")
    logger.info(f"Total tasks: {len(checkpoint)}")
    logger.info(f"Completed: {completed}")
    logger.info(f"Partial: {partial}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success rate: {100.0 * completed / len(checkpoint):.1f}%")
    if failed > 0:
        logger.info("\nFailed tasks:")
        for task_id, info in checkpoint.items():
            if info["status"] == "failed":
                logger.info(f"  - {task_id}: {info.get('error', 'Unknown error')}")
    if partial > 0:
        logger.info("\nPartial tasks:")
        for task_id, info in checkpoint.items():
            if info["status"] == "partial":
                logger.info(f"  - {task_id}: {info.get('error', 'Some slices remaining')}")


if __name__ == "__main__":
    main()
