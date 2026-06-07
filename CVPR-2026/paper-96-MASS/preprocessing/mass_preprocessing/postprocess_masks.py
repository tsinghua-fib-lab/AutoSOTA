#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Postprocess SAM2 auto-generated masks.

This script deduplicates overlapping SAM2 masks with NMS, filters tiny or
low-quality components, and packs the surviving binary masks into compact NIfTI
label volumes before the final resampling/export stage.
"""

import os
import numpy as np
import torch
import SimpleITK as sitk
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
import glob
from collections import defaultdict
import multiprocessing as mp
import time
import gc
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_pattern_tokens(pattern: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Robustly parse a pattern like:
      <sample_name>_axis<d>_slice<k>
    and return (sample_name, 'axis<d>', 'slice<k>').
    Everything before '_axis' is treated as the sample name.
    """
    parts = pattern.split('_')

    axis_idx = next((i for i, p in enumerate(parts) if p.startswith('axis')), None)
    axis_info = None
    slice_info = None

    if axis_idx is not None:
        # Sample name is *everything* before the 'axis' segment (can contain underscores)
        sample_name = '_'.join(parts[:axis_idx]) or parts[0]
        axis_info = parts[axis_idx]

        for p in parts[axis_idx + 1:]:
            if p.startswith('slice'):
                slice_info = p
                break
    else:
        sample_name = parts[0]

    return sample_name, axis_info, slice_info


def build_pattern_suffix_from_pattern(pattern: str) -> str:
    """
    Build the output suffix used in filenames,
    preferring 'axisX_sliceY', else 'axisX', else 'filtered'.
    """
    _, axis_info, slice_info = parse_pattern_tokens(pattern)
    if axis_info and slice_info:
        return f"{axis_info}_{slice_info}"
    elif axis_info:
        return axis_info
    else:
        return "filtered"


@dataclass
class GPUMemoryManager:
    """Manages GPU memory allocation and batch sizes dynamically."""
    device: torch.device
    reserved_memory_gb: float = 5.0  # Reserve 5GB for other operations
    
    def __post_init__(self):
        self.total_memory = torch.cuda.get_device_properties(self.device).total_memory
        self.usable_memory = self.total_memory - (self.reserved_memory_gb * 1024**3)
    
    def get_available_memory(self) -> int:
        """Get currently available GPU memory in bytes."""
        return torch.cuda.mem_get_info(self.device)[0]
    
    def estimate_batch_size(self, element_size: int, elements_per_batch: int) -> int:
        """Estimate optimal batch size based on available memory."""
        available = self.get_available_memory()
        memory_per_batch = element_size * elements_per_batch
        
        # Use 70% of available memory to leave buffer
        safe_batches = int((available * 0.7) / memory_per_batch)
        return max(1, safe_batches)
    
    def cleanup(self):
        """Force GPU memory cleanup."""
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(self.device)


def downsample_masks_streaming(masks: List[np.ndarray], factor: int, 
                               device: torch.device, memory_manager: GPUMemoryManager,
                               dtype: torch.dtype = torch.bool) -> torch.Tensor:
    """
    Downsample masks with streaming approach to minimize memory usage.
    
    Args:
        masks: List of mask arrays
        factor: Downsampling factor
        device: GPU device
        memory_manager: GPU memory manager
        dtype: Data type for masks (bool uses 1/8 memory of float32)
    
    Returns:
        Tensor of downsampled masks
    """
    if factor == 1:
        return torch.stack([torch.from_numpy(m > 0).to(device, dtype=dtype) for m in masks])
    
    n_masks = len(masks)
    logger.info(f"Downsampling {n_masks} masks by factor {factor} using {dtype}")
    
    H, W, D = masks[0].shape
    
    pad_h = (factor - H % factor) % factor
    pad_w = (factor - W % factor) % factor
    pad_d = (factor - D % factor) % factor
    
    # Output dimensions
    out_h = (H + pad_h) // factor
    out_w = (W + pad_w) // factor
    out_d = (D + pad_d) // factor
    
    # Estimate memory per mask
    input_memory_per_mask = H * W * D * 4  # float32 input
    output_memory_per_mask = out_h * out_w * out_d * (1 if dtype == torch.bool else 4)
    total_memory_per_mask = input_memory_per_mask + output_memory_per_mask
    
    # Dynamic batch size based on available memory
    batch_size = memory_manager.estimate_batch_size(total_memory_per_mask, 1)
    batch_size = min(batch_size, 100)  # Cap at 100 for stability
    
    logger.info(f"  Using dynamic batch size: {batch_size}")
    
    # Pre-allocate output
    downsampled_masks = torch.zeros((n_masks, out_h, out_w, out_d), 
                                   dtype=dtype, device=device)
    
    for batch_start in range(0, n_masks, batch_size):
        batch_end = min(batch_start + batch_size, n_masks)
        
        batch_list = []
        for i in range(batch_start, batch_end):
            mask_bool = torch.from_numpy(masks[i] > 0).to(device)
            mask_4d = mask_bool.unsqueeze(0).unsqueeze(0).float()  # Need float for pooling
            
            if pad_h > 0 or pad_w > 0 or pad_d > 0:
                mask_4d = torch.nn.functional.pad(mask_4d, (0, pad_d, 0, pad_w, 0, pad_h))
            
            batch_list.append(mask_4d)
        
        # Stack and downsample
        batch_tensor = torch.cat(batch_list, dim=0)
        
        # Use max pooling for downsampling
        downsampled_batch = torch.nn.functional.max_pool3d(
            batch_tensor,
            kernel_size=factor,
            stride=factor,
            padding=0
        )
        
        downsampled_masks[batch_start:batch_end] = downsampled_batch.squeeze(1) > 0
        
        del batch_tensor, batch_list, downsampled_batch
        
        if batch_end % 500 == 0:
            logger.info(f"  Downsampled {batch_end}/{n_masks} masks")
            memory_manager.cleanup()
    
    return downsampled_masks


def compute_iou_streaming(query_masks: torch.Tensor, target_masks: torch.Tensor,
                         chunk_size: int = 100) -> torch.Tensor:
    """
    Compute IoU between query and target masks using streaming approach.
    
    Args:
        query_masks: Query masks [N_q, H, W, D]
        target_masks: Target masks [N_t, H, W, D]
        chunk_size: Size of chunks for computation
        
    Returns:
        IoU matrix [N_q, N_t]
    """
    n_query = query_masks.shape[0]
    n_target = target_masks.shape[0]
    device = query_masks.device
    
    query_flat = query_masks.view(n_query, -1).float()
    target_flat = target_masks.view(n_target, -1).float()
    
    # Pre-compute sums for efficiency
    query_sums = query_flat.sum(dim=1, keepdim=True)  # [N_q, 1]
    target_sums = target_flat.sum(dim=1)  # [N_t]
    
    # Allocate output
    iou_matrix = torch.zeros(n_query, n_target, device=device, dtype=torch.float32)
    
    for i in range(0, n_query, chunk_size):
        end_i = min(i + chunk_size, n_query)
        query_chunk = query_flat[i:end_i]  # [chunk_q, HWD]
        query_sum_chunk = query_sums[i:end_i]  # [chunk_q, 1]
        
        for j in range(0, n_target, chunk_size):
            end_j = min(j + chunk_size, n_target)
            target_chunk = target_flat[j:end_j]  # [chunk_t, HWD]
            target_sum_chunk = target_sums[j:end_j]  # [chunk_t]
            
            intersection = torch.mm(query_chunk, target_chunk.t())  # [chunk_q, chunk_t]
            
            union = query_sum_chunk + target_sum_chunk.unsqueeze(0) - intersection
            
            chunk_iou = intersection / (union + 1e-6)
            
            iou_matrix[i:end_i, j:end_j] = chunk_iou
    
    return iou_matrix


def apply_nms_and_downsample(masks: List[np.ndarray], metadata: List[Dict], 
                            iou_threshold: float, device: torch.device,
                            memory_manager: GPUMemoryManager,
                            downsample_factor: int = 4) -> Tuple[List[int], torch.Tensor]:
    """
    Apply NMS and return both keep indices and downsampled masks for reuse.
    
    Returns:
        Tuple of (keep_indices, downsampled_masks_tensor)
    """
    if len(masks) == 0:
        return [], None
    
    n_masks = len(masks)
    logger.info(f"Applying memory-efficient NMS to {n_masks} masks")
    
    start_time = time.time()
    
    with torch.no_grad():
        downsampled_masks = downsample_masks_streaming(
            masks, downsample_factor, device, memory_manager, dtype=torch.bool
        )
    
    mask_sizes = downsampled_masks.sum(dim=(1, 2, 3)).float()
    sorted_indices = torch.argsort(mask_sizes, descending=True)
    
    keep_mask = torch.ones(n_masks, dtype=torch.bool, device=device)
    
    for idx in range(n_masks):
        if idx % 100 == 0:
            logger.info(f"  Processing mask {idx}/{n_masks} ({idx/n_masks*100:.1f}%)")
            memory_manager.cleanup()
        
        current_idx = sorted_indices[idx]
        
        if not keep_mask[current_idx]:
            continue
        
        remaining_indices = sorted_indices[idx+1:]
        remaining_valid = keep_mask[remaining_indices]
        
        if not remaining_valid.any():
            continue
        
        valid_indices = remaining_indices[remaining_valid]
        
        if len(valid_indices) == 0:
            continue
        
        current_mask = downsampled_masks[current_idx:current_idx+1]
        
        chunk_size = min(memory_manager.estimate_batch_size(
            downsampled_masks[0].numel(), 1
        ), 500)
        
        for chunk_start in range(0, len(valid_indices), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(valid_indices))
            chunk_indices = valid_indices[chunk_start:chunk_end]
            
            chunk_masks = downsampled_masks[chunk_indices]
            
            current_flat = current_mask.view(1, -1).float()
            chunk_flat = chunk_masks.view(len(chunk_indices), -1).float()
            
            # Intersection
            intersection = torch.mv(chunk_flat, current_flat.squeeze())
            
            # Union
            current_sum = current_flat.sum()
            chunk_sums = chunk_flat.sum(dim=1)
            union = current_sum + chunk_sums - intersection
            
            # IoU
            iou = intersection / (union + 1e-6)
            
            # Suppress high IoU masks
            suppress_mask = iou > iou_threshold
            keep_mask[chunk_indices[suppress_mask]] = False
    
    keep_indices = torch.where(keep_mask)[0].cpu().numpy().tolist()
    
    total_time = time.time() - start_time
    logger.info(f"Memory-efficient NMS completed in {total_time:.2f}s")
    logger.info(f"Kept {len(keep_indices)}/{n_masks} masks ({len(keep_indices)/n_masks*100:.1f}%)")
    
    return keep_indices, downsampled_masks


def pack_masks_with_downsampled(masks: List[np.ndarray], metadata: List[Dict], 
                               original_shape: tuple, device: torch.device,
                               memory_manager: GPUMemoryManager,
                               downsampled_masks: torch.Tensor,
                               keep_indices: List[int]) -> Tuple[List[np.ndarray], Dict]:
    """
    Memory-efficient mask packing using pre-computed downsampled masks.
    
    Args:
        masks: Original resolution masks
        metadata: Mask metadata
        original_shape: Original volume shape
        device: GPU device
        memory_manager: Memory manager
        downsampled_masks: Pre-computed downsampled masks from NMS
        keep_indices: Indices of masks to keep from NMS
    """
    num_masks = len(masks)
    if num_masks == 0:
        return [], {}
    
    logger.info(f"Packing {num_masks} masks using pre-computed downsampled masks")
    
    filtered_downsampled = downsampled_masks[keep_indices]
    
    overlap_time = time.time()
    overlap_matrix = torch.zeros((num_masks, num_masks), dtype=torch.bool, device=device)
    
    chunk_size = min(memory_manager.estimate_batch_size(
        filtered_downsampled[0].numel(), 2
    ), 400)
    
    logger.info(f"  Computing overlaps with chunk size {chunk_size}")
    
    for i in range(0, num_masks, chunk_size):
        end_i = min(i + chunk_size, num_masks)
        
        # Stack chunk of masks
        chunk_i = filtered_downsampled[i:end_i]
        chunk_i_flat = chunk_i.view(end_i - i, -1)
        
        for j in range(i, num_masks, chunk_size):
            end_j = min(j + chunk_size, num_masks)
            
            chunk_j = filtered_downsampled[j:end_j]
            chunk_j_flat = chunk_j.view(end_j - j, -1)
            
            # Fast overlap detection using matrix multiplication
            with torch.cuda.amp.autocast(dtype=torch.float16):
                intersection = torch.mm(chunk_i_flat.float(), chunk_j_flat.t().float())
            
            has_overlap = intersection > 0
            
            overlap_matrix[i:end_i, j:end_j] = has_overlap
            if i != j:
                overlap_matrix[j:end_j, i:end_i] = has_overlap.t()
        
        if end_i % 500 == 0:
            logger.info(f"    Computed overlaps for {end_i}/{num_masks} masks ({(end_i/num_masks)*100:.1f}%)")
    
    overlap_matrix_cpu = overlap_matrix.cpu().numpy()
    
    del filtered_downsampled, overlap_matrix
    torch.cuda.empty_cache()
    
    overlap_time = time.time() - overlap_time
    logger.info(f"  Overlap computation completed in {overlap_time:.2f}s")
    
    coloring_time = time.time()
    colors = np.full(num_masks, -1, dtype=np.int16)
    max_color = -1
    
    for i in range(num_masks):
        neighbors = np.where(overlap_matrix_cpu[i])[0]
        neighbor_colors = colors[neighbors[neighbors < i]]
        neighbor_colors = neighbor_colors[neighbor_colors >= 0]
        
        if len(neighbor_colors) == 0:
            colors[i] = 0
        else:
            unique_colors = np.unique(neighbor_colors)
            if len(unique_colors) == 0 or unique_colors[0] > 0:
                colors[i] = 0
            else:
                gaps = np.diff(unique_colors)
                gap_idx = np.where(gaps > 1)[0]
                if len(gap_idx) > 0:
                    colors[i] = unique_colors[gap_idx[0]] + 1
                else:
                    colors[i] = unique_colors[-1] + 1
        
        max_color = max(max_color, colors[i])
    
    coloring_time = time.time() - coloring_time
    logger.info(f"  Graph coloring completed in {coloring_time:.2f}s")
    
    # Pack masks by color
    num_colors = max_color + 1
    packed_volumes = []
    obj_id_mapping = {}
    
    logger.info(f"  Packing into {num_colors} volumes")
    packing_time = time.time()
    
    for color in range(num_colors):
        packed_volume = np.zeros(original_shape, dtype=np.uint16)
        label = 1
        
        color_indices = np.where(colors == color)[0]
        
        for i in color_indices:
            packed_volume[masks[i] > 0] = label
            
            source_pattern = metadata[i].get('pattern', 'unknown')
            new_obj_id = f"{source_pattern}_filtered_{i}"
            
            obj_id_mapping[new_obj_id] = {
                'volume_idx': int(color),
                'label_value': int(label),
                'original_metadata': {
                    'obj_id': metadata[i].get('obj_id', ''),
                    'pattern': metadata[i].get('pattern', 'unknown'),
                    'vol_idx': int(metadata[i].get('vol_idx', -1)),
                    'label_value': int(metadata[i].get('label_value', -1)),
                    'source_slice': int(metadata[i].get('source_slice', -1)),
                    'axis': int(metadata[i].get('axis', -1))
                }
            }
            label += 1
        
        packed_volumes.append(packed_volume)
    
    packing_time = time.time() - packing_time
    logger.info(f"  Packing completed in {packing_time:.2f}s")
    
    total_time = overlap_time + coloring_time + packing_time
    logger.info(f"Packed {num_masks} masks into {len(packed_volumes)} volumes in {total_time:.2f}s total")
    
    return packed_volumes, obj_id_mapping


def process_single_image(image_name: str, pattern_groups: List[Dict], 
                        output_dir: str, iou_threshold: float,
                        device: torch.device, memory_manager: GPUMemoryManager,
                        downsample_factor: int = 4):
    """
    Process masks for a single image with memory-efficient approach and reused downsampling.
    """
    if check_if_processed(image_name, pattern_groups, output_dir):
        logger.info(f"Skipping {image_name} - already processed (found existing output)")
        return 0, 0  # Return 0s to indicate skipped

    logger.info(f"Processing {image_name} with {len(pattern_groups)} slice patterns")
    
    masks, metadata, reference_sitk = load_masks_from_patterns(pattern_groups)
    original_shape = sitk.GetArrayFromImage(reference_sitk).shape
    
    logger.info(f"Loaded {len(masks)} individual masks from all slices")
    
    if len(masks) == 0:
        logger.warning(f"No masks found for {image_name}")
        return 0, 0
    
    keep_indices, downsampled_masks = apply_nms_and_downsample(
        masks, metadata, iou_threshold, device, 
        memory_manager, downsample_factor
    )
    
    filtered_masks = [masks[i] for i in keep_indices]
    filtered_metadata = [metadata[i] for i in keep_indices]
    
    original_count = len(masks)
    del masks
    memory_manager.cleanup()
    
    # Pack filtered masks
    if len(filtered_masks) == 0:
        logger.warning(f"No masks remaining after NMS for {image_name}")
        return original_count, 0
    
    # Use memory-efficient packing with pre-computed downsampled masks
    packed_volumes, obj_id_mapping = pack_masks_with_downsampled(
        filtered_masks, filtered_metadata, original_shape, device, memory_manager,
        downsampled_masks, keep_indices
    )
    
    del downsampled_masks
    memory_manager.cleanup()
    
    first_pattern = pattern_groups[0]['pattern']
    pattern_suffix = build_pattern_suffix_from_pattern(first_pattern)
    
    save_filtered_masks(packed_volumes, obj_id_mapping, reference_sitk,
                       output_dir, image_name, pattern_suffix)
    
    return original_count, len(keep_indices)


def parse_mask_files(mask_dir: str) -> Dict[str, List[Dict]]:
    """Parse mask directory and group files by image (sample name before '_axis')."""
    all_files = os.listdir(mask_dir)

    mapping_files = {}
    vol_files_by_pattern = defaultdict(list)

    for filename in all_files:
        if filename.endswith('_mapping.json'):
            # Pattern is everything before the "_mapping.json" suffix
            pattern = filename[:-13]
            mapping_files[pattern] = os.path.join(mask_dir, filename)
        elif '_vol' in filename and filename.endswith('.nii.gz'):
            vol_pos = filename.rfind('_vol')
            if vol_pos > 0:
                pattern = filename[:vol_pos]  # '<sample>_axisX_sliceY'
                vol_files_by_pattern[pattern].append(os.path.join(mask_dir, filename))

    image_patterns = defaultdict(set)

    for pattern in mapping_files:
        if pattern not in vol_files_by_pattern:
            continue

        sample_name, _, _ = parse_pattern_tokens(pattern)
        image_patterns[sample_name].add(pattern)

    image_masks = {}
    for image_name, patterns in image_patterns.items():
        pattern_list = []
        for pattern in sorted(patterns):
            vol_files = sorted(vol_files_by_pattern[pattern])
            mapping_file = mapping_files[pattern]
            pattern_list.append({
                'pattern': pattern,
                'mapping_file': mapping_file,
                'vol_files': vol_files
            })
        image_masks[image_name] = pattern_list

    for image_name, patterns in image_masks.items():
        total_files = sum(len(p['vol_files']) for p in patterns)
        logger.info(f"{image_name}: {len(patterns)} patterns, {total_files} volume files")

    return image_masks


def load_masks_from_patterns(pattern_groups: List[Dict]) -> Tuple[List[np.ndarray], List[Dict], sitk.Image]:
    """Load all masks from a list of pattern groups."""
    all_masks = []
    all_metadata = []
    reference_sitk = None
    
    for pattern_info in pattern_groups:
        pattern = pattern_info['pattern']
        mapping_file = pattern_info['mapping_file']
        vol_files = pattern_info['vol_files']
        
        logger.info(f"Loading pattern {pattern} with {len(vol_files)} volumes")
        
        with open(mapping_file, 'r') as f:
            mapping_info = json.load(f)
        
        volumes = {}
        for vol_file in vol_files:
            vol_idx = int(vol_file.split('_vol')[-1].split('.nii.gz')[0])
            
            vol_sitk = sitk.ReadImage(vol_file)
            if reference_sitk is None:
                reference_sitk = vol_sitk
            
            vol_array = sitk.GetArrayFromImage(vol_sitk)
            volumes[vol_idx] = vol_array
        
        masks_extracted = 0
        for obj_id, obj_info in mapping_info['original_objects'].items():
            vol_idx = obj_info['volume_idx']
            label_value = obj_info['label_value']
            
            if vol_idx in volumes:
                vol_array = volumes[vol_idx]
                mask = (vol_array == label_value).astype(np.bool_)
                
                if np.any(mask):
                    all_masks.append(mask)
                    all_metadata.append({
                        'obj_id': obj_id,
                        'pattern': pattern,
                        'vol_idx': vol_idx,
                        'label_value': label_value,
                        'source_slice': mapping_info.get('source_slice', -1),
                        'axis': mapping_info.get('axis', -1)
                    })
                    masks_extracted += 1
        
        logger.info(f"  Extracted {masks_extracted} non-empty masks from pattern {pattern}")
        del volumes
    
    return all_masks, all_metadata, reference_sitk


def save_filtered_masks(packed_volumes: List[np.ndarray], obj_id_mapping: Dict,
                       reference_sitk: sitk.Image, output_dir: str, 
                       image_name: str, pattern_suffix: str):
    """Save filtered masks in the same format as input."""
    output_pattern = f"{image_name}_{pattern_suffix}_nms"
    
    for vol_idx, volume in enumerate(packed_volumes):
        vol_sitk = sitk.GetImageFromArray(volume)
        vol_sitk.SetSpacing(reference_sitk.GetSpacing())
        vol_sitk.SetOrigin(reference_sitk.GetOrigin())
        vol_sitk.SetDirection(reference_sitk.GetDirection())
        
        output_filename = f"{output_pattern}_vol{vol_idx}.nii.gz"
        output_path = os.path.join(output_dir, output_filename)
        sitk.WriteImage(vol_sitk, output_path)
    
    slice_stats = defaultdict(int)
    for obj_id, obj_info in obj_id_mapping.items():
        pattern = obj_info['original_metadata'].get('pattern', 'unknown')
        slice_stats[pattern] += 1
    
    mapping_info = {
        'original_objects': obj_id_mapping,
        'total_volumes': len(packed_volumes),
        'total_objects': len(obj_id_mapping),
        'source_slices': dict(slice_stats),
        'processing_info': {
            'type': 'NMS_filtered',
            'image_name': image_name
        }
    }
    
    mapping_filename = f"{output_pattern}_mapping.json"
    mapping_path = os.path.join(output_dir, mapping_filename)
    
    with open(mapping_path, 'w') as f:
        json.dump(mapping_info, f, indent=2)
    
    logger.info(f"Saved {len(packed_volumes)} filtered volumes for {image_name}")
    logger.info(f"Masks from {len(slice_stats)} different slices retained")


def worker_process(task_queue: mp.Queue, result_queue: mp.Queue, 
                  worker_id: int, gpu_id: int, output_dir: str,
                  iou_threshold: float, downsample_factor: int):
    """Worker process for parallel processing with memory management."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")
    
    memory_manager = GPUMemoryManager(device)
    
    logger.info(f"Worker {worker_id} started on GPU {gpu_id}")
    logger.info(f"  Total GPU memory: {memory_manager.total_memory / 1024**3:.1f} GB")
    logger.info(f"  Usable GPU memory: {memory_manager.usable_memory / 1024**3:.1f} GB")
    
    while True:
        try:
            task = task_queue.get(timeout=1)
            
            if task is None:
                break
            
            image_name, pattern_groups = task
            
            try:
                masks_before, masks_after = process_single_image(
                    image_name, pattern_groups, output_dir, 
                    iou_threshold, device, memory_manager,
                    downsample_factor
                )
                
                result_queue.put((True, image_name, masks_before, masks_after))
                
                memory_manager.cleanup()
                
            except Exception as e:
                logger.error(f"Error processing {image_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                result_queue.put((False, image_name, 0, 0))
                
        except mp.queues.Empty:
            continue
    
    logger.info(f"Worker {worker_id} finished")


def check_if_processed(image_name: str, pattern_groups: List[Dict], output_dir: str) -> bool:
    """
    Check if an image has already been processed by looking for the output mapping file.
    
    Args:
        image_name: Name of the image
        pattern_groups: List of pattern groups for this image
        output_dir: Output directory
        
    Returns:
        True if already processed, False otherwise
    """
    # Recreate the pattern suffix logic from process_single_image
    if not pattern_groups:
        return False

    first_pattern = pattern_groups[0]['pattern']
    pattern_suffix = build_pattern_suffix_from_pattern(first_pattern)

    output_pattern = f"{image_name}_{pattern_suffix}_nms"
    mapping_filename = f"{output_pattern}_mapping.json"
    mapping_path = os.path.join(output_dir, mapping_filename)

    return os.path.exists(mapping_path)





def main():
    parser = argparse.ArgumentParser(
        description="Memory-efficient NMS-based mask deduplication for SAM2"
    )
    
    parser.add_argument("--mask_dir", type=str, required=True,
                       help="Directory containing mask files")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save filtered masks")
    parser.add_argument("--iou_threshold", type=float, default=0.95,
                       help="IoU threshold for considering masks as duplicates")
    parser.add_argument("--gpu_ids", type=int, nargs='+', default=[0],
                       help="GPU IDs to use (e.g., 0 1 2 3)")
    parser.add_argument("--downsample_factor", type=int, default=4,
                       help="Downsampling factor for NMS and overlap computation")
    parser.add_argument("--images", type=str, nargs='+', default=None,
                       help="Specific image names to process")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Parsing mask files in {args.mask_dir}")
    image_masks = parse_mask_files(args.mask_dir)
    logger.info(f"Found masks for {len(image_masks)} images")
    
    if args.images:
        image_masks = {k: v for k, v in image_masks.items() if k in args.images}
        logger.info(f"Processing only specified images: {args.images}")
    
    original_count = len(image_masks)
    image_masks = {
        k: v for k, v in image_masks.items() 
        if not check_if_processed(k, v, args.output_dir)
    }
    skipped_count = original_count - len(image_masks)
    if skipped_count > 0:
        logger.info(f"Found {skipped_count} already processed images, skipping them")
    logger.info(f"Images to process: {len(image_masks)}")
    

    tasks = list(image_masks.items())
    
    if len(args.gpu_ids) == 1:
        device = torch.device(f"cuda:{args.gpu_ids[0]}")
        memory_manager = GPUMemoryManager(device)
        
        logger.info(f"Single GPU processing on device {args.gpu_ids[0]}")
        logger.info(f"Total GPU memory: {memory_manager.total_memory / 1024**3:.1f} GB")
        
        total_before = 0
        total_after = 0
        
        for image_name, pattern_groups in tasks:
            masks_before, masks_after = process_single_image(
                image_name, pattern_groups, args.output_dir,
                args.iou_threshold, device, memory_manager,
                args.downsample_factor
            )
            total_before += masks_before
            total_after += masks_after
            
            memory_manager.cleanup()
        
        logger.info(f"\nOverall Summary:")
        logger.info(f"Total masks before NMS: {total_before}")
        logger.info(f"Total masks after NMS: {total_after}")
        if total_before > 0:
            logger.info(f"Reduction: {(1 - total_after/total_before)*100:.1f}%")
        
    else:
        mp.set_start_method('spawn', force=True)
        
        task_queue = mp.Queue()
        result_queue = mp.Queue()
        
        for task in tasks:
            task_queue.put(task)
        
        for _ in range(len(args.gpu_ids)):
            task_queue.put(None)
        
        # Start workers
        processes = []
        for i, gpu_id in enumerate(args.gpu_ids):
            p = mp.Process(
                target=worker_process,
                args=(task_queue, result_queue, i, gpu_id,
                      args.output_dir, args.iou_threshold, 
                      args.downsample_factor)
            )
            p.start()
            processes.append(p)
        
        results = []
        for _ in range(len(tasks)):
            result = result_queue.get()
            results.append(result)
        
        # Wait for workers
        for p in processes:
            p.join()
        
        total_before = sum(r[2] for r in results if r[0])
        total_after = sum(r[3] for r in results if r[0])
        failed = sum(1 for r in results if not r[0])
        
        logger.info(f"\nOverall Summary:")
        logger.info(f"Successfully processed: {len(results) - failed}/{len(results)} images")
        logger.info(f"Total masks before NMS: {total_before}")
        logger.info(f"Total masks after NMS: {total_after}")
        if total_before > 0:
            logger.info(f"Reduction: {(1 - total_after/total_before)*100:.1f}%")
    
    logger.info("Memory-efficient NMS deduplication complete!")


if __name__ == "__main__":
    main()
