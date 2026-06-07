#!/usr/bin/env python
"""
Prepare MASS training data from processed NIfTI images and auto masks.

This stage runs after SAM2 mask generation. It resamples the cropped
original-resolution images, GT labels, and auto masks to the training spacing,
then saves high-frequency image/GT arrays as .npy files and stores the large
auto-mask stack in a compressed HDF5 dataset compatible with the training
dataset loaders.
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import logging
import traceback
import h5py
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import time
import hashlib
import tempfile
import platform

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [PID:%(process)d] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Cross-process locking
# -----------------------------------------------------------------------------
class CrossProcessLock:
    """File-based lock that works across different processes"""
    
    def __init__(self, lockfile_path):
        self.lockfile_path = lockfile_path
        self.lockfile = None
        self.is_windows = platform.system() == 'Windows'
        
    def acquire(self, timeout=300):  # 5 minute timeout
        """Acquire the lock with timeout"""
        start_time = time.time()
        os.makedirs(os.path.dirname(self.lockfile_path), exist_ok=True)
        
        while True:
            try:
                if self.is_windows:
                    import msvcrt
                    self.lockfile = open(self.lockfile_path, 'w')
                    msvcrt.locking(self.lockfile.fileno(), msvcrt.LK_NBLCK, 1)
                    return True
                else:
                    self.lockfile = open(self.lockfile_path, 'w')
                    fcntl.flock(self.lockfile.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return True
            except (IOError, OSError):
                if self.lockfile:
                    self.lockfile.close()
                    self.lockfile = None
                    
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Could not acquire lock after {timeout} seconds")
                time.sleep(0.1)
    
    def release(self):
        """Release the lock"""
        if self.lockfile:
            try:
                if self.is_windows:
                    import msvcrt
                    msvcrt.locking(self.lockfile.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(self.lockfile.fileno(), fcntl.LOCK_UN)
            except:
                pass
            finally:
                try:
                    self.lockfile.close()
                except:
                    pass
                self.lockfile = None
    
    def __enter__(self):
        self.acquire()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class TaskManager:
    """Manages task distribution across multiple program instances"""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.task_dir = os.path.join(output_dir, ".tasks")
        os.makedirs(self.task_dir, exist_ok=True)
        
        self.master_lock_path = os.path.join(self.task_dir, "master.lock")
        self.h5_lock_path = os.path.join(self.task_dir, "h5.lock")
        
    def get_task_file(self, image_name):
        """Get the path to the task file for an image"""
        # Use hash to avoid filesystem issues with special characters
        name_hash = hashlib.md5(image_name.encode()).hexdigest()[:8]
        return os.path.join(self.task_dir, f"{image_name}_{name_hash}.task")
    
    def get_done_file(self, image_name):
        """Get the path to the done file for an image"""
        name_hash = hashlib.md5(image_name.encode()).hexdigest()[:8]
        return os.path.join(self.task_dir, f"{image_name}_{name_hash}.done")
    
    def claim_task(self, image_name):
        """Try to claim a task. Returns True if claimed, False if already claimed."""
        task_file = self.get_task_file(image_name)
        done_file = self.get_done_file(image_name)
        
        with CrossProcessLock(self.master_lock_path):
            if os.path.exists(done_file):
                return False
            
            if os.path.exists(task_file):
                return False
            
            try:
                with open(task_file, 'w') as f:
                    f.write(f"{os.getpid()}\n{time.time()}")
                return True
            except Exception as e:
                logger.error(f"Failed to claim task {image_name}: {e}")
                return False
    
    def complete_task(self, image_name):
        """Mark a task as complete"""
        task_file = self.get_task_file(image_name)
        done_file = self.get_done_file(image_name)
        
        with CrossProcessLock(self.master_lock_path):
            try:
                if os.path.exists(task_file):
                    os.rename(task_file, done_file)
                elif not os.path.exists(done_file):
                    with open(done_file, 'w') as f:
                        f.write(f"{os.getpid()}\n{time.time()}\ncompleted")
            except FileNotFoundError:
                # Another process might have completed it already
                if not os.path.exists(done_file):
                    try:
                        with open(done_file, 'w') as f:
                            f.write(f"{os.getpid()}\n{time.time()}\ncompleted")
                    except:
                        pass
            except Exception as e:
                logger.error(f"Error completing task {image_name}: {e}")
                if not os.path.exists(done_file):
                    try:
                        with open(done_file, 'w') as f:
                            f.write(f"{os.getpid()}\n{time.time()}\ncompleted_with_error")
                    except:
                        pass
    
    def is_task_done(self, image_name):
        """Check if a task is already done"""
        done_file = self.get_done_file(image_name)
        return os.path.exists(done_file)
    
    def is_task_claimed(self, image_name):
        """Check if a task is claimed (but not necessarily done)"""
        task_file = self.get_task_file(image_name)
        return os.path.exists(task_file)

    def clear_done(self, image_name):
        """Remove a stale done marker so an incomplete output can be regenerated."""
        done_file = self.get_done_file(image_name)
        with CrossProcessLock(self.master_lock_path):
            if os.path.exists(done_file):
                os.remove(done_file)
    
    def cleanup_stale_tasks(self, timeout=3600):  # 1 hour timeout
        """Clean up tasks that have been claimed but not completed"""
        current_time = time.time()
        cleaned_count = 0
        
        with CrossProcessLock(self.master_lock_path):
            for task_file in glob.glob(os.path.join(self.task_dir, "*.task")):
                try:
                    with open(task_file, 'r') as f:
                        lines = f.readlines()
                        if len(lines) >= 2:
                            claim_time = float(lines[1].strip())
                            if current_time - claim_time > timeout:
                                logger.warning(f"Removing stale task: {os.path.basename(task_file)}")
                                os.remove(task_file)
                                cleaned_count += 1
                except Exception as e:
                    logger.error(f"Error checking task file {task_file}: {e}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} stale task(s)")
        
        return cleaned_count


# -----------------------------------------------------------------------------
# Global concurrency primitives
# -----------------------------------------------------------------------------
H5_LOCK = None
GPU_SEMA = None
TASK_MANAGER = None
H5_CROSS_LOCK = None


def _init_worker(h5_lock, gpu_sema, gpu_id, use_cuda, output_dir):
    """
    Initializer for each worker process.
    """
    global H5_LOCK, GPU_SEMA, TASK_MANAGER, H5_CROSS_LOCK
    
    H5_LOCK = h5_lock
    GPU_SEMA = gpu_sema
    TASK_MANAGER = TaskManager(output_dir)
    H5_CROSS_LOCK = CrossProcessLock(os.path.join(output_dir, ".tasks", "h5.lock"))
    
    if use_cuda and torch.cuda.is_available():
        try:
            torch.cuda.set_device(gpu_id)
        except Exception as e:
            logger.warning(f"Worker failed to set CUDA device {gpu_id}: {e}")


class GPUResampler:
    """GPU-accelerated resampling using PyTorch"""
    
    def __init__(self, device='cuda'):
        self.device = device
        if torch.cuda.is_available() and str(device).startswith('cuda'):
            try:
                torch.cuda.set_per_process_memory_fraction(0.8)
            except Exception:
                pass
    
    def _to_device(self, arr):
        """Convert numpy array to pinned tensor and move to device."""
        t = torch.as_tensor(arr, dtype=torch.float32, device='cpu')
        if t.ndim == 3:
            t = t.unsqueeze(0)
        t = t.unsqueeze(1).pin_memory()
        return t.to(self.device, non_blocking=True)
    
    @torch.no_grad()
    def resample_volume(self, volume, current_spacing, target_spacing, is_label=False, batch=False):
        """Resample a 3D volume or batch of volumes to target spacing using GPU acceleration."""
        try:
            if np.allclose(current_spacing, target_spacing, rtol=1e-5):
                logger.info(f"  Spacing already at target {target_spacing}, skipping resampling")
                
                if batch:
                    vols = volume if isinstance(volume, list) else [volume]
                    if is_label:
                        return np.array([v.astype(np.uint8) for v in vols])
                    else:
                        return np.array([v.astype(np.float32) for v in vols])
                else:
                    if is_label:
                        return volume.astype(np.uint8)
                    else:
                        return volume.astype(np.float32)
            else:
                logger.info(f"  Resampling {current_spacing} to {target_spacing}")

            if not batch:
                vols = [volume]
            else:
                vols = volume if isinstance(volume, list) else [volume]

            current_size = vols[0].shape[-3:] if batch else volume.shape
            scale_factors = [
                current_spacing[i] / target_spacing[i] for i in range(3)
            ]
            new_size = [
                int(round(current_size[i] * scale_factors[i])) for i in range(3)
            ]
            
            if len(vols) > 1 or batch:
                cpu_stack = np.stack(vols, axis=0).astype(np.float32)
            else:
                cpu_stack = vols[0].astype(np.float32)
                if cpu_stack.ndim == 3:
                    cpu_stack = cpu_stack[np.newaxis, ...]
            
            x = self._to_device(cpu_stack)
            
            mode = 'nearest' if is_label else 'trilinear'
            align = None if is_label else True
            
            y = F.interpolate(x, size=new_size, mode=mode, align_corners=align)
            
            out = y.squeeze(1).cpu().numpy()
            
            del x, y
            if torch.cuda.is_available() and str(self.device).startswith('cuda'):
                torch.cuda.empty_cache()
            
            if is_label:
                out = np.round(out).astype(np.uint8)
            
            return out[0] if not batch else out
            
        except Exception as e:
            logger.error(f"GPU resampling failed: {e}")
            raise


def guess_modality(image):
    """Best-effort modality guess used only when --modality auto is selected."""
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return "ct"
    low, high = np.percentile(finite, [0.5, 99.5])
    if low < -200 and high > 300:
        return "ct"
    if low >= 0 and high > 5:
        return "pet"
    return "mr"


def normalize_for_training(
    image,
    modality,
    ct_clip=(-991.0, 500.0),
    non_ct_percentiles=(2.0, 98.0),
    eps=1e-6,
):
    """
    Match MASS training intensity preprocessing.

    CT volumes are clipped to a fixed HU window. MR/PET volumes are clipped to
    per-volume percentiles. All modalities are then z-score normalized.
    """
    image = image.astype(np.float32, copy=False)
    modality = modality.lower()
    if modality == "auto":
        modality = guess_modality(image)
        logger.info(f"Auto-detected modality for intensity preprocessing: {modality}")
    if modality == "mri":
        modality = "mr"

    finite = image[np.isfinite(image)]
    if finite.size == 0:
        logger.warning("Image has no finite voxels; returning zeros after normalization")
        return np.zeros_like(image, dtype=np.float32)

    if modality == "ct":
        image = np.clip(image, float(ct_clip[0]), float(ct_clip[1]))
        logger.info(f"Applied CT clip: [{ct_clip[0]}, {ct_clip[1]}]")
    elif modality in ("mr", "pet"):
        p_low, p_high = np.percentile(finite, non_ct_percentiles)
        if p_high > p_low:
            image = np.clip(image, p_low, p_high)
            logger.info(
                f"Applied {modality.upper()} percentile clip: "
                f"{non_ct_percentiles[0]}-{non_ct_percentiles[1]}% "
                f"-> [{p_low:.3f}, {p_high:.3f}]"
            )
        else:
            logger.warning(
                f"Skipped percentile clipping because upper <= lower: "
                f"{p_low:.3f}, {p_high:.3f}"
            )
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    mean = float(np.mean(image))
    std = float(np.std(image))
    if std > eps:
        image = (image - mean) / std
    else:
        image = image - mean

    return image.astype(np.float32, copy=False)


def save_npy_atomic(path, array):
    """Atomically save a .npy file so interrupted runs can safely retry."""
    tmp_path = f"{path}.tmp.{os.getpid()}.npy"
    np.save(tmp_path, array)
    os.replace(tmp_path, path)


def calculate_pad_width(shape, min_size=128):
    """Calculate padding width for each dimension."""
    pad_width = []
    for dim in shape:
        if dim < min_size:
            total_pad = min_size - dim
            pad_before = total_pad // 2
            pad_after = total_pad - pad_before
            pad_width.append((pad_before, pad_after))
        else:
            pad_width.append((0, 0))
    return pad_width


def pad_to_min_size(volume, min_size=128, is_stack=False):
    """Pad a volume to ensure minimum size in each spatial dimension."""
    if is_stack:
        current_shape = volume.shape[1:]
        pad_width = calculate_pad_width(current_shape, min_size)
        pad_width = [(0, 0)] + pad_width
    else:
        current_shape = volume.shape
        pad_width = calculate_pad_width(current_shape, min_size)
    
    padded_volume = np.pad(volume, pad_width, mode='constant', constant_values=0)
    
    return padded_volume


def find_auto_label_files(auto_label_dir, image_name):
    """Find all auto label files for a given image."""
    pattern = f"{image_name}_*_nms_vol*.nii.gz"
    vol_files = glob.glob(os.path.join(auto_label_dir, pattern))
    
    mapping_pattern = f"{image_name}_*_nms_mapping.json"
    mapping_files = glob.glob(os.path.join(auto_label_dir, mapping_pattern))
    
    if not vol_files or not mapping_files:
        return [], None
    
    patterns = {}
    for vol_file in vol_files:
        basename = os.path.basename(vol_file)
        pattern_name = '_'.join(basename.split('_')[:-1])
        if pattern_name not in patterns:
            patterns[pattern_name] = []
        patterns[pattern_name].append(vol_file)
    
    for pattern_name in patterns:
        patterns[pattern_name] = sorted(patterns[pattern_name])
    
    return patterns, mapping_files[0]


def load_auto_label_volumes(auto_label_dir, image_name):
    """Load auto-generated label volumes."""
    patterns, mapping_file = find_auto_label_files(auto_label_dir, image_name)
    
    if not patterns:
        logger.warning(f"No auto label files found for {image_name}")
        return {}, {}
    
    logger.info(f"Found {len(patterns)} auto label patterns for {image_name}")
    
    mapping_info = {}
    if mapping_file and os.path.exists(mapping_file):
        with open(mapping_file, 'r') as f:
            mapping_info = json.load(f)
    
    all_volumes = {}
    for pattern_name, vol_files in patterns.items():
        logger.info(f"  Loading pattern {pattern_name} with {len(vol_files)} volumes")
        
        for vol_file in vol_files:
            basename = os.path.basename(vol_file)
            vol_idx_str = basename.split('_vol')[-1].replace('.nii.gz', '')
            vol_idx = int(vol_idx_str)
            
            vol_sitk = sitk.ReadImage(vol_file)
            vol_array = sitk.GetArrayFromImage(vol_sitk).astype(np.int32)
            all_volumes[vol_idx] = vol_array
    
    return all_volumes, mapping_info


def write_masks_streaming(h5_group, resampled_label_vols, mapping_info, out_shape, pad_width, compressor='lzf'):
    """Stream individual masks to HDF5 without stacking in memory."""
    mask_list = []
    if 'original_objects' in mapping_info:
        # SAM objects can be split across volume files; mapping_info restores
        # the object id -> volume/value lookup.
        for obj_id, obj_info in mapping_info['original_objects'].items():
            vol_idx = obj_info.get('volume_idx', -1)
            label_value = obj_info.get('label_value', -1)
            if vol_idx in resampled_label_vols and label_value > 0:
                mask_list.append((vol_idx, label_value))
    else:
        for vol_idx, vol in resampled_label_vols.items():
            unique_labels = np.unique(vol)
            unique_labels = unique_labels[unique_labels > 0]
            for label_val in unique_labels:
                mask_list.append((vol_idx, int(label_val)))
    
    n_masks = len(mask_list)
    
    if n_masks == 0:
        h5_group.create_dataset('auto_masks', data=np.zeros((0, *out_shape), dtype=np.uint8))
        logger.info(f"  No masks to save")
        return
    
    logger.info(f"  Streaming {n_masks} masks to HDF5")
    
    z, y, x = out_shape
    bytes_per_voxel = 1
    target_chunk_bytes = 16 * 1024 * 1024
    voxels_per_mask = z * y * x
    
    # Keep chunks near a fixed byte budget so large volumes stay readable.
    if voxels_per_mask > target_chunk_bytes:
        vox_per_chunk = target_chunk_bytes // bytes_per_voxel
        cz = min(z, max(1, int(vox_per_chunk // (y * x))))
        cy = min(y, max(1, int(vox_per_chunk // (cz * x))))
        cx = min(x, max(1, int(vox_per_chunk // (cz * cy))))
        chunks = (1, cz, cy, cx)
    else:
        masks_per_chunk = max(1, target_chunk_bytes // voxels_per_mask)
        chunks = (min(masks_per_chunk, n_masks), z, y, x)
    
    dset = h5_group.create_dataset(
        'auto_masks',
        shape=(n_masks, z, y, x),
        dtype=np.uint8,
        compression=compressor,
        chunks=chunks,
        shuffle=True
    )
    
    for idx, (vol_idx, label_val) in enumerate(mask_list):
        mask = (resampled_label_vols[vol_idx] == label_val).astype(np.uint8)
        
        if pad_width is not None:
            mask = np.pad(mask, pad_width, mode='constant', constant_values=0)
        
        dset[idx] = mask
        
        if (idx + 1) % 100 == 0:
            logger.info(f"    Written {idx + 1}/{n_masks} masks")
    
    logger.info(f"  Successfully streamed {n_masks} masks to HDF5")


def process_single_image(
    image_path,
    label_path,
    auto_label_dir,
    output_dir,
    target_spacing,
    device,
    min_size=128,
    modality="auto",
    ct_clip=(-991.0, 500.0),
    non_ct_percentiles=(2.0, 98.0),
):
    """Process a single image with cross-process safety."""
    image_name = os.path.basename(image_path).replace('.nii.gz', '').replace('.nii', '')
    
    try:
        has_gt = label_path is not None and os.path.exists(label_path)
        has_auto = auto_label_dir is not None
        
        image_npy_path = os.path.join(output_dir, f"{image_name}_image.npy")
        gt_npy_path = os.path.join(output_dir, f"{image_name}_gt.npy")
        h5_file_path = os.path.join(output_dir, "dataset.h5")
        
        image_ready = os.path.exists(image_npy_path)
        gt_ready = (not has_gt) or os.path.exists(gt_npy_path)
        auto_ready = not has_auto
        if has_auto:
            with H5_CROSS_LOCK:
                try:
                    with h5py.File(h5_file_path, 'r') as h5f:
                        auto_ready = image_name in h5f and 'auto_masks' in h5f[image_name]
                except (OSError, KeyError, FileNotFoundError):
                    auto_ready = False

        if image_ready and gt_ready and auto_ready:
            logger.info(f"{image_name} already processed, marking as done...")
            TASK_MANAGER.complete_task(image_name)
            return True

        if TASK_MANAGER.is_task_done(image_name):
            logger.warning(
                f"{image_name} has a done marker but expected outputs are missing; "
                "clearing marker and regenerating"
            )
            TASK_MANAGER.clear_done(image_name)

        if not TASK_MANAGER.claim_task(image_name):
            logger.info(f"{image_name} already being processed, skipping...")
            return True

        logger.info(f"[Instance] Claimed task for {image_name}")
        
        mode_desc = []
        if has_gt:
            mode_desc.append("GT")
        if has_auto:
            mode_desc.append("Auto")
        logger.info(f"Processing {image_name} in mode: Image + {' + '.join(mode_desc) if mode_desc else 'No Labels'}")
        
        resampler = GPUResampler(device=device)
        
        logger.info(f"Loading image for {image_name}...")
        image_sitk = sitk.ReadImage(image_path)
        image_array = sitk.GetArrayFromImage(image_sitk)
        
        # SimpleITK reports spacing as (x, y, z), while numpy arrays are [z, y, x].
        spacing_xyz = image_sitk.GetSpacing()
        current_spacing = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
        
        label_array = None
        resampled_gt = None
        if has_gt:
            logger.info(f"Loading GT label for {image_name}...")
            label_sitk = sitk.ReadImage(label_path)
            label_array = sitk.GetArrayFromImage(label_sitk)
        
        label_volumes = {}
        mapping_info = {}
        if has_auto:
            logger.info(f"Loading auto-generated label volumes for {image_name}...")
            label_volumes, mapping_info = load_auto_label_volumes(auto_label_dir, image_name)
            
            if len(label_volumes) == 0:
                logger.warning(f"No auto-generated label volumes found for {image_name}")
                if not has_gt:
                    logger.warning(f"No GT label and no auto masks for {image_name}, skipping...")
                    TASK_MANAGER.complete_task(image_name)
                    return False
        
        # Only one worker uses the GPU at a time, while CPU workers can still
        # load files and write finished outputs.
        use_cuda_here = torch.cuda.is_available() and str(device).startswith('cuda')
        if use_cuda_here and GPU_SEMA is not None:
            GPU_SEMA.acquire()
        
        try:
            logger.info(f"Resampling image...")
            resampled_image = resampler.resample_volume(
                image_array, current_spacing, target_spacing, is_label=False
            )
            
            if has_gt and label_array is not None:
                logger.info(f"Resampling GT label...")
                resampled_gt = resampler.resample_volume(
                    label_array, current_spacing, target_spacing, is_label=True
                )
            
            resampled_label_volumes = {}
            if has_auto and len(label_volumes) > 0:
                logger.info(f"Resampling {len(label_volumes)} label volumes...")
                
                batch_size = 10
                vol_indices = list(label_volumes.keys())
                
                for i in range(0, len(vol_indices), batch_size):
                    batch_indices = vol_indices[i:i+batch_size]
                    batch_volumes = [label_volumes[idx] for idx in batch_indices]
                    
                    logger.info(f"  Resampling batch {i//batch_size + 1}/{(len(vol_indices) + batch_size - 1)//batch_size}")
                    
                    if len(batch_volumes) > 1:
                        resampled_batch = resampler.resample_volume(
                            batch_volumes, current_spacing, target_spacing, 
                            is_label=True, batch=True
                        )
                        for j, idx in enumerate(batch_indices):
                            resampled_label_volumes[idx] = resampled_batch[j]
                    else:
                        resampled_label_volumes[batch_indices[0]] = resampler.resample_volume(
                            batch_volumes[0], current_spacing, target_spacing, is_label=True
                        )
                    
                    if torch.cuda.is_available() and str(device).startswith('cuda'):
                        torch.cuda.empty_cache()
                        
        finally:
            if use_cuda_here and GPU_SEMA is not None:
                GPU_SEMA.release()
        
        # The saved image array is already intensity-normalized for training.
        logger.info(f"Applying training intensity preprocessing...")
        normalized_image = normalize_for_training(
            resampled_image,
            modality=modality,
            ct_clip=ct_clip,
            non_ct_percentiles=non_ct_percentiles,
        )
        
        pad_width = calculate_pad_width(normalized_image.shape, min_size=min_size)
        
        logger.info(f"Padding image to minimum size {min_size}...")
        normalized_image = pad_to_min_size(normalized_image, min_size=min_size, is_stack=False)
        
        if resampled_gt is not None:
            resampled_gt = pad_to_min_size(resampled_gt, min_size=min_size, is_stack=False)
        
        logger.info(f"Saving image npy: {image_npy_path}")
        save_npy_atomic(image_npy_path, normalized_image.astype(np.float32, copy=False))
        if resampled_gt is not None:
            logger.info(f"Saving GT npy: {gt_npy_path}")
            save_npy_atomic(gt_npy_path, resampled_gt.astype(np.uint8, copy=False))
        
        if has_auto:
            temp_h5 = tempfile.NamedTemporaryFile(suffix='.h5', delete=False, dir=output_dir)
            temp_h5_path = temp_h5.name
            temp_h5.close()

            logger.info(f"Writing auto masks to temporary HDF5...")

            try:
                # Write each case to a temporary file first; the main dataset.h5
                # is touched only during the short locked merge below.
                with h5py.File(temp_h5_path, 'w') as temp_h5f:
                    group = temp_h5f.create_group(image_name)

                    if len(resampled_label_volumes) > 0:
                        write_masks_streaming(
                            group, resampled_label_volumes, mapping_info,
                            normalized_image.shape, pad_width, compressor='lzf'
                        )
                    else:
                        group.create_dataset(
                            'auto_masks',
                            data=np.zeros((0, *normalized_image.shape), dtype=np.uint8)
                        )
                        logger.info(f"  Saved empty auto masks dataset")

                logger.info(f"Merging auto masks into main HDF5...")
                with H5_CROSS_LOCK:
                    with h5py.File(h5_file_path, 'a') as main_h5f:
                        with h5py.File(temp_h5_path, 'r') as temp_h5f:
                            if image_name in main_h5f:
                                del main_h5f[image_name]
                            temp_h5f.copy(image_name, main_h5f)

                    logger.info(f"Successfully saved {image_name}/auto_masks to main HDF5")

            finally:
                try:
                    if os.path.exists(temp_h5_path):
                        os.remove(temp_h5_path)
                except:
                    pass
        
        TASK_MANAGER.complete_task(image_name)
        logger.info(f"Successfully processed {image_name}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing {image_name}: {e}")
        logger.error(traceback.format_exc())
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Leave the task incomplete so a later run can retry it.
        return False


def main():
    parser = argparse.ArgumentParser(description="Multi-process safe data preparation")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing images and optional *_gt.nii.gz labels")
    parser.add_argument("--auto_label_dir", type=str, default=None, 
                       help="Directory containing auto-generated labels")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for *_image.npy, optional *_gt.npy, and dataset.h5")
    parser.add_argument("--target_spacing", type=float, nargs=3, default=[1.5, 1.5, 1.5], 
                       help="Target spacing for resampling (z, y, x)")
    parser.add_argument("--modality", choices=["auto", "ct", "mr", "mri", "pet"], default="auto",
                       help="Intensity preprocessing mode for saved *_image.npy")
    parser.add_argument("--ct_clip", type=float, nargs=2, default=[-991.0, 500.0],
                       help="CT clipping range before z-score normalization")
    parser.add_argument("--non_ct_percentiles", type=float, nargs=2, default=[2.0, 98.0],
                       help="MR/PET percentile clipping range before z-score normalization")
    parser.add_argument("--min_size", type=int, default=135,
                       help="Minimum size for each spatial dimension after padding")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of CPU workers")
    parser.add_argument("--images", type=str, nargs='+', default=None,
                       help="Specific image names to process (without extension)")
    parser.add_argument("--skip_auto_labels", action='store_true', 
                       help="Skip processing auto labels")
    parser.add_argument("--skip_gt_labels", action='store_true',
                       help="Skip processing GT labels")
    parser.add_argument("--instance_id", type=int, default=0, 
                       help="Instance ID when running multiple programs")
    parser.add_argument("--stale_timeout", type=int, default=3600,
                       help="Timeout in seconds for stale task cleanup (default: 3600)")
    
    args = parser.parse_args()
    
    logger.info(f"Starting instance {args.instance_id} with PID {os.getpid()}")
    
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    
    if args.skip_auto_labels:
        args.auto_label_dir = None
        logger.info("Skipping auto labels as requested")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    task_manager = TaskManager(args.output_dir)
    cleaned = task_manager.cleanup_stale_tasks(timeout=args.stale_timeout)
    
    if torch.cuda.is_available():
        device = f'cuda:{args.gpu_id}'
        torch.cuda.set_device(args.gpu_id)
        logger.info(f"Using GPU {args.gpu_id}")
    else:
        device = 'cpu'
        logger.warning("CUDA not available, using CPU")
    
    target_spacing = tuple(args.target_spacing)
    logger.info(f"Target spacing: {target_spacing}")
    logger.info(f"Image modality preprocessing: {args.modality}")
    
    # Standardized data uses image.nii.gz and optional image_gt.nii.gz naming.
    all_nii_files = sorted(glob.glob(os.path.join(args.data_dir, "*.nii.gz")))
    if not all_nii_files:
        all_nii_files = sorted(glob.glob(os.path.join(args.data_dir, "*.nii")))
    
    image_paths = []
    for file_path in all_nii_files:
        if not file_path.endswith('_gt.nii.gz') and not file_path.endswith('_gt.nii'):
            image_paths.append(file_path)
    
    logger.info(f"Found {len(image_paths)} images")
    
    if args.images:
        filtered_paths = []
        for img_path in image_paths:
            img_name = os.path.basename(img_path).replace('.nii.gz', '').replace('.nii', '')
            if img_name in args.images:
                filtered_paths.append(img_path)
        image_paths = filtered_paths
        logger.info(f"Processing only specified images: {args.images}")
        logger.info(f"Found {len(image_paths)} matching images")
    
    tasks = []
    skipped_count = 0
    for image_path in image_paths:
        image_name = os.path.basename(image_path).replace('.nii.gz', '').replace('.nii', '')

        # GT is optional for SSL pretraining, but auto masks or GT must exist.
        label_path = None
        if not args.skip_gt_labels:
            candidate = os.path.join(args.data_dir, f"{image_name}_gt.nii.gz")
            if not os.path.exists(candidate):
                candidate = os.path.join(args.data_dir, f"{image_name}_gt.nii")
            if os.path.exists(candidate):
                label_path = candidate
            else:
                logger.warning(f"GT label not found for {image_name}")
        
        if label_path is None and args.auto_label_dir is None:
            logger.error(f"No GT label and no auto label directory for {image_name}, skipping...")
            continue
        
        tasks.append((image_path, label_path))
    
    logger.info(f"Instance {args.instance_id}: Processing {len(tasks)} remaining tasks")
    logger.info(f"Already completed: {skipped_count}")
    
    successful = 0
    failed = 0
    
    if len(tasks) == 0:
        logger.info(f"\nInstance {args.instance_id} complete:")
        logger.info(f"No tasks to process")
        return
    
    use_cuda = torch.cuda.is_available()
    h5_lock = mp.Lock()
    gpu_sema = mp.Semaphore(1) if use_cuda else None
    
    max_workers = min(len(tasks), args.num_workers)
    logger.info(f"Running with {max_workers} worker process(es)")
    
    futures = []
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(h5_lock, gpu_sema, args.gpu_id, use_cuda, args.output_dir),
    ) as executor:
        for (image_path, label_path) in tasks:
            # Each case writes independent npy files; only dataset.h5 merge is locked.
            futures.append(
                executor.submit(
                    process_single_image,
                    image_path, label_path, args.auto_label_dir,
                    args.output_dir, target_spacing, device, args.min_size,
                    args.modality,
                    tuple(args.ct_clip),
                    tuple(args.non_ct_percentiles),
                )
            )
        
        pbar_desc = f"Instance {args.instance_id} processing"
        for fut in tqdm(as_completed(futures), total=len(futures), desc=pbar_desc):
            try:
                success = fut.result()
            except Exception as e:
                logger.error(f"Worker raised exception: {e}")
                logger.error(traceback.format_exc())
                success = False
            
            if success:
                successful += 1
            else:
                failed += 1
    
    logger.info(f"\nInstance {args.instance_id} complete:")
    logger.info(f"Successfully processed: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Skipped (already done): {skipped_count}")
    logger.info(f"Total processed in this run: {successful + failed}")


if __name__ == "__main__":
    main()
