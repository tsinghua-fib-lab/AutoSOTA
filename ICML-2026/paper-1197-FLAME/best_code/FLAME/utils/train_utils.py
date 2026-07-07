"""Utility helpers for experiment naming and dataloader collation."""

from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, List

import torch


logger = logging.getLogger(__name__)


def get_param_strings() -> List[str]:
    """
    Extract parameter strings from the calling function's arguments.

    Returns:
        List of formatted parameter strings for experiment naming
    """
    frame = inspect.currentframe()
    if frame is None or frame.f_back is None:
        logger.warning("get_param_strings called outside a frame; returning empty list")
        return []

    args = inspect.getargvalues(frame.f_back)

    # Exclude certain parameters from the experiment name
    exclude_params = {
        "model", "loader", "optimizer", "device", "save_path", "lambda_focal",
        "train_sam_iou", "img_size", "mult-edit-data-root", "use_multi_edit_only",
        "downscale", "lambda-iou", "use-ema", "use-ema", "lambda-iou", "scheduler-type", "ema-decay",
        "lambda_iou", "use_ema", "scheduler_type", "ema_decay", "authentic_source_dir", "weight_decay",
        "authentic_ratio", "pretrained_checkpoint", "checkpoint_path", "resume_checkpoint",
    }

    param_strings = []
    for param_name, param_value in args.locals.items():
        if param_name not in exclude_params:
            if param_value is None:
                continue
            if param_name == "use_feature_fusion":
                # change name to: uff
                clean_name = "uff"
            elif param_name == "use_spatial_gating":
                # change name to: usg
                clean_name = "usg"
            elif param_name == "use_coarse_block":
                # change name to: ucb
                clean_name = "ucb"
            elif param_name == "use_uncertainty":
                # change name to: uu
                clean_name = "uu"
            else:
                clean_name = param_name.replace("_", "-")
                
                
            clean_value = str(param_value).replace(".", "p").replace("/", "-")
            if not clean_value:
                continue
            param_strings.append(f"{clean_name}{clean_value}")

    return param_strings


def custom_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Custom collate function to handle variable-sized elements in multi-edit batches.

    Args:
        batch: List of sample dictionaries from the dataset

    Returns:
        Collated batch dictionary with proper tensor stacking
    """
    # Filter out invalid samples (required tensors are None)
    valid_samples = []
    for item in batch:
        mask_ok = item.get("mask", None) is not None
        if not mask_ok:
            logger.warning("Skipping invalid sample with missing mask")
            continue

        if item.get("local_patch", None) is not None:
            required = ("global_image", "norm_coords")
            if all(item.get(k, None) is not None for k in required):
                valid_samples.append(item)
            else:
                logger.warning("Skipping invalid sample with missing global/local views")
        elif item.get("orig", None) is not None:
            valid_samples.append(item)
        else:
            logger.warning("Skipping invalid sample with missing image tensor")
    
    if not valid_samples:
        logger.warning("No valid samples in batch, returning empty batch")
        return {}
    
    batch = valid_samples
    collated_batch = {}

    # Handle streams - each sample can have a different number of streams
    if 'streams' in batch[0]:
        # Get the maximum number of streams across all samples
        max_streams = max(len(item['streams']) for item in batch)

        # Stack streams by index
        collated_streams = []
        for stream_idx in range(max_streams):
            stream_batch = []
            for item in batch:
                if stream_idx < len(item['streams']):
                    stream_batch.append(item['streams'][stream_idx])
                else:
                    # Pad with the last available stream if needed
                    stream_batch.append(item['streams'][-1])

            try:
                collated_streams.append(torch.stack(stream_batch))
            except RuntimeError:
                logger.exception("Error stacking stream %s", stream_idx)
                continue

        collated_batch['streams'] = collated_streams

    # Handle standard tensors (fixed size) - these should stack normally
    fixed_keys = [
        'mask',
        'orig',
        'source',
        # Global-guided local refinement (training-only; inference keeps these as None)
        'global_image',
        'local_patch',
        'local_patch_raw',
        'norm_coords',
    ]  # 'source' is optional but useful for visualization

    # Legacy support - handle old format if present
    if 'pert' in batch[0]:
        fixed_keys.append('pert')
    if batch[0].get('sharp') is not None:
        fixed_keys.append('sharp')

    for key in fixed_keys:
        if key in batch[0] and batch[0][key] is not None:
            try:
                collated_batch[key] = torch.stack([item[key] for item in batch])
            except RuntimeError:
                # Variable-sized tensors (e.g., full-resolution inference) can't be stacked.
                # Keep them as lists and let downstream code handle per-sample processing.
                collated_batch[key] = [item[key] for item in batch]

    # Handle variable-sized elements - keep as lists
    variable_keys = ['prev_masks', 'turn_idx', 'source_base', 'target_file', 'instruction', 'dataset_name', 'is_authentic', 'sample_type', 'orig_pil']
    for key in variable_keys:
        if key in batch[0]:
            collated_batch[key] = [item[key] for item in batch]

    return collated_batch
