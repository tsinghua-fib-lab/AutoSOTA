"""Dice metric helpers for segmentation evaluation.

The functions in this file compute binary and multi-class Dice scores while
chunking large 3D volumes to avoid excessive GPU memory use.
"""

import torch
import numpy as np
from typing import Tuple


def calculate_dice_split(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    C: int, 
    block_size: int = 64*64*64
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate Dice coefficient by processing in blocks to avoid memory issues.
    
    Args:
        pred: Predicted binary segmentation tensor [C, N] with C classes and N voxels
        target: Target binary segmentation tensor [C, N] with C classes and N voxels
        C: Number of classes
        block_size: Block size for processing large tensors
        
    Returns:
        Tuple of (dice_coefficients, intersection, union) as torch tensors
    """
    N = pred.shape[1]
    assert C == pred.shape[0], f"Number of classes {C} doesn't match prediction shape {pred.shape[0]}"
    
    split_num = N // block_size
    total_intersection = torch.zeros(C, device=pred.device)
    total_sum = torch.zeros(C, device=pred.device)
    
    for i in range(split_num):
        dice, intersection, summ = calculate_dice(
            pred[:, i*block_size:(i+1)*block_size], 
            target[:, i*block_size:(i+1)*block_size], 
            C
        )
        total_intersection += intersection
        total_sum += summ
    
    if N % block_size != 0:
        dice, intersection, summ = calculate_dice(
            pred[:, split_num*block_size:], 
            target[:, split_num*block_size:], 
            C
        )
        total_intersection += intersection
        total_sum += summ
    
    eps = 1e-5
    dice = 2 * total_intersection / (total_sum + eps)
    
    return dice, total_intersection, total_sum


def calculate_dice(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    C: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate Dice coefficient for a single block.
    
    Args:
        pred: Predicted binary segmentation tensor [C, N] with C classes and N voxels
        target: Target binary segmentation tensor [C, N] with C classes and N voxels
        C: Number of classes
        
    Returns:
        Tuple of (dice_coefficients, intersection, union) as torch tensors
    """
    assert C == pred.shape[0], f"Number of classes {C} doesn't match prediction shape {pred.shape[0]}"
    
    intersection = (pred * target).sum(1)
    summ = pred.sum(1) + target.sum(1)
    
    intersection = intersection.float()
    summ = summ.float()
    
    eps = 1e-5
    dice = 2 * intersection / (summ + eps)
    
    return dice, intersection, summ


def calculate_multiclass_dice(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    num_classes: int
) -> torch.Tensor:
    """
    Calculate Dice coefficient for multi-class segmentation.
    
    Args:
        pred: Predicted class indices [B, H, W, D] or one-hot [B, C, H, W, D]
        target: Target class indices [B, H, W, D] or one-hot [B, C, H, W, D]
        num_classes: Number of classes
        
    Returns:
        Dice coefficients for each class [C]
    """
    if pred.ndim == target.ndim and pred.shape[1] != num_classes:
        pred = torch.nn.functional.one_hot(pred, num_classes).permute(0, 4, 1, 2, 3)
        target = torch.nn.functional.one_hot(target, num_classes).permute(0, 4, 1, 2, 3)
    
    dice_scores = torch.zeros(num_classes, device=pred.device)
    
    for c in range(num_classes):
        pred_c = pred[:, c].reshape(-1)
        target_c = target[:, c].reshape(-1)
        
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        
        if union > 0:
            dice_scores[c] = 2.0 * intersection / union
    
    return dice_scores
