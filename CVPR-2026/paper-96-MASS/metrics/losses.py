"""Loss functions used by MASS training.

The registered criteria here include binary Dice and BCE losses used by the
mask-guided self-supervised objective and downstream segmentation examples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Union

from utils.registry import register_criterion


@register_criterion("BinaryDiceLoss")
class BinaryDiceLoss(nn.Module):
    """
    Dice loss for binary segmentation (foreground/background).
    Supports dynamic class weights and adaptive alpha-beta balancing.
    """
    
    def __init__(self, reduction: bool = True):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self, 
        preds: torch.Tensor, 
        class_mask: torch.Tensor, 
        tgt_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for Dice loss calculation.
        
        Args:
            preds: Predicted segmentation logits, shape [B, C, D, H, W]
            class_mask: Target segmentation mask, shape [B, C, D, H, W]
            tgt_idx: Target class indices, shape [B, C]. -1 indicates classes to ignore
            
        Returns:
            Calculated Dice loss
        """
        N, C = preds.shape[0], preds.shape[1]
        
        P = torch.sigmoid(preds)
        smooth = 1e-6  # Smoothing factor to avoid numerical issues
        
        ones = torch.ones_like(preds)
        P_ = ones - P  # Probability of background
        class_mask_ = ones - class_mask  # Background mask
        
        TP = P * class_mask
        FP = P * class_mask_
        FN = P_ * class_mask
        
        # Adaptive alpha calculation (balancing false positives and negatives)
        alpha = FP.reshape(N*C, -1).sum(dim=1) / ((FP.reshape(N*C, -1).sum(dim=1) + FN.reshape(N*C, -1).sum(dim=1)) + smooth)
        alpha = torch.clamp(alpha, min=0.2, max=0.8)  # Keep alpha in reasonable range
        beta = 1 - alpha
        
        num = torch.sum(TP.reshape(N*C, -1), dim=1).float()
        den = num + alpha * torch.sum(FP.reshape(N*C, -1), dim=1).float() + beta * torch.sum(FN.reshape(N*C, -1), dim=1).float()
        
        dice = num / (den + smooth)
        
        loss = 1 - dice
        
        if tgt_idx is not None:
            tgt_idx_flat = tgt_idx.reshape(N*C)
            # - 1 for valid GT classes (tgt_idx >= 0)
            # - 1 for auto mask channels (tgt_idx == -2) 
            # - 0 for padding/invalid (tgt_idx == -1)
            weights = ((tgt_idx_flat >= 0) | (tgt_idx_flat == -2)).float()
            
            loss = loss * weights
            
            num_valid = weights.sum()
            
            if num_valid > 0:
                loss = loss.sum() / num_valid
            else:
                # No valid classes, return 0
                loss = torch.tensor(0.0, device=preds.device)
        elif not self.reduction:
            return loss
        else:
            loss = loss.mean()
        
        return loss

@register_criterion("BinaryCrossEntropyLoss")
class BinaryCrossEntropyLoss(nn.Module):
    """
    Binary cross entropy loss with dynamic class weights support.
    """
    
    def __init__(self):
        super().__init__()
        # Use BCEWithLogitsLoss for numerical stability
        self.loss = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(
        self, 
        preds: torch.Tensor, 
        class_mask: torch.Tensor, 
        tgt_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for BCE loss calculation.
        
        Args:
            preds: Predicted segmentation logits, shape [B, C, D, H, W]
            class_mask: Target segmentation mask, shape [B, C, D, H, W]
            tgt_idx: Target class indices, shape [B, C]. -1 indicates classes to ignore
            
        Returns:
            Calculated BCE loss
        """
        N, C = preds.shape[0], preds.shape[1]
        
        loss = self.loss(preds, class_mask.float())
        
        loss = torch.mean(loss.reshape(N*C, -1), dim=1)
        
        if tgt_idx is not None:
            tgt_idx_flat = tgt_idx.reshape(N*C)
            # - 1 for valid GT classes (tgt_idx >= 0)
            # - 1 for auto mask channels (tgt_idx == -2) 
            # - 0 for padding/invalid (tgt_idx == -1)
            weights = ((tgt_idx_flat >= 0) | (tgt_idx_flat == -2)).float()
            
            loss = loss * weights
            
            num_valid = weights.sum()
            
            if num_valid > 0:
                loss = loss.sum() / num_valid
            else:
                # No valid classes, return 0
                loss = torch.tensor(0.0, device=preds.device)
        else:
            loss = loss.mean()
        
        return loss

@register_criterion("MultiClassDiceLoss")
class MultiClassDiceLoss(nn.Module):
    """
    Dice loss for multi-class segmentation with adaptive alpha-beta balancing.
    Efficient vectorized implementation.
    
    Args:
        num_classes: Number of classes
        smooth: Smoothing factor to avoid division by zero
        size_average: Whether to average the loss over classes
        reduce: Whether to reduce the loss or return per-class losses
    """
    
    def __init__(
        self, 
        num_classes: int,
        smooth: float = 1e-5,
        size_average: bool = True,
        reduce: bool = True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.size_average = size_average
        self.reduce = reduce
    
    def forward(
        self, 
        preds: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with adaptive alpha-beta balancing.
        
        Args:
            preds: Predicted logits [B, C, D, H, W]
            targets: Target one-hot encoded [B, C, D, H, W] or class indices [B, D, H, W]
            
        Returns:
            Dice loss value
        """
        N = preds.size(0)
        C = preds.size(1)
        
        P = F.softmax(preds, dim=1)
        
        if targets.dim() == preds.dim() and targets.shape[1] == C:
            # Already one-hot encoded
            class_mask = targets.float()
        else:
            class_mask = torch.zeros_like(preds)
            class_mask.scatter_(1, targets.unsqueeze(1), 1.)
        
        ones = torch.ones_like(P)
        P_ = ones - P
        class_mask_ = ones - class_mask
        
        TP = P * class_mask
        FP = P * class_mask_
        FN = P_ * class_mask
        
        # Reshape for efficient computation: [C, N*D*H*W]
        TP = TP.transpose(0, 1).reshape(C, -1)
        FP = FP.transpose(0, 1).reshape(C, -1)
        FN = FN.transpose(0, 1).reshape(C, -1)
        
        # Sum over spatial dimensions
        TP_sum = TP.sum(dim=1)
        FP_sum = FP.sum(dim=1)
        FN_sum = FN.sum(dim=1)
        
        # Adaptive alpha calculation
        smooth_tensor = torch.full((C,), self.smooth, device=preds.device)
        alpha = FP_sum / (FP_sum + FN_sum + smooth_tensor)
        alpha = torch.clamp(alpha, min=0.2, max=0.8)
        beta = 1 - alpha
        
        numerator = TP_sum
        denominator = numerator + alpha * FP_sum + beta * FN_sum
        dice = numerator / (denominator + smooth_tensor)
        
        loss = 1 - dice
        
        
        if not self.reduce:
            return loss
        
        # Sum over classes
        loss = loss.sum()
        
        if self.size_average and C > 0:
            loss = loss.mean()
        
        return loss
