"""Evaluation metrics and plotting utilities for FLAME training."""

from __future__ import annotations

from typing import Dict, Iterable

from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def compute_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    mean: bool = True,
    straight_up_binary_mask: bool = False,
) -> torch.Tensor:

    # Convert logits to binary mask
    pred = (torch.sigmoid(pred) > threshold).float() if not straight_up_binary_mask else pred
    target = (target > 0.5).float()

    # Compute intersection and union
    intersection = (pred * target).sum(dim=[1,2,3])
    union = (pred + target).sum(dim=[1,2,3]) - intersection

    # Handle empty masks case
    batch_size = pred.shape[0]
    ious = torch.zeros(batch_size, device=pred.device)

    # Only compute IoU for non-empty unions
    valid_mask = union > 0
    if valid_mask.any():
        ious[valid_mask] = intersection[valid_mask] / union[valid_mask]

    # If both pred and target are empty (union=0), IoU=1
    empty_mask = (union == 0)
    if empty_mask.any():
        ious[empty_mask] = 1.0

    return ious.mean() if mean else ious

    
def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    loss_on_multimask: bool = False,
    padding_mask: torch.Tensor = None,
) -> torch.Tensor:
    # Clamp logits to a safe range for numerical stability
    inputs = torch.clamp(inputs, -10, 10)
    # Compute probabilities
    prob = inputs.sigmoid()
    # Small epsilon to prevent 0^gamma or 1^gamma
    eps = 1e-6
    p_t = torch.clamp(prob * targets + (1 - prob) * (1 - targets), min=eps, max=1 - eps)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    
    # Apply padding mask if provided
    if padding_mask is not None:
        loss = loss * padding_mask
        # Calculate mean only over non-padding regions
        num_valid = padding_mask.sum()
        if num_valid > 0:
            return loss.sum() / num_valid
        else:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)
    
    return torch.nanmean(loss)


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    padding_mask: torch.Tensor = None,
    eps: float = 1e-6,
    pred_thresh: float = 0.5,
) -> torch.Tensor:
    """
    Soft-Dice that:
      - skips loss (returns 0) if BOTH target and (thresholded) prediction are empty,
      - penalizes spurious predictions on empty GT,
      - otherwise computes 1 - Dice on probabilities.
      - ignores padding regions if padding_mask is provided
    """
    probs = torch.sigmoid(inputs)
    B = probs.shape[0]
    
    # Apply padding mask if provided
    if padding_mask is not None:
        # Flatten for easier calculation
        probs_flat = (probs * padding_mask).view(B, -1)
        targs_flat = (targets * padding_mask).view(B, -1)
        # Calculate sum of padding mask for each image (to know valid area)
        mask_sum = padding_mask.view(B, -1).sum(dim=1)
    else:
        probs_flat = probs.view(B, -1)
        targs_flat = targets.view(B, -1)
        mask_sum = None

    # Soft Dice components
    intersection = (probs_flat * targs_flat).sum(dim=1)
    union = probs_flat.sum(dim=1) + targs_flat.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    loss_per_image = 1 - dice

    # Emptiness check using a binary threshold
    has_gt = targs_flat.sum(dim=1) > 0
    has_pred = (probs > pred_thresh).view(B, -1).any(dim=1)

    # Include in loss unless BOTH are empty
    use_dice = has_gt | has_pred

    if use_dice.any():
        return torch.nanmean(loss_per_image[use_dice])
    return torch.tensor(0.0, device=inputs.device, requires_grad=True)

def plot_all_metrics(metrics_history: Dict[str, Iterable], train_sam_iou: bool, save_path: str) -> None:
    """Plot training and validation curves with dataset-specific breakdowns."""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # Create a more complex figure layout
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.25)
    
    fig.suptitle("Training Progress with Dataset-Specific Metrics", fontsize=16)

    # Define consistent colors and markers for datasets
    colors = plt.cm.tab10(range(10))
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'X', '+']
    size_styles = {'small': ':', 'medium': '--', 'large': '-'}  # Line styles for sizes
    
    # Plot 1: Training Loss
    ax1 = fig.add_subplot(gs[0, 0])
    losses = metrics_history['train_losses']
    if len(losses) >= 25:
        losses = np.convolve(losses, np.ones(25)/25, mode='valid')
    else:
        losses = np.array(losses)
    ax1.plot(losses, alpha=0.7)
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Overall Validation Metrics
    ax2 = fig.add_subplot(gs[0, 1])
    if metrics_history['val_ious']:
        ax2.plot(metrics_history['val_ious'], label="IoU", marker='o', markersize=4)
        ax2.plot(metrics_history['val_f1s'], label="F1", marker='s', markersize=4)
    ax2.set_title("Overall Validation")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Overall Size-stratified IoU
    ax3 = fig.add_subplot(gs[0, 2])
    if metrics_history.get('small_iou') and len(metrics_history['small_iou']) > 0:
        epochs = range(1, len(metrics_history['small_iou']) + 1)
        size_colors = ['red', 'orange', 'green']
        for i, size in enumerate(['small', 'medium', 'large']):
            if metrics_history[f'{size}_iou']:
                ax3.plot(epochs, metrics_history[f'{size}_iou'],
                        label=f'{size.title()} (<5%/5-20%/>20%)', 
                        color=size_colors[i], marker='o', markersize=4)
        ax3.set_title("Overall IoU by Forgery Size")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No size-stratified data yet', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title("Overall IoU by Forgery Size")

    # Plot 4: IoU by Dataset
    ax4 = fig.add_subplot(gs[1, 0])
    if 'dataset_metrics' in metrics_history:
        for i, (dataset, metrics) in enumerate(metrics_history['dataset_metrics'].items()):
            if metrics.get('iou_history'):
                epochs = range(1, len(metrics['iou_history']) + 1)
                marker = markers[i % len(markers)]
                ax4.plot(epochs, metrics['iou_history'], 
                        label=dataset, color=colors[i], marker=marker, markersize=4)
        ax4.set_title("IoU by Dataset")
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)

    # Plot 5: F1 by Dataset  
    ax5 = fig.add_subplot(gs[1, 1])
    if 'dataset_metrics' in metrics_history:
        for i, (dataset, metrics) in enumerate(metrics_history['dataset_metrics'].items()):
            if metrics.get('f1_history'):
                epochs = range(1, len(metrics['f1_history']) + 1)
                marker = markers[i % len(markers)]
                ax5.plot(epochs, metrics['f1_history'],
                        label=dataset, color=colors[i], marker=marker, markersize=4)
        ax5.set_title("F1 by Dataset")
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)

    # Plot 6: IoU Divergence or Sample Counts
    ax6 = fig.add_subplot(gs[1, 2])
    if metrics_history.get('iou_divergence') and len(metrics_history['iou_divergence']) > 0:
        iou_epochs = range(2, len(metrics_history['iou_divergence']) + 2)
        ax6.plot(iou_epochs, metrics_history['iou_divergence'],
                label="Train-Val IoU Gap", color='purple', marker='o', markersize=4)
        ax6.axhline(y=0, color='red', linestyle=':', alpha=0.5)
        ax6.set_title("IoU Divergence (Epoch 2+)")
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)
        ax6.set_xlabel("Epoch")
        
        if len(metrics_history['iou_divergence']) > 0:
            ax6.text(0.02, 0.98, 'IoU training disabled for epoch 1', 
                    transform=ax6.transAxes, fontsize=8, 
                    verticalalignment='top', alpha=0.7)
    elif 'dataset_metrics' in metrics_history:
        for i, (dataset, metrics) in enumerate(metrics_history['dataset_metrics'].items()):
            if metrics.get('count_history'):
                epochs = range(1, len(metrics['count_history']) + 1)
                marker = markers[i % len(markers)]
                ax6.plot(epochs, metrics['count_history'],
                        label=dataset, color=colors[i], marker=marker, markersize=4)
        ax6.set_title("Sample Counts by Dataset")
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)

    # NEW: Plots 7-9: IoU Size Decomposition per Dataset
    # We'll create one plot per size category showing all datasets
    for plot_idx, size in enumerate(['small', 'medium', 'large']):
        ax = fig.add_subplot(gs[2, plot_idx])
        
        if 'dataset_size_metrics' in metrics_history:
            for i, (dataset, size_metrics) in enumerate(metrics_history['dataset_size_metrics'].items()):
                key = f'{size}_iou_history'
                if key in size_metrics and size_metrics[key]:
                    epochs = range(1, len(size_metrics[key]) + 1)
                    ax.plot(epochs, size_metrics[key],
                           label=dataset, color=colors[i % len(colors)], 
                           marker=markers[i % len(markers)], markersize=3,
                           linewidth=1.5, alpha=0.8)
            
            ax.set_title(f'{size.title()} Forgery IoU by Dataset')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('IoU')
            ax.legend(fontsize=7, loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
        else:
            ax.text(0.5, 0.5, f'No {size} size data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{size.title()} Forgery IoU by Dataset')

    # NEW: Plots 10-12: F1 Size Decomposition per Dataset
    for plot_idx, size in enumerate(['small', 'medium', 'large']):
        ax = fig.add_subplot(gs[3, plot_idx])
        
        if 'dataset_size_metrics' in metrics_history:
            for i, (dataset, size_metrics) in enumerate(metrics_history['dataset_size_metrics'].items()):
                key = f'{size}_f1_history'
                if key in size_metrics and size_metrics[key]:
                    epochs = range(1, len(size_metrics[key]) + 1)
                    ax.plot(epochs, size_metrics[key],
                           label=dataset, color=colors[i % len(colors)], 
                           marker=markers[i % len(markers)], markersize=3,
                           linewidth=1.5, alpha=0.8)
            
            ax.set_title(f'{size.title()} Forgery F1 by Dataset')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('F1 Score')
            ax.legend(fontsize=7, loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
        else:
            ax.text(0.5, 0.5, f'No {size} F1 data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{size.title()} Forgery F1 by Dataset')

    plt.suptitle("Training Progress with Dataset & Size-Specific Metrics", fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()