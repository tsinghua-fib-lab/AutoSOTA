"""
Accuracy metric for node classification in graph neural networks.

This module provides accuracy computation for evaluating GNN models
on node classification tasks.
"""

import torch
import numpy as np


def accuracy(output, target):
    """
    Compute classification accuracy for node-level predictions.

    Args:
        output: Model predictions, either:
            - Log probabilities of shape [num_nodes, num_classes]
            - Raw logits of shape [num_nodes, num_classes]
            - Already predicted labels of shape [num_nodes]
            - Can be numpy array or torch tensor
        target: Ground truth labels of shape [num_nodes]
            - Can be numpy array or torch tensor

    Returns:
        acc: Accuracy as a float between 0 and 1
    """
    # Convert to numpy for consistent handling
    if isinstance(output, torch.Tensor):
        output = output.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    # Ensure arrays
    output = np.asarray(output)
    target = np.asarray(target)

    # Get predictions
    if output.ndim == 1:
        # Already predicted labels
        pred = output
    elif output.ndim == 2:
        # Logits - need to argmax
        pred = output.argmax(axis=1)
    else:
        raise ValueError(f"Output must be 1D or 2D, got shape {output.shape}")

    # Ensure target is 1D
    if target.ndim > 1:
        target = target.flatten()

    # Compute accuracy
    correct = np.sum(pred == target)
    total = len(target)
    acc = float(correct) / float(total)

    return acc


def accuracy_by_class(output, target, num_classes):
    """
    Compute per-class accuracy for detailed evaluation.

    Args:
        output: Model predictions of shape [num_nodes, num_classes]
            - Can be numpy array or torch tensor
        target: Ground truth labels of shape [num_nodes]
            - Can be numpy array or torch tensor
        num_classes: Number of classes in the dataset

    Returns:
        class_acc: Dictionary mapping class index to accuracy
    """
    # Convert to numpy for consistent handling
    if isinstance(output, torch.Tensor):
        output = output.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    # Ensure arrays
    output = np.asarray(output)
    target = np.asarray(target)

    # Get predictions
    if output.ndim == 1:
        pred = output
    elif output.ndim == 2:
        pred = output.argmax(axis=1)
    else:
        raise ValueError(f"Output must be 1D or 2D, got shape {output.shape}")

    # Ensure target is 1D
    if target.ndim > 1:
        target = target.flatten()

    # Compute per-class accuracy
    class_acc = {}
    for c in range(num_classes):
        mask = target == c
        total = np.sum(mask)
        if total > 0:
            correct = np.sum(pred[mask] == target[mask])
            class_acc[c] = float(correct) / float(total)
        else:
            class_acc[c] = 0.0

    return class_acc
