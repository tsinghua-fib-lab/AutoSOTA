"""Focal Loss for long-tail classification on TEDBench.

Implements Focal Loss (Lin et al., ICCV 2017) with optional inverse-square-root
class frequency weighting for the 965-class CATH topology classification task.

Usage:
    from tedbench.model.focal_loss import FocalLoss, compute_class_weights
    weights = compute_class_weights(dataset, num_classes=965, weight_type='sqrt')
    loss_fn = FocalLoss(gamma=2.0, alpha=weights)
"""

import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    The focal term ``(1 - p_t)^gamma`` down-weights well-classified examples
    (typically head classes) and focuses gradient on hard/misclassified examples
    (typically tail classes).

    Args:
        gamma: Focusing parameter. gamma=0 reduces to standard weighted CE.
            Higher gamma increases focus on hard examples. Default: 2.0.
        alpha: Optional per-class weight tensor of shape (num_classes,).
            If None, no class weighting is applied.
        reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            inputs: Logits of shape (N, C).
            targets: Class indices of shape (N,).

        Returns:
            Scalar loss if reduction is 'mean' or 'sum', else per-sample loss.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            at = self.alpha.gather(0, targets)
            focal_loss = at * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def compute_class_weights_from_dataset(
    dataset,
    num_classes: int = 965,
    weight_type: str = "sqrt",
) -> torch.Tensor:
    """Compute per-class weights from training dataset label distribution.

    Args:
        dataset: PyTorch Dataset that returns (coords, residue_index, seq_ids,
            protein_chain, label) tuples.
        num_classes: Total number of classes (965 for TEDBench CATH topologies).
        weight_type: Weighting scheme:
            - 'sqrt': inverse-square-root frequency (recommended)
            - 'inv': inverse frequency
            - 'uniform': all ones

    Returns:
        Tensor of shape (num_classes,) with per-class weights.
    """
    import numpy as np

    label_counts = np.zeros(num_classes, dtype=np.float64)
    for i in range(len(dataset)):
        _, _, _, _, label = dataset[i]
        label_counts[label] += 1

    if weight_type == "uniform":
        weights = np.ones(num_classes)
    elif weight_type == "inv":
        # Add 1 to avoid division by zero
        weights = 1.0 / (label_counts + 1)
    elif weight_type == "sqrt":
        weights = 1.0 / np.sqrt(label_counts + 1)
    else:
        raise ValueError(f"Unknown weight_type: {weight_type}")

    # Normalize so mean weight is 1.0
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def compute_class_weights_from_labels(
    labels: list,
    num_classes: int = 965,
    weight_type: str = "sqrt",
) -> torch.Tensor:
    """Compute per-class weights from a list of labels.

    Args:
        labels: List or array of integer class labels.
        num_classes: Total number of classes.
        weight_type: 'sqrt', 'inv', or 'uniform'.

    Returns:
        Tensor of shape (num_classes,) with per-class weights.
    """
    import numpy as np

    label_counts = np.bincount(labels, minlength=num_classes).astype(np.float64)

    if weight_type == "uniform":
        weights = np.ones(num_classes)
    elif weight_type == "inv":
        weights = 1.0 / (label_counts + 1)
    elif weight_type == "sqrt":
        weights = 1.0 / np.sqrt(label_counts + 1)
    else:
        raise ValueError(f"Unknown weight_type: {weight_type}")

    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)
