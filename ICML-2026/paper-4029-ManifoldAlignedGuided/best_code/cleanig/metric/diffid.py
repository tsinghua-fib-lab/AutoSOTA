import numpy as np
from typing import Optional, List, Tuple, Union

import torch
import torch.nn.functional as F


def compute_diffid_score(
    model: torch.nn.Module,
    images: torch.Tensor,
    attributions: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    ratios: Optional[List[float]] = None,
    baseline_method: str = 'mean',
    use_soft_metric: bool = False,
    return_curves: bool = False
) -> Union[float, Tuple[float, dict]]:
    """
    Compute DiffID score for attribution quality evaluation.

    DiffID measures the difference between insertion and deletion curves:
    - Deletion: Remove most important pixels first (good attributions → rapid accuracy drop)
    - Insertion: Remove least important pixels first (good attributions → slow accuracy drop)
    - DiffID = Insertion Accuracy - Deletion Accuracy (higher is better)

    Args:
        model: Classification model to evaluate
        images: Input images tensor [batch_size, C, H, W]
        attributions: Attribution maps [batch_size, C, H, W]
        labels: True labels [batch_size]. If None, uses model's original predictions
        ratios: List of removal ratios to evaluate (default: [0.05, 0.1, ..., 0.95])
        baseline_method: Method for replacing pixels ('mean', 'zero', 'blur')
        use_soft_metric: If True, use confidence drop instead of binary accuracy (NEW)
        return_curves: If True, also return insertion/deletion curves

    Returns:
        If return_curves=False: Average DiffID score across all ratios
        If return_curves=True: Tuple of (DiffID score, dict with curves)
    """

    if ratios is None:
        # ratios = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    device = images.device
    batch_size = images.shape[0]

    # Get original predictions if labels not provided
    if labels is None:
        with torch.no_grad():
            outputs = model(images)
            labels = torch.argmax(outputs, dim=1)

    # Get original confidences for soft metric
    if use_soft_metric:
        with torch.no_grad():
            original_outputs = model(images)
            original_probs = F.softmax(original_outputs, dim=1)
            # Get probability of correct class for each sample
            original_confidences = original_probs[torch.arange(batch_size), labels]

    # Flatten for easier manipulation
    # Use reshape instead of view to handle non-contiguous tensors (e.g., from WebP compression)
    num_pixels = images.shape[1] * images.shape[2] * images.shape[3]
    flat_images = images.reshape(batch_size, -1)
    flat_attributions = torch.abs(attributions).reshape(batch_size, -1)

    insertion_scores = []
    deletion_scores = []

    for ratio in ratios:
        num_perturb = int(num_pixels * ratio)

        if use_soft_metric:
            # Deletion: Remove most important pixels
            deletion_score = _evaluate_perturbation_soft(
                model, flat_images, flat_attributions, labels,
                num_perturb, images.shape, original_confidences,
                descending=True, baseline_method=baseline_method
            )
            deletion_scores.append(deletion_score)

            # Insertion: Remove least important pixels
            insertion_score = _evaluate_perturbation_soft(
                model, flat_images, flat_attributions, labels,
                num_perturb, images.shape, original_confidences,
                descending=False, baseline_method=baseline_method
            )
            insertion_scores.append(insertion_score)
        else:
            # Deletion: Remove most important pixels
            deletion_score = _evaluate_perturbation(
                model, flat_images, flat_attributions, labels,
                num_perturb, images.shape, descending=True, baseline_method=baseline_method
            )
            deletion_scores.append(deletion_score)

            # Insertion: Remove least important pixels
            insertion_score = _evaluate_perturbation(
                model, flat_images, flat_attributions, labels,
                num_perturb, images.shape, descending=False, baseline_method=baseline_method
            )
            insertion_scores.append(insertion_score)

    # Compute DiffID scores
    diffid_scores = [ins - del_ for ins, del_ in zip(insertion_scores, deletion_scores)]
    avg_diffid = np.mean(diffid_scores)

    if return_curves:
        return avg_diffid, {
            'ratios': ratios,
            'insertion_scores': insertion_scores,
            'deletion_scores': deletion_scores,
            'diffid_scores': diffid_scores,
            'metric_type': 'soft' if use_soft_metric else 'binary'
        }

    return avg_diffid


def _evaluate_perturbation(
    model: torch.nn.Module,
    flat_images: torch.Tensor,
    flat_attributions: torch.Tensor,
    labels: torch.Tensor,
    num_perturb: int,
    original_shape: tuple,
    descending: bool,
    baseline_method: str = 'mean'
) -> float:
    """
    Evaluate model accuracy after perturbing pixels based on attribution importance.

    Args:
        model: Classification model
        flat_images: Flattened images [batch_size, num_pixels]
        flat_attributions: Flattened attribution maps [batch_size, num_pixels]
        labels: True labels
        num_perturb: Number of pixels to perturb
        original_shape: Original image shape for reshaping
        descending: If True, perturb most important pixels (deletion)
                   If False, perturb least important pixels (insertion)
        baseline_method: Method for replacing pixels

    Returns:
        Accuracy after perturbation
    """
    batch_size = flat_images.shape[0]
    device = flat_images.device

    # Sort pixels by attribution importance
    sorted_indices = torch.argsort(flat_attributions, dim=1, descending=descending)
    perturb_indices = sorted_indices[:, :num_perturb]

    # Create baseline values
    if baseline_method == 'mean':
        # Use mean of remaining pixels
        perturb_mask = torch.ones_like(flat_images)
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)
        perturb_mask[batch_indices, perturb_indices] = 0

        # Calculate mean of non-perturbed pixels
        sum_preserved = (flat_images * perturb_mask).sum(dim=1, keepdim=True)
        count_preserved = perturb_mask.sum(dim=1, keepdim=True)
        baseline_values = sum_preserved / (count_preserved + 1e-8)

    elif baseline_method == 'zero':
        baseline_values = torch.zeros(batch_size, 1, device=device)

    elif baseline_method == 'blur':
        # Use blurred version of image
        images_reshaped = flat_images.view(original_shape)
        blurred = F.avg_pool2d(images_reshaped, kernel_size=11, stride=1, padding=5)
        baseline_values = blurred.view(batch_size, -1)
    else:
        raise ValueError(f"Unknown baseline method: {baseline_method}")

    # Apply perturbation
    perturbed_images = flat_images.clone()
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)

    if baseline_method == 'blur':
        perturbed_images[batch_indices, perturb_indices] = baseline_values[batch_indices, perturb_indices]
    else:
        perturbed_images[batch_indices, perturb_indices] = baseline_values.expand_as(perturbed_images)[batch_indices, perturb_indices]

    # Reshape and evaluate
    perturbed_images = perturbed_images.view(original_shape)

    with torch.no_grad():
        outputs = model(perturbed_images)
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == labels).float().mean().item()

    return accuracy


def _evaluate_perturbation_soft(
    model: torch.nn.Module,
    flat_images: torch.Tensor,
    flat_attributions: torch.Tensor,
    labels: torch.Tensor,
    num_perturb: int,
    original_shape: tuple,
    original_confidences: torch.Tensor,
    descending: bool,
    baseline_method: str = 'mean'
) -> float:
    """
    Evaluate model confidence retention after perturbing pixels (SOFT METRIC).

    Instead of binary accuracy, this measures how much the confidence
    in the correct class is retained after perturbation.

    Args:
        model: Classification model
        flat_images: Flattened images [batch_size, num_pixels]
        flat_attributions: Flattened attribution maps [batch_size, num_pixels]
        labels: True labels
        num_perturb: Number of pixels to perturb
        original_shape: Original image shape for reshaping
        original_confidences: Original confidence scores for correct classes
        descending: If True, perturb most important pixels (deletion)
                   If False, perturb least important pixels (insertion)
        baseline_method: Method for replacing pixels

    Returns:
        Average confidence retention ratio (0 to 1)
    """
    batch_size = flat_images.shape[0]
    device = flat_images.device

    # Sort pixels by attribution importance
    sorted_indices = torch.argsort(flat_attributions, dim=1, descending=descending)
    perturb_indices = sorted_indices[:, :num_perturb]

    # Create baseline values
    if baseline_method == 'mean':
        # Use mean of remaining pixels
        perturb_mask = torch.ones_like(flat_images)
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)
        perturb_mask[batch_indices, perturb_indices] = 0

        # Calculate mean of non-perturbed pixels
        sum_preserved = (flat_images * perturb_mask).sum(dim=1, keepdim=True)
        count_preserved = perturb_mask.sum(dim=1, keepdim=True)
        baseline_values = sum_preserved / (count_preserved + 1e-8)

    elif baseline_method == 'zero':
        baseline_values = torch.zeros(batch_size, 1, device=device)

    elif baseline_method == 'blur':
        # Use blurred version of image
        images_reshaped = flat_images.view(original_shape)
        blurred = F.avg_pool2d(images_reshaped, kernel_size=11, stride=1, padding=5)
        baseline_values = blurred.view(batch_size, -1)
    else:
        raise ValueError(f"Unknown baseline method: {baseline_method}")

    # Apply perturbation
    perturbed_images = flat_images.clone()
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)

    if baseline_method == 'blur':
        perturbed_images[batch_indices, perturb_indices] = baseline_values[batch_indices, perturb_indices]
    else:
        perturbed_images[batch_indices, perturb_indices] = baseline_values.expand_as(perturbed_images)[batch_indices, perturb_indices]

    # Reshape and evaluate
    perturbed_images = perturbed_images.view(original_shape)

    with torch.no_grad():
        outputs = model(perturbed_images)
        probs = F.softmax(outputs, dim=1)
        # Get probability of correct class after perturbation
        perturbed_confidences = probs[torch.arange(batch_size), labels]

    # Calculate confidence retention ratio
    # Ratio of confidence after perturbation to original confidence
    confidence_retention = (perturbed_confidences / (original_confidences + 1e-8)).mean().item()

    # Clamp to [0, 1] range (in case of numerical issues)
    confidence_retention = max(0.0, min(1.0, confidence_retention))

    return confidence_retention