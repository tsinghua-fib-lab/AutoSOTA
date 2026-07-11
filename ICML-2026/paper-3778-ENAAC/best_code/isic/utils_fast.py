"""
Fast GPU-based version of add_artificial_bias using pure PyTorch operations.
Replaces the slow PIL-based implementation.
"""
import torch
import random
import numpy as np


def add_artificial_bias_fast(batch_images, batch_labels, class_bias_probs, image_size=(224, 224)):
    """
    Fast GPU-based implementation of add_artificial_bias.
    Uses PyTorch operations instead of PIL for ~100x speedup.

    Args:
        batch_images (Tensor): (B, C, H, W) normalized images
        batch_labels (Tensor): (B,) binary class labels (0 or 1)
        class_bias_probs (dict): e.g., {0: 0.9, 1: 0.2}
        image_size (tuple): image size (H, W)

    Returns:
        Tuple[Tensor, Tensor]: (augmented_images, patch_masks)
    """
    B, C, H, W = batch_images.shape
    device = batch_images.device

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # De-normalize
    images = batch_images * std + mean
    images = torch.clamp(images, 0, 1)

    # Create patch masks (batch_size, 1, H, W)
    masks = torch.zeros(B, 1, H, W, device=device)

    for i in range(B):
        label = batch_labels[i].item()
        p_bias = class_bias_probs.get(label, 0.0)

        if random.random() < p_bias:
            side = random.choice(['left', 'right', 'top', 'bottom'])

            if side in ('left', 'right'):
                w_ellipse = random.randint(50, 60)
                h_ellipse = random.randint(120, 224)
            else:
                w_ellipse = random.randint(120, 224)
                h_ellipse = random.randint(50, 60)

            # Determine ellipse position (similar to original, slightly off-image)
            if side == 'left':
                x0 = random.randint(-40, -10)
                y0 = random.randint(0, H - h_ellipse)
            elif side == 'right':
                x0 = random.randint(W - w_ellipse + 10, W - w_ellipse + 40)
                y0 = random.randint(0, H - h_ellipse)
            elif side == 'top':
                x0 = random.randint(0, W - w_ellipse)
                y0 = random.randint(-40, -10)
            else:  # bottom
                x0 = random.randint(0, W - w_ellipse)
                y0 = random.randint(H - h_ellipse + 10, H - h_ellipse + 40)

            x1 = x0 + w_ellipse
            y1 = y0 + h_ellipse

            # Generate ellipse mask using PyTorch
            # Create meshgrid for the ellipse
            yy, xx = torch.meshgrid(
                torch.arange(H, device=device, dtype=torch.float32),
                torch.arange(W, device=device, dtype=torch.float32),
                indexing='ij'
            )

            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            rx = w_ellipse / 2.0
            ry = h_ellipse / 2.0

            # Rotation handled approximately (original PIL rotation also approximate)
            # For simplicity, use the ellipse equation without rotation
            # (Original code rotated ellipses by ±30 degrees; we approximate with axis-aligned)

            # Clamp coordinates to valid image range for mask calculation
            mask_x0 = max(0, x0)
            mask_x1 = min(W, x1)
            mask_y0 = max(0, y0)
            mask_y1 = min(H, y1)

            # Create ellipse: ((x-cx)/rx)^2 + ((y-cy)/ry)^2 <= 1
            ellipse = ((xx - cx) / max(rx, 1)) ** 2 + ((yy - cy) / max(ry, 1)) ** 2 <= 1.0
            ellipse = ellipse.float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

            # Create random color
            color = torch.rand(3, device=device).view(1, 3, 1, 1)

            # Apply patch: blend ellipse region with random color
            images[i:i+1] = images[i:i+1] * (1 - ellipse) + color * ellipse

            masks[i:i+1] = ellipse

    # Re-normalize
    images = (images - mean) / std

    return images, masks


# Monkey-patch the original function
def patch_utils():
    """Replace the slow PIL-based function with the fast PyTorch version."""
    import utils
    utils.add_artificial_bias = add_artificial_bias_fast
    return utils
