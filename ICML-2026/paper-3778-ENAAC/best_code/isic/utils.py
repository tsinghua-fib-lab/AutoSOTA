import torch
import random
import numpy as np
from PIL import Image, ImageDraw
import os
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torch.utils.data import Subset
from collections import defaultdict
from tqdm import tqdm
import torch.nn.functional as F

def add_artificial_bias(batch_images, batch_labels, class_bias_probs, image_size=(224, 224)):
    """
    Adds random elliptical patches to a batch of normalized images (e.g., ResNet-ready inputs),
    based on class-specific probabilities. Also returns binary masks of added patches.

    Args:
        batch_images (Tensor): (B, C, H, W) normalized images
        batch_labels (Tensor): (B,) binary class labels (0 or 1)
        class_bias_probs (dict): e.g., {0: 0.9, 1: 0.2}
        image_size (tuple): image size (H, W)

    Returns:
        Tuple[Tensor, Tensor]: (augmented_images, patch_masks)
    """
    B, C, H, W = batch_images.shape
    mean = torch.tensor([0.485, 0.456, 0.406], device=batch_images.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=batch_images.device).view(3, 1, 1)

    output_images = []
    output_masks = []

    for i in range(B):
        label = batch_labels[i].item()
        p_bias = class_bias_probs.get(label, 0.0)

        # De-normalize
        img = batch_images[i] * std + mean
        img = torch.clamp(img, 0, 1)
        img_pil = transforms.ToPILImage()(img.cpu())
        mask_pil = Image.new("L", (W, H), 0)  # Binary mask (single channel)

        if random.random() < p_bias:
            draw_img = ImageDraw.Draw(img_pil)
            draw_mask = ImageDraw.Draw(mask_pil)

            # Side to place it
            side = random.choice(['left', 'right', 'top', 'bottom'])

            # Random ellipse size
            if side == 'left' or side == 'right':
                w_ellipse = random.randint(50, 60)
                h_ellipse = random.randint(120, 224)
            if side == 'top' or side == 'bottom':
                w_ellipse = random.randint(120, 224)
                h_ellipse = random.randint(50, 60)

            # Random rotation
            angle = random.uniform(-30, 30)

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

            # Create the rotated ellipse as a transparent layer
            patch = Image.new("RGBA", (W, H))
            mask_patch = Image.new("L", (W, H), 0)
            patch_draw = ImageDraw.Draw(patch)
            mask_draw = ImageDraw.Draw(mask_patch)

            ellipse_color = tuple(np.random.randint(0, 255, 3)) + (255,)  # RGBA
            patch_draw.ellipse([x0, y0, x1, y1], fill=ellipse_color)
            mask_draw.ellipse([x0, y0, x1, y1], fill=255)

            # Composite onto base image and mask
            img_pil = Image.alpha_composite(img_pil.convert("RGBA"), patch).convert("RGB")
            mask_pil = mask_patch

        # Back to tensor and normalize
        img_tensor = transforms.ToTensor()(img_pil).to(batch_images.device)
        img_tensor = (img_tensor - mean) / std

        mask_tensor = transforms.ToTensor()(mask_pil).to(batch_images.device)  # 1xHxW, 0–1 float

        output_images.append(img_tensor)
        output_masks.append(mask_tensor)

    return torch.stack(output_images), torch.stack(output_masks)

def create_balanced_subset(dataset, targets, max_per_class=None, seed=42):
    """
    Create a balanced subset of a dataset with equal number of samples per class.

    Args:
        dataset: PyTorch dataset
        targets: List or tensor of class labels, same order as dataset
        max_per_class: Optional[int], limit per class (defaults to min class count)
        seed: Random seed for reproducibility

    Returns:
        Subset: a torch.utils.data.Subset containing balanced data
    """
    print('Balancing dataset...')
    random.seed(seed)

    # Map each class to its indices
    class_indices = defaultdict(list)
    for idx, label in enumerate(targets):
        class_indices[int(label)].append(idx)

    # Get minimum count (or a manual limit)
    if max_per_class is None:
        min_class_count = min(len(idxs) for idxs in class_indices.values())
    else:
        min_class_count = min(max_per_class, min(len(idxs) for idxs in class_indices.values()))

    # Sample equal number from each class
    selected_indices = []
    for cls, idxs in class_indices.items():
        selected = random.sample(idxs, min_class_count)
        selected_indices.extend(selected)

    random.shuffle(selected_indices)
    return Subset(dataset, selected_indices)

def dilate_masks_torch(masks, dilation_pixels=10):
    """
    Dilates binary masks by N pixels using max pooling (PyTorch only).

    Args:
        masks (Tensor): shape (B, 1, H, W), binary masks (0s and 1s)
        dilation_pixels (int): number of pixels to dilate

    Returns:
        Tensor: dilated masks, same shape and device
    """
    kernel_size = 2 * dilation_pixels + 1
    padding = dilation_pixels

    # Ensure binary input
    masks = (masks > 0.5).float()

    # Max pool to simulate dilation
    dilated = F.max_pool2d(masks, kernel_size=kernel_size, stride=1, padding=padding)

    return dilated