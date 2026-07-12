import torch
import random
import numpy as np


def set_all_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_mask(data, mask_type, obs_ratio):
    if mask_type == 'random':
        mask = torch.zeros_like(data).reshape(-1)
        num_obs = int(obs_ratio * mask.numel())
        mask[:num_obs] = 1
        perm = torch.randperm(mask.numel())
        mask = mask[perm].reshape(data.shape)
    else:
        raise NotImplementedError(f'Unsupported mask type {mask_type}!')
    return mask


def image2vdt256(x):
    x = x.permute(1, 2, 0)  # C H W -> H W C
    nway = [2] * 16 + [3]
    x = x.reshape(nway)
    per_nway = [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15, 16]
    x = x.permute(per_nway)
    x = x.reshape([4, 4, 4, 4, 4, 4, 4, 4, 3])
    x = x.permute([8, 0, 1, 2, 3, 4, 5, 6, 7])
    return x


def vdt2image256(x):
    x = x.permute([1, 2, 3, 4, 5, 6, 7, 8, 0])
    nway = [2] * 16 + [3]
    x = x.reshape(nway)
    per_nway = [0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15, 16]
    x = x.permute(per_nway)
    x = x.reshape([256, 256, 3])
    x = x.permute(2, 0, 1)  # H W C -> C H W
    return x


def image2patches(image, patch_size, stride=None):
    """
    Convert image into patches and stack them into a 4D tensor.
    
    Args:
        image (torch.Tensor): Input image of shape (C, H, W) or (3, 256, 256)
        patch_size (int or tuple): Size of each patch. If int, uses square patches.
        stride (int or tuple, optional): Stride for patch extraction. If None, uses patch_size (no overlap).
    
    Returns:
        torch.Tensor: 4D tensor of shape (C, patch_h, patch_w, num_patches)
                     where num_patches = num_patches_h * num_patches_w
    """
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    
    if stride is None:
        stride = patch_size
    elif isinstance(stride, int):
        stride = (stride, stride)
    
    C, H, W = image.shape
    patch_h, patch_w = patch_size
    stride_h, stride_w = stride
    
    # Calculate number of patches
    num_patches_h = (H - patch_h) // stride_h + 1
    num_patches_w = (W - patch_w) // stride_w + 1
    num_patches = num_patches_h * num_patches_w
    
    # Use unfold to extract patches
    # unfold(dimension, size, step) extracts sliding windows
    patches = image.unfold(1, patch_h, stride_h).unfold(2, patch_w, stride_w)
    # Shape after unfold: (C, num_patches_h, num_patches_w, patch_h, patch_w)
    
    # Reshape to (C, patch_h, patch_w, num_patches)
    patches = patches.contiguous().view(C, num_patches_h * num_patches_w, patch_h, patch_w)
    patches = patches.permute(0, 2, 3, 1)  # (C, patch_h, patch_w, num_patches)
    
    return patches


def patches2image(patches, image_size, patch_size, stride=None, window_type="hann"):
    """
    Reconstruct image from patches (reverse operation of image2patches).
    
    Args:
        patches (torch.Tensor): 4D tensor of shape (C, patch_h, patch_w, num_patches)
        image_size (tuple): Original image size (H, W)
        patch_size (int or tuple): Size of each patch
        stride (int or tuple, optional): Stride used for patch extraction. If None, uses patch_size.
        window_type (str): Window type for smooth blending ("hann", "uniform"). Default "hann".
    
    Returns:
        torch.Tensor: Reconstructed image of shape (C, H, W)
    """
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    
    if stride is None:
        stride = patch_size
    elif isinstance(stride, int):
        stride = (stride, stride)
    
    C, patch_h, patch_w, num_patches = patches.shape
    H, W = image_size
    stride_h, stride_w = stride
    
    # Calculate number of patches
    num_patches_h = (H - patch_h) // stride_h + 1
    num_patches_w = (W - patch_w) // stride_w + 1
    
    # Initialize reconstructed image and weight accumulator
    image = torch.zeros(C, H, W, device=patches.device, dtype=patches.dtype)
    weight = torch.zeros(C, H, W, device=patches.device, dtype=patches.dtype)
    
    # Reshape patches to (C, num_patches, patch_h, patch_w)
    patches = patches.permute(0, 3, 1, 2)
    
    # ALGO-02: Pre-compute 2D window for smooth blending
    if window_type == "hann":
        h_hann = torch.hann_window(patch_h, device=patches.device, dtype=patches.dtype)
        w_hann = torch.hann_window(patch_w, device=patches.device, dtype=patches.dtype)
        window_2d = h_hann.unsqueeze(1) * w_hann.unsqueeze(0)  # (patch_h, patch_w)
    else:
        window_2d = torch.ones(patch_h, patch_w, device=patches.device, dtype=patches.dtype)
    
    # Place patches back into image with window weighting
    patch_idx = 0
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            h_start = i * stride_h
            h_end = h_start + patch_h
            w_start = j * stride_w
            w_end = w_start + patch_w
            
            # Apply window weighting
            weighted_patch = patches[:, patch_idx, :, :] * window_2d.unsqueeze(0)
            image[:, h_start:h_end, w_start:w_end] += weighted_patch
            weight[:, h_start:h_end, w_start:w_end] += window_2d.unsqueeze(0)
            patch_idx += 1
    
    # Normalize by accumulated weights
    image = image / weight.clamp(min=1e-8)
    
    return image
