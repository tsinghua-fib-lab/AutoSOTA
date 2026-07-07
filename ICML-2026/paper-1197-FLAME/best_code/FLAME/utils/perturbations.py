"""Image perturbation utilities for constructing contrastive training streams."""

from __future__ import annotations

import io
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image


def create_spatial_gaussian_kernel(
    kernel_size: int,
    sigma: float,
    channels: int,
    device: torch.device
) -> torch.Tensor:
    """Create a 2D Gaussian kernel for spatial blurring.

    Args:
        kernel_size: Size of the kernel (odd number)
        sigma: Standard deviation of the Gaussian
        channels: Number of channels (for grouped convolution)
        device: Torch device

    Returns:
        2D Gaussian kernel
    """
    radius = kernel_size // 2
    coords = torch.arange(-radius, radius+1, device=device, dtype=torch.float32)
    x, y = torch.meshgrid(coords, coords, indexing='ij')
    gaussian = torch.exp(-(x**2 + y**2) / (2*sigma**2))
    gaussian /= gaussian.sum()
    # Shape for grouped convolution [C, 1, K, K]
    return gaussian.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)


def apply_spatial_gaussian_blur(
    video_tensor: torch.Tensor,
    sigma: float,
    kernel_sizes: Tuple[int, ...] = (5, ),
) -> torch.Tensor:
    """Apply a sequence of spatial Gaussian blurs to a video tensor.

    Args:
        video_tensor: Tensor [B, C, T, H, W]
        sigma: Std-dev for spatial Gaussian kernels
        kernel_sizes: Iterable of odd ints ∈ {3,5,7,9,11,13,15}

    Returns:
        Blurred video tensor of same shape
    """
    B, C, T, H, W = video_tensor.shape
    device = video_tensor.device

    valid_sizes = (3, 5, 7, 9, 11, 13, 15)
    for k in kernel_sizes:
        if k not in valid_sizes:
            raise ValueError(f"Invalid kernel size: {k}, must be one of {valid_sizes}")

    x = video_tensor
    for kernel_size in kernel_sizes:
        kernel = create_spatial_gaussian_kernel(kernel_size, sigma, C, device)
        padding = kernel_size // 2

        # Merge B and T dims for 2D conv
        xt = x.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
        xt = F.conv2d(xt, kernel, padding=padding, groups=C)
        x = xt.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)

    return x


def add_gaussian_noise(image, mean=0, std=0.1):
    """Add Gaussian noise to a torch tensor image."""
    noise = torch.randn_like(image) * std + mean
    noisy_image = image + noise
    return torch.clamp(noisy_image, 0, 1)  # Ensure values are in [0, 1] range


def apply_blur_to_image(image, sigma=0.7, kernel_sizes=(5,)):
    """Apply spatial Gaussian blur to a PIL image using the existing apply_spatial_gaussian_blur function.

    Args:
        image: PIL image
        sigma: Blur strength (standard deviation)
        kernel_sizes: Tuple of kernel sizes to use

    Returns:
        Blurred PIL image
    """
    import torch
    import torch.nn.functional as F

    # Convert PIL to tensor [C, H, W]
    img_tensor = TF.to_tensor(image)

    # Add batch and time dimensions to match expected input [B, C, T, H, W]
    video_tensor = img_tensor.unsqueeze(0).unsqueeze(2)  # [1, C, 1, H, W]

    # Move to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video_tensor = video_tensor.to(device)

    # Apply blur using existing function
    blurred_tensor = apply_spatial_gaussian_blur(
        video_tensor,
        sigma=sigma,
        kernel_sizes=kernel_sizes
    )

    return video_tensor.squeeze(0).squeeze(1).cpu(), blurred_tensor.squeeze(0).squeeze(1).cpu()


def apply_contrastive_blur_to_image(image, sigma=0.7, kernel_sizes=(5,)):
    """Apply contrastive blur to a PIL image, creating both sharp and blurred versions.
    
    For contrastive blur:
    - orig = original image
    - pert = orig + delta (blurred version)  
    - sharp = orig - delta (sharpened version)
    
    Args:
        image: PIL image
        sigma: Blur strength (standard deviation)
        kernel_sizes: Tuple of kernel sizes to use
        
    Returns:
        tuple: (sharp_tensor, pert_tensor) where sharp is the sharpened version
    """
    import torch
    import torch.nn.functional as F
    
    # Convert PIL to tensor [C, H, W]
    img_tensor = TF.to_tensor(image)
    
    # Add batch and time dimensions to match expected input [B, C, T, H, W]
    video_tensor = img_tensor.unsqueeze(0).unsqueeze(2)  # [1, C, 1, H, W]
    
    # Move to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video_tensor = video_tensor.to(device)
    
    # Apply blur to get the blurred version
    blurred_tensor = apply_spatial_gaussian_blur(
        video_tensor,
        sigma=sigma,
        kernel_sizes=kernel_sizes
    )
    
    # Calculate delta: blur_delta = blurred - original
    blur_delta = blurred_tensor - video_tensor
    
    # Create sharp version: sharp = original - delta
    sharp_tensor = video_tensor - blur_delta
    
    # Clamp to valid range [0, 1]
    sharp_tensor = torch.clamp(sharp_tensor, 0, 1)
    pert_tensor = torch.clamp(blurred_tensor, 0, 1)
    
    return sharp_tensor.squeeze(0).squeeze(1).cpu(), pert_tensor.squeeze(0).squeeze(1).cpu()


def apply_jpeg_compression(image, quality=75):
    """Apply JPEG compression to a PIL image.
    
    Args:
        image: PIL image
        quality: JPEG quality factor (1-100, lower = more compression artifacts)
        
    Returns:
        PIL image with JPEG compression artifacts
    """
    # Save to memory buffer as JPEG
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    
    # Load back from buffer
    compressed_image = Image.open(buffer)
    
    return compressed_image


def apply_contrastive_jpeg_compression(image, quality=50):
    """Apply contrastive JPEG compression, creating sharp and compressed versions.
    
    For contrastive compression:
    - orig = original image
    - pert = orig + delta (more compressed version)
    - sharp = orig - delta (less compressed/enhanced version)
    
    Args:
        image: PIL image
        quality: JPEG quality factor for compressed version (lower = more artifacts)
        
    Returns:
        tuple: (sharp_tensor, pert_tensor) where sharp is enhanced, pert is compressed
    """
    # Convert PIL to tensor [C, H, W]
    orig_tensor = TF.to_tensor(image)
    
    # Create compressed version (pert = orig + delta)
    compressed_pil = apply_jpeg_compression(apply_jpeg_compression(image, quality), quality=quality)
    compressed_tensor = TF.to_tensor(compressed_pil)
    
    # Calculate delta: compression_delta = compressed - original  
    compression_delta = compressed_tensor - orig_tensor
    
    # Create sharp version: sharp = original - delta
    sharp_tensor = orig_tensor - compression_delta
    
    # Clamp to valid range [0, 1]
    sharp_tensor = torch.clamp(sharp_tensor, 0, 1)
    pert_tensor = torch.clamp(compressed_tensor, 0, 1)
    
    return sharp_tensor, pert_tensor


def apply_blur_to_image_tensor(img_tensor, sigma):
    """
    Apply Gaussian blur to an image tensor.
    
    Args:
        img_tensor: Tensor of shape (C, H, W) with values in [0, 1]
        sigma: Standard deviation for Gaussian blur
    
    Returns:
        Blurred tensor of same shape
    """
    if sigma <= 0:
        return img_tensor.clone()
    
    # Add batch and time dimensions to match expected input [B, C, T, H, W]
    video_tensor = img_tensor.unsqueeze(0).unsqueeze(2)  # [1, C, 1, H, W]
    
    # Apply blur using existing function
    blurred_tensor = apply_spatial_gaussian_blur(
        video_tensor,
        sigma=sigma,
        kernel_sizes=(5,)
    )
    
    return blurred_tensor.squeeze(0).squeeze(1)


def apply_jpeg_compression_to_tensor(img_tensor, quality):
    """
    Apply JPEG compression to a tensor image.
    
    Args:
        img_tensor: Tensor of shape (C, H, W) with values in [0, 1]
        quality: JPEG quality (1-100, lower = more compression)
    
    Returns:
        Compressed tensor of same shape
    """
    # Convert tensor to PIL Image
    pil_img = TF.to_pil_image(img_tensor)
    
    # Apply existing JPEG compression function
    compressed_pil = apply_jpeg_compression(pil_img, quality)
    
    # Convert back to tensor
    compressed_tensor = TF.to_tensor(compressed_pil)
    
    return compressed_tensor
