"""Utilities for computing and managing Wiener filter matrices."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader

LOGGER = logging.getLogger(__name__)


def compute_wiener_filter(
    dataloader: DataLoader,
    device: torch.device,
    resolution: int,
    n_channels: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the covariance matrix S and mean from a dataset.
    
    Parameters
    ----------
    dataloader : DataLoader
        DataLoader yielding batches of images.
    device : torch.device
        Device on which to perform computation.
    resolution : int
        Spatial resolution of images (assumed square).
    n_channels : int
        Number of image channels.
    
    Returns
    -------
    S : torch.Tensor
        Covariance matrix of shape (n_pixels, n_pixels).
    mean : torch.Tensor
        Mean image of shape (n_pixels,).
    """
    
    LOGGER.info("Computing dataset statistics for Wiener filter...")
    LOGGER.info("Expected resolution: %dx%d, Channels: %d", resolution, resolution, n_channels)
    
    # Infer actual dimensions from first batch
    sum_images = None
    total_samples = 0
    n_pixels = None

    
    # First pass: compute mean
    for batch in dataloader:
        if isinstance(batch, (tuple, list)):
            images = batch[0]
        else:
            images = batch
        
        images = images.to(device).flatten(start_dim=1)  # [batch, n_pixels]
        
        # Initialize accumulator on first batch
        if sum_images is None:
            n_pixels = images.shape[1]
            sum_images = torch.zeros(n_pixels, device=device)
            LOGGER.info("Actual n_pixels from data: %d", n_pixels)
        
        sum_images += images.sum(dim=0)
        total_samples += images.shape[0]
    
    if sum_images is None or n_pixels is None:
        raise RuntimeError("No data found in dataloader")
    
    mean = sum_images / total_samples
    LOGGER.info("Computed mean from %d samples", total_samples)
    
    # Second pass: compute covariance
    cov_accumulator = torch.zeros(n_pixels, n_pixels, device=device)
    
    for batch in dataloader:
        if isinstance(batch, (tuple, list)):
            images = batch[0]
        else:
            images = batch
        
        images = images.to(device).flatten(start_dim=1)  # [batch, n_pixels]
        centered = images - mean.unsqueeze(0)  # [batch, n_pixels]
        cov_accumulator += centered.T @ centered  # accumulate outer product
    
    S = cov_accumulator / (total_samples - 1)
    LOGGER.info("Computed covariance matrix of shape %s", tuple(S.shape))
    
    return S, mean


def save_wiener_filter(
    U: torch.Tensor,
    LA: torch.Tensor,
    Vh: torch.Tensor,
    mean: torch.Tensor,
    save_path: Path,
) -> None:
    """Save the Wiener filter SVD components to disk.
    
    Parameters
    ----------
    U : torch.Tensor
        U matrix from SVD.
    LA : torch.Tensor
        Singular values (diagonal).
    Vh : torch.Tensor
        Vh matrix from SVD.
    mean : torch.Tensor
        Mean image vector.
    save_path : Path
        Directory in which to save the matrices.
    """
    save_path.mkdir(parents=True, exist_ok=True)
    
    torch.save(U.cpu(), save_path / "U.pt")
    torch.save(LA.cpu(), save_path / "LA.pt")
    torch.save(Vh.cpu(), save_path / "Vh.pt")
    torch.save(mean.cpu(), save_path / "mean.pt")
    
    LOGGER.info("Saved Wiener filter SVD components to %s", save_path)


def load_wiener_filter(
    load_path: Path,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load the Wiener filter SVD components from disk.
    
    If only the covariance matrix S is found (legacy), it computes SVD,
    saves the components, and returns them.
    
    Parameters
    ----------
    load_path : Path
        Directory from which to load the matrices.
    device : torch.device, optional
        Device to which to load the tensors.
    
    Returns
    -------
    U : torch.Tensor
    LA : torch.Tensor
    Vh : torch.Tensor
    mean : torch.Tensor
    """
    U_path = load_path / "U.pt"
    LA_path = load_path / "LA.pt"
    Vh_path = load_path / "Vh.pt"
    mean_path = load_path / "mean.pt"
    
    # Check if we saved the SVD components
    if U_path.exists() and LA_path.exists() and Vh_path.exists() and mean_path.exists():
        U = torch.load(U_path, map_location=device, weights_only=True)
        LA = torch.load(LA_path, map_location=device, weights_only=True)
        Vh = torch.load(Vh_path, map_location=device, weights_only=True)
        mean = torch.load(mean_path, map_location=device, weights_only=True)
        LOGGER.info("Loaded Wiener filter SVD components from %s", load_path)
        return U, LA, Vh, mean

    raise FileNotFoundError(
        f"Wiener filter not found at {load_path}. "
        f"Expected SVD components (U.pt, LA.pt, Vh.pt, mean.pt) to be present in the directory."
    )
