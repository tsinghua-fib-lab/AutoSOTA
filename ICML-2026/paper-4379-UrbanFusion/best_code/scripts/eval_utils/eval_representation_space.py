#!/usr/bin/env python3
"""
Description: This script contains functions to evaluate the learned
representation space of a model. This includes plotting the representations
using t-SNE and PCA.
"""

import os
import re

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import hsv_to_rgb
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def parse_text_attributes(text_data: list) -> tuple:
    """
    Parse text data to extract color and size information.
    Only for the SyntheticMultiModalDataset.

    Parameters
    ----------
    text_data : list of str
        List of text descriptions corresponding to each representation.

    Returns
    -------
    tuple
        Tuple of lists containing color and size information.
    """
    color_mapping = {"red": "red", "green": "green", "blue": "blue"}
    size_mapping = {"small": 5, "medium": 25, "large": 50, "huge": 100}
    colors = []
    sizes = []
    for batch in text_data:
        for text in batch:
            color = "black"  # Default color
            size = 100  # Default size
            for word in text.split():
                if word in color_mapping:
                    color = color_mapping[word]
                if word in size_mapping:
                    size = size_mapping[word]

            colors.append(color)
            sizes.append(size)
    return colors, sizes


def plot_synthetic_representations(
    representations: torch.Tensor,
    images: torch.Tensor,
    text_data: list,
    token_index: int,
    epoch: int = None,
    stage: str = "",
    only_cls: bool = False,
    title: str = "",
    save_plots: bool = False,
    show_plots: bool = False,
    save_dir: str = None,
) -> None:
    """
    Apply t-SNE and PCA on the representations and plot the 2D visualization
    with color and size. Only suitable for SyntheticMultiModalDataset.

    Parameters
    ----------
    representations : torch.Tensor
        The learned representations (B, num_tokens, contrastive_dim).
    images : torch.Tensor
        The corresponding images (B, C, H, W) to compute intensity for
        coloring.
    text_data : list of str
        List of text descriptions corresponding to each representation.
    token_index : int
        Index of the token to visualize per sample.
    only_cls : bool, optional
        Whether output is only the CLS token, by default False.
    title : str, optional
        Title of the plot, by default "".
    save_plots : bool, optional
        Whether to save the plots, by default False.
    show_plots : bool, optional
        Whether to show the plots, by default False.
    save_dir : str, optional
        Directory to save the plots, by default None.
    """
    # Add epoch and stage to title
    if epoch is not None:
        epoch = epoch + 1  # Start from 1
        title = f"{title} (Epoch {epoch}, {stage})"

    # Format title for saving plots
    formatted_title = re.sub(r"[,\(\)]", "", title.replace(" ", "_")).lower()

    # Ensure the directory exists
    if save_plots and save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Parse text attributes for color and size
    colors, sizes = parse_text_attributes(text_data)

    # Compute average intensity per image
    avg_intensity_red = images[:, 0, :, :].mean(dim=(1, 2))
    avg_intensity_green = images[:, 1, :, :].mean(dim=(1, 2))
    avg_intensity_blue = images[:, 2, :, :].mean(dim=(1, 2))

    # Use torch.max to compute the element-wise maximum across tensors
    avg_intensity = torch.max(
        torch.stack(
            [avg_intensity_red, avg_intensity_green, avg_intensity_blue]
        ),
        dim=0,
    ).values

    # Normalize intensity values between 0 and 1 for filling
    intensity_min = avg_intensity.min()
    intensity_max = avg_intensity.max()
    normalized_intensity = (avg_intensity - intensity_min) / (
        intensity_max - intensity_min
    )
    normalized_intensity = normalized_intensity.cpu().numpy()

    # Select only one token per sample
    if only_cls:
        selected_representations = representations
    else:
        selected_representations = representations[:, token_index, :]

    # Convert to numpy
    selected_representations = selected_representations.cpu().numpy()

    # Apply PCA
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(selected_representations)

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=10, random_state=42)
    tsne_results = tsne.fit_transform(selected_representations)

    # Plot and save PCA
    plt.figure(figsize=(8, 6))
    for i in range(len(pca_results)):
        facecolor = mcolors.to_rgba(
            colors[i], alpha=normalized_intensity[i]
        )  # Apply alpha only to face color
        plt.scatter(
            pca_results[i, 0],
            pca_results[i, 1],
            facecolors=facecolor,
            edgecolors=colors[i],  # Edge remains solid
            s=sizes[i],
        )
    plt.title(f"PCA Visualization of Representations: {title}")
    plt.xlabel("PCA Dimension 1")
    plt.ylabel("PCA Dimension 2")
    if save_plots:
        plt.savefig(f"{save_dir}/pca_{formatted_title}.png")
    if show_plots:
        plt.show()

    # Plot and save t-SNE
    plt.figure(figsize=(8, 6))
    for i in range(len(tsne_results)):
        facecolor = mcolors.to_rgba(
            colors[i], alpha=normalized_intensity[i]
        )  # Apply alpha only to face color
        plt.scatter(
            tsne_results[i, 0],
            tsne_results[i, 1],
            facecolors=facecolor,
            edgecolors=colors[i],  # Edge remains solid
            s=sizes[i],
        )
    plt.title(f"t-SNE Visualization of Representations: {title}")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    if save_plots:
        plt.savefig(f"{save_dir}/tsne_{formatted_title}.png")
    if show_plots:
        plt.show()


def interpolate_color(val, val_min, val_max, color_low, color_high):
    """
    Linearly interpolate between two colors based on the normalized value.
    """
    # Avoid division by zero if min equals max.
    t = (val - val_min) / (val_max - val_min) if val_max != val_min else 0.5
    low_rgb = mcolors.to_rgb(color_low)
    high_rgb = mcolors.to_rgb(color_high)
    interp_rgb = tuple(
        low + t * (high - low) for low, high in zip(low_rgb, high_rgb)
    )
    return interp_rgb


def integrated_color(lon, lat, lon_min, lon_max, lat_min, lat_max):
    """
    Compute an integrated color from longitude and latitude using the HSV
    color space.

    Parameters
    ----------
    lon : float
        Longitude value.
    lat : float
        Latitude value.
    lon_min, lon_max : float
        Minimum and maximum longitude values in the dataset.
    lat_min, lat_max : float
        Minimum and maximum latitude values in the dataset.

    Returns
    -------
    tuple
        The RGB tuple after converting from HSV.
    """
    # Normalize the coordinates to [0, 1]
    norm_lon = (
        (lon - lon_min) / (lon_max - lon_min) if lon_max != lon_min else 0.5
    )
    norm_lat = (
        (lat - lat_min) / (lat_max - lat_min) if lat_max != lat_min else 0.5
    )

    # Map normalized lon to hue (0 to 0.7) and norm_lat to saturation, with
    # fixed brightness
    hue = 0.7 * norm_lon  # Adjust the upper limit as desired
    saturation = norm_lat  # Direct mapping for saturation
    value = 0.9  # Fixed brightness

    hsv = np.array([hue, saturation, value])
    return hsv_to_rgb(hsv)


def plot_milan_representations(
    representations: torch.Tensor,
    modality_coordinates: torch.Tensor,
    token_index: int,
    epoch: int = None,
    stage: str = "",
    only_cls: bool = False,
    title: str = "",
    save_plots: bool = False,
    show_plots: bool = False,
    save_dir: str = None,
) -> None:
    """
    Apply PCA and t-SNE on the representations and plot the 2D visualization
    with colors derived from modality coordinates (lon, lat) for the Milan
    dataset. The integrated color mapping uses longitude for hue and latitude
    for saturation.

    Parameters
    ----------
    representations : torch.Tensor
        The learned representations (B, feature_dim) to be reduced.
    modality_coordinates : torch.Tensor or np.array
        The modality coordinates as a 2D vector (lon, lat) for each sample
        (B, 2).
    token_index : int
        Index of the token to visualize per sample.
    epoch : int, optional
        Current epoch, by default None.
    stage : str, optional
        Stage of processing (e.g., training/validation), by default "".
    title : str, optional
        Title of the plot, by default "".
    save_plots : bool, optional
        Whether to save the plots, by default False.
    show_plots : bool, optional
        Whether to display the plots, by default False.
    save_dir : str, optional
        Directory to save the plots, by default None.
    """
    # Add epoch and stage to title if provided
    if epoch is not None:
        epoch = epoch + 1  # Start numbering epochs at 1
        title = f"{title} (Epoch {epoch}, {stage})"

    # Format title for saving
    formatted_title = re.sub(r"[,\(\)]", "", title.replace(" ", "_")).lower()

    # Ensure the directory exists for saving plots
    if save_plots and save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Select only one token per sample
    if only_cls:
        selected_representations = representations
    else:
        selected_representations = representations[:, token_index, :]

    # Convert tensors to numpy arrays if needed
    if isinstance(selected_representations, torch.Tensor):
        representations_np = selected_representations.cpu().numpy()
    else:
        representations_np = selected_representations

    if isinstance(modality_coordinates, torch.Tensor):
        coords = modality_coordinates.cpu().numpy()
    else:
        coords = modality_coordinates

    # Extract longitude and latitude
    lons = coords[:, 0]
    lats = coords[:, 1]

    # Determine min and max values for normalization
    lon_min, lon_max = lons.min(), lons.max()
    lat_min, lat_max = lats.min(), lats.max()

    # Reduce representations using PCA
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(representations_np)

    # Reduce representations using t-SNE
    tsne = TSNE(n_components=2, perplexity=10, random_state=42)
    tsne_results = tsne.fit_transform(representations_np)

    # Plot PCA results with integrated color mapping
    plt.figure(figsize=(8, 6))
    for i in range(len(pca_results)):
        color = integrated_color(
            lons[i], lats[i], lon_min, lon_max, lat_min, lat_max
        )
        plt.scatter(
            pca_results[i, 0],
            pca_results[i, 1],
            facecolors=mcolors.to_rgba(color, alpha=0.8),
            edgecolors=color,
            s=50,
        )
    plt.title(f"PCA Visualization of Milan Representations: {title}")
    plt.xlabel("PCA Dimension 1")
    plt.ylabel("PCA Dimension 2")
    if save_plots and save_dir:
        plt.savefig(f"{save_dir}/pca_{formatted_title}.png")
    if show_plots:
        plt.show()

    # Plot t-SNE results with integrated color mapping
    plt.figure(figsize=(8, 6))
    for i in range(len(tsne_results)):
        color = integrated_color(
            lons[i], lats[i], lon_min, lon_max, lat_min, lat_max
        )
        plt.scatter(
            tsne_results[i, 0],
            tsne_results[i, 1],
            facecolors=mcolors.to_rgba(color, alpha=0.8),
            edgecolors=color,
            s=50,
        )
    plt.title(f"t-SNE Visualization of Milan Representations: {title}")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    if save_plots and save_dir:
        plt.savefig(f"{save_dir}/tsne_{formatted_title}.png")
    if show_plots:
        plt.show()
