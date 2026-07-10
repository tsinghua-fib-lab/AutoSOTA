"""
Transport Plan Visualization

Functions for visualizing optimal transport plans and color/texture transfer between point clouds and meshes.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection

from utils.implementation.eot import barycentric_projection
from utils.math.functions import normalize
from utils.viz.data import plot_pointcloud, plot_mesh


def _default_colors(points):
    """
    Generate default position-based colors for points.

    Creates a color mapping where spatial position determines color.

    Args:
        points: Point cloud, shape (N, D) where D ∈ {2, 3}

    Returns:
        torch.Tensor: RGB colors, shape (N, 3) with values in [0, 1]
    """
    D = points.shape[1]

    if D == 2:
        points_min, points_max = points.min(axis=0, keepdims=True)[0], points.max(axis=0, keepdims=True)[0]
        coords = (points - points_min) / (points_max - points_min)
        colors = torch.stack([coords[:, 0], coords[:, 1], (1 - coords[:, 0]) * (1 - coords[:, 1])], dim=1)

    else:
        norm_points = normalize(points)
        norm_points = norm_points / (norm_points ** 2).sum(dim=1, keepdims=True).sqrt()

        points_min, points_max = norm_points.min(axis=0, keepdims=True)[0], norm_points.max(axis=0, keepdims=True)[0]
        colors = (norm_points - points_min) / (points_max - points_min)

    return colors


def plot_transport(points1, points2, transport, savefile=None, normalize=True, **kwargs):
    """
    Visualize a transport plan as lines connecting matched points.

    Displays two point clouds with lines connecting them, where line thickness represents transport mass.

    Args:
        points1: Source point cloud, shape (N, D)
        points2: Target point cloud, shape (M, D)
        transport: Transport plan/coupling matrix, shape (N, M)
                  where transport[i,j] = mass transported from i to j
        savefile: Path to save figure. If None, doesn't save
        normalize: If True, normalize line weights to [0, 1]
        **kwargs: Additional arguments passed to plot_pointcloud

    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    ax = plot_pointcloud(points1, ax=ax, c="green", **kwargs)
    ax = plot_pointcloud(points2, ax=ax, c="orange", **kwargs)

    matchings = np.argwhere(transport > (1e-3 * transport.max()))

    weights = transport[matchings[:, 0], matchings[:, 1]]
    if normalize:
        weights = weights / np.max(weights)

    edges = np.transpose([points1[matchings[:, 0]], points2[matchings[:, 1]]], axes=(1, 0, 2))
    lines = LineCollection(edges, alpha=.5, color='black', lw=weights)
    ax.add_collection(lines)

    ax.set_aspect('equal')
    plt.tight_layout()

    if savefile is not None: plt.savefig(savefile)
    plt.show()


def plot_transfer(points1, points2, transport, source_dim=0, lazy=False, colors_source=None, savefile=None, show=True,
                  **kwargs):
    """
    Visualize color transfer via optimal transport.

    Shows how colors from source points are transported to target points through the transport plan. Creates
    side-by-side comparison showing source colors and their projection onto the target.

    Args:
        points1: Source point cloud, shape (N, D1)
        points2: Target point cloud, shape (M, D2)
        transport: Transport plan, shape (N, M) or (M, N) depending on source_dim
        source_dim: Dimension to transport from (0 for row-wise, 1 for column-wise)
        lazy: If True, use LazyTensor implementation
        colors_source: Source colors, shape (N, 3). If None, uses position-based colors
        savefile: Path to save figure
        show: If True, display the figure
        **kwargs: Additional arguments passed to plot_pointcloud
    """
    D1, D2 = points1.shape[1], points2.shape[1]

    if colors_source is None:
        colors_source = _default_colors(points1)

    transfered_color = barycentric_projection(colors_source, transport, lazy=lazy, source_dim=source_dim)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    if D1 == 3:
        ss = axs[0].get_subplotspec()
        axs[0].remove()
        axs[0] = fig.add_subplot(ss, projection='3d')
    if D2 == 3:
        ss = axs[1].get_subplotspec()
        axs[1].remove()
        axs[1] = fig.add_subplot(ss, projection='3d')

    axs[0] = plot_pointcloud(points1.cpu().numpy(), ax=axs[0], c=colors_source.cpu().numpy(), **kwargs)
    axs[1] = plot_pointcloud(points2.cpu().numpy(), ax=axs[1], c=transfered_color.cpu().numpy(), **kwargs)
    plt.tight_layout()

    if savefile is not None: plt.savefig(savefile)
    if show: plt.show()

    plt.close()


def plot_mesh_transfer(mesh1, mesh2, transport, source_dim=0, lazy=False, pl_source=None, pl_target=None,
                       colors_source=None, uv_source=None, texture_path=None, **kwargs):
    """
        Visualize color or texture transfer between meshes via optimal transport.

        Transfers vertex attributes (colors or UV coordinates) from source mesh to
        target mesh according to the transport plan. Useful for texture transfer,
        correspondence visualization, and evaluating mesh alignment quality.

        Args:
            mesh1: Source mesh dictionary with 'pos' and 'face'
            mesh2: Target mesh dictionary with 'pos' and 'face'
            transport: Transport plan between vertices, shape (N, M)
            source_dim: Transport direction (0 for source→target, 1 for target→source)
            lazy: If True, use LazyTensor implementation
            pl_source: PyVista Plotter for source. If None, creates new plotter
            pl_target: PyVista Plotter for target. If None, creates new plotter
            colors_source: Source vertex colors, shape (N, 3)
            uv_source: Source UV coordinates, shape (N, 2)
            texture_path: Path to texture image. Required if using uv_source
            **kwargs: Additional arguments passed to plot_mesh

        Returns:
            tuple: (pl_source, pl_target) - PyVista plotters for both meshes
        """
    points1, points2 = mesh1['pos'], mesh2['pos']

    transfered_colors, transfered_uv = None, None
    if colors_source is not None:
        transfered_colors = barycentric_projection(colors_source, transport, lazy=lazy, source_dim=source_dim)
    elif uv_source is not None:
        transfered_uv = barycentric_projection(uv_source, transport, lazy=lazy, source_dim=source_dim)
    else:
        colors_source = (points1 - points1.min(axis=0, keepdims=True)[0]) / (
                points1.max(axis=0, keepdims=True)[0] - points1.min(axis=0, keepdims=True)[0])
        transfered_colors = barycentric_projection(colors_source, transport, lazy=lazy)

    pl_source = plot_mesh(mesh1, color=colors_source, uv=uv_source, texture_path=texture_path, pl=pl_source, **kwargs)
    pl_target = plot_mesh(mesh2, color=transfered_colors, uv=transfered_uv, texture_path=texture_path, pl=pl_target,
                          **kwargs)

    return pl_source, pl_target
