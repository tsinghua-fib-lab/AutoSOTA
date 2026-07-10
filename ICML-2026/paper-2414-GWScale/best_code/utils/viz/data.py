"""
Data Visualization Utilities

Functions for visualizing point clouds and meshes in 2D and 3D using
matplotlib and PyVista.
"""
import numpy as np
import pyvista as pv
from matplotlib import pyplot as plt


def plot_pointcloud(points, ax=None, c='black', savefile=None, show=None, **kwargs):
    """
    Plot a point cloud in 1D, 2D, or 3D.

    Automatically detects dimensionality and creates appropriate visualization. Can plot on existing axes or create new
    figure.

    Args:
        points: Point cloud tensor, shape (N, D) where D is in {1, 2, 3}
        ax: Matplotlib axes to plot on. If None, creates new figure
        c: Color specification
        savefile: Path to save figure. If None, doesn't save
        **kwargs: Additional arguments passed to ax.scatter()

    Returns:
        matplotlib.axes.Axes: Axes object containing the plot

    Behavior by Dimension:
        - D=1: Plots points on horizontal line at y=1
        - D=2: Standard 2D scatter plot
        - D=3: 3D scatter plot with projection='3d'

    Note:
        The figure is shown only if ax=None. When providing ax, caller controls when to show the figure.
    """
    D = points.shape[1]
    if show is None:
        show = (ax is None)

    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d') if D == 3 else fig.add_subplot(111)

    if D == 1:
        ax.scatter(points[:, 0], np.ones(points.shape[0]), c=c, **kwargs)
    elif D == 2:
        ax.scatter(points[:, 0], points[:, 1], c=c, **kwargs)
    elif D == 3:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=c, **kwargs)

    ax.set_aspect('equal')

    plt.tight_layout()

    if savefile is not None:
        plt.savefig(savefile, bbox_inches='tight', dpi=300)

    if show:
        plt.show()

    return ax


def plot_mesh(mesh, color=None, uv=None, texture_path=None, pl=None, **kwargs):
    """
    Render a 3D triangular mesh using PyVista.

    Supports three visualization modes:
        1. Solid color or per-vertex coloring
        2. Texture mapping with UV coordinates
        3. Default material appearance

    Args:
        mesh: Dictionary containing:
            - 'pos': Vertex positions, shape (M, 3)
            - 'face': Triangle faces, shape (F, 3)
        color: Per-vertex RGB colors, shape (M, 3) with values in [0, 1].
               If provided, applies vertex coloring
        uv: Texture coordinates, shape (M, 2). Required if using texture_path
        texture_path: Path to texture image file. Requires uv coordinates
        pl: PyVista Plotter object. If None, creates new plotter
        **kwargs: Additional arguments passed to pl.add_mesh()

    Returns:
        pv.Plotter: PyVista plotter object containing the mesh

    Note:
        The figure is shown only if pl=None. When providing pl, caller controls when to show the figure.
    """
    show = (pl is None)
    if pl is None:  pl = pv.Plotter(border=True, border_width=5)

    pos_np = mesh['pos'].cpu().numpy()  # Size (M, D)
    face_np = mesh['face'].cpu().numpy()  # Size (Q, 3)

    pv_mesh = pv.PolyData(pos_np, np.hstack(np.c_[np.full(face_np.shape[0], 3), face_np]))

    if pl is None: pl = pv.Plotter(border=False, window_size=(2000, 2000))

    if color is not None:
        pl.add_mesh(pv_mesh, scalars=color.cpu().numpy(), rgb=True, **kwargs)
    elif uv is not None and texture_path is not None:
        pv_mesh.active_texture_coordinates = uv.cpu().numpy()
        pl.add_mesh(pv_mesh, texture=pv.read_texture(texture_path), **kwargs)
    else:
        pl.add_mesh(pv_mesh, **kwargs)

    if show:
        pl.show()
        pl.close()
        pl.deep_clean()

    return pl
