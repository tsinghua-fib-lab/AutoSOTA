import matplotlib.pyplot as plt
import torch
import numpy as np
from matplotlib.collections import LineCollection


def plot_gradients(X, grad, ax=None, savefile=None, **kwargs):
    show = (ax is None)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    ax.quiver(X[:, 0], X[:, 1], grad[:, 0], grad[:, 1], **kwargs)

    ax.set_aspect('equal')

    plt.tight_layout()

    if savefile is not None:
        plt.savefig(savefile, bbox_inches='tight')

    if show:
        plt.show()

    return ax


def plot_flow(X_list, X_0, ax=None, show = None, savefile=None):
    if show is None:
        show = (ax is None)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    edges = torch.stack(X_list, dim=0).cpu().numpy().transpose((1, 0, 2))
    t_index = np.zeros(edges.shape[:2])
    t_index[:, np.arange(len(X_list))] = np.arange(len(X_list)) / len(X_list)

    edges_flat = np.zeros((*edges.shape, 2))
    edges_flat[:, 1:] = np.stack([edges[:, :-1], edges[:, 1:]], axis=2)
    edges_flat[:, 0] = np.stack([X_0, edges[:, 0]], axis=1)
    edges_flat = edges_flat.reshape(-1, 2, 2)

    lines = LineCollection(edges_flat, alpha=0.5, lw=0.5, colors='lightgray')
    ax.add_collection(lines)
    ax.scatter(edges[:, :, 0].flatten(), edges[:, :, 1].flatten(), marker='o', c='black', s=4,
               alpha=0.2 + 0.3 * np.arange(len(X_list)) / len(X_list))

    ax.scatter(X_0[:, 0], X_0[:, 1], color='lightgray', s=10, alpha=0.3)

    X_f = X_list[-1]
    ax.scatter(X_f[:, 0], X_f[:, 1], color='black', s=40,
               edgecolors='black', linewidths=0.2, alpha=1, zorder=3)

    ax.set(aspect='equal')
    plt.tight_layout()

    if savefile is not None:
        plt.savefig(savefile, bbox_inches='tight', dpi=300)

    if show:
        plt.show()

    return ax
