import warnings

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from utils.helpers import pca_reduce

###  DKL-LVMOGP  ###

@torch.no_grad()
def visualize_embeddings(model, data_dict, save_path=None, figsize=(10, 8)):
    """
    Visualize transformed inputs and inducing points in the embedding space.
    
    Args:
        model: The DKL-LVMOGP model with fnet, qH, and Z attributes
        data_dict: Dictionary containing 'all_X', 'train_mask', 'test_mask'
        save_path: Optional path to save the figure (e.g., 'embeddings.pdf')
        figsize: Tuple specifying figure size (width, height)
        
    Returns:
        matplotlib.figure.Figure: The created figure object
    """
    # Prepare
    _X = data_dict["all_X"].unsqueeze(-2)  # [..., N_all, 1, D_X]
    train_mask = data_dict["train_mask"]  # [..., N_all, P]
    test_mask = data_dict["test_mask"]  # [..., N_all, P]
    _H = model.qH.mean_qH.unsqueeze(-3)  # [..., 1, P, D_H]
    N_data, P = _X.size(-3), _H.size(-2)
    X_exp = _X.expand(*_X.shape[:-2], P, *_X.shape[-1:])  # [..., N_all, P, D_X]
    H_exp = _H.expand(*_H.shape[:-3], N_data, *_H.shape[-2:])  # [..., N_all, P, D_H]
    XH_concat = torch.cat([X_exp, H_exp], dim=-1)  # [..., N_all, P, D_X + D_H]

    # Feed forward through neural net
    X_trans = model.fnet(XH_concat)  # transformed inputs, [..., N_all, P, D_T]
    D_T = X_trans.size(-1)
    assert D_T >= 2, "Embedding dimension D_T must be at least 2 for visualization."

    inducing_points = model.Z.inducing_points  # [..., M, D_T]

    def ensure_2d(X):
        # X: [..., *, D_T] -> [..., *, 2]
        if D_T > 2:
            # perform PCA for last two dims
            return pca_reduce(X, k=2)
        return X

    _X_trans_2d = ensure_2d(
        X_trans.view(*X_trans.shape[:-3], N_data*P, D_T)
    )  # [..., N_all*P, D_T] -> [..., N_all*P, 2]
    X_trans_2d = _X_trans_2d.view(*X_trans.shape[:-3], N_data, P, 2)  # [..., N_all, P, 2]
    Z_2d = ensure_2d(inducing_points)  # [..., M, 2]

    def first_slice_3d(t):
        if t.ndim == 3:  # [N_all, P, 2]
            return t
        warnings.warn(f"Only the first 3D slice of original {t.ndim}-dimensional tensors will be plotted.")
        idx = (0,) * (t.ndim - 3)
        return t[idx]  # [N_all, P, 2]

    def first_slice_masks(m):
        if m.ndim == 2:
            return m
        warnings.warn(f"Only the first 2D slice of original {m.ndim}-dimensional masks will be used.")
        idx = (0,) * (m.ndim - 2)
        return m[idx]  # [N_all, P]

    X_plot = first_slice_3d(X_trans_2d).cpu().numpy()  # [N_all, P, 2]
    Z_plot = (Z_2d if Z_2d.ndim == 2 else Z_2d[(0,) * (Z_2d.ndim-2)]).cpu().numpy()  # [M, 2]
    tr_mask = first_slice_masks(train_mask).cpu().numpy()  # [N_all, P]
    te_mask = first_slice_masks(test_mask).cpu().numpy()  # [N_all, P]

    # ---- build palettes: same hue per output, lighter tint for test ----
    base_cmap = get_cmap('tab10' if P <= 10 else 'tab20')

    def lighten(rgb, factor=0.5):
        # simple linear mix toward white: factor in (0,1), higher -> lighter
        r, g, b, *rest = rgb
        w = 1.0
        return (r + (w - r) * factor, g + (w - g) * factor, b + (w - b) * factor)

    colours_train = [base_cmap(i % base_cmap.N) for i in range(P)]
    colours_test = [lighten(c, 0.55) for c in colours_train]

    # ---- Plotting ----
    plt.figure(figsize=figsize)

    # per-output scatter; train: circles, test: triangles
    for p in range(P):
        Xp = X_plot[:, p, :]  # [N_all, 2]

        # training
        if tr_mask[:, p].any():
            pts = Xp[tr_mask[:, p]]
            plt.scatter(pts[:, 0], pts[:, 1],
                        s=22, marker='o', alpha=0.8,
                        edgecolors='none',
                        label=None, c=[colours_train[p]])

        # test
        if te_mask[:, p].any():
            pts = Xp[te_mask[:, p]]
            plt.scatter(pts[:, 0], pts[:, 1],
                        s=22, marker='^', alpha=0.9,
                        edgecolors='black', linewidths=0.3,
                        label=None, c=[colours_test[p]])

    # Plot inducing points
    plt.scatter(
        Z_plot[:, 0], Z_plot[:, 1], c='none', alpha=0.95, s=90, marker='X',
        edgecolors='black', linewidth=1, label='Inducing Points',
    )
    
    # Formatting
    plt.xlabel('First Principal Component' if D_T > 2 else 'Dimension 1')
    plt.ylabel('Second Principal Component' if D_T > 2 else 'Dimension 2')
    plt.title('Embedding Space Visualization\n(Training, Test, and Inducing Points)')

    # legends: (1) split legend; (2) outputs legend
    split_handles = [
        Line2D([0], [0], marker='o', linestyle='None', markersize=7,
               markerfacecolor='gray', markeredgecolor='none', label='Train'),
        Line2D([0], [0], marker='^', linestyle='None', markersize=7,
               markerfacecolor='white', markeredgecolor='black', label='Test'),
        Line2D([0], [0], marker='X', linestyle='None', markersize=8,
               markerfacecolor='white', markeredgecolor='black', label='Inducing')
    ]
    first_legend = plt.legend(handles=split_handles, loc='upper right', frameon=True, title='Split')
    plt.gca().add_artist(first_legend)

    output_handles = [Patch(facecolor=colours_train[p], edgecolor='none', label=f'Output {p}')
                      for p in range(P)]
    plt.legend(handles=output_handles, loc='lower left', frameon=True, title='Outputs')

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save if path provided
    if save_path is not None:
        plt.savefig(save_path, dpi=2000, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.close()
