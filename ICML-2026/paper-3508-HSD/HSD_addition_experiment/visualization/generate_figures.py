#!/usr/bin/env python3
"""
Generate key visualizations for the addition experiments:
  1. 3D vector field comparison (HSD vs HAMLET on ellipsoid & torus)
  2. Level set contours showing topological fidelity
  3. Hodge component energy distribution

Usage:
    python visualization/generate_figures.py
"""
import os, sys, pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'hodge-spectral-operator'))

from hodge_spectral.operators import HighOrderSpectralOperators
from hodge_spectral.models.unified import UnifiedHSD, train_unified_hsd
from hodge_spectral.utils import LpLoss
from baselines.recent_baselines import HAMLET, train_recent_baseline

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)


class DS(Dataset):
    def __init__(s, c0, c1, c2, xr, yr):
        s.c0, s.c1, s.c2, s.xr, s.yr = c0, c1, c2, xr, yr
    def __len__(s): return len(s.c0)
    def __getitem__(s, i):
        return (torch.from_numpy(s.c0[i]), torch.from_numpy(s.c1[i]),
                torch.from_numpy(s.c2[i]), torch.zeros(1),
                torch.from_numpy(s.xr[i]), torch.from_numpy(s.yr[i]))


def load_and_train(dataset_name):
    """Load data, train HSD + HAMLET, return predictions on a test sample."""
    path = f'data/{dataset_name}/{dataset_name}_dataset.pkl'
    with open(path, 'rb') as f:
        d = pickle.load(f)
    pts = d['points'].astype(np.float64)
    faces = d['faces'].astype(np.int64)
    X, Y = d['X_data'], d['Y_data']
    n, k = len(pts), 64

    host = HighOrderSpectralOperators(pts, faces, k_list=(k, k, k))
    Phi0, Phi1 = host.Phi0[:, :k], host.Phi1[:, :k]

    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=42)
    Xtr, Xva, Ytr, Yva = train_test_split(Xtr, Ytr, test_size=0.15, random_state=42)
    xs = np.max(np.abs(Xtr)) + 1e-9
    ys = np.max(np.abs(Ytr)) + 1e-9

    def proc(Xn, Yn):
        x2 = (Xn / xs).astype(np.float32)
        y2 = (Yn / ys).astype(np.float32)
        fX, gX, _ = host.lift_signal(x2)
        return ((fX @ Phi0).astype(np.float32), (gX @ Phi1).astype(np.float32),
                np.zeros((len(x2), k), dtype=np.float32), x2, y2)

    tr, va, te = proc(Xtr, Ytr), proc(Xva, Yva), proc(Xte, Yte)

    # Train HSD
    Phi_out = torch.from_numpy(Phi0.astype(np.float32)).to(DEVICE)
    hsd = UnifiedHSD(host, 1, k, k, k, n, k, Phi_out,
                     hidden_dims=(256, 192), res_hidden=128).to(DEVICE)
    hsd = train_unified_hsd(hsd, DataLoader(DS(*tr), batch_size=64, shuffle=True),
                            DataLoader(DS(*va), batch_size=64), 120, DEVICE, lr=3e-3, patience=25)

    # Train HAMLET
    pts_np = pts.astype(np.float32)
    xt = torch.from_numpy((Xtr / xs).astype(np.float32)).to(DEVICE)
    yt = torch.from_numpy((Ytr / ys).astype(np.float32)).to(DEVICE)
    xv = torch.from_numpy((Xva / xs).astype(np.float32)).to(DEVICE)
    yv = torch.from_numpy((Yva / ys).astype(np.float32)).to(DEVICE)
    hamlet = HAMLET(n, 3, 96, 4, 3).to(DEVICE)
    hamlet, _, _ = train_recent_baseline(hamlet, xt, yt, xv, yv, pts_np, 100,
                                          {'lr': 1e-3, 'epochs': 100, 'batch_size': 64}, DEVICE)

    # Predict on test sample
    idx = 3
    c0, c1, c2, xr, yr = [torch.from_numpy(te[j][idx:idx+1]).to(DEVICE) for j in range(5)]
    hsd.eval(); hamlet.eval()
    with torch.no_grad():
        pred_hsd = hsd(c0, c1, c2, xr)[0].cpu().numpy()[0]
        xe_t = torch.from_numpy((Xte[idx:idx+1] / xs).astype(np.float32)).to(DEVICE)
        pts_t = torch.from_numpy(pts_np).to(DEVICE)
        pred_hamlet = hamlet(xe_t, pts_t).cpu().numpy()[0]

    gt = te[4][idx]  # (N, 3)
    return pts.astype(np.float32), faces, gt, pred_hsd, pred_hamlet, dataset_name


def plot_3d_vector_field(pts, faces, gt, hsd_pred, hamlet_pred, name, out_path):
    """3D surface with quiver arrows showing vector field."""
    fig = plt.figure(figsize=(16, 5))

    gt_mag = np.linalg.norm(gt, axis=-1)
    hsd_mag = np.linalg.norm(hsd_pred, axis=-1)
    hamlet_mag = np.linalg.norm(hamlet_pred, axis=-1)
    vmax = gt_mag.max()

    # Subsample for quiver clarity
    step = max(1, len(pts) // 300)
    sub = np.arange(0, len(pts), step)

    for i, (vecs, title, mag) in enumerate([
        (gt, 'Ground Truth', gt_mag),
        (hsd_pred, 'HSD (ours)', hsd_mag),
        (hamlet_pred, 'HAMLET', hamlet_mag),
    ]):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')
        # Surface colored by magnitude
        ax.plot_trisurf(pts[:, 0], pts[:, 1], pts[:, 2], triangles=faces,
                        alpha=0.3, color='lightgray', edgecolor='none')
        # Quiver
        scale = 0.15
        ax.quiver(pts[sub, 0], pts[sub, 1], pts[sub, 2],
                  vecs[sub, 0] * scale, vecs[sub, 1] * scale, vecs[sub, 2] * scale,
                  color=plt.cm.RdYlBu_r(mag[sub] / (vmax + 1e-9)),
                  arrow_length_ratio=0.3, linewidth=0.8)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlim(pts[:, 0].min(), pts[:, 0].max())
        ax.set_ylim(pts[:, 1].min(), pts[:, 1].max())
        ax.set_zlim(pts[:, 2].min(), pts[:, 2].max())
        ax.set_axis_off()
        # Set consistent viewpoint
        if 'torus' in name:
            ax.view_init(elev=25, azim=45)
        else:
            ax.view_init(elev=20, azim=30)

    short = 'Ellipsoid' if 'ellipsoid' in name else 'Torus'
    fig.suptitle(f'{short}: Vector Field Comparison', fontsize=13, fontweight='bold',
                 x=0.02, ha='left', y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


def plot_magnitude_levelset(pts, faces, gt, hsd_pred, hamlet_pred, name, out_path):
    """2D level set contours of velocity magnitude (projected)."""
    gt_mag = np.linalg.norm(gt, axis=-1)
    hsd_mag = np.linalg.norm(hsd_pred, axis=-1)
    hamlet_mag = np.linalg.norm(hsd_pred, axis=-1)
    hsd_err = np.abs(hsd_mag - gt_mag)
    hamlet_err = np.abs(np.linalg.norm(hamlet_pred, axis=-1) - gt_mag)

    # Project to 2D
    if 'torus' in name:
        # Torus: use (atan2(z, sqrt(x²+y²)-R), atan2(y,x))
        R = 1.0
        r_xy = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        u = np.arctan2(pts[:, 1], pts[:, 0])
        v = np.arctan2(pts[:, 2], r_xy - R)
        xlabel, ylabel = 'Toroidal angle', 'Poloidal angle'
    else:
        # Ellipsoid: spherical projection
        r = np.linalg.norm(pts, axis=1)
        u = np.arctan2(pts[:, 1], pts[:, 0])
        v = np.arccos(np.clip(pts[:, 2] / (r + 1e-9), -1, 1))
        xlabel, ylabel = 'θ', 'φ'

    from matplotlib.tri import Triangulation
    tri = Triangulation(u, v)
    # Remove wrap-around triangles
    x_tri = u[tri.triangles]
    valid = np.all(np.abs(np.diff(x_tri, axis=1)) < 2.0, axis=1)
    tri = Triangulation(u, v, tri.triangles[valid])

    vmax = gt_mag.max()
    levels = np.linspace(0, vmax * 0.95, 15)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top row: GT and HSD
    for ax, data, title in [(axes[0, 0], gt_mag, 'Ground Truth'),
                             (axes[0, 1], np.linalg.norm(hsd_pred, axis=-1), 'HSD (ours)')]:
        cs = ax.tricontourf(tri, data, levels=levels, cmap='RdYlBu_r', extend='both')
        ax.tricontour(tri, data, levels=levels, colors='k', linewidths=0.3, alpha=0.4)
        ax.set_title(title, fontsize=11, fontweight='bold', loc='left')
        ax.set_aspect('equal')
        cax = ax.inset_axes([0.6, 0.88, 0.37, 0.04])
        plt.colorbar(cs, cax=cax, orientation='horizontal')
        cax.tick_params(labelsize=6)

    # Bottom left: HAMLET
    ax = axes[1, 0]
    hamlet_data = np.linalg.norm(hamlet_pred, axis=-1)
    cs = ax.tricontourf(tri, hamlet_data, levels=levels, cmap='RdYlBu_r', extend='both')
    ax.tricontour(tri, hamlet_data, levels=levels, colors='k', linewidths=0.3, alpha=0.4)
    ax.set_title('HAMLET', fontsize=11, fontweight='bold', loc='left')
    ax.set_aspect('equal')
    cax = ax.inset_axes([0.6, 0.88, 0.37, 0.04])
    plt.colorbar(cs, cax=cax, orientation='horizontal')
    cax.tick_params(labelsize=6)

    # Bottom right: Error comparison
    ax = axes[1, 1]
    err_max = max(hsd_err.max(), hamlet_err.max())
    err_levels = np.linspace(0, err_max * 0.7, 10)
    cs = ax.tricontourf(tri, hamlet_err, levels=err_levels, cmap='Reds', extend='both')
    ax.tricontour(tri, hsd_err, levels=err_levels, colors='blue', linewidths=0.8, alpha=0.6)
    ax.set_title('Error: HAMLET (fill) vs HSD (blue contours)', fontsize=10,
                 fontweight='bold', loc='left')
    ax.set_aspect('equal')
    cax = ax.inset_axes([0.6, 0.88, 0.37, 0.04])
    plt.colorbar(cs, cax=cax, orientation='horizontal')
    cax.tick_params(labelsize=6)

    short = 'Ellipsoid' if 'ellipsoid' in name else 'Torus'
    fig.suptitle(f'{short}: Level Set Contours & Error Topology',
                 fontsize=13, fontweight='bold', x=0.02, ha='left', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


def main():
    print("=" * 50)
    print("Generating Visualizations")
    print("=" * 50)

    for ds_name in ['ellipsoid_aero', 'torus_helmholtz']:
        short = 'ellipsoid' if 'ellipsoid' in ds_name else 'torus'
        print(f"\n--- {ds_name} ---")
        pts, faces, gt, hsd_pred, hamlet_pred, name = load_and_train(ds_name)

        plot_3d_vector_field(pts, faces, gt, hsd_pred, hamlet_pred, name,
                            os.path.join(OUT_DIR, f'vector_field_3d_{short}.png'))

        plot_magnitude_levelset(pts, faces, gt, hsd_pred, hamlet_pred, name,
                               os.path.join(OUT_DIR, f'levelset_{short}.png'))


if __name__ == "__main__":
    main()
