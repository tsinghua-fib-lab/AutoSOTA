#!/usr/bin/env python3
"""
TDA Task: Betti Number Prediction from HSD Spectral Features

A genuine supervised TDA task:
  - Generate point clouds from shapes with known topology
  - Build Vietoris-Rips filtration at multiple scales
  - At each scale, compute HSD spectral features (L₁ eigenvalues, Md₀ structure)
  - Train a small MLP to predict b₁ from spectral features
  - Evaluate on UNSEEN point clouds (different samples, same shape classes)

This demonstrates HSD as a spectral feature extractor for topological inference.

Usage:
    python tda_demo/hsd_tda_classifier.py
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import os


# ==============================
# Point cloud generators
# ==============================
def sample_circle(n=40, noise=0.05):
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    pts = np.stack([np.cos(t), np.sin(t)], axis=1)
    pts += noise * np.random.randn(*pts.shape)
    return pts  # b1=1

def sample_figure8(n=40, noise=0.05):
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    pts = np.stack([np.sin(t), np.sin(t)*np.cos(t)], axis=1)
    pts += noise * np.random.randn(*pts.shape)
    return pts  # b1=2

def sample_line(n=40, noise=0.05):
    t = np.linspace(-1, 1, n)
    pts = np.stack([t, np.zeros(n)], axis=1)
    pts += noise * np.random.randn(*pts.shape)
    return pts  # b1=0

def sample_annulus(n=40, noise=0.05):
    t = np.linspace(0, 2*np.pi, n//2, endpoint=False)
    inner = 0.5 * np.stack([np.cos(t), np.sin(t)], axis=1)
    outer = 1.0 * np.stack([np.cos(t), np.sin(t)], axis=1)
    pts = np.vstack([inner, outer])
    pts += noise * np.random.randn(*pts.shape)
    return pts  # b1=1


# ==============================
# Simplicial complex + HSD spectral features
# ==============================
def build_vr_complex(pts, epsilon):
    n = len(pts)
    dists = np.linalg.norm(pts[:, None] - pts[None, :], axis=-1)
    edges = [(i, j) for i in range(n) for j in range(i+1, n) if dists[i, j] < epsilon]

    triangles = []
    if len(edges) < 1500:
        adj = {i: set() for i in range(n)}
        for i, j in edges:
            adj[i].add(j); adj[j].add(i)
        for i in range(n):
            nb = sorted(j for j in adj[i] if j > i)
            for a, j in enumerate(nb):
                for k in nb[a+1:]:
                    if k in adj[j]:
                        triangles.append((i, j, k))
                        if len(triangles) > 10000:
                            return edges, triangles
    return edges, triangles


def build_boundary_ops(n_verts, edges, triangles):
    if not edges:
        return csr_matrix((0, n_verts)), csr_matrix((0, max(1, 0)))
    ne = len(edges)
    r, c, v = [], [], []
    for idx, (i, j) in enumerate(edges):
        r.extend([idx, idx]); c.extend([i, j]); v.extend([-1., 1.])
    B1 = csr_matrix((v, (r, c)), shape=(ne, n_verts))

    if not triangles:
        return B1, csr_matrix((0, ne))
    e2i = {(min(e), max(e)): idx for idx, e in enumerate(edges)}
    r, c, v = [], [], []
    for idx, (i, j, k) in enumerate(triangles):
        for (a, b), s in [((j,k), 1.), ((i,k), -1.), ((i,j), 1.)]:
            key = (min(a,b), max(a,b))
            if key in e2i:
                r.append(idx); c.append(e2i[key]); v.append(s)
    B2 = csr_matrix((v, (r, c)), shape=(len(triangles), ne))
    return B1, B2


def compute_hsd_spectral_features(pts, epsilon, n_eigs=10):
    """
    Compute HSD spectral features at a given filtration scale.
    Returns: feature vector (spectral signature of the Hodge Laplacians).
    """
    n = len(pts)
    edges, tris = build_vr_complex(pts, epsilon)
    ne, nt = len(edges), len(tris)

    if ne < 2:
        return np.zeros(n_eigs * 2 + 4), 0  # not enough structure

    B1, B2 = build_boundary_ops(n, edges, tris)

    # Hodge Laplacians
    L0 = (B1.T @ B1).tocsr()
    L1 = (B1 @ B1.T).tocsr()
    if nt > 0:
        L1 = L1 + (B2.T @ B2).tocsr()

    # Spectral features from L0
    n0 = L0.shape[0]
    k0 = min(n_eigs, n0 - 1)
    if n0 < 200:
        eigs0 = np.sort(np.maximum(np.linalg.eigvalsh(L0.toarray()), 0))[:n_eigs]
    else:
        from scipy.sparse.linalg import eigsh
        try:
            eigs0 = np.sort(np.maximum(eigsh(L0, k=k0, which='SM', tol=1e-6,
                                              return_eigenvectors=False), 0))[:n_eigs]
        except:
            eigs0 = np.zeros(n_eigs)

    # Spectral features from L1
    n1 = L1.shape[0]
    k1 = min(n_eigs, n1 - 1)
    if k1 < 1:
        eigs1 = np.zeros(n_eigs)
    elif n1 < 200:
        eigs1 = np.sort(np.maximum(np.linalg.eigvalsh(L1.toarray()), 0))[:n_eigs]
    else:
        from scipy.sparse.linalg import eigsh
        try:
            eigs1 = np.sort(np.maximum(eigsh(L1, k=k1, which='SM', tol=1e-6,
                                              return_eigenvectors=False), 0))[:n_eigs]
        except:
            eigs1 = np.zeros(n_eigs)

    # Pad to fixed size
    eigs0_pad = np.zeros(n_eigs); eigs0_pad[:len(eigs0)] = eigs0[:n_eigs]
    eigs1_pad = np.zeros(n_eigs); eigs1_pad[:len(eigs1)] = eigs1[:n_eigs]

    # Spectral de Rham coupling: ||Md0|| norm
    # Md0 = Phi1^T B1 Phi0 (approximate with truncated eigenvectors)
    md0_norm = 0.0
    if ne > 2 and n0 > 2 and k0 > 0 and k1 > 0:
        try:
            if n0 < 200:
                _, Phi0 = np.linalg.eigh(L0.toarray())
                Phi0 = Phi0[:, :min(10, n0)]
            else:
                from scipy.sparse.linalg import eigsh
                _, Phi0 = eigsh(L0, k=min(10, n0-1), which='SM', tol=1e-6)

            if n1 < 200:
                _, Phi1 = np.linalg.eigh(L1.toarray())
                Phi1 = Phi1[:, :min(10, n1)]
            else:
                _, Phi1 = eigsh(L1, k=min(10, n1-1), which='SM', tol=1e-6)

            Md0 = Phi1.T @ B1.toarray() @ Phi0
            md0_norm = np.linalg.norm(Md0)
        except:
            md0_norm = 0.0

    # Feature vector: [L0_eigs, L1_eigs, ne, nt, epsilon, md0_norm]
    features = np.concatenate([eigs0_pad, eigs1_pad, [ne, nt, epsilon, md0_norm]])

    # True b1 from threshold
    b1 = int(np.sum(eigs1_pad < 1e-6))
    return features, b1


# ==============================
# Dataset generation
# ==============================
def generate_dataset(n_samples_per_class=50, n_scales=8):
    """Generate (features, b1_true) pairs from multiple shapes at multiple scales."""
    classes = [
        ('line', sample_line, 0),
        ('circle', sample_circle, 1),
        ('annulus', sample_annulus, 1),
        ('figure8', sample_figure8, 2),
    ]

    all_features, all_labels = [], []
    all_meta = []

    for class_name, sampler, expected_b1 in classes:
        for i in range(n_samples_per_class):
            pts = sampler(n=35 + np.random.randint(10), noise=0.03 + 0.04*np.random.rand())
            epsilons = np.linspace(0.2, 0.7, n_scales)

            for eps in epsilons:
                feat, b1 = compute_hsd_spectral_features(pts, eps)
                all_features.append(feat)
                all_labels.append(b1)
                all_meta.append({'class': class_name, 'eps': eps, 'expected_b1': expected_b1})

    return np.array(all_features, dtype=np.float32), np.array(all_labels, dtype=np.int64), all_meta


# ==============================
# Betti number predictor (small MLP)
# ==============================
class BettiPredictor(nn.Module):
    def __init__(self, in_dim, n_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ==============================
# Baselines
# ==============================
def compute_geometry_features(pts, epsilon, n_feat=24):
    """Simple geometry features (no topology): pairwise distance statistics."""
    dists = np.linalg.norm(pts[:, None] - pts[None, :], axis=-1)
    mask = dists < epsilon
    degrees = mask.sum(axis=1) - 1  # neighbor count

    feats = [
        np.mean(dists), np.std(dists), np.median(dists),
        np.min(dists[dists > 0]), np.max(dists),
        np.mean(degrees), np.std(degrees), np.max(degrees),
        len(pts), epsilon,
        np.percentile(dists, 25), np.percentile(dists, 75),
    ]
    # Pad to n_feat
    feats = feats[:n_feat] + [0.0] * max(0, n_feat - len(feats))
    return np.array(feats, dtype=np.float32)


# ==============================
# Main
# ==============================
def main():
    np.random.seed(42)
    torch.manual_seed(42)
    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("TDA Task: Betti Number Prediction via HSD Spectral Features")
    print("=" * 60)

    # Generate train and test data (different random samples)
    print("\n--- Generating training data ---")
    X_train, y_train, meta_train = generate_dataset(n_samples_per_class=40, n_scales=6)
    print(f"  Train: {len(X_train)} samples, feature dim = {X_train.shape[1]}")
    print(f"  Label distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")

    print("\n--- Generating test data ---")
    np.random.seed(123)  # different seed for test
    X_test, y_test, meta_test = generate_dataset(n_samples_per_class=20, n_scales=6)
    print(f"  Test: {len(X_test)} samples")

    # Standardize features
    mu, std = X_train.mean(0), X_train.std(0) + 1e-10
    X_train_n = (X_train - mu) / std
    X_test_n = (X_test - mu) / std

    # --- Train HSD spectral feature predictor ---
    print("\n--- Training Betti predictor (HSD spectral features) ---")
    n_classes = int(y_train.max()) + 1
    model = BettiPredictor(X_train_n.shape[1], n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    train_ds = TensorDataset(torch.from_numpy(X_train_n), torch.from_numpy(y_train))
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)

    for ep in range(100):
        model.train()
        for xb, yb in train_dl:
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        pred_train = model(torch.from_numpy(X_train_n)).argmax(1).numpy()
        pred_test = model(torch.from_numpy(X_test_n)).argmax(1).numpy()
    acc_train = (pred_train == y_train).mean()
    acc_test = (pred_test == y_test).mean()
    print(f"  Train accuracy: {acc_train:.1%}")
    print(f"  Test accuracy:  {acc_test:.1%}")

    # --- Baseline: geometry features ---
    print("\n--- Baseline: geometry features (no topology) ---")
    # Regenerate geometry features
    np.random.seed(42)
    geo_train, geo_test = [], []
    for meta in meta_train:
        pts = (sample_line if meta['class'] == 'line' else
               sample_circle if meta['class'] == 'circle' else
               sample_annulus if meta['class'] == 'annulus' else sample_figure8)(n=40)
        geo_train.append(compute_geometry_features(pts, meta['eps']))
    for meta in meta_test:
        pts = (sample_line if meta['class'] == 'line' else
               sample_circle if meta['class'] == 'circle' else
               sample_annulus if meta['class'] == 'annulus' else sample_figure8)(n=40)
        geo_test.append(compute_geometry_features(pts, meta['eps']))
    G_train = np.array(geo_train)
    G_test = np.array(geo_test)
    gmu, gstd = G_train.mean(0), G_train.std(0) + 1e-10
    G_train_n = (G_train - gmu) / gstd
    G_test_n = (G_test - gmu) / gstd

    geo_model = BettiPredictor(G_train_n.shape[1], n_classes)
    geo_opt = torch.optim.Adam(geo_model.parameters(), lr=1e-3, weight_decay=1e-4)
    geo_ds = TensorDataset(torch.from_numpy(G_train_n), torch.from_numpy(y_train))
    geo_dl = DataLoader(geo_ds, batch_size=128, shuffle=True)
    for ep in range(100):
        geo_model.train()
        for xb, yb in geo_dl:
            geo_opt.zero_grad()
            crit(geo_model(xb), yb).backward()
            geo_opt.step()
    geo_model.eval()
    with torch.no_grad():
        geo_pred = geo_model(torch.from_numpy(G_test_n)).argmax(1).numpy()
    geo_acc = (geo_pred == y_test).mean()
    print(f"  Test accuracy: {geo_acc:.1%}")

    # --- Per-class accuracy ---
    print(f"\n--- Per-class test accuracy ---")
    print(f"  {'b₁':>4s}  {'HSD Spectral':>14s}  {'Geometry':>10s}")
    print(f"  {'-'*32}")
    for b in sorted(np.unique(y_test)):
        mask = y_test == b
        hsd_acc = (pred_test[mask] == y_test[mask]).mean()
        g_acc = (geo_pred[mask] == y_test[mask]).mean()
        print(f"  {b:>4d}  {hsd_acc:>14.1%}  {g_acc:>10.1%}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  HSD Spectral Features → b₁ prediction: {acc_test:.1%} accuracy")
    print(f"  Geometry Baseline → b₁ prediction:      {geo_acc:.1%} accuracy")
    print(f"  Improvement: +{(acc_test - geo_acc)*100:.1f}pp")

    # --- Figure ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (preds, title) in zip(axes, [(pred_test, f'HSD Spectral ({acc_test:.1%})'),
                                          (geo_pred, f'Geometry ({geo_acc:.1%})')]):
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, preds, labels=list(range(n_classes)))
        im = ax.imshow(cm, cmap='Blues', aspect='auto')
        ax.set_xlabel('Predicted b₁'); ax.set_ylabel('True b₁')
        ax.set_title(title)
        ax.set_xticks(range(n_classes)); ax.set_yticks(range(n_classes))
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > cm.max()/2 else 'black')
    plt.suptitle('Betti Number Prediction: HSD Spectral vs Geometry Features')
    plt.tight_layout()
    fig_path = os.path.join(out_dir, 'hsd_tda_classifier.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Figure: {fig_path}")


if __name__ == "__main__":
    main()
