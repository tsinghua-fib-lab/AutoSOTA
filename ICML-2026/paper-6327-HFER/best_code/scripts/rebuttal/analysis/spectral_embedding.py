"""
Spectral Embedding / Conjecture Space (Day 2 CPU)
====================================================
Addresses: Reviewer MEs9 ("deeper understanding")

Input: data/results/rebuttal/llama8b_full_extraction.json
Output: 4-panel UMAP figure, clustering analysis

Usage:
    python scripts/rebuttal/spectral_embedding.py
"""

import os, sys, json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    'font.family': 'serif', 'font.size': 10, 'figure.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
})

INPUT_FILE = "data/results/rebuttal/llama8b_full_extraction.json"
METRICS = ["hfer", "fiedler", "smoothness", "entropy"]


def parse_source(filename):
    if filename.startswith('aime'): return 'AIME'
    elif filename.startswith('amc'): return 'AMC'
    elif filename.startswith('imo'): return 'IMO'
    elif filename.startswith('mathd'): return 'MathD'
    elif filename.startswith('algebra'): return 'Algebra'
    elif filename.startswith('induction'): return 'Induction'
    elif filename.startswith('numbertheory'): return 'NumberTheory'
    return 'Other'


def run_embedding():
    print("=" * 70)
    print("  SPECTRAL EMBEDDING / CONJECTURE SPACE")
    print("=" * 70)

    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: {INPUT_FILE} not found. Run day1_llama8b_batch.py first.")
        sys.exit(1)

    os.makedirs("output/rebuttal", exist_ok=True)
    os.makedirs("data/results/rebuttal", exist_ok=True)

    with open(INPUT_FILE) as f:
        data = json.load(f)

    # Build feature matrix: 4 metrics × N layers = features per proof
    samples = []
    for entry in data:
        spec = entry.get("spectral", {})
        if isinstance(spec, dict) and "error" not in spec:
            feats = []
            for layer_key in sorted(spec.keys(), key=lambda x: int(x.split("_")[1])):
                layer_data = spec[layer_key]
                for m in METRICS:
                    v = layer_data.get(m)
                    feats.append(v if v is not None else 0.0)

            is_platonic = (entry["label_original"] == "invalid" and entry["is_valid"])

            samples.append({
                "file": entry["file"],
                "label": "valid" if entry["is_valid"] else "invalid",
                "label_original": entry["label_original"],
                "platonic": "platonic" if is_platonic else ("valid" if entry["is_valid"] else "invalid"),
                "source": parse_source(entry["file"]),
                "features": feats,
            })

    X = np.array([s["features"] for s in samples])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    n_layers = X.shape[1] // len(METRICS)
    print(f"Feature matrix: {X.shape} ({len(METRICS)} metrics × {n_layers} layers)")

    # PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=min(50, X.shape[1]))
    X_pca = pca.fit_transform(X_scaled)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_95 = int(np.searchsorted(cumvar, 0.95) + 1)
    print(f"PCA: {n_95} components for 95% variance")

    # Dimensionality reduction to 2D
    try:
        from umap import UMAP
        reducer = UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        X_2d = reducer.fit_transform(X_pca[:, :n_95])
        method = "UMAP"
    except ImportError:
        from sklearn.manifold import TSNE
        X_2d = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_pca[:, :n_95])
        method = "t-SNE"

    # K-means
    best_k, best_sil = 3, -1
    for k in [3, 4, 5, 6]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_pca[:, :n_95])
        sil = silhouette_score(X_pca[:, :n_95], km.labels_)
        print(f"  K={k}: silhouette={sil:.3f}")
        if sil > best_sil: best_k, best_sil = k, sil

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(X_pca[:, :n_95])

    # 4-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # (a) Validity
    c_a = ['#1565C0' if s['label'] == 'valid' else '#C62828' for s in samples]
    axes[0,0].scatter(X_2d[:, 0], X_2d[:, 1], c=c_a, alpha=0.5, s=15)
    axes[0,0].set_title('(a) Validity (blue=valid, red=invalid)')

    # (b) Source
    src_colors = {'AMC': '#4CAF50', 'AIME': '#FF9800', 'IMO': '#F44336',
                  'MathD': '#2196F3', 'Algebra': '#9C27B0', 'Induction': '#00BCD4',
                  'NumberTheory': '#795548', 'Other': '#9E9E9E'}
    c_b = [src_colors.get(s['source'], '#9E9E9E') for s in samples]
    axes[0,1].scatter(X_2d[:, 0], X_2d[:, 1], c=c_b, alpha=0.5, s=15)
    axes[0,1].set_title('(b) Problem Source')

    # (c) K-means clusters
    cluster_colors = plt.cm.tab10(np.linspace(0, 1, best_k))
    c_c = [cluster_colors[l] for l in km.labels_]
    axes[1,0].scatter(X_2d[:, 0], X_2d[:, 1], c=c_c, alpha=0.5, s=15)
    axes[1,0].set_title(f'(c) K-means Clusters (k={best_k}, sil={best_sil:.2f})')

    # (d) Platonic validity
    plat_colors = {'valid': '#1565C0', 'invalid': '#C62828', 'platonic': '#FFD600'}
    c_d = [plat_colors.get(s['platonic'], '#9E9E9E') for s in samples]
    axes[1,1].scatter(X_2d[:, 0], X_2d[:, 1], c=c_d, alpha=0.5, s=15)
    axes[1,1].set_title('(d) Platonic Validity (gold=reclaimed)')

    for ax in axes.flat:
        ax.set_xlabel(f'{method} 1'); ax.set_ylabel(f'{method} 2')
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('output/rebuttal/spectral_embedding.pdf')
    plt.savefig('output/rebuttal/spectral_embedding.png')
    print(f"\nSaved: output/rebuttal/spectral_embedding.pdf")

    # Check platonic clustering
    platonic_idxs = [i for i, s in enumerate(samples) if s['platonic'] == 'platonic']
    valid_idxs = [i for i, s in enumerate(samples) if s['platonic'] == 'valid']
    if platonic_idxs and valid_idxs:
        plat_cluster_dist = np.bincount(km.labels_[platonic_idxs], minlength=best_k)
        valid_cluster_dist = np.bincount(km.labels_[valid_idxs], minlength=best_k)
        plat_pct = plat_cluster_dist / plat_cluster_dist.sum()
        valid_pct = valid_cluster_dist / valid_cluster_dist.sum()
        print(f"\n  Platonic proofs cluster distribution: {plat_pct}")
        print(f"  Valid proofs cluster distribution:    {valid_pct}")

    # Save
    with open("data/results/rebuttal/spectral_embedding.json", "w") as f:
        json.dump({
            "n_samples": len(samples), "n_features": X.shape[1],
            "pca_95pct_components": n_95, "best_k": best_k,
            "silhouette": round(float(best_sil), 3), "method": method,
        }, f, indent=2)
    print(f"Saved: data/results/rebuttal/spectral_embedding.json")


if __name__ == "__main__":
    run_embedding()
