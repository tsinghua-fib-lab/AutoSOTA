"""
Scalability Benchmarks for Spectral Analysis
=============================================
Addresses: Reviewer mxpA Q4 ("scaling to 100k+ tokens with O(N³) eigendecomposition")

Benchmarks eigendecomposition time vs sequence length, comparing full and approximate
methods. Also tests HFER approximation quality with partial eigendecompositions.

No real data needed — uses synthetic random Laplacians.

Usage:
    cd geometry-of-reason
    python scripts/rebuttal/scalability_bench.py
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from scipy.sparse.linalg import eigsh
from sklearn.utils.extmath import randomized_svd

# Paper-quality matplotlib settings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})


def generate_attention_laplacian(n, seed=42):
    """Generate a random symmetric positive semidefinite Laplacian
    simulating an attention weight matrix."""
    rng = np.random.RandomState(seed)
    # Simulate attention weights (softmax-like, non-negative)
    W = rng.exponential(scale=1.0, size=(n, n))
    # Symmetrize
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0)
    # Build Laplacian: L = D - W
    D = np.diag(W.sum(axis=1))
    L = D - W
    return L.astype(np.float64)


def compute_hfer_full(eigenvalues, signal=None):
    """Compute HFER from full eigenvalue decomposition.
    HFER = high-frequency energy ratio using median cutoff."""
    eigenvalues = np.sort(np.abs(eigenvalues))
    n = len(eigenvalues)
    if signal is None:
        # Use uniform signal for consistent comparison
        signal = np.ones(n) / np.sqrt(n)
    cutoff = n // 2
    low_energy = np.sum(eigenvalues[:cutoff] * signal[:cutoff]**2)
    high_energy = np.sum(eigenvalues[cutoff:] * signal[cutoff:]**2)
    total = low_energy + high_energy
    if total < 1e-15:
        return 0.5
    return high_energy / total


def compute_hfer_partial(eigenvalues_partial, k, n):
    """Approximate HFER from partial (k smallest) eigenvalues.
    Assumes remaining eigenvalues contribute to high-frequency energy."""
    eigenvalues_partial = np.sort(np.abs(eigenvalues_partial))
    cutoff = n // 2
    # Low-frequency energy from known small eigenvalues
    k_low = min(k, cutoff)
    low_energy = np.sum(eigenvalues_partial[:k_low])
    # For eigenvalues beyond k, estimate from the largest known eigenvalue
    if k < cutoff:
        # Extrapolate remaining low-freq eigenvalues
        est_val = eigenvalues_partial[-1] if k > 0 else 1.0
        low_energy += est_val * (cutoff - k)
    # High-frequency: estimate total energy minus low
    high_from_known = np.sum(eigenvalues_partial[k_low:])
    est_remaining = eigenvalues_partial[-1] * (n - k) if k > 0 else n
    high_energy = high_from_known + est_remaining
    total = low_energy + high_energy
    if total < 1e-15:
        return 0.5
    return high_energy / total


def benchmark_method(method_fn, L, repeats=5):
    """Time a method over multiple repeats, return median time."""
    times = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = method_fn(L)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.median(times), result


def run_scalability_benchmark():
    """Main benchmark loop."""
    print("=" * 70)
    print("  SCALABILITY BENCHMARK: Eigendecomposition vs Sequence Length")
    print("=" * 70)

    sizes = [100, 250, 500, 1000, 2000, 5000, 10000]
    methods = {
        'Full (eigh)': lambda L: np.linalg.eigh(L)[0],
        'Partial k=20': lambda L: eigsh(L, k=min(20, L.shape[0]-2), which='SM', return_eigenvectors=False),
        'Partial k=50': lambda L: eigsh(L, k=min(50, L.shape[0]-2), which='SM', return_eigenvectors=False),
        'Randomized SVD k=50': lambda L: randomized_svd(L, n_components=min(50, L.shape[0]-1), random_state=42)[1],
    }

    results = {name: [] for name in methods}
    results['N'] = []

    for n in sizes:
        print(f"\n--- N = {n} ---")
        L = generate_attention_laplacian(n)
        results['N'].append(n)

        for name, method_fn in methods.items():
            try:
                k_val = 20 if 'k=20' in name else 50
                if n <= k_val + 2 and 'Partial' in name:
                    print(f"  {name}: SKIPPED (N too small for k={k_val})")
                    results[name].append(None)
                    continue
                median_time, _ = benchmark_method(method_fn, L, repeats=5)
                print(f"  {name}: {median_time*1000:.1f} ms")
                results[name].append(median_time)
            except Exception as e:
                print(f"  {name}: ERROR ({e})")
                results[name].append(None)

    # --- HFER Approximation Quality ---
    print("\n" + "=" * 70)
    print("  HFER APPROXIMATION QUALITY (N=1000)")
    print("=" * 70)

    n_test = 1000
    L_test = generate_attention_laplacian(n_test, seed=123)
    full_evals = np.linalg.eigh(L_test)[0]
    hfer_full = compute_hfer_full(full_evals)
    print(f"  Full spectrum HFER: {hfer_full:.6f}")

    approx_results = []
    for k in [2, 5, 10, 20, 50, 100, 200]:
        if k >= n_test - 1:
            continue
        partial_evals = eigsh(L_test, k=k, which='SM', return_eigenvectors=False)
        hfer_approx = compute_hfer_partial(partial_evals, k, n_test)
        pct_diff = abs(hfer_approx - hfer_full) / hfer_full * 100
        print(f"  k={k:>4}: HFER={hfer_approx:.6f}  (Δ={pct_diff:.2f}%)")
        approx_results.append({'k': k, 'hfer_approx': hfer_approx, 'pct_diff': pct_diff})

    # Key insight: Fiedler only needs λ₂ → k=2 suffices
    fiedler_full = full_evals[1]
    fiedler_partial = eigsh(L_test, k=2, which='SM', return_eigenvectors=False)
    fiedler_approx = np.sort(fiedler_partial)[1] if len(fiedler_partial) > 1 else fiedler_partial[0]
    fiedler_diff = abs(fiedler_approx - fiedler_full) / abs(fiedler_full) * 100
    print(f"\n  Fiedler value (full): {fiedler_full:.6f}")
    print(f"  Fiedler value (k=2):  {fiedler_approx:.6f}  (Δ={fiedler_diff:.4f}%)")

    # --- Extrapolation ---
    print("\n" + "=" * 70)
    print("  EXTRAPOLATED WALL-CLOCK TIMES")
    print("=" * 70)

    # Fit power law: t = a * N^b
    for name in ['Full (eigh)', 'Partial k=50']:
        valid_n = []
        valid_t = []
        for i, n in enumerate(sizes):
            t = results[name][i]
            if t is not None:
                valid_n.append(np.log(n))
                valid_t.append(np.log(t))
        if len(valid_n) > 2:
            coeffs = np.polyfit(valid_n, valid_t, 1)
            b = coeffs[0]
            a = np.exp(coeffs[1])
            print(f"  {name}: t ≈ {a:.2e} × N^{b:.2f}")
            for n_ext in [32000, 100000, 1000000]:
                t_ext = a * n_ext**b
                if t_ext < 60:
                    print(f"    N={n_ext:>8,}: {t_ext:.2f} sec")
                elif t_ext < 3600:
                    print(f"    N={n_ext:>8,}: {t_ext/60:.1f} min")
                else:
                    print(f"    N={n_ext:>8,}: {t_ext/3600:.1f} hours")

    # --- Generate Plot ---
    os.makedirs('output/rebuttal', exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: log-log scaling plot
    colors = {'Full (eigh)': '#2196F3', 'Partial k=20': '#4CAF50',
              'Partial k=50': '#FF9800', 'Randomized SVD k=50': '#9C27B0'}
    markers = {'Full (eigh)': 'o', 'Partial k=20': 's',
               'Partial k=50': '^', 'Randomized SVD k=50': 'D'}

    for name in methods:
        valid_n = []
        valid_t = []
        for i, n in enumerate(sizes):
            t = results[name][i]
            if t is not None:
                valid_n.append(n)
                valid_t.append(t * 1000)  # convert to ms
        if valid_n:
            ax1.loglog(valid_n, valid_t, marker=markers[name],
                      color=colors[name], label=name, linewidth=2, markersize=6)

    # Add reference slopes
    ns = np.array([100, 10000])
    ax1.loglog(ns, 1e-3 * (ns/100)**3, '--', color='gray', alpha=0.4, linewidth=1, label='O(N³)')
    ax1.loglog(ns, 1e-3 * (ns/100)**2, ':', color='gray', alpha=0.4, linewidth=1, label='O(N²)')

    ax1.set_xlabel('Sequence Length (N)')
    ax1.set_ylabel('Wall-Clock Time (ms)')
    ax1.set_title('Eigendecomposition Scaling')
    ax1.legend(fontsize=8, loc='upper left')
    ax1.grid(True, alpha=0.3, which='both')

    # Right: HFER approximation error
    ks = [r['k'] for r in approx_results]
    pct_diffs = [r['pct_diff'] for r in approx_results]
    ax2.semilogy(ks, pct_diffs, 'o-', color='#E91E63', linewidth=2, markersize=6)
    ax2.axhline(y=2.0, color='green', linestyle='--', alpha=0.6, label='2% threshold')
    ax2.axhline(y=5.0, color='orange', linestyle='--', alpha=0.6, label='5% threshold')
    ax2.set_xlabel('Number of Eigenvalues (k)')
    ax2.set_ylabel('HFER Approximation Error (%)')
    ax2.set_title('HFER Quality vs Partial Eigendecomposition')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = 'output/rebuttal/scalability.pdf'
    plt.savefig(fig_path)
    plt.savefig(fig_path.replace('.pdf', '.png'))
    print(f"\nSaved: {fig_path}")

    # --- Save JSON ---
    os.makedirs('data/results/rebuttal', exist_ok=True)
    save_data = {
        'scaling': {
            'N': sizes,
            'methods': {}
        },
        'hfer_approximation': approx_results,
        'fiedler_approximation': {
            'full': float(fiedler_full),
            'partial_k2': float(fiedler_approx),
            'pct_diff': float(fiedler_diff)
        }
    }
    for name in methods:
        save_data['scaling']['methods'][name] = [
            float(t) if t is not None else None for t in results[name]
        ]

    json_path = 'data/results/rebuttal/scalability.json'
    with open(json_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"Saved: {json_path}")


if __name__ == '__main__':
    run_scalability_benchmark()
"""
Description: Scalability benchmarks for eigendecomposition methods.
Addresses reviewer mxpA Q4 about O(N³) scaling concerns.
"""
