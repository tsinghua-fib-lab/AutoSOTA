"""Analyze prefill linearity across all Qwen3 models + Llama."""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent

def load_data(path):
    with open(path) as f:
        data = json.load(f)
    results = data["results"]
    L = np.array([r["prefill_chunk_size"] for r in results], dtype=float)
    T = np.array([r["mean_time_ms"] for r in results], dtype=float)
    return L, T

def fit_linear(x, y):
    A = np.column_stack([np.ones_like(x), x])
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    y_pred = A @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return coeffs, 1 - ss_res / ss_tot, y_pred

def fit_quadratic(x, y):
    A = np.column_stack([np.ones_like(x), x, x**2])
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    y_pred = A @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return coeffs, 1 - ss_res / ss_tot, y_pred

models = [
    ("Qwen3-4B\n(d=2560)", ROOT / "prefill_linearity_qwen3_4b_128k.json", 2560),
    ("Qwen3-8B\n(d=4096)", ROOT / "prefill_linearity_qwen3_8b_128k.json", 4096),
    ("Qwen3-14B\n(d=5120)", ROOT / "prefill_linearity_qwen3_14b_128k.json", 5120),
    ("Qwen3-30B-A3B\n(d=2048)", ROOT / "prefill_linearity_qwen3_30b_128k.json", 2048),
    ("Llama-3.2-1B\n(d=2048)", ROOT / "prefill_linearity_128k.json", 2048),
    ("Llama-3.1-8B\n(d=4096)", ROOT / "prefill_linearity_llama_8b_128k.json", 4096),
    ("Mistral-Nemo-12B\n(d=5120)", ROOT / "prefill_linearity_mistral_nemo_128k.json", 5120),
]

# --- R² table ---
print("=" * 90)
print("LINEAR R² BY MODEL AND RANGE")
print("=" * 90)
header = f"{'Range':>8}"
for name, _, _ in models:
    short = name.split('\n')[0]
    header += f" {short:>14}"
print(header)
print("-" * 90)

thresholds = [4096, 8192, 16384, 32768, 65536, 131072]
for t in thresholds:
    label = f"≤{t//1024}K"
    row = f"{label:>8}"
    for name, path, d in models:
        L, T = load_data(path)
        mask = L <= t
        if mask.sum() < 3:
            row += f" {'—':>14}"
            continue
        _, r2_lin, _ = fit_linear(L[mask], T[mask])
        row += f" {r2_lin:>14.4f}"
    print(row)

# Quadratic R²
print(f"\n{'QUADRATIC R²':>8}")
print("-" * 90)
for t in thresholds:
    label = f"≤{t//1024}K"
    row = f"{label:>8}"
    for name, path, d in models:
        L, T = load_data(path)
        mask = L <= t
        if mask.sum() < 3:
            row += f" {'—':>14}"
            continue
        _, r2_quad, _ = fit_quadratic(L[mask], T[mask])
        row += f" {r2_quad:>14.4f}"
    print(row)

# Per-token slope at ≤8K vs ≤32K
print(f"\n{'MODEL COMPARISON SUMMARY':>8}")
print("-" * 90)
print(f"{'Model':<18} {'d':>6} {'L/(6d)@32K':>12} {'Lin R²≤8K':>12} {'Lin R²≤32K':>12} {'Quad R²≤32K':>12}")
print("-" * 90)
for name, path, d in models:
    short = name.split('\n')[0]
    L, T = load_data(path)
    ratio_32k = 32768 / (6 * d)
    mask_8k = L <= 8192
    mask_32k = L <= 32768
    _, r2_8k, _ = fit_linear(L[mask_8k], T[mask_8k])
    _, r2_32k, _ = fit_linear(L[mask_32k], T[mask_32k])
    _, r2_32k_q, _ = fit_quadratic(L[mask_32k], T[mask_32k])
    print(f"{short:<18} {d:>6} {ratio_32k:>12.2f} {r2_8k:>12.4f} {r2_32k:>12.4f} {r2_32k_q:>12.4f}")

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Model groups for plotting
# 128K models for left plot (full range)
models_128k = [
    ("Llama-3.2-1B", ROOT / "prefill_linearity_128k.json", 2048, '#9C27B0'),
    ("Llama-3.1-8B", ROOT / "prefill_linearity_llama_8b_128k.json", 4096, '#E91E63'),
    ("Mistral-Nemo-12B", ROOT / "prefill_linearity_mistral_nemo_128k.json", 5120, '#00BCD4'),
]

# All models for right plot (R²)
models_r2 = [
    # Qwen3 - dashed
    ("Qwen3-4B", ROOT / "prefill_linearity_qwen3_4b_128k.json", 2560, '#2196F3', '--'),
    ("Qwen3-8B", ROOT / "prefill_linearity_qwen3_8b_128k.json", 4096, '#F44336', '--'),
    ("Qwen3-14B", ROOT / "prefill_linearity_qwen3_14b_128k.json", 5120, '#4CAF50', '--'),
    ("Qwen3-30B-A3B", ROOT / "prefill_linearity_qwen3_30b_128k.json", 2048, '#FF9800', '--'),
    # Other 128K models - solid
    ("Llama-3.2-1B", ROOT / "prefill_linearity_128k.json", 2048, '#9C27B0', '-'),
    ("Llama-3.1-8B", ROOT / "prefill_linearity_llama_8b_128k.json", 4096, '#E91E63', '-'),
    ("Mistral-Nemo-12B", ROOT / "prefill_linearity_mistral_nemo_128k.json", 5120, '#00BCD4', '-'),
]

# Left: all models, measured data only (one line per model)
ax = axes[0]
for idx, (name, path, d) in enumerate(models):
    L, T = load_data(path)
    short = name.split('\n')[0]
    color = models_r2[idx][3]
    ls = models_r2[idx][4]
    marker = 's' if ls == '--' else 'o'
    ax.plot(L / 1000, T / 1000, marker + '-', color=color, markersize=4, linewidth=1.5,
            label=f"{short} (d={d})")
ax.axvspan(0, 8.192, alpha=0.08, color='green')
ax.text(4.0, 0.5, 'Typical token\nbudget (2K–8K)', fontsize=7, color='green', ha='center', alpha=0.8)
ax.set_xlabel('Prefill Length L (K tokens)', fontsize=11)
ax.set_ylabel('Iteration Time (s)', fontsize=11)
ax.set_title('Prefill Time vs. Sequence Length (≤128K)', fontsize=12, fontweight='bold')
ax.legend(fontsize=7, loc='upper left')
ax.grid(True, alpha=0.3)

# Right: Linear R² vs range for all models
ax2 = axes[1]
for name, path, d, color, ls in models_r2:
    L, T = load_data(path)
    r2s = []
    valid_thresholds = []
    max_L = L.max()
    for t in thresholds:
        if t > max_L * 1.01:  # allow small margin (e.g. 131071 vs 131072)
            break
        mask = L <= t
        if mask.sum() >= 3:
            _, r2, _ = fit_linear(L[mask], T[mask])
            r2s.append(r2)
            valid_thresholds.append(t / 1000)
    marker = 's' if ls == '--' else 'o'
    ax2.plot(valid_thresholds, r2s, marker + ls, color=color, markersize=5, linewidth=1.5,
             label=f"{name} (d={d})")

ax2.axhline(0.99, color='gray', linestyle=':', alpha=0.5, linewidth=1)
ax2.text(125, 0.991, 'R²=0.99', fontsize=8, color='gray', ha='right')
ax2.set_xlabel('Fit Range Upper Bound (K tokens)', fontsize=11)
ax2.set_ylabel('Linear R²', fontsize=11)
ax2.set_title('Linear R² Degradation by Model Size', fontsize=12, fontweight='bold')
ax2.legend(fontsize=6.5, loc='lower left', ncol=2)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0.93, 1.005)
ax2.set_xlim(0, 135)

plt.tight_layout()
out_path = Path(__file__).parent / "prefill_linearity_all_models.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {out_path}")
plt.close()
