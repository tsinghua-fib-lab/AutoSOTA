"""Analyze prefill linearity from benchmark results and generate plot + table."""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load data
data_path = Path(__file__).parent.parent.parent / "prefill_linearity_long.json"
with open(data_path) as f:
    data = json.load(f)

results = data["results"]
L = np.array([r["prefill_chunk_size"] for r in results], dtype=float)
T = np.array([r["mean_time_ms"] for r in results], dtype=float)
T_std = np.array([r["std_time_ms"] for r in results], dtype=float)

# --- Fitting ---
def fit_linear(x, y):
    """T = a + b*L"""
    A = np.column_stack([np.ones_like(x), x])
    coeffs, residuals, _, _ = np.linalg.lstsq(A, y, rcond=None)
    y_pred = A @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return coeffs, r2, y_pred

def fit_quadratic(x, y):
    """T = a + b*L + c*L^2"""
    A = np.column_stack([np.ones_like(x), x, x**2])
    coeffs, residuals, _, _ = np.linalg.lstsq(A, y, rcond=None)
    y_pred = A @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return coeffs, r2, y_pred

# Fit on different ranges
ranges = [
    ("≤4K", L <= 4096),
    ("≤8K", L <= 8192),
    ("≤16K", L <= 16384),
    ("≤32K", L <= 32768),
]

print("=" * 80)
print("PREFILL LINEARITY ANALYSIS — Qwen3-4B, RTX PRO 6000")
print("=" * 80)

# Raw data table
print("\n--- Raw Data ---")
print(f"{'L':>8} {'Time (ms)':>12} {'Std (ms)':>10} {'Tok/s':>12} {'Δslope (ms/tok)':>16}")
print("-" * 62)
for i, r in enumerate(results):
    slope = ""
    if i > 0:
        dt = T[i] - T[i-1]
        dl = L[i] - L[i-1]
        slope = f"{dt/dl:.5f}"
    print(f"{int(L[i]):>8} {T[i]:>12.2f} {T_std[i]:>10.3f} {r['throughput_tokens_per_sec']:>12.0f} {slope:>16}")

# Fitting table
print("\n--- R² Comparison ---")
print(f"{'Range':>8} {'Linear R²':>12} {'Quadratic R²':>14} {'Improvement':>14}")
print("-" * 52)

for name, mask in ranges:
    _, r2_lin, _ = fit_linear(L[mask], T[mask])
    _, r2_quad, _ = fit_quadratic(L[mask], T[mask])
    print(f"{name:>8} {r2_lin:>12.6f} {r2_quad:>14.6f} {r2_quad - r2_lin:>14.6f}")

# Full range fits for plotting
lin_coeffs, r2_lin_full, T_lin = fit_linear(L, T)
quad_coeffs, r2_quad_full, T_quad = fit_quadratic(L, T)

print(f"\n--- Full Range Coefficients ---")
print(f"Linear:    T = {lin_coeffs[0]:.3f} + {lin_coeffs[1]:.6f} × L  (R² = {r2_lin_full:.6f})")
print(f"Quadratic: T = {quad_coeffs[0]:.3f} + {quad_coeffs[1]:.6f} × L + {quad_coeffs[2]:.2e} × L²  (R² = {r2_quad_full:.6f})")

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: T vs L with fits
ax = axes[0]
L_dense = np.linspace(0, 35000, 500)
T_lin_dense = lin_coeffs[0] + lin_coeffs[1] * L_dense
T_quad_dense = quad_coeffs[0] + quad_coeffs[1] * L_dense + quad_coeffs[2] * L_dense**2

ax.scatter(L / 1000, T, color='black', zorder=5, s=50, label='Measured')
ax.plot(L_dense / 1000, T_lin_dense, '--', color='#2196F3', linewidth=2,
        label=f'Linear (R²={r2_lin_full:.4f})')
ax.plot(L_dense / 1000, T_quad_dense, '-', color='#F44336', linewidth=2,
        label=f'Quadratic (R²={r2_quad_full:.4f})')
# Shade the typical chunk_size region
ax.axvspan(0, 8.192, alpha=0.08, color='green', label='Typical max_num_batched_tokens\n(2K–8K)')
ax.set_xlabel('Prefill Length L (K tokens)', fontsize=12)
ax.set_ylabel('Iteration Time (ms)', fontsize=12)
ax.set_title('Prefill Time vs. Sequence Length', fontsize=13)
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3)

# Right: Residual (% deviation from linear fit)
ax2 = axes[1]
residual_pct = (T - T_lin) / T_lin * 100
ax2.bar(range(len(L)), residual_pct, color=['#4CAF50' if abs(r) < 5 else '#FF9800' if abs(r) < 15 else '#F44336' for r in residual_pct],
        edgecolor='black', linewidth=0.5)
ax2.set_xticks(range(len(L)))
ax2.set_xticklabels([f'{int(l/1000)}K' if l >= 1000 else str(int(l)) for l in L], fontsize=9)
ax2.set_xlabel('Prefill Length L', fontsize=12)
ax2.set_ylabel('Deviation from Linear Fit (%)', fontsize=12)
ax2.set_title('Linear Model Residuals', fontsize=13)
ax2.axhline(0, color='black', linewidth=0.8)
ax2.axhspan(-5, 5, alpha=0.1, color='green', label='±5%')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
out_path = Path(__file__).parent / "prefill_linearity_long.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {out_path}")
plt.close()
