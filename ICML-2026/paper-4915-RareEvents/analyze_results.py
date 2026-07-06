import json
import numpy as np

with open("reproduction_output/reproduction_results.json") as f:
    r = json.load(f)

# Metric 1: Minimum Accessible Probability Density
print("=== Metric 1: Minimum Accessible Probability Density ===")
m1 = r["metrics"]
print("Density at ARI~14: %.6e" % m1["minimum_accessible_probability_density"])
print("  at bin center: %.2f" % m1["minimum_accessible_probability_density_at_ari"])
print("Min density (right tail, ARI>5, direct=0): %.6e" % m1["min_density_right_tail_beyond_direct"])
print("Min density (overall): %.6e" % m1["min_density_overall"])

# Metric 2: Relative CI Half Width
print("\n=== Metric 2: Relative CI Half Width ===")
print("MBAR min: %.6e" % m1["mbar_min_relative_ci_half_width"])
print("MBAR median: %.6e" % m1["mbar_median_relative_ci_half_width"])
print("Direct min: %.6e" % m1["direct_min_relative_ci_half_width"])
print("Direct median: %.6e" % m1["direct_median_relative_ci_half_width"])

# Fix direct CI (the Wilson interval swapped upper/lower in notebook code)
bin_centers = np.array(r["histogram"]["bin_centers"])
direct_heights = np.array(r["histogram"]["direct_heights"])
ci_lower = np.array(r["histogram"]["ci_lower"])
ci_upper = np.array(r["histogram"]["ci_upper"])
mbar_heights = np.array(r["histogram"]["mbar_heights"])

# MBAR CI half-width is correct (bootstrap percentiles)
ci_hw_mbar = (ci_upper - ci_lower) / 2
rel_ci_hw_mbar = ci_hw_mbar / np.where(mbar_heights > 0, mbar_heights, np.inf)

peak_mask = (bin_centers >= 0) & (bin_centers <= 5) & (mbar_heights > 0)
if np.any(peak_mask):
    peak_rel = rel_ci_hw_mbar[peak_mask]
    print("\nMBAR min rel CI hw (peak 0-5): %.6e" % np.min(peak_rel))
    print("  at ARI~%.2f" % bin_centers[peak_mask][np.argmin(peak_rel)])
    print("MBAR median rel CI hw (peak 0-5): %.6e" % np.median(peak_rel))

# Also check density values at key ARI values
print("\n=== Density at key ARI values ===")
for ari_target in [0, 5, 10, 12, 13, 13.5, 14, 14.5, 15]:
    idx = np.argmin(np.abs(bin_centers - ari_target))
    print("ARI~%.2f: MBAR=%.6e, Direct=%.6e" % (bin_centers[idx], mbar_heights[idx], direct_heights[idx]))

# Compute direct relative CI half-width correctly
print("\n=== Direct Wilson CI (corrected) ===")
# Wilson interval: bin count + zscore correction
n = r["num_samples"]["total_direct_histogram"]
zscore = 2.0537  # 96% CI
n_s = direct_heights * (r["bins"]["bin_width"]) * n  # convert density back to counts
n_f = n - n_s
p_hat = (n_s + 0.5 * (zscore**2)) / (n + zscore**2)
diff = (zscore / (n + zscore**2)) * np.sqrt(((n_s * n_f) / n) + (zscore**2 / 4))
ci_hw_direct_corrected = diff / r["bins"]["bin_width"]
rel_ci_hw_direct_corrected = ci_hw_direct_corrected / np.where(direct_heights > 0, direct_heights, np.inf)

direct_peak = (bin_centers >= 0) & (bin_centers <= 5) & (direct_heights > 0)
if np.any(direct_peak):
    print("Direct min rel CI hw (peak 0-5, corrected): %.6e" % np.min(rel_ci_hw_direct_corrected[direct_peak]))
    print("Direct median rel CI hw (peak 0-5, corrected): %.6e" % np.median(rel_ci_hw_direct_corrected[direct_peak]))

# Summary
print("\n=== SUMMARY ===")
print("Paper MBAR density at ARI=14: ~1e-10")
print("Our MBAR density at ARI~14: %.2e" % m1["density_at_ari_14"] if "density_at_ari_14" in m1 else m1["minimum_accessible_probability_density"])
print("Paper MBAR min rel CI hw: ~0.003")
print("Our MBAR min rel CI hw: %.4f" % np.min(peak_rel) if np.any(peak_mask) else float('nan'))
