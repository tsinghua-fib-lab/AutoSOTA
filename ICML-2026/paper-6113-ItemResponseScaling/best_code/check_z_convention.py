"""
Check the z-score sign convention by comparing z-scores with actual pass rates.
If higher z means easier: high z should correlate with high pass@1
If higher z means harder: high z should correlate with low pass@1
"""

import torch
import numpy as np
from scipy.stats import spearmanr
from huggingface_hub import snapshot_download

FILE_NAME = "irsl_testtime_resmat1"
cache_dir = snapshot_download(repo_id=f"stair-lab/{FILE_NAME}", repo_type="dataset")

# Load data
testtime_resmat = torch.load(f"{cache_dir}/resmat.pt", weights_only=False)
data_tensor = testtime_resmat["data_tensor"].numpy()
helm_zs = np.array(testtime_resmat["zs"])

# Load recalibrated z-scores
testtime_calibrated = torch.load("monkey/monkey_analysis/irsl_testtime_resmat1_withz.pt", weights_only=False)
testtime_zs = testtime_calibrated["zs"]
if hasattr(testtime_zs, 'numpy'):
    testtime_zs = testtime_zs.numpy()

# Compute average pass@1 across all models for each question
overall_pass_at_1 = np.mean(data_tensor, axis=(0, 2))  # Average over models and samples

print("="*80)
print("CHECKING Z-SCORE SIGN CONVENTION")
print("="*80)

# Remove NaN values before correlation
valid_mask = ~np.isnan(overall_pass_at_1)
helm_zs_clean = helm_zs[valid_mask]
testtime_zs_clean = testtime_zs[valid_mask]
overall_pass_at_1_clean = overall_pass_at_1[valid_mask]

# Check HELM z-scores
helm_corr, helm_p = spearmanr(helm_zs_clean, overall_pass_at_1_clean)
print(f"\nHELM z-scores vs overall pass@1:")
print(f"Spearman correlation: {helm_corr:.4f} (p={helm_p:.2e})")
if helm_corr > 0:
    print("✓ Positive correlation → Higher z = EASIER (z represents easiness)")
else:
    print("✓ Negative correlation → Higher z = HARDER (z represents difficulty)")

# Check test-time z-scores
testtime_corr, testtime_p = spearmanr(testtime_zs_clean, overall_pass_at_1_clean)
print(f"\nTest-time z-scores vs overall pass@1:")
print(f"Spearman correlation: {testtime_corr:.4f} (p={testtime_p:.2e})")
if testtime_corr > 0:
    print("✓ Positive correlation → Higher z = EASIER (z represents easiness)")
else:
    print("✓ Negative correlation → Higher z = HARDER (z represents difficulty)")

# Show some examples
print(f"\n{'='*80}")
print("EXAMPLE QUESTIONS")
print(f"{'='*80}")

# Sort by HELM z-score
sorted_idx = np.argsort(helm_zs)

print("\n5 LOWEST HELM z-scores (should be hardest if paper convention, easiest if code convention):")
for idx in sorted_idx[:5]:
    print(f"  HELM z={helm_zs[idx]:6.3f}, Test-time z={testtime_zs[idx]:6.3f}, Pass@1={overall_pass_at_1[idx]:.3f}")

print("\n5 HIGHEST HELM z-scores (should be easiest if paper convention, hardest if code convention):")
for idx in sorted_idx[-5:]:
    print(f"  HELM z={helm_zs[idx]:6.3f}, Test-time z={testtime_zs[idx]:6.3f}, Pass@1={overall_pass_at_1[idx]:.3f}")

# Check test-time z-scores
sorted_idx_tt = np.argsort(testtime_zs)

print("\n5 LOWEST Test-time z-scores:")
for idx in sorted_idx_tt[:5]:
    print(f"  HELM z={helm_zs[idx]:6.3f}, Test-time z={testtime_zs[idx]:6.3f}, Pass@1={overall_pass_at_1[idx]:.3f}")

print("\n5 HIGHEST Test-time z-scores:")
for idx in sorted_idx_tt[-5:]:
    print(f"  HELM z={helm_zs[idx]:6.3f}, Test-time z={testtime_zs[idx]:6.3f}, Pass@1={overall_pass_at_1[idx]:.3f}")

print(f"\n{'='*80}")
print("CONCLUSION")
print(f"{'='*80}")

if helm_corr > 0 and testtime_corr > 0:
    print("\n✓ Both HELM and test-time use: Higher z = EASIER")
    print("✓ The CODE convention (σ(θ + z)) is being used")
    print("✗ This CONTRADICTS the paper which says 'higher z = more difficult'")
    print("\n⚠️  THE PAPER AND CODE USE OPPOSITE SIGN CONVENTIONS!")
elif helm_corr < 0 and testtime_corr < 0:
    print("\n✓ Both HELM and test-time use: Higher z = HARDER")
    print("✓ The PAPER convention (σ(θ - z)) is being used")
    print("✓ This matches the paper definition")
else:
    print("\n⚠️  HELM and test-time use DIFFERENT conventions!")
    print(f"HELM: {'Higher z = easier' if helm_corr > 0 else 'Higher z = harder'}")
    print(f"Test-time: {'Higher z = easier' if testtime_corr > 0 else 'Higher z = harder'}")
