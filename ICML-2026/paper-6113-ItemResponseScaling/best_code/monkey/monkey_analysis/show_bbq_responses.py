"""
Show actual model responses for BBQ questions to understand the transfer failure.
"""

import torch
import numpy as np
from huggingface_hub import snapshot_download
import pandas as pd

# Load data
FILE_NAME = "irsl_testtime_resmat1"
cache_dir = snapshot_download(repo_id=f"stair-lab/{FILE_NAME}", repo_type="dataset")

print("Loading data...")
testtime_resmat = torch.load(f"{cache_dir}/resmat.pt", map_location="cpu", weights_only=False)
data_tensor = testtime_resmat["data_tensor"].numpy()
models = testtime_resmat["models"]
questions = testtime_resmat["questions"]
scenarios = testtime_resmat["scenarios"]
helm_zs = np.array(testtime_resmat["zs"])

# Load calibrated data
import os
calibrated_path = os.path.join(os.path.dirname(__file__), f"{FILE_NAME}_withz.pt")
testtime_calibrated = torch.load(calibrated_path, weights_only=False)
testtime_zs = testtime_calibrated["zs"]
if hasattr(testtime_zs, 'numpy'):
    testtime_zs = testtime_zs.numpy()

# Find BBQ questions with high discrepancy
bbq_mask = np.array([s == "bbq" for s in scenarios])
bbq_indices = np.where(bbq_mask)[0]
bbq_z_diffs = np.abs(testtime_zs[bbq_mask] - helm_zs[bbq_mask])

# Get top discrepancy questions
top_bbq_idxs = bbq_indices[np.argsort(bbq_z_diffs)[-3:]][::-1]

print("\n" + "="*80)
print("BBQ QUESTIONS WITH HIGHEST Z-SCORE DISCREPANCIES")
print("="*80)

for rank, q_idx in enumerate(top_bbq_idxs, 1):
    print(f"\n{'='*80}")
    print(f"QUESTION #{rank}")
    print(f"{'='*80}")

    question = questions[q_idx]
    helm_z = helm_zs[q_idx]
    testtime_z = testtime_zs[q_idx]
    z_diff = abs(helm_z - testtime_z)

    print(f"\nQuestion:\n{question}\n")
    print(f"HELM z-score:      {helm_z:7.3f} ({'EASY' if helm_z < 0 else 'HARD'})")
    print(f"Test-time z-score: {testtime_z:7.3f} ({'EASY' if testtime_z < 0 else 'HARD'})")
    print(f"Difference:        {z_diff:7.3f}")

    # Get responses for this question
    responses = data_tensor[:, q_idx, :]  # (8 models, 10000 samples)

    print(f"\n{'='*80}")
    print("MODEL RESPONSES (showing first 20 samples per model)")
    print(f"{'='*80}")

    for model_idx, model_name in enumerate(models):
        model_responses = responses[model_idx, :20]  # First 20 samples
        pass_at_1 = np.mean(responses[model_idx, :])
        std = np.std(responses[model_idx, :])

        print(f"\n{model_name}:")
        print(f"  Pass@1: {pass_at_1:.3f}, Std: {std:.3f}")
        print(f"  Sample responses (0=incorrect, 1=correct):")
        print(f"  {model_responses}")

        # Show pattern
        correct_count = np.sum(model_responses)
        print(f"  → {correct_count}/20 correct in this sample")

    print(f"\n{'='*80}")
    print("AGGREGATE STATISTICS")
    print(f"{'='*80}")

    model_means = np.mean(responses, axis=1)
    model_stds = np.std(responses, axis=1)

    print(f"Pass@1 by model: {', '.join([f'{m:.3f}' for m in model_means])}")
    print(f"Std by model:    {', '.join([f'{s:.3f}' for s in model_stds])}")
    print(f"\nIntra-model variance (avg): {np.mean(model_stds**2):.4f}")
    print(f"Inter-model variance:       {np.var(model_means):.4f}")

    # Interpretation
    print(f"\n{'='*80}")
    print("INTERPRETATION")
    print(f"{'='*80}")

    if helm_z < 0 and testtime_z > 0:
        print("⚠️  HELM perceived this as EASY (z < 0)")
        print("⚠️  Test-time perceived this as HARD (z > 0)")
        print("⚠️  This is a MASSIVE perception shift!")
    elif helm_z > 0 and testtime_z < 0:
        print("⚠️  HELM perceived this as HARD (z > 0)")
        print("⚠️  Test-time perceived this as EASY (z < 0)")
        print("⚠️  This is a MASSIVE perception shift!")
    else:
        print("Both evaluations agree on easy/hard direction, but magnitudes differ")

    avg_std = np.mean(model_stds)
    if avg_std < 0.1:
        print(f"✓ Low response variance ({avg_std:.3f}) - models are VERY consistent")
    elif avg_std < 0.3:
        print(f"→ Moderate response variance ({avg_std:.3f})")
    else:
        print(f"✗ High response variance ({avg_std:.3f}) - models are inconsistent")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
