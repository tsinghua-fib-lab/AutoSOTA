"""Evaluation script for GenUnfold reproduction (paper 4082).
Computes Rel l2, FID, Force-JSD, and Energy-JSD from pre-generated test curves.
"""
import numpy as np
import sys
sys.path.insert(0, '/repo')

from src.evaluation.metrics import (
    calculate_relative_l2_error,
    compute_fid,
    evaluate_mechanical_properties,
)

def main():
    data_dir = '/repo/scripts/data'
    true_curves = np.load(f'{data_dir}/true_curves.npy', allow_pickle=True)
    gen_curves = np.load(f'{data_dir}/generated_curves.npy', allow_pickle=True)
    gen_curves = np.clip(gen_curves, 0, 1)

    rel_l2 = calculate_relative_l2_error(true_curves, gen_curves)
    fid = compute_fid(true_curves, gen_curves)

    peak_params = {"height": 0, "distance": 50, "prominence": 0.02}
    props = evaluate_mechanical_properties(
        true_curves, gen_curves,
        property_extraction_params={"find_peaks": peak_params},
    )

    force_jsd = props["max_force"]["Jensen-Shannon Divergence"]
    energy_jsd = props["unfolding_energy"]["Jensen-Shannon Divergence"]

    print(f"Rel_l2={rel_l2:.6f}")
    print(f"FID={fid:.6f}")
    print(f"Force-JSD={force_jsd:.6f}")
    print(f"Energy-JSD={energy_jsd:.6f}")

    # Paper reference values and CIs
    print("\n--- Paper comparison ---")
    print(f"Rel_l2:    {rel_l2:.4f} vs paper 0.2070±0.0050 (CI: [0.2020, 0.2120])")
    print(f"FID:       {fid:.4f} vs paper 0.0117±0.0011 (CI: [0.0106, 0.0128])")
    print(f"Force-JSD: {force_jsd:.4f} vs paper 0.0553±0.0032 (CI: [0.0521, 0.0585])")
    print(f"Energy-JSD:{energy_jsd:.4f} vs paper 0.0338±0.0038 (CI: [0.0300, 0.0376])")

    return {
        "rel_l2": float(rel_l2),
        "fid": float(fid),
        "force_jsd": float(force_jsd),
        "energy_jsd": float(energy_jsd),
    }

if __name__ == "__main__":
    main()
