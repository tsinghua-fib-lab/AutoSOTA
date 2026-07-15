"""RSA-CP ImageNet Evaluation Script - Paper 6081 Reproduction.

Reproduces Figure 4 from RSA-CP (ICML 2026): ImageNet classification
conformal prediction with m=15 calibration samples, N=1000 reference scores,
alpha in {0.05, 0.10}, beta=0.4, 100 trials.

Usage:
    cd /repo/Codes/real_data && python3 /repo/eval_imagenet.py
"""
import sys, os, time, numpy as np, pandas as pd, json

sys.path.insert(0, "/repo/Codes/real_data")

from real_data_pipeline import (
    load_imagenet_data, precompute_imagenet_aps,
    run_imagenet_experiment, CONFIG_DIR,
    RAW_DIR, SUMMARY_DIR, summarize
)
import real_data_pipeline as rdp

# Paper settings
rdp.N_SEEDS_REAL_DATA = 100
rdp.BETA_IMAGENET = 0.4
ALPHAS = [0.05, 0.10]

def main():
    print("=" * 60)
    print("RSA-CP Paper 6081 Reproduction - ImageNet Experiment")
    print("=" * 60)
    print("Settings: alpha=[0.05, 0.10], n_cal=15, n_ref=1000, beta=0.4, n_seeds=100")
    t0 = time.time()

    imagenet = load_imagenet_data(CONFIG_DIR / "imagenet_clip_marginal.yml")
    comp = precompute_imagenet_aps(imagenet)

    results = run_imagenet_experiment(
        imagenet, comp,
        figure="Figure 4",
        alpha_values=ALPHAS,
        n_cal_values=[15],
        n_ref_values=[1000],
        prior_scale=0.5,
        real_weight=1.1,
    )

    elapsed = time.time() - t0
    print("\nTotal time: {:.1f}s".format(elapsed))

    # Save results
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(str(RAW_DIR / "figure4_imagenet_main_raw.csv"), index=False)
    summary = summarize(results, ["figure", "dataset", "Method", "alpha", "n_cal", "n_ref", "n_test"])
    summary.to_csv(str(SUMMARY_DIR / "figure4_imagenet_main_summary.csv"), index=False)

    # Print results
    for alpha in ALPHAS:
        print("\n--- alpha={:.2f} (target coverage={:.2f}) ---".format(alpha, 1 - alpha))
        sub = results[results["alpha"] == alpha]
        for method in ["SCP", "Synthetic-only", "SPI", "RSA-CP (OT) (Ours)"]:
            s = sub[sub["Method"] == method]
            if len(s) == 0:
                continue
            cov = s["Coverage"].mean()
            cov_s = s["Coverage"].std()
            sz = s["Length"].mean()
            sz_s = s["Length"].std()
            print("  {:25s}: Coverage={:.4f}+/-{:.4f},  Set Size={:.2f}+/-{:.2f}".format(
                method, cov, cov_s, sz, sz_s))

    # Print key metric for reproduction
    sub05 = results[results["alpha"] == 0.05]
    rsacp = sub05[sub05["Method"] == "RSA-CP (OT) (Ours)"]
    print("\n=== KEY RESULT (alpha=0.05) ===")
    print("RSA-CP Coverage: {:.4f}".format(rsacp["Coverage"].mean()))
    print("RSA-CP Set Size: {:.2f} / 30 classes".format(rsacp["Length"].mean()))
    print("Rubric Coverage CI: [0.921, 0.959]")

    coverage_val = float(rsacp["Coverage"].mean())
    within_ci = 0.921 <= coverage_val <= 0.959
    print("Within CI: {}".format(within_ci))

    return results, summary

if __name__ == "__main__":
    main()
