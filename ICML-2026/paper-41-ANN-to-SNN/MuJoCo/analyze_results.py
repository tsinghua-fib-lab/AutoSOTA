"""Analyze ANN-to-SNN conversion results and compute APR."""
import numpy as np
import os

# Paper-reported ANN baselines (Table 10)
ANN_BASELINES = {
    "Ant-v4": 6505.0,
    "HalfCheetah-v4": 13193.0,
    "Hopper-v4": 3594.0,
    "Walker2d-v4": 4582.0,
}

ENVS = ["Hopper-v4", "HalfCheetah-v4", "Walker2d-v4", "Ant-v4"]
SEEDS = list(range(5))

results_dir = "/repo/MuJoCo/IF-3M/8"

print("=" * 80)
print("TD3 + IF + T=8 ANN-to-SNN Conversion Results")
print("=" * 80)

all_ratios = []

for env in ENVS:
    ann_baseline = ANN_BASELINES[env]
    env_ratios_alpha0 = []
    env_ratios_alpha2 = []

    print(f"\n--- {env} (ANN baseline: {ann_baseline:.1f}) ---")

    for seed in SEEDS:
        fname = f"TD3_{env}_{seed}.npy"
        fpath = os.path.join(results_dir, fname)
        if not os.path.exists(fpath):
            print(f"  seed {seed}: MISSING")
            continue

        data = np.load(fpath)  # shape [11, 10]
        mean_alpha0 = data[0].mean()   # alpha=0.0 (no CRPI)
        mean_alpha2 = data[2].mean()   # alpha=0.2 (CRPI, paper optimal)

        ratio0 = mean_alpha0 / ann_baseline * 100
        ratio2 = mean_alpha2 / ann_baseline * 100

        env_ratios_alpha0.append(ratio0)
        env_ratios_alpha2.append(ratio2)

        print(f"  seed {seed}: α=0 → {mean_alpha0:.1f} ({ratio0:.2f}%), α=0.2 → {mean_alpha2:.1f} ({ratio2:.2f}%)")

    if env_ratios_alpha2:
        avg_r0 = np.mean(env_ratios_alpha0)
        avg_r2 = np.mean(env_ratios_alpha2)
        std_r2 = np.std(env_ratios_alpha2)
        print(f"  AVG: α=0 → {avg_r0:.2f}%, α=0.2 → {avg_r2:.2f}% ± {std_r2:.2f}")
        all_ratios.append(avg_r2)

print("\n" + "=" * 80)
if all_ratios:
    overall_apr = np.mean(all_ratios)
    print(f"OVERALL APR (α=0.2, CRPI): {overall_apr:.2f}%")
    print(f"Per-environment ratios: {[f'{r:.2f}%' for r in all_ratios]}")

    # Also compute baseline APR (α=0)
    all_ratios_alpha0 = []
    for env in ENVS:
        ratios0 = []
        for seed in SEEDS:
            fpath = os.path.join(results_dir, f"TD3_{env}_{seed}.npy")
            if os.path.exists(fpath):
                data = np.load(fpath)
                ratios0.append(data[0].mean() / ANN_BASELINES[env] * 100)
        if ratios0:
            all_ratios_alpha0.append(np.mean(ratios0))

    baseline_apr = np.mean(all_ratios_alpha0)
    print(f"BASELINE APR (α=0, no CRPI): {baseline_apr:.2f}%")
else:
    print("NO RESULTS FOUND")

print("=" * 80)
