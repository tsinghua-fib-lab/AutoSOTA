"""Full alpha analysis per environment."""
import numpy as np
import os

ANN_BASELINES = {
    'Ant-v4': 6505.0, 'HalfCheetah-v4': 13193.0,
    'Hopper-v4': 3594.0, 'Walker2d-v4': 4582.0,
}

ENVS = ['Hopper-v4', 'HalfCheetah-v4', 'Walker2d-v4', 'Ant-v4']

for env in ENVS:
    ann = ANN_BASELINES[env]
    all_data = []
    for seed in range(5):
        fpath = f'/repo/MuJoCo/IF-3M/8/TD3_{env}_{seed}.npy'
        if os.path.exists(fpath):
            all_data.append(np.load(fpath))
    if not all_data:
        continue
    stacked = np.stack(all_data, axis=0)  # [5, 11, 10]

    print(f'\n=== {env} (ANN baseline: {ann:.0f}) ===')
    print(f'{"Alpha":>6}  {"Mean Return":>12}  {"APR%":>8}  {"Per-seed returns"}')
    for alpha_idx in range(11):
        alpha = alpha_idx / 10.0
        seed_means = stacked[:, alpha_idx, :].mean(axis=1)
        overall_mean = seed_means.mean()
        apr = overall_mean / ann * 100
        seed_str = ' '.join([f'{s:.0f}' for s in seed_means])
        print(f'{alpha:>6.1f}  {overall_mean:>12.1f}  {apr:>8.2f}  {seed_str}')

# Overall APR with alpha=0.2 (paper CRPI setting)
print("\n=== OVERALL APR (alpha=0.2) ===")
ratios = []
for env in ENVS:
    ann = ANN_BASELINES[env]
    seed_means = []
    for seed in range(5):
        fpath = f'/repo/MuJoCo/IF-3M/8/TD3_{env}_{seed}.npy'
        if os.path.exists(fpath):
            data = np.load(fpath)
            seed_means.append(data[2].mean())
    if seed_means:
        env_ratio = np.mean(seed_means) / ann * 100
        ratios.append(env_ratio)
        print(f"  {env}: {env_ratio:.2f}%")
print(f"  Overall APR: {np.mean(ratios):.2f}%")
print(f"  Paper APR: 72.26%")
print(f"  Paper baseline APR: 64.71%")
