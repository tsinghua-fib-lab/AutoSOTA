"""
Job 1: Full Eigenvalue Decomposition of ∇²_aa Q.
Computes all eigenvalues of the action Hessian at each (s,a) timestep.
Reports: negative definiteness rate, λ_max statistics.

Base vs PAVE, Lunar + Walker, 5 seeds.
Run from PAVE_Merge root: python td3/tests/eval_eigenvalue.py
"""
import sys, os, csv
import numpy as np
import torch as th

sys.path.append(os.path.join(os.path.dirname(__file__), "modules"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.envs import (make_lunar_env, make_pendulum_env, make_reacher_env,
                           make_ant_env, make_hopper_env, make_walker_env)
from stable_baselines3 import TD3

MAX_EPISODES = 10
EVAL_SEED_FILE = "./sac/tests/validation_seeds.txt"
TRAIN_SEEDS = [178132, 410580, 922852, 787576, 660993]
DEVICE = "cuda" if th.cuda.is_available() else "cpu"

OUTPUT_DIR = "./Full/td3/pths/eval_eigenvalue/"

CONFIGS = {
    "lunar": {"func": make_lunar_env, "dir": "./Full/td3/pths/lunar/"},
    "pendulum": {"func": make_pendulum_env, "dir": "./Full/td3/pths/pendulum/"},
    "reacher": {"func": make_reacher_env, "dir": "./Full/td3/pths/reacher/"},
    "ant": {"func": make_ant_env, "dir": "./Full/td3/pths/ant/"},
    "hopper": {"func": make_hopper_env, "dir": "./Full/td3/pths/hopper/"},
    "walker": {"func": make_walker_env, "dir": "./Full/td3/pths/walker/"},
}
METHODS = ["base_td3", "pave_td3"]

def load_eval_seeds(filepath):
    with open(filepath, "r") as f:
        return [int(line.strip()) for line in f if line.strip()]

def find_seed_file(save_dir, al_name, train_seed):
    for fname in os.listdir(save_dir):
        if al_name in fname and str(train_seed) in fname:
            full = os.path.join(save_dir, fname)
            if os.path.isdir(full):
                final = os.path.join(full, "final.zip")
                if os.path.isfile(final):
                    return os.path.join(fname, "final.zip")
    return None

def measure_eigenvalues(model, env, eval_seed, max_episodes):
    """Compute full eigenvalue decomposition of ∇²_aa Q at each timestep."""
    all_eigenvalues = []
    all_traces = []
    neg_def_count = 0
    total_count = 0

    obs, info = env.reset(seed=eval_seed)

    for ep in range(max_episodes):
        while True:
            action, _ = model.predict(obs, deterministic=True)

            obs_tensor, _ = model.policy.obs_to_tensor(obs)
            action_tensor = th.as_tensor(action, device=obs_tensor.device).unsqueeze(0)
            action_tensor.requires_grad_(True)

            q1 = model.critic(obs_tensor, action_tensor)[0].sum()

            # Step 1: ∂Q/∂a
            grad_a = th.autograd.grad(q1, action_tensor,
                                       create_graph=True, retain_graph=True)[0]

            action_dim = action_tensor.shape[1]

            # Step 2: Full Hessian ∇²_aa Q (d×d)
            hessian = th.zeros(action_dim, action_dim, device=action_tensor.device)
            for i in range(action_dim):
                grad2_i = th.autograd.grad(
                    grad_a[0, i], action_tensor,
                    retain_graph=True, create_graph=False
                )[0]
                hessian[i, :] = grad2_i[0, :]

            # Step 3: Symmetrize
            hessian = (hessian + hessian.T) / 2

            # Step 4: Eigenvalue decomposition
            eigenvalues = th.linalg.eigvalsh(hessian)
            eigs_np = eigenvalues.detach().cpu().numpy()
            all_eigenvalues.append(eigs_np)

            # Step 5: Trace (for cross-check with B.7)
            trace = float(eigs_np.sum())
            all_traces.append(trace)

            # Step 6: Negative definiteness
            lambda_max = float(eigs_np.max())
            if lambda_max < 0:
                neg_def_count += 1
            total_count += 1

            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
                break

    return {
        'eigenvalues': np.array(all_eigenvalues),  # (N, d)
        'traces': np.array(all_traces),
        'neg_def_rate': neg_def_count / total_count if total_count > 0 else 0,
        'total_count': total_count,
    }

def main():
    eval_seeds = load_eval_seeds(EVAL_SEED_FILE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== Full Eigenvalue Decomposition ===\n")

    for env_name, cfg in CONFIGS.items():
        env = cfg["func"]("rgb_array")()
        save_dir = cfg["dir"]

        for method in METHODS:
            print(f"--- {method} / {env_name} ---")

            seed_neg_def_rates = []
            seed_lambda_max_means = []
            seed_lambda_max_stds = []
            seed_lambda_min_means = []
            seed_trace_concavity_rates = []

            all_csv_rows = []

            for si, train_seed in enumerate(TRAIN_SEEDS):
                filename = find_seed_file(save_dir, method, train_seed)
                if filename is None:
                    print(f"  WARNING: seed {train_seed} not found")
                    continue

                eval_seed = eval_seeds[si]
                model = TD3.load(os.path.join(save_dir, filename), env=env)

                result = measure_eigenvalues(model, env, eval_seed, MAX_EPISODES)

                seed_neg_def_rates.append(result['neg_def_rate'])

                # λ_max statistics per seed
                lambda_maxs = result['eigenvalues'].max(axis=1)  # per-timestep max eigenvalue
                lambda_mins = result['eigenvalues'].min(axis=1)
                seed_lambda_max_means.append(np.mean(lambda_maxs))
                seed_lambda_max_stds.append(np.std(lambda_maxs))
                seed_lambda_min_means.append(np.mean(lambda_mins))

                # trace-based concavity (cross-check with B.7)
                trace_rate = np.mean(result['traces'] < -1.0)
                seed_trace_concavity_rates.append(trace_rate)

                # CSV rows
                for t in range(len(result['eigenvalues'])):
                    row = {
                        'seed': train_seed,
                        'timestep': t,
                        'trace': result['traces'][t],
                        'lambda_max': result['eigenvalues'][t].max(),
                        'lambda_min': result['eigenvalues'][t].min(),
                        'is_neg_def': 1 if result['eigenvalues'][t].max() < 0 else 0,
                    }
                    # Add individual eigenvalues
                    for d in range(result['eigenvalues'].shape[1]):
                        row[f'lambda_{d}'] = result['eigenvalues'][t, d]
                    all_csv_rows.append(row)

                print(f"  seed {train_seed}: neg_def={result['neg_def_rate']:.3f}, "
                      f"λ_max={np.mean(lambda_maxs):.3f}±{np.std(lambda_maxs):.3f}, "
                      f"trace_concavity={trace_rate:.3f}, "
                      f"n={result['total_count']}")

            # Summary
            if seed_neg_def_rates:
                print(f"  SUMMARY: neg_def_rate={np.mean(seed_neg_def_rates):.3f}±{np.std(seed_neg_def_rates):.3f}, "
                      f"λ_max_mean={np.mean(seed_lambda_max_means):.3f}, "
                      f"trace_concavity={np.mean(seed_trace_concavity_rates):.3f}")

            # Save CSV
            if all_csv_rows:
                csv_path = os.path.join(OUTPUT_DIR, f"eigenvalues_{method}_{env_name}.csv")
                fieldnames = list(all_csv_rows[0].keys())
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(all_csv_rows)
                print(f"  Saved: {csv_path}")

            print()

        env.close()

    print("Done!")

if __name__ == "__main__":
    main()
