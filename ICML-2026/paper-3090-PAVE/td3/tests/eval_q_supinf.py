"""
Q2 Rebuttal: M (sup) and μ (inf) measurement — seed-wise sup/inf, then across-seed mean±std.
Matches Proposition 4.2 definitions:
  M = sup_s ||∇²_sa Q||_2
  μ = inf_s |λ_max(∇²_aa Q)|  (approximated by |tr(∇²_aa Q)|)

Uses first 5 seeds only. Does NOT modify any existing code.
Run from PAVE_Merge root: python td3/tests/eval_q_supinf.py
"""
import sys, os, csv
import gymnasium as gym
import numpy as np
import torch as th

sys.path.append(os.path.join(os.path.dirname(__file__), "modules"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# For loading custom model classes (grad_td3, asap_td3, pave_td3)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.envs import (make_ant_env, make_hopper_env, make_lunar_env,
                           make_pendulum_env, make_reacher_env, make_walker_env)
from stable_baselines3 import TD3

# ── Config ──
MAX_EPISODES = 10
EVAL_SEED_FILE = "./sac/tests/validation_seeds.txt"  # for env.reset during eval
TRAIN_SEEDS = [178132, 410580, 922852, 787576, 660993]  # seeds #1-#5 (in folder names)
DEVICE = "cuda" if th.cuda.is_available() else "cpu"

SAVE_DIR_ROOT = "./Full/td3/pths/"
PTH_NAMES = ["base_td3", "caps_td3", "grad_td3", "asap_td3", "pave_td3"]
OUTPUT_DIR = "./Full/td3/pths/eval_q_supinf/"

ENVS = {
    "lunar": make_lunar_env,
    "pendulum": make_pendulum_env,
    "reacher": make_reacher_env,
    "ant": make_ant_env,
    "hopper": make_hopper_env,
    "walker": make_walker_env,
}

def load_eval_seeds(filepath):
    """Load seeds for env.reset during evaluation (NOT training seeds)."""
    with open(filepath, "r") as f:
        return [int(line.strip()) for line in f if line.strip()]

def find_matching_files(save_dir, al_name):
    """Same logic as q_extractor2.py"""
    if not os.path.isdir(save_dir):
        return []
    matched = []
    for fname in os.listdir(save_dir):
        full_path = os.path.join(save_dir, fname)
        if al_name in fname and os.path.isdir(full_path):
            final = os.path.join(full_path, "final.zip")
            if os.path.isfile(final):
                matched.append(os.path.join(fname, "final.zip"))
    return sorted(matched)  # sort for reproducibility

def measure_one_seed(model, env, seed, max_episodes):
    """Run max_episodes episodes, return per-timestep sa_norms and aa_traces."""
    sa_norms = []
    aa_traces = []

    obs, info = env.reset(seed=seed)
    for ep in range(max_episodes):
        while True:
            action, _ = model.predict(obs, deterministic=True)

            obs_tensor, _ = model.policy.obs_to_tensor(obs)
            action_tensor = th.as_tensor(action, device=obs_tensor.device).unsqueeze(0)
            obs_tensor.requires_grad_(True)
            action_tensor.requires_grad_(True)

            q1 = model.critic(obs_tensor, action_tensor)[0].sum()

            # ∂Q/∂a
            grad_a = th.autograd.grad(q1, action_tensor, create_graph=True, retain_graph=True)[0]

            # ||∇²_sa Q||₂ = spectral norm (max singular value)
            # Build full d_a × d_s mixed Hessian matrix
            action_dim = grad_a.shape[1]
            state_dim = obs_tensor.shape[1]
            H_sa = th.zeros(action_dim, state_dim, device=obs_tensor.device)
            for i in range(action_dim):
                grad_sa_i = th.autograd.grad(grad_a[0, i], obs_tensor,
                                              create_graph=True, retain_graph=True)[0]
                H_sa[i, :] = grad_sa_i[0, :]
            sa_norm = th.linalg.svdvals(H_sa)[0].item()  # largest singular value = spectral norm
            sa_norms.append(sa_norm)

            # tr(∇²_aa Q) = Σ_i ∂²Q/∂a_i²
            aa_trace = 0.0
            for i in range(grad_a.shape[1]):
                grad_aa_i = th.autograd.grad(grad_a[0, i], action_tensor,
                                              create_graph=True, retain_graph=True)[0]
                aa_trace += grad_aa_i[0, i].item()
            aa_traces.append(aa_trace)  # keep sign (negative = concave)

            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
                break

    return np.array(sa_norms), np.array(aa_traces)

def main():
    eval_seeds = load_eval_seeds(EVAL_SEED_FILE)  # for env.reset
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    env_names = list(ENVS.keys())

    # Result tables: M_sup, mu_inf, M_mean (for comparison), mu_mean
    headers = ["al_name"] + env_names
    rows_M_sup_mean = [headers]
    rows_M_sup_std = [headers]
    rows_mu_inf_mean = [headers]
    rows_mu_inf_std = [headers]
    rows_M_mean_mean = [headers]  # old-style mean for comparison
    rows_mu_mean_mean = [headers]

    for al_name in PTH_NAMES:
        print(f"\n=== {al_name} ===")
        row_M_sup_m = [al_name]
        row_M_sup_s = [al_name]
        row_mu_inf_m = [al_name]
        row_mu_inf_s = [al_name]
        row_M_mean_m = [al_name]
        row_mu_mean_m = [al_name]

        for env_name in env_names:
            save_dir = os.path.join(SAVE_DIR_ROOT, env_name, "")
            files = find_matching_files(save_dir, al_name)

            if not files:
                print(f"  {env_name}: no files found")
                for row in [row_M_sup_m, row_M_sup_s, row_mu_inf_m, row_mu_inf_s, row_M_mean_m, row_mu_mean_m]:
                    row.append(0.0)
                continue

            # Match files to training seeds by parsing seed from filename
            seed_to_file = {}
            for f in files:
                for s in TRAIN_SEEDS:
                    if str(s) in f:
                        seed_to_file[s] = f
                        break

            seed_M_sups = []   # per-seed sup of ||∇²_sa Q||
            seed_mu_infs = []  # per-seed inf of |tr(∇²_aa Q)|
            seed_M_means = []  # per-seed mean (for comparison)
            seed_mu_means = [] # per-seed mean (for comparison)

            env = ENVS[env_name]("rgb_array")()

            for si, train_seed in enumerate(TRAIN_SEEDS):
                if train_seed not in seed_to_file:
                    print(f"  WARNING: train seed {train_seed} not found for {al_name}/{env_name}")
                    continue
                filename = seed_to_file[train_seed]
                eval_seed = eval_seeds[si]  # use corresponding eval seed for env.reset
                model = TD3.load(os.path.join(save_dir, filename), env=env)

                sa_norms, aa_traces = measure_one_seed(model, env, eval_seed, MAX_EPISODES)

                # M: sup of ||∇²_sa Q||
                seed_M_sups.append(np.max(sa_norms))
                seed_M_means.append(np.mean(sa_norms))

                # μ: inf of |tr(∇²_aa Q)| — trace is negative when concave
                # |tr| gives magnitude, inf of magnitude = tightest curvature
                seed_mu_infs.append(np.min(np.abs(aa_traces)))
                seed_mu_means.append(np.mean(np.abs(aa_traces)))

            env.close()

            M_sup_mean = np.mean(seed_M_sups)
            M_sup_std = np.std(seed_M_sups)
            mu_inf_mean = np.mean(seed_mu_infs)
            mu_inf_std = np.std(seed_mu_infs)
            M_mean_mean = np.mean(seed_M_means)
            mu_mean_mean = np.mean(seed_mu_means)

            print(f"  {env_name}: M_sup={M_sup_mean:.1f}±{M_sup_std:.1f}, "
                  f"mu_inf={mu_inf_mean:.2f}±{mu_inf_std:.2f}, "
                  f"M_mean={M_mean_mean:.1f}, mu_mean={mu_mean_mean:.2f}")

            row_M_sup_m.append(M_sup_mean)
            row_M_sup_s.append(M_sup_std)
            row_mu_inf_m.append(mu_inf_mean)
            row_mu_inf_s.append(mu_inf_std)
            row_M_mean_m.append(M_mean_mean)
            row_mu_mean_m.append(mu_mean_mean)

        rows_M_sup_mean.append(row_M_sup_m)
        rows_M_sup_std.append(row_M_sup_s)
        rows_mu_inf_mean.append(row_mu_inf_m)
        rows_mu_inf_std.append(row_mu_inf_s)
        rows_M_mean_mean.append(row_M_mean_m)
        rows_mu_mean_mean.append(row_mu_mean_m)

    # Write CSVs
    for name, rows in [
        ("M_sup_mean.csv", rows_M_sup_mean),
        ("M_sup_std.csv", rows_M_sup_std),
        ("mu_inf_mean.csv", rows_mu_inf_mean),
        ("mu_inf_std.csv", rows_mu_inf_std),
        ("M_mean_mean.csv", rows_M_mean_mean),
        ("mu_mean_mean.csv", rows_mu_mean_mean),
    ]:
        path = os.path.join(OUTPUT_DIR, name)
        with open(path, "w", newline="") as f:
            csv.writer(f).writerows(rows)
        print(f"Saved: {path}")

    print("\nDone!")

if __name__ == "__main__":
    main()
