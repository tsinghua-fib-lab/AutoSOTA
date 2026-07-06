"""
Hessian proxy validation — compare Eq.11 (finite-difference) vs Eq.12 (exact Hessian).
At the same (s,a) points:
  Proxy:  ||∇_a Q(s+ε, a) - ∇_a Q(s, a)||²
  Truth:  σ² ||∇²_sa Q||²_F

Reports Pearson correlation + saves paired values for scatter plot.
Base vs PAVE, lunar + walker, 5 seeds.

Run from PAVE_Merge root: python td3/tests/eval_hessian_proxy.py
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
SIGMA = 0.01  # same as PAVE training
DEVICE = "cuda" if th.cuda.is_available() else "cpu"

OUTPUT_DIR = "./Full/td3/pths/eval_hessian_proxy/"

CONFIGS = {
    "lunar": {"func": make_lunar_env, "dir": "./Full/td3/pths/lunar/"},
    "pendulum": {"func": make_pendulum_env, "dir": "./Full/td3/pths/pendulum/"},
    "reacher": {"func": make_reacher_env, "dir": "./Full/td3/pths/reacher/"},
    "ant": {"func": make_ant_env, "dir": "./Full/td3/pths/ant/"},
    "hopper": {"func": make_hopper_env, "dir": "./Full/td3/pths/hopper/"},
    "walker": {"func": make_walker_env, "dir": "./Full/td3/pths/walker/"},
}
METHODS = ["base_td3"]

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

def measure_proxy_vs_truth(model, env, eval_seed, max_episodes, sigma):
    """At each (s,a), compute proxy and ground truth."""
    proxy_vals = []
    truth_vals = []

    obs, info = env.reset(seed=eval_seed)

    for ep in range(max_episodes):
        while True:
            action, _ = model.predict(obs, deterministic=True)

            obs_tensor, _ = model.policy.obs_to_tensor(obs)
            action_tensor = th.as_tensor(action, device=obs_tensor.device).unsqueeze(0)

            # === Ground truth: σ² ||∇²_sa Q||²_F ===
            obs_tensor.requires_grad_(True)
            action_tensor_gt = action_tensor.clone().detach().requires_grad_(True)

            q1_gt = model.critic(obs_tensor, action_tensor_gt)[0].sum()
            grad_a = th.autograd.grad(q1_gt, action_tensor_gt, create_graph=True, retain_graph=True)[0]

            # ∇²_sa Q = ∂(∇_a Q)/∂s — full Jacobian
            grad_sa = th.autograd.grad(grad_a.sum(), obs_tensor, retain_graph=True)[0]
            hsa_frob_sq = (grad_sa ** 2).sum().item()  # ||∇²_sa Q||²_F
            truth = sigma ** 2 * hsa_frob_sq

            # === Proxy: ||∇_a Q(s+ε, a) - ∇_a Q(s, a)||² ===
            obs_np = obs.copy()
            epsilon = np.random.randn(*obs_np.shape) * sigma
            obs_pert = obs_np + epsilon

            # Q-gradient at original s
            obs_t1, _ = model.policy.obs_to_tensor(obs_np)
            act_t1 = th.as_tensor(action, device=obs_t1.device).unsqueeze(0).requires_grad_(True)
            q1_orig = model.critic(obs_t1, act_t1)[0].sum()
            grad_a_orig = th.autograd.grad(q1_orig, act_t1)[0].detach()

            # Q-gradient at perturbed s+ε
            obs_t2, _ = model.policy.obs_to_tensor(obs_pert)
            act_t2 = th.as_tensor(action, device=obs_t2.device).unsqueeze(0).requires_grad_(True)
            q1_pert = model.critic(obs_t2, act_t2)[0].sum()
            grad_a_pert = th.autograd.grad(q1_pert, act_t2)[0].detach()

            proxy = ((grad_a_pert - grad_a_orig) ** 2).sum().item()

            proxy_vals.append(proxy)
            truth_vals.append(truth)

            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
                break

    return np.array(proxy_vals), np.array(truth_vals)

def main():
    eval_seeds = load_eval_seeds(EVAL_SEED_FILE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"=== Hessian Proxy Validation (σ={SIGMA}) ===\n")

    for env_name, cfg in CONFIGS.items():
        env = cfg["func"]("rgb_array")()
        save_dir = cfg["dir"]

        for method in METHODS:
            all_proxy = []
            all_truth = []

            for si, train_seed in enumerate(TRAIN_SEEDS):
                filename = find_seed_file(save_dir, method, train_seed)
                if filename is None:
                    print(f"  WARNING: {method}/{env_name} seed {train_seed} not found")
                    continue

                model = TD3.load(os.path.join(save_dir, filename), env=env)
                eval_seed = eval_seeds[si]

                proxy, truth = measure_proxy_vs_truth(model, env, eval_seed, MAX_EPISODES, SIGMA)
                all_proxy.extend(proxy)
                all_truth.extend(truth)

            all_proxy = np.array(all_proxy)
            all_truth = np.array(all_truth)

            # Pearson correlation
            if len(all_proxy) > 10:
                corr = np.corrcoef(all_proxy, all_truth)[0, 1]
                ratio_mean = np.mean(all_proxy / (all_truth + 1e-10))
                print(f"  {method:10s} {env_name:8s}: Pearson r={corr:.4f}, "
                      f"mean(proxy/truth)={ratio_mean:.3f}, "
                      f"n={len(all_proxy)} points")

                # Save paired values for scatter plot
                out_path = os.path.join(OUTPUT_DIR, f"proxy_vs_truth_{method}_{env_name}.csv")
                with open(out_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["proxy", "truth"])
                    for p, t in zip(all_proxy, all_truth):
                        writer.writerow([p, t])

        env.close()
        print()

    print("Done!")

if __name__ == "__main__":
    main()
