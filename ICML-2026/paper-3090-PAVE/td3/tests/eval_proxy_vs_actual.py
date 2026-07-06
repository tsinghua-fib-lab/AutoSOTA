"""
Job 2: Proxy vs Actual Hessian pointwise comparison.
PAVE only, 6 envs, 5 seeds, n_epsilon=5 averaging.
Computes scaled_proxy (= proxy/σ²) vs actual (= ‖grad_sa‖²).

Run from PAVE_Merge root: python td3/tests/eval_proxy_vs_actual.py
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
SIGMA = 0.01
N_EPSILON = 5
DEVICE = "cuda" if th.cuda.is_available() else "cpu"

OUTPUT_DIR = "./Full/td3/pths/eval_proxy_vs_actual/"

CONFIGS = {
    "lunar": {"func": make_lunar_env, "dir": "./Full/td3/pths/lunar/"},
    "pendulum": {"func": make_pendulum_env, "dir": "./Full/td3/pths/pendulum/"},
    "reacher": {"func": make_reacher_env, "dir": "./Full/td3/pths/reacher/"},
    "ant": {"func": make_ant_env, "dir": "./Full/td3/pths/ant/"},
    "hopper": {"func": make_hopper_env, "dir": "./Full/td3/pths/hopper/"},
    "walker": {"func": make_walker_env, "dir": "./Full/td3/pths/walker/"},
}

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

def measure_proxy_vs_actual(model, env, eval_seed, max_episodes, sigma, n_epsilon):
    """Compute proxy and actual at each (s,a), with n_epsilon averaging for proxy."""
    results = []

    obs, info = env.reset(seed=eval_seed)

    for ep in range(max_episodes):
        t = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)

            # === Actual: ‖∇²_sa Q‖²_F (true Frobenius norm) ===
            obs_tensor, _ = model.policy.obs_to_tensor(obs)
            action_tensor = th.as_tensor(action, device=obs_tensor.device).unsqueeze(0)
            obs_tensor.requires_grad_(True)
            action_tensor_gt = action_tensor.clone().detach().requires_grad_(True)

            q1 = model.critic(obs_tensor, action_tensor_gt)[0].sum()
            grad_a = th.autograd.grad(q1, action_tensor_gt, create_graph=True, retain_graph=True)[0]

            # Frobenius: Σ_i ‖∂(∂Q/∂aᵢ)/∂s‖² — iterate over each action dim
            action_dim = grad_a.shape[1]
            actual = 0.0
            for i in range(action_dim):
                grad_sa_i = th.autograd.grad(grad_a[0, i], obs_tensor,
                                              retain_graph=(i < action_dim - 1))[0]
                actual += (grad_sa_i ** 2).sum().item()

            # === Proxy: average over n_epsilon samples ===
            proxy_values = []
            obs_np = obs.copy()
            for _ in range(n_epsilon):
                epsilon = np.random.randn(*obs_np.shape).astype(np.float32) * sigma
                obs_pert = obs_np + epsilon

                # ∇_a Q at (s, a)
                obs_t1, _ = model.policy.obs_to_tensor(obs_np)
                act_t1 = th.as_tensor(action, device=obs_t1.device).unsqueeze(0).requires_grad_(True)
                q1_orig = model.critic(obs_t1, act_t1)[0].sum()
                grad_a_orig = th.autograd.grad(q1_orig, act_t1)[0].detach()

                # ∇_a Q at (s+ε, a)
                obs_t2, _ = model.policy.obs_to_tensor(obs_pert)
                act_t2 = th.as_tensor(action, device=obs_t2.device).unsqueeze(0).requires_grad_(True)
                q1_pert = model.critic(obs_t2, act_t2)[0].sum()
                grad_a_pert = th.autograd.grad(q1_pert, act_t2)[0].detach()

                proxy_val = ((grad_a_pert - grad_a_orig) ** 2).sum().item()
                proxy_values.append(proxy_val)

            proxy_mean = np.mean(proxy_values)
            scaled_proxy = proxy_mean / (sigma ** 2)

            results.append({
                'episode': ep,
                'timestep': t,
                'proxy_raw': proxy_mean,
                'scaled_proxy': scaled_proxy,
                'actual': actual,
            })

            obs, reward, terminated, truncated, info = env.step(action)
            t += 1
            if terminated or truncated:
                obs, info = env.reset()
                break

    return results

def main():
    eval_seeds = load_eval_seeds(EVAL_SEED_FILE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"=== Proxy vs Actual (PAVE, σ={SIGMA}, n_ε={N_EPSILON}) ===\n")

    from scipy.stats import spearmanr

    for env_name, cfg in CONFIGS.items():
        env = cfg["func"]("rgb_array")()
        save_dir = cfg["dir"]
        method = "pave_td3"

        all_results = []

        for si, train_seed in enumerate(TRAIN_SEEDS):
            filename = find_seed_file(save_dir, method, train_seed)
            if filename is None:
                print(f"  WARNING: {env_name} seed {train_seed} not found")
                continue

            eval_seed = eval_seeds[si]
            model = TD3.load(os.path.join(save_dir, filename), env=env)

            seed_results = measure_proxy_vs_actual(
                model, env, eval_seed, MAX_EPISODES, SIGMA, N_EPSILON
            )
            for r in seed_results:
                r['seed'] = train_seed
            all_results.extend(seed_results)

        env.close()

        if all_results:
            scaled = [r['scaled_proxy'] for r in all_results]
            actual = [r['actual'] for r in all_results]
            rho, pval = spearmanr(scaled, actual)

            rel_errors = [abs(s - a) / (a + 1e-10) for s, a in zip(scaled, actual)]
            median_rel_err = np.median(rel_errors)

            print(f"  {env_name}: Spearman ρ={rho:.3f}, p={pval:.1e}, "
                  f"median_rel_err={median_rel_err:.3f}, n={len(all_results)}")

            # Save CSV
            csv_path = os.path.join(OUTPUT_DIR, f"proxy_vs_actual_{env_name}.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
                writer.writeheader()
                writer.writerows(all_results)

    print("\nDone!")

if __name__ == "__main__":
    main()
