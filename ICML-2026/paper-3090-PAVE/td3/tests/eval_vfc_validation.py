"""
VFC Proxy Validation: Eq.13 vs Eq.14.
Proxy (Eq.13): ‖∇_a Q(s_{t+1}, a_t) - ∇_a Q(s_t, a_t)‖²
Actual (Eq.14): ‖∇²_sa Q(s_t, a_t) · Δs‖²  where Δs = s_{t+1} - s_t

PAVE only, 6 envs, 5 seeds, 10 episodes.
Run from PAVE_Merge root: python td3/tests/eval_vfc_validation.py
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

OUTPUT_DIR = "./Full/td3/pths/eval_vfc_validation/"

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

def compute_grad_a(model, obs_np, action):
    """Compute ∇_a Q(s, a) for given obs and action."""
    obs_t, _ = model.policy.obs_to_tensor(obs_np)
    act_t = th.as_tensor(action, device=obs_t.device).unsqueeze(0).requires_grad_(True)
    q1 = model.critic(obs_t, act_t)[0].sum()
    grad_a = th.autograd.grad(q1, act_t)[0].detach()
    return grad_a  # (1, d_a)

def measure_vfc_validation(model, env, eval_seed, max_episodes):
    """Compare VFC proxy (Eq.13) vs actual (Eq.14) at each transition."""
    results = []
    obs, info = env.reset(seed=eval_seed)

    for ep in range(max_episodes):
        t = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)

            # Step environment to get s_{t+1}
            obs_next, reward, terminated, truncated, info = env.step(action)
            delta_s = (obs_next - obs).astype(np.float32)

            # === Proxy (Eq.13): ‖∇_a Q(s_{t+1}, a_t) - ∇_a Q(s_t, a_t)‖² ===
            grad_a_curr = compute_grad_a(model, obs, action)       # ∇_a Q(s_t, a_t)
            grad_a_next = compute_grad_a(model, obs_next, action)  # ∇_a Q(s_{t+1}, a_t)
            proxy_vfc = ((grad_a_next - grad_a_curr) ** 2).sum().item()

            # === Actual (Eq.14): ‖∇²_sa Q · Δs‖² ===
            obs_t, _ = model.policy.obs_to_tensor(obs)
            act_t = th.as_tensor(action, device=obs_t.device).unsqueeze(0)
            obs_t.requires_grad_(True)
            act_t_gt = act_t.clone().detach().requires_grad_(True)

            q1 = model.critic(obs_t, act_t_gt)[0].sum()
            grad_a = th.autograd.grad(q1, act_t_gt, create_graph=True, retain_graph=True)[0]

            # Build full H_sa (d_a × d_s)
            action_dim = grad_a.shape[1]
            state_dim = obs_t.shape[1]
            H_sa = th.zeros(action_dim, state_dim, device=obs_t.device)
            for i in range(action_dim):
                grad_sa_i = th.autograd.grad(grad_a[0, i], obs_t,
                                              retain_graph=(i < action_dim - 1))[0]
                H_sa[i, :] = grad_sa_i[0, :]

            # H_sa · Δs
            delta_s_t = th.as_tensor(delta_s, device=obs_t.device)
            H_delta = H_sa @ delta_s_t  # (d_a,)
            actual_vfc = (H_delta ** 2).sum().item()

            results.append({
                'episode': ep,
                'timestep': t,
                'proxy_vfc': proxy_vfc,
                'actual_vfc': actual_vfc,
            })

            obs = obs_next
            t += 1
            if terminated or truncated:
                obs, info = env.reset()
                break

    return results

def main():
    eval_seeds = load_eval_seeds(EVAL_SEED_FILE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== VFC Proxy Validation (Eq.13 vs Eq.14) ===\n")

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

            seed_results = measure_vfc_validation(model, env, eval_seed, MAX_EPISODES)
            for r in seed_results:
                r['seed'] = train_seed
            all_results.extend(seed_results)

        env.close()

        if all_results:
            proxy_vals = [r['proxy_vfc'] for r in all_results]
            actual_vals = [r['actual_vfc'] for r in all_results]

            proxy_mean = np.mean(proxy_vals)
            actual_mean = np.mean(actual_vals)
            ratio = proxy_mean / actual_mean if actual_mean > 0 else float('inf')

            print(f"  {env_name}: proxy={proxy_mean:.4f}, actual={actual_mean:.4f}, "
                  f"ratio={ratio:.2f}, n={len(all_results)}")

            # Save CSV
            csv_path = os.path.join(OUTPUT_DIR, f"vfc_validation_{env_name}.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
                writer.writeheader()
                writer.writerows(all_results)

    print("\nDone!")

if __name__ == "__main__":
    main()
