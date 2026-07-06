"""
Job: During-Training Assumption Verification.
At each mid checkpoint, measures:
  1) M_sup (trajectory-wise max of ‖∇²_sa Q‖)
  2) Eigenvalue neg def rate (all eigenvalues < 0)
  3) Trace concavity rate (tr < -1.0, cross-check with B.8)

Base vs PAVE, Lunar + Walker, 5 seeds.
Run from PAVE_Merge root: python td3/tests/eval_during_training_full.py
"""
import sys, os, csv
import numpy as np
import torch as th

sys.path.append(os.path.join(os.path.dirname(__file__), "modules"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.envs import make_lunar_env, make_walker_env
from stable_baselines3 import TD3

MAX_EPISODES = 10
EVAL_SEED_FILE = "./sac/tests/validation_seeds.txt"
TRAIN_SEEDS = [178132, 410580, 922852, 787576, 660993]
DELTA = 1.0
DEVICE = "cuda" if th.cuda.is_available() else "cpu"

OUTPUT_DIR = "./Full/td3/pths/eval_during_training_full/"

CONFIGS = [
    {
        "env_name": "lunar",
        "env_func": make_lunar_env,
        "base_dir": "./Full/td3/pths/lunar/",
        "methods": ["base_td3", "pave_td3"],
    },
    {
        "env_name": "walker",
        "env_func": make_walker_env,
        "base_dir": "./Full/td3/pths/walker/",
        "methods": ["base_td3", "pave_td3"],
    },
]

def load_eval_seeds(filepath):
    with open(filepath, "r") as f:
        return [int(line.strip()) for line in f if line.strip()]

def measure_all(model, env, eval_seed, max_episodes):
    """Measure M_sup, neg_def_rate, trace_concavity at each timestep."""
    sa_norms = []
    neg_def_count = 0
    trace_concave_count = 0
    total_count = 0

    obs, info = env.reset(seed=eval_seed)

    for ep in range(max_episodes):
        while True:
            action, _ = model.predict(obs, deterministic=True)

            obs_tensor, _ = model.policy.obs_to_tensor(obs)
            action_tensor = th.as_tensor(action, device=obs_tensor.device).unsqueeze(0)
            obs_tensor.requires_grad_(True)
            action_tensor.requires_grad_(True)

            q1 = model.critic(obs_tensor, action_tensor)[0].sum()

            # ∂Q/∂a
            grad_a = th.autograd.grad(q1, action_tensor,
                                       create_graph=True, retain_graph=True)[0]

            # 1) ‖∇²_sa Q‖₂ = spectral norm (max singular value)
            action_dim_sa = action_tensor.shape[1]
            state_dim_sa = obs_tensor.shape[1]
            H_sa = th.zeros(action_dim_sa, state_dim_sa, device=obs_tensor.device)
            for i in range(action_dim_sa):
                grad_sa_i = th.autograd.grad(grad_a[0, i], obs_tensor,
                                              create_graph=True, retain_graph=True)[0]
                H_sa[i, :] = grad_sa_i[0, :]
            sa_norm = th.linalg.svdvals(H_sa)[0].item()
            sa_norms.append(sa_norm)

            # 2) Full eigenvalue decomposition
            action_dim = action_tensor.shape[1]
            hessian = th.zeros(action_dim, action_dim, device=action_tensor.device)
            for i in range(action_dim):
                grad2_i = th.autograd.grad(
                    grad_a[0, i], action_tensor,
                    retain_graph=True, create_graph=False
                )[0]
                hessian[i, :] = grad2_i[0, :]

            hessian = (hessian + hessian.T) / 2
            eigenvalues = th.linalg.eigvalsh(hessian)
            lambda_max = eigenvalues.max().item()

            if lambda_max < 0:
                neg_def_count += 1

            # 3) Trace concavity
            trace = float(eigenvalues.sum().item())
            if trace < -DELTA:
                trace_concave_count += 1

            total_count += 1

            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
                break

    return {
        'm_sup': np.max(sa_norms) if sa_norms else 0,
        'm_mean': np.mean(sa_norms) if sa_norms else 0,
        'neg_def_rate': neg_def_count / total_count if total_count > 0 else 0,
        'trace_conc_rate': trace_concave_count / total_count if total_count > 0 else 0,
        'total_count': total_count,
    }

def main():
    eval_seeds = load_eval_seeds(EVAL_SEED_FILE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== During-Training Full Assumption Verification ===\n")

    all_csv_rows = []

    for config in CONFIGS:
        env_name = config["env_name"]
        base_dir = config["base_dir"]
        env = config["env_func"]("rgb_array")()

        print(f"--- {env_name} ---")

        for method in config["methods"]:
            # Find checkpoint steps from seed #1
            seed1_dir = os.path.join(base_dir, f"{method}_{TRAIN_SEEDS[0]}")
            if not os.path.isdir(seed1_dir):
                print(f"  {method}: dir not found")
                continue

            step_list = []
            for f in sorted(os.listdir(seed1_dir)):
                if f.startswith("mid_") and f.endswith(".zip"):
                    step_str = f.replace("mid_", "").replace("_steps.zip", "")
                    try:
                        step = int(step_str)
                        if step > 0:
                            step_list.append(step)
                    except ValueError:
                        pass
            step_list.sort()

            print(f"  {method} (checkpoints: {step_list}):")

            for step in step_list:
                seed_m_sups = []
                seed_neg_def_rates = []
                seed_trace_conc_rates = []

                for si, train_seed in enumerate(TRAIN_SEEDS):
                    model_dir = os.path.join(base_dir, f"{method}_{train_seed}")
                    ckpt_path = os.path.join(model_dir, f"mid_{step:05d}_steps.zip")
                    if not os.path.exists(ckpt_path):
                        ckpt_path = os.path.join(model_dir, f"mid_{step}_steps.zip")
                    if not os.path.exists(ckpt_path):
                        continue

                    try:
                        model = TD3.load(ckpt_path, env=env)
                        eval_seed = eval_seeds[si]
                        result = measure_all(model, env, eval_seed, MAX_EPISODES)

                        seed_m_sups.append(result['m_sup'])
                        seed_neg_def_rates.append(result['neg_def_rate'])
                        seed_trace_conc_rates.append(result['trace_conc_rate'])
                    except Exception as e:
                        print(f"    ERROR seed {train_seed}: {e}")

                if seed_m_sups:
                    m_sup = np.mean(seed_m_sups)
                    m_sup_std = np.std(seed_m_sups)
                    neg_def = np.mean(seed_neg_def_rates)
                    neg_def_std = np.std(seed_neg_def_rates)
                    trace_conc = np.mean(seed_trace_conc_rates)
                    trace_conc_std = np.std(seed_trace_conc_rates)

                    print(f"    step={step:>8d}: M_sup={m_sup:.1f}±{m_sup_std:.1f}, "
                          f"neg_def={neg_def:.3f}±{neg_def_std:.3f}, "
                          f"trace_conc={trace_conc:.3f}±{trace_conc_std:.3f} "
                          f"({len(seed_m_sups)} seeds)")

                    all_csv_rows.append({
                        'env': env_name,
                        'method': method,
                        'step': step,
                        'm_sup_mean': round(m_sup, 1),
                        'm_sup_std': round(m_sup_std, 1),
                        'neg_def_mean': round(neg_def, 4),
                        'neg_def_std': round(neg_def_std, 4),
                        'trace_conc_mean': round(trace_conc, 4),
                        'trace_conc_std': round(trace_conc_std, 4),
                        'n_seeds': len(seed_m_sups),
                    })

        env.close()
        print()

    # Save CSV
    if all_csv_rows:
        csv_path = os.path.join(OUTPUT_DIR, "during_training_assumptions.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_csv_rows)
        print(f"Saved: {csv_path}")

    print("\nDone!")

if __name__ == "__main__":
    main()
