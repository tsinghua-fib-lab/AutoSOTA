"""
Concavity measurement module — follows q_extractor2.py structure exactly.
Measures tr(∇²_aa Q) and computes fraction satisfying tr < -δ.
"""
import sys
import os
import gymnasium as gym
import numpy as np
import csv
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from .envs import make_ant_env, make_hopper_env, make_humanoid_env, make_lunar_env, make_pendulum_env, make_reacher_env, make_walker_env

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from stable_baselines3 import TD3
import torch as th

max_loop = 10
seed_file_path = "./sac/tests/validation_seeds.txt"
DEVICE = "cuda" if th.cuda.is_available() else "cpu"
DELTA = 1.0  # curvature margin (same as training default)

def load_seeds(filepath):
    with open(filepath, "r") as f:
        return [int(line.strip()) for line in f if line.strip()]

train_envs_dict = dict({
    "ant" : make_ant_env,
    "hopper" : make_hopper_env,
    "humanoid" : make_humanoid_env,
    "lunar" : make_lunar_env,
    "pendulum" : make_pendulum_env,
    "reacher" : make_reacher_env,
    "walker" : make_walker_env
})

def find_matching_files(save_dir: str, al_name: str) -> list[str]:
    if not os.path.isdir(save_dir):
        return []
    matched_paths = []
    for fname in os.listdir(save_dir):
        full_dir_path = os.path.join(save_dir, fname)
        if al_name in fname:
            if os.path.isdir(full_dir_path):
                try:
                    subfiles = [f for f in os.listdir(full_dir_path) if f.endswith(".zip")]
                except OSError: continue
                target_zip = None
                if "final.zip" in subfiles: target_zip = "final.zip"
                elif "best_model.zip" in subfiles: target_zip = "best_model.zip"
                elif subfiles: target_zip = subfiles[0]
                if target_zip:
                    matched_paths.append(os.path.join(fname, target_zip))
            elif os.path.isfile(full_dir_path) and fname.endswith(".zip"):
                matched_paths.append(fname)
    return matched_paths


def test_concavity(root_dir, al_name, env_name, deterministic=True, mode="rgb_array"):
    """
    Returns: (satisfy_rate, trace_mean, trace_std, total_steps)
    satisfy_rate = fraction of steps where tr(∇²_aa Q) < -DELTA
    """
    try:
        seeds = load_seeds(seed_file_path)
        counter = 0
        if env_name not in train_envs_dict:
            print(f"invalid env : {env_name}")
            return 0.0, 0.0, 0.0, 0

        save_dir = root_dir + env_name + "/"
        env = train_envs_dict[env_name](mode)()
        files = find_matching_files(save_dir, al_name)

        trace_list = []

        for filename in files:
            model = TD3.load(f"{save_dir}{filename}", env=env)
            obs, info = env.reset(seed=seeds[counter])
            counter += 1
            for _ in range(max_loop):
                while True:
                    action, _states = model.predict(obs, deterministic=deterministic)

                    obs_tensor, _ = model.policy.obs_to_tensor(obs)
                    action_tensor = th.as_tensor(action, device=obs_tensor.device).unsqueeze(0)
                    action_tensor.requires_grad_(True)

                    q1_pred, q2_pred = model.critic(obs_tensor, action_tensor)
                    q1 = q1_pred.sum()

                    # ∂Q / ∂a
                    grad_a = th.autograd.grad(
                        q1, action_tensor,
                        create_graph=True, retain_graph=True
                    )[0]

                    # tr(∇²_aa Q) via exact diagonal
                    trace_sum = 0.0
                    for i in range(grad_a.shape[1]):
                        grad_aa_i = th.autograd.grad(
                            grad_a[0, i], action_tensor,
                            create_graph=True, retain_graph=True
                        )[0]
                        trace_sum += grad_aa_i[0, i].item()

                    trace_list.append(trace_sum)  # keep sign!

                    obs, reward, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        obs, info = env.reset(seed=seeds[counter])
                        counter += 1
                        break

        traces = np.array(trace_list)
        satisfy_rate = float((traces < -DELTA).mean())
        trace_mean = float(traces.mean())
        trace_std = float(traces.std())

        return satisfy_rate, trace_mean, trace_std, len(traces)

    except Exception as e:
        print(f"ERROR ({al_name}, {env_name}): {e}")
        import traceback; traceback.print_exc()
        return 0.0, 0.0, 0.0, 0


def test_some_path_concavity(root_dir, deterministic=True, add_al_names: list[str] = [], sub_dir=""):
    basic_al = []
    basic_envs = list(train_envs_dict.keys())
    all_al = basic_al + add_al_names

    satisfy_rows = [["al_name"] + basic_envs]
    trace_mean_rows = [["al_name"] + basic_envs]
    trace_std_rows = [["al_name"] + basic_envs]

    combined_path = os.path.join(root_dir, sub_dir)
    os.makedirs(combined_path, exist_ok=True)

    print("Testing concavity satisfaction (tr(∇²_aa Q) < -δ)...")

    for al_name in all_al:
        satisfy_row = [al_name]
        trace_mean_row = [al_name]
        trace_std_row = [al_name]

        for env_name in basic_envs:
            print(f"  [{al_name}] {env_name} ...", end=" ", flush=True)
            sat, t_mean, t_std, n_steps = test_concavity(root_dir, al_name, env_name, deterministic)
            print(f"satisfy={sat:.3f}, trace={t_mean:.2f}±{t_std:.2f}, steps={n_steps}")

            satisfy_row.append(sat)
            trace_mean_row.append(t_mean)
            trace_std_row.append(t_std)

        satisfy_rows.append(satisfy_row)
        trace_mean_rows.append(trace_mean_row)
        trace_std_rows.append(trace_std_row)

    for fname, rows in [
        ("concavity_satisfy.csv", satisfy_rows),
        ("concavity_trace_mean.csv", trace_mean_rows),
        ("concavity_trace_std.csv", trace_std_rows),
    ]:
        path = os.path.join(combined_path, fname)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for row in rows:
                writer.writerow(row)
        print(f"Saved: {path}")
