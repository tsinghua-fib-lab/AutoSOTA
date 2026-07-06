import glob
import os
import re
import sys

import gymnasium as gym
import imageio
import numpy as np
from scipy.io import loadmat

DATA_ROOT = "./Data"
OUTPUT_ROOT = "./Rendered"

DEFAULT_HIDDEN_SIZE = 16
DEFAULT_NUM_LAYERS = 2

TARGET_TASKS = range(18)
FIXED_SEEDS = [42, 123, 999]

ENV_NAMES = [
    "MountainCarContinuous-v0",
    "MountainCar-v0",
    "Pendulum-v1",
    "CartPole-v1",
    "Acrobot-v1",
    "LunarLander-v3",
    "BipedalWalker-v3",
    "InvertedPendulum-v5",
    "InvertedDoublePendulum-v5",
    "Reacher-v5",
    "Pusher-v5",
    "HalfCheetah-v5",
    "Hopper-v5",
    "Walker2d-v5",
    "Swimmer-v5",
    "Ant-v5",
    "Humanoid-v5",
    "HumanoidStandup-v5",
]


def sanitize_matlab_data(data):
    """
    Recursively cleans data loaded from MATLAB .mat files and converts it to a flattened float32 array.
    """
    if data is None:
        return None
    if isinstance(data, (list, tuple)):
        data = np.array(data)
    while isinstance(data, np.ndarray):
        if data.size == 0:
            return np.array([], dtype=np.float32)
        if data.dtype == "object" and data.size == 1:
            data = data.flat[0]
        elif data.ndim > 1 and (data.shape[0] == 1 or data.shape[1] == 1):
            data = data.flatten()
        else:
            break
    try:
        return np.ascontiguousarray(np.array(data).flatten().astype(np.float32))
    except:
        return np.array([], dtype=np.float32)


def get_clean_params(dec_matrix, task_idx):
    """
    Extracts policy parameters for a specific task from the decision matrix and removes NaN padding.
    """
    if task_idx >= dec_matrix.shape[0]:
        raise ValueError(f"Task index {task_idx} out of bounds.")
    raw_row = dec_matrix[task_idx]
    clean_row = raw_row[~np.isnan(raw_row)]
    return sanitize_matlab_data(clean_row)


def normalize_obs(obs, mean, std):
    """
    Applies Z-score normalization to observations using running statistics.
    """
    if mean is None or std is None or mean.size == 0:
        return obs
    return (obs - mean) / (std + 1e-8)


def mlp_forward_prop(obs, params, obs_dim, act_dim, hidden_size, num_layers):
    """
    Dynamic Multi-Layer Perceptron forward propagation based on flattened weight vectors.
    The parameter vector is parsed sequentially to reconstruct W and b for each layer.
    """
    obs_vec = obs.reshape(1, obs_dim).astype(np.float32)
    offset = 0

    b1_len = hidden_size
    b1 = params[offset : offset + b1_len].reshape(1, hidden_size)
    offset += b1_len

    middle_layers = []
    for _ in range(num_layers - 1):
        w_len = hidden_size * hidden_size
        W_mid = params[offset : offset + w_len].reshape(1, hidden_size, hidden_size)
        offset += w_len
        b_len = hidden_size
        b_mid = params[offset : offset + b_len].reshape(1, hidden_size)
        offset += b_len
        middle_layers.append((W_mid, b_mid))

    w_out_len = hidden_size * act_dim
    W_end = params[offset : offset + w_out_len].reshape(1, hidden_size, act_dim)
    offset += w_out_len

    b_out_len = act_dim
    b_end = params[offset : offset + b_out_len].reshape(1, act_dim)
    offset += b_out_len

    w1_len = obs_dim * hidden_size
    W1 = params[offset : offset + w1_len].reshape(1, obs_dim, hidden_size)
    offset += w1_len

    h = np.tanh(np.einsum("ni,nih->nh", obs_vec, W1) + b1)

    for W_mid, b_mid in middle_layers:
        h = np.tanh(np.einsum("nh,nhh->nh", h, W_mid) + b_mid)

    logits = np.einsum("nh,nha->na", h, W_end) + b_end
    return logits


def run_simulation(
    env_name, params, norm_mean, norm_std, hidden_size, num_layers, seed, render=False
):
    """
    Executes an episode in the Gymnasium environment using the evolved policy.
    Returns the cumulative reward and an array of RGB frames if rendering is enabled.
    """
    try:
        mode = "rgb_array" if render else None
        env = gym.make(env_name, render_mode=mode)

        obs_space = env.observation_space
        act_space = env.action_space
        is_discrete_act = isinstance(act_space, gym.spaces.Discrete)
        is_discrete_obs = isinstance(obs_space, gym.spaces.Discrete)
        obs_dim = obs_space.n if is_discrete_obs else obs_space.shape[0]
        act_dim = act_space.n if is_discrete_act else act_space.shape[0]

        scale, offset_a = 0, 0
        if not is_discrete_act:
            scale = (act_space.high - act_space.low) / 2
            offset_a = (act_space.high + act_space.low) / 2

        obs, _ = env.reset(seed=seed)
        total_reward = 0
        frames = []
        terminated = False
        truncated = False
        step_count = 0

        while not (terminated or truncated) and step_count < 1000:
            if render:
                frames.append(env.render())

            if is_discrete_obs:
                obs_in = np.eye(obs_dim)[obs].flatten().astype(np.float32)
            else:
                obs_f32 = obs.astype(np.float32)
                obs_in = normalize_obs(obs_f32, norm_mean, norm_std)

            logits = mlp_forward_prop(
                obs_in, params, obs_dim, act_dim, hidden_size, num_layers
            )

            if is_discrete_act:
                action = np.argmax(logits, axis=1)[0]
            else:
                a = np.tanh(logits)
                action_vec = a * scale + offset_a
                action_vec = np.clip(action_vec, act_space.low, act_space.high)
                action = action_vec[0]

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            step_count += 1

        env.close()
        return total_reward, frames
    except Exception as e:
        print(f"    Error in simulation: {e}")
        return -float("inf"), []


def process_algo_tasks(algo_name, dec_files):
    """
    Evaluates all trials for a given algorithm, identifies the best trial per task,
    and renders the top-performing policies into GIF files.
    """
    algo_out_dir = os.path.join(OUTPUT_ROOT, algo_name)
    os.makedirs(algo_out_dir, exist_ok=True)
    print(f"\nProcessing Algorithm: {algo_name} ({len(dec_files)} files)")

    loaded_trials = []
    for dec_path in dec_files:
        match = re.search(r"_Dec_(\d{8}_\d{6})\.mat$", dec_path)
        if not match:
            continue
        timestamp = match.group(1)
        norm_path = dec_path.replace(
            f"_Dec_{timestamp}.mat", f"_Normalizer_{timestamp}.mat"
        )
        if not os.path.exists(norm_path):
            continue

        try:
            dec_data = loadmat(dec_path)
            norm_data = loadmat(norm_path)
            loaded_trials.append(
                {
                    "ts": timestamp,
                    "dec": dec_data["Dec"],
                    "norm": norm_data["normalizer"],
                }
            )
        except:
            continue

    for task_idx in TARGET_TASKS:
        if task_idx >= len(ENV_NAMES):
            break
        env_name = ENV_NAMES[task_idx]
        print(f"\n> Task {task_idx}: {env_name}")

        best_avg_reward = -float("inf")
        best_trial_data = None

        for trial in loaded_trials:
            try:
                params = get_clean_params(trial["dec"], task_idx)
            except:
                continue

            mean, std = None, None
            hidden_size = DEFAULT_HIDDEN_SIZE
            num_layers = DEFAULT_NUM_LAYERS

            try:
                task_struct = trial["norm"][0, task_idx]

                if "mean" in task_struct.dtype.names:
                    mean = sanitize_matlab_data(task_struct["mean"])
                    std = sanitize_matlab_data(task_struct["std"])

                if "hiddenLayers" in task_struct.dtype.names:
                    hl_val = sanitize_matlab_data(task_struct["hiddenLayers"])
                    if hl_val.size > 0:
                        num_layers = int(hl_val.item())

                if "hiddenSize" in task_struct.dtype.names:
                    hs_val = sanitize_matlab_data(task_struct["hiddenSize"])
                    if hs_val.size > 0:
                        hidden_size = int(hs_val.item())
            except:
                pass

            rewards = []
            for seed in FIXED_SEEDS:
                r, _ = run_simulation(
                    env_name,
                    params,
                    mean,
                    std,
                    hidden_size,
                    num_layers,
                    seed,
                    render=False,
                )
                rewards.append(r)

            avg_r = np.mean(rewards)

            if avg_r > best_avg_reward:
                best_avg_reward = avg_r
                best_trial_data = {
                    "ts": trial["ts"],
                    "params": params,
                    "mean": mean,
                    "std": std,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                }

        if best_trial_data:
            print(
                f"  * Best Trial: {best_trial_data['ts']} (Avg Reward: {best_avg_reward:.2f})"
            )
            ts = best_trial_data["ts"]
            params = best_trial_data["params"]
            mean = best_trial_data["mean"]
            std = best_trial_data["std"]
            h_size = best_trial_data["hidden_size"]
            n_layers = best_trial_data["num_layers"]

            for seed in FIXED_SEEDS:
                r, frames = run_simulation(
                    env_name, params, mean, std, h_size, n_layers, seed, render=True
                )

                if len(frames) > 0:
                    clean_name = env_name.replace("/", "_")
                    gif_name = f"T{task_idx + 1}_{clean_name}_Best_{ts}_Seed{seed}_Rew{r:.0f}.gif"
                    gif_path = os.path.join(algo_out_dir, gif_name)
                    imageio.mimsave(gif_path, frames, fps=30, loop=0)
                    print(f"    Saved: {gif_name}")
        else:
            print("  No valid trials found for this task.")


if __name__ == "__main__":
    if not os.path.exists(DATA_ROOT):
        print("Data directory not found.")
        sys.exit()

    subfolders = [
        f for f in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, f))
    ]
    for algo in subfolders:
        folder_path = os.path.join(DATA_ROOT, algo)
        dec_files = glob.glob(os.path.join(folder_path, "*_Dec_*.mat"))
        if dec_files:
            process_algo_tasks(algo, dec_files)
