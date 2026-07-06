import time

import gymnasium as gym
import numpy as np

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


# ---------------------------------------------------------
# Helper: Welford Online Update (Numpy Version)
# ---------------------------------------------------------
def welford_update_numpy(current_stats, new_batch_obs):
    """
    Welford's Online Algorithm:
    Updates running statistics (count, mean, M2) incrementally with a new batch
    of data, without storing the full history.
    M2 = sum((x - mean)^2).

    Args:
        current_stats: tuple (n, mean, m2) - Accumulated stats so far.
        new_batch_obs: numpy array (batch_size, obs_dim) - New observations.
    """
    (n_a, mean_a, m2_a) = current_stats

    n_b = float(new_batch_obs.shape[0])
    if n_b == 0:
        return current_stats

    # Calculate local stats for the new batch
    mean_b = np.mean(new_batch_obs, axis=0)
    m2_b = np.sum((new_batch_obs - mean_b) ** 2, axis=0)

    n_new = n_a + n_b

    # Handle first update initialization
    if n_a < 1e-6:
        return (n_new, mean_b, m2_b)

    # Welford Merge Formula (Numerically stable)
    delta = mean_b - mean_a
    mean_new = mean_a + delta * (n_b / n_new)
    m2_new = m2_a + m2_b + (delta**2) * (n_a * n_b / n_new)

    return (n_new, mean_new, m2_new)


def run_episode_single(env, params, hidden_size, num_layers, mean, std, seed=None):
    """
    Execute a single episode (Single-threaded).

    Args:
        seed: Random seed for environment reset (optional here, usually handled by caller).

    [Key Parameter Layout]:
    Structure: b1 -> [W_mid, b_mid]... -> [Wend, bend] -> W1 (End)
    """
    obs_space = env.single_observation_space
    act_space = env.single_action_space
    obs_is_discrete = isinstance(obs_space, gym.spaces.Discrete)
    act_is_discrete = isinstance(act_space, gym.spaces.Discrete)
    obs_dim = obs_space.n if obs_is_discrete else obs_space.shape[0]
    act_dim = act_space.n if act_is_discrete else act_space.shape[0]
    max_steps = getattr(env.spec, "max_episode_steps", None) or 1000

    mean = np.array(mean).reshape(1, -1)
    std = np.array(std).reshape(1, -1)
    safe_std = std + 1e-8

    # --- Parameter Unpacking ---
    offset = 0

    # 1. b1 (Layer 1 Bias)
    b1 = params[offset : offset + hidden_size].reshape(1, hidden_size)
    offset += hidden_size

    # 2. Middle Layers (Dynamic Parsing)
    middle_layers = []
    for _ in range(num_layers - 1):
        # W_mid: [hidden, hidden]
        w_size = hidden_size * hidden_size
        W_mid = params[offset : offset + w_size].reshape(1, hidden_size, hidden_size)
        offset += w_size

        # b_mid: [hidden]
        b_size = hidden_size
        b_mid = params[offset : offset + b_size].reshape(1, hidden_size)
        offset += b_size

        middle_layers.append((W_mid, b_mid))

    # 3. Output Layer (Hidden -> Action)
    w_out_size = hidden_size * act_dim
    W_end = params[offset : offset + w_out_size].reshape(1, hidden_size, act_dim)
    offset += w_out_size

    b_out_size = act_dim
    b_end = params[offset : offset + b_out_size].reshape(1, act_dim)
    offset += b_out_size

    # 4. Input Layer Weights "W1" (Obs -> Hidden) -> Placed at the END
    w1_size = obs_dim * hidden_size
    W1 = params[offset : offset + w1_size].reshape(1, obs_dim, hidden_size)
    offset += w1_size

    if not act_is_discrete:
        scale = (act_space.high - act_space.low) / 2
        offset_a = (act_space.high + act_space.low) / 2

    # Reset environment with seed
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    episode_obs = []

    # --- Simulation Loop ---
    for _ in range(max_steps):
        if obs_is_discrete:
            obs_vec = np.eye(obs_dim)[obs].reshape(1, obs_dim)
        else:
            obs_raw = obs.reshape(1, obs_dim)
            episode_obs.append(obs_raw)
            # Apply Z-Score Normalization
            obs_vec = (obs_raw - mean) / safe_std

        # Forward Pass (MLP)
        # 1. Input -> Hidden 1
        x = np.tanh(np.einsum("ni,nih->nh", obs_vec, W1) + b1)

        # 2. Middle Layers
        for W_mid, b_mid in middle_layers:
            x = np.tanh(np.einsum("nh,nhh->nh", x, W_mid) + b_mid)

        # 3. Output Layer
        logits = np.einsum("nh,nha->na", x, W_end) + b_end

        if act_is_discrete:
            # Probabilistic action selection
            logits -= np.max(logits, axis=1, keepdims=True)
            prob = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
            # Use np.random (seeded globally by caller)
            action = [np.random.choice(act_dim, p=prob[0])]

            # Deterministic action selection
            # action = np.argmax(logits, axis=1)
        else:
            a = np.tanh(logits)
            action = np.clip(a * scale + offset_a, act_space.low, act_space.high)

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward[0]
        if terminated or truncated:
            break

    # Stack observations for statistical update
    if len(episode_obs) > 0:
        stacked_obs = np.vstack(episode_obs)
    else:
        stacked_obs = np.zeros((0, obs_dim))

    # NOTE: Do NOT close env here if using SyncVectorEnv in Sequential mode wrapper
    return total_reward, stacked_obs


def run_episode_seq(
    env_idx, params, hidden_size, num_layers, num_rollouts, mean, std, seed
):
    """
    Sequential Execution:
    Used for environments that do not support or are unstable with vectorization.
    """
    # Set Global Random Seed (Controls np.random.choice/uniform)
    np.random.seed(seed)

    env_name = ENV_NAMES[env_idx]
    env = gym.vector.SyncVectorEnv([(lambda: gym.make(env_name))])

    n_params = params.shape[0]
    avg_rewards = np.zeros(n_params, dtype=float)

    # Initialize local statistics
    obs_dim = mean.shape[0] if len(mean.shape) > 0 else len(mean)
    run_n = 0.0
    run_mean = np.zeros(obs_dim)
    run_m2 = np.zeros(obs_dim)

    for i in range(n_params):
        total_reward = 0.0
        for r in range(num_rollouts):
            # Generate a deterministic unique seed for each rollout
            # e.g., base_seed + param_index * 100 + rollout_index
            current_seed = seed + (i * num_rollouts) + r

            reward, ep_obs = run_episode_single(
                env, params[i], hidden_size, num_layers, mean, std, seed=current_seed
            )
            total_reward += reward

            # Update stats with data from this episode
            if ep_obs.shape[0] > 0:
                (run_n, run_mean, run_m2) = welford_update_numpy(
                    (run_n, run_mean, run_m2), ep_obs
                )

        avg_rewards[i] = total_reward / num_rollouts

    env.close()
    return [avg_rewards.tolist(), float(run_n), run_mean.tolist(), run_m2.tolist()]


def run_episode_vec(
    env_idx, params, hidden_size, num_layers, num_rollouts, mean, std, seed
):
    """
    Vectorized Execution:
    Evaluates multiple environment instances in parallel for maximum throughput.
    """
    # Set Global Random Seed
    np.random.seed(seed)

    env_name = ENV_NAMES[env_idx]
    n_params = params.shape[0]
    # Repeat params to match rollout count
    repeated = np.repeat(params, num_rollouts, axis=0)
    n_envs = repeated.shape[0]

    env = gym.vector.SyncVectorEnv(
        [(lambda env_=env_name: lambda: gym.make(env_))() for _ in range(n_envs)]
    )

    # ... (Environment space retrieval) ...
    obs_space = env.single_observation_space
    act_space = env.single_action_space
    obs_is_discrete = isinstance(obs_space, gym.spaces.Discrete)
    act_is_discrete = isinstance(act_space, gym.spaces.Discrete)
    obs_dim = obs_space.n if obs_is_discrete else obs_space.shape[0]
    act_dim = act_space.n if act_is_discrete else act_space.shape[0]
    max_steps = getattr(env.spec, "max_episode_steps", None) or 1000

    mean = np.array(mean).reshape(1, -1)
    std = np.array(std).reshape(1, -1)
    safe_std = std + 1e-8

    run_n = 0.0
    run_mean = np.zeros(obs_dim)
    run_m2 = np.zeros(obs_dim)

    # --- Vectorized Parameter Unpacking ---
    offset = 0
    # 1. b1
    b1 = repeated[:, offset : offset + hidden_size]
    offset += hidden_size
    # 2. Middle Layers
    middle_layers = []
    for _ in range(num_layers - 1):
        w_size = hidden_size * hidden_size
        W_mid = repeated[:, offset : offset + w_size].reshape(
            n_envs, hidden_size, hidden_size
        )
        offset += w_size
        b_size = hidden_size
        b_mid = repeated[:, offset : offset + b_size]
        offset += b_size
        middle_layers.append((W_mid, b_mid))
    # 3. Output Layer
    w_out_size = hidden_size * act_dim
    W_end = repeated[:, offset : offset + w_out_size].reshape(
        n_envs, hidden_size, act_dim
    )
    offset += w_out_size
    b_out_size = act_dim
    b_end = repeated[:, offset : offset + b_out_size]
    offset += b_out_size
    # 4. Input Layer (W1)
    w1_size = obs_dim * hidden_size
    W1 = repeated[:, offset : offset + w1_size].reshape(n_envs, obs_dim, hidden_size)
    offset += w1_size

    # Reset with Seed (SyncVectorEnv automatically distributes seed+i to sub-envs)
    obs, _ = env.reset(seed=seed)

    total_rewards = np.zeros(n_envs, dtype=float)
    done_flags = np.zeros(n_envs, dtype=bool)

    if not act_is_discrete:
        scale = (act_space.high - act_space.low) / 2
        offset_a = (act_space.high + act_space.low) / 2

    for _ in range(max_steps):
        if np.all(done_flags):
            break
        active_mask = ~done_flags

        if obs_is_discrete:
            obs_vec = np.zeros((n_envs, obs_dim))
            obs_vec[active_mask] = np.eye(obs_dim)[obs[active_mask]]
        else:
            active_obs = obs[active_mask]
            if active_obs.shape[0] > 0:
                (run_n, run_mean, run_m2) = welford_update_numpy(
                    (run_n, run_mean, run_m2), active_obs
                )
            obs_vec = np.zeros_like(obs)
            obs_vec[active_mask] = (obs[active_mask] - mean) / safe_std

        # Forward Pass
        h = np.tanh(np.einsum("ni,nih->nh", obs_vec, W1) + b1)
        for W_mid, b_mid in middle_layers:
            h = np.tanh(np.einsum("nh,nhh->nh", h, W_mid) + b_mid)
        logits = np.einsum("nh,nha->na", h, W_end) + b_end

        if act_is_discrete:
            # Probabilistic action selection
            logits -= np.max(logits, axis=1, keepdims=True)
            probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

            # Inverse transform sampling
            r = np.random.rand(n_envs, 1)
            cum_probs = np.cumsum(probs, axis=1)
            # argmax finds the first index where cum_probs > r
            actions = np.argmax(cum_probs > r, axis=1)
            actions[~active_mask] = 0

            # Deterministic action selection
            # actions = np.argmax(logits, axis=1)
        else:
            a = np.tanh(logits)
            actions = a * scale + offset_a
            actions = np.clip(actions, act_space.low, act_space.high)
            actions[~active_mask] = 0.0

        obs_next, rewards, terminateds, truncateds, _ = env.step(actions)
        total_rewards += rewards * active_mask
        done_flags = done_flags | terminateds | truncateds
        obs = obs_next

    env.close()
    grouped = total_rewards.reshape(n_params, num_rollouts)
    final_scores = np.mean(grouped, axis=1).tolist()
    return [final_scores, float(run_n), run_mean.tolist(), run_m2.tolist()]


def run_episode(
    env_idx, params, hidden_size, num_rollouts, mean, std, num_layers=2, seed=42
):
    """
    Standard Interface called by MATLAB.
    Added 'seed' parameter to control randomness from MATLAB.
    """
    env_idx = int(env_idx)
    hidden_size = int(hidden_size)
    num_rollouts = int(num_rollouts)
    num_layers = int(num_layers)
    seed = int(seed)

    if env_idx in [3, 5, 6, 7, 8, 9, 12, 13, 16]:
        return run_episode_seq(
            env_idx, params, hidden_size, num_layers, num_rollouts, mean, std, seed
        )
    elif env_idx in [0, 1, 2, 4, 10, 11, 14, 15, 17]:
        return run_episode_vec(
            env_idx, params, hidden_size, num_layers, num_rollouts, mean, std, seed
        )
    else:
        raise ValueError(f"Invalid environment index: {env_idx}")


def compute_utilities(rewards):
    """SNES Helper: Transform raw rewards into rank-based utilities."""
    rewards = np.array(rewards)
    n = len(rewards)
    ranks = np.argsort(np.argsort(rewards))
    utilities = ranks / (n - 1) - 0.5
    return utilities


def snes_loop(
    env_idx,
    mode,
    generations,
    pop_size,
    param_dim,
    hidden_size,
    num_layers,
    mean_obs,
    std_obs,
):
    """
    Test Loop:
    Uses a fixed internal seed for reproducibility in standalone tests.
    """
    obs_dim = len(mean_obs)
    fixed_part = (
        hidden_size
        + (hidden_size * hidden_size + hidden_size) * (num_layers - 1)
        + obs_dim * hidden_size
    )
    act_dim = (param_dim - fixed_part) // (hidden_size + 1)

    # Init weights (LeCun)
    mu_parts = []
    mu_parts.append(np.zeros(hidden_size))  # b1
    for _ in range(num_layers - 1):
        limit_mid = np.sqrt(3 / hidden_size)
        mu_parts.append(
            np.random.uniform(-limit_mid, limit_mid, size=hidden_size * hidden_size)
        )
        mu_parts.append(np.zeros(hidden_size))
    limit_out = np.sqrt(3 / hidden_size)
    mu_parts.append(
        np.random.uniform(-limit_out, limit_out, size=hidden_size * act_dim)
    )
    mu_parts.append(np.zeros(act_dim))  # bend
    limit_in = np.sqrt(3 / obs_dim)
    mu_parts.append(
        np.random.uniform(-limit_in, limit_in, size=obs_dim * hidden_size)
    )  # W1

    mu = np.concatenate(mu_parts)

    if env_idx == 2:  # Pendulum-v1
        sigma = np.ones(param_dim) * 1.0
    elif env_idx <= 6:  # Classic Control & Box2D
        sigma = np.ones(param_dim) * 0.5
    elif env_idx == 15:  # Ant-v5
        sigma = np.ones(param_dim) * 0.15
    else:
        sigma = np.ones(param_dim) * 0.3

    learning_rate = 0.1

    test_n = 1e-4
    test_mean = np.zeros(obs_dim)
    test_m2 = np.zeros(obs_dim)
    test_std = np.ones(obs_dim)

    # Fixed seed for standalone test consistency
    base_seed = 100

    start_time = time.time()

    for gen in range(generations):
        # Generate new seed for each generation
        current_gen_seed = base_seed + gen

        # Reset numpy seed for parameter sampling
        np.random.seed(current_gen_seed)

        z = np.random.randn(pop_size, param_dim)
        params = mu + sigma * z

        if mode == "seq":
            result = run_episode_seq(
                env_idx,
                params,
                hidden_size,
                num_layers,
                3,
                test_mean,
                test_std,
                current_gen_seed,
            )
        else:
            result = run_episode_vec(
                env_idx,
                params,
                hidden_size,
                num_layers,
                3,
                test_mean,
                test_std,
                current_gen_seed,
            )

        rewards = result[0]
        batch_n = result[1]
        batch_mean = np.array(result[2])
        batch_m2 = np.array(result[3])

        if batch_n > 0:
            n_new = test_n + batch_n
            delta = batch_mean - test_mean
            mean_new = test_mean + delta * (batch_n / n_new)
            m2_new = test_m2 + batch_m2 + (delta**2) * (test_n * batch_n / n_new)
            test_n = n_new
            test_mean = mean_new
            test_m2 = m2_new
            test_std = np.sqrt(test_m2 / max(1.0, test_n - 1) + 1e-8)

        utilities = compute_utilities(rewards)
        grad_mu = np.dot(utilities, z) / pop_size
        grad_sigma = np.dot(utilities, z**2 - 1) / pop_size

        mu += learning_rate * sigma * grad_mu
        sigma *= np.exp(learning_rate * grad_sigma)
        sigma = np.clip(sigma, 1e-5, 1.0)

    end_time = time.time()
    return end_time - start_time


def main():
    hidden_size = 16
    hidden_layers = 2
    pop_size = 50
    generations = 50

    print(
        f"Starting running time comparison tests: {generations} gens, pop {pop_size}, hidden {hidden_size}, layers {hidden_layers}...\n"
    )
    header = (
        f"{'ID':<3} | {'Environment':<25} | {'Seq':<12} | {'Vec':<12} | {'Speedup':<8}"
    )
    print(header)
    print("-" * len(header))

    for env_idx, env_name in enumerate(ENV_NAMES):
        try:
            try:
                temp_env = gym.make(env_name)
            except Exception as e:
                print(f"{env_idx:<3} | {env_name:<25} | {'N/A (Load Fail)':<40}")
                continue

            obs_space = temp_env.observation_space
            act_space = temp_env.action_space
            obs_is_discrete = isinstance(obs_space, gym.spaces.Discrete)
            act_is_discrete = isinstance(act_space, gym.spaces.Discrete)
            obs_dim = obs_space.n if obs_is_discrete else obs_space.shape[0]
            act_dim = act_space.n if act_is_discrete else act_space.shape[0]
            temp_env.close()

            param_dim = (
                hidden_size
                + (hidden_size * hidden_size + hidden_size) * (hidden_layers - 1)
                + hidden_size * act_dim
                + act_dim
                + obs_dim * hidden_size
            )

            mean_obs = np.zeros(obs_dim)
            std_obs = np.ones(obs_dim)

            time_seq = snes_loop(
                env_idx,
                "seq",
                generations,
                pop_size,
                param_dim,
                hidden_size,
                hidden_layers,
                mean_obs,
                std_obs,
            )
            time_vec = snes_loop(
                env_idx,
                "vec",
                generations,
                pop_size,
                param_dim,
                hidden_size,
                hidden_layers,
                mean_obs,
                std_obs,
            )
            speedup = time_seq / time_vec if time_vec > 0 else 0.0

            print(
                f"{env_idx:<3} | {env_name:<25} | {time_seq:<12.4f} | {time_vec:<12.4f} | {speedup:<6.2f}x"
            )

        except Exception as e:
            print(f"{env_idx:<3} | {env_name:<25} | Error: {str(e)[:30]}...")

    print("-" * len(header))


if __name__ == "__main__":
    main()
