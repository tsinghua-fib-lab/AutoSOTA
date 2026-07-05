"""aero.py: Helper functions for training and evaluation on the Quanser Aero 2 gymnasium env.
"""
import wandb
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from polyagents.polynomial_policies import PolynomialARSPolicy, PolynomialPPOPolicy
from aero_envs.utils.trajectories import RandomStepTrajectory, EvaluationStepTrajectory, RandomTrajectory, EvaluationTrajectory
from stable_baselines3 import PPO
from sb3_contrib import ARS
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from utils import exp_run


__author__ = "Hannes Unger"
__version__ = "1.0"
__email__ = "hannes.unger@fh-salzburg.ac.at"


tensorboard_log_dir = "./tensorboard_logs/aero"

# Aero env parameters
steps = 150000
steps_ars = 4000000
power_penalty = 0.0 # the "vanilla" reward setting from the original paper
norm_action = True
norm_observation = True

evaluation_stop_time = 80.0 # episode length = stop time * sample time = 80.0 * 0.1 = 800
evaluation_initial_tilt = 0.0


def get_aero_train_env(episodic=False, step=False):
    if step:
        training_tilt_function = RandomStepTrajectory
    else:
        training_tilt_function = RandomTrajectory

    stop_time = 200 if episodic else steps*0.1 # stop_time in seconds. sample_time = 0.1s per default. with 10000 training steps, stop_time should be 10000*0.1 = 1000
    return gym.make(
        "AeroSimulationEnv-v0",
        #render_mode="human",
        target_tilt=training_tilt_function(),
        stop_time=stop_time, 
        norm_action=norm_action,
        norm_observation=norm_observation,
        initial_tilt=lambda: np.random.uniform(-40 * np.pi / 180, 40 * np.pi / 180), # callable, for infinite horizon reset occurs only once
        power_penalty_weight=power_penalty
    )


def get_aero_eval_env(step=False):
    if step:
        evaluation_tilt_function = EvaluationStepTrajectory
    else:
        evaluation_tilt_function = EvaluationTrajectory
    return Monitor(gym.make(
        "AeroSimulationEnv-v0",
        #render_mode="human",
        target_tilt=evaluation_tilt_function(),
        stop_time=evaluation_stop_time, # stop_time in seconds. sample_time = 0.1s per default. with 10000 training steps, stop_time should be 10000*0.1 = 1000
        norm_action=norm_action,
        norm_observation=norm_observation,
        initial_tilt=evaluation_initial_tilt, # callable, for infinite horizon reset occurs only once
        power_penalty_weight=power_penalty)
    )


def get_aero_eval_callback(flatten_observation=False):
    env = Monitor(get_aero_eval_env())
    return EvalCallback(
        FlattenObservation(env) if flatten_observation else env,
        eval_freq=10_000,
        deterministic=True,
        render=False,
        n_eval_episodes=1,
    )


def evaluate_aero_agent(model, render=True, flatten_observation=False, return_obs =False, step=False):
    if step:
        evaluation_tilt_function = EvaluationStepTrajectory
    else:
        evaluation_tilt_function = EvaluationTrajectory

    if render:
        env = gym.make("AeroSimulationEnv-v0", render_mode="human", initial_tilt=evaluation_initial_tilt, stop_time=evaluation_stop_time, target_tilt=evaluation_tilt_function(), norm_action=norm_action, norm_observation=norm_observation, power_penalty_weight=power_penalty)
    else:
        env = gym.make("AeroSimulationEnv-v0", initial_tilt=evaluation_initial_tilt, stop_time=evaluation_stop_time, target_tilt=evaluation_tilt_function(), norm_action=norm_action, norm_observation=norm_observation, power_penalty_weight=power_penalty)

    if flatten_observation:
        env = FlattenObservation(env)

    observations = []
    actions = []
    powers = []
    obs, info = env.reset()
    observations.append(obs)
    done = False
    reward = []
    while not done:
        action = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, info = env.step(action)
        observations.append(obs)
        reward.append(r)
        actions.append(action)
        powers.append(env.unwrapped.power)
        done = terminated or truncated
    env.close()
    if return_obs:
        return np.mean(reward), observations, actions, powers
    else:
        return np.mean(reward)


def run_ppo_training(seed=0):
    print('.')
    log_prefix = 'PPO'
    log_dir = tensorboard_log_dir + f'/{log_prefix}'
    name = f'{log_prefix}_{exp_run.get_datetime_string()}'
    model = PPO(env=get_aero_train_env(episodic=True), policy='MultiInputPolicy', tensorboard_log=log_dir, device='cpu', seed=seed) # MultiInputPolicy is required for dict observation space
    model.learn(total_timesteps=steps, tb_log_name=name, callback=get_aero_eval_callback())
    print('#')
    return name, model.policy.state_dict() 


def run_ch_ppo_training(args):
    seed, degree = args
    print('.')
    log_prefix = 'CH_PPO'
    log_dir = tensorboard_log_dir + f'/{log_prefix}'
    name = f'{log_prefix}_{exp_run.get_datetime_string()}'
    model = PPO(env=FlattenObservation(get_aero_train_env(episodic=True)), policy=PolynomialPPOPolicy, tensorboard_log=log_dir, device='cpu', learning_rate=0.0006, clip_range=0.4, clip_range_vf = 0.4, 
                n_steps=2048, n_epochs=10, batch_size=64, policy_kwargs=dict(degree=degree), seed=seed)
    # model = PPO(env=FlattenObservation(get_aero_train_env(episodic=True)), policy=PolynomialPPOPolicy, tensorboard_log=log_dir, device='cpu', learning_rate=0.0004, clip_range=0.4, clip_range_vf = 0.4, 
    #             n_steps=2048, n_epochs=5, batch_size=64, policy_kwargs=dict(degree=degree), seed=seed) # hyperparameters from pendulum env experiments
    model.learn(total_timesteps=steps, tb_log_name=name, callback=get_aero_eval_callback(flatten_observation=True))
    print('#')
    return name, model.policy.parameters()


def run_ch_ars_training(args):
    seed, degree = args
    print('.')
    log_prefix = 'CH_ARS'
    log_dir = tensorboard_log_dir + f'/{log_prefix}'
    name = f'{log_prefix}_{exp_run.get_datetime_string()}'
    model = ARS(PolynomialARSPolicy, env=FlattenObservation(get_aero_train_env(episodic=True)), tensorboard_log=log_dir, n_delta=4, policy_kwargs=dict(degree=degree), learning_rate=0.0044, delta_std=0.01, n_eval_episodes=1, seed=seed)
    model.learn(total_timesteps=steps_ars, tb_log_name=name, callback=get_aero_eval_callback(flatten_observation=True))
    print('#')
    return name, model.policy.parameters()


def run_mlp_ars_training(seed=0):
    print('.')
    log_prefix = 'ARS'
    log_dir = tensorboard_log_dir + f'/{log_prefix}'
    name = f'{log_prefix}_{exp_run.get_datetime_string()}'
    model = ARS('MlpPolicy', env=FlattenObservation(get_aero_train_env(episodic=True)), tensorboard_log=log_dir, n_eval_episodes=1, n_delta=4, learning_rate=0.0044, delta_std=0.01, seed=seed)
    model.learn(total_timesteps=steps_ars, tb_log_name=name, callback=get_aero_eval_callback(flatten_observation=True))
    print('#')
    return name, model.policy.action_net.state_dict() 


def train_ars_wandb():
    log_prefix = 'ARS'
    log_dir = tensorboard_log_dir + f'/{log_prefix}'

    config_defaults = {
        "learning_rate": 0.05,
        "n_delta": 8,
        "delta_std": 0.05,
        "n_eval_episodes": 1,
        "policy_kwargs": [64, 64]
    }

    wandb.init(
        project="sb3-sweep-ch-aero-ars",
        config=config_defaults,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True
    )

    config = wandb.config

    model = ARS('MlpPolicy', 
                env=FlattenObservation(get_aero_train_env(episodic=True)), 
                tensorboard_log=log_dir, 
                learning_rate=config.learning_rate, 
                n_delta=config.n_delta, 
                delta_std=config.delta_std, 
                n_eval_episodes=config.n_eval_episodes, 
                policy_kwargs=dict(net_arch=config.policy_kwargs),
                seed=0)

    model.learn(
        total_timesteps=steps_ars,
        callback=get_aero_eval_callback(flatten_observation=True),
        tb_log_name=f'{log_prefix}_{exp_run.get_datetime_string()}'
    )

    wandb.finish()


def train_ch_ars_wandb():
    log_prefix = 'CH_ARS'
    log_dir = tensorboard_log_dir + f'/{log_prefix}'

    config_defaults = {
        "learning_rate": 0.05,
        "n_delta": 8,
        "delta_std": 0.05,
        "n_eval_episodes": 1,
        "degree": 6,
    }

    wandb.init(
        project="sb3-sweep-ch-aero-ars",
        config=config_defaults,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True
    )

    config = wandb.config

    model = ARS(PolynomialARSPolicy, 
                env=FlattenObservation(get_aero_train_env(episodic=True)), 
                tensorboard_log=log_dir, 
                learning_rate=config.learning_rate, 
                n_delta=config.n_delta, 
                delta_std=config.delta_std, 
                n_eval_episodes=config.n_eval_episodes, 
                policy_kwargs=dict(degree=config.degree),
                seed=0)

    model.learn(
        total_timesteps=steps_ars,
        callback=get_aero_eval_callback(flatten_observation=True),
        tb_log_name=f'{log_prefix}_{exp_run.get_datetime_string()}'
    )

    wandb.finish()


def train_ch_ppo_wandb():
    log_prefix = 'CH_PPO'
    log_dir = tensorboard_log_dir + f'/{log_prefix}'

    config_defaults = {
        "learning_rate": 0.001,
        "n_steps": 512, 
        "n_epochs": 1, 
        "batch_size": 32,
        "clip_range": 0.4,
        "degree": 3,
        "std_schedule": 0,
        "total_steps": 250_000
    }

    wandb.init(
        project="sb3-sweep-ch-aero-ppo",
        config=config_defaults,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True
    )

    config = wandb.config

    if config.std_schedule == 0:
        policy_kwargs=dict(degree=config.degree, use_fixed_std_schedule=False, use_expln=True)
    elif config.std_schedule == 1:
        policy_kwargs=dict(degree=config.degree, use_fixed_std_schedule=False, use_expln=False)
    else:
        policy_kwargs=dict(degree=config.degree, use_fixed_std_schedule=True)

    model = PPO(env=FlattenObservation(get_aero_train_env(episodic=True)), 
                policy=PolynomialPPOPolicy, 
                tensorboard_log=log_dir, 
                device='cpu', 
                learning_rate=config.learning_rate, 
                n_steps=config.n_steps, 
                n_epochs=config.n_epochs, 
                batch_size=config.batch_size, 
                clip_range=config.clip_range, 
                clip_range_vf=config.clip_range,
                policy_kwargs=policy_kwargs, 
                seed=0)

    model.learn(
        total_timesteps=config.total_steps,
        callback=get_aero_eval_callback(flatten_observation=True),
        tb_log_name=f'{log_prefix}'
    )

    wandb.finish()


def plot_aero_policy(X_flat, Y_flat, Z_flat, actions_flat, vmin=None, vmax=None, ax=None, title=None):
    fig = None
    standalone = False
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        standalone = True
    if vmin is not None and vmax is not None:
        p = ax.scatter(X_flat, Y_flat, Z_flat, c=actions_flat, cmap='viridis', vmin=vmin, vmax=vmax, marker='.', alpha=0.01)
    else:
        p = ax.scatter(X_flat, Y_flat, Z_flat, c=actions_flat, cmap='viridis', marker='.', alpha=0.01)
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$\dot{\theta}$')
    ax.set_zlabel(r'$r$')
    ax.set_title(title)
    if standalone:
        cb = plt.colorbar(p, ax=ax, pad=0.15)
        cb.solids.set(alpha=1)  # force solid colorbar
        plt.show()
    else:
        return p


def plot_tilt_series(series_list, width=12, height=8):
    """
    series_list: list of [name, pitch_data, action_data, power_data]
        pitch_data rows: [tilt, velocity, target_tilt]
        action_data: array-like, 1D list of actions
    """
    processed = [
        (name, np.array(pitch_data), np.array(action_data), np.array(power_data))
        for name, pitch_data, action_data, power_data in series_list
    ]

    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width, height), sharex=True)

    # ===========================
    #  Top subplot: tilt vs target
    # ===========================

    # target tilt from the first series
    first_name, first_pitch, _, _ = processed[0]
    target_tilt = first_pitch[:, 1]
    ax1.plot(target_tilt, label="target", color="black", linestyle="--")

    # tilt from each series
    for name, pitch_data, _, _ in processed:
        tilt = pitch_data[:, 0]
        ax1.plot(tilt, label=name)

    ax1.set_ylabel("Pitch (°)")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x*90:.0f}")) # denormalize. values are normalized to 90°
    ax1.set_xlabel("Time Step")
    ax1.set_title("Pitch Angle")
    ax1.grid(True)
    ax1.legend()

    # ===========================
    #  Bottom subplot: actions
    # ===========================

    for name, _, actions, _ in processed:
        ax2.plot(actions, label=f"{name} action")

    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Action (V)")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x*24:.0f}")) # denormalize. values are normalized to 24V
    ax2.set_title("Control Action")
    ax2.grid(True)
    ax2.legend()

    # ===========================
    #  Compute summary metrics
    # ===========================

    summary_lines = []
    for name, pitch_data, _, power_data in processed:
        avg_power = np.mean(power_data)
        avg_tilt_dev = get_average_deviation(pitch_data) #*90 = degrees

        summary_lines.append(
            f"{name}: Avg power consumption per step = {avg_power:.3f}, Avg pitch deviation = {avg_tilt_dev:.3f} rad"
        )

    # Combine into a single suptitle
    fig.suptitle("\n".join(summary_lines), fontsize=12, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def convert_observation_dict_to_arr(obs_dict):
    keys = ["pitch", "target", 'velocity']
    return [[d[k] for k in keys] for d in obs_dict]

def convert_arr_obs_to_dict_obs(obs):
    return {"pitch": np.array([obs[0]]), "target": np.array([obs[1]]), 'velocity': np.array([obs[2]])}

def extract_action_sequence(actions):
    return [a[0][0] for a in actions]

def get_average_deviation(series_list):
    data = np.array(series_list)
    pitch = data[:, 0]
    target = data[:, 1]
    return np.mean(np.abs(pitch - target))