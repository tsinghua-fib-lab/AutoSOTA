"""exp_run.py: Code for (parallel) experiment execution with SB3 compatible agents.
"""

import time
import numpy as np
from stable_baselines3 import PPO
from sb3_contrib import ARS
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium.envs.registration import register
from polyagents.polynomial_policies import PolynomialARSPolicy, PolynomialPPOPolicy
from utils.preprocessing import get_normalized_vec_env


__author__ = "Hannes Unger"
__version__ = "1.0"
__email__ = "hannes.unger@fh-salzburg.ac.at"


def get_sb3_polynomial_model_and_eval_env(basis='chebyshev', degree=3, algo='ars', learning_rate=0.0018, clip_range=0.4, clip_range_vf=None, batch_size=64, n_steps = 1024, n_epochs = 1, use_expln=True, coeffs=None, 
                                          zero_policy=False, delta_std=0.05, n_delta=8, n_top=None, use_fixed_std_schedule=False, buffer_size = 1000000, ent_coef = 'auto', 
                                          gamma = 0.99, gradient_steps = 1, learning_starts = 100, tau = 0.005, train_freq = 1, vf_coef= 0.5, max_grad_norm= 0.5,
                                          env_name="MountainCarContinuous-v0", tensorboard_log_dir="./tensorboard_logs/", seed=None, verbose=1, initialization='random', 
                                          normalize_actions=True, min_action=-1.0, max_action=1.0, observation_space_low=None, observation_space_high=None, new_min_obs=-1.0, new_max_obs=1.0,
                                          render=False):
    # Normalize actions to range [min_action, max_action], observations to range [new_min_obs, new_max_obs]
    env, eval_env = get_normalized_vec_env(env_name=env_name, normalize_actions=normalize_actions, min_action=min_action, max_action=max_action, observation_space_low=observation_space_low, 
                                           observation_space_high=observation_space_high, new_min=new_min_obs, new_max=new_max_obs, render=render)

    if algo == 'ars':
        model = ARS(PolynomialARSPolicy, env=env, verbose=verbose, tensorboard_log=tensorboard_log_dir, device='cpu', learning_rate=learning_rate, zero_policy=zero_policy, delta_std=delta_std, 
                    n_delta=n_delta, n_top=n_top, policy_kwargs=dict(degree=degree, basis=basis, initialization=initialization, coeffs=coeffs), seed=seed)
    elif algo == 'ars_mlp':
        model = ARS("MlpPolicy", env=env, verbose=verbose, tensorboard_log=tensorboard_log_dir, device='cpu', learning_rate=learning_rate, zero_policy=zero_policy, delta_std=delta_std, 
                    n_delta=n_delta, n_top=n_top, policy_kwargs=dict(net_arch=[16]), seed=seed)
        if coeffs is not None:
            model.policy.load_state_dict(coeffs)
    elif algo == 'ppo':
        model = PPO(PolynomialPPOPolicy, env=env, verbose=verbose, tensorboard_log=tensorboard_log_dir, device='cpu', learning_rate=learning_rate, clip_range=clip_range, clip_range_vf=clip_range_vf, 
                    n_steps=n_steps, n_epochs=n_epochs, batch_size=batch_size, policy_kwargs=dict(degree=degree, basis=basis, initialization=initialization, coeffs=coeffs, 
                                                                                                      use_fixed_std_schedule=use_fixed_std_schedule), seed=seed, vf_coef= 0.5, max_grad_norm= 0.5)
    elif algo == 'ppo_mlp':
        model = PPO('MlpPolicy', env=env, verbose=verbose, tensorboard_log=tensorboard_log_dir, device='cpu', learning_rate=learning_rate, clip_range=clip_range, clip_range_vf=clip_range_vf, 
                    n_steps=n_steps, n_epochs=n_epochs, batch_size=batch_size, seed=seed, vf_coef= 0.5, max_grad_norm= 0.5) 
        if coeffs is not None:
            model.policy.load_state_dict(coeffs)       
    else:
        raise Exception('RL algorithm not implemented.')
        
    return model, eval_env


def run_sb3_model(model, env, seed=None, episodes=1, verbose=False, render=False, options=None, return_observations=False):
    rewards = []
    if seed:
        env.seed(seed)
    for _ in range(episodes):
        observations = []
        reward = 0.0

        if options is not None:
            obs = env.venv.envs[0].reset(options=options)
        else:
            obs = env.reset() 
        if verbose:
            print(f'start: {env.get_original_obs()}')
        observations.append([obs[0]])
        terminated = [False]
        truncated = [{'TimeLimit.truncated': False}]

        while not (terminated[0] or truncated[0]['TimeLimit.truncated']):
            if render:
                env.render()
            action, _ = model.predict(obs[0], deterministic=True)
            obs, r, terminated, truncated = env.step([action])
            observations.append(obs)
            reward += r
        rewards.append(reward)
    if return_observations:
        observations.pop() # remove last element since it is the one after reset
        return rewards, observations
    else:
        return np.mean(rewards), np.std(rewards)


def run_sb3_polyagent_training(kwargs):
    try:
        if 'degree' in kwargs:
            degree = kwargs['degree']
        else:
            degree = 3
        if 'basis' in kwargs:
            basis = kwargs['basis']
        else:
            basis = 'chebyshev'  
        if 'algo' in kwargs:
            algo = kwargs['algo']
        else:
            algo = 'ars'                    
        if 'learning_rate' in kwargs:
            learning_rate = kwargs['learning_rate']
        else:
            learning_rate = 0.0003
        if 'steps' in kwargs:
            steps = kwargs['steps']
        else:
            steps = 100000
        if 'evaluate_every_n_steps' in kwargs:
            evaluate_every_n_steps = kwargs['evaluate_every_n_steps']
        else:
            evaluate_every_n_steps = 10000  
        if 'env_name' in kwargs:
            env_name = kwargs['env_name']
        else:
            raise Exception("No environment specified")      
        if 'tensorboard_log_dir' in kwargs:
            tensorboard_log_dir = kwargs['tensorboard_log_dir']
        else:
            tensorboard_log_dir = "./tensorboard_logs/"
        if 'name' in kwargs:
            name = kwargs['name']
        else:
            name = 'exp_run'
        if 'seed' in kwargs:
            seed = kwargs['seed']
        else:
            seed = 'None'
        if 'delta_std' in kwargs:
            delta_std = kwargs['delta_std']
        else:
            delta_std = 0.05
        if 'n_delta' in kwargs:
            n_delta = kwargs['n_delta']
        else:
            n_delta = 8
        if 'n_top' in kwargs:
            n_top = kwargs['n_top']
        else:
            n_top = None
        if 'verbose' in kwargs:
            verbose = kwargs['verbose']
        else:
            verbose = 0
        if 'initialization' in kwargs:
            initialization = kwargs['initialization']
        else:
            initialization = 'random'
        if 'normalize_actions' in kwargs:
            normalize_actions = kwargs['normalize_actions']
        else:
            normalize_actions = True
        if 'min_action' in kwargs:
            min_action = kwargs['min_action']
        else:
            min_action = -1.0
        if 'max_action' in kwargs:
            max_action = kwargs['max_action']
        else:
            max_action = 1.0
        if 'observation_space_low' in kwargs:
            observation_space_low = kwargs['observation_space_low']
        else:
            observation_space_low = None
        if 'observation_space_high' in kwargs:
            observation_space_high = kwargs['observation_space_high']
        else:
            observation_space_high = None
        if 'new_min_obs' in kwargs:
            new_min_obs = kwargs['new_min_obs']
        else:
            new_min_obs = -1.0
        if 'new_max_obs' in kwargs:
            new_max_obs = kwargs['new_max_obs']
        else:
            new_max_obs = 1.0
        if 'env_save_path' in kwargs:
            env_save_path = kwargs['env_save_path']
        else:
            env_save_path = None
        if 'clip_range' in kwargs:
            clip_range = kwargs['clip_range']
        else:
            clip_range = 0.4
        if 'clip_range_vf' in kwargs:
            clip_range_vf = kwargs['clip_range_vf']
        else:
            clip_range_vf = 0.4
        if 'n_steps' in kwargs:
            n_steps = kwargs['n_steps']
        else:
            n_steps = 2048
        if 'n_epochs' in kwargs:
            n_epochs = kwargs['n_epochs']
        else:
            n_epochs = 2
        if 'batch_size' in kwargs:
            batch_size = kwargs['batch_size']
        else:
            batch_size = 64
        if 'coeffs' in kwargs:
            coeffs = kwargs['coeffs']
        else:
            coeffs = None

        model, eval_env = get_sb3_polynomial_model_and_eval_env(algo=algo, basis=basis, degree=degree, learning_rate=learning_rate, use_expln=True, env_name=env_name, 
                                                                clip_range=clip_range, delta_std=delta_std, n_delta=n_delta, n_top=n_top, tensorboard_log_dir=tensorboard_log_dir, 
                                                                clip_range_vf=clip_range_vf, n_steps=n_steps, n_epochs=n_epochs, batch_size=batch_size, coeffs=coeffs,
                                                                seed=seed, verbose=verbose, initialization=initialization, normalize_actions=normalize_actions, min_action=min_action, 
                                                                max_action=max_action, observation_space_low=observation_space_low, observation_space_high=observation_space_high,
                                                                new_min_obs=new_min_obs, new_max_obs=new_max_obs)
        eval_callback = EvalCallback(
            eval_env,
            eval_freq=evaluate_every_n_steps,
            deterministic=True,
            render=False,
            n_eval_episodes=1,
        )

        start = time.time()

        name = f'{name}_{get_datetime_string()}'
        print(f'{name}: .')
        model.learn(total_timesteps=steps, tb_log_name=name, callback=eval_callback) 
        print(f'{name}: #')

        duration = f'{time.time()-start:.0f}'

        if env_save_path is not None:
            eval_env.save(env_save_path + name + '_env')

        if "mlp" in algo:
            return name, duration, model.policy.state_dict()        
        else:
            return name, duration, model.policy.parameters()

    except Exception as e:
        print(f'An Exception ocurred: {e}')
        return e


def job_get_episode_reward(params, kwargs):
    try:
        if 'basis' in kwargs:
            basis = kwargs['basis']
        else:
            basis = 'chebyshev'  
        if 'algo' in kwargs:
            algo = kwargs['algo']
        else:
            algo = 'ars'  
        if 'env_name' in kwargs:
            env_name = kwargs['env_name']
        else:
            env_name = 'MountainCarContinuous-v0'
            
        model, _ = get_sb3_polynomial_model_and_eval_env(algo=algo, basis=basis, env_name=env_name, coeffs=params[-2])

        return params[0], params[-1], run_sb3_model(model, start_loc=params[-1])

    except Exception as e:
        print(f'An Exception ocurred: {e}')
        return e    


def evaluate_chebyshev_pendulum_deterministic_starting_states(chebyshev_results, initial_states, env_name, algo='ars'):
    chebyshev_eval_results = []
    rewards = []

    for c in chebyshev_results:
        rewards = []
        model, eval_env = get_sb3_polynomial_model_and_eval_env(basis='chebyshev', env_name=env_name, coeffs=c[-1], algo=algo)
        for s in initial_states:
            rewards.append(run_sb3_model(model, eval_env, options={'x_init': s[0], 'y_init': s[1]})[0])
        chebyshev_eval_results.append([np.mean(rewards), np.std(rewards), np.min(rewards), np.max(rewards)])
    
    return chebyshev_eval_results


def evaluate_chebyshev_pendulum_deterministic_starting_states_single_coeffs(coeffs, initial_states, algo='ars', action_space_normalization=True, env_name='DeterministicPendulum-v1'):
    register_deterministic_pendulum_env(env_name)
    chebyshev_eval_results = []
    rewards = []
    model, eval_env = get_sb3_polynomial_model_and_eval_env(basis='chebyshev', env_name=env_name, coeffs=coeffs, algo=algo, normalize_actions=action_space_normalization)
    for s in initial_states:
        rewards.append(run_sb3_model(model, eval_env, options={'x_init': s[0], 'y_init': s[1]})[0])
    chebyshev_eval_results.append([np.mean(rewards), np.std(rewards), np.min(rewards), np.max(rewards)])
    return chebyshev_eval_results


def get_datetime_string():
    now = time.time()
    t = time.localtime(now)
    micros = int((now - int(now)) * 1_000_000)
    return time.strftime("%Y%m%d_%H%M_", t) + str(int(t.tm_sec)) + f"_{micros:06d}"


def register_deterministic_pendulum_env(deterministic_pendulum_env_name):
    # Register the environment
    register(
        id=f'{deterministic_pendulum_env_name}',
        entry_point='envs.custom_gymnasium:DeterministicPendulumEnv',  
        max_episode_steps=200,
    )