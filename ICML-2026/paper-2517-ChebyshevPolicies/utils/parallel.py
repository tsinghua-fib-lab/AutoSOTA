"""parallel.py: Code for parallel experiment execution with TrainableMountainCarMRPWrapper compatible agents.
"""

import copy
import numpy as np
import gymnasium as gym
import multiprocessing as mp
import multiprocessing.pool # need this import also if unused: https://stackoverflow.com/questions/75996502/module-multiprocessing-has-no-attribute-pool-error
from algorithms import polynomial_agents
from concurrent.futures import ProcessPoolExecutor, as_completed
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.envs.registration import register
from datetime import datetime
import os
import yaml

# # Create an alias so pickle can find 'gym' when unpickling
# sys.modules['gym'] = gym

from rl_zoo3 import ALGOS, create_test_env, get_saved_hyperparams
from rl_zoo3.load_from_hub import download_from_hub
from rl_zoo3.utils import StoreDict, get_model_path
from stable_baselines3.common.utils import set_random_seed

from envs import custom_gymnasium

__author__ = "Hannes Unger"
__version__ = "2.0"
__email__ = "hannes.unger@fh-salzburg.ac.at"


MAX_WORKERS = 5
EPSILON = 1e-3
UG_MIN_EPSILON = 1e-2
V_EPSILON = 1e-6


def get_rl_zoo3_model_and_generate_env(algo="ppo", folder="rl-trained-agents", env_name="MountainCarContinuous-v0"):
    '''
    Retrieve rlzoo model from the huggingface hub or load from disk if already present
    repo_id = id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name})
    filename = name of the model zip file from the repository including the extension .zip
    checkpoint = load_from_hub(
        repo_id="sb3/ppo-MountainCarContinuous-v0",
        filename="ppo-MountainCarContinuous-v0.zip",
    )
    '''
    env_name=env_name
    organization="sb3"
    folder=folder
    exp_id=0
    repo_name=None
    force=False
    requires_download = False

    try:
        print("Loading model from local disk")
        _, model_path, log_path = get_model_path(
            exp_id,
            folder,
            algo,
            env_name
        )

        stats_path = os.path.join(log_path, env_name)
        hyperparams, maybe_stats_path = get_saved_hyperparams(stats_path, norm_reward=False, test_mode=True)
    except:
        print("Model not found on local disk, starting download")
        requires_download = True
    
    if requires_download:
        download_from_hub(
            algo=algo,
            env_name=env_name,
            organization=organization,
            folder=folder,
            exp_id=exp_id,
            repo_name=repo_name,
            force=force,
        )
        _, model_path, log_path = get_model_path(
            exp_id,
            folder,
            algo,
            env_name
        )

    stats_path = os.path.join(log_path, env_name)
    hyperparams, maybe_stats_path = get_saved_hyperparams(stats_path, norm_reward=False, test_mode=True)

    env_kwargs = {}
    args_path = os.path.join(log_path, env_name, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path) as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]

    env = create_test_env(
        env_name,
        n_envs=1,
        stats_path=maybe_stats_path,
        seed=set_random_seed(0),
        log_dir="logs/",
        should_render=False,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )

    model = ALGOS[algo].load(model_path)

    return model, env


def job_reinforce_train(kwargs):
    try:
        if 'degree' in kwargs:
            degree = kwargs['degree']
        else:
            degree = 3
        if 'alpha_mu' in kwargs:
            alpha_mu = kwargs['alpha_mu']
        else:
            alpha_mu = 0.0003
        if 'alpha_sigma' in kwargs:
            alpha_sigma = kwargs['alpha_sigma']
        else:
            alpha_sigma = 0.00003
        if 'alpha_critic' in kwargs:
            alpha_critic = kwargs['alpha_critic']
        else:
            alpha_critic = 0.00003
        if 'episodes' in kwargs:
            episodes = kwargs['episodes']
        else:
            episodes = 50
        if 'discount' in kwargs:
            discount = kwargs['discount']
        else:
            discount = 0.9
        if 'initial_sigma' in kwargs:
            initial_sigma = kwargs['initial_sigma']
        else:
            initial_sigma = 0.25
        if 'mu_optimizer' in kwargs:
            mu_optimizer = kwargs['mu_optimizer']
        else:
            mu_optimizer = 'adam'
        if 'sigma_optimizer' in kwargs:
            sigma_optimizer = kwargs['sigma_optimizer']
        else:
            sigma_optimizer = 'adam'
        if 'critic_optimizer' in kwargs:
            critic_optimizer = kwargs['critic_optimizer']
        else:
            critic_optimizer = 'adam'
        if 'method' in kwargs:
            method = kwargs['method']
        else:
            method = 'reinforce_autodiff'
        if 'normalize_observations' in kwargs:
            normalize_observations = kwargs['normalize_observations']
        else:
            normalize_observations = True
        if 'approximator' in kwargs:
            approximator = kwargs['approximator']
        else:
            approximator = 'polynomial'
        if 'mlp_n_hidden_nodes' in kwargs:
            mlp_n_hidden_nodes = kwargs['mlp_n_hidden_nodes']
        else:
            mlp_n_hidden_nodes = 4
        if 'mlp_n_input_nodes' in kwargs:
            mlp_n_input_nodes = kwargs['mlp_n_input_nodes']
        else:
            mlp_n_input_nodes = 2
        if 'mlp_n_output_nodes' in kwargs:
            mlp_n_output_nodes = kwargs['mlp_n_output_nodes']
        else:
            mlp_n_output_nodes = 1
        if 'env_name' in kwargs:
            env_name = kwargs['env_name']
        else:
            env_name = "MountainCarContinuous-v0"
        if 'mu_coeffs' in kwargs:
            mu_coeffs = kwargs['mu_coeffs']
        else:
            mu_coeffs = None
        if 'sigma_coeffs' in kwargs:
            sigma_coeffs = kwargs['sigma_coeffs']
        else:
            sigma_coeffs = None
        if 'net_arch' in kwargs:
            net_arch = kwargs['net_arch']
        else:
            net_arch = None

        print(f"start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        env = gym.make(env_name)

        if approximator == 'mlp':
            reinforce_trainable_mrp = polynomial_agents.TrainableContinuousMRPWrapper(env, basis='mlp',
                                                                                    initial_sigma=initial_sigma,
                                                                                    normalize_observations=False,
                                                                                    initialization='constant',
                                                                                    mlp_n_hidden_nodes=mlp_n_hidden_nodes,
                                                                                    mlp_n_input_nodes=mlp_n_input_nodes,
                                                                                    mlp_n_output_nodes=mlp_n_output_nodes,
                                                                                    net_arch=net_arch)
            reinforce_rewards = []
            reinforce_steps = []
            reinforce_loss = []
            _, _, _ = reinforce_trainable_mrp.train(alpha_mu=alpha_mu, alpha_sigma=alpha_sigma, epochs=episodes, discount=discount, method='reinforce_autodiff',
                                                                        learning_history=reinforce_rewards,
                                                                        steps_history=reinforce_steps,
                                                                        loss_history=reinforce_loss,
                                                                        mu_optimizer=mu_optimizer,
                                                                        sigma_optimizer=sigma_optimizer,
                                                                        critic_optimizer=critic_optimizer,
                                                                        verbose=False)
            return [np.sum(reinforce_rewards), reinforce_rewards, reinforce_loss, reinforce_steps,
                    reinforce_trainable_mrp.agent.sigma_approximator.model.state_dict(), reinforce_trainable_mrp.agent.mu_approximator.model.state_dict()]
        else:
            reinforce_trainable_mrp = polynomial_agents.TrainableContinuousMRPWrapper(env, basis='chebyshev',
                                                                                            degree=degree,
                                                                                            normalize_observations=normalize_observations,
                                                                                            initial_sigma=initial_sigma,
                                                                                            mu_coeffs=mu_coeffs,
                                                                                            sigma_coeffs=sigma_coeffs)
            reinforce_rewards = []
            reinforce_steps = []
            reinforce_coeffs = []
            reinforce_loss = []
            _, _, _, _ = reinforce_trainable_mrp.train(alpha_mu=alpha_mu, alpha_sigma=alpha_sigma, alpha_critic=alpha_critic, epochs=episodes, discount=discount,
                                                                    method=method,
                                                                    learning_history=reinforce_rewards,
                                                                    steps_history=reinforce_steps,
                                                                    loss_history=reinforce_loss,
                                                                    coeffs_history=reinforce_coeffs,
                                                                    mu_optimizer=mu_optimizer,
                                                                    sigma_optimizer=sigma_optimizer,
                                                                    critic_optimizer=critic_optimizer,
                                                                    verbose=False)

            print(f"end: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            return [np.sum(reinforce_rewards), reinforce_rewards, reinforce_loss, reinforce_steps,
                    reinforce_coeffs, reinforce_trainable_mrp.agent.sigma_approximator.coeffs.numpy(), reinforce_trainable_mrp.agent.mu_approximator.coeffs.numpy()]

    except Exception as e:
        print(f'An Exception ocurred: {e}')
        return e


def job_evaluate(kwargs):
    try:
        if 'degree' in kwargs:
            degree = kwargs['degree']
        else:
            degree = 3
        if 'episodes' in kwargs:
            episodes = kwargs['episodes']
        else:
            episodes = 50
        if 'normalize_observations' in kwargs:
            normalize_observations = kwargs['normalize_observations']
        else:
            normalize_observations = True
        if 'mu_coeffs' in kwargs:
            mu_coeffs = kwargs['mu_coeffs']
        else:
            mu_coeffs = None
        if 'basis' in kwargs:
            basis = kwargs['basis']
        else:
            basis = 'chebyshev'
        if 'mlp_n_input_nodes' in kwargs:
            mlp_n_input_nodes = kwargs['mlp_n_input_nodes']
        else:
            mlp_n_input_nodes = 2
        if 'mlp_n_hidden_nodes' in kwargs:
            mlp_n_hidden_nodes = kwargs['mlp_n_hidden_nodes']
        else:
            mlp_n_hidden_nodes = 4
        if 'mlp_n_output_nodes' in kwargs:
            mlp_n_output_nodes = kwargs['mlp_n_output_nodes']
        else:
            mlp_n_output_nodes = 1
        if 'env_name' in kwargs:
            env_name = kwargs['env_name']
        else:
            env_name = "MountainCarContinuous-v0"
        if 'return_coeffs' in kwargs:
            return_coeffs = True
        else:
            return_coeffs = False
        if 'net_arch' in kwargs:
            net_arch = kwargs['net_arch']
        else:
            net_arch = None
        if 'start_loc' in kwargs:
            options = {'low': kwargs['start_loc'], 'high': kwargs['start_loc']}
        else:
            options = None
        if env_name == 'DeterministicPendulum-v1':
            register_deterministic_pendulum_env(env_name)
            options={'x_init': kwargs['x_init'], 'y_init': kwargs['y_init']}

        env = gym.make(env_name)
        mrp = polynomial_agents.TrainableContinuousMRPWrapper(env, basis=basis, degree=degree,
                                                                        normalize_observations=normalize_observations,
                                                                        initial_sigma=0.25,
                                                                        mu_coeffs=mu_coeffs,
                                                                        mlp_n_input_nodes=mlp_n_input_nodes,
                                                                        mlp_n_hidden_nodes=mlp_n_hidden_nodes,
                                                                        mlp_n_output_nodes=mlp_n_output_nodes,
                                                                        net_arch=net_arch)        

        print(f"start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        episode_rewards = []
        rewards = []

        # Run actions from trained policy
        obs = mrp.reset(options=options)[0]
        i = 0
        while i < episodes:
            obs, reward, terminated, truncated, info, action, _ = mrp.step(obs)
            episode_rewards.append(reward)
            if terminated or truncated:
                obs = mrp.reset()[0]
                rewards.append(np.sum(episode_rewards))
                episode_rewards = []
                i += 1

        print(f"end: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if not return_coeffs:
            return rewards
        else:
            return [mu_coeffs, rewards]

    except Exception as e:
        print(f'An Exception ocurred: {e}')
        return e


def job_evaluate_nruns(mu_coeffs, kwargs):
    try:
        if 'n_runs' in kwargs:
            n_runs = kwargs['n_runs']
        else:
            n_runs = 5

        kwargs['mu_coeffs'] = mu_coeffs
        args = [kwargs for i in range(n_runs)]

        pool = NestablePool(n_runs)
        results = pool.map(job_evaluate, args)

        pool.close()
        pool.join()

        return results

    except Exception as e:
        print(f'An Exception ocurred: {e}')
        return e


def job_evaluate_ncoeffs(mu_coeffs, kwargs):
    try:
        args = [copy.deepcopy(kwargs) for _ in range(len(mu_coeffs))]

        for i, arg in enumerate(args):
            arg['mu_coeffs'] = copy.deepcopy(mu_coeffs[i])   

        results = []
        
        # Use ProcessPoolExecutor to limit number of workers
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all jobs
            futures = [executor.submit(job_evaluate, arg) for arg in args]
            
            # As each job completes, add the result to `results`
            for future in as_completed(futures):
                results.append(future.result())

        return results

    except Exception as e:
        print(f'An Exception ocurred: {e}')
        return e


def job_reinforce_optimizers(optimizer, kwargs):
    try:
        if 'n_runs' in kwargs:
            n_runs = kwargs['n_runs']
        else:
            n_runs = 5

        kwargs['mu_optimizer'] = optimizer
        kwargs['sigma_optimizer'] = optimizer
        kwargs['critic_optimizer'] = optimizer

        args = [kwargs for _ in range(n_runs)]
        results = []
        
        # Use ProcessPoolExecutor to limit number of workers
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all jobs
            futures = [executor.submit(job_reinforce_train, arg) for arg in args]
            
            # As each job completes, add the result to `results`
            for future in as_completed(futures):
                results.append(future.result())

        return results

    except Exception as e:
        print(f'An Exception ocurred: {e}')
        return e


def job_delta_max_a_policy(delta, kwargs):
    try:
        if 'start_loc' in kwargs:
            start_loc = kwargs['start_loc']
        else:
            start_loc = -np.pi/6
        if 'left_wall' in kwargs:
            left_wall = kwargs['left_wall']
        else:
            left_wall = -1.2
        if 'opt_last_stroke' in kwargs:
            opt_last_stroke = kwargs['opt_last_stroke']
        else:
            opt_last_stroke = False
        if 'alphapoly' in kwargs:
            alphapoly = kwargs['alphapoly']
        else:
            alphapoly = False

        env = gym.make("MountainCarContinuous-v0")

        if alphapoly:
            alphapolycoeffs = [ 0.03905354, -0.0670672 ,  0.12519343, -0.00250992, -0.52350381]
            alphapoly = np.poly1d(alphapolycoeffs)
            alpha = alphapoly(delta)
        else:
            alpha = -np.pi/6

        eps = 1e-5
        left_wall += eps
        
        opt_endgame = 21 * [0] + [1, 1, 1, 1, 1, 0.61829] + 1000 * [0]
        minimum_gravity_potential_pos = -np.pi/6
        radius = delta/2
        neg_x_star = np.float32(-2*np.pi/6-0.45)

        actions = []
        velocities = []
        observations = []
        positions = []
        observations.append(env.reset(options={'low': start_loc, 'high': start_loc})[0])

        i = 0
        j = 0
        r = 0.0
        action=None
        x_min = minimum_gravity_potential_pos
        endgame = False

        while True:
            if observations[i][0] <= left_wall:
                endgame = True
            if endgame and opt_last_stroke:
                if left_wall - eps <= neg_x_star:
                    action = 0.0 # potential is high enough to reach flag without action
                else:
                    action = opt_endgame[j]
                    j+=1
            else:
                disttogravminpot = alpha - observations[i][0]
                if (np.fabs(disttogravminpot) <= radius): # if in interval
                    #print(f"{observations[i][0]} at dist to min of {disttogravminpot}")
                    # apply maximum force into the direction of movement
                    if observations[i][1] < 0: 
                        action=-1.0 
                    else:
                        action = 1.0
                else:
                    action = 0.0
            actions.append(action)
            
            obs, reward, terminated, truncated, info = env.step([action])
            observations.append(obs)
            velocities.append(obs[1])
            positions.append(obs[0])
            r += reward
            
            i+=1

            if observations[i][0] < x_min:
                x_min = observations[i][0]

            if terminated:
                return r, observations[i][1], x_min, len(observations), actions, velocities, positions
            if truncated:
                return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    except Exception as e:
        print(f'An Exception ocurred: {e}')
        return e


def job_analytic_solution_policy(cin, kwargs):
    try:
        if 'start_loc' in kwargs:
            start_loc = kwargs['start_loc']
        else:
            start_loc = -np.pi/6
        if 'left_wall' in kwargs:
            left_wall = kwargs['left_wall']
        else:
            left_wall = -1.2
        if 'cin_mode' in kwargs:
            cin_mode = kwargs['cin_mode']
        else:
            cin_mode = 'absolute'
        if 'minimum' in kwargs:
            minimum = kwargs['minimum']
        else:
            minimum = -np.pi/6
        if 'stop_at_left_wall' in kwargs:
            stop_at_left_wall = kwargs['stop_at_left_wall']
        else:
            stop_at_left_wall = False
        if 'single_phase' in kwargs:
            single_phase = kwargs['single_phase']
        else:
            single_phase = False

        env = gym.make("MountainCarContinuous-v0")

        v_max = 0.07
        maxc = 1.0 / v_max

        if cin_mode == 'relative':
            cin = [f*maxc for f in cin]
        
        if cin[0] > maxc or cin[1] > maxc:
            raise Exception(f'Maximum C ({maxc}) exceeded')

        actions = []
        abs_velocities = []
        observations = []
        positions = []
        observations.append(env.reset(options={'low': start_loc, 'high': start_loc})[0])

        i = 0
        r = 0.0
        action=None
        c = cin[0]
        endgame=False

        while True:
            if not endgame and np.fabs(observations[i][0] - left_wall) < EPSILON:
                if stop_at_left_wall:
                    return cin, r
                endgame = True
                if not single_phase:
                    c=cin[1]
            vel = observations[i][1]
            # if np.fabs(vel) < EPSILON and np.fabs(observations[i][0] - minimum) < EPSILON:
            #     action = 0.1
            # else:
            action = c * np.abs(vel)
            if np.fabs(observations[i][0] - minimum) < UG_MIN_EPSILON:
                action = max(0.1, action)
            if vel <= -0.0:
                action = -action
            actions.append(action)
            
            obs, reward, terminated, truncated, info = env.step([action])
            observations.append(obs)
            abs_velocities.append(np.abs(obs[1]))
            positions.append(obs[0])
            r += reward
            
            if terminated or truncated:
                break

            i+=1

        if terminated:
            xi_star = start_loc + np.sum(abs_velocities)
            total_distance = xi_star - start_loc
            return cin, r, total_distance, obs[1], observations
        if truncated:
            return cin, np.nan, np.nan, np.nan, np.nan

    except Exception as e:
        print(f'An Exception ocurred: {e}')
        return e


def job_get_episode_reward(coeffs, kwargs):
    try:
        if 'start_loc' in kwargs:
            start_loc = kwargs['start_loc']
        else:
            start_loc = -np.pi/6
        if 'degree' in kwargs:
            degree = kwargs['degree']
        else:
            degree = 3
        if 'env_name' in kwargs:
            env_name = kwargs['env_name']
        else:
            env_name = "MountainCarContinuous-v0"
        if 'options' in kwargs:
            options = kwargs['options']

        env = gym.make(env_name)

        mrp = polynomial_agents.TrainableContinuousMRPWrapper(env, basis='chebyshev', degree=degree,
                                                                    normalize_observations=True,
                                                                    mu_coeffs=coeffs)

        r = 0.0
        observations = []
        unnormalized_obs = []
        if 'options' in kwargs:
            obs = mrp.reset(options=options)[0]
        else:
            obs = mrp.reset(options={'low': start_loc, 'high': start_loc})[0]
        observations.append(obs)
        #print(f'Starting at {mrp.unnormalize(observations[0], np.array([0.6, 0.07]), np.array([-1.2, -0.07]))}')
        for i in range(1000):
            obs, reward, terminated, truncated, info, action, _ = mrp.step(obs)
            r += reward
            observations.append(obs)
            unnormalized_obs.append(mrp.unnormalize(obs, np.array([0.6, 0.07]), np.array([-1.2, -0.07])))
            #print(reward, r)
            if terminated or truncated:
                break
        return r, observations, unnormalized_obs

    except Exception as e:
        print(f'An Exception ocurred: {e}')
        return e


def job_get_episode_reward_rl_zoo3_model(start_loc, kwargs):
    if 'algo' in kwargs:
        algo = kwargs['algo']
    else:
        raise Exception("Need algorithm.")
    if 'folder' in kwargs:
        folder = kwargs['folder']
    else:
        folder = "rl-trained-agents"
    
    try:
        model, env = get_rl_zoo3_model_and_generate_env(algo, folder)

        env.set_options({'low': start_loc, 'high': start_loc})
        env.seed(seed=0)
        obs = env.reset()

        
        try:
            if env.norm_obs: # If observations are normalized, get "unnormalized" version
                print(f'start: {env.get_original_obs()}')
            else:
                print(f'start: {obs}')
        except:
            print(f'start: {obs}')

        try:
            if env.norm_reward:
                raise Exception("Normalized reward, aborting.")
        except:
            pass

        episode_reward = 0.0
        episode_rewards, episode_lengths = [], []
        ep_len = 0
        observations = []

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = env.step(action)

            try:
                if env.norm_obs:
                    observations.append(env.get_original_obs())        
                else:
                    observations.append(obs)  
            except:
                observations.append(obs)  

            #env.render("human")

            episode_reward += reward[0]
            ep_len += 1

            if done:
                print(f"Episode Reward: {episode_reward:.2f}")
                print("Episode Length", ep_len)
                episode_rewards.append(episode_reward)
                episode_lengths.append(ep_len)
                try:
                    if env.norm_obs:
                        unnormalized_terminal_obs = infos[0]['terminal_observation']*np.sqrt(env.obs_rms.var + env.epsilon) + env.obs_rms.mean # denormalize value manually
                        v_target = unnormalized_terminal_obs[1]
                    else:
                        v_target = infos[0]['terminal_observation'][1]
                except:
                    v_target = infos[0]['terminal_observation'][1]

                return episode_reward, ep_len, v_target

    except Exception as e:
        print(f'An Exception ocurred: {e}')
        return e


def job_get_episodes_rewards_rl_zoo3_model_pendulum(kwargs):
    if 'algo' in kwargs:
        algo = kwargs['algo']
    else:
        raise Exception("Need algorithm.")
    if 'folder' in kwargs:
        folder = kwargs['folder']
    else:
        folder = "rl-trained-agents"
    if 'episodes' in kwargs:
        episodes = kwargs['episodes']
    else:
        episodes = 1
    if 'options' in kwargs:
        options = kwargs['options']
    else:
        options = None
    
    try:
        model, env = get_rl_zoo3_model_and_generate_env(algo, folder, 'Pendulum-v1')

        if 'options' in kwargs:
            # Non-clean way of working around non-deterministic initialization
            env.envs[0].env.env.env.env = custom_gymnasium.DeterministicPendulumEnv()
            env.reset()
            obs = np.array([env.envs[0].env.env.env.env.reset(options=options)[0]])
        else:
            obs = env.reset()

        try:
            if env.norm_reward:
                raise Exception("Normalized reward, aborting.")
        except:
            pass

        rewards = []

        for _ in range(episodes):
            episode_reward = 0.0
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, infos = env.step(action)

                episode_reward += reward[0]

                if done:
                    rewards.append(episode_reward)
                    break
        
        return rewards

    except Exception as e:
        print(f'An Exception ocurred: {e}')
        return e


def job_get_episode_reward_rl_zoo3_model_pendulum_deterministic_start(kwargs):
    if 'algo' in kwargs:
        algo = kwargs['algo']
    else:
        raise Exception("Need algorithm.")
    if 'start_loc' in kwargs:
        start_loc = kwargs['start_loc']
    else:
        raise Exception("Need start loc.")
    folder = "rl-trained-agents"
    options = {'x_init': start_loc[0], 'y_init': start_loc[1]}
    
    try:
        model, env = get_rl_zoo3_model_and_generate_env(algo, folder, 'Pendulum-v1')

        # Non-clean way of working around non-deterministic initialization
        env.envs[0].env.env.env.env = custom_gymnasium.DeterministicPendulumEnv()
        env.reset()
        obs = np.array([env.envs[0].env.env.env.env.reset(options=options)[0]])
        start_loc = copy.deepcopy(obs)

        try:
            if env.norm_reward:
                raise Exception("Normalized reward, aborting.")
        except:
            pass

        episode_reward = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = env.step(action)

            episode_reward += reward[0]

            if done:
                break
        
        return algo, start_loc, episode_reward

    except Exception as e:
        print(f'An Exception ocurred: {e}')
        return e


def register_deterministic_pendulum_env(deterministic_pendulum_env_name):
    # Register the environment
    register(
        id=f'{deterministic_pendulum_env_name}',
        entry_point='envs.custom_gymnasium:DeterministicPendulumEnv',  
        max_episode_steps=200,
    )


# https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
class NoDaemonProcess(mp.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(mp.get_context())):
    Process = NoDaemonProcess

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class NestablePool(mp.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(NestablePool, self).__init__(*args, **kwargs)


if __name__ == '__main__':
    p = Process(target=sweep_param)
    p.start()
