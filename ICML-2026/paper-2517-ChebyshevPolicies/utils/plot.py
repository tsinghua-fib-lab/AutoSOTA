"""exp_run.py: Code for visualization of results.
"""
import torch
import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from collections import defaultdict
from tensorboard.backend.event_processing import event_accumulator
from utils.preprocessing import normalize
from utils.preprocessing import denormalize as denormalize_fun


__author__ = "Hannes Unger"
__version__ = "1.0"
__email__ = "hannes.unger@fh-salzburg.ac.at"


def get_logdirs_with_prefix(root_dir, prefix):
    """
    Returns a list of subdirectories in `root_dir` that start with `prefix`
    """
    all_dirs = [
        os.path.join(root_dir, d)
        for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d)) and d.startswith(prefix)
    ]
    return sorted(all_dirs)


def load_tensorboard_scalars(logdir, tag):
    """
    Loads Tensorboard event data with tag in specified logdir.
    """
    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()
    
    if tag not in ea.Tags()["scalars"]:
        raise ValueError(f"Tag '{tag}' not found in {logdir}")
    
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    
    return pd.DataFrame({'step': steps, 'value': values})


def aggregate_runs(logdirs, tag, step_resolution=1):
    """
    Aggregates all the Tensorboard test runs with tag specified in logdirs and
    returns common steps, mean, min and max vals.
    """
    all_dfs = []
    for logdir in logdirs:
        try:
            df = load_tensorboard_scalars(logdir, tag)
            df = df.groupby('step').mean().reset_index()
            all_dfs.append(df)
        except Exception as e:
            print(f'Attention: {e}')

    common_steps = np.arange(
        max(df['step'].min() for df in all_dfs),
        min(df['step'].max() for df in all_dfs),
        step_resolution
    )

    interpolated = [
        np.interp(common_steps, df['step'], df['value']) for df in all_dfs
    ]

    interpolated = np.vstack(interpolated)
    mean = np.mean(interpolated, axis=0)
    min_vals = np.min(interpolated, axis=0)
    max_vals = np.max(interpolated, axis=0)

    return common_steps, mean, min_vals, max_vals


def find_best_run_by_last_n_points(root_logdir, tag, n_points=10, step_resolution=1, folder_filter=None):
    """
    Find the test run with the highest mean value over the last n data points.
    
    Args:
        root_logdir: Root directory containing all log subdirectories
        tag: The scalar tag to analyze (e.g., 'eval/mean_reward')
        n_points: Number of last data points to consider for mean calculation (default: 10)
        step_resolution: Step resolution for interpolation (default: 1)
        folder_filter: Optional string that must be contained in folder names to consider them.
                      If None, all folders with tensorboard files are considered (default: None)
    
    Returns:
        dict: Contains the best run info with keys:
            - 'best_logdir': Path to the best performing run
            - 'best_mean': Mean value over last n points for the best run
            - 'top_five_runs': List of tuples (logdir, mean_value) for top 5 runs, sorted by performance
            - 'last_n_values': The actual last n values for the best run
            - 'discovered_logdirs': List of all discovered log directories
            - 'folder_filter': The filter that was applied
    """
    import os
    
    # Discover all subdirectories that contain tensorboard files
    logdirs = []
    for item in os.listdir(root_logdir):
        item_path = os.path.join(root_logdir, item)
        if os.path.isdir(item_path):
            # Apply folder filter if specified
            if folder_filter is not None and folder_filter not in item:
                continue
                
            # Check if directory contains tensorboard event files
            has_tb_files = any(f.startswith('events.out.tfevents') for f in os.listdir(item_path) 
                              if os.path.isfile(os.path.join(item_path, f)))
            if has_tb_files:
                logdirs.append(item_path)
    
    if not logdirs:
        filter_msg = f" matching filter '{folder_filter}'" if folder_filter else ""
        raise ValueError(f"No tensorboard log directories{filter_msg} found in {root_logdir}")
    
    filter_info = f" (filtered by '{folder_filter}')" if folder_filter else ""
    print(f"Discovered {len(logdirs)} log directories{filter_info}: {[os.path.basename(d) for d in logdirs]}")
    
    run_means = []
    
    # Load and process each run individually
    for logdir in logdirs:
        try:
            df = load_tensorboard_scalars(logdir, tag)
            df = df.groupby('step').mean().reset_index()
            
            # Get the last n_points from this run
            if len(df) >= n_points:
                last_n_values = df['value'].tail(n_points).values
                mean_last_n = np.mean(last_n_values)
                run_means.append((logdir, mean_last_n, last_n_values))
            else:
                # If run has fewer than n_points, use all available points
                all_values = df['value'].values
                mean_all = np.mean(all_values)
                run_means.append((logdir, mean_all, all_values))
                print(f'Warning: {logdir} has only {len(df)} points, using all for mean calculation')
                
        except Exception as e:
            print(f'Error processing {logdir}: {e}')
            continue
    
    if not run_means:
        raise ValueError("No valid runs found")
    
    # Find the run with highest mean
    best_run = max(run_means, key=lambda x: x[1])
    best_logdir, best_mean, last_n_values = best_run
    
    # Prepare return information with folder names only
    all_means = [(os.path.basename(logdir), mean_val) for logdir, mean_val, _ in run_means]
    # Sort by mean value (descending) and take top 5
    top_five_runs = sorted(all_means, key=lambda x: x[1], reverse=True)[:5]
    
    result = {
        'best_logdir': os.path.basename(best_logdir),
        'best_mean': best_mean,
        'top_five_runs': top_five_runs,
        'last_n_values': last_n_values,
        'n_points_used': len(last_n_values),
        'discovered_logdirs': [os.path.basename(d) for d in logdirs],
        'folder_filter': folder_filter
    }
    
    return result


def print_best_run_analysis(root_logdir, tag, n_points=10, folder_filter=None):
    """
    Convenience function to print a nice analysis of the best run retrieved via find_best_run_by_last_n_points().
    """
    result = find_best_run_by_last_n_points(root_logdir, tag, n_points, folder_filter=folder_filter)
    
    print(f"\n=== Best Run Analysis (last {result['n_points_used']} points) ===")
    print(f"Best performing run: {result['best_logdir']}")
    print(f"Mean value over last {result['n_points_used']} points: {result['best_mean']:.4f}")
    print(f"Last {result['n_points_used']} values: {result['last_n_values']}")
    
    print(f"\n=== Top 5 Runs Ranking ===")
    # Display the top 5 runs
    for i, (logdir, mean_val) in enumerate(result['top_five_runs'], 1):
        marker = " ⭐ (BEST)" if logdir == result['best_logdir'] else ""
        print(f"{i}. {logdir}: {mean_val:.4f}{marker}")


def plot_mean_min_max(ax, steps, mean, min_vals=None, max_vals=None, label="Mean", xlabel="Step", ylabel="Reward", color=None):
    """
    Plot mean curve with min-max shading on the given axis.

    Parameters:
    - ax: matplotlib.axes.Axes object
    - steps: array of x values (steps)
    - mean: array of mean values
    - min_vals: array of minimum values
    - max_vals: array of maximum values
    - label: label for the mean curve
    - color: optional color
    """
    ax.plot(steps, mean, label=label, color=color)
    if min_vals is not None and max_vals is not None:
        ax.fill_between(steps, min_vals, max_vals, alpha=0.3, color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)


def aggregate_rewards(data):
    """
    data: list of [name, reward]
    returns: list of [name, min, mean, max]
    """
    rewards_by_name = defaultdict(list)
    
    # Group rewards by name
    for name, reward in data:
        rewards_by_name[name].append(reward)
    
    result = []
    for name, rewards in rewards_by_name.items():
        min_r = min(rewards)
        max_r = max(rewards)
        mean_r = sum(rewards) / len(rewards)
        result.append([name, min_r, mean_r, max_r])
    
    return result


def best_mean_reward(aggregated_data):
    """
    aggregated_data: list of [name, min, mean, max]
    returns: [name, , min, mean, max] with the highest mean
    """
    if not aggregated_data:
        return None  
    
    best_row = max(aggregated_data, key=lambda row: row[2])  # row[2] is mean
    return [best_row[0], best_row[1], best_row[2], best_row[3]]


def get_top_5_training_results_nested(training_names, mean_rewards, top_k=5):
    """
    Get the top-k performing training results from nested data structure.
    training_names[i] correspond to mean_rewards[i].
    
    Args:
        training_names: List of lists, where training_names[i] contains names 
                       corresponding to rewards in mean_rewards[i]
        mean_rewards: List of lists, where mean_rewards[i] contains rewards
                     corresponding to names in training_names[i]
        top_k: Number of top results to return (default: 5)
    
    Returns:
        List of tuples: [(name, reward, group_index), ...] sorted by reward (highest first)
    """
    if not training_names or not mean_rewards:
        raise ValueError("Both lists must be non-empty")
    
    if len(training_names) != len(mean_rewards):
        raise ValueError("training_names and mean_rewards must have the same length")
    
    # Flatten all results into a single list
    all_results = []
    
    for group_idx, (names, rewards) in enumerate(zip(training_names, mean_rewards)):
        if len(names) != len(rewards):
            raise ValueError(f"Group {group_idx}: names and rewards must have the same length")
        
        for name, reward in zip(names, rewards):
            all_results.append((name, reward, group_idx))
    
    # Sort by reward (descending) and return top k
    sorted_results = sorted(all_results, key=lambda x: x[1], reverse=True)
    return sorted_results[:min(top_k, len(sorted_results))]


def get_top_k_per_group(training_names, mean_rewards, top_k=5):
    """
    Get top-k results from each group separately.
    
    Returns:
        List of lists: [[(name, reward), ...], ...] where each inner list 
        contains top-k results from the corresponding group
    """
    if not training_names or not mean_rewards:
        raise ValueError("Both lists must be non-empty")
    
    if len(training_names) != len(mean_rewards):
        raise ValueError("training_names and mean_rewards must have the same length")
    
    results_per_group = []
    
    for group_idx, (names, rewards) in enumerate(zip(training_names, mean_rewards)):
        if len(names) != len(rewards):
            raise ValueError(f"Group {group_idx}: names and rewards must have the same length")
        
        # Get top-k for this group
        combined = list(zip(names, rewards))
        sorted_group = sorted(combined, key=lambda x: x[1], reverse=True)
        top_k_group = sorted_group[:min(top_k, len(sorted_group))]
        
        results_per_group.append(top_k_group)
    
    return results_per_group


def print_top_overall_results(results, title="Top Overall Results"):
    """
    Pretty print overall top results.
    
    Args:
        results: List of tuples [(name, reward, group_index), ...]
        title: Title for the output
    """
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    if not results:
        print("No results to display.")
        return
    
    print(f"{'Rank':<6} {'Model Name':<25} {'Reward':<10}")
    print(f"{'-'*6} {'-'*25} {'-'*10}")
    
    for i, (name, reward, group_idx) in enumerate(results, 1):
        print(f"{i:<6} {name:<25} {reward:<10.4f} {group_idx:<8}")
    
    print(f"{'='*60}")


def print_mean_reward_of_top_results(results, title="Combined Mean Reward of Top Results"):
    """
    Pretty print the mean reward of top results.
    Args:
        results: List of tuples [(name, reward, group_index), ...]
        title: Title for the output
    """
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    if not results:
        print("No results to display.")
        return
    
    # Calculate mean reward
    rewards = [reward for _, reward, _ in results]
    mean_reward = sum(rewards) / len(rewards)
    
    print(f"Mean reward: {mean_reward:.4f}")
    print(f"{'='*60}")


def print_top_per_group_results(results_per_group, title="Top Results Per Group", group_names=None):
    """
    Pretty print top results for each group.
    
    Args:
        results_per_group: List of lists [[(name, reward), ...], ...]
        title: Title for the output
        group_names: Optional list of group names (defaults to "Group 0", "Group 1", etc.)
    """
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}")
    
    if not results_per_group:
        print("No results to display.")
        return
    
    for group_idx, group_results in enumerate(results_per_group):
        # Determine group name
        if group_names and group_idx < len(group_names):
            group_name = group_names[group_idx]
        else:
            group_name = f"Group {group_idx}"
        
        print(f"\n{group_name}")
        print(f"{'-'*len(group_name)}")
        
        if not group_results:
            print("  No results in this group.")
            continue
        
        print(f"{'Rank':<6} {'Model Name':<30} {'Reward':<10}")
        print(f"{' '*6} {'-'*30} {'-'*10}")
        
        for i, (name, reward) in enumerate(group_results, 1):
            print(f"{i:<6} {name:<30} {reward:<10.4f}")
    
    print(f"\n{'='*70}")


def print_comprehensive_results(training_names, mean_rewards, top_k_overall=5, top_k_per_group=3, group_names=None):
    """
    Print both overall and per-group results in a comprehensive format.
    
    Args:
        training_names: Nested list of training names
        mean_rewards: Nested list of corresponding rewards
        top_k_overall: Number of top results to show overall
        top_k_per_group: Number of top results to show per group
        group_names: Optional list of group names
    """
    # Get the results
    top_overall = get_top_5_training_results_nested(training_names, mean_rewards, top_k_overall)
    top_per_group = get_top_k_per_group(training_names, mean_rewards, top_k_per_group)
    
    # Print overall results
    print_top_overall_results(top_overall, f"Top {top_k_overall} Overall Results")

    # Print per-group results
    print_top_per_group_results(top_per_group, f"Top {top_k_per_group} Results Per Group", group_names)
    
    # Summary statistics
    total_models = sum(len(group) for group in training_names)
    num_groups = len(training_names)
    
    print(f"\n{'='*50}")
    print(f"{'SUMMARY':^50}")
    print(f"{'='*50}")
    print(f"Total number of groups: {num_groups}")
    print(f"Total number of models: {total_models}")
    if top_overall:
        best_model, best_reward, best_group = top_overall[0]
        print(f"Best performing model: {best_model}")
        print(f"Best reward: {best_reward:.4f}")
        print(f"From group: {best_group}")
    print(f"{'='*50}")


def print_eval_results(eval_results, training_results, env_name, print_per_group_results=False):
    names = [[r[0] for r in category] for category in training_results]
    rewards = [[r[0] for r in category] for category in eval_results]

    top_5 = get_top_5_training_results_nested(names, rewards)
    print_top_overall_results(top_5, f'{env_name}: Top evaluation results after training')
    print_mean_reward_of_top_results(top_5)
    if print_per_group_results:
        top_3_per_group = get_top_k_per_group(names, rewards, 3)
        print_top_per_group_results(top_3_per_group, f'{env_name}: Top group results after training')


def get_parallel_degree_experiment_kwargs_with_distinct_seeds(kwargs, num_experiments_per_degree, min_degree, max_degree, name_prefix, seed):
    args = []
    logdirs = []
    # num_experiments_per_degree experiments per degree with distinct seeds
    for d in range(min_degree, max_degree+1):
        deg_name = name_prefix+f'{d}'
        logdirs.append(deg_name)
        for i in range(num_experiments_per_degree):
            entry = copy.deepcopy(kwargs)
            entry['seed'] = (seed + i) * 123
            entry['name'] = deg_name
            entry['degree'] = d
            args.append(entry)
    return args, logdirs


def get_kwargs_with_distinct_seeds(kwargs, num_experiments, name_prefix, seed, degree=None):
    args = []
    logdirs = []
    # num_experiments_per_degree experiments per degree with distinct seeds
    for d in range(num_experiments):
        logdirs.append(name_prefix)
        entry = copy.deepcopy(kwargs)
        entry['seed'] = (seed + d) * 123
        entry['name'] = name_prefix
        if degree is not None:
            entry['degree'] = degree
        args.append(entry)
    return args, logdirs


def plot_results(tensorboard_dir, num_experiments, name_prefix, seed, label=None, title=None, color=None, ax=None):
    tag = 'eval/mean_reward'
    _, logdirs_chebyshev = get_kwargs_with_distinct_seeds(kwargs={}, num_experiments=num_experiments, name_prefix=name_prefix, seed=seed)

    if ax is None:
        fig, ax = plt.subplots() 
        fig.set_figwidth(10)
        fig.set_figheight(8)
    if not color:
        color='#1f77b4'
    try:
        dirs = get_logdirs_with_prefix(tensorboard_dir, logdirs_chebyshev[0])
        steps, mean, min_vals, max_vals = aggregate_runs(dirs, tag)
        plot_mean_min_max(ax, steps, mean, min_vals, max_vals, label=label, color=color)
        if title is not None:
            ax.set_title(title)
        ax.legend()
    except Exception as e:
        print(e)


def plot_tensorboard_rewards_min_mean_max(tensorboard_dirs, label=None, title=None, color=None, ax=None):
    tag = 'eval/mean_reward'

    if ax is None:
        fig, ax = plt.subplots() 
        fig.set_figwidth(10)
        fig.set_figheight(8)
    if not color:
        color='#1f77b4'
    try:
        steps, mean, min_vals, max_vals = aggregate_runs(tensorboard_dirs, tag)
        plot_mean_min_max(ax, steps, mean, min_vals, max_vals, label=label, color=color)
        if title is not None:
            ax.set_title(title)
        ax.legend()
    except Exception as e:
        print(e)


def plot_sb3_mountaincar_policy(model, ax=None, fig=None, title=None, normalized_env=True, denormalize=True, trajectory=None, trajectory_label=None, save_to_file=None, actionbar=True, actionbaraxis=None, xlim=(-1.2, 0.45), fontsize=14, fontsize_title=20, unnormalized_upper=[0.6, 0.07], unnormalized_lower=[-1.2, -0.07], xticks=[-1.0, -0.35, 0.3], yticks=[-0.06, 0.0, 0.06], hideyticks=True, yticksright=False, hideylabel=True):
    env = model.env
    env.seed(seed=0)
    
    try:
        if env.norm_obs or normalized_env: 
            normalized = True
        else:
            normalized = False
    except:
        normalized = False

    if not normalized:
    # Discretize the state space
        x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 100)
        y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 100)
    else:
        # env.observation_space.low[0] and high[0] give original bounds, not normalized ones
        # Normalize bounds
        normalized_lower = env.normalize_obs([env.observation_space.low[0], env.observation_space.low[1]])
        normalized_upper = env.normalize_obs([env.observation_space.high[0], env.observation_space.high[1]])
        x = np.linspace(normalized_lower[0], normalized_upper[0], 100)
        y = np.linspace(normalized_lower[1], normalized_upper[1], 100)

    X, Y = np.meshgrid(x, y)
    states = np.vstack([X.ravel(), Y.ravel()]).T

    # Predict actions for each state
    actions = np.array([model.predict(state, deterministic=True)[0] for state in states]).flatten()

    # Reshape to match the grid for heatmap plotting
    Z = actions.reshape(X.shape)

    levels=np.linspace(-1.0, 1.0, 21)

    # Plot the heatmap
    contourf = ax.contourf(X, Y, Z, levels=levels, cmap="viridis")  # Use a colormap like 'viridis' or 'plasma'

    if fig and actionbar:
        if actionbaraxis:
            cbar = fig.colorbar(contourf, cax=actionbaraxis, label="Action Magnitude")
            cbar.ax.set_ylabel("Action Magnitude", fontsize=fontsize)
            cbar.ax.tick_params(labelsize=fontsize)        
        else:
            cbar = fig.colorbar(contourf, label="Action Magnitude")
            cbar.ax.set_ylabel("Action Magnitude", fontsize=fontsize)
            cbar.ax.tick_params(labelsize=fontsize)

    contour = ax.contour(X, Y, Z, levels=levels, cmap="viridis")  # Adjust levels for granularity
    #ax.clabel(contour, inline=True, fontsize=8)  # Add labels to the contour lines

    # Add a specific contour for the 0.0 level with custom styling
    zero_contour = ax.contour(X, Y, Z, levels=[0.0], colors="red", linewidths=2.5, zorder=5)

    if trajectory:
        ax.plot([row[0][0] for row in trajectory], [row[0][1] for row in trajectory], label=trajectory_label, color='white', zorder=4)
        ax.legend(loc='best', fontsize=fontsize)

    ax.set_xlabel(r"$x$", fontsize=fontsize+5)
    if not hideylabel:
        ax.set_ylabel(r"$\dot{x}$", fontsize=fontsize+5)
    ax.set_title(title, fontsize=fontsize_title)
    if xticks:
        if normalized_env:
            xticks = [normalize(x, unnormalized_upper[0], unnormalized_lower[0]) for x in xticks]
        ax.set_xticks(xticks)
    if yticks:
        if normalized_env:
            yticks = [normalize(y, unnormalized_upper[1], unnormalized_lower[1]) for y in yticks]
        ax.set_yticks(yticks)
    if hideyticks:
        ax.tick_params(axis='y', which='both', labelleft=False)
    else:
        if yticksright:
            ax.yaxis.tick_right()
    ax.tick_params(axis='both', labelsize=fontsize)

    if denormalize:
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda val, _: f"{denormalize_fun(val, unnormalized_upper[0], unnormalized_lower[0]):.1f}")
        )
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda val, _: f"{denormalize_fun(val, unnormalized_upper[1], unnormalized_lower[1]):.1f}")
        )

    if save_to_file:
        plt.savefig(save_to_file)

    if fig and actionbar and not actionbaraxis:
        return cbar


def plot_sb3_chebyshev_mountaincar_policy(model, title=None, denormalize=True, save_to_file=None, actionbar=True, original_low=np.array([-1.2, -0.07]), original_high=np.array([0.6, 0.07]), normed_min_obs=-1.0, normed_max_obs=1.0, algo='ppo'):
    """
    We expect a Chebyshev policy model to have a normalized observation space ([-1, 1]) here.
    """
    if algo == 'ppo':
        fig, axes = plt.subplots(1, 3) 
        axes = axes.flatten()
        fig.set_figwidth(26)
        fig.set_figheight(6)
        #fig.suptitle(f'Approximator plots for best result out of {n_runs} runs')
    elif algo == 'sac':
        fig, axes = plt.subplots(1, 2) 
        axes = axes.flatten()
        fig.set_figwidth(18)
        fig.set_figheight(6)        
    else:
        fig, ax = plt.subplots() 
        axes = [ax]
        #axes = axes.flatten()
        #fig.set_figwidth(26)
        #fig.set_figheight(6)
        #fig.suptitle(f'Approximator plots for best result out of {n_runs} runs')

    env = model.env
    env.seed(seed=0)

    if denormalize:
        x = np.linspace(original_low[0], original_high[0], 100)
        y = np.linspace(original_low[1], original_high[1], 100)
    else:
        x = np.linspace(env.observation_space.low[0], env.observation_space.high[1], 100)
        y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 100)

    X, Y = np.meshgrid(x, y)
    states = np.vstack([X.ravel(), Y.ravel()]).T

    with torch.no_grad():
        if denormalize:
            actions = np.array([model.predict(torch.tensor(normalize(state, max_value=original_high, min_value=original_low, new_max=normed_max_obs, new_min=normed_min_obs)), deterministic=True)[0] for state in states])
            if algo == 'ppo':
                values = np.array([model.policy.predict_values(torch.tensor([normalize(state, max_value=original_high, min_value=original_low, new_max=normed_max_obs, new_min=normed_min_obs)])) for state in states])
                sigmas = np.array([model.policy.policy.evaluate_std_at(torch.tensor([normalize(state, max_value=original_high, min_value=original_low, new_max=normed_max_obs, new_min=normed_min_obs)])) for state in states])
            if algo == 'sac':
                sigmas = np.array([model.policy.actor.policy.evaluate_std_at(torch.tensor([normalize(state, max_value=original_high, min_value=original_low, new_max=normed_max_obs, new_min=normed_min_obs)])) for state in states])
        else:
            actions = np.array([model.predict(state, deterministic=True)[0] for state in states])
            if algo == 'ppo':
                values = np.array([model.policy.predict_values(torch.tensor([state])) for state in states])
                sigmas = np.array([model.policy.policy.evaluate_std_at(torch.tensor([state])) for state in states])
            if algo == 'sac':
                sigmas = np.array([model.policy.actor.policy.evaluate_std_at(torch.tensor([state])) for state in states])

    # Extract or compute the value to visualize (e.g., magnitude of action vector)
    # For a single action dimension, use `actions[:, 0]` or similar.
    #action_magnitude = np.linalg.norm(actions, axis=1)
    action_magnitude = actions[:, 0]
    Z_actions = action_magnitude.reshape(X.shape)
    levels_actions=np.linspace(min(action_magnitude), max(action_magnitude), 21)
    if algo == 'ppo':
        values_magnitude = values[:, 0]
        sigmas_magnitude = sigmas[:, 0]
        Z_values = values_magnitude.reshape(X.shape)
        Z_sigmas = sigmas_magnitude.reshape(X.shape)
        levels_values=np.linspace(min(values_magnitude), max(values_magnitude), 21)
        levels_sigmas=np.linspace(min(sigmas_magnitude), max(sigmas_magnitude), 21)
    if algo == 'sac':
        sigmas_magnitude = sigmas[:, 0]
        Z_sigmas = sigmas_magnitude.reshape(X.shape)
        levels_sigmas=np.linspace(min(sigmas_magnitude), max(sigmas_magnitude), 21)

    # Plot the heatmap
    contourf_actions = axes[0].contourf(X, Y, Z_actions, levels=levels_actions, cmap="viridis")
    contour_actions = axes[0].contour(X, Y, Z_actions, levels=levels_actions, cmap="viridis")
    axes[0].clabel(contour_actions, inline=True, fontsize=8)
    zero_contour = axes[0].contour(X, Y, Z_actions, levels=[0.0], colors="red", linewidths=2.5, zorder=5)
    if algo == 'ppo':
        contourf_values = axes[1].contourf(X, Y, Z_values, levels=levels_values, cmap="viridis")
        contour_values = axes[1].contour(X, Y, Z_values, levels=levels_values, cmap="viridis")
        #axes[1].clabel(contour_values, inline=True, fontsize=8)
        contourf_sigmas = axes[2].contourf(X, Y, Z_sigmas, levels=levels_sigmas, cmap="viridis")
        contour_sigmas = axes[2].contour(X, Y, Z_sigmas, levels=levels_sigmas, cmap="viridis")
        #axes[2].clabel(contour_sigmas, inline=True, fontsize=8)
    if algo == 'sac':
        contourf_sigmas = axes[1].contourf(X, Y, Z_sigmas, levels=levels_sigmas, cmap="viridis")
        contour_sigmas = axes[1].contour(X, Y, Z_sigmas, levels=levels_sigmas, cmap="viridis")
        #axes[1].clabel(contour_sigmas, inline=True, fontsize=8)

    plt.colorbar(contourf_actions, ax=axes[0]) 
    axes[0].set_title(r'$\pi(s)$', fontsize=18)
    if algo == 'ppo':
        plt.colorbar(contourf_values, ax=axes[1]) 
        plt.colorbar(contourf_sigmas, ax=axes[2]) 
        axes[1].set_title(r'$v_{\pi}(s)$', fontsize=18)
        axes[2].set_title(r'$\sigma$', fontsize=18)
    if algo == 'sac':
        plt.colorbar(contourf_sigmas, ax=axes[1]) 
        axes[1].set_title(r'$\sigma$', fontsize=18)

    for ax in axes:
        ax.set_xlabel(r"$x$", fontsize=14)
        ax.set_ylabel(r"$\dot{x}$", fontsize=14)
        ax.tick_params(axis='both', labelsize=14)

    if save_to_file:
        plt.savefig(save_to_file)