
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial
import time
from core.algorithm import serpant_algorithm, serpant_algorithm_covariate
from .data_generator import generate_true_probs, generate_true_probs_with_covariates
from utils.helpers import safe_mean, safe_sd


def run_single_simulation_with_seed(args, m, alpha, max_t, sampling_method, sd, 
                                   max_tournament_samples, top_k=None, record_interval=100,
                                   use_covariates=False, priority_mode="weighted",
                                   uncertainty_weight=0.2, signal_strength_weight=0.8):

    sim_idx, seed = args
    np.random.seed(seed)
    
    if use_covariates:
        true_probs_info = generate_true_probs_with_covariates(
            m=m, 
            sd_x=sd,
            sd_beta=0.1,
            sd_alpha=0.1
        )
    else:
        true_probs_info = generate_true_probs(m, sd)
    
    true_probs = true_probs_info['probs']
    
    all_correct_pairs = np.sum((true_probs > 0.5) & ~np.eye(m, dtype=bool))
    
    result = serpant_algorithm(
        m=m,
        alpha=alpha,
        true_probs=true_probs,
        max_t=max_t,
        sampling_method=sampling_method,
        verbose=False,
        max_tournament_samples=max_tournament_samples,
        top_k=top_k,
        priority_mode=priority_mode,
        uncertainty_weight=uncertainty_weight,
        signal_strength_weight=signal_strength_weight
    )
    
    sim_results = []
    cumulative_rejected_pairs = np.zeros((m, m), dtype=bool)
    
    for t_result in result['results']:
        rejected_idx = t_result['rejected_pairs']
        if len(rejected_idx) > 0:
            cumulative_rejected_pairs[rejected_idx[:, 0], rejected_idx[:, 1]] = True
        
        t = t_result['time']
        if t % record_interval != 0:
            continue
        
        false_rejections = 0
        correct_rejections = 0
        if cumulative_rejected_pairs.sum() > 0:
            rejected_idx = np.argwhere(cumulative_rejected_pairs)
            for j, i in rejected_idx:
                if true_probs[j, i] <= 0.5:
                    false_rejections += 1
                else:
                    correct_rejections += 1
        
        current_power = correct_rejections / all_correct_pairs if all_correct_pairs > 0 else 0
        
        if top_k is not None:
            topk_set = t_result['top_k_set'] if t_result['top_k_set'] is not None else []
            true_topk = true_probs_info['true_ranking'][:top_k]
            missing_topk = list(set(true_topk) - set(topk_set))
            
            sim_results.append({
                'simulation': sim_idx,
                'time': t_result['time'],
                'topk_set_size': len(topk_set),
                'missing_count': len(missing_topk),
                'has_missing': len(missing_topk) > 0,
                'rejected_count': t_result['rejected_count'],
                'active_count': t_result['active_count'],
                'round_rejections': t_result['round_rejections'],
                'false_rejections': false_rejections,
                'correct_rejections': correct_rejections,
                'power': current_power,
                'has_false_rejection': false_rejections > 0
            })
        else:
            sim_results.append({
                'simulation': sim_idx,
                'time': t_result['time'],
                'rejected_count': t_result['rejected_count'],
                'active_count': t_result['active_count'],
                'round_rejections': t_result['round_rejections'],
                'false_rejections': false_rejections,
                'correct_rejections': correct_rejections,
                'power': current_power,
                'has_false_rejection': false_rejections > 0
            })
    
    return pd.DataFrame(sim_results)


def evaluate_fwer_multiple_parallel(m, alpha, num_simulations=100, max_t=1000,
                                    sampling_method="all_active", sd=4,
                                    max_tournament_samples=60, num_cores=None,
                                    random_seed=None, use_covariates=False, priority_mode="weighted",
                                    uncertainty_weight=0.2, signal_strength_weight=0.8):

    if num_cores is None:
        num_cores = cpu_count() - 1
    
    print(f"Using {num_cores} cores for parallel computation")
    
    if random_seed is not None:
        np.random.seed(random_seed)
    sim_seeds = np.random.randint(0, 2**31 - 1, size=num_simulations)
    
    start_time = time.time()
    
    with Pool(num_cores) as pool:
        func = partial(run_single_simulation_with_seed, m=m, alpha=alpha, max_t=max_t,
                      sampling_method=sampling_method, sd=sd,
                      max_tournament_samples=max_tournament_samples,
                      use_covariates=use_covariates, priority_mode=priority_mode,
                      uncertainty_weight=uncertainty_weight,
                      signal_strength_weight=signal_strength_weight)
        all_results = pool.map(func, [(i+1, sim_seeds[i]) for i in range(num_simulations)])
    
    end_time = time.time()
    print(f"Parallel computation completed in {end_time - start_time:.2f} seconds")
    
    combined_results = pd.concat(all_results, ignore_index=True)
    
    fwer_over_time = combined_results.groupby('time').agg({
        'has_false_rejection': 'mean',
        'false_rejections': ['mean', 'std'],
        'correct_rejections': 'mean',
        'power': ['mean', 'std'],
        'rejected_count': ['mean', 'std'],
        'simulation': 'nunique'
    }).reset_index()
    
    fwer_over_time.columns = ['time', 'fwer', 'avg_false_rejections', 'sd_false_rejections',
                              'avg_correct_rejections', 'avg_power', 'sd_power',
                              'avg_total_rejections', 'sd_total_rejections', 'n_simulations']
    
    final_results = combined_results.groupby('simulation').last().reset_index()
    
    return {
        'fwer_over_time': fwer_over_time,
        'detailed_results': combined_results,
        'empirical_fwer': fwer_over_time['fwer'].iloc[-1],
        'avg_final_rejections': final_results['rejected_count'].mean(),
        'sd_final_rejections': final_results['rejected_count'].std(),
        'avg_final_power': final_results['power'].mean(),
        'sd_final_power': final_results['power'].std(),
        'final_rejections': final_results['rejected_count'].values,
        'final_powers': final_results['power'].values
    }


def evaluate_topk_fwer_multiple_parallel(m, alpha, top_k, num_simulations=100,
                                        max_t=1000, sampling_method="all_active",
                                        sd=4, max_tournament_samples=60,
                                        num_cores=None, record_interval=100,
                                        random_seed=None, use_covariates=False,
                                        priority_mode="weighted"):
    if num_cores is None:
        num_cores = cpu_count() - 1
    
    print(f"Using {num_cores} cores for parallel computation")
    
    if random_seed is not None:
        np.random.seed(random_seed)
    sim_seeds = np.random.randint(0, 2**31 - 1, size=num_simulations)
    
    start_time = time.time()
    
    with Pool(num_cores) as pool:
        func = partial(
            run_single_simulation_with_seed,
            m=m,
            alpha=alpha,
            max_t=max_t,
            sampling_method=sampling_method,
            sd=sd,
            max_tournament_samples=max_tournament_samples,
            top_k=top_k,
            record_interval=record_interval,
            use_covariates=use_covariates,
            priority_mode=priority_mode,
        )
        all_results = pool.map(
            func, [(i + 1, sim_seeds[i]) for i in range(num_simulations)]
        )
    
    end_time = time.time()
    print(f"Parallel computation completed in {end_time - start_time:.2f} seconds")
    
    combined_results = pd.concat(all_results, ignore_index=True)
    
    topk_fwer_over_time = combined_results.groupby("time").agg(
        {
            "has_missing": "mean",
            "topk_set_size": ["mean", "std"],
            "missing_count": ["mean", "std"],
            "has_false_rejection": "mean",
            "power": ["mean", "std"],
            "simulation": "nunique",
        }
    ).reset_index()

    topk_fwer_over_time.columns = [
        "time",
        "topk_fwer",
        "avg_set_size",
        "sd_set_size",
        "avg_missing_count",
        "sd_missing_count",
        "fwer",
        "avg_power",
        "sd_power",
        "n_simulations",
    ]
    
    return {
        "topk_fwer_over_time": topk_fwer_over_time,
        "detailed_results": combined_results,
    }


def compare_sd_effects_parallel(m, alpha, num_simulations=50, max_t=2000,
                               sampling_method="random_pair", sd_values=[1, 2, 4],
                               max_tournament_samples=60, num_cores=None,
                               random_seed=123):
    if num_cores is None:
        num_cores = cpu_count() - 1
    
    all_sd_results = []
    summary_stats = []
    
    for sd_val in sd_values:
        print(f"\n=== Running parallel simulations with sd = {sd_val:.1f} ===")
        
        fwer_results = evaluate_fwer_multiple_parallel(
            m=m, alpha=alpha, num_simulations=num_simulations,
            max_t=max_t, sampling_method=sampling_method, sd=sd_val,
            max_tournament_samples=max_tournament_samples, num_cores=num_cores,
            random_seed=random_seed
        )
        
        fwer_data = fwer_results['fwer_over_time'].copy()
        fwer_data['sd'] = sd_val
        all_sd_results.append(fwer_data)
        
        summary_stats.append({
            'sd': sd_val,
            'final_fwer': fwer_data['fwer'].iloc[-1],
            'final_power': fwer_data['avg_power'].iloc[-1],
            'final_rejections': fwer_data['avg_total_rejections'].iloc[-1]
        })
        
        print(f"sd = {sd_val:.1f}: Final FWER = {fwer_data['fwer'].iloc[-1]:.3f}, "
              f"Final Power = {fwer_data['avg_power'].iloc[-1]:.3f}, "
              f"Final Rejections = {fwer_data['avg_total_rejections'].iloc[-1]:.1f}")
    
    combined_data = pd.concat(all_sd_results, ignore_index=True)
    summary_stats_df = pd.DataFrame(summary_stats)
    
    return {
        'combined_data': combined_data,
        'summary_stats': summary_stats_df
    }


def compare_sampling_methods_power(m, alpha, num_simulations=50, max_t=2000,
                                  sd=4, max_tournament_samples=60, num_cores=None,
                                  random_seed=123):
    if num_cores is None:
        num_cores = cpu_count() - 1
    
    all_method_results = []
    summary_stats = []
    sampling_methods = ["random_pair", "tournament"]
    
    for method in sampling_methods:
        print(f"\n=== Running parallel simulations with method = {method} ===")
        
        fwer_results = evaluate_fwer_multiple_parallel(
            m=m, alpha=alpha, num_simulations=num_simulations,
            max_t=max_t, sampling_method=method, sd=sd,
            max_tournament_samples=max_tournament_samples, num_cores=num_cores,
            random_seed=random_seed
        )
        
        fwer_data = fwer_results['fwer_over_time'].copy()
        fwer_data['method'] = method
        all_method_results.append(fwer_data)
        
        summary_stats.append({
            'method': method,
            'final_fwer': fwer_data['fwer'].iloc[-1],
            'final_power': fwer_data['avg_power'].iloc[-1],
            'final_rejections': fwer_data['avg_total_rejections'].iloc[-1]
        })
        
        print(f"Method = {method}: Final FWER = {fwer_data['fwer'].iloc[-1]:.3f}, "
              f"Final Power = {fwer_data['avg_power'].iloc[-1]:.3f}, "
              f"Final Rejections = {fwer_data['avg_total_rejections'].iloc[-1]:.1f}")
    
    combined_data = pd.concat(all_method_results, ignore_index=True)
    summary_stats_df = pd.DataFrame(summary_stats)
    
    return {
        'combined_data': combined_data,
        'summary_stats': summary_stats_df
    }


def compare_methods_and_sd_fwer_power(m, alpha, num_simulations=100, max_t=3000,
                                     sd_values=[1, 2, 4], 
                                     sampling_methods=["random_pair", "tournament"],
                                     max_tournament_samples=100, num_cores=None,
                                     record_interval=100, random_seed=123,
                                     priority_mode="weighted_no_proximity"):
    if num_cores is None:
        num_cores = cpu_count() - 1
    
    all_results = []
    summary_stats = []
    
    for sd_val in sd_values:
        for method in sampling_methods:
            print(f"\n=== Running: sd = {sd_val:.1f}, method = {method} ===")
            
            fwer_results = evaluate_fwer_multiple_parallel(
                m=m, alpha=alpha, num_simulations=num_simulations,
                max_t=max_t, sampling_method=method, sd=sd_val,
                max_tournament_samples=max_tournament_samples, num_cores=num_cores,
                random_seed=random_seed, priority_mode=priority_mode
            )
            
            fwer_data = fwer_results['fwer_over_time'].copy()
            sampled_times = [t for t in fwer_data['time'] if t % record_interval == 0 or t == fwer_data['time'].max()]
            fwer_data = fwer_data[fwer_data['time'].isin(sampled_times)].copy()
            
            fwer_data['sd'] = sd_val
            fwer_data['method'] = method
            all_results.append(fwer_data)
            
            summary_stats.append({
                'sd': sd_val,
                'method': method,
                'final_fwer': fwer_data['fwer'].iloc[-1],
                'final_power': fwer_data['avg_power'].iloc[-1],
                'final_power_sd': fwer_data['sd_power'].iloc[-1],
                'final_rejections': fwer_data['avg_total_rejections'].iloc[-1]
            })
            
            print(f"Completed: Final FWER = {fwer_data['fwer'].iloc[-1]:.3f}, "
                  f"Final Power = {fwer_data['avg_power'].iloc[-1]:.3f}")
    
    combined_data = pd.concat(all_results, ignore_index=True)
    summary_stats_df = pd.DataFrame(summary_stats)
    
    return {
        'combined_data': combined_data,
        'summary_stats': summary_stats_df,
        'sd_values': sd_values,
        'methods': sampling_methods
    }


def compare_topk_methods_and_sd(m, alpha, top_k, num_simulations=100, max_t=3000,
                               sd_values=[1, 2, 4], sampling_methods=["random_pair", "tournament"],
                               max_tournament_samples=100, num_cores=None, record_interval=100,
                               random_seed=123, priority_mode="weighted_no_proximity"):
    if num_cores is None:
        num_cores = cpu_count() - 1
    
    all_results = []
    
    for sd_val in sd_values:
        for method in sampling_methods:
            print(f"\n=== Running: sd = {sd_val:.1f}, method = {method} ===")
            
            result = evaluate_topk_fwer_multiple_parallel(
                m=m, alpha=alpha, top_k=top_k, num_simulations=num_simulations,
                max_t=max_t, sampling_method=method, sd=sd_val,
                max_tournament_samples=max_tournament_samples,
                num_cores=num_cores, record_interval=record_interval,
                random_seed=random_seed, priority_mode=priority_mode
            )
            
            result_data = result['topk_fwer_over_time'].copy()
            result_data['sd'] = sd_val
            result_data['method'] = method
            all_results.append(result_data)
            
            print(f"Completed: Final FWER = {result_data['topk_fwer'].iloc[-1]:.3f}, "
                  f"Final Set Size = {result_data['avg_set_size'].iloc[-1]:.1f}")
    
    combined_data = pd.concat(all_results, ignore_index=True)
    
    return {
        'combined_data': combined_data,
        'sd_values': sd_values,
        'methods': sampling_methods
    }


def compare_methods_and_sd_with_covariates(m, alpha, num_simulations=100, max_t=3000,
                                           sd_x_values=[0.5, 1, 2], 
                                           sampling_methods=["random_pair", "tournament"],
                                           max_tournament_samples=100, num_cores=None,
                                           record_interval=100, random_seed=123):
    if num_cores is None:
        num_cores = cpu_count() - 1
    
    all_results = []
    summary_stats = []
    
    for sd_x in sd_x_values:
        for method in sampling_methods:
            print(f"\n=== Running (Covariates): sd_x = {sd_x:.1f}, method = {method} ===")
            
            fwer_results = evaluate_fwer_multiple_parallel(
                m=m, alpha=alpha, num_simulations=num_simulations,
                max_t=max_t, sampling_method=method, sd=sd_x,
                max_tournament_samples=max_tournament_samples, num_cores=num_cores,
                random_seed=random_seed, use_covariates=True
            )
            
            fwer_data = fwer_results['fwer_over_time'].copy()
            sampled_times = [t for t in fwer_data['time'] if t % record_interval == 0 or t == fwer_data['time'].max()]
            fwer_data = fwer_data[fwer_data['time'].isin(sampled_times)].copy()
            
            fwer_data['sd_x'] = sd_x
            fwer_data['method'] = method
            all_results.append(fwer_data)
            
            summary_stats.append({
                'sd_x': sd_x,
                'method': method,
                'final_fwer': fwer_data['fwer'].iloc[-1],
                'final_power': fwer_data['avg_power'].iloc[-1],
                'final_power_sd': fwer_data['sd_power'].iloc[-1],
                'final_rejections': fwer_data['avg_total_rejections'].iloc[-1]
            })
            
            print(f"Completed: Final FWER = {fwer_data['fwer'].iloc[-1]:.3f}, "
                  f"Final Power = {fwer_data['avg_power'].iloc[-1]:.3f}")
    
    combined_data = pd.concat(all_results, ignore_index=True)
    summary_stats_df = pd.DataFrame(summary_stats)
    
    return {
        'combined_data': combined_data,
        'summary_stats': summary_stats_df,
        'sd_x_values': sd_x_values,
        'methods': sampling_methods
    }


def _run_covariate_assisted_simulation(args, m, alpha, max_t, sd_x, sd_beta, sd_alpha,
                                        sampling_method, max_tournament_samples,
                                        theta_update_interval, record_interval,
                                        priority_mode="weighted_no_proximity"):

    sim_idx, seed = args
    np.random.seed(seed)
    
    true_probs_info = generate_true_probs_with_covariates(
        m=m, sd_x=sd_x, sd_beta=sd_beta, sd_alpha=sd_alpha
    )
    true_probs = true_probs_info['probs']
    
    all_correct_pairs = np.sum((true_probs > 0.5) & ~np.eye(m, dtype=bool))
    
    result = serpant_algorithm_covariate(
        m=m,
        alpha=alpha,
        true_probs=true_probs,
        covariate_info=true_probs_info,
        max_t=max_t,
        sampling_method=sampling_method,
        verbose=False,
        max_tournament_samples=max_tournament_samples,
        theta_update_interval=theta_update_interval,
        priority_mode=priority_mode
    )
    
    sim_results = []
    cumulative_rejected_pairs = np.zeros((m, m), dtype=bool)
    
    for t_result in result['results']:
        rejected_idx = t_result['rejected_pairs']
        if len(rejected_idx) > 0:
            cumulative_rejected_pairs[rejected_idx[:, 0], rejected_idx[:, 1]] = True
        
        false_rejections = 0
        correct_rejections = 0
        if cumulative_rejected_pairs.sum() > 0:
            rejected_idx = np.argwhere(cumulative_rejected_pairs)
            for j, i in rejected_idx:
                if true_probs[j, i] <= 0.5:
                    false_rejections += 1
                else:
                    correct_rejections += 1
        
        current_power = correct_rejections / all_correct_pairs if all_correct_pairs > 0 else 0
        
        sim_results.append({
            'simulation': sim_idx,
            'time': t_result['time'],
            'rejected_count': t_result['rejected_count'],
            'active_count': t_result['active_count'],
            'round_rejections': t_result['round_rejections'],
            'false_rejections': false_rejections,
            'correct_rejections': correct_rejections,
            'power': current_power,
            'has_false_rejection': false_rejections > 0
        })
    
    return pd.DataFrame(sim_results)


def evaluate_covariate_assisted_fwer_parallel(m, alpha, num_simulations=100, max_t=3000,
                                              sampling_method="random_pair", sd_x=1.0,
                                              sd_beta=0.1, sd_alpha=0.1,
                                              max_tournament_samples=100, num_cores=None,
                                              theta_update_interval=1, record_interval=100,
                                              random_seed=123, priority_mode="weighted_no_proximity"):
    if num_cores is None:
        num_cores = cpu_count() - 1
    
    print(f"Using {num_cores} cores for parallel computation")
    
    if random_seed is not None:
        np.random.seed(random_seed)
    sim_seeds = np.random.randint(0, 2**31 - 1, size=num_simulations)
    
    start_time = time.time()
    
    with Pool(num_cores) as pool:
        func = partial(_run_covariate_assisted_simulation, 
                      m=m, alpha=alpha, max_t=max_t, sd_x=sd_x,
                      sd_beta=sd_beta, sd_alpha=sd_alpha,
                      sampling_method=sampling_method,
                      max_tournament_samples=max_tournament_samples,
                      theta_update_interval=theta_update_interval,
                      record_interval=record_interval,
                      priority_mode=priority_mode)
        all_results = pool.map(func, [(i+1, sim_seeds[i]) for i in range(num_simulations)])
    
    end_time = time.time()
    print(f"Parallel computation completed in {end_time - start_time:.2f} seconds")
    
    combined_results = pd.concat(all_results, ignore_index=True)
    
    fwer_over_time = combined_results.groupby('time').agg({
        'has_false_rejection': 'mean',
        'false_rejections': ['mean', 'std'],
        'correct_rejections': 'mean',
        'power': ['mean', 'std'],
        'rejected_count': ['mean', 'std'],
        'simulation': 'nunique'
    }).reset_index()
    
    fwer_over_time.columns = ['time', 'fwer', 'avg_false_rejections', 'sd_false_rejections',
                              'avg_correct_rejections', 'avg_power', 'sd_power',
                              'avg_total_rejections', 'sd_total_rejections', 'n_simulations']
    
    sampled_times = [t for t in fwer_over_time['time'] if t % record_interval == 0 or t == fwer_over_time['time'].max()]
    fwer_over_time = fwer_over_time[fwer_over_time['time'].isin(sampled_times)].copy()
    
    return {
        'fwer_over_time': fwer_over_time,
        'detailed_results': combined_results,
        'empirical_fwer': fwer_over_time['fwer'].iloc[-1],
        'avg_final_power': fwer_over_time['avg_power'].iloc[-1],
        'sd_final_power': fwer_over_time['sd_power'].iloc[-1]
    }


def compare_covariate_sd_x_effects(m, alpha, num_simulations=100, max_t=3000,
                                    sd_x_values=[0.5, 1.0, 2.0],
                                    sampling_method="random_pair",
                                    sd_beta=0.1, sd_alpha=0.1,
                                    max_tournament_samples=100, num_cores=None,
                                    theta_update_interval=1, record_interval=100,
                                    random_seed=123):
    if num_cores is None:
        num_cores = cpu_count() - 1
    
    all_results = []
    summary_stats = []
    
    for sd_x in sd_x_values:
        print(f"\n=== Running Covariate-Assisted: sd_x = {sd_x:.1f}, method = {sampling_method} ===")
        
        fwer_results = evaluate_covariate_assisted_fwer_parallel(
            m=m, alpha=alpha, num_simulations=num_simulations,
            max_t=max_t, sampling_method=sampling_method, sd_x=sd_x,
            sd_beta=sd_beta, sd_alpha=sd_alpha,
            max_tournament_samples=max_tournament_samples, num_cores=num_cores,
            theta_update_interval=theta_update_interval,
            record_interval=record_interval, random_seed=random_seed
        )
        
        fwer_data = fwer_results['fwer_over_time'].copy()
        fwer_data['sd_x'] = sd_x
        fwer_data['method'] = sampling_method
        fwer_data['algorithm'] = 'covariate_assisted'
        all_results.append(fwer_data)
        
        summary_stats.append({
            'sd_x': sd_x,
            'method': sampling_method,
            'algorithm': 'covariate_assisted',
            'final_fwer': fwer_data['fwer'].iloc[-1],
            'final_power': fwer_data['avg_power'].iloc[-1],
            'final_power_sd': fwer_data['sd_power'].iloc[-1]
        })
        
        print(f"Completed: Final FWER = {fwer_data['fwer'].iloc[-1]:.3f}, "
              f"Final Power = {fwer_data['avg_power'].iloc[-1]:.3f}")
    
    combined_data = pd.concat(all_results, ignore_index=True)
    summary_stats_df = pd.DataFrame(summary_stats)
    
    return {
        'combined_data': combined_data,
        'summary_stats': summary_stats_df,
        'sd_x_values': sd_x_values,
        'method': sampling_method
    }


def compare_original_vs_covariate_sd_x(m, alpha, num_simulations=100, max_t=3000,
                                        sd_x_values=[0.5, 1.0, 2.0],
                                        sampling_methods=["random_pair", "tournament"],
                                        sd_beta=0.1, sd_alpha=0.1,
                                        max_tournament_samples=100, num_cores=None,
                                        theta_update_interval=1, record_interval=100,
                                        random_seed=123, priority_mode="weighted_no_proximity"):
    if num_cores is None:
        num_cores = cpu_count() - 1
    
    if isinstance(sampling_methods, str):
        sampling_methods = [sampling_methods]
    
    all_results = []
    summary_stats = []
    
    for sd_x in sd_x_values:
        for sampling_method in sampling_methods:
            print(f"\n=== Running Original Algorithm: sd_x = {sd_x:.1f}, method = {sampling_method} ===")
            
            original_results = evaluate_fwer_multiple_parallel(
                m=m, alpha=alpha, num_simulations=num_simulations,
                max_t=max_t, sampling_method=sampling_method, sd=sd_x,
                max_tournament_samples=max_tournament_samples, num_cores=num_cores,
                random_seed=random_seed, use_covariates=True, priority_mode=priority_mode
            )
            
            original_data = original_results['fwer_over_time'].copy()
            sampled_times = [t for t in original_data['time'] if t % record_interval == 0 or t == original_data['time'].max()]
            original_data = original_data[original_data['time'].isin(sampled_times)].copy()
            original_data['sd_x'] = sd_x
            original_data['method'] = sampling_method
            original_data['algorithm'] = 'original'
            all_results.append(original_data)
            
            summary_stats.append({
                'sd_x': sd_x,
                'method': sampling_method,
                'algorithm': 'original',
                'final_fwer': original_data['fwer'].iloc[-1],
                'final_power': original_data['avg_power'].iloc[-1],
                'final_power_sd': original_data['sd_power'].iloc[-1]
            })
            
            print(f"Original ({sampling_method}) - Final FWER = {original_data['fwer'].iloc[-1]:.3f}, "
                  f"Final Power = {original_data['avg_power'].iloc[-1]:.3f}")
            
            print(f"\n=== Running Covariate-Assisted Algorithm: sd_x = {sd_x:.1f}, method = {sampling_method} ===")
            
            covariate_results = evaluate_covariate_assisted_fwer_parallel(
                m=m, alpha=alpha, num_simulations=num_simulations,
                max_t=max_t, sampling_method=sampling_method, sd_x=sd_x,
                sd_beta=sd_beta, sd_alpha=sd_alpha,
                max_tournament_samples=max_tournament_samples, num_cores=num_cores,
                theta_update_interval=theta_update_interval,
                record_interval=record_interval, random_seed=random_seed,
                priority_mode=priority_mode
            )
            
            covariate_data = covariate_results['fwer_over_time'].copy()
            covariate_data['sd_x'] = sd_x
            covariate_data['method'] = sampling_method
            covariate_data['algorithm'] = 'covariate_assisted'
            all_results.append(covariate_data)
            
            summary_stats.append({
                'sd_x': sd_x,
                'method': sampling_method,
                'algorithm': 'covariate_assisted',
                'final_fwer': covariate_data['fwer'].iloc[-1],
                'final_power': covariate_data['avg_power'].iloc[-1],
                'final_power_sd': covariate_data['sd_power'].iloc[-1]
            })
            
            print(f"Covariate ({sampling_method}) - Final FWER = {covariate_data['fwer'].iloc[-1]:.3f}, "
                  f"Final Power = {covariate_data['avg_power'].iloc[-1]:.3f}")
    
    combined_data = pd.concat(all_results, ignore_index=True)
    summary_stats_df = pd.DataFrame(summary_stats)
    
    return {
        'combined_data': combined_data,
        'summary_stats': summary_stats_df,
        'sd_x_values': sd_x_values,
        'sampling_methods': sampling_methods
    }

