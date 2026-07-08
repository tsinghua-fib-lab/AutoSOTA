
import numpy as np
import pandas as pd
import time
from multiprocessing import Pool, cpu_count
from functools import partial
from core import serpant_algorithm, serpant_algorithm_covariate
from simulation import (
    generate_true_probs,
    generate_true_probs_with_covariates,
    evaluate_results,
    evaluate_topk_fwer_multiple_parallel,
    evaluate_fwer_multiple_parallel,
    compare_topk_methods_and_sd,
    compare_methods_and_sd_fwer_power,
    compare_methods_and_sd_with_covariates,
    compare_covariate_sd_x_effects,
    compare_original_vs_covariate_sd_x
)
from visualization import (
    plot_topk_fwer_over_time,
    plot_topk_set_size_over_time,
    plot_topk_comparison_grid,
    plot_fwer_power_comparison_grid,
    plot_fwer_power_time_analysis,
    plot_covariate_comparison_fwer_power,
    plot_covariate_sd_x_comparison,
    plot_original_vs_covariate_comparison,
    plot_covariate_sd_x_grid,
    plot_covariate_methods_comparison,
    plot_covariate_methods_grid
)


def main():
    print("\n=== Starting Top-k FWER evaluation ===\n")
    
    topk_fwer_results = evaluate_topk_fwer_multiple_parallel(
        m=10,
        alpha=0.1,
        top_k=5,
        num_simulations=1000,
        max_t=8000,
        sampling_method="tournament",
        sd=1,
        max_tournament_samples=800,
        num_cores=None,
        record_interval=100,
        random_seed=123,
        priority_mode="weighted_no_proximity"
    )
    
    fig1 = plot_topk_fwer_over_time(
        topk_fwer_results, 
        "Top-5 FWER over Time",
        save_path="topk_fwer.png"
    )
    
    fig2 = plot_topk_set_size_over_time(
        topk_fwer_results,
        "Top-5 Confidence Set Size over Time",
        save_path="topk_set_size.png"
    )
    
    print(topk_fwer_results['topk_fwer_over_time'])
    
    topk_comparison = compare_topk_methods_and_sd(
        m=20,
        alpha=0.1,
        top_k=10,
        num_simulations=1000,
        max_t=8000,
        sd_values=[0.2, 0.5, 1],
        sampling_methods=["random_pair", "tournament"],
        max_tournament_samples=800,
        num_cores=None,
        record_interval=100,
        random_seed=123
    )
    
    fig3 = plot_topk_comparison_grid(
        topk_comparison,
        alpha=0.1,
        title_prefix="Top-5 Comparison: Methods and SD",
        save_path="topk_comparison_grid.png"
    )
    

def experiment_fwer_power_comparison():
    comparison_results = compare_methods_and_sd_fwer_power(
        m=20,
        alpha=0.1,
        num_simulations=1000,
        max_t=8000,
        sd_values=[0.5, 1, 2],
        sampling_methods=["random_pair", "tournament"],
        max_tournament_samples=800,
        num_cores=None,
        record_interval=100,
        random_seed=111
    )
    
    print(comparison_results['summary_stats'].to_string(index=False))
    
    fig = plot_fwer_power_comparison_grid(
        comparison_results,
        alpha=0.1,
        title_prefix="FWER and Power: Methods x SD Comparison",
        save_path="fwer_power_comparison.png"
    )
    
    return comparison_results


def example_single_run():
    m = 10
    alpha = 0.1
    max_t = 8000
    sd = 0.1
    true_probs_info = generate_true_probs(m, sd)
    true_probs = true_probs_info['probs']
    
    result = serpant_algorithm(
        m, alpha, true_probs, max_t,
        sampling_method="random_pair",
        verbose=True,
        max_tournament_samples=400,
        top_k=3
    )
    
    eval_results = evaluate_results(result, true_probs_info, m, top_k=3)
    
    return result, eval_results


def _run_covariate_comparison_simulation(args, m, alpha, max_t, time_points,
                                          sd_x=1.0, sd_beta=0.1, sd_alpha=0.1,
                                          max_tournament_samples=400,
                                          theta_update_interval=1):
    sim_idx, seed = args
    np.random.seed(seed)
    
    true_probs_info = generate_true_probs_with_covariates(
        m=m, sd_x=sd_x, sd_beta=sd_beta, sd_alpha=sd_alpha
    )
    true_probs = true_probs_info['probs']
    
    all_correct_pairs = np.sum(true_probs > 0.5)
    
    results = {}
    
    methods = [
        ('random_pair_original', serpant_algorithm, {'sampling_method': 'random_pair'}),
        ('tournament_original', serpant_algorithm, {'sampling_method': 'tournament', 'priority_mode': 'weighted_no_proximity'}),
        ('random_pair_covariate', serpant_algorithm_covariate, {'sampling_method': 'random_pair', 'covariate_info': true_probs_info, 'theta_update_interval': theta_update_interval}),
        ('tournament_covariate', serpant_algorithm_covariate, {'sampling_method': 'tournament', 'covariate_info': true_probs_info, 'theta_update_interval': theta_update_interval, 'priority_mode': 'weighted_no_proximity'}),
    ]
    
    for method_name, algorithm, kwargs in methods:
        if 'covariate_info' in kwargs:
            covariate_info = kwargs.pop('covariate_info')
            theta_interval = kwargs.pop('theta_update_interval', 1)
            result = algorithm(
                m, alpha, true_probs, covariate_info, max_t,
                verbose=False, max_tournament_samples=max_tournament_samples,
                theta_update_interval=theta_interval, **kwargs
            )
        else:
            result = algorithm(
                m, alpha, true_probs, max_t,
                verbose=False, max_tournament_samples=max_tournament_samples, **kwargs
            )
        
        method_results = {'fwer': [], 'power': [], 'false_discoveries': []}
        
        time_idx = 0
        current_rejected = np.zeros((m, m), dtype=bool)
        
        for step_result in result['results']:
            t = step_result['time']
            rejected_pairs = step_result['rejected_pairs']
            for pair in rejected_pairs:
                current_rejected[pair[0], pair[1]] = True
            
            if time_idx < len(time_points) and t >= time_points[time_idx]:
                false_disc = 0
                correct_disc = 0
                for j in range(m):
                    for i in range(m):
                        if current_rejected[j, i]:
                            if true_probs[j, i] > 0.5:
                                correct_disc += 1
                            else:
                                false_disc += 1
                
                has_fwer = 1 if false_disc > 0 else 0
                power = correct_disc / all_correct_pairs if all_correct_pairs > 0 else 0
                
                method_results['fwer'].append(has_fwer)
                method_results['power'].append(power)
                method_results['false_discoveries'].append(false_disc)
                
                time_idx += 1
        
        while len(method_results['fwer']) < len(time_points):
            method_results['fwer'].append(method_results['fwer'][-1] if method_results['fwer'] else 0)
            method_results['power'].append(method_results['power'][-1] if method_results['power'] else 0)
            method_results['false_discoveries'].append(method_results['false_discoveries'][-1] if method_results['false_discoveries'] else 0)
        
        results[method_name] = method_results
    
    return results


def experiment_with_covariates():
    m = 10
    alpha = 0.1
    max_t = 5000
    num_simulations = 10
    record_interval = 100
    num_cores = cpu_count() - 1
    
    sd_x = 1.0
    sd_beta = 0.1
    sd_alpha = 0.1
    max_tournament_samples = 400
    theta_update_interval = 5
    
    time_points = list(range(record_interval, max_t + 1, record_interval))
    np.random.seed(123)
    sim_seeds = np.random.randint(0, 2**31 - 1, size=num_simulations)
    
    start_time = time.time()
    
    with Pool(num_cores) as pool:
        func = partial(_run_covariate_comparison_simulation, 
                      m=m, alpha=alpha, max_t=max_t, time_points=time_points,
                      sd_x=sd_x, sd_beta=sd_beta, sd_alpha=sd_alpha,
                      max_tournament_samples=max_tournament_samples,
                      theta_update_interval=theta_update_interval)
        all_results = pool.map(func, [(i+1, sim_seeds[i]) for i in range(num_simulations)])
    
    end_time = time.time()
    
    methods_list = ['random_pair_original', 'tournament_original', 
                    'random_pair_covariate', 'tournament_covariate']
    method_labels = {
        'random_pair_original': 'Random Pair (Original)',
        'tournament_original': 'Tournament (Original)',
        'random_pair_covariate': 'Random Pair (Covariate)',
        'tournament_covariate': 'Tournament (Covariate)'
    }
    
    summary = {method: {'fwer': [], 'power_mean': [], 'power_std': []} for method in methods_list}
    
    for method in methods_list:
        for t_idx in range(len(time_points)):
            fwer_values = [all_results[sim][method]['fwer'][t_idx] for sim in range(num_simulations)]
            power_values = [all_results[sim][method]['power'][t_idx] for sim in range(num_simulations)]
            
            summary[method]['fwer'].append(np.mean(fwer_values))
            summary[method]['power_mean'].append(np.mean(power_values))
            summary[method]['power_std'].append(np.std(power_values))
    
    plot_covariate_comparison_fwer_power(
        summary=summary,
        time_points=time_points,
        methods_list=methods_list,
        method_labels=method_labels,
        m=m,
        alpha=alpha,
        num_simulations=num_simulations,
        save_path="covariate_comparison_fwer_power.png"
    )
    
    for method in methods_list:
        fwer_final = summary[method]['fwer'][-1]
        power_mean_final = summary[method]['power_mean'][-1]
        power_std_final = summary[method]['power_std'][-1]
        print(f"{method_labels[method]:<30} {fwer_final:<10.3f} {power_mean_final:.3f} +/- {power_std_final:.3f}")
    
    return {
        'summary': summary,
        'time_points': time_points,
        'methods': methods_list,
        'method_labels': method_labels
    }


def experiment_m10_random_pair():
    comparison_results = compare_methods_and_sd_fwer_power(
        m=10,
        alpha=0.1,
        num_simulations=1000,
        max_t=10000,
        sd_values=[0.1, 0.2, 0.5, 1],
        sampling_methods=["random_pair"],
        max_tournament_samples=400,
        num_cores=None,
        record_interval=100,
        random_seed=123
    )

    fig = plot_fwer_power_time_analysis(
        comparison_results,
        alpha=0.1,
        m=10,
        title_prefix="FWER and Power Analysis: m=10, random_pair",
        save_path="experiment_m10_random_pair.png"
    )

    return comparison_results


def experiment_m50_random_pair():
    comparison_results = compare_methods_and_sd_fwer_power(
        m=50,
        alpha=0.1,
        num_simulations=1000,
        max_t=20000,
        sd_values=[0.2, 0.6, 1, 2],
        sampling_methods=["random_pair"],
        max_tournament_samples=800,
        num_cores=None,
        record_interval=200,
        random_seed=123
    )

    fig = plot_fwer_power_time_analysis(
        comparison_results,
        alpha=0.1,
        m=50,
        title_prefix="FWER and Power Analysis: m=50, random_pair",
        save_path="experiment_m50_random_pair.png"
    )

    return comparison_results


def experiment_topk_with_covariates():
    m = 10
    alpha = 0.1
    top_k = 5
    max_t = 5000
    sd_x = 0.5
    num_simulations = 1000
    sampling_method = "random_pair"

    from simulation import evaluate_topk_fwer_multiple_parallel

    results = evaluate_topk_fwer_multiple_parallel(
        m=m,
        alpha=alpha,
        top_k=top_k,
        num_simulations=num_simulations,
        max_t=max_t,
        sampling_method=sampling_method,
        sd=sd_x,
        max_tournament_samples=400,
        num_cores=None,
        record_interval=100,
        random_seed=123,
        use_covariates=True,
    )

    topk_data = results["topk_fwer_over_time"]

    final_row = topk_data.iloc[-1]
    plot_topk_fwer_over_time(
        results,
        title=f"Top-{top_k} FWER over Time (Covariates, sd_x={sd_x}, {sampling_method})",
        save_path="topk_fwer_covariates.png",
    )
    plot_topk_set_size_over_time(
        results,
        title=f"Top-{top_k} Set Size over Time (Covariates, sd_x={sd_x}, {sampling_method})",
        save_path="topk_set_size_covariates.png",
    )

    return results

def experiment_compare_covariates():
    comparison_results = compare_methods_and_sd_with_covariates(
        m=10,
        alpha=0.1,
        num_simulations=100,
        max_t=5000,
        sd_x_values=[0.5, 1, 2],
        sampling_methods=["random_pair", "tournament"],
        max_tournament_samples=800,
        num_cores=None,
        record_interval=100,
        random_seed=111
    )
    
    plot_data = comparison_results['combined_data'].copy()
    plot_data['sd'] = plot_data['sd_x']
    
    plot_results = {
        'combined_data': plot_data,
        'summary_stats': comparison_results['summary_stats'],
        'sd_values': comparison_results['sd_x_values'],
        'methods': comparison_results['methods']
    }
    
    fig = plot_fwer_power_comparison_grid(
        plot_results,
        alpha=0.1,
        title_prefix="FWER and Power: Methods x sd_x (Covariates)",
        save_path="fwer_power_comparison_covariates.png",
        sd_label="sd_x"
    )
    
    return comparison_results


def experiment_covariate_sd_x_comparison():
    comparison_results = compare_covariate_sd_x_effects(
        m=10,
        alpha=0.1,
        num_simulations=500,
        max_t=5000,
        sd_x_values=[0.5, 1.0, 2.0],
        sampling_method="random_pair",
        sd_beta=0.1,
        sd_alpha=0.1,
        max_tournament_samples=800,
        num_cores=None,
        theta_update_interval=10,
        record_interval=100,
        random_seed=123
    )
    
    fig = plot_covariate_sd_x_comparison(
        comparison_results,
        alpha=0.1,
        m=10,
        title_prefix="Covariate-Assisted Algorithm: sd_x Effects",
        save_path="covariate_sd_x_comparison.png"
    )
    
    return comparison_results


def experiment_original_vs_covariate_sd_x():
    comparison_results = compare_original_vs_covariate_sd_x(
        m=10,
        alpha=0.1,
        num_simulations=100,
        max_t=4000,
        sd_x_values=[0.5, 0.8, 1],
        sampling_methods=["random_pair", "tournament"],
        sd_beta=0.1,
        sd_alpha=1,
        max_tournament_samples=1000,
        num_cores=None,
        theta_update_interval=10,
        record_interval=100,
        random_seed=123,
        priority_mode="weighted_no_proximity"
    )
    
    fig1 = plot_covariate_methods_comparison(
        comparison_results,
        alpha=0.1,
        save_path="covariate_methods_comparison_facet.png"
    )
    
    fig2 = plot_covariate_methods_grid(
        comparison_results,
        alpha=0.1,
        save_path="covariate_methods_comparison_grid.png"
    )
    
    return comparison_results


def experiment_tournament_priority_modes():
    m = 10
    alpha = 0.1
    num_simulations = 100
    max_t = 6000
    sd_values = [0.2, 0.5, 1]
    max_tournament_samples = 400
    num_cores = None
    record_interval = 100
    random_seed = 123
    
    all_results = []
    summary_stats = []
    
    for sd_val in sd_values:
        for priority_mode, mode_label in [("max", "Max"), 
                                           ("weighted_no_proximity", "Weighted_Sum")]:
            print(f"\n=== Running: sd = {sd_val:.1f}, priority_mode = {mode_label} ===")
            
            fwer_results = evaluate_fwer_multiple_parallel(
                m=m, 
                alpha=alpha, 
                num_simulations=num_simulations,
                max_t=max_t, 
                sampling_method="tournament",
                sd=sd_val,
                max_tournament_samples=max_tournament_samples, 
                num_cores=num_cores,
                random_seed=random_seed,
                priority_mode=priority_mode
            )
            
            fwer_data = fwer_results['fwer_over_time']
            fwer_data['sd'] = sd_val
            fwer_data['priority_mode'] = mode_label
            all_results.append(fwer_data)
            
            final_row = fwer_data.iloc[-1]
            summary_stats.append({
                'sd': sd_val,
                'priority_mode': mode_label,
                'final_time': final_row['time'],
                'fwer': final_row['fwer'],
                'avg_power': final_row['avg_power'],
                'sd_power': final_row['sd_power']
            })
            
            print(f"  Final FWER: {final_row['fwer']:.3f}")
            print(f"  Final Power: {final_row['avg_power']:.3f} (+/- {final_row['sd_power']:.3f})")
    
    combined_data = pd.concat(all_results, ignore_index=True)
    summary_df = pd.DataFrame(summary_stats)
    
    combined_data['method'] = combined_data['priority_mode']
    
    comparison_results = {
        'combined_data': combined_data,
        'summary_stats': summary_df,
        'sd_values': sd_values,
        'priority_modes': ["Max", "Weighted_Sum"]
    }
    
    fig = plot_fwer_power_time_analysis(
        comparison_results,
        alpha=alpha,
        m=m,
        title_prefix="Tournament Priority Modes Comparison",
        save_path="tournament_priority_modes_comparison.png"
    )
    
    return comparison_results


def experiment_tournament_uncertainty_weights():
    m = 10
    alpha = 0.1
    num_simulations = 100
    max_t = 4000
    sd_values = [0.1]
    max_tournament_samples = 400
    num_cores = None
    record_interval = 100
    random_seed = 123
    
    uncertainty_weights = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    all_results = []
    summary_stats = []
    
    for sd_val in sd_values:
        for unc_weight in uncertainty_weights:
            sig_weight = 1.0 - unc_weight
            weight_label = f"unc={unc_weight:.1f}"
            
            print(f"\n=== Running: sd={sd_val:.1f}, uncertainty_weight={unc_weight:.1f}, signal_strength_weight={sig_weight:.1f} ===")
            
            fwer_results = evaluate_fwer_multiple_parallel(
                m=m, 
                alpha=alpha, 
                num_simulations=num_simulations,
                max_t=max_t, 
                sampling_method="tournament",
                sd=sd_val,
                max_tournament_samples=max_tournament_samples, 
                num_cores=num_cores,
                random_seed=random_seed,
                priority_mode="weighted_no_proximity",
                uncertainty_weight=unc_weight,
                signal_strength_weight=sig_weight
            )
            
            fwer_data = fwer_results['fwer_over_time']
            fwer_data['sd'] = sd_val
            fwer_data['uncertainty_weight'] = unc_weight
            fwer_data['weight_label'] = weight_label
            all_results.append(fwer_data)
            
            final_row = fwer_data.iloc[-1]
            summary_stats.append({
                'sd': sd_val,
                'uncertainty_weight': unc_weight,
                'signal_strength_weight': sig_weight,
                'final_time': final_row['time'],
                'fwer': final_row['fwer'],
                'avg_power': final_row['avg_power'],
                'sd_power': final_row['sd_power']
            })
            
            

    combined_data = pd.concat(all_results, ignore_index=True)
    summary_df = pd.DataFrame(summary_stats)
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    sns.set_style("whitegrid")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(uncertainty_weights)))
    
    for sd_idx, sd_val in enumerate(sd_values):
        sd_data = combined_data[combined_data['sd'] == sd_val]
        
        sd_data_filtered = sd_data[sd_data['time'] % record_interval == 0].copy()
        
        ax_fwer = axes[0]
        for unc_idx, unc_weight in enumerate(uncertainty_weights):
            weight_data = sd_data_filtered[sd_data_filtered['uncertainty_weight'] == unc_weight]
            ax_fwer.plot(weight_data['time'], weight_data['fwer'], 
                        label=f'unc={unc_weight:.1f}', color=colors[unc_idx], 
                        linewidth=2, marker='o', markersize=4)
        
        ax_fwer.axhline(y=alpha, color='red', linestyle='--', linewidth=1.5, label=f'alpha={alpha}')
        ax_fwer.set_xlabel('Time Steps', fontsize=12)
        ax_fwer.set_ylabel('FWER', fontsize=12)
        # ax_fwer.set_title(f'FWER over Time (sd={sd_val:.1f})', fontsize=14, fontweight='bold')
        ax_fwer.set_ylim(0, 0.15)
        ax_fwer.legend(fontsize=10, loc='best')
        ax_fwer.grid(True, alpha=0.3)
        
        ax_power = axes[1]
        for unc_idx, unc_weight in enumerate(uncertainty_weights):
            weight_data = sd_data_filtered[sd_data_filtered['uncertainty_weight'] == unc_weight]
            ax_power.plot(weight_data['time'], weight_data['avg_power'], 
                         label=f'unc={unc_weight:.1f}', color=colors[unc_idx], 
                         linewidth=2, marker='o', markersize=4)
        
        ax_power.set_xlabel('Time Steps', fontsize=12)
        ax_power.set_ylabel('Power', fontsize=12)
        ax_power.legend(fontsize=10, loc='best')
        ax_power.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tournament_uncertainty_weights_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'combined_data': combined_data,
        'summary_stats': summary_df,
        'sd_values': sd_values,
        'uncertainty_weights': uncertainty_weights
    }


def experiment_real_mode(config_path: str = None, checkpoint_csv: str = None):
    from real_world import load_real_mode_config, run_real_mode_experiment

    if config_path:
        config = load_real_mode_config(config_path)
        results = run_real_mode_experiment(config, checkpoint_csv=checkpoint_csv)
        return results

    from real_world import quick_test_real_mode

    results = quick_test_real_mode(
        model_names=["model_A", "model_B", "model_C"],
        questions=[
            "What is machine learning?",
            "Explain quantum computing in one sentence.",
        ],
        alpha=0.1,
        max_t=30,
        sampling_method="random_pair",
        judge_type="heuristic",
    )
    return results


def experiment_real_custom_tasks():

    model_names = ["Model_A", "Model_B", "Model_C", "Model_D", "Model_E"]
    questions = [
        "What is machine learning?", 
        "Explain quantum computing.", 
        "How to bake a cake?",
        "Write a python function to sort a list."
    ]
    
    from real_world.config import RealModeConfig, ModelConfig, JudgeConfig, OutputConfig
    from real_world import run_real_mode_experiment
    
    config_task1 = RealModeConfig(
        alpha=0.1,
        max_t=1000,
        sampling_method="random_pair",
        questions=questions,
        models=[ModelConfig(name=n, provider="stub") for n in model_names],
        judge=JudgeConfig(type="heuristic"),
        output=OutputConfig(dir="real_results/task1_pairwise"),
        verbose=False
    )
    
    results_task1 = run_real_mode_experiment(config_task1)
    
    serpant_res = results_task1["serpant_results"]
    final_rejected = serpant_res["final_rejected"]
    model_names_list = [m.name for m in config_task1.models]
    m = len(model_names_list)
    
    partial_order_list = []
    for j in range(m):
        for i in range(m):
            if final_rejected[j, i]:
                partial_order_list.append({
                    "Winner": model_names_list[j],
                    "Loser": model_names_list[i],
                    "Relationship": "Better"
                })
    
    df_partial_order = pd.DataFrame(partial_order_list)
    csv_path = "real_results/task1_pairwise/partial_order.csv"
    df_partial_order.to_csv(csv_path, index=False)
    
    k = 2
    config_task2 = RealModeConfig(
        alpha=0.1,
        max_t=1000,
        sampling_method="tournament",
        top_k=k,
        questions=questions,
        models=[ModelConfig(name=n, provider="stub") for n in model_names],
        judge=JudgeConfig(type="heuristic"),
        output=OutputConfig(dir="real_results/task2_topk"),
        verbose=False
    )
    
    results_task2 = run_real_mode_experiment(config_task2)
    
    topk_indices = results_task2["serpant_results"].get("top_k_confidence_set", [])
    topk_models = [model_names_list[i] for i in topk_indices]
    
    for name in topk_models:
        print(f"  - {name}")
    return results_task1, results_task2


def parse_args():
    import argparse
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["simulation", "real"],
        default="simulation",
        help="Mode: simulation (simulation) or real (real data)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Configuration file path for real data mode (YAML or JSON)"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Checkpoint CSV file path (for resuming, only used in real data mode)"
    )
    
    parser.add_argument(
        "--experiment",
        type=str,
        default="fwer_power",
        help="Experiment name to run in simulation mode"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if args.mode == "real":
        experiment_real_mode(args.config, checkpoint_csv=args.checkpoint)
    
    else:
        # main()
        # experiment_topk_with_covariates()
        # experiment_fwer_power_comparison()
        # example_single_run()
        # experiment_with_covariates()
        # experiment_compare_covariates()
        experiment_m10_random_pair()
        # experiment_m50_random_pair()
        # experiment_covariate_sd_x_comparison()
        # experiment_original_vs_covariate_sd_x()
        # experiment_tournament_priority_modes()
        # experiment_tournament_uncertainty_weights()
        # experiment_real_custom_tasks()
