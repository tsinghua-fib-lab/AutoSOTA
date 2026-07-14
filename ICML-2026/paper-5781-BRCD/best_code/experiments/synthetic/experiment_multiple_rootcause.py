import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import logging
# Reset root logging configuration
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.WARNING)
import models.brcd_k as brcd
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import pyAgrum as gum
import argparse
import time
import random
import networkx as nx
from models.rcd.rcd import top_k_rc
from models.epsilon_diagnosis import e_diagnosis
from models.CausalRCA_code.causalRCA import causalRCA
from models.automap import auto_map
from baro.root_cause_analysis import robust_scorer
from utils import get_cpdag, compute_mutual_information_ranking, pdag_to_causallearn_generalgraph
from models.rcg.rcg import rank_variables
import re
from collections import defaultdict
from models.SCORE_ORDERING import score_ordering
from dowhy.gcm import RescaledMedianCDFQuantileScorer, ITAnomalyScorer
import gc
import torch
from causallearn.search.ConstraintBased.PC import pc
from sklearn.preprocessing import KBinsDiscretizer
from multiprocessing import Process, Queue
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
import multiprocessing as mp
ctx = mp.get_context("spawn") 
from itertools import cycle
from graphical_models.classes.dags.pdag import PDAG
from causallearn.graph.Endpoint import Endpoint

def causal_learn_graph_to_nx_digraph(G_cl, column_names):
    G_nx = nx.DiGraph()
    id_to_col = {i: name for i, name in enumerate(column_names)}
    for node in G_cl.get_nodes():
        node_id = G_cl.node_map[node]
        G_nx.add_node(id_to_col[node_id])
    arcs = []
    edges = []
    for edge in G_cl.get_graph_edges():
        n1 = id_to_col[G_cl.node_map[edge.node1]]
        n2 = id_to_col[G_cl.node_map[edge.node2]]
        ep1 = edge.endpoint1
        ep2 = edge.endpoint2
        if ep1 == Endpoint.TAIL and ep2 == Endpoint.ARROW:
            arcs.append((n1, n2))
        elif ep2 ==  Endpoint.TAIL and ep1 == Endpoint.ARROW:
            arcs.append((n2, n1))
        elif ep2 ==  Endpoint.TAIL and ep1 == Endpoint.TAIL:
            edges.append((n2, n1))
    return arcs, edges

def _causalrca_worker(q, df_int, obs_cols):
    """Child process so we can hard-kill on timeout."""
    try:
        # re-import inside child
        from models.CausalRCA_code.causalRCA import causalRCA
        res = causalRCA(df_int.astype(int), metric_names=None)
        crca_root_causes = res.get('root_cause', [])
        crca_indices = [list(obs_cols).index(r) for r in crca_root_causes]
        q.put(("ok", (crca_root_causes, crca_indices)))
    except Exception as e:
        q.put(("err", repr(e)))


def _run_worker(q, root_path, data_path, true_root_cause):
    """Child process: run RUN_model_wrapper and return result (or error) via queue."""
    try:
        # re-import inside the child to avoid pickling issues
        from models.RUN.model_wrapper import RUN_model_wrapper
        # ensure project root is importable if needed
        sys.path.append(os.getcwd())
        out = RUN_model_wrapper(root_path, data_path, true_root_cause)
        q.put(("ok", out))
    except Exception as e:
        q.put(("err", repr(e)))


def load_experiment_data(exp_dir):
    """
    Load data from an experiment directory.
    """
    # Load observational and interventional data
    df_obs = pd.read_csv(os.path.join(exp_dir, 'observational_data.csv'))
    df_int = pd.read_csv(os.path.join(exp_dir, 'interventional_data.csv'))
    output_path = os.path.join(exp_dir, "combined.csv")
    if not os.path.exists(output_path):
        # 2) concatenate them (resetting the index so it's clean)
        combined = pd.concat([df_obs, df_int], ignore_index=True)
        # 3) write to CSV
        combined.to_csv(output_path, index=False)
    # Load metadata
    with open(os.path.join(exp_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    # Load Bayesian networks
    obs_bn = gum.loadBN(os.path.join(exp_dir, 'observational_bn.bif'))
    int_bn = gum.loadBN(os.path.join(exp_dir, 'interventional_bn.bif'))

    
    return df_obs, df_int, metadata, obs_bn, int_bn




        
def run_models(df_obs, df_int, true_root_cause, obs_bn, int_bn, args, root_path, data_path, multiple_root_causes=False):
    """
    Run all models on the data and return their results.
    """
    obs_ground_truth = args.gt
    results = {}
    
    # Create a mapping between V-prefixed names and numeric names
    # = {f'{i}': str(i) for i in range(len(df_obs.columns))}
    #rev_node_map = {v: k for k, v in node_map.items()}
    
    # Rename columns to numeric for models that expect it
    #df_obs_num = df_obs.rename(columns=node_map)
    #df_int_num = df_int.rename(columns=node_map)
    
    # Get numeric version of true root cause

    print('rc:{}'.format(true_root_cause))
    if multiple_root_causes:
        if isinstance(true_root_cause, str):
            true_root_cause_num = list(df_obs.columns).index(true_root_cause)
            true_root_cause_num = [true_root_cause_num]
            true_root_cause = [true_root_cause]
        else:
            true_root_cause_num = [list(df_obs.columns).index(r) for r in true_root_cause]
    else:
        true_root_cause_num = list(df_obs.columns).index(true_root_cause)
    print('rc_index:{}'.format(true_root_cause_num))
    colnames = df_obs.columns
    # Create a DAG from the Bayesian network
    cpdag = get_cpdag(obs_bn)
    # print(cpdag.edges)
    # print(cpdag.arcs)
    #cg = pc(df_obs.to_numpy(), indep_test = 'chisq')
    #arcs, edges = causal_learn_graph_to_nx_digraph(cg.G, colnames)

    #print(cg.G)
    #cpdag = PDAG(nodes=colnames, arcs=arcs, edges=edges)

    
    # BRCD

    # sorted_root_causes, sorted_posterior, brcd_time = brcd.brcd(df_obs, df_int, cpdag,
    #                                                             isdiscrete=True,
    #                                                             node_transform = "none",      
    #                                                             transform_parents = False, 
    #                                                             version='brcd_u',
    #                                                             num_root_causes_candidates = 2)
    # print("BRCD-U predicted:{}".format(sorted_root_causes[:5]))
    # print("BRCD-U posterior:{}".format(sorted_posterior[:5]))

    # results['BRCD_U'] = {
    #     'Accuracy': 1 if all(elem in true_root_cause for elem in sorted_root_causes[0]) else 0,
    #     'time': brcd_time
    # }

    
    
    # # RCD
    # start_time = time.time()
    # rcd_result = top_k_rc(df_obs, df_int, k=2, bins=5, localized=True, verbose=False)
    

    # rcd_time = time.time() - start_time
    # rcd_sorted_nodes = rcd_result['root_cause']
  
    # print('RCD output:{}'.format(rcd_sorted_nodes))

    # results['RCD'] = {
    # 'Accuracy':  1 if all(elem in true_root_cause for elem in rcd_sorted_nodes) else 0,
    # 'time': rcd_time}


    
    
    
    # Clean up saved_models directory
    saved_models_dir = os.path.join(os.getcwd(), 'saved_models')
    if os.path.exists(saved_models_dir):
        import shutil
        shutil.rmtree(saved_models_dir)
        print(f"Cleaned up {saved_models_dir}")

        
    # RCG
    # alpha_r = rank_variables(df_obs.astype(int), df_int.astype(int), pdag_to_causallearn_generalgraph(cpdag), l=5)
    # # alpha_r = {'time': end, 'root_cause': result, 'tests': n_df.shape[1] - 1}
    # rcg_rootcauses = alpha_r['root_cause']
    # results['RCG'] = {
    #     'Accuracy':  1 if all(elem in true_root_cause for elem in rcg_rootcauses[:2]) else 0,
    #     'time': alpha_r['time']
    # }
    print("Result:")
    print(results)
    return results




def plot_accuracy_comparison_with_errorbars_full_boxed_and_save(nodes, xlabel, algorithms_results_top1, 
                                                                algorithms_results_top3,
                                                                algorithms_results_top5,
                                                                output_filename='average_accuracy.pdf',
                                                                get_top_five_plot= False,
                                                                algorithms_results_accuracy=None):
    # Set consistent style
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 100,
    })

    # Set up the figure and axes for 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
    colors = ['#006600',  '#88179a', '#0437ff', '#e39c19', '#808080' , '#ff3333', '#009999' ,'#cc0066', "#9966ff", "#ff9933"]

    # Sort nodes to ensure consistent plotting order
    nodes = sorted(nodes)
    
    if get_top_five_plot:
        fig, ax = plt.subplots(figsize=(10, 6))

        color_cycle = cycle(colors)  # robust to more methods than colors
        x_all = np.arange(len(nodes))

        for label, results in algorithms_results_accuracy.items():
            # mask points that don't exist for this method
            mask = np.array([size in results for size in nodes], dtype=bool)
            y = np.array([results[size][0] if size in results else np.nan for size in nodes], dtype=float)
            e = np.array([results[size][1] if size in results else np.nan for size in nodes], dtype=float)

            # plot only available points (avoids None/NaN in yerr)
            ax.errorbar(x_all[mask], y[mask], yerr=e[mask],
                        label=label, linestyle='--', marker='o',
                        color=next(color_cycle), linewidth=1)

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_xticks(x_all)
        ax.set_xticklabels(nodes)
        ax.set_ylim(0, 1)
        ax.grid(False)

        for side in ['top', 'bottom', 'left', 'right']:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_edgecolor('black')
            ax.spines[side].set_linewidth(2)

        # single legend (no fig.legend) placed above
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12),
                ncol=2, frameon=False)

        # leave room for the legend
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(output_filename, format='pdf', bbox_inches='tight')
        plt.close()
    else:
        # Plot top-1 accuracy
        for (label, results), color in zip(algorithms_results_top1.items(), colors):
            accuracies = []
            errors = []
            for size in nodes:
                if size in results:
                    accuracies.append(results[size][0])
                    errors.append(results[size][1])
                else:
                    accuracies.append(None)
                    errors.append(None)
            # axs[0].errorbar(nodes, accuracies, yerr=errors, label=label, linestyle='--', marker='o', color=color, linewidth=1)
            axs[0].errorbar(range(len(nodes)), accuracies, yerr=errors, label=label, linestyle='--', marker='o', color=color, linewidth=1)

        axs[0].set_title('Top-1')
        axs[0].set_xlabel(xlabel, fontsize=12)
        #axs[0].set_xticks(nodes)
        axs[0].set_xticks(range(len(nodes)))
        axs[0].set_xticklabels(nodes)
        axs[0].set_ylim(0, 1)

        # Plot top-3 accuracy
        for (label, results), color in zip(algorithms_results_top3.items(), colors):
            accuracies = []
            errors = []
            for size in nodes:
                if size in results:
                    accuracies.append(results[size][0])
                    errors.append(results[size][1])
                else:
                    accuracies.append(None)
                    errors.append(None)
            #axs[1].errorbar(nodes, accuracies, yerr=errors, label=label, linestyle='--', marker='o', color=color, linewidth=1)
            axs[1].errorbar(range(len(nodes)), accuracies, yerr=errors, label=label, linestyle='--', marker='o', color=color, linewidth=1)
            
        axs[1].set_title('Top-3')
        axs[1].set_xlabel(xlabel, fontsize=12)
        #axs[1].set_xticks(nodes)
        axs[1].set_xticks(range(len(nodes)))
        axs[1].set_xticklabels(nodes)
        axs[1].set_ylim(0, 1)

        # Plot top-5 accuracy
        for (label, results), color in zip(algorithms_results_top5.items(), colors):
            accuracies = []
            errors = []
            for size in nodes:
                if size in results:
                    accuracies.append(results[size][0])
                    errors.append(results[size][1])
                else:
                    accuracies.append(None)
                    errors.append(None)
            #axs[2].errorbar(nodes, accuracies, yerr=errors, label=label, linestyle='--', marker='o', color=color, linewidth=1)
            axs[2].errorbar(range(len(nodes)), accuracies, yerr=errors, label=label, linestyle='--', marker='o', color=color, linewidth=1)
        axs[2].set_title('Top-5')
        axs[2].set_xlabel(xlabel, fontsize=12)
        #axs[2].set_xticks(nodes)
        axs[2].set_xticks(range(len(nodes)))
        axs[2].set_xticklabels(nodes)
        axs[2].set_ylim(0, 1)

        # Remove the grid and ensure that the borders on all edges of each subplot are visible
        for ax in axs:
            ax.grid(False)  # Turn off the grid
            # Ensure all spines (top, bottom, left, right) are visible to create a box-like border
            for spine_position in ['top', 'bottom', 'left', 'right']:
                ax.spines[spine_position].set_visible(True)
                ax.spines[spine_position].set_edgecolor('black')  # Set the border color to black
                ax.spines[spine_position].set_linewidth(2)  # Set the border thickness

        # Create a single combined legend above all the plots that spans across the entire figure
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.97), frameon=False)  # Moving legend further down

        # Adjust layout to ensure space for the legend
        plt.tight_layout(rect=[0, 0, 1, 0.85])  # Increasing the available space above the plot
        
        # Save the figure to a PDF file
        fig.savefig(output_filename, format='pdf')
        plt.close()




_DIR_PATTERN = re.compile(
    r'^n_(?P<n>\d+)_obsSampleSize_(?P<obsSampleSize>\d+)_intSampleSize_(?P<intSampleSize>\d+)(?:_rc_(?P<rc>\d+))?$'
)


def make_key_fn(focus: str):
    if focus not in ("n", "obsSampleSize", "intSampleSize", "rc"):  # added rc
        raise ValueError("focus must be one of 'n','obsSampleSize','intSampleSize','rc'")
    pat = re.compile(rf"{re.escape(focus)}[_-](\d+)")
    def key_fn(folder_name: str):
        m = pat.search(folder_name)
        return (0, int(m.group(1))) if m else (1, folder_name.lower())
    return key_fn

def main():
    parser = argparse.ArgumentParser(description='Run experiments on generated data')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing experiment data')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--gt', type=bool, default=False, help='Determine whether to use ground truth obs. distribution')
    parser.add_argument('--xaxis', type=str, default='n', help='the x-axis of plot: n|obsSampleSize|intSampleSize|rc')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize results dictionariee
    algorithms_results_accuracy = {}
    time_results = {}
    
    # Get list of sample size directories
    pattern = re.compile(r'(n|obsSampleSize|intSampleSize|rc)_([0-9]+)')
    values = defaultdict(set)
    for d in os.listdir(args.data_dir):
        for key, val in pattern.findall(d):
            values[key].add(int(val))
    print(values)
    result = {key: sorted(vals) for key, vals in values.items()}
    
    
    all_entries = os.listdir(args.data_dir)
    exp_dirs = [d for d in all_entries if _DIR_PATTERN.match(d)]
    if not exp_dirs:
        raise RuntimeError(f"No valid experiment dirs in {args.data_dir!r} matching {_DIR_PATTERN.pattern}")

    # 2) Sort them by the chosen numeric field
    key_fn = make_key_fn(args.xaxis)
    sorted_dirs = sorted(exp_dirs, key=key_fn)

    # 3) Traverse in that order
    for config_dir in sorted_dirs:
        print('Working on :{}'.format(config_dir))
        m = _DIR_PATTERN.match(config_dir)
        n  = int(m.group("n"))
        obs_sample_size = int(m.group("obsSampleSize"))
        int_sample_size = int(m.group("intSampleSize"))
        rc = int(m.group("rc")) if m.group("rc") is not None else None

        exp_dirs = [d for d in os.listdir(os.path.join(args.data_dir, config_dir)) if d.startswith('experiment_')]
        # exp_dirs.sort()
        # Track per-experiment accuracies and times per model for this config
        exp_accuracy_by_model = defaultdict(list)
        exp_time_by_model = defaultdict(list)

        # Process each experiment
        for exp_idx, exp_dir in enumerate(tqdm(exp_dirs, desc="Experiment started...")):
            # Load experiment data
            df_obs, df_int, metadata, obs_bn, int_bn = load_experiment_data(
                os.path.join(args.data_dir, config_dir, exp_dir)
            )
            
            # Run models
            print(f'Processing experiment {exp_idx + 1}/{len(exp_dirs)}: {exp_dir}')
            if args.xaxis == 'rc':
                model_results = run_models(df_obs, df_int, metadata['true_root_cause'], obs_bn, int_bn, args, os.path.join(args.data_dir, config_dir, exp_dir), 'combined.csv',  multiple_root_causes=True)
            else:
                model_results = run_models(df_obs, df_int, metadata['true_root_cause'], obs_bn, int_bn, args, os.path.join(args.data_dir, config_dir, exp_dir), 'combined.csv')

            # Accumulate per-experiment metrics for per-config summary
            for model, metrics in model_results.items():
                if 'Accuracy' in metrics:
                    exp_accuracy_by_model[model].append(metrics['Accuracy'])
                if 'time' in metrics:
                    exp_time_by_model[model].append(metrics['time'])

        # After processing all experiments in this config, compute per-model averages and standard errors
        per_model_summary = {}
        all_models = set(list(exp_accuracy_by_model.keys()) + list(exp_time_by_model.keys()))
        for model in all_models:
            acc_list = exp_accuracy_by_model.get(model, [])
            time_list = exp_time_by_model.get(model, [])

            acc_mean = float(np.mean(acc_list)) if len(acc_list) > 0 else None
            acc_stderr = float(np.std(acc_list) / float(np.sqrt(len(acc_list)))) if len(acc_list) > 0 else None

            time_mean = float(np.mean(time_list)) if len(time_list) > 0 else None
            time_stderr = float(np.std(time_list) / float(np.sqrt(len(time_list)))) if len(time_list) > 0 else None

            per_model_summary[model] = {
                'accuracy': {
                    'mean': acc_mean,
                    'stderr': acc_stderr
                },
                'time': {
                    'mean': time_mean,
                    'stderr': time_stderr
                }
            }

        print('Per-config per-model avg/stderr:')
        print(per_model_summary)

        # Save per-config summary to JSON
        summary_path = os.path.join(args.output_dir, f'per_experiment_summary_{config_dir}.json')
        with open(summary_path, 'w') as f:
            json.dump(per_model_summary, f, indent=2)
        print(f'Saved per-experiment summary to {summary_path}')
 

if __name__ == "__main__":
    import torch.multiprocessing as tmp
    try:
        tmp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  

    main()
