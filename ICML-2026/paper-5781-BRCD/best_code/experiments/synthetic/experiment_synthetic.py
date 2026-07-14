import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import logging
# Reset root logging configuration
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.WARNING)
import models.brcd as brcd
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

    sorted_root_causes, sorted_posterior, brcd_time = brcd.brcd(df_obs, df_int, cpdag, obs_bn, int_bn, obs_ground_truth, int_ground_truth=False, version='brcd_u')
    print("BRCD-U predicted:{}".format(sorted_root_causes[:5]))
    print("BRCD-U posterior:{}".format(sorted_posterior[:5]))
    if multiple_root_causes:
        check = 1 
        for r in true_root_cause:
            if not r in sorted_root_causes[:5]:
                check = 0
                break
        results['BRCD_U'] = {
            'Accuracy': check,
            'time': brcd_time
        }
    else:
        results['BRCD_U'] = {
            'top1': 1 if sorted_root_causes and true_root_cause == sorted_root_causes[0] else 0,
            'top3': 1 if true_root_cause in sorted_root_causes[:3] else 0,
            'top5': 1 if true_root_cause in sorted_root_causes[:5] else 0,
            'time': brcd_time
        }

    sorted_root_causes, sorted_posterior, brcd_time = brcd.brcd(df_obs, df_int, cpdag, obs_bn, int_bn, obs_ground_truth, int_ground_truth=False, version='brcd_c')
    print("BRCD-C predicted:{}".format(sorted_root_causes[:5]))
    print("BRCD-C posterior:{}".format(sorted_posterior[:5]))
    if multiple_root_causes:
        check = 1 
        for r in true_root_cause:
            if not r in sorted_root_causes[:5]:
                check = 0
                break
        results['BRCD_C'] = {
            'Accuracy': check,
            'time': brcd_time
        }
    else:
        results['BRCD_C'] = {
            'top1': 1 if sorted_root_causes and true_root_cause == sorted_root_causes[0] else 0,
            'top3': 1 if true_root_cause in sorted_root_causes[:3] else 0,
            'top5': 1 if true_root_cause in sorted_root_causes[:5] else 0,
            'time': brcd_time
        }

    
    # BARO
    df_for_baro = pd.concat([df_obs, df_int], ignore_index=False).astype(int)
    df_for_baro['time'] = df_for_baro.index
    baronames = [f'X{i+1}_cpu' for i in range(len(df_obs.columns))]
    baro_id_map = {f'X{i+1}_cpu': i for i in range(len(df_obs.columns))}
    df_for_baro.columns = baronames + ['time']
    anomalies = [len(df_obs)]
    start_time = time.time()
    root_causes = robust_scorer(df_for_baro, anomalies=anomalies)["ranks"]
    root_causes_idx = [baro_id_map[r] for r in root_causes]
    baro_time = time.time() - start_time
    if multiple_root_causes:
        check = 1 
        for r in true_root_cause_num:
            if not r in root_causes_idx[:5]:
                check = 0
                break
        results['BARO'] = {
            'Accuracy': check,
            'time': baro_time
        }
    else:
        results['BARO'] = {
                'top1': 1 if root_causes and root_causes_idx[0] == true_root_cause_num  else 0,
                'top3': 1 if true_root_cause_num in root_causes_idx[:3] else 0,
                'top5': 1 if true_root_cause_num in root_causes_idx[:5] else 0,
                'time': baro_time}

    print('BARO output top 5:{}'.format(root_causes_idx[:5]))

    
    # RCD
    start_time = time.time()
    rcd_result_top1 = top_k_rc(df_obs, df_int, k=1, bins=5, localized=True, verbose=False)
    rcd_result_top3 = top_k_rc(df_obs, df_int, k=3, bins=5, localized=True, verbose=False)
    rcd_result_top5 = top_k_rc(df_obs, df_int, k=5, bins=5, localized=True, verbose=False)

    rcd_time = time.time() - start_time
    rcd_sorted_nodes_top1 = rcd_result_top1['root_cause']
    rcd_sorted_nodes_top3 = rcd_result_top3['root_cause']
    rcd_sorted_nodes_top5 = rcd_result_top5['root_cause']
    print('RCD output top 5:{}'.format(rcd_sorted_nodes_top5))

    if multiple_root_causes:
        check = 1 
        for r in true_root_cause:
            if not r in rcd_sorted_nodes_top5:
                check = 0
                break
        results['RCD'] = {
            'Accuracy': check,
            'time': rcd_time
        }
    else:
        results['RCD'] = {
            'top1': 1 if rcd_sorted_nodes_top1 and rcd_sorted_nodes_top1[0] == true_root_cause else 0,
            'top3': 1 if rcd_sorted_nodes_top3  and true_root_cause in rcd_sorted_nodes_top3[:3] else 0,
            'top5': 1 if rcd_sorted_nodes_top3 and true_root_cause in rcd_sorted_nodes_top5[:5] else 0,
            'time': rcd_time
        }

        
    # RCG
    alpha_r = rank_variables(df_obs.astype(int), df_int.astype(int), pdag_to_causallearn_generalgraph(cpdag), l=5)
    # alpha_r = {'time': end, 'root_cause': result, 'tests': n_df.shape[1] - 1}
    rcg_rootcauses = alpha_r['root_cause']
    if multiple_root_causes:
        check = 1 
        for r in true_root_cause:
            if not r in rcg_rootcauses[:5]:
                check = 0
                break
        results['RCG'] = {
            'Accuracy': check,
            'time': alpha_r['time']
        }
    else:
        results['RCG'] = {
                'top1': 1 if rcg_rootcauses and rcg_rootcauses[0] == true_root_cause else 0,
                'top3': 1 if rcg_rootcauses  and true_root_cause in rcg_rootcauses[:3] else 0,
                'top5': 1 if rcg_rootcauses and true_root_cause in rcg_rootcauses[:5] else 0,
                'time': alpha_r['time']
            }

    
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

def plot_execution_time_with_errorbars_and_save(
    int_sample_sizes,
    xlabel, 
    time_results,
    output_filename='Execution_time_comparison.pdf'
):
    # Set consistent style
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 14,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.dpi': 100,
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold'
    })

    # Set up the figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Define colors
    colors = ['#006600',  '#88179a', '#0437ff', '#e39c19', '#808080' , '#009999' ,'#cc0066', "#9966ff", "#ff9933", '#ff3333']


    # Sort sample sizes to ensure consistent plotting order
    int_sample_sizes = sorted(int_sample_sizes)
    positions = range(len(int_sample_sizes))
    for (label, results), color in zip(time_results.items(), colors):
        times = []
        errors = []
        for size in int_sample_sizes:
            if size in results:
                times.append(results[size][0])
                errors.append(results[size][1])
            else:
                times.append(None)
                errors.append(None)
        ax.errorbar(positions, times, yerr=errors, label=label, linestyle='--', marker='o', color=color, linewidth=1)

    # Set titles and labels
    ax.set_title('Execution Time', fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Time (seconds)', fontsize=14, fontweight='bold')
    # Set xticks and limits
    ax.set_xticks(positions)
    ax.set_xticklabels(int_sample_sizes)  # shows [10,20,100,1000] equally spaced
    ax.set_ylim(0)  # Assuming non-negative execution times

    # Remove grid and style the spines
    ax.grid(False)
    for spine_position in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine_position].set_visible(True)
        ax.spines[spine_position].set_edgecolor('black')
        ax.spines[spine_position].set_linewidth(2)

    #handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        loc='upper left', 
        bbox_to_anchor=(1, 1),   # x=1 (just outside right), y=0.5 (centered vertically)
        frameon=False, fontsize=12
    )

    # Add legend
    #ax.legend(loc='upper left', frameon=False, fontsize=12)

    # Adjust layout for a clean look
    plt.tight_layout()
    plt.subplots_adjust(right=0.8)
    # Save the figure to a PDF file
    fig.savefig(output_filename, format='pdf', bbox_inches="tight")
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
    
    # Initialize results dictionaries
    algorithms_results_top1 = {}
    algorithms_results_top3 = {}
    algorithms_results_top5 = {}
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
            
            if args.xaxis == 'n':
                parameters_to_average = result['n']
            elif args.xaxis == 'obsSampleSize':
                parameters_to_average = result['obsSampleSize']
            elif args.xaxis == 'intSampleSize':
                # interventional sample sizes
                parameters_to_average = result['intSampleSize']
            elif args.xaxis == 'rc':                              # NEW
                parameters_to_average = result['rc']
            else:
                assert NotImplementedError

            # update results
            for model, metrics in model_results.items():
                # Update accuracy results
                if model not in algorithms_results_top1:
                    algorithms_results_top1[model] = {size: [] for size in parameters_to_average}
                if model not in algorithms_results_top3:
                    algorithms_results_top3[model] = {size: [] for size in parameters_to_average}
                if model not in algorithms_results_top5:
                    algorithms_results_top5[model] = {size: [] for size in parameters_to_average}
                if model not in algorithms_results_accuracy:
                    algorithms_results_accuracy[model] = {size: [] for size in parameters_to_average}
                if model not in time_results:
                    time_results[model] = {size: [] for size in parameters_to_average}
                
                if args.xaxis == 'n':
                    selected_para = n
                elif args.xaxis == 'obsSampleSize':
                    selected_para = obs_sample_size
                elif args.xaxis == 'intSampleSize':
                    selected_para = int_sample_size
                elif args.xaxis == 'rc':
                    selected_para = rc
                else:
                    assert NotImplementedError
                if args.xaxis == 'rc':
                    algorithms_results_accuracy[model][selected_para].append(metrics['Accuracy'])
                else:
                    algorithms_results_top1[model][selected_para].append(metrics['top1'])
                    algorithms_results_top3[model][selected_para].append(metrics['top3'])
                    algorithms_results_top5[model][selected_para].append(metrics['top5'])

                time_results[model][selected_para].append(metrics['time'])

        # Calculate means and standard errors for each sample size after all experiments in this config
        temp_algorithms_results_top1 = {}
        temp_algorithms_results_top3 = {}
        temp_algorithms_results_top5 = {}
        temp_algorithms_results_accuracy = {}
        temp_time_results = {}

        for model in algorithms_results_top1:
            # Calculate accuracy statistics for each sample size
            temp_algorithms_results_top1[model] = {}
            temp_algorithms_results_top3[model] = {}
            temp_algorithms_results_top5[model] = {}
            temp_algorithms_results_accuracy[model] = {}
            temp_time_results[model] = {}

            for size in parameters_to_average:
                top1_values = algorithms_results_top1[model][size]
                top3_values = algorithms_results_top3[model][size]
                top5_values = algorithms_results_top5[model][size]
                accuracy_values = algorithms_results_accuracy[model][size]
                time_values = time_results[model][size]

                temp_algorithms_results_top1[model][size] = (float(np.mean(top1_values)), float(np.std(top1_values) / float(np.sqrt(len(top1_values)))))
                temp_algorithms_results_top3[model][size] = (float(np.mean(top3_values)), float(np.std(top3_values) / float(np.sqrt(len(top3_values)))))
                temp_algorithms_results_top5[model][size] = (float(np.mean(top5_values)), float(np.std(top5_values) / float(np.sqrt(len(top5_values)))))
                if args.xaxis == 'rc':
                    temp_algorithms_results_accuracy[model][size] = (float(np.mean(accuracy_values)), float(np.std(accuracy_values) / float(np.sqrt(len(accuracy_values)))))
                temp_time_results[model][size] = (float(np.mean(time_values)), float(np.std(time_values) / float(np.sqrt(len(time_values)))))
               
        # Plot results after all experiments in this config
        if args.xaxis == 'n':
            xlabel = 'number of nodes'
        elif args.xaxis == 'obsSampleSize':
            xlabel = 'obs. sample size'
        elif args.xaxis == 'intSampleSize':
            xlabel = 'int. sample size'
        elif args.xaxis == 'rc':
            xlabel = 'number of root causes'
        else:
            assert NotImplementedError

        # Create a subdirectory for this config's results
        config_output_dir = os.path.join(args.output_dir, config_dir)
        os.makedirs(config_output_dir, exist_ok=True)
        print("top1 results:")
        print(temp_algorithms_results_top1)
        print("top3 results:")
        print(temp_algorithms_results_top3)
        print("top5 results:")
        print(temp_algorithms_results_top5)
        print("accuracy results:")
        print(temp_algorithms_results_accuracy)
        print("Time results:")
        print(temp_time_results)

    
        if args.xaxis == 'rc':
            plot_accuracy_comparison_with_errorbars_full_boxed_and_save(
                parameters_to_average,
                xlabel,
                temp_algorithms_results_top1,
                temp_algorithms_results_top3,
                temp_algorithms_results_top5,
                output_filename=os.path.join(config_output_dir, 'accuracy_results.pdf'),
                get_top_five_plot=True,
                algorithms_results_accuracy=temp_algorithms_results_accuracy
            )
        else:
            plot_accuracy_comparison_with_errorbars_full_boxed_and_save(
                parameters_to_average,
                xlabel,
                temp_algorithms_results_top1,
                temp_algorithms_results_top3,
                temp_algorithms_results_top5,
                output_filename=os.path.join(config_output_dir, 'accuracy_results.pdf'),
                get_top_five_plot=False
            )

        plot_execution_time_with_errorbars_and_save(
            parameters_to_average,
            xlabel,
            temp_time_results,
            output_filename=os.path.join(config_output_dir, 'execution_time.pdf')
        )
        
        # Save numerical results for this config
        results = {
            'top1': temp_algorithms_results_top1,
            'top3': temp_algorithms_results_top3,
            'top5': temp_algorithms_results_top5,
            'time': temp_time_results
        }
        with open(os.path.join(config_output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"Results for config {config_dir} saved to {config_output_dir}/")

    # Final results calculation and plotting (keeping the original code for final results)
    for model in algorithms_results_top1:
        # Calculate accuracy statistics for each sample size
        for size in parameters_to_average:
            top1_values = algorithms_results_top1[model][size]
            top3_values = algorithms_results_top3[model][size]
            top5_values = algorithms_results_top5[model][size]
            accuracy_values = algorithms_results_accuracy[model][size]
            time_values = time_results[model][size]
            
            algorithms_results_top1[model][size] = (float(np.mean(top1_values)), float(np.std(top1_values) / float(np.sqrt(len(top1_values)))))
            algorithms_results_top3[model][size] = (float(np.mean(top3_values)), float(np.std(top3_values) / float(np.sqrt(len(top3_values)))))
            algorithms_results_top5[model][size] = (float(np.mean(top5_values)), float(np.std(top5_values) / float(np.sqrt(len(top5_values)))))
            if args.xaxis == 'rc':
                algorithms_results_accuracy[model][size] = (float(np.mean(accuracy_values)), float(np.std(accuracy_values) / float(np.sqrt(len(accuracy_values)))))
            time_results[model][size] = (float(np.mean(time_values)), float(np.std(time_values) / float(np.sqrt(len(time_values)))))
    
    # Plot final results
    if args.xaxis == 'rc':
        plot_accuracy_comparison_with_errorbars_full_boxed_and_save(
            parameters_to_average,
            xlabel,
            algorithms_results_top1,
            algorithms_results_top3,
            algorithms_results_top5,
            output_filename=os.path.join(args.output_dir, 'final_accuracy_results.pdf'),
            get_top_five_plot=True,
            algorithms_results_accuracy=algorithms_results_accuracy
        )
    else:
        plot_accuracy_comparison_with_errorbars_full_boxed_and_save(
            parameters_to_average,
            xlabel,
            algorithms_results_top1,
            algorithms_results_top3,
            algorithms_results_top5,
            output_filename=os.path.join(args.output_dir, 'final_accuracy_results.pdf'),
            get_top_five_plot=False
        )
    
    plot_execution_time_with_errorbars_and_save(
        parameters_to_average,
        xlabel,
        time_results,
        output_filename=os.path.join(args.output_dir, 'final_execution_time.pdf')
    )
    
    # Save final numerical results
    if args.xaxis == 'rc':
        results = {
            'Accuracy': algorithms_results_accuracy,
            'time': time_results
        }
    else:
        results = {
            'top1': algorithms_results_top1,
            'top3': algorithms_results_top3,
            'top5': algorithms_results_top5,
            'time': time_results
        }
    with open(os.path.join(args.output_dir, 'final_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    
    print(f"Final results saved to {args.output_dir}/")

    
 

if __name__ == "__main__":
    import torch.multiprocessing as tmp
    try:
        tmp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  

    main()
