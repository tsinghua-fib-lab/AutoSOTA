import torch
import sys
import os
from pathlib import Path
import glob
import json
import re
import yaml
from my_paths import SUBMIT_HOST, CONDA_PATH, CONDA_ENV, REMOTE_SCRIPT_PATH, output_dir, HISTORY_FILE, RUNNING_EXPERIMENTS_FILE, path_to_saved_model, CACHE_DIR

sys.path.append(os.path.abspath(".."))
from draw_curve_utils import get_pareto_frontier, get_num_element
from convert_to_code import convert_config_to_code_qk_pruned
from patching_helper_functions import get_full_possible_config_for_pruning


def plot_multiple_pareto_seaborn(all_data, colors):
    """
    all_data: list of datasets
        each dataset is a list of (acc, num_edges, exp_name)
        sorted by num_edges, last entry is original model
    """
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_style("white")
    plt.figure(figsize=(8, 4))
    # Single color for everything
    visited_color = set()
    for i, (data, color) in enumerate(zip(all_data, colors)):
        pruned = data[:-1]
        original = data[-1]

        # ---- Pareto frontier ----
        plt.step(
            np.array([x[1] for x in data]), np.array([x[0] for x in data]),
            where="post",
            color=color,
            linewidth=2,
            alpha=0.3,   # lighter so multiple curves don't dominate
            zorder=1,
            label="Pareto Frontiers" if color not in visited_color else None
        )
        visited_color.add(color)

        # ---- Pruned points ----
        sns.scatterplot(
            x=np.array([x[1] for x in pruned]), y=np.array([x[0] for x in pruned]),
            color="red",
            marker="X",
            s=40,
            alpha=0.6,
            zorder=3,
            label="Pruned Models" if i == 0 else None
        )

        # ---- Original model ----
        sns.scatterplot(
            x=[original[1]], y=[original[0]],
            color="gold",
            edgecolor="black",
            marker="*",
            s=180,
            alpha=0.8,
            zorder=4,
            label="Original Models" if i == 0 else None
        )

    # Labels
    plt.xlabel("Proportion of Remaining Number of Edges (%)")
    plt.ylabel("Match Accuracy (%)")

    ax = plt.gca()

    # Very faint grid (paper-friendly)
    ax.grid(True, which="major", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

    # plt.fill_between(edges, accs, step="post", alpha=0.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig("trash2.png", dpi=200)

if __name__ == "__main__":
    all_curves = []
    mark_list = ["D3", "D12", "tomita4", "ababstar", "aaaastar", "012_0_2"]
    colors = []
    for path in glob.glob("newprune*", root_dir=output_dir):
        print(path)
        series_path = Path(output_dir) / path
        data_points = []
        LLNA_acc = 0
        for exp_name in glob.glob("exp*", root_dir=series_path):
            if (series_path / exp_name / "output.json").exists():
                with open(series_path / exp_name / "output.json") as f:
                    output_dict = json.load(f)
                stages = []
                for key in output_dict:
                    if match := re.search(r"result_patching_config_global_iteration_(\d+)", key):
                        stages.append(int(match.group(1)))
                if max(stages) == 0:
                    with open(series_path / exp_name / "config.yaml") as f:
                        linear_LN = yaml.safe_load(f).get("linear_LN", True)
                    if linear_LN:
                        exp_stage = 1
                    else:
                        exp_stage = 0
                else:
                    exp_stage = max(stages)+1
                
                if exp_stage == 1:
                    LLNA_acc = max(LLNA_acc, output_dict["acc_match"])
                if exp_stage != 0:
                    continue
            else:
                continue
  
            final_config = output_dict[f"result_patching_config_global_iteration_{max(stages)}"]
            num_edges = get_num_element(final_config)
            acc_match = output_dict["acc_match"]
            data_points.append((acc_match, num_edges, exp_name))
        
        if len(data_points) == 0 or LLNA_acc >= 0.9:
            continue

        frontier = get_pareto_frontier(data_points) # (acc, num_edges, exp_name) ascending by num_edges


        num_heads_per_layer = {layer_idx: len(final_config[str(layer_idx)]["k"]) for layer_idx in range(len(final_config)-1)}
        with open(series_path / exp_name / "config.yaml") as f:
            split_mlps = yaml.safe_load(f)["split_mlps"]
        full_config = get_full_possible_config_for_pruning(num_heads_per_layer)
        
        orig_num_edges = get_num_element(full_config)
        frontier = [(acc*100, num_edges/orig_num_edges*100, exp_name) for acc, num_edges, exp_name in frontier]
        # if frontier[0][0] > 0:
        #     frontier.insert(0, (0, frontier[0][1], None))
        frontier.append((100, 100, None))

        all_curves.append(frontier)

        if any(f"bce_{s}" in path for s in mark_list):
            colors.append("green")
        else:
            colors.append("orange")


    plot_multiple_pareto_seaborn(all_curves, colors)



