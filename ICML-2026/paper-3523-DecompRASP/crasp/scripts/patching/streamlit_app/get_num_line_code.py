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
from draw_curve_utils import get_pareto_frontier
from convert_to_code import convert_config_to_code_qk_pruned

def plot_pareto_seaborn(data):
    """
    data: list of (acc, num_edges, exp_name)
          sorted by num_edges, last entry is original model
    """
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Separate pruned and original
    pruned = data[:-1]
    original = data[-1]

    # Extract arrays
    edges = np.array([x[1] for x in pruned])
    accs  = np.array([x[0] for x in pruned])

    # ---- Paper-style aesthetics ----
    sns.set_style("white")
    plt.figure(figsize=(8, 4))

    # ---- Pareto frontier (background) ----
    plt.step(
        edges, accs,
        where="post",
        color="orange",
        linewidth=2,
        alpha=0.6,
        zorder=1,
        label="Pareto Frontier"
    )

    # ---- Pruned models ----
    sns.scatterplot(
        x=edges, y=accs,
        color="red", marker="X", s=100,
        zorder=3,
        label="Pruned Models"
    )

    # ---- Original model ----
    sns.scatterplot(
        x=[original[1]], y=[original[0]],
        color="gold", edgecolor="black",
        marker="*", s=250,
        zorder=4,
        label="Original Model\nw/ Linear LN"
    )

    # Labels
    plt.xlabel("Proportion of Remaining Lines of Code")
    plt.ylabel("Match Accuracy")

    ax = plt.gca()

    # ---- Very faint grid (paper-friendly) ----
    ax.grid(True, which="major", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)  # keep grid behind data

    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_multiple_pareto_seaborn(all_data, labels=None):
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

    for i, data in enumerate(all_data):
        pruned = data[:-1]
        original = data[-1]

        edges = np.array([x[1] for x in data])
        accs  = np.array([x[0] for x in data])

        # ---- Pareto frontier ----
        plt.step(
            edges, accs,
            where="post",
            color="orange",
            linewidth=2,
            alpha=0.3,   # lighter so multiple curves don't dominate
            zorder=1,
            label="Pareto Frontiers" if i == 0 else None
        )

        # ---- Pruned points ----
        sns.scatterplot(
            x=edges, y=accs,
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
            label="Original Models\nw/ Linear LN" if i == 0 else None
        )

    # Labels
    plt.xlabel("Proportion of Remaining Lines of Code (%)")
    plt.ylabel("Match Accuracy (%)")

    ax = plt.gca()

    # Very faint grid (paper-friendly)
    ax.grid(True, which="major", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

    # plt.fill_between(edges, accs, step="post", alpha=0.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig("trash.png", dpi=200)

def determine_total_line(num_layer, num_head, split_mlps):
    num_v = 2
    num_line = 0
    for i in range(num_layer):
        num_line += (num_v ** 2 + num_v) * num_head
        num_v += num_v * num_head

        if split_mlps:
            num_line += num_v
            num_v += num_v
        else:
            num_line += 1
            num_v += 1
    
    num_line += num_v
    num_line += 1 # bias term
    num_line += 1 # prediction
    return num_line


if __name__ == "__main__":
    all_curves = []
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
                if exp_stage != 3:
                    continue
            else:
                continue
  
            final_config = output_dict[f"result_patching_config_global_iteration_{max(stages)}"]
            
            if not (series_path / exp_name / "converted_mlp.pt").exists():
                print("warning: no converted_mlp for", series_path)
                continue
            converted_mlp = torch.load(series_path / exp_name / "converted_mlp.pt", map_location="cpu", weights_only=False)
            with open(series_path / exp_name / "config.yaml") as f:
                split_mlps = yaml.safe_load(f).get("split_mlps", False)
            code, heatmap_to_config = convert_config_to_code_qk_pruned(final_config, split_mlps, converted_mlp)

            acc_match = output_dict["acc_match"]
            data_points.append((acc_match, len(code), exp_name))
        
        if len(data_points) == 0 or LLNA_acc < 0.9:
            continue

        match = re.search(r"(\d+)l(\d+)h(\d+)d([34])lr(0[01])drop", series_path.name.split("-")[2])
        n_layer, n_head, d_model, lr, dropout = match.groups()
        n_layer, n_head, d_model = int(n_layer), int(n_head), int(d_model)
        total_line = determine_total_line(n_layer, n_head, split_mlps)

        frontier = get_pareto_frontier(data_points) # (acc, num_edges, exp_name) ascending by num_edges

        frontier = [(acc*100, num_edges/total_line*100, exp_name) for acc, num_edges, exp_name in frontier]
        # if frontier[0][0] > 0:
        #     frontier.insert(0, (0, frontier[0][1], None))
        frontier.append((LLNA_acc*100, 100, None))

        all_curves.append(frontier)


    plot_multiple_pareto_seaborn(all_curves)



