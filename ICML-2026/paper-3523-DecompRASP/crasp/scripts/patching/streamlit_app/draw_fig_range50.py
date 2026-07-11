import glob
from my_paths import SUBMIT_HOST, CONDA_PATH, CONDA_ENV, REMOTE_SCRIPT_PATH, output_dir, HISTORY_FILE, RUNNING_EXPERIMENTS_FILE, path_to_saved_model, CACHE_DIR
from pathlib import Path
import json
import re
import yaml
import sys
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from collections import OrderedDict

sys.path.append(os.path.abspath(".."))
from convert_to_code import convert_config_to_code_full_paths, convert_config_to_code_one_step_paths, convert_config_to_code_qk_pruned, convert_keys_to_int
from patching_data import get_tokenizer_and_dataset_for_task
from draw_curve_utils import get_pareto_frontier, get_num_element
from show_heatmap import *
from patching_helper_functions import get_full_possible_config_for_pruning
from pruning_model import PruningModelWithHooksForQK, convert_config_fullpaths_, convert_config_fullpaths_incl_mlp_, convert_full_paths_config_to_prune_inside_kq_config_

from get_data_checkpoints import read_data_points, get_length_gen_performance

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm


def plot_models_frontiers(model_data):
    """
    model_data: dict
        { model_name: (frontier1, frontier2) }
        each frontier is list of (acc, num_edges, exp_name)
        sorted by num_edges, last entry is original model
    """

    sns.set_style("white")

    model_names = list(model_data.keys())
    # assert len(model_names) == 6, "Expect exactly 6 models"

    # fixed colors per frontier
    color_f1, line_f1 = "orange", "-"
    color_f2, line_f2 = "green", "--"

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=False, sharey=True)
    axes = axes.flatten()

    for idx, (ax, model_name) in enumerate(zip(axes, model_names)):
        f1, f2 = model_data[model_name]

        for frontier, color, line, frontier_label in [
            (f1, color_f1, line_f1, "Pruned on $\leq 150$"),
            (f2, color_f2, line_f2, "Pruned on $\leq 50$"),
        ]:
            pruned = frontier[:-1]
            original = frontier[-1]
            # ---- Pareto frontier ----
            ax.step(
                np.array([x[1] for x in pruned]),
                np.array([x[0] for x in pruned]),
                where="post",
                color=color,
                linewidth=2,
                alpha=0.3,
                zorder=1,
                label=frontier_label if idx == 0 else None,
                linestyle=line,
            )

            # ---- Pruned models ----
            ax.scatter(
                np.array([x[1] for x in pruned]),
                np.array([x[0] for x in pruned]),
                marker="X",
                s=40,
                alpha=0.6,
                zorder=3,
                color="red",
                label="Pruned Models" if idx == 0 and frontier is f1 else None,
            )

            # ---- Original model ----
            ax.scatter(
                [original[1]],
                [original[0]],
                marker="*",
                s=180,
                alpha=0.8,
                zorder=4,
                color="gold",
                edgecolor="black",
                label="Original Models" if idx == 0 and frontier is f1 else None,
            )

        ax.set_title(re.sub(r'[@#%^]', '', model_name), fontsize=11)

        # faint grid
        ax.grid(True, which="major", alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)

    # global labels
    fig.supxlabel("Number of Remaining Edges")
    fig.supylabel("Match Accuracy (%)")

    # global legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig("trash3.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    model_list = ["unique_copy-#4l4h256d4lr00drop", "unique_bigram_copy-#4l2h64d3lr00drop",
                  "repeat_copy-@4l4h256d3lr00drop", "count-#1l4h256d3lr01drop", 
                  "unique_reverse-#2l4h256d4lr01drop", "unique_copy-^2l1h64d3lr01drop-1300"]
    paths = glob.glob("newprune-unique_bigram_copy-*", root_dir=output_dir) # count, unique_reverse
    all_curves = OrderedDict()
    for model in model_list:
        print(model)
        path = glob.glob(f"newprune-{model}-*", root_dir=output_dir)
        if len(path) == 0:
            print("skip", model)
            continue
        assert len(path) == 1
        path = path[0]

        data_points, orig_num_edges = read_data_points(path, selected_stage=1)
        frontier = get_pareto_frontier(data_points) # (acc, num_edges, exp_name) ascending by num_edges
        if frontier[0][0] > 0:
            frontier.insert(0, (0, frontier[0][1], None))
        if frontier[-1][1] < orig_num_edges:
            frontier.append((frontier[-1][0], orig_num_edges, None))
        frontier.append((1, orig_num_edges, None))

        all_curves[model] = (frontier, )

        path = glob.glob(f"range50-{model}-*", root_dir=output_dir)
        assert len(path) == 1
        path = path[0]

        data_points, orig_num_edges = read_data_points(path, selected_stage=1)
        frontier = get_pareto_frontier(data_points) # (acc, num_edges, exp_name) ascending by num_edges
        if frontier[0][0] > 0:
            frontier.insert(0, (0, frontier[0][1], None))
        if frontier[-1][1] < orig_num_edges:
            frontier.append((frontier[-1][0], orig_num_edges, None))
        frontier.append((1, orig_num_edges, None))
        
        all_curves[model] = all_curves[model] + (frontier, )

    plot_models_frontiers(all_curves)