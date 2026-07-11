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


sys.path.append(os.path.abspath(".."))
from convert_to_code import convert_config_to_code_full_paths, convert_config_to_code_one_step_paths, convert_config_to_code_qk_pruned, convert_keys_to_int
from patching_data import get_tokenizer_and_dataset_for_task
from draw_curve_utils import get_pareto_frontier, get_num_element
from show_heatmap import *
from patching_helper_functions import get_full_possible_config_for_pruning
from pruning_model import PruningModelWithHooksForQK, convert_config_fullpaths_, convert_config_fullpaths_incl_mlp_, convert_full_paths_config_to_prune_inside_kq_config_

from get_data_checkpoints import read_data_points, get_length_gen_performance



def plot_colored_curves_plotly(all_curves):
    import matplotlib.cm as cm
    import plotly.graph_objects as go

    cmap = cm.get_cmap("RdYlGn")
    scale_factor = 0.9

    # Line style cycle (<= 3 curves as you stated)
    line_styles = ["solid", "dot", "dash"]  

    fig = go.Figure()

    for i, (x, y, label, (model_name, acc1, acc2, acc3)) in enumerate(all_curves):
        # Map label to adjusted colormap position
        adjusted_label = label * scale_factor
        rgba = cmap(adjusted_label)
        r, g, b = int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)

        line_rgba_color = f"rgba({r}, {g}, {b}, 0.9)"
        marker_rgba_color = f"rgba({r}, {g}, {b}, 0.9)"

        # Pick line style
        dash_style = line_styles[i % len(line_styles)]

        # Legend label text
        legend_label = (
            f"{model_name} | "
            f"{acc1*100:.0f}%, {acc2*100:.0f}%, {acc3*100:.1f}%"
        )

        fig.add_trace(go.Scatter(
            x=[v * 100 for v in x],
            y=[v * 100 for v in y],
            mode="lines+markers",
            name=legend_label,   # <-- shown in legend
            line=dict(
                color=line_rgba_color,
                width=2,
                dash=dash_style
            ),
            marker=dict(
                color=marker_rgba_color,
                size=4,
                symbol="circle"
            ),
            hovertemplate=(
                f"<b>Model:</b> {model_name}<br>"
                f"<b>Label:</b> {label:.4f}<br>"
                f"<b>Original Task Acc:</b> {acc1*100:.0f}%, {acc2*100:.0f}%, {acc3*100:.1f}%"
                "<extra></extra>"
            ),
            showlegend=True
        ))

    # ----- Custom colorbar for label mapping -----
    colorscale = []
    num_points = 100
    for j in range(num_points + 1):
        label_value = j / num_points
        cmap_position = label_value * scale_factor
        rgba = cmap(cmap_position)
        rgb = f"rgb({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)})"
        colorscale.append([label_value, rgb])

    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker=dict(
            colorscale=colorscale,
            showscale=True,
            cmin=0,
            cmax=1,
            size=0,
            colorbar=dict(
                title="Accuracy of<br>Original Model on<br>length [101,150]<br>&nbsp;",
                tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                ticktext=["0%", "25%", "50%", "75%", "100%"],
                thickness=15
            )
        ),
        hoverinfo="none",
        showlegend=False
    ))

    def auto_legend_position(all_curves):
        # Count points in 4 quadrants
        counts = {"tl":0, "tr":0, "bl":0, "br":0}

        for x, y, *_ in all_curves:
            xm = sum(x)/len(x)
            ym = sum(y)/len(y)

            if xm < 0.5 and ym > 0.5: counts["tl"] += 1
            if xm > 0.5 and ym > 0.5: counts["tr"] += 1
            if xm < 0.5 and ym < 0.5: counts["bl"] += 1
            if xm > 0.5 and ym < 0.5: counts["br"] += 1

        # pick least occupied quadrant
        pos = min(counts, key=counts.get)

        mapping = {
            "tl": dict(x=0.02, y=0.98, xanchor="left",  yanchor="top"),
            "tr": dict(x=0.98, y=0.98, xanchor="right", yanchor="top"),
            "bl": dict(x=0.02, y=0.02, xanchor="left",  yanchor="bottom"),
            "br": dict(x=0.85, y=0.02, xanchor="right", yanchor="bottom"),  # avoid colorbar
        }
        return mapping[pos]

    legend_pos = auto_legend_position(all_curves)

    # ----- Layout -----
    fig.update_layout(
        width=650,
        height=600,
        xaxis_title="Percentage of Remained Edges",
        yaxis_title="Match Accuracy",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial, sans-serif", size=12, color="black"),
        legend=dict(
            title="Model | Task acc on <50%, [51,100]%, [101,150]%",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="black",
            borderwidth=1,
            **legend_pos
        ),
    )

    # Axes styling
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#E0E0E0",
        gridwidth=1,
        zeroline=True,
        zerolinecolor="gray",
        showline=True,
        linecolor="black",
        linewidth=1,
        ticksuffix="%",
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="#E0E0E0",
        gridwidth=1,
        zeroline=True,
        zerolinecolor="gray",
        showline=True,
        linecolor="black",
        linewidth=1,
        ticksuffix="%",
    )

    fig.show()

def plot_colored_curves_seaborn(all_curves):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.cm as cm

    sns.set_theme(style="whitegrid")

    cmap = cm.get_cmap("RdYlGn")
    scale_factor = 0.9

    # Line styles for <= 3 curves
    line_styles = ["solid", "dotted", "dashed"]

    fig, ax = plt.subplots(figsize=(7, 6))

    for i, (x, y, label, (model_name, acc1, acc2, acc3)) in enumerate(all_curves):
        adjusted_label = label * scale_factor
        rgba = cmap(adjusted_label)
        color = rgba[:3]  # matplotlib wants RGB tuple in [0,1]

        linestyle = line_styles[i % len(line_styles)]

        legend_label = (
            f"{model_name} | "
            f"{acc1*100:.0f}%, {acc2*100:.0f}%, {acc3*100:.1f}%"
        )

        ax.plot(
            [v * 100 for v in x],
            [v * 100 for v in y],
            linestyle=linestyle,
            linewidth=2,
            marker=None,
            markersize=4,
            color=color,
            label=legend_label,
            alpha=0.9,
        )

    # Axis labels
    ax.set_xlabel("Percentage of Remained Edges (%)", fontsize=14)
    ax.set_ylabel("Match Accuracy (%)", fontsize=14)

    # Auto legend placement (THIS is what Plotly cannot do)
    ax.legend(
        title="Model | Task Acc on <50, [51,100], [101,150]",
        loc="best",   # <-- automatic non-overlapping placement
        frameon=True,
        framealpha=0.9,
    )


    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    # cbar = plt.colorbar(sm, ax=ax, orientation="horizontal", pad=0.15, fraction=0.05)
    cbar.set_label("Accuracy of Original Model on length [101,150]")

    # Clean spines for paper look
    sns.despine()
    ax.set_facecolor("white")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    paths = glob.glob("newprune-unique_bigram_copy-*", root_dir=output_dir) # count, unique_reverse
    all_curves = []
    for path in tqdm(paths):    
        if "^" in path:
            continue
        model_name = "-".join(path.split("-")[1:-1])
        results = get_length_gen_performance(model_name)

        data_points, orig_num_edges = read_data_points(path, selected_stage=1)
        
        if len(data_points) == 0:
            continue

        frontier = get_pareto_frontier(data_points) # (acc, num_edges, exp_name) ascending by num_edges

        # optional: complete the frontier by assume the worst
        if frontier[0][0] > 0:
            frontier.insert(0, (0, frontier[0][1], None))
        if frontier[-1][1] < orig_num_edges:
            frontier.append((frontier[-1][0], orig_num_edges, None))

        frontier_accs, frontier_num_edges, _ = zip(*frontier)

        frontier_edge_percentage = [n/orig_num_edges for n in frontier_num_edges]
        all_curves.append((frontier_edge_percentage, frontier_accs, results["eval_len101-150_acc"], (model_name.split("-")[1][1:], results["eval_len0-50_acc"], results["eval_len51-100_acc"], results["eval_len101-150_acc"])))


    plot_colored_curves_seaborn(all_curves)

