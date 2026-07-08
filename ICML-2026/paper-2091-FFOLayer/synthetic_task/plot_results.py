import os, re, glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")


sns.set_theme(style="whitegrid", context="talk")
palette = sns.color_palette()

batch_size = 8
BASE_DIR = f"../synthetic_results_general_{batch_size}"
# BASE_DIR = f"../synthetic_results_1_compare_SCS_OSQP_dim200_debug"
# METHODS = [
#     "cvxpylayer",
#     "lpgd",
#     "bpqp",
#     "ffocp_eq",
#     "dqp",
#     "qpth",
#     "ffoqp_eq",   
# ]
METHODS = [
    "cvxpylayer",
    "qpth",
    "lpgd",
    "bpqp",
    "dqp",
    "ffocp_eq",
    "ffoqp_eq_schur"  
]
METHODS_LEGEND = {
    "cvxpylayer": "CvxpyLayer",
    "lpgd": "LPGD",
    "bpqp": "BPQP",
    "ffocp_eq": "FFOCP",
    "dqp": "dQP",
    "qpth": "qpth",
    "ffoqp_eq_schur": "FFOQP",
}

METHODS_STEPS = [method+"_steps" for method in METHODS]

method_order = [METHODS_LEGEND[m] for m in METHODS]

markers = ["o", "s", "D", "^", "v", "x", "P", "s"]
markers_dict = {method: markers[i] for i, method in enumerate(method_order)}

LINEWIDTH = 1.5

def is_subsequence(a, b):
    it = iter(b)
    return all(x in it for x in a)

def load_results_CP(base_dir=BASE_DIR, methods=METHODS, methods_legend=None):
    if methods_legend is None:
        methods_legend = METHODS_LEGEND
    dfs = []
    for m in methods:
        pattern = os.path.join(base_dir, m, "*.csv")
        for fp in sorted(glob.glob(pattern)):
            m = m.removesuffix("_steps")
            
            df = pd.read_csv(fp)
            df["method"] = methods_legend[m]

            fname = os.path.basename(fp)
            def grab(pat, cast=float):
                mo = re.search(pat, fname)
                return cast(mo.group(1)) if mo else np.nan

            df["seed"] = grab(r"_seed(\d+)", int)
            df["n"] = grab(r"n(\d+)", int)
            df["lr"]   = grab(r"lr([0-9eE\.\-]+)", float)
            df["ydim"] = grab(r"ydim(\d+)", int)
            df["backwardTol"] = grab(r"backwardTol([0-9eE\+\-\.]+?)(?:_|\.csv)", float)
            dfs.append(df)
        
        print("method: ", m)
        # print(df)
        
    if not dfs:
        raise FileNotFoundError(f"No CSVs found under {base_dir}.")
    return pd.concat(dfs, ignore_index=True, sort=False)

def load_results_QP(base_dir=BASE_DIR, methods=METHODS, methods_legend=None):
    if methods_legend is None:
        methods_legend = METHODS_LEGEND
    dfs = []
    for m in methods:
        pattern = os.path.join(base_dir, m, "*.csv")
        for fp in sorted(glob.glob(pattern)):
            m = m.removesuffix("_steps")
            if m == "ffocp_eq" and "backwardTol" not in os.path.basename(fp):
                continue
            
            df = pd.read_csv(fp)
            df["method"] = methods_legend[m]

            fname = os.path.basename(fp)
            def grab(pat, cast=float):
                mo = re.search(pat, fname)
                return cast(mo.group(1)) if mo else np.nan

            df["seed"] = grab(r"_seed(\d+)", int)
            df["n"] = grab(r"n(\d+)", int)
            df["lr"]   = grab(r"lr([0-9eE\.\-]+)", float)
            df["ydim"] = grab(r"ydim(\d+)", int)
            df["backwardTol"] = grab(r"backwardTol([0-9eE\+\-\.]+?)(?:_|\.csv)", float)
            dfs.append(df)
        
        print("method: ", m)
        # print(df)
        
    if not dfs:
        raise FileNotFoundError(f"No CSVs found under {base_dir}.")
    return pd.concat(dfs, ignore_index=True, sort=False)

def plot_time_vs_method(df, time_names=['forward_time', 'backward_time'], plot_path=BASE_DIR, plot_name_tag=""):
    df_avg_method = df.groupby('method')[time_names].mean().reset_index()

    # Convert wide → long format so Seaborn can handle grouped bars
    df_long = df_avg_method.melt(id_vars='method', 
                                value_vars=time_names,
                                var_name='Metrics', 
                                value_name='Time')

    plt.figure(figsize=(8,5))
    sns.barplot(data=df_long, x='method', y='Time', hue='Metrics')
    plt.ylabel("Time")
    plt.title("Forward and Backward Time")
    plt.savefig(f"{plot_path}/{plot_name_tag}_time_vs_method.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_time_vs_epoch(df, time_names=['forward_time', 'backward_time'], iteration_name='epoch', plot_path=BASE_DIR, plot_name_tag=""):
    df_avg_epoch = df.groupby(['method', iteration_name])[time_names].mean().reset_index()

    # --- Forward Time Figure ---
    plt.figure(figsize=(8,5))
    sns.lineplot(data=df_avg_epoch, x=iteration_name, y=time_names[0], hue='method', marker=None, dashes=False, linewidth=LINEWIDTH)
    plt.ylabel("Forward Time")
    plt.title(f"Forward Time vs {iteration_name}")
    
    plt.savefig(f"{plot_path}/{plot_name_tag}_forward_time_vs_{iteration_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --- Backward Time Figure ---
    plt.figure(figsize=(8,5))
    sns.lineplot(data=df_avg_epoch, x=iteration_name, y=time_names[1], hue='method', marker=None, dashes=False, linewidth=LINEWIDTH)
    plt.ylabel("Backward Time")
    plt.title(f"Backward Time vs {iteration_name}")
    
    plt.savefig(f"{plot_path}/{plot_name_tag}_backward_time_vs_{iteration_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

# def plot_total_time_vs_method(df, time_names=['forward_time', 'backward_time'], plot_path=BASE_DIR, plot_name_tag=""):
#     # Group by method, average over epochs and seeds
#     df_avg_method = df.groupby('method')[time_names].mean().reset_index()

#     # --- Stacked Bar Chart ---
#     methods = df_avg_method['method']
#     forward = df_avg_method[time_names[0]]
#     backward = df_avg_method[time_names[1]]

#     plt.figure(figsize=(8,5))
#     plt.bar(methods, forward, label=time_names[0], color=palette[0])
#     plt.bar(methods, backward, bottom=forward, label=time_names[1], color=palette[1])
#     plt.ylabel("Time")
#     plt.title("Total Time vs Method")
#     plt.legend()
#     plt.savefig(f"{plot_path}/{plot_name_tag}_total_time_vs_method.pdf", dpi=300, bbox_inches='tight')
#     # plt.savefig(f"{plot_path}/{plot_name_tag}_total_time_vs_method.png", dpi=300, bbox_inches='tight')
#     plt.close()

def plot_total_time_vs_method(
    df,
    task_title_tag=None,
    time_names=["forward_time", "backward_time"],
    plot_path=BASE_DIR,
    plot_name_tag="",
):
    available_methods = set(df["method"].unique())
    # print("hhhhhhh\n", df[df['method']=='FFOQP'])
    # assert("FFOQP" in available_methods)
    filtered_method_order = [m for m in method_order if m in available_methods]
    print("filtered_method_order:", filtered_method_order)
    print("available method: ", available_methods)

    df_avg_method = df.groupby("method")[time_names].mean().reindex(filtered_method_order)

    methods = df_avg_method.index.tolist()
    forward = df_avg_method[time_names[0]].to_numpy()
    backward = df_avg_method[time_names[1]].to_numpy()

    total_times = forward + backward
    max_time = np.nanmax(total_times)
    min_time = 0
    y_max = max_time * 1.05
    
    #### get groups of method to plot them close together
    groups = [
        ["CvxpyLayer", "qpth"], # KKT based
        ["LPGD","BPQP", "dQP"], # optimization based
        ["FFOCP", "FFOQP"], # our methods
    ]
    
    # assert groups respect the filtered method order
    flat_groups = [m for g in groups for m in g]
   
    groups = [[m for m in g if m in filtered_method_order] for g in groups]
    groups = [g for g in groups if len(g) > 0]
    
    flat_groups = [m for g in groups for m in g]
    print("groups:", flat_groups)
    print("methods:", methods)
    assert is_subsequence(flat_groups, methods), "Method groups do not respect the method order."

    # dashed_methods = {"CvxpyLayer", "LPGD", "BPQP", "FFOCP"}  # CP
    
    ######### set bar positions with custom gaps
    # inner_gap = 0.25 #1.5     # spacing between bars inside a group
    # group_gap = len(methods)*0.5/7   ##so that when num methods is 7, group gap is 3.0   # extra spacing between groups
    width = 0.08 #0.75 # bar width
    inner_gap = 0.08 #1.5     # spacing between bars inside a group
    group_gap = len(methods)*0.125/7   ##so that when num methods is 7, group gap is 3.0   # extra spacing between groups


    x = []
    # labels = []
    pos = 0.0

    for g in groups:
        for m in g:
            x.append(pos)
            # labels.append(m)
            pos += inner_gap
        pos += group_gap  # extra space after each group
        
    ########## set bar label positions with custom gaps
    # inner_gap = 1.5     # spacing between bars inside a group
    # group_gap = len(methods)*2/7   ##so that when num methods is 7, group gap is 3.0   # extra spacing between groups
    # label_x = []
    # pos = 0.0
    # for g in groups:
    #     for m in g:
    #         label_x.append(pos)
    #         pos += inner_gap
    #     pos += group_gap  # extra space after each group
    label_x = x

    
    # x = np.arange(len(methods))
    # width = 0.25 #0.75

    fig, ax = plt.subplots(figsize=(14, 7))
    sf = mticker.ScalarFormatter(useOffset=False)
    sf.set_scientific(False)
    ax.yaxis.set_major_formatter(sf)
    ax.yaxis.get_offset_text().set_visible(False)

    for i, m in enumerate(methods):
        # is_dashed = (m in dashed_methods)
        # ls = (0, (4, 2)) if is_dashed else "solid"
        # lw = 2.0 if is_dashed else 1.5

        # ax.bar(x[i], forward[i], width=width,
        #        color=palette[0], edgecolor="black", linewidth=lw, linestyle=ls)
        # ax.bar(x[i], backward[i], bottom=forward[i], width=width,
        #        color=palette[1], edgecolor="black", linewidth=lw, linestyle=ls)
        
        ax.bar(x[i], forward[i], width=width,
               color=palette[0], edgecolor="black")
        ax.bar(x[i], backward[i], bottom=forward[i], width=width,
               color=palette[1], edgecolor="black")


    ax.set_xticks(label_x)
    methods = ["Cvxpy\nLayer" if m == "CvxpyLayer" else m for m in methods]
    ax.set_xticklabels(methods)
    ax.set_ylabel("Time")
    if task_title_tag is not None:
        ax.set_title(f"{task_title_tag}: Total Time vs Methods")
    else:
        ax.set_title("Total Time vs Methods")
    ax.set_ylim(min_time, y_max)
    min_x = min([min(x), min(label_x)])
    max_x = max([max(x), max(label_x)])
    ax.set_xlim(min_x - width, max_x + width)

    handles = [
        Line2D([], [], linestyle="none", label="Comp. Time"),
        Patch(facecolor=palette[0], edgecolor="black", label="Forward"),
        Patch(facecolor=palette[1], edgecolor="black", label="Backward"),
        # Line2D([], [], linestyle="none", label=""),
        # Line2D([], [], linestyle="none", label="Solvers"),
        # Line2D([0], [0], color="black", linewidth=2, linestyle=(0, (4, 2)), label="CP methods"),
        # Line2D([0], [0], color="black", linewidth=2, linestyle="solid", label="QP methods"),
    ]

    leg = ax.legend(
        handles=handles,
        loc="upper right",
        frameon=True,
        handlelength=2.2,
        handletextpad=0.8,
        borderpad=0.7,
        labelspacing=0.1,
    )
    leg._legend_box.align = "left"

    for h, t in zip(leg.legend_handles, leg.get_texts()):
        txt = t.get_text()
        if txt in ("Comp. Time", "Solvers"):
            t.set_weight("bold")
            if hasattr(h, "set_visible"):
                h.set_visible(False)
        if txt == "":
            if hasattr(h, "set_visible"):
                h.set_visible(False)
            t.set_color((0, 0, 0, 0))

    fig.savefig(f"{plot_path}/{plot_name_tag}_total_time_vs_method.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
def plot_loss_vs_epoch(df, loss_metric_name, task_title_tag=None, iteration_name='epoch', plot_path=BASE_DIR, plot_name_tag="", loss_range=None, stride=50):
    df_avg_epoch = df.groupby(['method', iteration_name])[[loss_metric_name]].mean().reset_index()
    
    # print(df_avg_epoch)
    # epoch_0_loss = -0.00950829166918993
    # df_avg_epoch.loc[df_avg_epoch[iteration_name] == 0, loss_metric_name] = epoch_0_loss
    # df_avg_epoch = df_avg_epoch[df_avg_epoch[iteration_name] % stride == 0]

    # --- Forward Time Figure ---
    plt.figure(figsize=(8,5))
    ax = sns.lineplot(data=df_avg_epoch, x=iteration_name, y=loss_metric_name, hue='method', dashes=False, linewidth=LINEWIDTH, hue_order=method_order)
    plt.ylabel("loss")
    if task_title_tag is not None:
        plt.title(f"{task_title_tag}: Loss vs iteration")
    else:
        plt.title("Loss vs iteration")
    
    # plt.legend(
    #     title=None,
    #     loc="lower center",
    #     bbox_to_anchor=(0.5, -0.25),
    #     ncol=(df["method"].nunique()),           
    #     frameon=False
    # )
    
    if loss_range is not None:
        ax.set_ylim(loss_range)
    
    plt.savefig(f"{plot_path}/{plot_name_tag}_{loss_metric_name}_vs_{iteration_name}.pdf", dpi=300, bbox_inches='tight')
    plt.close()

def plot_total_time_vs_method_by_ydim(df, plot_path, time_names=('forward_time','backward_time'), tag="syn"):
    os.makedirs(plot_path, exist_ok=True)
    ydims = sorted(df["ydim"].dropna().unique().astype(int))
    for y in ydims:
        d = df[df["ydim"] == y]
        plot_total_time_vs_method(
            d,
            time_names=list(time_names),
            plot_path=plot_path,
            plot_name_tag=f"{tag}_ydim{y}"
        )

def plot_total_time_vs_ydim_for_methods(
    df,
    methods_subset,
    markers_dict=markers_dict,
    time_names=('forward_time', 'backward_time'),
    plot_path=BASE_DIR,
    plot_name_tag="syn_methods_vs_ydim"
):
    os.makedirs(plot_path, exist_ok=True)
    df_sub = df[df["method"].isin(methods_subset)].copy()
    df_sub = df_sub.dropna(subset=["ydim"])
    df_sub["ydim"] = df_sub["ydim"].astype(int)

    df_group = (
        df_sub
        .groupby(["method", "ydim"])[list(time_names)]
        .mean()
        .reset_index()
    )

    df_group["total_time"] = df_group[time_names[0]] + df_group[time_names[1]]

    plt.figure(figsize=(8, 5))

    for idx, method in enumerate(methods_subset):
        d = df_group[df_group["method"] == method].copy()
        d = d.sort_values("ydim")
        marker = markers_dict.get(method, markers[idx % len(markers)])
        plt.plot(
            d["ydim"],
            d["total_time"],
            label=method,
            marker=marker,
            linewidth=LINEWIDTH,
        )

    plt.xlabel("y_dim")
    plt.ylabel("Total Time")
    plt.title("Total Time vs y_dim (selected methods)")
    
    # Reorder legend items to: CvxpyLayer, qpth, LPGD, BPQP, dQP, FFOCP, FFOQP
    legend_order = ["CvxpyLayer", "qpth", "LPGD", "BPQP", "dQP", "FFOCP", "FFOQP"]
    handles, labels = plt.gca().get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))
    ordered_handles = [label_to_handle[l] for l in legend_order if l in label_to_handle]
    ordered_labels = [l for l in legend_order if l in label_to_handle]
    plt.legend(ordered_handles, ordered_labels, ncol=len(ordered_labels))
    
    plt.tight_layout()
    plt.savefig(
        f"{plot_path}/{plot_name_tag}_total_time_vs_ydim.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_total_loss_vs_method_by_ydim(df, plot_path):
    os.makedirs(plot_path, exist_ok=True)
    ydims = sorted(df["ydim"].dropna().unique().astype(int))
    for y in ydims:
        d = df[df["ydim"] == y]
        plot_loss_vs_epoch(
            d,
            loss_metric_name="train_df_loss",
            iteration_name="iter",
            plot_path=plot_path,
            plot_name_tag=f"syn_ydim{y}"
        )


def plot_total_time_vs_method_by_backwardTol(df, time_names=['forward_time', 'backward_time'],
                                             plot_path=BASE_DIR, plot_name_tag=""):
    """
    Plots stacked bar chart of total time vs backward tolerance for each method.
    Each method is plotted separately in a for-loop to ensure only one method at a time.
    """
    os.makedirs(plot_path, exist_ok=True)

    df = df.dropna(subset=['method', 'backwardTol']).copy()

    # Get unique methods
    methods = df['method'].unique()
    assert(len(methods)==1)

    for method in methods:
        d = df[df['method'] == method].copy()

        # Remove duplicate backwardTol per method (take mean)
        d = d.groupby('backwardTol')[time_names].mean().reset_index()

        plt.figure(figsize=(8,5))
        plt.bar(d['backwardTol'].astype(str), d[time_names[0]], label=time_names[0], color=palette[0])
        plt.bar(d['backwardTol'].astype(str), d[time_names[1]], bottom=d[time_names[0]],
                label=time_names[1], color=palette[1])
        
        plt.xlabel("backward tolerance")
        plt.ylabel("Time")
        plt.title(f"Total Time vs Backward Tolerance ({method})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{plot_path}/{plot_name_tag}_total_time_vs_backwardTol_{method}.pdf", 
                    dpi=300, bbox_inches='tight')
        plt.close()


def plot_loss_vs_epoch_method_tol(df, loss_metric='train_df_loss', iteration='epoch',
                                  plot_path=BASE_DIR, plot_name_tag=""):
    """
    Simple: plot loss vs epoch, each line = one (method, backwardTol) tuple.
    """
    import matplotlib.pyplot as plt
    import os

    os.makedirs(plot_path, exist_ok=True)
    df = df.dropna(subset=['method', 'backwardTol', iteration, loss_metric])

    plt.figure(figsize=(8,5))

    for (method, tol), d in df.groupby(['method', 'backwardTol']):
        d = d.sort_values(iteration)
        plt.plot(d[iteration], d[loss_metric], label=f"{method}-tol{tol:.0e}")

    plt.xlabel(iteration)
    plt.ylabel("loss")
    plt.title(f"Loss vs {iteration}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plot_path}/{plot_name_tag}_{loss_metric}_vs_{iteration}.pdf", dpi=300, bbox_inches='tight')
    plt.close()

def plot_time_scaling_vs_ydim(
    df,
    time_names=("forward_time", "backward_time"),
    plot_path=BASE_DIR,
    plot_name_tag="syn",
    methods_order=None,
    markers_dict=markers_dict,
    dashed_methods=("CvxpyLayer", "LPGD", "BPQP", "FFOCP"),  # CP
    agg="mean",
    logy=False,
    y_min=None,
    y_max=None,
    filter_backwardTol=None,
):
    os.makedirs(plot_path, exist_ok=True)

    d = df.copy().dropna(subset=["ydim", "method"])
    d["ydim"] = d["ydim"].astype(int)

    if "backwardTol" in d.columns and filter_backwardTol is not None:
        d = d[np.isclose(d["backwardTol"].astype(float), float(filter_backwardTol))]

    gfunc = "median" if agg == "median" else "mean"
    g = getattr(d.groupby(["method", "ydim"], as_index=False)[list(time_names)], gfunc)()
    g["total_time"] = g[time_names[0]] + g[time_names[1]]

    if methods_order is None:
        methods_order = (
            [m for m in d["method"].cat.categories if m in g["method"].unique()]
            if isinstance(d["method"].dtype, pd.CategoricalDtype)
            else sorted(g["method"].unique())
        )
    else:
        methods_order = [m for m in methods_order if m in g["method"].unique()]

    plt.rcParams.update({
        "font.size": 18, "axes.titlesize": 20, "axes.labelsize": 18,
        "legend.fontsize": 16, "xtick.labelsize": 16, "ytick.labelsize": 16,
    })

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2), sharex=True)
    panels = [
        ("total_time", "Total time"),
        (time_names[0], "Forward time"),
        (time_names[1], "Backward time"),
    ]

    for ax, (col, title) in zip(axes, panels):
        for i, method in enumerate(methods_order):
            gg = g[g["method"] == method].sort_values("ydim")
            if gg.empty:
                continue

            ax.plot(
                gg["ydim"], gg[col],
                label=method,
                marker=markers_dict.get(method, markers[i % len(markers)]),
                markersize=5.5,
                linewidth=2.0,
                linestyle="solid",
            )

        ax.set_title(title)
        ax.set_xlabel("y dim")
        ax.grid(True, which="major", alpha=0.25)
        ax.grid(True, which="minor", alpha=0.12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if logy:
            ax.set_yscale("log")
        if y_min is not None or y_max is not None:
            ax.set_ylim(bottom=y_min, top=y_max)

        ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
        ax.yaxis.get_offset_text().set_visible(False)

    axes[0].set_ylabel("time (s)")

    # Reorder legend items to: CvxpyLayer, qpth, LPGD, BPQP, dQP, FFOCP, FFOQP (single row)
    legend_order = ["CvxpyLayer", "qpth", "LPGD", "BPQP", "dQP", "FFOCP", "FFOQP"]
    handles, labels = axes[0].get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))
    ordered_handles = [label_to_handle[l] for l in legend_order if l in label_to_handle]
    ordered_labels = [l for l in legend_order if l in label_to_handle]
    fig.legend(
        ordered_handles, ordered_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        fontsize=18,
        ncol=len(ordered_labels),
        frameon=False,
        handlelength=2.4,
        handletextpad=0.6,
        columnspacing=1.0,
    )

    # leave space at bottom for legend
    fig.tight_layout(rect=[0, 0.10, 1, 1])

    out = os.path.join(plot_path, f"{plot_name_tag}_time_scaling_vs_ydim.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

def discover_batch_dirs(root_dir="..", prefix="synthetic_results_"):
    cand = glob.glob(os.path.join(root_dir, f"{prefix}*"))
    out = []
    for p in cand:
        if not os.path.isdir(p):
            continue
        name = os.path.basename(p.rstrip("/"))
        mo = re.match(rf"^{re.escape(prefix)}(\d+)$", name)
        if mo:
            out.append((int(mo.group(1)), p))
    out.sort(key=lambda x: x[0])
    if not out:
        raise FileNotFoundError(f"No dirs like {prefix}{{int}} under {root_dir}")
    return out


def load_results_across_batches(
    root_dir="..",
    batch_sizes=None,
    loader_fn=None,
    methods=None,
    methods_legend=None,
    prefix="synthetic_results_",
):
    if loader_fn is None:
        raise ValueError("Please pass loader_fn=load_results_CP or load_results_QP")
    if methods is None:
        methods = METHODS
    if methods_legend is None:
        methods_legend = METHODS_LEGEND

    if batch_sizes is None:
        batch_dirs = discover_batch_dirs(root_dir=root_dir, prefix=prefix)
    else:
        batch_dirs = [(int(bs), os.path.join(root_dir, f"{prefix}{int(bs)}")) for bs in batch_sizes]

    dfs = []
    for bs, dpath in batch_dirs:
        if not os.path.isdir(dpath):
            print(f"[WARN] missing dir: {dpath}, skip")
            continue

        df = loader_fn(base_dir=dpath, methods=methods, methods_legend=methods_legend)
        df = df.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)
        df["batch_size"] = int(bs)
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError("No results loaded across batches. Check dirs / patterns.")
    out = pd.concat(dfs, ignore_index=True, sort=False)
    return out


# def plot_time_scaling_vs_batch(
#     df,
#     time_names=("forward_time", "backward_time"),
#     plot_path=".",
#     plot_name_tag="syn",
#     methods_order=None,
#     dashed_methods=("CvxpyLayer", "LPGD", "BPQP", "FFOCP"),  # CP
#     agg="mean",
#     logx=True,
#     logy=False,
#     y_min=None,
#     y_max=None,
#     filter_ydim=None,
#     filter_backwardTol=None,
# ):
#     os.makedirs(plot_path, exist_ok=True)
#     markers = ["o", "s", "D", "^", "v", "x", "P", "s"]
#     markers_dict = {method: markers[i] for i, method in enumerate(methods_order)}

#     d = df.copy()

#     if "batch_size" not in d.columns:
#         raise KeyError("df must contain column 'batch_size' (use load_results_across_batches).")
#     if "method" not in d.columns:
#         raise KeyError("df must contain column 'method'.")

#     if filter_ydim is not None:
#         if "ydim" not in d.columns:
#             raise KeyError("filter_ydim is set but df has no 'ydim' column.")
#         d = d.dropna(subset=["ydim"])
#         d = d[d["ydim"].astype(int) == int(filter_ydim)]

#     if "backwardTol" in d.columns and filter_backwardTol is not None:
#         d = d.dropna(subset=["backwardTol"])
#         d = d[np.isclose(d["backwardTol"].astype(float), float(filter_backwardTol))]

#     d = d.dropna(subset=["batch_size", "method"])
#     d["batch_size"] = d["batch_size"].astype(int)

#     gfunc = "median" if agg == "median" else "mean"
#     g = getattr(d.groupby(["method", "batch_size"], as_index=False)[list(time_names)], gfunc)()
#     g["total_time"] = g[time_names[0]] + g[time_names[1]]

#     if methods_order is None:
#         if isinstance(d["method"].dtype, pd.CategoricalDtype):
#             methods_order = [m for m in d["method"].cat.categories if m in g["method"].unique()]
#         else:
#             methods_order = sorted(g["method"].unique())
#     else:
#         methods_order = [m for m in methods_order if m in g["method"].unique()]

#     plt.rcParams.update({
#         "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
#         "legend.fontsize": 11, "xtick.labelsize": 12, "ytick.labelsize": 12,
#     })

#     fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2), sharex=True)
#     panels = [
#         ("total_time", "Total time"),
#         (time_names[0], "Forward time"),
#         (time_names[1], "Backward time"),
#     ]
#     dashed_methods = set(dashed_methods)

#     for ax, (col, title) in zip(axes, panels):
#         for i, method in enumerate(methods_order):
#             gg = g[g["method"] == method].sort_values("batch_size")
#             if gg.empty:
#                 continue

#             is_dashed = method in dashed_methods
#             ax.plot(
#                 gg["batch_size"], gg[col],
#                 label=method,
#                 marker=markers_dict.get(method, markers[i % len(markers)]),
#                 markersize=5.5,
#                 linewidth=2.2 if is_dashed else 2.0,
#                 linestyle=(0, (4, 2)) if is_dashed else "solid",
#             )

#         ax.set_title(title)
#         ax.set_xlabel("batch size")
#         ax.grid(True, which="major", alpha=0.25)
#         ax.grid(True, which="minor", alpha=0.12)
#         ax.spines["top"].set_visible(False)
#         ax.spines["right"].set_visible(False)

#         if logx:
#             ax.set_xscale("log", base=2)
#             ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
#             ax.set_xticks(sorted(g["batch_size"].unique()))
#             ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())

#         if logy:
#             ax.set_yscale("log")
#         if y_min is not None or y_max is not None:
#             ax.set_ylim(bottom=y_min, top=y_max)

#         ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
#         ax.yaxis.get_offset_text().set_visible(False)

#     axes[0].set_ylabel("time (s)")

#     handles, labels = axes[0].get_legend_handles_labels()
#     ncol = min(len(labels), 4)
#     fig.legend(
#         handles, labels,
#         loc="lower center",
#         bbox_to_anchor=(0.5, -0.02),
#         fontsize=14,
#         ncol=ncol,
#         frameon=False,
#         handlelength=2.4,
#         handletextpad=0.6,
#         columnspacing=1.2,
#     )

#     fig.tight_layout(rect=[0, 0.12, 1, 1])

#     suffix = ""
#     if filter_ydim is not None:
#         suffix += f"_ydim{int(filter_ydim)}"
#     if filter_backwardTol is not None:
#         suffix += f"_bwdTol{filter_backwardTol:g}"

#     out = os.path.join(plot_path, f"{plot_name_tag}_time_scaling_vs_batch{suffix}.pdf")
#     fig.savefig(out, dpi=300, bbox_inches="tight")
#     plt.close(fig)

#     print(f"[saved] {out}")

def plot_time_scaling_vs_batch(
    df,
    plot_path=".",
    plot_name_tag="syn",
    methods_order=None,
    markers_dict=None,
    dashed_methods=("CvxpyLayer", "LPGD", "BPQP", "FFOCP"),
    agg="mean",                 # "mean" or "median"
    use_log2_x=True,
    logy=False,
    y_min=None,
    y_max=None,
    filter_backwardTol=None,    # optional if df has backwardTol
):
    os.makedirs(plot_path, exist_ok=True)
    d = df.copy().dropna(subset=["batch_size", "method"])
    d["batch_size"] = d["batch_size"].astype(int)

    if "backwardTol" in d.columns and filter_backwardTol is not None:
        d = d[np.isclose(d["backwardTol"].astype(float), float(filter_backwardTol))]

    for c in ["forward_time", "backward_time"]:
        if c not in d.columns:
            raise ValueError(f"missing column: {c}")
        d[c] = d[c].astype(float)

    # derive step/overhead
    if "overhead_time" in d.columns:
        d["overhead_time"] = d["overhead_time"].astype(float)
        d["step_time"] = d["forward_time"] + d["backward_time"] + d["overhead_time"]
    else:
        if "step_time" in d.columns:
            d["step_time"] = d["step_time"].astype(float)
        elif "total_time" in d.columns:
            d["step_time"] = d["total_time"].astype(float)
        else:
            d["step_time"] = d["forward_time"] + d["backward_time"]
        d["overhead_time"] = (d["step_time"] - d["forward_time"] - d["backward_time"]).clip(lower=0.0)

    gfunc = "median" if agg == "median" else "mean"
    g = getattr(
        d.groupby(["method", "batch_size"], as_index=False)[
            ["forward_time", "backward_time", "overhead_time", "step_time"]
        ],
        gfunc,
    )()

    eps = 1e-12
    g["throughput"] = g["batch_size"] / g["step_time"].clip(lower=eps)
    g["per_sample"] = g["step_time"] / g["batch_size"].clip(lower=1)       # sec/sample

    # method order (match your style)
    if methods_order is None:
        methods_order = (
            [m for m in d["method"].cat.categories if m in g["method"].unique()]
            if isinstance(d["method"].dtype, pd.CategoricalDtype)
            else sorted(g["method"].unique())
        )
    else:
        methods_order = [m for m in methods_order if m in g["method"].unique()]

    # style (copy yours)
    plt.rcParams.update({
        "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
        "legend.fontsize": 11, "xtick.labelsize": 12, "ytick.labelsize": 12,
    })

    dashed_methods = set(dashed_methods)
    markers = ["o", "s", "D", "^", "v", "P", "X", "*", "<", ">"]

    xvals = np.array(sorted(g["batch_size"].unique().tolist()), dtype=int)
    xplot = np.log2(xvals) if use_log2_x else xvals
    xticks = xplot
    xtick_labels = [str(v) for v in xvals]
    xlabel = "batch size" if use_log2_x else "batch size"

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2), sharex=True)
    panels = [
        ("throughput", "Throughput (samples/sec)"),
        ("step_time", "Time per step (sec)"),
        ("per_sample", "Per-sample cost (sec/sample)"),
    ]

    for ax, (col, title) in zip(axes, panels):
        for i, method in enumerate(methods_order):
            gg = g[g["method"] == method].sort_values("batch_size")
            if gg.empty:
                continue

            xx = np.log2(gg["batch_size"].to_numpy()) if use_log2_x else gg["batch_size"].to_numpy()
            is_dashed = method in dashed_methods
            mk = (markers_dict.get(method) if markers_dict else None) or markers[i % len(markers)]

            ax.plot(
                xx, gg[col].to_numpy(),
                label=method,
                marker=mk,
                markersize=5.5,
                linewidth=2.2 if is_dashed else 2.0,
                linestyle=(0, (4, 2)) if is_dashed else "solid",
            )

            # decomposition shading only on panel 2 (step_time)
            if col == "step_time":
                fwd = gg["forward_time"].to_numpy()
                bwd = gg["backward_time"].to_numpy()
                ovh = gg["overhead_time"].to_numpy()
                ax.fill_between(xx, 0, fwd, alpha=0.06)
                ax.fill_between(xx, fwd, fwd + bwd, alpha=0.06)
                ax.fill_between(xx, fwd + bwd, fwd + bwd + ovh, alpha=0.06)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels)

        ax.grid(True, which="major", alpha=0.25)
        ax.grid(True, which="minor", alpha=0.12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if logy:
            ax.set_yscale("log")
        if y_min is not None or y_max is not None:
            ax.set_ylim(bottom=y_min, top=y_max)

        ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
        ax.yaxis.get_offset_text().set_visible(False)

    # shared legend (same as your function)
    handles, labels = axes[0].get_legend_handles_labels()
    ncol = min(len(labels), 4)
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        fontsize=14,
        ncol=ncol,
        frameon=False,
        handlelength=2.4,
        handletextpad=0.6,
        columnspacing=1.2,
    )
    fig.tight_layout(rect=[0, 0.12, 1, 1])

    out = os.path.join(plot_path, f"{plot_name_tag}_batch_scaling_3panels.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def load_comp_grad_results_metrics(
    base_dir,
    methods,
    methods_legend,
    parse_backwardTol=True,
):
    dfs = []
    for mm in methods:
        pattern = os.path.join(base_dir, mm, "*.csv")
        for fp in sorted(glob.glob(pattern)):
            m = mm.removesuffix("_steps")
            df = pd.read_csv(fp)
            df["method"] = methods_legend[m]

            fname = os.path.basename(fp)
            def grab(pat, cast=float):
                mo = re.search(pat, fname)
                return cast(mo.group(1)) if mo else np.nan

            df["seed"] = grab(r"_seed(\d+)", int)
            df["ydim"] = grab(r"ydim(\d+)", int)
            df["lr"]   = grab(r"lr([0-9eE\.\-]+)", float)

            if parse_backwardTol:
                df["backwardTol"] = grab(r"backwardTol([0-9eE\+\-\.]+?)(?:_|\.csv)", float)

            dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No CSVs found under {base_dir}.")
    out = pd.concat(dfs, ignore_index=True, sort=False)
    out = out.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)
    return out


def plot_comp_grad_metrics_vs_iter_by_ydim(
    df,
    metrics=("cosine_sim_val", "l2_dist_val", "l_inf_dist_val"),
    plot_path=".",
    plot_name_tag="syn",
    filter_method=None,
    filter_backwardTol=None,
    agg="mean",
    logy_dist=True,                     
    cosine_ylim=None,                   
    ydim_order=None,                    
    markers=("o","s","D","^","v","x","P","*"),
    linewidth=2.0,
    legend_ncol=5,
    legend_y=-0.06,
    legend_fontsize=11,
):
    os.makedirs(plot_path, exist_ok=True)

    d = df.copy().dropna(subset=["iter", "ydim"])
    d["iter"] = d["iter"].astype(int)
    d["ydim"] = d["ydim"].astype(int)

    if filter_method is not None:
        d = d[d["method"] == filter_method]

    if filter_backwardTol is not None and "backwardTol" in d.columns:
        d = d[np.isclose(d["backwardTol"].astype(float), float(filter_backwardTol))]

    if "backwardTol" in d.columns and filter_backwardTol is None and d["backwardTol"].nunique() > 1:
        bt = d["backwardTol"].mode().iloc[0]
        d = d[np.isclose(d["backwardTol"].astype(float), float(bt))]

    gfunc = "median" if agg == "median" else "mean"
    g = getattr(d.groupby(["ydim", "iter"], as_index=False)[list(metrics)], gfunc)()

    if ydim_order is None:
        ydim_order = sorted(g["ydim"].unique())

    titles = {
        "cosine_sim_val": "Cosine similarity",
        "l2_dist_val":    r"$\ell_2$ distance",
        "l_inf_dist_val": r"$\ell_\infty$ distance",
    }

    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2), sharex=True)

    for ax, metric in zip(axes, metrics):
        for i, ydim in enumerate(ydim_order):
            gg = g[g["ydim"] == ydim].sort_values("iter")
            if gg.empty:
                continue
            ax.plot(
                gg["iter"], gg[metric],
                label=f"ydim={ydim}",
                marker=markers[i % len(markers)],
                markersize=5.0,
                linewidth=linewidth,
            )

        ax.set_title(titles.get(metric, metric))
        ax.set_xlabel("iterations")
        ax.grid(True, which="major", alpha=0.25)
        ax.grid(True, which="minor", alpha=0.12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
        ax.yaxis.get_offset_text().set_visible(False)

        if metric == "cosine_sim_val" and cosine_ylim is not None:
            ax.set_ylim(*cosine_ylim)
        if metric != "cosine_sim_val" and logy_dist:
            ax.set_yscale("log")

    axes[0].set_ylabel("value")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=min(len(labels), legend_ncol),
        frameon=False,
        fontsize=legend_fontsize,
        handlelength=2.4,
        handletextpad=0.6,
        columnspacing=1.2,
        borderaxespad=0.0,
    )

    fig.tight_layout(rect=[0, 0.14, 1, 1])

    tag = plot_name_tag
    if filter_method is not None:
        tag += f"_{filter_method}"
    out = os.path.join(plot_path, f"{tag}_metrics_vs_iter_by_ydim.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)



if __name__=="__main__":
    task_title = "SOCP"
    df = load_results_CP()
    df = df.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)
    df["method"] = pd.Categorical(df["method"], categories=method_order, ordered=True)
    df = df[df["ydim"] == 800]

    print("loaded df")
    print(df)
    # assert(1==0)

    plot_total_time_vs_method(df, task_title_tag=task_title, time_names=['forward_time', 'backward_time'], plot_path=BASE_DIR, plot_name_tag="syn_soc_ydim800")
    # plot_total_time_vs_method_by_backwardTol(df, time_names=['forward_time', 'backward_time'],
    #                                          plot_path=BASE_DIR, plot_name_tag="syn")
   
   #########################################
    df = load_results_CP(methods=METHODS_STEPS)
    df = df.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)
    df["method"] = pd.Categorical(df["method"], categories=method_order, ordered=True)
    df = df[df["ydim"] == 800]

    print("loaded df steps")

    # plot_total_time_vs_method(df, time_names=['forward_solve_time', 'backward_solve_time'], plot_path=BASE_DIR, plot_name_tag="syn_steps_solve")
    # plot_total_time_vs_method(df, time_names=['forward_setup_time', 'backward_setup_time'], plot_path=BASE_DIR, plot_name_tag="syn_steps_setup")
    
    plot_loss_vs_epoch(df, "train_df_loss", task_title_tag=task_title, iteration_name='iter', plot_path=BASE_DIR, plot_name_tag="syn_soc_steps_ydim800")
    # plot_loss_vs_epoch_method_tol(df, loss_metric='train_df_loss', iteration='iter',
    #                               plot_path=BASE_DIR, plot_name_tag="syn_steps")
    