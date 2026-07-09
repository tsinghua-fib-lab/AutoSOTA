# %%
import argparse
import math
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# %%


def set_multicolor_xlabel(
    ax,
    x_idx,
    line1,
    line2,
    line2_color="red",
    align_h=None,
    rotation=0,
    dx=0,
    dy=0,
):
    """
    Replaces the x-label at x_idx with a custom multicolor label.
    Safe to call multiple times for different indices.
    """
    labels = ax.get_xticklabels()
    if x_idx >= len(labels):
        return

    target_label = labels[x_idx]

    # 1. Get coordinates (Data X, Axis Y)
    pos = target_label.get_position()
    pos_x = pos[0]

    # 2. CRITICAL FIX: Set the original text to empty string
    # This effectively "deletes" it from the visual plot while keeping the object alive
    target_label.set_text("")

    # 3. Determine Alignment
    # For rotated labels (e.g. 30 deg), 'right' alignment is standard
    align_h = "right" if rotation > 10 else "center"

    # 4. Shared Properties
    # Note: We use a small vertical offset (-5) to mimic standard tick padding
    common_props = dict(
        xy=(pos_x, 0),
        xycoords=ax.get_xaxis_transform(),
        xytext=(0 + dx, -5 + dy),
        textcoords="offset points",
        ha=align_h,
        va="top",
        rotation=rotation,
        fontsize=10,  # Match your plot's font size
    )

    # 5. Add "Top" Line (Black)
    # The trailing newline "\n" pushes the text up, leaving space below
    ax.annotate(line1 + "\n", color="black", **common_props)

    # 6. Add "Bottom" Line (Color)
    # The leading newline "\n" pushes the text down, leaving space above
    ax.annotate(
        "\n" + line2, color=line2_color, fontweight="bold", **common_props
    )


def get_legend_by_row(handles, labels, ncol=None):

    ncol = ncol if ncol else len(labels) // 2

    # Reorder handles and labels to simulate 'row-by-row' filling
    # Calculate how many rows are needed
    nrows = math.ceil(len(handles) / ncol)

    reordered_handles = []
    reordered_labels = []

    # Iterate columns first, then rows, picking the item that belongs at (row, col)
    for c in range(ncol):
        for r in range(nrows):
            # Calculate the index of the item in the original list
            idx = r * ncol + c
            if idx < len(handles):
                reordered_handles.append(handles[idx])
                reordered_labels.append(labels[idx])

    return reordered_handles, reordered_labels


def make_boxplot(
    df_plot,
    yaxis_var,
    xaxis_var,
    xaxis_order,
    new_x_labels,
    hue_var,
    hue_order,
    new_hue_labels,
    box_width=0.6,
    box_gap=0.1,
    xlabel_visible=True,
    hline_val=None,
    hline_label=None,
    draw_legend=True,
    legend_title=None,
    legend_by_row=True,
    legend_ncol=None,
    ax=None,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    # 1. Plotting
    # Ensure xaxis_order matches the data type of df_plot[xaxis_var] (likely strings)
    # If df_plot has "1", "2", and xaxis_order has 1, 2, the plot will be empty.
    xaxis_order = [str(x) for x in xaxis_order]

    sns.boxplot(
        data=df_plot,
        x=xaxis_var,
        y=yaxis_var,
        order=xaxis_order,
        hue=hue_var,
        hue_order=hue_order,
        palette="Set2",
        fill=True,
        showfliers=False,
        dodge="auto",
        gap=box_gap,
        width=box_width,
        linewidth=1.0,
        ax=ax,
    )

    if new_x_labels is not None:
        ax.set_xticks(range(len(new_x_labels)))
        if xlabel_visible:
            ax.set_xticklabels(new_x_labels)
        else:
            ax.set_xticklabels([])  # Hide text for top plot

    if hline_val is not None:
        # Note: We omit 'label' here to prevent it from messing up get_legend_handles_labels()
        # We will add it manually to the legend later if needed.
        ax.axhline(
            y=hline_val,
            color="gray",
            linestyle="--",
            linewidth=1.5,
        )

    # 2. Legend Logic
    # Get handles from the boxplot.
    # This usually grabs 6 items (if hue_order has 6).
    handles, _ = ax.get_legend_handles_labels()

    # Force use of new labels
    # Use existing labels sliced to n_hue as fallback
    current_labels = new_hue_labels if new_hue_labels else hue_order

    # Safety: Ensure lists are same length before reordering to avoid IndexError
    # (In case handles includes extra items or fewer items due to empty data)
    min_len = min(len(handles), len(current_labels))
    handles = handles[:min_len]
    current_labels = current_labels[:min_len]

    if legend_by_row:
        handles, current_labels = get_legend_by_row(
            handles, current_labels, ncol=legend_ncol
        )

    # Add the horizontal line to legend manually if requested
    if hline_label is not None:
        # Create a proxy artist for the line
        line_handle = Line2D(
            [0], [0], color="gray", linestyle="--", linewidth=1.5
        )
        handles.append(line_handle)
        current_labels.append(hline_label)

    # 3. Create Legend
    # USE ax.legend INSTEAD OF sns.move_legend
    if draw_legend:
        ax.legend(
            handles=handles,
            labels=current_labels,
            # loc="lower center",
            # bbox_to_anchor=(0.5, 1),
            ncol=legend_ncol,
            title=legend_title,
            frameon=True,
        )
    else:
        ax.get_legend().remove()

    return ax, handles, current_labels


def split_labels(strings):
    """
    Identifies the unique part of each string by removing the common
    prefix and common suffix among a list of strings.
    """
    if not strings:
        return [], ""
    if len(strings) == 1:
        return "", strings

    # Find common prefix
    prefix = os.path.commonprefix(strings)

    # Find common suffix
    reversed_strings = [s[::-1] for s in strings]
    suffix = os.path.commonprefix(reversed_strings)[::-1]

    # Strip common prefix and suffix
    labels = [s[len(prefix) : len(s) - len(suffix)] for s in strings]
    return labels, prefix + "LABELS" + suffix


def gather_results(
    results_dir,
    exp_grp,
    exp_ids,
    ckpt_strategy,
    metric_key,
    suffix,
    exclude_sim_ids=None,
):

    labels, base_name = split_labels(exp_ids)

    metric_key = metric_key.replace("/", "_")
    if exclude_sim_ids:
        suffix += "_no" + "".join(map(str, exclude_sim_ids))

    exp_map = {}
    for label, exp_id in zip(labels, exp_ids):
        # Clean up labels like '_meanvar_' or '-poly2-'
        clean_label = label.strip("_-")
        filepath = os.path.join(
            results_dir,
            exp_grp,
            f"summary_{exp_id}_{ckpt_strategy}_{metric_key}_{suffix}.csv",
        )
        exp_map[clean_label] = filepath

    all_data = []
    for label, filepath in exp_map.items():
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df["group_var"] = label
            all_data.append(df)
        else:
            print(f"Warning: File not found: {filepath}")

    if not all_data:
        print("No data available to plot.")
        return
    else:
        df_all = pd.concat(all_data, ignore_index=True)
        return df_all


# %%
# Different mixing functions

mixing_fns = [
    "normalclamppolymix1",
    "normalclamppolymix2",
    "normalclamppolymix3",
]
exp_grp = "new_mlpnormenc_inv_polyind_ms100"
exp_ids = [
    f"new_{mix_fn}_mlpnormenc_poly2inv_polyind_ms100" for mix_fn in mixing_fns
] + ["new_invmlpmix_mlpnormenc_mlpreludec_poly2inv_polyind_ms100"]
ckpt_strategy = "last"
metric_key = "val/inv_loss"
exclude_sim_ids = [0, 4, 8]

df_all = gather_results(
    results_dir="results",
    exp_grp=exp_grp,
    exp_ids=exp_ids,
    ckpt_strategy=ckpt_strategy,
    metric_key=metric_key,
    suffix="insample",
    exclude_sim_ids=exclude_sim_ids,
)

df_plot = df_all[df_all["pop_num"] == -1].copy()
df_plot = df_plot[
    df_plot["instrument"].isin(
        [
            "Z",
            "hW",
            "hWchV",
            "tslsCondPop",
            "limlCondPop",
            "MREgger",
            "MREggerCondPop",
        ]
    )
]
df_plot["bias"] = df_plot["estimate"] - 1.0

fig, ax = plt.subplots(figsize=(5, 3.5))
ax, _, _ = make_boxplot(
    df_plot=df_plot,
    yaxis_var="bias",
    hue_var="group_var",
    hue_order=[
        "normalclamppolymix1_mlpnormen",
        "normalclamppolymix2_mlpnormen",
        "normalclamppolymix3_mlpnormen",
        "invmlpmix_mlpnormenc_mlprelude",
    ],
    new_hue_labels=["Poly 1", "Poly 2", "Poly 3", "Inv. MLP"],
    xaxis_var="instrument",
    xaxis_order=[
        "hW",
        "hWchV",
        "Z",
        "tslsCondPop",
        # "limlCondPop",
        "MREgger",
        "MREggerCondPop",
    ],
    new_x_labels=[
        r"",
        r"",
        r"2SLS($Z$)",
        r"PO($K$)-2SLS($Z$)",
        # r"PO($K$)-LIML($Z$)",
        r"Egger",
        r"PO($K$)-Egger",
    ],
    box_width=0.7,
    box_gap=0.2,
    hline_val=0.0,
    hline_label=None,
    ax=ax,
    legend_title="Mixing function",
    legend_by_row=True,
    legend_ncol=2,
)
ax.set_xlabel("Estimator")
ax.set_ylabel("Bias")
ax.tick_params(axis="x", rotation=30)
set_multicolor_xlabel(
    ax,
    x_idx=0,
    line1=r"2SLS($\widehat{W}$)",
    line2="(ours)",
    line2_color="#d62728",  # 'Tab:Red'
    rotation=30,
    dx=20,
    dy=0,
)
set_multicolor_xlabel(
    ax,
    x_idx=1,
    line1=r"PO($\widehat{V}$)-2SLS($\widehat{W}$)",
    line2="(ours)",
    line2_color="#d62728",  # 'Tab:Red'
    rotation=30,
    dx=30,
    dy=0,
)
fig.tight_layout()
plt.savefig("results/writeup_simu_fig1.png", dpi=300, bbox_inches="tight")

# %%
# Mis-specified latent dimensions
exp_ids = [
    f"new_normalclamppolymix3_mlpnormenc_inv_polyind_hw{ii}_ms100"
    for ii in range(1, 5)
]
ckpt_strategy = "last"
metric_key = "val/inv_loss"
exclude_sim_ids = [0, 4, 8]

df_all = gather_results(
    results_dir="results",
    exp_grp="new_normalclamppolymix3_mlpnormenc_polyinv_polyind_hw",
    exp_ids=exp_ids,
    ckpt_strategy=ckpt_strategy,
    metric_key=metric_key,
    suffix="insample",
    exclude_sim_ids=exclude_sim_ids,
)

df_plot = df_all[df_all["pop_num"] == -1].copy()
df_plot = df_plot[df_plot["instrument"].isin(["hW", "hWchV"])]
df_plot["bias"] = df_plot["estimate"] - 1.0

fig, ax = plt.subplots(figsize=(5, 3))
ax, _, _ = make_boxplot(
    df_plot=df_plot,
    yaxis_var="bias",
    xaxis_var="group_var",
    xaxis_order=[1, 2, 3, 4],
    new_x_labels=[1, 2, 3, 4],
    hue_var="instrument",
    hue_order=[
        "hW",
        "hWchV",
    ],
    new_hue_labels=[
        r"2SLS($\widehat{W}$)",
        r"PO($\widehat{V}$)-2SLS($\widehat{W}$)",
    ],
    box_width=0.5,
    box_gap=0.2,
    hline_val=0.0,
    hline_label=None,
    ax=ax,
    legend_title="Estimator",
    legend_by_row=True,
    legend_ncol=2,
)
ax.set_xlabel(r"$\hat{p}$")
ax.set_ylabel("Bias")
fig.tight_layout()
plt.savefig("results/writeup_simu_fig2.png", dpi=300, bbox_inches="tight")

# %%
# With and without independence penalty

mixing_fns = [
    "normalclamppolymix1",
    "normalclamppolymix2",
    "normalclamppolymix3",
]
exp_grp = "new_mlpnormenc_inv_polyind_ms100"
exp_ids = [
    f"new_{mix_fn}_mlpnormenc_poly2inv_polyind_ms100" for mix_fn in mixing_fns
] + ["new_invmlpmix_mlpnormenc_mlpreludec_poly2inv_polyind_ms100"]
ckpt_strategy = "last"
metric_key = "val/inv_loss"

exclude_sim_ids = [1, 2, 3, 5, 6, 7, 9, 10, 11]
df_all_inv_only = gather_results(
    results_dir="results",
    exp_grp=exp_grp,
    exp_ids=exp_ids,
    ckpt_strategy=ckpt_strategy,
    metric_key=metric_key,
    suffix="insample",
    exclude_sim_ids=exclude_sim_ids,
)
df_all_inv_only["penalty"] = "inv_only"

exclude_sim_ids = [0, 4, 8]
df_all_inv_ind = gather_results(
    results_dir="results",
    exp_grp=exp_grp,
    exp_ids=exp_ids,
    ckpt_strategy=ckpt_strategy,
    metric_key=metric_key,
    suffix="insample",
    exclude_sim_ids=exclude_sim_ids,
)
df_all_inv_ind["penalty"] = "inv_ind"
df_all = pd.concat([df_all_inv_only, df_all_inv_ind], axis=0)

df_plot = df_all[df_all["pop_num"] == -1].copy()
df_plot["bias"] = df_plot["estimate"] - 1.0

df_plot1 = df_plot[df_plot["penalty"] == "inv_ind"].copy().reset_index()
df_plot2 = df_plot[df_plot["penalty"] == "inv_only"].copy().reset_index()
df_plots = [df_plot1, df_plot2]

fig, axs = plt.subplots(2, 1, figsize=(5, 3), sharex=True)
shared_handles = None
shared_labels = None
for ii in range(2):
    axs[ii], h, l = make_boxplot(
        df_plot=df_plots[ii],
        yaxis_var="bias",
        xaxis_var="group_var",
        xaxis_order=[
            "normalclamppolymix1_mlpnormen",
            "normalclamppolymix2_mlpnormen",
            "normalclamppolymix3_mlpnormen",
            "invmlpmix_mlpnormenc_mlprelude",
        ],
        new_x_labels=["Poly 1", "Poly 2", "Poly 3", "Inv. MLP"],
        hue_var="instrument",
        hue_order=["hW", "hWchV"],
        new_hue_labels=[
            r"2SLS($\widehat{W}$)",
            r"PO($\widehat{V}$)-2SLS($\widehat{W}$)",
        ],
        box_width=0.7,
        box_gap=0.2,
        xlabel_visible=(ii == 1),
        hline_val=0.0,
        hline_label=None,
        ax=axs[ii],
        legend_title="Penalty",
        legend_by_row=True,
        legend_ncol=2,
        draw_legend=False,
    )
    if ii == 0:
        shared_handles = h
        shared_labels = l

title_handle = Line2D([], [], color="none", marker="none", linestyle="none")

# Prepend a "title" handle so the legend has a section header
final_handles = [title_handle] + shared_handles
final_labels = ["Estimator"] + shared_labels
fig.legend(
    handles=final_handles,
    labels=final_labels,
    loc="lower center",  # Anchor point of the legend box
    bbox_to_anchor=(0.5, 0.92),  # Position: (X=Center, Y=Top of figure)
    ncol=3,
    title=None,  # Title for the shared legend
    frameon=False,
)
axs[1].set_xlabel(r"Mixing function")
axs[0].set_ylabel(r"w/ ind." + "\n" + "Bias")
axs[1].set_ylabel(r"w/o ind." + "\n" + "Bias")
fig.tight_layout()
plt.savefig("results/writeup_simu_fig3.png", dpi=300, bbox_inches="tight")

# %%
# Three population example

mixing_fns = [
    "normalclamppolymix1",
    "normalclamppolymix2",
    "normalclamppolymix3",
]
exp_grp = "new_3pop_ms123"
exp_ids = [f"new2_3pop_{mix_fn}_ms123" for mix_fn in mixing_fns] + [
    "new2_3pop_invmlpmix_mlpnormenc_mlpreludec_poly2inv_polyind_ms123"
]
ckpt_strategy = "last"
metric_key = "val/inv_loss"
exclude_sim_ids = [0, 4, 8]

df_all = gather_results(
    results_dir="results",
    exp_grp=exp_grp,
    exp_ids=exp_ids,
    ckpt_strategy=ckpt_strategy,
    metric_key=metric_key,
    suffix="insample",
    exclude_sim_ids=exclude_sim_ids,
)

df_plot = df_all[df_all["pop_num"] == -1].copy()
df_plot = df_plot[
    df_plot["instrument"].isin(
        [
            "Z",
            "hW",
            "hWchV",
            "tslsCondPop",
            "limlCondPop",
            "MREgger",
            "MREggerCondPop",
        ]
    )
]
df_plot["bias"] = df_plot["estimate"] - 1.0

fig, ax = plt.subplots(figsize=(5, 3.5))
ax, _, _ = make_boxplot(
    df_plot=df_plot,
    yaxis_var="bias",
    hue_var="group_var",
    hue_order=[
        "normalclamppolymix1",
        "normalclamppolymix2",
        "normalclamppolymix3",
        "invmlpmix_mlpnormenc_mlpreludec_poly2inv_polyind",
    ],
    new_hue_labels=["Poly 1", "Poly 2", "Poly 3", "Inv. MLP"],
    xaxis_var="instrument",
    xaxis_order=[
        "hW",
        "hWchV",
        "Z",
        "tslsCondPop",
        # "limlCondPop",
        "MREgger",
        "MREggerCondPop",
    ],
    new_x_labels=[
        r"",
        r"",
        r"2SLS($Z$)",
        r"PO($K$)-2SLS($Z$)",
        # r"PO($K$)-LIML($Z$)",
        r"Egger",
        r"PO($K$)-Egger",
    ],
    box_width=0.7,
    box_gap=0.2,
    hline_val=0.0,
    hline_label=None,
    ax=ax,
    legend_title="Mixing function",
    legend_by_row=True,
    legend_ncol=2,
)
ax.set_xlabel("Estimator")
ax.set_ylabel("Bias")
ax.tick_params(axis="x", rotation=30)
set_multicolor_xlabel(
    ax,
    x_idx=0,
    line1=r"2SLS($\widehat{W}$)",
    line2="(ours)",
    line2_color="#d62728",  # 'Tab:Red'
    rotation=30,
    dx=20,
    dy=0,
)
set_multicolor_xlabel(
    ax,
    x_idx=1,
    line1=r"PO($\widehat{V}$)-2SLS($\widehat{W}$)",
    line2="(ours)",
    line2_color="#d62728",  # 'Tab:Red'
    rotation=30,
    dx=30,
    dy=0,
)
fig.tight_layout()
plt.savefig("results/writeup_simu_fig4.png", dpi=300, bbox_inches="tight")

# %%
# Different invariance loss functions
inv_losses = ["meanvar", "poly2", "poly3"]
exp_grp = "new_mlpnormenc_inv_polyind_ms100"
exp_ids = [
    f"new_normalclamppolymix3_mlpnormenc_{inv_loss}inv_polyind_ms100"
    for inv_loss in inv_losses
] + [
    f"new_invmlpmix_mlpnormenc_mlpreludec_{inv_loss}inv_polyind_ms100"
    for inv_loss in inv_losses
]
ckpt_strategy = "last"
metric_key = "val/inv_loss"
exclude_sim_ids = [0, 4, 8]

df_all = gather_results(
    results_dir="results",
    exp_grp=exp_grp,
    exp_ids=exp_ids,
    ckpt_strategy=ckpt_strategy,
    metric_key=metric_key,
    suffix="insample",
    exclude_sim_ids=exclude_sim_ids,
)

df_plot = df_all[df_all["pop_num"] == -1].copy()
df_plot = df_plot[
    df_plot["instrument"].isin(
        [
            # "Z",
            "hW",
            "hWchV",
            # "tslsCondPop",
            # "limlCondPop",
            # "MREgger",
            # "MREggerCondPop",
        ]
    )
]
df_plot["bias"] = df_plot["estimate"] - 1.0

fig, ax = plt.subplots(figsize=(5, 3))
ax, _, _ = make_boxplot(
    df_plot=df_plot,
    yaxis_var="bias",
    hue_var="group_var",
    hue_order=[
        "normalclamppolymix3_mlpnormenc_meanvar",
        "normalclamppolymix3_mlpnormenc_poly2",
        "normalclamppolymix3_mlpnormenc_poly3",
        #    'invmlpmix_mlpnormenc_mlpreludec_meanvar',
        #    'invmlpmix_mlpnormenc_mlpreludec_poly2',
        #    'invmlpmix_mlpnormenc_mlpreludec_poly3'
    ],
    new_hue_labels=["Mean-Var", "Poly 2", "Poly 3"],
    xaxis_var="instrument",
    xaxis_order=[
        "hW",
        "hWchV",
        # "Z",
        # "tslsCondPop",
        # "limlCondPop",
        # "MREgger",
        # "MREggerCondPop",
    ],
    new_x_labels=[
        r"2SLS($\widehat{W}$)",
        r"PO($\widehat{V}$)-2SLS($\widehat{W}$)",
        # r"2SLS($Z$)",
        # r"PO($K$)-2SLS($Z$)",
        # # r"PO($K$)-LIML($Z$)",
        # r"Egger",
        # r"PO($K$)-Egger",
    ],
    box_width=0.7,
    box_gap=0.2,
    hline_val=0.0,
    hline_label=None,
    ax=ax,
    legend_title="Invariance loss",
    legend_by_row=True,
    legend_ncol=3,
)
ax.set_xlabel("Estimator")
ax.set_ylabel("Bias")
ax.tick_params(axis="x", rotation=0)
fig.tight_layout()
plt.savefig("results/writeup_simu_fig5.png", dpi=300, bbox_inches="tight")

# %%
# Effect of relatedness loss weight (lam3) — replaces Table 4 in rebuttal_tables.md.

lam3_values = ["0", "0.001", "0.01", "0.1"]
ckpt_strategy = "last"
metric_key = "val/inv_loss"
exclude_sim_ids = [0, 4, 8]
suffix = "insample_no" + "".join(map(str, exclude_sim_ids))
metric_key_clean = metric_key.replace("/", "_")

# lam3=0 baseline lives under a different exp_grp/exp_id (polyind was the old name for poly2ind).
lam3_files = {
    "0":     ("new_mlpnormenc_inv_polyind_ms100",
              "new_normalclamppolymix3_mlpnormenc_poly2inv_polyind_ms100"),
    "0.001": ("new_normalclamppolymix3_mlpnormenc_polyinv_polyind_hw",
              "new_normalclamppolymix3_mlpnormenc_poly2inv_poly2ind_0.001lam3_ms100"),
    "0.01":  ("new_normalclamppolymix3_mlpnormenc_polyinv_polyind_hw",
              "new_normalclamppolymix3_mlpnormenc_poly2inv_poly2ind_0.01lam3_ms100"),
    "0.1":   ("new_normalclamppolymix3_mlpnormenc_polyinv_polyind_hw",
              "new_normalclamppolymix3_mlpnormenc_poly2inv_poly2ind_0.1lam3_ms100"),
}

lam3_dfs = []
for v, (exp_grp, exp_id) in lam3_files.items():
    fp = os.path.join("results", exp_grp,
                      f"summary_{exp_id}_{ckpt_strategy}_{metric_key_clean}_{suffix}.csv")
    d = pd.read_csv(fp)
    d["group_var"] = v
    lam3_dfs.append(d)
df_all = pd.concat(lam3_dfs, ignore_index=True)

df_plot = df_all[df_all["pop_num"] == -1].copy()
df_plot = df_plot[df_plot["instrument"].isin(["hW", "hWchV"])]
df_plot["bias"] = df_plot["estimate"] - 1.0

fig, ax = plt.subplots(figsize=(5, 3))
ax, handles, labels = make_boxplot(
    df_plot=df_plot,
    yaxis_var="bias",
    xaxis_var="group_var",
    xaxis_order=lam3_values,
    new_x_labels=[r"$\lambda_3=0$", r"$\lambda_3=0.001$", r"$\lambda_3=0.01$", r"$\lambda_3=0.1$"],
    hue_var="instrument",
    hue_order=["hW", "hWchV"],
    new_hue_labels=[r"2SLS($\widehat{W}$)", r"PO($\widehat{V}$)-2SLS($\widehat{W}$)"],
    box_width=0.7,
    box_gap=0.2,
    hline_val=0.0,
    hline_label=None,
    ax=ax,
    legend_title="Estimator",
    legend_by_row=True,
    legend_ncol=2,
)
title_handle = Line2D([], [], color="none", marker="none", linestyle="none")
ax.legend(
    handles=[title_handle] + handles,
    labels=["Estimator"] + labels,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=3,
    title=None,
    frameon=False,
)
ax.set_xlabel("Relatedness weight")
ax.set_ylabel("Bias")
ax.tick_params(axis="x", rotation=0)
fig.tight_layout()
plt.savefig("results/writeup_simu_fig6.png", dpi=300, bbox_inches="tight")

# %%
