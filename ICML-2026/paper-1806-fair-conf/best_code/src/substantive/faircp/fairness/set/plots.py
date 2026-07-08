import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict
from math import ceil

from substantive.faircp.structs.enums import ConformalMethod
from substantive.faircp.structs.fairness_input import FairnessInput


def plot_set_size_distribution(fairness_input: FairnessInput, filename: str):
    # Collect counts per method
    counts_by_method = defaultdict(Counter)
    for instance in fairness_input.instances:
        for method, preds in instance.predictions.items():
            if method == ConformalMethod.TOP_K:
                continue
            size = len(preds)
            counts_by_method[method][size] += 1

    # Get all possible sizes
    all_sizes = sorted(set(size for c in counts_by_method.values() for size in c))

    methods = list(counts_by_method.keys())
    x = range(len(all_sizes))
    width = 0.8 / len(methods)  # space out bars

    plt.figure(figsize=(16, 9))
    for i, method in enumerate(methods):
        sizes = [counts_by_method[method].get(s, 0) for s in all_sizes]
        plt.bar(
            [xx + i * width for xx in x], sizes, width=width, label=str(method.value)
        )

    plt.xticks([xx + width * (len(methods) - 1) / 2 for xx in x], all_sizes)
    plt.xlabel("Set size")
    plt.ylabel("Count")
    plt.title("Distribution of conformal set sizes")
    plt.legend(title="Conformal Method")
    plt.tight_layout()
    # plt.show()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_shape_heatmap(
    fairness_input: FairnessInput, method: ConformalMethod, filename: str
):
    # n_labels = len(fairness_input.label_map)
    # mat = np.zeros((n_labels, n_labels), dtype=int)

    # for instance in fairness_input.instances:
    #     true_label = instance.label
    #     preds = instance.predictions[method]
    #     for p in preds:
    #         mat[true_label, p] += 1

    all_label_indices = set()
    for instance in fairness_input.instances:
        all_label_indices.add(instance.label)
        all_label_indices.update(instance.predictions[method])
    
    sorted_indices = sorted(all_label_indices)
    n_labels = len(sorted_indices)
    
    idx_to_pos = {idx: pos for pos, idx in enumerate(sorted_indices)}
    
    mat = np.zeros((n_labels, n_labels), dtype=int)

    for instance in fairness_input.instances:
        true_label_pos = idx_to_pos[instance.label]
        preds = instance.predictions[method]
        for p in preds:
            pred_pos = idx_to_pos[p]
            mat[true_label_pos, pred_pos] += 1

    # Normalize rows to probabilities
    row_sums = mat.sum(axis=1, keepdims=True)
    heatmap = mat / np.maximum(row_sums, 1)

    plt.figure(figsize=(12, 10))
    plt.imshow(heatmap, cmap="Blues", aspect="auto")
    plt.colorbar(label="Inclusion probability")
    plt.xticks(
        range(n_labels),
        [fairness_input.label_map[i] for i in sorted_indices],
        rotation=45,
    )
    plt.yticks(range(n_labels), [fairness_input.label_map[i] for i in sorted_indices])
    plt.xlabel("Label in prediction set")
    plt.ylabel("True label")
    plt.title(f"Shape heatmap for {method}")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_set_size_by_group(fairness_input: FairnessInput, filename: str):
    # counts[method][group][set_size] = count
    counts = defaultdict(lambda: defaultdict(Counter))
    total_by_method_group = defaultdict(lambda: defaultdict(int))

    for inst in fairness_input.instances:
        g = inst.group
        for method, preds in inst.predictions.items():
            if method == ConformalMethod.TOP_K:
                continue
            size = len(preds)
            counts[method][g][size] += 1
            total_by_method_group[method][g] += 1

    groups = sorted(fairness_input.group_map.keys())
    methods = list(counts.keys())
    colors = plt.cm.Set2.colors  # distinct palette

    fig, axes = plt.subplots(
        len(methods), 1, figsize=(14, 7 * len(methods)), sharex=True
    )

    if len(methods) == 1:
        axes = [axes]  # handle single method case

    for ax, method in zip(axes, methods):
        # collect all possible sizes for this method
        all_sizes = sorted(set(size for g in counts[method].values() for size in g))
        x = range(len(all_sizes))
        width = 0.8 / len(groups)

        for i, g in enumerate(groups):
            values = [
                counts[method][g].get(s, 0) / total_by_method_group[method][g] * 100
                for s in all_sizes
            ]
            ax.bar(
                [xx + i * width for xx in x],
                values,
                width=width,
                label=fairness_input.group_map[g],
                color=colors[i % len(colors)],
            )

        ax.set_title(f"Conformal Method: {method}")
        ax.set_ylabel("Percentage (%)")
        ax.legend()

    axes[-1].set_xlabel("Set size")
    axes[-1].set_xticks([xx + width * (len(groups) - 1) / 2 for xx in x])
    axes[-1].set_xticklabels(all_sizes)
    plt.tight_layout()
    # plt.show()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_shape_heatmap_by_group(
    fairness_input: FairnessInput,
    method: ConformalMethod,
    filename: str,
    normalize="row",
    cmap="Blues",
    share_scale=True,
    figsize_per_group=(6, 6),
    hspace=0.4,
):
    all_label_indices = set()
    for inst in fairness_input.instances:
        all_label_indices.add(inst.label)
        all_label_indices.update(inst.predictions[method])
    
    sorted_indices = sorted(all_label_indices)
    n_labels = len(sorted_indices)
    
    idx_to_pos = {idx: pos for pos, idx in enumerate(sorted_indices)}
    
    groups = sorted(fairness_input.group_map.keys())

    # 1) Build raw count matrices per group
    counts_by_group = {g: np.zeros((n_labels, n_labels), dtype=float) for g in groups}
    for inst in fairness_input.instances:
        g = inst.group
        true_label_pos = idx_to_pos[inst.label]
        preds = inst.predictions[method]
        for p in preds:
            pred_pos = idx_to_pos[p]
            counts_by_group[g][true_label_pos, pred_pos] += 1

    # 2) Convert to display matrices according to normalization mode
    mats = {}
    if normalize == "none":
        mats = {g: counts_by_group[g] for g in groups}
    elif normalize == "global":
        global_max = max(mat.max() for mat in counts_by_group.values())
        global_max = max(global_max, 1.0)  # avoid div-by-zero
        mats = {g: counts_by_group[g] / global_max for g in groups}
    elif normalize == "row":
        for g in groups:
            mat = counts_by_group[g].astype(float)
            row_sums = mat.sum(axis=1, keepdims=True)  # shape (n_labels, 1)
            # replace zeros with 1 so rows that have no instances remain zeros after division
            safe_row_sums = np.where(row_sums == 0, 1.0, row_sums)
            mats[g] = mat / safe_row_sums
    else:
        raise ValueError("normalize must be one of {'none','global','row'}")

    # 3) Compute color scale (vmin/vmax)
    if share_scale:
        all_mins = [mat.min() for mat in mats.values()]
        all_maxs = [mat.max() for mat in mats.values()]
        vmin = float(np.min(all_mins))
        vmax = float(np.max(all_maxs))
        # if everything is zero, avoid equal vmin/vmax which errors in some matplotlib versions
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-6
    else:
        vmin = vmax = None

    # 4) Plot vertical subplots, one per group
    fig_h = figsize_per_group[1] * max(1, len(groups))
    fig, axes = plt.subplots(
        len(groups), 1, figsize=(figsize_per_group[0], fig_h), sharex=True, sharey=True
    )
    if len(groups) == 1:
        axes = [axes]

    # prepare tick labels
    #labels_names = [fairness_input.label_map[i] for i in range(n_labels)]
    for ax, g in zip(axes, groups):
        ax.imshow(mats[g], aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"Group: {fairness_input.group_map[g]}")
        ax.set_ylabel("True label")
        ax.set_yticks(range(n_labels))
        # ax.set_yticklabels(labels_names)
        #ax.set_yticklabels(range(len(labels_names)))
        ax.set_yticklabels([fairness_input.label_map[idx] for idx in sorted_indices])
        ax.set_xticks(range(n_labels))
        # ax.set_xticklabels(labels_names, rotation=45, ha="right")
        #ax.set_xticklabels(range(len(labels_names)), rotation=45, ha="right")
        ax.set_xticklabels([fairness_input.label_map[idx] for idx in sorted_indices], rotation=45, ha="right")

    axes[-1].set_xlabel("Label in prediction set")

    fig.subplots_adjust(hspace=hspace, right=0.85)
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_grouped_bar_by_label(
    fairness_input: FairnessInput,
    method: ConformalMethod,
    filename: str,
    figsize_per_subplot=(6, 4),
    max_cols=2,
    hspace=1,
    wspace=0.3,
):
    """
    Plot grouped bar plots for each true label, showing inclusion frequencies of predicted labels
    across groups, for a given conformal method.

    Args:
        fairness_input: FairnessInput object with instances, label_map, and group_map.
        method: ConformalMethod to plot.
        filename: Output file path for saving the plot.
        figsize_per_subplot: Tuple of (width, height) per subplot in inches.
        max_cols: Maximum number of columns for subplot grid (used if n_labels > 6).
        hspace: Vertical spacing between subplots.
        wspace: Horizontal spacing between subplots.
    """
    n_labels = len(fairness_input.label_map)
    groups = sorted(fairness_input.group_map.keys())
    n_groups = len(groups)

    # Compute inclusion counts per true label, predicted label, and group
    counts_by_label_group = {
        true_label: {g: np.zeros(n_labels, dtype=float) for g in groups}
        for true_label in range(n_labels)
    }
    instance_counts = {
        true_label: {g: 0 for g in groups} for true_label in range(n_labels)
    }
    for inst in fairness_input.instances:
        true_label = inst.label
        g = inst.group
        instance_counts[true_label][g] += 1
        preds = inst.predictions[method]
        for p in preds:
            counts_by_label_group[true_label][g][p] += 1

    # Convert counts to inclusion probabilities
    probs_by_label_group = {
        true_label: {
            g: counts_by_label_group[true_label][g]
            / max(instance_counts[true_label][g], 1)
            for g in groups
        }
        for true_label in range(n_labels)
    }

    # Determine layout: vertical stack for small n_labels, grid for large
    if n_labels <= 6:
        n_rows, n_cols = n_labels, 1
        fig_width = figsize_per_subplot[0]
        fig_height = figsize_per_subplot[1] * n_labels
    else:
        n_cols = min(max_cols, n_labels)
        n_rows = ceil(n_labels / n_cols)
        fig_width = figsize_per_subplot[0] * n_cols
        fig_height = figsize_per_subplot[1] * n_rows

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(fig_width, fig_height), sharex=False, sharey=False
    )
    axes = np.array(axes).flatten() if n_labels > 1 else [axes]

    # Color palette for groups
    colors = plt.cm.Set2.colors
    group_names = [fairness_input.group_map[g] for g in groups]

    # Plot each true label
    for idx, true_label in enumerate(range(n_labels)):
        ax = axes[idx]
        # Get predicted labels with non-zero counts for this true label
        total_counts = sum(counts_by_label_group[true_label][g] for g in groups)
        active_labels = [i for i in range(n_labels) if total_counts[i] > 0]
        if not active_labels:
            ax.text(0.5, 0.5, "No predictions", ha="center", va="center")
            ax.set_title(f"True Label: {fairness_input.label_map[true_label]}")
            ax.axis("off")
            continue

        # Prepare data for grouped bars
        x = np.arange(len(active_labels))
        bar_width = 0.8 / n_groups  # Adjust width based on number of groups
        for g_idx, g in enumerate(groups):
            probs = [probs_by_label_group[true_label][g][p] for p in active_labels]
            ax.bar(
                x + g_idx * bar_width - 0.4 + bar_width / 2,
                probs,
                bar_width,
                label=group_names[g_idx] if idx == 0 else None,
                color=colors[g_idx],
            )

        # Customize subplot
        ax.set_title(f"True Label: {fairness_input.label_map[true_label]}")
        # ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Inclusion Probability")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [fairness_input.label_map[p] for p in active_labels],
            rotation=45,
            ha="right",
        )

    # Hide unused subplots
    for idx in range(n_labels, len(axes)):
        axes[idx].axis("off")

    # Add shared legend
    if n_groups > 1:
        fig.legend(
            loc="center right",
            bbox_to_anchor=(1.0, 0.5),
            title="Groups",
            labels=group_names,
        )

    # Adjust layout
    fig.tight_layout()
    fig.subplots_adjust(hspace=hspace, wspace=wspace, right=0.85)
    plt.suptitle(f"Grouped Bar Plot for {method}", y=1.02)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
