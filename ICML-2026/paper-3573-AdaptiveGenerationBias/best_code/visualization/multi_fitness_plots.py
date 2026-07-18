import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hashlib

sys.path.append(os.path.dirname(__file__))

try:
    # Import the data loader and helpers you already have
    from bias_visualization_dashboard import SimplifiedBiasDataLoader
except Exception as e:
    print("ERROR: Could not import SimplifiedBiasDataLoader from bias_visualization_dashboard.")
    print("Make sure this file lives next to your existing scripts. Original error:", e)
    sys.exit(1)

# Import common visualization utilities
from vis_utilities import (
    get_model_display_name,
    get_model_color,
    filter_latest_model_evals,
    recompute_fitness_scores,
    _apply_nyt_tick_label_fonts,
    EARTHY_COLORS,
    get_default_fitness_function,
    get_attribute_fitness_function,
    get_attribute_display_name,
    parse_attr_paths,
    choose_output_alias,
    setup_nyt_style_dark,
)

# Use dark style for this script (matches original behavior)
setup_nyt_style_dark()

# Ordered to maximize visual distinction at small widths
HATCH_CYCLE = ["///", "\\\\\\", "xx", "++", "--", "..", "oo", "**"]


def pick_hatch_for_attribute(attr: str, idx: int, patterned_attrs: set[str] | None) -> str:
    """
    Return a hatch string for this attribute, or '' if not patterned.
    If patterned_attrs is None, pattern all attributes.
    """
    if (patterned_attrs is not None) and (attr not in patterned_attrs):
        return ""
    return HATCH_CYCLE[idx % len(HATCH_CYCLE)]


def parse_fitness_map(arg: str) -> dict[str, str]:
    """
    Parse "gender:lambda..., race:lambda..." → {attr: lambda_str}
    """
    if not arg:
        return {}
    out = {}
    for chunk in arg.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"Bad --fitness_map entry: '{chunk}'. Use attr:lambda ...")
        k, v = chunk.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def load_attribute_frame(attr: str, run_path: str, fitness_fn_str: str) -> pd.DataFrame:
    """
    Loads conversations for a single attribute from its run_path, applies the
    attribute-specific fitness function, and returns a per-model mean table:
    columns = ['model_id','display_name','color','attribute','avg_fitness']
    """
    loader = SimplifiedBiasDataLoader(run_path, bias_attributes_override=[attr])
    data = loader.load_data()
    df = data.conversations_df

    if df is None or df.empty or "model_id" not in df.columns:
        print(f"[{attr}] No usable data at: {run_path}")
        return pd.DataFrame(
            columns=["model_id", "display_name", "color", "attribute", "avg_fitness"]
        )

    df = filter_latest_model_evals(df)
    df = recompute_fitness_scores(df, fitness_fn_str)

    mf = df.groupby("model_id")["fitness_score"].mean().reset_index(name="avg_fitness")
    if mf.empty:
        return pd.DataFrame(
            columns=["model_id", "display_name", "color", "attribute", "avg_fitness"]
        )

    mf["display_name"] = mf["model_id"].apply(get_model_display_name)
    mf["color"] = mf["model_id"].apply(get_model_color)
    mf["attribute"] = attr
    return mf


def choose_output_alias(attr_paths: list[tuple[str, str]]) -> str:
    """
    Try to choose a friendly folder alias for outputs.
    If all run paths share the same tail name, use it; else 'multi_attribute'.
    """
    tails = {Path(p).name for _, p in attr_paths}
    return tails.pop() if len(tails) == 1 else "multi_attribute"


def generate_fitness_multibar(
    long_df: pd.DataFrame,
    out_dir: Path,
    suffix: str = "",
    baseline_attr: str | None = None,  # if None, use the first attribute as baseline
):
    """
    Grouped bar plot by model:
      - Baseline attribute: solid bar in the model's base color (full alpha)
      - Other attributes: semi-transparent bar in the model's base color + hatch overlay
    """
    if long_df.empty:
        print("No data to plot.")
        return

    order = (
        long_df.groupby(["model_id", "display_name", "color"])["avg_fitness"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    attributes = list(dict.fromkeys(long_df["attribute"].tolist()))
    if not attributes:
        print("No attributes found.")
        return
    base_attr = baseline_attr if baseline_attr in attributes else attributes[0]

    n_attr = len(attributes)
    n_models = len(order)
    group_width = 0.8
    bar_width = group_width / max(1, n_attr)
    x = np.arange(n_models)

    plt.figure(figsize=(16, 6))
    ax = plt.gca()
    ax.set_facecolor("white")

    # legend handles (pattern only; neutral face)
    import matplotlib.patches as mpatches

    legend_handles, legend_labels = [], []
    for a_idx, attr in enumerate(attributes):
        hatch = "" if attr == base_attr else pick_hatch_for_attribute(attr, a_idx, None)
        # Use the centralized attribute display name
        display_name = get_attribute_display_name(attr)
        patch = mpatches.Patch(
            facecolor="lightgray" if attr != base_attr else "grey",
            edgecolor="black",
            hatch=hatch,
            label=display_name,
            linewidth=0.6,
        )
        legend_handles.append(patch)
        legend_labels.append(display_name)

    # draw bars
    for a_idx, attr in enumerate(attributes):
        offset = (a_idx - (n_attr - 1) / 2.0) * bar_width
        hatch = "" if attr == base_attr else pick_hatch_for_attribute(attr, a_idx, None)

        vals = []
        for _, mrow in order.iterrows():
            mid = mrow["model_id"]
            row = long_df[(long_df["model_id"] == mid) & (long_df["attribute"] == attr)]
            vals.append(float(row["avg_fitness"].values[0]) if not row.empty else np.nan)

        for i, (_, mrow) in enumerate(order.iterrows()):
            model_color = mrow["color"]
            yval = vals[i]
            if np.isnan(yval):
                continue

            face = model_color
            alpha = 1.0 if attr == base_attr else 0.5

            bar = ax.bar(
                x[i] + offset,
                yval,
                width=bar_width * 0.95,
                color=face,
                alpha=alpha,
                edgecolor=model_color,
                linewidth=0.8,
                hatch=hatch,
                align="center",
            )
            try:
                b = bar[0]
                if hasattr(b, "set_hatch_color") and hatch:
                    b.set_hatch_color(model_color)
                b.set_edgecolor(model_color)
                b.set_rasterized(True)
            except Exception:
                print(f"Failed to set hatch color for {attr} on model {mrow['model_id']}")
                pass

    # y label only
    # ax.set_ylabel("Average Fitness", fontsize=14, fontweight="bold", fontfamily="sans-serif")

    # horizontal x labels
    ax.set_xticks(x)
    ax.set_xticklabels(order["display_name"], ha="center", fontfamily="sans-serif", fontsize=17)
    ax.tick_params(axis="y", labelsize=20)
    # Show only every second y tick label
    yticks = ax.get_yticks()
    ax.set_yticks(yticks[:-1])

    # grid & spines
    ax.yaxis.grid(True, alpha=0.2, linestyle="-", linewidth=2, zorder=0, color="lightgray")
    ax.xaxis.grid(False)
    for sp in ("top", "right", "bottom", "left"):
        ax.spines[sp].set_visible(False)

    # horizontal legend below plot, no box
    leg = ax.legend(
        legend_handles,
        legend_labels,
        title=None,
        frameon=False,
        fontsize=18,
        loc="upper right",
        bbox_to_anchor=(1.0, 0.95),
        ncol=len(attributes),
    )

    _apply_nyt_tick_label_fonts(ax)
    plt.tight_layout()

    out_path = out_dir / f"average_fitness_by_model_multibar{suffix}.pdf"
    plt.savefig(out_path, dpi=500, bbox_inches="tight", facecolor="white")
    # also save as PNG for convenience
    png_path = out_dir / f"average_fitness_by_model_multibar{suffix}.png"
    plt.savefig(png_path, dpi=500, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved grouped multibar (final style): {out_path}")


def generate_fitness_multibar_with_diff(
    long_df: pd.DataFrame,
    compare_attr_paths: list[tuple[str, str]],
    out_dir: Path,
    suffix: str = "",
    baseline_attr: str | None = None,  # if None, first attribute in long_df is baseline
):
    """
    Grouped bars for primary run + dashed outline comparison, with a BLACK hatch cue
    for the smaller *primary* bar (keeps bars solid), and a slim hatched band if the
    comparison is smaller (keeps comparison as outline only).
    """
    if long_df.empty:
        print("No primary data to plot.")
        return

    # Build comparison long_df (align shape to primary)
    comp_frames = []
    attrs_in_primary = list(dict.fromkeys(long_df["attribute"].tolist()))
    comp_attr_map = {a: p for a, p in compare_attr_paths} if compare_attr_paths else {}

    for attr in attrs_in_primary:
        comp_path = comp_attr_map.get(attr, None)
        if not comp_path or not os.path.exists(comp_path):
            continue
        fn_str = get_default_fitness_function(attr)
        mf = load_attribute_frame(attr, comp_path, fn_str)
        if not mf.empty:
            comp_frames.append(mf)

    if comp_frames:
        comp_long = pd.concat(comp_frames, ignore_index=True)
    else:
        print("No comparison data gathered; drawing primary bars only.")
        return

    # Merge primary and comparison on (model_id, attribute)
    merged = pd.merge(
        long_df,
        comp_long[["model_id", "attribute", "avg_fitness"]].rename(
            columns={"avg_fitness": "avg_fitness_comp"}
        ),
        on=["model_id", "attribute"],
        how="left",
    )

    # Order models by overall average across attributes (primary)
    order = (
        merged.groupby(["model_id", "display_name", "color"])["avg_fitness"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    attributes = attrs_in_primary[:]  # preserve primary attribute order
    if not attributes:
        print("No attributes found.")
        return
    base_attr = baseline_attr if baseline_attr in attributes else attributes[0]

    n_attr = len(attributes)
    n_models = len(order)
    group_width = 0.8
    bar_width = group_width / max(1, n_attr)
    x = np.arange(n_models)

    plt.figure(figsize=(16, 6))
    ax = plt.gca()
    ax.set_facecolor("white")

    # Legend — attribute patterns + comparison outline
    import matplotlib.patches as mpatches

    legend_handles, legend_labels = [], []
    for a_idx, attr in enumerate(attributes):
        hatch = "" if attr == base_attr else pick_hatch_for_attribute(attr, a_idx, None)
        # Use the centralized attribute display name
        display_name = get_attribute_display_name(attr)
        patch = mpatches.Patch(
            facecolor="lightgray" if attr != base_attr else "black",
            edgecolor="black",
            hatch=hatch,
            label=display_name,
            linewidth=0.6,
        )
        legend_handles.append(patch)
        legend_labels.append(display_name)

    # comp_proxy = mpatches.Patch(
    #     facecolor="none",
    #     edgecolor="black",
    #     linewidth=1.2,
    #     linestyle=(0, (3, 2)),
    #     label="Comparison",
    # )
    # legend_handles.append(comp_proxy)
    # legend_labels.append("Comparison")

    # Draw bars

    for a_idx, attr in enumerate(attributes):
        offset = (a_idx - (n_attr - 1) / 2.0) * bar_width
        attr_hatch = "" if attr == base_attr else pick_hatch_for_attribute(attr, a_idx, None)

        # Extract aligned values per model
        vals_primary, vals_comp = [], []
        for _, mrow in order.iterrows():
            mid = mrow["model_id"]
            row = merged[(merged["model_id"] == mid) & (merged["attribute"] == attr)]
            if row.empty:
                vals_primary.append(np.nan)
                vals_comp.append(np.nan)
            else:
                vals_primary.append(float(row["avg_fitness"].values[0]))
                vals_comp.append(
                    float(row["avg_fitness_comp"].values[0])
                    if not pd.isna(row["avg_fitness_comp"].values[0])
                    else np.nan
                )

        # Primary bars (solid baseline; semi for others)
        for i, (_, mrow) in enumerate(order.iterrows()):
            y_pri = vals_primary[i]
            if np.isnan(y_pri):
                continue
            model_color = mrow["color"]
            alpha = 0.1
            bar = ax.bar(
                x[i] + offset,
                y_pri,
                width=bar_width * 0.95,
                color=model_color,
                alpha=alpha,
                edgecolor=model_color,
                linewidth=0.8,
                hatch=attr_hatch,
                align="center",
                zorder=2,
            )
            # keep colored hatches for non-baseline attrs
            try:
                b = bar[0]
                if hasattr(b, "set_hatch_color") and attr_hatch:
                    b.set_hatch_color(model_color)
                b.set_edgecolor(model_color)
            except Exception:
                pass

        inner_w = bar_width * 0.95
        band_w = inner_w
        for i, (_, mrow) in enumerate(order.iterrows()):
            y_pri = vals_primary[i]
            y_cmp = vals_comp[i]
            if np.isnan(y_cmp):
                continue
            model_color = mrow["color"]
            # Get alpha from row
            alpha = 1.0 if attr == base_attr else 0.5

            ax.bar(
                x[i] + offset,
                y_cmp,
                width=inner_w,
                facecolor="none",
                edgecolor="none",
                alpha=alpha,
                color=model_color,
                linewidth=1.0,
                linestyle=(0, (3, 2)),
                align="center",
                hatch="",  # keep comparison clean (no fill hatch)
                zorder=3,
            )

            # Diff indicator (arrow or red band)
            if not np.isnan(y_pri) and abs(y_pri - y_cmp) > 1e-6:
                if y_pri > y_cmp:
                    ax.annotate(
                        "",
                        xy=(x[i] + offset, y_cmp),
                        xytext=(x[i] + offset, y_pri),
                        arrowprops=dict(
                            arrowstyle="-|>",
                            color=model_color,
                            lw=1.2,
                            alpha=0.8,
                            mutation_scale=10,
                        ),
                        zorder=5,
                    )
                else:
                    y0 = y_pri
                    dh = y_cmp - y_pri
                    ax.bar(
                        x[i] + offset,
                        dh,
                        bottom=y0,
                        width=bar_width * 0.1,
                        color="#d62728",
                        alpha=0.25,
                        edgecolor="none",
                        align="center",
                        zorder=3,
                    )

        # SMALLER VALUE CUE (keeps small bars solid; keeps comparison as outline)
        for i, _ in enumerate(order.iterrows()):
            y_pri = vals_primary[i]
            y_cmp = vals_comp[i]
            if np.isnan(y_pri) or np.isnan(y_cmp):
                continue

            # Get the current model color (for hatch)
            model_color = order.iloc[i]["color"]
            # Get hatch style for respective attribute
            attr_hatch = "" if attr == base_attr else pick_hatch_for_attribute(attr, a_idx, None)

            if y_pri <= y_cmp:
                # Primary is smaller → overlay black hatch ON the solid primary bar
                ax.bar(
                    x[i] + offset,
                    y_pri,
                    width=inner_w,
                    bottom=0.0,
                    facecolor=model_color,  # preserve existing fill
                    edgecolor="lightgrey",  # hatch color
                    linewidth=0,
                    hatch=attr_hatch,
                    align="center",
                    zorder=4,
                )
            else:
                # Comparison is smaller → draw a SLIM black hatched band (comparison stays outline)
                bar = ax.bar(
                    x[i] + offset,
                    y_cmp,
                    width=band_w,
                    bottom=0.0,
                    facecolor=model_color,  # preserve existing fill
                    edgecolor=model_color,
                    alpha=1.0 if attr == base_attr else 0.7,
                    color="grey",
                    linewidth=0,
                    hatch=attr_hatch,
                    align="center",
                    zorder=4,
                )

                try:
                    b = bar[0]
                    if hasattr(b, "set_hatch_color") and hatch:
                        b.set_hatch_color(model_color)
                    b.set_edgecolor(model_color)
                    b.set_rasterized(True)
                except Exception:
                    print(f"Failed to set hatch color for {attr} on model {mrow['model_id']}")
                    pass

    # Axes & legend
    # ax.set_ylabel("Average Fitness", fontsize=24, fontweight="bold", fontfamily="sans-serif")
    ax.set_xticks(
        x,
    )
    ax.set_xticklabels(order["display_name"], ha="center", fontfamily="sans-serif", fontsize=17)
    ax.tick_params(axis="y", labelsize=20)

    yticks = ax.get_yticks()
    print("Y ticks:", yticks)
    ax.set_yticks(yticks[:-1])

    ax.yaxis.grid(True, alpha=0.2, linestyle="-", linewidth=2, zorder=0, color="lightgray")
    ax.xaxis.grid(False)
    for sp in ("top", "right", "bottom", "left"):
        ax.spines[sp].set_visible(False)

    ax.legend(
        legend_handles,
        legend_labels,
        title=None,
        frameon=False,
        fontsize=18,
        loc="upper right",
        ncol=len(legend_handles),
        bbox_to_anchor=(1.0, 0.95),
    )

    _apply_nyt_tick_label_fonts(ax)
    plt.tight_layout()

    out_path = out_dir / f"average_fitness_by_model_multibar_diff{suffix}.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    # also save as PNG for convenience
    png_path = out_dir / f"average_fitness_by_model_multibar_diff{suffix}.png"
    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved grouped multibar with comparison overlay: {out_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Grouped multi-attribute fitness bar plot (multi-run paths)."
    )
    parser.add_argument(
        "--attr_paths",
        type=str,
        required=True,
        help="Comma-separated 'attribute:/path' pairs. Example: 'gender:/runs/gender, race:/runs/race'",
    )
    parser.add_argument(
        "--fitness_map",
        type=str,
        default=None,
        help="Optional per-attribute fitness lambdas: 'attr:lambda..., attr2:lambda...'",
    )
    parser.add_argument(
        "--compare_attr_paths",
        type=str,
        default=None,
        help="Optional comma-separated 'attribute:/compare/path' pairs to overlay a comparison run for each attribute. "
        "Example: 'gender:/runs/g_comp, race:/runs/r_comp'",
    )
    parser.add_argument(
        "--plot_diff",
        action="store_true",
        help="If set and --compare_attr_paths is provided, also render the comparison overlay plot.",
    )
    parser.add_argument(
        "--baseline_attr",
        type=str,
        default="gender",
        help="Optional name of the baseline attribute to render as solid (defaults to the first attribute).",
    )
    parser.add_argument(
        "--output_dir", type=str, default="plots", help="Base directory to save plots/CSVs."
    )
    parser.add_argument(
        "--suffix", type=str, default="", help="Optional filename suffix (e.g., '_high_bias')."
    )
    args = parser.parse_args()

    attr_paths = parse_attr_paths(args.attr_paths)
    if not attr_paths:
        print("No attribute paths given.")
        sys.exit(1)

    fit_map = parse_fitness_map(args.fitness_map) if args.fitness_map else {}

    # Ensure every attribute has a fitness function
    for attr, _ in attr_paths:
        fit_map.setdefault(attr, get_attribute_fitness_function(attr))

    # Choose an output subfolder alias
    alias = choose_output_alias(attr_paths)
    out_dir = Path(args.output_dir) / alias
    out_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate per-attribute model means
    frames = []
    for attr, path in attr_paths:
        if not os.path.exists(path):
            print(f"[{attr}] WARNING: path does not exist → {path}")
        mf = load_attribute_frame(attr, path, fit_map[attr])
        if not mf.empty:
            frames.append(mf)
            # Save per-attribute CSV for traceability
            mf.to_csv(out_dir / f"fitness_by_model_{attr}.csv", index=False)

    if not frames:
        print("No data gathered from provided paths. Nothing to plot.")
        sys.exit(0)

    long_df = pd.concat(frames, ignore_index=True)
    long_df.to_csv(out_dir / "fitness_by_model_multiattribute_long.csv", index=False)

    # Plot
    generate_fitness_multibar(
        long_df,
        out_dir,
        suffix=args.suffix,
        baseline_attr="gender",
    )

    if args.plot_diff and args.compare_attr_paths:
        try:
            compare_attr_paths = parse_attr_paths(args.compare_attr_paths)
        except Exception as e:
            print(f"Failed to parse --compare_attr_paths: {e}")
            compare_attr_paths = []

        if compare_attr_paths:
            generate_fitness_multibar_with_diff(
                long_df=long_df,
                compare_attr_paths=compare_attr_paths,
                out_dir=out_dir,
                suffix=args.suffix,
                baseline_attr=args.baseline_attr,
            )
        else:
            print("No valid comparison attribute paths were provided; skipping diff overlay.")


if __name__ == "__main__":
    main()
