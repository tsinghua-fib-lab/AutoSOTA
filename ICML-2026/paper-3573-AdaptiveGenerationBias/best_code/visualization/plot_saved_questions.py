from __future__ import annotations
import argparse
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from collections import defaultdict

# Try to use your NYT-style util module; fall back gracefully.
try:
    from vis_utilities import (
        setup_nyt_style,
        apply_nyt_style_to_axes,
        get_attribute_color,
        get_attribute_display_name,
    )

    HAVE_VIS_UTILS = True
except Exception:
    HAVE_VIS_UTILS = False

    def setup_nyt_style():
        plt.rcParams.update(
            {
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "axes.edgecolor": "lightgray",
                "axes.linewidth": 0.8,
                "axes.grid": True,
                "grid.alpha": 0.2,
                "grid.color": "lightgray",
                "grid.linewidth": 1.0,
                "axes.axisbelow": True,
                "font.size": 12,
                "axes.labelsize": 14,
                "axes.titlesize": 18,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                "legend.fontsize": 12,
                "figure.titlesize": 20,
            }
        )

    def apply_nyt_style_to_axes(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.yaxis.grid(True, alpha=0.2, linestyle="-", linewidth=2, color="lightgray")
        ax.xaxis.grid(False)

    def get_attribute_color(attr: str) -> str:
        return {"gender": "#A65D4E", "race": "#C49A6C", "religion": "#7A8450"}.get(attr, "#4E5D73")

    def get_attribute_display_name(attr: str) -> str:
        return {"gender": "Sex", "race": "Race", "religion": "Religion"}.get(attr, attr.title())


def _normalize_attr_label(label: str) -> str:
    k = label.strip().lower()
    if k in ("sex", "gender"):
        return "gender"
    if k == "race":
        return "race"
    if k == "religion":
        return "religion"
    return k


def _iter_iteration_dirs(root: Path):
    """Yield (idx, path) for iteration_* under root, sorted by idx."""
    items = []
    for p in root.iterdir():
        if p.is_dir() and p.name.startswith("iteration_"):
            try:
                idx = int(p.name.split("_", 1)[1])
                items.append((idx, p))
            except Exception:
                pass
    return sorted(items, key=lambda t: t[0])


def _count_jsonl_lines(path: Path) -> int:
    if not path.exists() or not path.is_file():
        return 0
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def scan_top_level_run(
    run_path: Path,
    sb_dir: str = "sb_2",
    filename: str = "saved_questions.jsonl",
):
    """
    Scan a run folder that may contain iteration_* directly OR model folders
    that themselves contain iteration_*.

    Returns: (iterations_sorted_list, cumulative_counts_list)
    """
    if not run_path.exists():
        raise FileNotFoundError(run_path)

    # First, check if iterations live directly under run_path.
    direct_iters = _iter_iteration_dirs(run_path)

    # Aggregate per-iteration counts across all sources (possibly many models).
    per_iter_counts = defaultdict(int)

    if direct_iters:
        # Old layout
        for idx, ipath in direct_iters:
            per_iter_counts[idx] += _count_jsonl_lines(ipath / sb_dir / filename)
    else:
        # New layout: one extra level (model folders)
        for model_dir in sorted([d for d in run_path.iterdir() if d.is_dir()]):
            iters = _iter_iteration_dirs(model_dir)
            for idx, ipath in iters:
                per_iter_counts[idx] += _count_jsonl_lines(ipath / sb_dir / filename)

    # Build cumulative curve
    iters_sorted = sorted(per_iter_counts.keys())
    cumulative = []
    total = 0
    for i in iters_sorted:
        total += per_iter_counts[i]
        cumulative.append(total)

    return iters_sorted, cumulative


def parse_runs(entries):
    """
    Parse repeated --run args of the form 'label:/path'.
    """
    out = []
    for e in entries:
        if ":" not in e:
            raise ValueError(f"Bad --run entry: {e} (use label:/path)")
        label, path = e.split(":", 1)
        label = label.strip()
        path = Path(path.strip())
        out.append((label, path))
    return out


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Plot cumulative saved questions by iteration.")
    ap.add_argument(
        "--run",
        action="append",
        required=True,
        help='Repeatable: "label:/path/to/top_level_run_folder"',
    )
    ap.add_argument("--out", default="saved_questions_cumulative.pdf")
    ap.add_argument("--sb-dir", default="sb_2")
    ap.add_argument("--file", default="saved_questions.jsonl")
    ap.add_argument("--title", default="Cumulative Saved Questions by Iteration")
    args = ap.parse_args()

    runs = parse_runs(args.run)

    setup_nyt_style()
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for label, folder in runs:
        try:
            iters, cum = scan_top_level_run(folder, sb_dir=args.sb_dir, filename=args.file)
        except FileNotFoundError:
            print(f"[WARN] Folder not found: {folder}", file=sys.stderr)
            continue

        if not iters:
            print(f"[WARN] No iterations found under: {folder}", file=sys.stderr)
            continue

        attr_key = _normalize_attr_label(label)
        color = get_attribute_color(attr_key)
        display = (
            get_attribute_display_name(attr_key)
            if attr_key in ("gender", "race", "religion")
            else label
        )

        ax.plot(iters, cum, marker="o", linewidth=2.5, label=display, color=color)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cumulative saved questions")
    ax.set_title(args.title, fontfamily="serif")
    apply_nyt_style_to_axes(ax)
    ax.legend(frameon=False, title="Attribute" if len(runs) > 1 else None)

    fig.tight_layout()
    fig.savefig(args.out, dpi=300)
    print(f"Saved plot → {args.out}")


if __name__ == "__main__":
    main()
