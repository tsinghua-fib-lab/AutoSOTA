"""Generate publication-ready LaTeX tables from experiment CSV results."""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from dp_kfac.results import load_results_csv, aggregate_results


def format_mean_std(mean, std):
    """Format mean +/- std as LaTeX percentage."""
    if std is None or std != std:  # NaN check
        return f"${mean * 100:.1f}$"
    return f"${mean * 100:.1f} \\pm {std * 100:.1f}$"


def generate_main_table(results_dir, output_dir):
    """Generate the main results table across datasets and privacy budgets."""
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "MNIST / CNN": "cnn_mnist_results.csv",
        "CIFAR-100 / CrossViT": "crossvit_cifar100_results.csv",
        "IMDB / LogReg": "imdb_logreg_results.csv",
        "StackOverflow / RoBERTa": "stackoverflow_results.csv",
        "SST-2 / DistilBERT": "sst2_results.csv",
    }

    selected_epsilons = [1.0, 3.0, 8.0]

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Test accuracy (\\%) across datasets and privacy budgets. "
        "Mean $\\pm$ std over 5 seeds.}"
    )
    lines.append("\\label{tab:main_results}")
    lines.append("\\resizebox{\\textwidth}{!}{")

    eps_headers = " & ".join(
        [f"$\\varepsilon={e:.0f}$" for e in selected_epsilons]
    )
    lines.append(f"\\begin{{tabular}}{{ll{'c' * len(selected_epsilons)}}}")
    lines.append("\\toprule")
    lines.append(f"Dataset / Model & Method & {eps_headers} \\\\")
    lines.append("\\midrule")

    for ds_name, csv_name in datasets.items():
        csv_path = results_dir / csv_name
        if not csv_path.exists():
            continue

        df = load_results_csv(csv_path)
        df = df[df["Method"] != "Plain SGD"]
        methods = df["Method"].unique()

        for i, method in enumerate(methods):
            prefix = (
                f"\\multirow{{{len(methods)}}}{{*}}{{{ds_name}}}"
                if i == 0
                else ""
            )
            cells = []
            for eps in selected_epsilons:
                subset = df[
                    (df["Method"] == method)
                    & (abs(df["Epsilon"] - eps) < 0.1)
                ]
                if len(subset) > 0:
                    mean = subset["Accuracy"].mean()
                    std = subset["Accuracy"].std()
                    cells.append(format_mean_std(mean, std))
                else:
                    cells.append("--")

            row_cells = " & ".join(cells)
            lines.append(f"{prefix} & {method} & {row_cells} \\\\")

        lines.append("\\midrule")

    if lines[-1] == "\\midrule":
        lines[-1] = "\\bottomrule"

    lines.append("\\end{tabular}}")
    lines.append("\\end{table}")

    table_text = "\n".join(lines)

    output_path = output_dir / "main_results_table.tex"
    with open(output_path, "w") as f:
        f.write(table_text)

    print(f"Main results table saved to {output_path}")
    print()
    print(table_text)

    return table_text


def generate_ablation_table(results_dir, output_dir):
    """Generate an ablation table from the AdaDPS comparison CSV."""
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "adadps_comparison.csv"
    if not csv_path.exists():
        print(f"AdaDPS comparison CSV not found at {csv_path}, skipping ablation table.")
        return None

    df = load_results_csv(csv_path)
    agg = aggregate_results(df, group_by=["Method", "Epsilon"], metric="Accuracy")

    epsilons = sorted(agg["Epsilon"].unique())
    methods = agg["Method"].unique()

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Ablation: AdaDPS vs.\\ KFAC-based preconditioning "
        "(MNIST $\\to$ FashionMNIST). "
        "Mean $\\pm$ std over multiple seeds.}"
    )
    lines.append("\\label{tab:adadps_ablation}")

    if len(epsilons) > 1:
        eps_headers = " & ".join(
            [f"$\\varepsilon={e:.0f}$" for e in epsilons]
        )
        lines.append(f"\\begin{{tabular}}{{l{'c' * len(epsilons)}}}")
        lines.append("\\toprule")
        lines.append(f"Method & {eps_headers} \\\\")
        lines.append("\\midrule")

        for method in methods:
            cells = []
            for eps in epsilons:
                row = agg[
                    (agg["Method"] == method)
                    & (abs(agg["Epsilon"] - eps) < 0.1)
                ]
                if len(row) > 0:
                    mean = row["Accuracy_mean"].values[0]
                    std = row["Accuracy_std"].values[0]
                    cells.append(format_mean_std(mean, std))
                else:
                    cells.append("--")
            row_cells = " & ".join(cells)
            lines.append(f"{method} & {row_cells} \\\\")
    else:
        # Single epsilon -- show accuracy and loss columns
        eps_val = epsilons[0]
        lines.append("\\begin{tabular}{lcc}")
        lines.append("\\toprule")
        lines.append(
            f"Method & Accuracy (\\%) & Loss \\\\"
        )
        lines.append(
            f"\\multicolumn{{3}}{{c}}{{$\\varepsilon={eps_val:.0f}$}} \\\\"
        )
        lines.append("\\midrule")

        has_loss = "Loss" in df.columns
        for method in methods:
            subset = df[df["Method"] == method]
            acc_mean = subset["Accuracy"].mean()
            acc_std = subset["Accuracy"].std()
            acc_cell = format_mean_std(acc_mean, acc_std)
            if has_loss:
                loss_mean = subset["Loss"].mean()
                loss_std = subset["Loss"].std()
                if loss_std != loss_std:  # NaN
                    loss_cell = f"${loss_mean:.4f}$"
                else:
                    loss_cell = f"${loss_mean:.4f} \\pm {loss_std:.4f}$"
            else:
                loss_cell = "--"
            lines.append(f"{method} & {acc_cell} & {loss_cell} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    table_text = "\n".join(lines)

    output_path = output_dir / "adadps_ablation_table.tex"
    with open(output_path, "w") as f:
        f.write(table_text)

    print(f"Ablation table saved to {output_path}")
    print()
    print(table_text)

    return table_text


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate publication-ready LaTeX tables from experiment CSV results."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing experiment CSV files (default: results)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/tables",
        help="Directory to write .tex table files (default: results/tables)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_main_table(args.results_dir, args.output_dir)
    print()
    generate_ablation_table(args.results_dir, args.output_dir)
