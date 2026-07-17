import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def main():
    # === 1. Configuration via Argparse ===
    parser = argparse.ArgumentParser(description="Plot Comprehensive Downstream Metrics (Acc, F1, Loss, Transfer Error)")
    parser.add_argument('--dataset', type=str, default='Cora', choices=['Cora', 'Pubmed', 'ogbn-arxiv'], 
                        help='Dataset used for the results (default: Cora)')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of GNN layers (default: 3)')
    parser.add_argument('--hidden_channels', type=int, default=64, help='Hidden channel size (default: 64)')
    args = parser.parse_args()

    # === 2. Read Results ===
    csv_path = f"{args.dataset}_Test_{args.num_layers}_{args.hidden_channels}_metrics.csv"

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Error: Result file '{csv_path}' not found. Please run the evaluation script first.")

    df = pd.read_csv(csv_path)

    # === 3. SCI Publication Style Configuration ===
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 16,             
        "axes.labelsize": 18,        
        "axes.titlesize": 18,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "lines.linewidth": 2,
        "lines.markersize": 6,
    })

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    markers = ["o", "s", "^"]
    linestyles = ["-", "--", "-."]
    labels = ["Scheme I", "Scheme II", "Scheme III"]

    # Map the metrics to column names, y-axis labels, and file suffixes
    metrics = [
        ("mean_diff", "std_diff", "Transferability Difference", "TransferError"),
        ("mean_loss", "std_loss", "Cross-Entropy Loss", "Loss"),
        ("mean_acc", "std_acc", "Accuracy", "Accuracy"),
        ("mean_f1", "std_f1", "Macro-F1 Score", "F1Score")
    ]

    # === 4. Plot Independent Figures ===
    for mean_col, std_col, ylabel, file_suffix in metrics:
        fig, ax = plt.subplots(figsize=(6, 4.5)) 

        for scheme_id, color, marker, ls in zip(sorted(df["scheme_id"].unique()), colors, markers, linestyles):
            subdf = df[df["scheme_id"] == scheme_id]
            
            # Plot main trend line with error bars
            ax.errorbar(
                x=subdf["sample_size"],
                y=subdf[mean_col],
                yerr=subdf[std_col],
                marker=marker,
                linestyle=ls,
                color=color,
                capsize=4,
                elinewidth=1.2,
                markeredgewidth=1,
                alpha=0.9,
                label=labels[scheme_id - 1]
            )

        # Axis and grid configuration
        ax.set_xlabel("Graph Size")
        ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False) 
        
        # Legend
        ax.legend(frameon=False, loc="best")

        plt.tight_layout()
        
        # Save independent metric plot
        output_filename = f"{args.dataset}_plot_{file_suffix}_{args.num_layers}_{args.hidden_channels}.pdf"
        plt.savefig(output_filename, bbox_inches="tight")
        print(f"Plot successfully saved to {output_filename}")
        
        plt.close(fig) 

if __name__ == "__main__":
    main()