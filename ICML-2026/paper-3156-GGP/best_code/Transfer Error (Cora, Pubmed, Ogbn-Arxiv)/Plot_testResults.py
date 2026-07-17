import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # === Configuration via Argparse ===
    parser = argparse.ArgumentParser(description="Plot GCN Transferability Results")
    parser.add_argument('--dataset', type=str, default='Cora', choices=['Cora', 'Pubmed', 'ogbn-arxiv'], 
                        help='Dataset used for the results (default: Cora)')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of GNN layers (default: 2)')
    parser.add_argument('--hidden_channels', type=int, default=32, help='Hidden channel size (default: 32)')
    args = parser.parse_args()

    # === Read Results ===
    save_path = f"{args.dataset}_Test_{args.num_layers}_{args.hidden_channels}.csv"
    
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Result file '{save_path}' not found. Please run the test script first.")
        
    df = pd.read_csv(save_path)

    # === SCI Publication Style Configuration ===
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 16,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "legend.fontsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "lines.linewidth": 2,
        "lines.markersize": 6,
    })

    # Standard SCI color palette (Blue, Orange, Green)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    markers = ["o", "s", "^"]
    linestyles = ["-", "--", "-."]

    # === Plotting ===
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Scheme I", "Scheme II", "Scheme III"]

    for scheme_id, color, marker, ls in zip(sorted(df["scheme_id"].unique()), colors, markers, linestyles):
        subdf = df[df["scheme_id"] == scheme_id]
        
        # Plot the main trend line
        ax.plot(
            subdf["sample_size"], subdf["mean_diff"],
            label=labels[scheme_id - 1],
            marker=marker, linestyle=ls, color=color
        )
        
        # Add error bars using standard deviation
        ax.errorbar(
            x=subdf["sample_size"],
            y=subdf["mean_diff"],
            yerr=subdf["std_diff"],
            marker=marker,
            linestyle=ls,
            color=color,
            capsize=4,        # Cap length for error bars
            elinewidth=1,     # Line width for error bars
            markeredgewidth=1 # Edge width for markers to ensure clarity
        )

    # === Detail Configuration ===
    ax.set_xlabel("Graph Size")
    ax.set_ylabel("Transferability Difference")

    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.grid(False)
    plt.tight_layout()
    
    # Save the figure dynamically
    out_file = f"{args.dataset}_plot_color_{args.num_layers}_{args.hidden_channels}.pdf"
    plt.savefig(out_file, bbox_inches="tight")
    print(f"Plot successfully saved to {out_file}")
    
    plt.show()

if __name__ == "__main__":
    main()