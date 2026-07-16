import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from pathlib import Path
from matplotlib.ticker import PercentFormatter

# --- 1. Setup Directories ---
script_dir = Path(__file__).resolve().parent if "__file__" in locals() else Path.cwd()
SAVE_DIR = Path("figures/zs-comparisons_aggregated")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# --- 2. Data Definitions ---
experiments = [
    { "name": "KS 1D", "n_train": 30, "path": "/home/zanot/projects/Multi-Agent-DPC/examples/ks1d/decentralized/figures/ks_zs_scaling/ks1d_zs_results.csv", "color": "#1f77b4" },
    { "name": "KS 2D", "n_train": 100, "path": "/home/zanot/projects/Multi-Agent-DPC/examples/ks2d_correct/decentralized/figures/zs_scaling/ks2d_zs_results.csv", "color": "#ff7f0e" },
    { "name": "Turbulence 2D", "n_train": 64, "path": "/home/zanot/projects/Multi-Agent-DPC/examples/turbulence2d/decentralized/figures/turb_scaling/turb_zs_results.csv", "color": "#2ca02c" },
    { "name": "FKPP 1D", "n_train": 30, "path": "/home/zanot/projects/Multi-Agent-DPC/examples/fkpp1d/decentralized/figures/zs-comparisons/zs_scalability_results.csv", "color": "#d62728" },
    { "name": "Heat 1D", "n_train": 30, "path": "/home/zanot/projects/Multi-Agent-DPC/examples/heat1d/decentralized/figures/zs-comparisons/heat_1d_zs_results.csv", "color": "#9467bd" },
    { "name": "Heat 2D", "n_train": 16, "path": "/home/zanot/projects/Multi-Agent-DPC/examples/heat2D/decentralized/figures/zs-comparisons/heat2d_zs_results.csv", "color": "#8c564b" },
    { "name": "Density Transport 2D", "n_train": 16, "path": "/home/zanot/projects/Multi-Agent-DPC/examples/density/decentralized/figures/ns2d_zs_scaling/ns2d_zs_results_n16.csv", "color": "#e377c2" }
]

# --- 3. Plotting Logic ---

def main():
    # Use a cleaner style for papers
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Increase default font sizes for readability in papers
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'font.family': 'serif' # Matches most academic papers (Times/Computer Modern)
    })

    # Create figure - slightly wider to accommodate legend if needed
    fig, ax = plt.subplots(figsize=(9, 6))
    
    Y_LIMIT = 250.0
    
    # Define a list of markers to cycle through for accessibility
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X']

    # Track the maximum x-value found in data
    max_observed_x = 0.0

    print(f"{'Experiment':<25} | {'Points':<10} | {'Info'}")
    print("-" * 60)

    # Loop with index to assign markers
    for i, exp in enumerate(experiments):
        file_path = Path(exp["path"])
        
        if not file_path.exists():
            print(f"{exp['name']:<25} | MISSING    | File not found")
            continue
            
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"{exp['name']:<25} | ERROR      | {e}")
            continue

        required_cols = {'n_agents', 'relative_mse'}
        if not required_cols.issubset(df.columns):
            continue

        # Data Processing
        df['rel_agents'] = df['n_agents'] / exp['n_train']
        df = df.sort_values('rel_agents')
        
        # Update global max tracker
        if not df.empty:
            current_max = df['rel_agents'].max()
            if current_max > max_observed_x:
                max_observed_x = current_max

        # Determine marker based on index
        marker = markers[i % len(markers)]

        ax.plot(
            df['rel_agents'], 
            df['relative_mse'], 
            marker=marker, 
            linestyle='-', 
            linewidth=2.5,       # Slightly thicker lines
            markersize=8,        # Slightly larger markers
            color=exp['color'], 
            label=exp['name'],
            alpha=0.9,
            clip_on=True
        )

        print(f"{exp['name']:<25} | {len(df):<10} | Plotted")

    # --- 4. Styling ---
    
    # 1. Baseline Line (Thick, dashed grey)
    ax.axhline(y=100, color='#555555', linestyle='--', linewidth=2, alpha=0.7, label='Baseline (100%)', zorder=1)
    
    # 2. Training Scale Line (Vertical) - ANNOTATED DIRECTLY
    ax.axvline(x=1.0, color='black', linestyle=':', linewidth=2, alpha=0.6, zorder=1)
    ax.text(1.05, 5, 'Training Scale ($1.0x$)', rotation=90, verticalalignment='bottom', color='#333333', fontsize=10)

    # 3. Axis Limits and formatting
    ax.set_ylim(0, Y_LIMIT)
    
    # DYNAMIC X-AXIS FIX:
    # Strictly cut off the view at max_observed_x + buffer.
    ax.set_xlim(0, max_observed_x + 0.2)
    
    # Format Y axis as percentage
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    
    # Format X axis: Force ticks at integers (0, 1, 2...) without expanding the view limit
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

    ax.set_title("Zero-Shot Scalability: Relative MSE vs Swarm Size", fontweight='bold', pad=15)
    ax.set_xlabel(r"Relative Swarm Size ($N / N_{train}$)")
    ax.set_ylabel("Relative MSE (%)")

    # 4. Legend Placement (Outside Right)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0, frameon=False)

    plt.tight_layout()
    
    plot_path = SAVE_DIR / "aggregated_zs_scalability_paper.pdf"
    fig.savefig(plot_path, dpi=300, bbox_inches='tight') 
    print("-" * 60)
    print(f"Aggregated plot saved to: {plot_path}")

if __name__ == "__main__":
    main()