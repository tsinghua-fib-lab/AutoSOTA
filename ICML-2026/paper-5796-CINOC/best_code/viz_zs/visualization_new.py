import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from pathlib import Path
from matplotlib.ticker import PercentFormatter

# --- 1. Setup Directories ---
script_dir = Path(__file__).resolve().parent if "__file__" in locals() else Path.cwd()
SAVE_DIR = Path("figures/zs-comparisons_regimes")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# --- 2. Data Definitions ---
all_experiments = [
    # 1D Experiments
    { "name": "KS 1D", "n_train": 30, "path": "/home/zanot/projects/Multi-Agent-DPC/examples/ks1d/decentralized/figures/ks_zs_scaling/ks1d_zs_results.csv", "color": "#1f77b4" },
    { "name": "FKPP 1D", "n_train": 30, "path": "/home/zanot/projects/Multi-Agent-DPC/examples/fkpp1d/decentralized/figures/zs-comparisons/zs_scalability_results.csv", "color": "#d62728" },
    { "name": "Heat 1D", "n_train": 30, "path": "/home/zanot/projects/Multi-Agent-DPC/examples/heat1d/decentralized/figures/zs-comparisons/heat_1d_zs_results.csv", "color": "#9467bd" },
    
    # 2D Experiments
    { "name": "KS 2D", "n_train": 100, "path": "/home/zanot/projects/Multi-Agent-DPC/examples/ks2d_correct/decentralized/figures/zs_scaling/ks2d_zs_results.csv", "color": "#ff7f0e" },
    { "name": "Turbulence 2D", "n_train": 64, "path": "/home/zanot/projects/Multi-Agent-DPC/examples/turbulence2d/decentralized/figures/turb_scaling/turb_zs_results.csv", "color": "#2ca02c" },
    { "name": "Heat 2D", "n_train": 16, "path": "/home/zanot/projects/Multi-Agent-DPC/examples/heat2D/decentralized/figures/zs-comparisons/heat2d_zs_results.csv", "color": "#8c564b" },
    { "name": "Density Transport 2D", "n_train": 16, "path": "/home/zanot/projects/Multi-Agent-DPC/examples/density/decentralized/figures/ns2d_zs_scaling/ns2d_zs_results_n16.csv", "color": "#e377c2" }
]

markers = ['o', 's', '^', 'D', 'v', 'P', 'X']

# --- 3. Style Configuration ---
def setup_paper_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "legend.fontsize": 11,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.5,
        "grid.alpha": 0.3,
    })

# --- 4. Helper Functions ---
def load_and_process_data(exp_list):
    processed_data = []
    for i, exp in enumerate(exp_list):
        file_path = Path(exp["path"])
        if not file_path.exists():
            continue 
        try:
            df = pd.read_csv(file_path)
        except:
            continue

        if not {'n_agents', 'relative_mse'}.issubset(df.columns):
            continue

        df['rel_agents'] = df['n_agents'] / exp['n_train']
        df = df.sort_values('rel_agents')
        
        processed_data.append({
            "name": exp["name"], "color": exp["color"], "df": df, "marker": markers[i % len(markers)]
        })
    return processed_data

def add_baseline_line(ax):
    ax.axhline(y=100, color='#555555', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)
    
def add_training_line(ax):
    ax.axvline(x=1.0, color='black', linestyle=':', linewidth=1.5, alpha=0.6, zorder=1)

# --- 5. Plotting Functions ---

def plot_regimes(data):
    # Create wider figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True, gridspec_kw={'wspace': 0.1})
    ax_dec, ax_inc = axes

    Y_LIMIT = 250
    ax_dec.set_ylim(0, Y_LIMIT)
    ax_inc.set_ylim(0, Y_LIMIT)
    ax_dec.yaxis.set_major_formatter(PercentFormatter(decimals=0))

    # To collect handles for a shared legend
    legend_handles = {}

    # --- Plotting Loop ---
    for item in data:
        df = item['df']
        
        df_dec = df[df['rel_agents'] <= 1.001] 
        df_inc = df[df['rel_agents'] >= 0.999]
        
        # Plot Decreasing
        if not df_dec.empty:
            line, = ax_dec.plot(df_dec['rel_agents'], df_dec['relative_mse'], 
                        marker=item['marker'], linestyle='-', color=item['color'], 
                        alpha=0.9, markersize=8)
            legend_handles[item['name']] = line # Store for legend
            
        # Plot Increasing
        if not df_inc.empty:
            line, = ax_inc.plot(df_inc['rel_agents'], df_inc['relative_mse'], 
                        marker=item['marker'], linestyle='-', color=item['color'], 
                        alpha=0.9, markersize=8)
            legend_handles[item['name']] = line

    # --- Styling: Decreasing (Left) ---
    ax_dec.set_title(r"Decreasing Swarm Size ($M \leq M_{train}$)", fontweight='bold')
    ax_dec.set_xlabel(r"Relative Swarm Size")
    ax_dec.set_ylabel("Relative MSE")
    ax_dec.set_xscale('log')
    
    # Limits - Updated Left Limit
    ax_dec.set_xlim(left=0.25, right=1.05) 
    
    add_baseline_line(ax_dec)
    add_training_line(ax_dec)
    
    # Custom ticks - NO SCIENTIFIC NOTATION
    # Added 0.0625 (1/16) if it fits, or keep standard power of 2 fractions
    ticks = [0.25, 0.5, 1.0]
    ax_dec.set_xticks(ticks)
    ax_dec.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:g}')) 
    ax_dec.minorticks_off() 

    # --- Styling: Increasing (Right) ---
    ax_inc.set_title(r"Increasing Swarm Size ($M \geq M_{train}$)", fontweight='bold')
    ax_inc.set_xlabel(r"Relative Swarm Size")
    ax_inc.set_xscale('log')
    
    # Truncate at 16
    ax_inc.set_xlim(0.95, 16.5) 
    
    add_baseline_line(ax_inc)
    add_training_line(ax_inc)
    
    ax_inc.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax_inc.set_xticks([1, 2, 4, 8, 16])
    
    # Annotate "Training Scale"
    ax_dec.text(1.0, 5, 'Training Scale', rotation=90, va='bottom', ha='right', fontsize=10, color='#333333')
    ax_inc.text(1.0, 5, 'Training Scale', rotation=90, va='bottom', ha='left', fontsize=10, color='#333333')

    # --- Shared Legend ---
    from matplotlib.lines import Line2D
    baseline_line = Line2D([0], [0], color='#555555', linestyle='--', linewidth=1.5, alpha=0.7)
    
    names = list(legend_handles.keys())
    handles = list(legend_handles.values())
    
    names.append("Baseline (100%)")
    handles.append(baseline_line)
    
    # Place Legend OUTSIDE and ABOVE the figure
    fig.legend(handles, names, 
               loc='lower center', 
               bbox_to_anchor=(0.5, 0.92), 
               ncol=4, 
               frameon=False, 
               fontsize=11)

    plt.subplots_adjust(top=0.85, bottom=0.15, wspace=0.1)
    
    save_path = SAVE_DIR / "zs_scaling_regimes_final.pdf"
    fig.savefig(save_path, bbox_inches='tight')
    print(f"Saved final regime plot to {save_path}")

# --- 6. Main ---
def main():
    setup_paper_style()
    print("Loading Data...")
    all_data = load_and_process_data(all_experiments)
    print("Generating Regime Plot...")
    plot_regimes(all_data)

if __name__ == "__main__":
    main()