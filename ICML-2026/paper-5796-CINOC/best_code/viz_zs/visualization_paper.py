"""
Paper-Quality Visualization for Zero-Shot Scaling Regimes.
Generates the zs_scaling_regimes_final plot for publication.
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from pathlib import Path
from matplotlib.ticker import PercentFormatter
from matplotlib.lines import Line2D

# --- Setup Directories ---
script_dir = Path(__file__).resolve().parent
SAVE_DIR = script_dir / "figures" / "zs-comparisons_regimes"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# --- Data Definitions ---
all_experiments = [
    # 1D Experiments
    {"name": "KS 1D", "n_train": 30, 
     "path": script_dir.parent / "examples/ks1d/decentralized/figures/ks_zs_scaling/ks1d_zs_results.csv", 
     "color": "#1f77b4"},
    {"name": "FKPP 1D", "n_train": 30, 
     "path": script_dir.parent / "examples/fkpp1d/decentralized/figures/zs-comparisons/zs_scalability_results.csv", 
     "color": "#d62728"},
    {"name": "Heat 1D", "n_train": 30, 
     "path": script_dir.parent / "examples/heat1d/decentralized/figures/zs-comparisons/heat_1d_zs_results.csv", 
     "color": "#9467bd"},
    
    # 2D Experiments
    {"name": "KS 2D", "n_train": 100, 
     "path": script_dir.parent / "examples/ks2d_correct/decentralized/figures/zs_scaling/ks2d_zs_results.csv", 
     "color": "#ff7f0e"},
    {"name": "Turbulence 2D", "n_train": 64, 
     "path": script_dir.parent / "examples/turbulence2d/decentralized/figures/turb_scaling/turb_zs_results.csv", 
     "color": "#2ca02c"},
    {"name": "Heat 2D", "n_train": 16, 
     "path": script_dir.parent / "examples/heat2D/decentralized/figures/zs-comparisons/heat2d_zs_results.csv", 
     "color": "#8c564b"},
    {"name": "Density Transport 2D", "n_train": 16, 
     "path": script_dir.parent / "examples/density/decentralized/figures/ns2d_zs_scaling/ns2d_zs_results_n16.csv", 
     "color": "#e377c2"}
]

markers = ['o', 's', '^', 'D', 'v', 'P', 'X']


def setup_paper_style():
    """Configure matplotlib for publication-quality figures."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size":22,
        "axes.labelsize": 22,
        "axes.titlesize": 22,
        "legend.fontsize": 22,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.5,
        "grid.alpha": 0.3,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def load_and_process_data(exp_list):
    """Load CSV data for all experiments."""
    processed_data = []
    for i, exp in enumerate(exp_list):
        file_path = Path(exp["path"])
        if not file_path.exists():
            print(f"  ⚠ File not found: {file_path}")
            continue
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"  ⚠ Error loading {file_path}: {e}")
            continue

        if not {'n_agents', 'relative_mse'}.issubset(df.columns):
            print(f"  ⚠ Missing columns in {file_path}")
            continue

        df['rel_agents'] = df['n_agents'] / exp['n_train']
        df = df.sort_values('rel_agents')
        
        processed_data.append({
            "name": exp["name"], 
            "color": exp["color"], 
            "df": df, 
            "marker": markers[i % len(markers)]
        })
        print(f"  ✓ Loaded {exp['name']}: {len(df)} rows")
    return processed_data


def plot_scaling_regimes(data):
    """Generate the zero-shot scaling regimes plot."""
    # Original figure size - fonts will appear large
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True, gridspec_kw={'wspace': 0.1})
    ax_dec, ax_inc = axes

    Y_LIMIT = 250
    ax_dec.set_ylim(0, Y_LIMIT)
    ax_inc.set_ylim(0, Y_LIMIT)
    ax_dec.yaxis.set_major_formatter(PercentFormatter(decimals=0))

    legend_handles = {}

    # --- Plotting Loop ---
    for item in data:
        df = item['df']
        
        df_dec = df[df['rel_agents'] <= 1.001] 
        df_inc = df[df['rel_agents'] >= 0.999]
        
        # Plot Decreasing regime (left panel)
        if not df_dec.empty:
            line, = ax_dec.plot(df_dec['rel_agents'], df_dec['relative_mse'], 
                        marker=item['marker'], linestyle='-', color=item['color'], 
                        alpha=0.9, markersize=8)
            legend_handles[item['name']] = line
            
        # Plot Increasing regime (right panel)
        if not df_inc.empty:
            line, = ax_inc.plot(df_inc['rel_agents'], df_inc['relative_mse'], 
                        marker=item['marker'], linestyle='-', color=item['color'], 
                        alpha=0.9, markersize=8)
            legend_handles[item['name']] = line

    # --- Baseline Line ---
    ax_dec.axhline(y=100, color='#555555', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)
    ax_inc.axhline(y=100, color='#555555', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)
    
    # --- Training Scale Line ---
    ax_dec.axvline(x=1.0, color='black', linestyle=':', linewidth=1.5, alpha=0.6, zorder=1)
    ax_inc.axvline(x=1.0, color='black', linestyle=':', linewidth=1.5, alpha=0.6, zorder=1)

    # --- Left Panel: Decreasing ---
    ax_dec.set_title(r"Decreasing Swarm Size ($M \leq M_{train}$)")
    ax_dec.set_xlabel(r"Relative Swarm Size")
    ax_dec.set_ylabel("Relative MSE")
    ax_dec.set_xscale('log')
    ax_dec.set_xlim(left=0.20, right=1.05)  # Extended left to give 0.25 more space
    # Only use 3 tick values to avoid crowding
    ax_dec.set_xticks([0.25, 0.5, 1.0])
    ax_dec.set_xticklabels(['0.25', '0.5', '1'])
    ax_dec.minorticks_off()

    # --- Right Panel: Increasing ---
    ax_inc.set_title(r"Increasing Swarm Size ($M \geq M_{train}$)")
    ax_inc.set_xlabel(r"Relative Swarm Size")
    ax_inc.set_xscale('log')
    ax_inc.set_xlim(0.95, 7.5)
    # Only 3 tick values to avoid overlap with 20pt fonts
    ax_inc.set_xticks([1, 2, 5])
    ax_inc.set_xticklabels(['1', '2', '5'])
    ax_inc.minorticks_off()
    
    # Annotate "Training Scale" - left of line on left panel, right of line on right panel
    ax_dec.text(0.98, 5, 'Training\nScale', rotation=90, va='bottom', ha='right', fontsize=21, color='#333333')
    ax_inc.text(1.02, 5, 'Training\nScale', rotation=90, va='bottom', ha='left', fontsize=21, color='#333333')

    # --- Shared Legend ---
    baseline_line = Line2D([0], [0], color='#555555', linestyle='--', linewidth=1.5, alpha=0.7)
    names = list(legend_handles.keys()) + ["Baseline (100%)"]
    handles = list(legend_handles.values()) + [baseline_line]
    
    fig.legend(handles, names, 
               loc='lower center', 
               bbox_to_anchor=(0.5, 0.92), 
               ncol=4, 
               frameon=False, 
               fontsize=22)

    plt.subplots_adjust(top=0.85, bottom=0.15, wspace=0.1)
    
    # Save outputs
    save_path_pdf = SAVE_DIR / "zs_scaling_regimes_final.pdf"
    save_path_png = SAVE_DIR / "zs_scaling_regimes_final.png"
    fig.savefig(save_path_pdf, bbox_inches='tight')
    fig.savefig(save_path_png, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path_pdf}")
    print(f"✓ Saved: {save_path_png}")
    plt.close()


def main():
    print("=" * 60)
    print("Zero-Shot Scaling Regimes - Paper Visualization")
    print("=" * 60)
    
    setup_paper_style()
    
    print("\nLoading Data...")
    all_data = load_and_process_data(all_experiments)
    
    if not all_data:
        print("No data loaded! Check file paths.")
        return
    
    print(f"\nGenerating plot with {len(all_data)} experiments...")
    plot_scaling_regimes(all_data)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
