"""
p_vals.py

Create a heatmap of the p-values of the wilcoxon test for each baseline.
"""
import pandas as pd, numpy as np, matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
IN_CSV   = SCRIPTS_DIR / "all_results.csv"
OUT_DIR  = SCRIPTS_DIR / "figures"
OUT_DIR.mkdir(exist_ok=True)

dataset_to_name = {
    "227_cpu_small": "CPU Performance",
    "294_satellite_image": "Satellite Image",
    "503_wind": "Wind Power",
    "623_fri_c4_1000_10": "Synthetic Regression",
    "ConcreteCompressiveStrength": "Concrete Strength",
    "EnergyEfficiency": "Energy Efficiency",
    "HousePrice": "House Price",
    "ParkinsonsTelemonitoring": "Parkinson's Monitoring",
    "WineQuality": "Wine Quality",
}

# --------------------------------------------------
# 1. Read once, create train-fraction column
# --------------------------------------------------
df = pd.read_csv(IN_CSV)

df["dataset_name"] = df["dataset"].map(dataset_to_name) 

df["train_frac"] = (
    df.groupby("dataset")["sample_size"]
      .transform(lambda s: s / s.max())        # {0.2, 0.4, 0.6, 0.8, 1.0}
)

# Ordered labels for the x-axis
FRAC_LABELS = ["N/5", "2N/5", "3N/5", "4N/5", "N"]
FRAC_ORDER  = [0.2, 0.4, 0.6, 0.8, 1.0]       # ensures consistent column order

# --------------------------------------------------
# 2. Function to draw one heat-map
# --------------------------------------------------
def plot_heatmap(sub_df: pd.DataFrame, baseline_name: str) -> None:
    """Create and save a heat-map for a single baseline."""
    pivot = (
        sub_df.pivot_table(index="dataset_name",
                           columns="train_frac",
                           values="p_wilcoxon",
                           aggfunc="first")        
            .reindex(columns=FRAC_ORDER)          
    )

    logp = -np.log10(pivot)


    fig, ax = plt.subplots(figsize=(7, 3.5))
    sns.heatmap(
        logp,
        cmap="YlGnBu",
        vmin=0, vmax=4,                           
        linewidths=.4, linecolor="w",
        cbar_kws={
            "label": r"$-\log_{10}(p)$",
            "ticks": [0, 1, 1.3, 2, 3, 4]         
        },
        annot=pivot.round(3),                    
        fmt=".3f", annot_kws={"size":6},
        ax=ax
    )

    cbar  = ax.collections[0].colorbar
    cbar.ax.hlines(1.3, *cbar.ax.get_xlim(),
                   colors="k", linestyles="--", linewidth=.8)

    ax.set_xticklabels(FRAC_LABELS, rotation=0)
    ax.set_xlabel("sample size fraction")
    ax.set_ylabel("")            
    plt.tight_layout()

    # Save
    out_file = OUT_DIR / f"wilcoxon_heatmap_{baseline_name}.png"
    fig.savefig(out_file)
    plt.close(fig)
    print(f"saved {out_file}")

# --------------------------------------------------
# 3. Generate one figure per baseline value
# --------------------------------------------------
for baseline in df["baseline"].unique():
    plot_heatmap(df[df["baseline"] == baseline], baseline)
