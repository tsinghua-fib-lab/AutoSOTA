import pandas as pd
import numpy as np
from pathlib import Path

# --- 1. Configuration ---
MSE_THRESHOLD = 250.0  # The "Zero-Shot" success limit

experiments = [
    { "name": "Kuramoto-Sivashinsky 1D", "short_name": "KS 1D", "n_train": 30, "path": "/home/zanot/projects/Multi-Agent-DPC/examples/ks1d/decentralized/figures/ks_zs_scaling/ks1d_zs_results.csv" },
    { "name": "Kuramoto-Sivashinsky 2D", "short_name": "KS 2D", "n_train": 196, "path": "/home/zanot/projects/Multi-Agent-DPC/examples/ks2d_correct/decentralized/figures/zs_scaling/ks2d_zs_results.csv" },
    { "name": "Turbulence 2D", "short_name": "Turbulence 2D", "n_train": 64, "path": "/home/zanot/projects/Multi-Agent-DPC/examples/turbulence2d/decentralized/figures/turb_scaling/turb_zs_results.csv" },
    { "name": "FKPP 1D", "short_name": "FKPP 1D", "n_train": 30, "path": "/home/zanot/projects/Multi-Agent-DPC/examples/fkpp1d/decentralized/figures/zs-comparisons/zs_scalability_results.csv" },
    { "name": "Heat Equation 1D", "short_name": "Heat 1D", "n_train": 30, "path": "/home/zanot/projects/Multi-Agent-DPC/examples/heat1d/decentralized/figures/zs-comparisons/heat_1d_zs_results.csv" },
    { "name": "Heat Equation 2D", "short_name": "Heat 2D", "n_train": 16, "path": "/home/zanot/projects/Multi-Agent-DPC/examples/heat2D/decentralized/figures/zs-comparisons/heat2d_zs_results.csv" },
    { "name": "Density Transport 2D", "short_name": "Density 2D", "n_train": 16, "path": "/home/zanot/projects/Multi-Agent-DPC/examples/density/decentralized/figures/ns2d_zs_scaling/ns2d_zs_results_n16.csv" }
]

# Define the exact display order requested
TARGET_ORDER = [
    "FKPP 1D",
    "Heat 1D",
    "Heat 2D",
    "KS 2D",
    "KS 1D",
    "Turbulence 2D",
    "Density 2D"
]

def generate_table_latex():
    table_data = []

    # --- 2. Process Data ---
    for exp in experiments:
        file_path = Path(exp["path"])
        
        if not file_path.exists():
            continue

        try:
            df = pd.read_csv(file_path)
            valid_df = df[df['relative_mse'] < MSE_THRESHOLD]
            
            if valid_df.empty:
                continue

            idx_min = valid_df['n_agents'].idxmin()
            idx_max = valid_df['n_agents'].idxmax()

            row_min = valid_df.loc[idx_min]
            row_max = valid_df.loc[idx_max]

            m_train = exp['n_train']

            table_data.append({
                "name": exp['short_name'],
                "M": m_train,
                
                # Min Stats
                "n_min": int(row_min['n_agents']),
                "rel_min": row_min['n_agents'] / m_train,
                "mse_min": row_min['relative_mse'],
                
                # Max Stats
                "n_max": int(row_max['n_agents']),
                "rel_max": row_max['n_agents'] / m_train,
                "mse_max": row_max['relative_mse']
            })
            
        except Exception as e:
            print(f"Error processing {exp['name']}: {e}")
            continue

    # --- 3. Sort Data ---
    # Sort based on the index in TARGET_ORDER
    # Helper to safely get index (defaults to 999 if not found to push to bottom)
    def get_sort_index(row):
        try:
            return TARGET_ORDER.index(row['name'])
        except ValueError:
            return 999
            
    table_data.sort(key=get_sort_index)

    # --- 4. Generate LaTeX ---
    latex_lines = []
    
    latex_lines.append(r"\begin{table}[h]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{\textbf{Cardinality Invariance.} Range of deployment agents ($M_{test}$) where performance remains stable (defined as problem-specific loss $\le 250\%$ of the baseline where $M_{test} = M_{train}$) Values in parentheses denote scale relative to training size ($M_{train}$).}")
    latex_lines.append(r"\label{tab:zs_scalability}")
    latex_lines.append(r"\renewcommand{\arraystretch}{1.4}") 
    latex_lines.append(r"\begin{tabular}{l c c c}")
    latex_lines.append(r"\toprule")
    
    # --- HEADERS ---
    h_env   = r"\textbf{PDE}"
    h_train = r"\makecell{\textbf{Train} \\ ($M_{train}$)}"
    h_min   = r"\makecell{\textbf{Min Stable} \\ ($M_{test}$)}"
    h_max   = r"\makecell{\textbf{Max Stable} \\ ($M_{test}$)}"
    
    latex_lines.append(f"{h_env} & {h_train} & {h_min} & {h_max} \\\\")
    latex_lines.append(r"\midrule")

    for row in table_data:
        name = row['name']
        M = row['M']
        
        # Train Cell (Now includes 100.00 baseline MSE)
        train_cell = f"\\makecell{{{M} \\\\ \\color{{gray}}\\scriptsize MSE: 100.00}}"

        # Min Cell
        min_n = row['n_min']
        min_scale = row['rel_min']
        min_mse = row['mse_min']
        min_cell = f"\\makecell{{{min_n} ({min_scale:.1f}$\\times$) \\\\ \\color{{gray}}\\scriptsize MSE: {min_mse:.2f}}}"

        # Max Cell
        max_n = row['n_max']
        max_scale = row['rel_max']
        max_mse = row['mse_max']
        max_cell = f"\\makecell{{{max_n} ({max_scale:.1f}$\\times$) \\\\ \\color{{gray}}\\scriptsize MSE: {max_mse:.2f}}}"

        latex_lines.append(f"{name} & {train_cell} & {min_cell} & {max_cell} \\\\")

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table}")

    return "\n".join(latex_lines)

if __name__ == "__main__":
    print(generate_table_latex())