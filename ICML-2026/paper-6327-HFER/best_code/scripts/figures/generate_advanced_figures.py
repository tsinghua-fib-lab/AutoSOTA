import json
import os
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import glob

def load_data(results_file):
    with open(results_file, 'r') as f:
        data = json.load(f)
    return data

def safe_get(traj, layer_idx, metric):
    if layer_idx < len(traj):
        return traj[layer_idx].get(metric)
    return None

def generate_density_plot_tex(model_name, metric, valid_vals, invalid_vals):
    # Kernel Density Estimation
    if len(valid_vals) > 1 and len(invalid_vals) > 1:
        # Determine range
        min_v = min(min(valid_vals), min(invalid_vals))
        max_v = max(max(valid_vals), max(invalid_vals))
        margin = (max_v - min_v) * 0.2
        x_grid = np.linspace(min_v - margin, max_v + margin, 100)
        
        kde_valid = gaussian_kde(valid_vals)(x_grid)
        kde_invalid = gaussian_kde(invalid_vals)(x_grid)
        
        df = pd.DataFrame({'x': x_grid, 'y_valid': kde_valid, 'y_invalid': kde_invalid})
        data_str = df.to_csv(None, index=False, lineterminator='\n', na_rep='nan')
    else:
        # Fallback for insufficient data
        data_str = "x,y_valid,y_invalid\n0,0,0"

    metric_labels = {
        "fiedler_value": "Fiedler Value ($\\lambda_2$)",
        "hfer": "HFER (High Freq Energy Ratio)",
        "smoothness": "Smoothness Index ($\\eta$)",
        "entropy": "Spectral Entropy ($H$)"
    }
    label = metric_labels.get(metric, metric)

    return fR"""\documentclass[tikz,border=10pt]{{standalone}}
\usepackage{{pgfplots}}
\usepgfplotslibrary{{fillbetween}}
\pgfplotsset{{compat=1.18}}

\begin{{document}}
\begin{{tikzpicture}}
\pgfplotstableread[row sep=newline, col sep=comma]{{
{data_str}
}}\mydata

\begin{{axis}}[
    width=8cm, height=6cm,
    xlabel={{{label}}},
    ylabel={{Density}},
    title={{Distribution Sep. @ Final Layer}},
    grid=major,
    grid style={{dashed, gray!30}},
    legend pos=north east,
    legend style={{font=\small}},
    ymin=0,
    area style,
]

\addplot [name path=valid, draw=blue, fill=blue!10, fill opacity=0.5, thick] table [x=x, y=y_valid] {{\mydata}};
\addlegendentry{{Valid}}

\addplot [name path=invalid, draw=red, fill=red!10, fill opacity=0.5, thick] table [x=x, y=y_invalid] {{\mydata}};
\addlegendentry{{Invalid}}

\end{{axis}}
\end{{tikzpicture}}
\end{{document}}
"""

def generate_heatmap_tex(model_name, corr_matrix):
    # Prepare data for PGFPlots heatmap (x, y, meta)
    # x,y are indices, meta is the correlation value
    csv_data = "x,y,val\n"
    metrics = corr_matrix.columns
    for i, row_metric in enumerate(metrics):
        for j, col_metric in enumerate(metrics):
            val = corr_matrix.iloc[i, j]
            csv_data += f"{j},{i},{val:.2f}\n" # Swap i/j for correct orientation if needed, usually matrix plot expects x=col, y=row

    # Tick labels
    tick_labels = ",".join(metrics)
    
    return fR"""\documentclass[tikz,border=10pt]{{standalone}}
\usepackage{{pgfplots}}
\pgfplotsset{{compat=1.18}}

\begin{{document}}
\begin{{tikzpicture}}
\pgfplotstableread[row sep=newline, col sep=comma]{{
{csv_data}
}}\datatable

\begin{{axis}}[
    width=8cm, height=8cm,
    colorbar,
    colormap/coolwarm,
    point meta min=-1, point meta max=1,
    xtick={{0,1,2,3}},
    xticklabels={{Fiedler, HFER, Smoothness, entropy}},
    ytick={{0,1,2,3}},
    yticklabels={{Fiedler, HFER, Smoothness, entropy}},
    xticklabel style={{rotate=45, anchor=north east}},
    title={{Metric Correlation}},
    grid=none,
    y dir=reverse, % Matrix convention
    enlargelimits=false,
    nodes near coords={{\pgfmathprintnumber\pgfplotspointmeta}},
    nodes near coords style={{font=\scriptsize, color=black}},
]

\addplot[
    matrix plot*,
    mesh/cols=4, % Size of matrix
    point meta=explicit
] table [x=x, y=y, meta=val] {{\datatable}};

\end{{axis}}
\end{{tikzpicture}}
\end{{document}}
"""

def process_model_advanced(model_name, results_file, output_dir):
    print(f"Processing Advanced Stats for {model_name}...")
    data = load_data(results_file)
    
    model_safe = model_name.replace("/", "_").replace(" ", "_")
    model_dir = os.path.join(output_dir, model_safe)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    metrics = ["fiedler_value", "hfer", "smoothness", "entropy"]
    
    # --- 1. Density Plots (Final Layer) ---
    # Extract final values
    for metric in metrics:
        valid_vals = []
        invalid_vals = []
        
        for item in data['valid']:
            traj = item['trajectory']
            if traj:
                val = traj[-1].get(metric) # Last layer
                if val is not None: valid_vals.append(val)
        
        for item in data['invalid']:
            traj = item['trajectory']
            if traj:
                val = traj[-1].get(metric)
                if val is not None: invalid_vals.append(val)
                
        tex = generate_density_plot_tex(model_name, metric, valid_vals, invalid_vals)
        with open(os.path.join(model_dir, f"density_{metric}.tex"), "w") as f:
            f.write(tex)

    # --- 2. Correlation Matrix ---
    # Gather all final layer features into a DF
    rows = []
    for item in data['valid'] + data['invalid']:
        row = {}
        traj = item['trajectory']
        if traj:
            for m in metrics:
                val = traj[-1].get(m)
                if val is not None: row[m] = val
        if row: rows.append(row)
        
    if rows:
        df = pd.DataFrame(rows)
        corr = df.corr()
        tex = generate_heatmap_tex(model_name, corr)
        with open(os.path.join(model_dir, "correlation_matrix.tex"), "w") as f:
            f.write(tex)

if __name__ == "__main__":
    results_dir = "data/results"
    output_dir = "data/paper_figures"
    
    model_files = glob.glob(os.path.join(results_dir, "experiment_results_*.json"))
    
    for res_file in model_files:
        # Parse model name from filename
        # Format: experiment_results_Masked-Llama-3-8B.json
        base = os.path.basename(res_file)
        model_name = base.replace("experiment_results_", "").replace(".json", "")
        
        process_model_advanced(model_name, res_file, output_dir)
