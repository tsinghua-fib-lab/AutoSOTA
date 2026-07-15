import json
import os
import glob
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import gaussian_kde, ttest_ind, mannwhitneyu

# Best Metric/Layer configurations for Spotlight Curves (Distribution Plots)
SPOTLIGHT_CONFIG = {
    "Llama-3.2-1B": [("hfer", 0), ("fiedler_value", 0)],
    "Llama-3.2-3B": [("hfer", 11), ("hfer", 0)],
    "Meta-Llama-3.1-8B": [("hfer", 30), ("smoothness", 9)],
    "Qwen2.5-7B": [("hfer", 26), ("entropy", 2)],
    "Qwen2.5-0.5B": [("entropy", 0), ("energy", 19)],
    "Phi-3.5-mini": [("smoothness", 25), ("hfer", 26)],
    "Mistral-7B": [("smoothness", 26), ("hfer", 11)],
    "Qwen1.5-MoE-A2.7B-Chat": [("smoothness", 6), ("fiedler_value", 6)]
}

def load_data(results_file):
    with open(results_file, 'r') as f:
        data = json.load(f)
    return data

def safe_get(traj, layer_idx, metric):
    if layer_idx < len(traj):
        return traj[layer_idx].get(metric)
    return None

def process_model(model_name, results_file, list_b_file, output_dir):
    print(f"Processing {model_name}...")
    data = load_data(results_file)

    # --- RELABELING LOGIC ---
    reclaimed_files = set()
    if list_b_file and os.path.exists(list_b_file):
        with open(list_b_file, 'r') as f:
            raw = json.load(f)
            # Handle both list of dicts and list of strings
            if raw and isinstance(raw[0], dict):
                reclaimed_files = set(item['file'] for item in raw)
            else:
                reclaimed_files = set(raw)
    
    valid_items = data['valid'][:] # Copy
    invalid_items = []
    
    for item in data['invalid']:
        if item['file'] in reclaimed_files:
            valid_items.append(item)
        else:
            invalid_items.append(item)
    
    # Update data reference for subsequent steps
    data['valid'] = valid_items
    data['invalid'] = invalid_items
    print(f"  - Valid: {len(valid_items)}, Invalid: {len(invalid_items)} (Reclaimed {len(reclaimed_files)})")
    # ------------------------
    
    # Sample subset for traces
    n_samples = 20
    valid_samples = random.sample(data['valid'], min(len(data['valid']), n_samples))
    invalid_samples = random.sample(data['invalid'], min(len(data['invalid']), n_samples))
    
    metrics = ["fiedler_value", "hfer", "smoothness", "entropy"]
    metric_labels = {
        "fiedler_value": "Fiedler Value ($\\lambda_2$)",
        "hfer": "HFER (High Freq Energy Ratio)",
        "smoothness": "Smoothness Index ($\\eta$)",
        "entropy": "Spectral Entropy ($H$)"
    }
    
    # Detect max layers
    max_layers = 0
    all_trajs = [item['trajectory'] for item in data['valid'] + data['invalid']]
    if all_trajs:
        max_layers = max(len(t) for t in all_trajs)
    
    model_safe = model_name.replace("/", "_").replace(" ", "_").replace("Instruct", "").strip("_-") # Simplify name
    model_dir = os.path.join(output_dir, model_safe)
    print(f"DEBUG: Output dir: {model_dir}")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    # --- Generate Data Strings (Inline) ---
    all_stats_strs = {}
    all_sample_strs = {}
    all_sample_cmds = {}
    
    for metric in metrics:
        # 1. Statistics Data
        stats_data = []
        stats_data = []
        for l in range(max_layers):
            # Valid
            v_vals = [safe_get(x['trajectory'], l, metric) for x in data['valid']]
            v_vals = [v for v in v_vals if v is not None]
            # Invalid
            i_vals = [safe_get(x['trajectory'], l, metric) for x in data['invalid']]
            i_vals = [v for v in i_vals if v is not None]
            
            row = {'layer': l}
            if v_vals:
                row['valid_mean'] = np.mean(v_vals)
                row['valid_std'] = np.std(v_vals)
                row['valid_ci_upper'] = row['valid_mean'] + row['valid_std']
                row['valid_ci_lower'] = row['valid_mean'] - row['valid_std']
            else:
                row['valid_mean'] = np.nan
                row['valid_std'] = np.nan
                row['valid_ci_upper'] = np.nan
                row['valid_ci_lower'] = np.nan
            
            if i_vals:
                row['invalid_mean'] = np.mean(i_vals)
                row['invalid_std'] = np.std(i_vals)
                row['invalid_ci_upper'] = row['invalid_mean'] + row['invalid_std']
                row['invalid_ci_lower'] = row['invalid_mean'] - row['invalid_std']
            else:
                row['invalid_mean'] = np.nan
                row['invalid_std'] = np.nan
                row['invalid_ci_upper'] = np.nan
                row['invalid_ci_lower'] = np.nan
            stats_data.append(row)
        
        df_stats = pd.DataFrame(stats_data)
        stats_str = df_stats.to_csv(None, index=False, lineterminator='\n', na_rep='nan')
        all_stats_strs[metric] = stats_str
        
        # 2. Sample Traces Data
        sample_rows = []
        for l in range(max_layers):
            row = {'layer': l}
            # Add columns like v_0, v_1 ... i_0, i_1 ...
            for idx, item in enumerate(valid_samples):
                val = safe_get(item['trajectory'], l, metric)
                row[f'v_{idx}'] = val if val is not None else np.nan
            for idx, item in enumerate(invalid_samples):
                val = safe_get(item['trajectory'], l, metric)
                row[f'i_{idx}'] = val if val is not None else np.nan
            sample_rows.append(row)
            
        df_samples = pd.DataFrame(sample_rows)
        sample_str = df_samples.to_csv(None, index=False, lineterminator='\n', na_rep='nan')
        all_sample_strs[metric] = sample_str
        
        # Generator plot commands (using the Table macro mapping)
        cmds = ""
        # Valid Samples (Blue)
        for idx in range(len(valid_samples)):
            cmds += f"\\addplot [blue, opacity=0.1, line width=0.1pt, forget plot] table [x=layer, y=v_{idx}] {{\\mysampleddata}};\n"
        # Invalid Samples (Red)
        for idx in range(len(invalid_samples)):
            cmds += f"\\addplot [red, opacity=0.1, line width=0.1pt, forget plot] table [x=layer, y=i_{idx}] {{\\mysampleddata}};\n"
        all_sample_cmds[metric] = cmds
        
        # --- Generate LaTeX for Individual Plot ---
        # Note: We pass sample_str and cmds separate so we can use pgfplotstableread
        tex_content = generate_single_plot_tex(metric, metric_labels[metric], stats_str, sample_str, cmds, max_layers)
        with open(os.path.join(model_dir, f"plot_{metric}.tex"), "w") as f:
            f.write(tex_content)

    # --- Generate Combined Plot ---
    combined_tex = generate_combined_plot_tex(
        metrics, 
        metric_labels, 
        [all_stats_strs[m] for m in metrics], 
        [all_sample_strs[m] for m in metrics], 
        [all_sample_cmds[m] for m in metrics], 
        max_layers
    )
    with open(os.path.join(model_dir, "combined_plot.tex"), "w") as f:
        f.write(combined_tex)

    # --- Generate SPOTLIGHT Distribution Plots ---
    # Check if this model has any spotlights configured
    # We do partial matching on model name because SPOTLIGHT_CONFIG keys might be slightly different or substrings
    
    # Find best matching key
    config_key = None
    for k in SPOTLIGHT_CONFIG.keys():
        if k in model_name: 
            config_key = k
            break
    
    if config_key:
        print(f"  - Generating Spotlights for {config_key}: {SPOTLIGHT_CONFIG[config_key]}")
        for metric, layer in SPOTLIGHT_CONFIG[config_key]:
            # Extract data
            v_vals = [safe_get(x['trajectory'], layer, metric) for x in data['valid']]
            v_vals = [v for v in v_vals if v is not None]
            
            i_vals = [safe_get(x['trajectory'], layer, metric) for x in data['invalid']]
            i_vals = [v for v in i_vals if v is not None]
            
            if len(v_vals) > 1 and len(i_vals) > 1:
                dist_tex = generate_distribution_tex(metric, layer, v_vals, i_vals, metric_labels.get(metric, metric))
                fname = f"dist_{metric}_L{layer}.tex"
                with open(os.path.join(model_dir, fname), "w") as f:
                    f.write(dist_tex)

def generate_single_plot_tex(metric, label, stats_str, sample_str, sample_cmds, max_layers):
    return fR"""\documentclass[tikz,border=10pt]{{standalone}}
\usepackage{{pgfplots}}
\usepgfplotslibrary{{fillbetween}}
\pgfplotsset{{compat=1.18}}

\begin{{document}}
\begin{{tikzpicture}}
% Define Sample Data Table
\pgfplotstableread[row sep=newline, col sep=comma]{{
{sample_str}
}}\mysampleddata

\begin{{axis}}[
    width=10cm, height=6cm,
    xlabel={{Layer Index}},
    ylabel={{{label}}},
    title={{Layer-wise Trajectory}},
    grid=major,
    grid style={{dashed, gray!30}},
    xmin=0, xmax={max_layers-1},
    legend pos=north west,
    legend style={{font=\small}},
    mark size=0.5pt
]

% --- Sample Traces ---
{sample_cmds}

% Valid CI
\addplot [name path=v_upper, draw=none, forget plot] table [x=layer, y=valid_ci_upper, col sep=comma, row sep=newline] {{
{stats_str}
}};
\addplot [name path=v_lower, draw=none, forget plot] table [x=layer, y=valid_ci_lower, col sep=comma, row sep=newline] {{
{stats_str}
}};
\addplot [blue!10, forget plot] fill between [of=v_upper and v_lower];

% Invalid CI
\addplot [name path=i_upper, draw=none, forget plot] table [x=layer, y=invalid_ci_upper, col sep=comma, row sep=newline] {{
{stats_str}
}};
\addplot [name path=i_lower, draw=none, forget plot] table [x=layer, y=invalid_ci_lower, col sep=comma, row sep=newline] {{
{stats_str}
}};
\addplot [red!10, forget plot] fill between [of=i_upper and i_lower];

% Means
\addplot [blue, thick] table [x=layer, y=valid_mean, col sep=comma, row sep=newline] {{
{stats_str}
}};
\addlegendentry{{Valid Proofs}}

\addplot [red, thick] table [x=layer, y=invalid_mean, col sep=comma, row sep=newline] {{
{stats_str}
}};
\addlegendentry{{Invalid Proofs}}

\end{{axis}}
\end{{tikzpicture}}
\end{{document}}
"""

def generate_combined_plot_tex(metrics, metric_labels, stats_strs, sample_strs, sample_cmds_list, max_layers):
    
    plots_code = ""
    for i, (metric, s_str, smp_str, smp_cmd) in enumerate(zip(metrics, stats_strs, sample_strs, sample_cmds_list)):
        label = metric_labels[metric]
        plots_code += fR"""
    \nextgroupplot[
        title={{{label}}},
        xlabel={{Layer}},
        grid=major,
        xmin=0, xmax={max_layers-1},
        legend style={{font=\tiny}},
        tick label style={{font=\tiny}},
        label style={{font=\small}},
        title style={{font=\small}}
    ]
    % Define Local Sample Data (inside axis? No, inside group nextgroupplot scope usually works, or define before)
    % pgfplotstableread inside a groupplot can be tricky.
    % Safest is to define it inline for each plot OR use a unique macro name per plot.
    % Let's use inline read to a temporary macro.
    \pgfplotstableread[row sep=newline, col sep=comma]{{
{smp_str}
    }}\mysampleddata
    
    % --- Sample Traces ---
    {smp_cmd}

    % Valid CI
    \addplot [name path=v_upper, draw=none, forget plot] table [x=layer, y=valid_ci_upper, col sep=comma, row sep=newline] {{
{s_str}
}};
    \addplot [name path=v_lower, draw=none, forget plot] table [x=layer, y=valid_ci_lower, col sep=comma, row sep=newline] {{
{s_str}
}};
    \addplot [blue!10, forget plot] fill between [of=v_upper and v_lower];

    % Invalid CI
    \addplot [name path=i_upper, draw=none, forget plot] table [x=layer, y=invalid_ci_upper, col sep=comma, row sep=newline] {{
{s_str}
}};
    \addplot [name path=i_lower, draw=none, forget plot] table [x=layer, y=invalid_ci_lower, col sep=comma, row sep=newline] {{
{s_str}
}};
    \addplot [red!10, forget plot] fill between [of=i_upper and i_lower];

    % Means
    \addplot [blue, thick] table [x=layer, y=valid_mean, col sep=comma, row sep=newline] {{
{s_str}
}};
    \addplot [red, thick] table [x=layer, y=invalid_mean, col sep=comma, row sep=newline] {{
{s_str}
}};
"""
    
    return fR"""\documentclass[tikz,border=10pt]{{standalone}}
\usepackage{{pgfplots}}
\usepgfplotslibrary{{groupplots}}
\usepgfplotslibrary{{fillbetween}}
\pgfplotsset{{compat=1.18}}

\begin{{document}}
\begin{{tikzpicture}}
\usetikzlibrary{{calc}}
\begin{{groupplot}}[
    group style={{
        group size=4 by 1,
        horizontal sep=1.5cm,
        vertical sep=1cm
    }},
    width=5cm, height=4cm
]
{plots_code}
\end{{groupplot}}
% Legend (Global)
\node[anchor=north] at ($(group c2r1.south)!0.5!(group c3r1.south) + (0,-1cm)$) {{
    \begin{{tikzpicture}}
        \draw[blue, thick] (0,0) -- (0.5,0) node[right, black] {{Valid}};
        \draw[red, thick] (1.5,0) -- (2.0,0) node[right, black] {{Invalid}};
    \end{{tikzpicture}}
}};
\end{{tikzpicture}}
\end{{document}}
"""

def generate_distribution_tex(metric, layer, v_vals, i_vals, label):
    # 1. Calc Stats
    mu_v, std_v = np.mean(v_vals), np.std(v_vals)
    mu_i, std_i = np.mean(i_vals), np.std(i_vals)
    n_v, n_i = len(v_vals), len(i_vals)
    pooled_std = np.sqrt(((n_v - 1)*std_v**2 + (n_i - 1)*std_i**2) / (n_v + n_i - 2))
    d = (mu_v - mu_i) / pooled_std
    
    # Calculate both p-values
    stat_mw, p_mw = mannwhitneyu(v_vals, i_vals)
    stat_t, p_t = ttest_ind(v_vals, i_vals, equal_var=False)

    # 2. KDE using Scipy
    # Define range
    min_val = min(min(v_vals), min(i_vals))
    max_val = max(max(v_vals), max(i_vals))
    margin = (max_val - min_val) * 0.2
    x_grid = np.linspace(min_val - margin, max_val + margin, 200)
    
    kde_v = gaussian_kde(v_vals)
    kde_i = gaussian_kde(i_vals)
    
    y_v = kde_v(x_grid)
    y_i = kde_i(x_grid)
    
    # 3. Create DataFrame and CSV string
    df = pd.DataFrame({
        'x': x_grid,
        'y_valid': y_v,
        'y_invalid': y_i
    })
    
    csv_str = df.to_csv(None, index=False, lineterminator='\n')
    
    # Format p-values for title
    p_mw_exp = int(np.log10(p_mw)) if p_mw > 0 else -100
    p_t_exp = int(np.log10(p_t)) if p_t > 0 else -100
    
    # 4. Generate TeX
    return fR"""\documentclass[tikz,border=10pt]{{standalone}}
\usepackage{{pgfplots}}
\usepackage{{amsmath}}
\usepgfplotslibrary{{fillbetween}}
\pgfplotsset{{compat=1.18}}

\begin{{document}}
\begin{{tikzpicture}}
\pgfplotstableread[row sep=newline, col sep=comma]{{
{csv_str}
}}\mydistdata

\begin{{axis}}[
    width=8cm, height=5cm,
    xlabel={{{label}}},
    ylabel={{Density}},
    title={{Layer {layer}: $d={d:.2f}$, $p_{{MW}} < 10^{{{p_mw_exp}}}$, $p_{{t}} < 10^{{{p_t_exp}}}$}},
    grid=major,
    grid style={{dashed, gray!30}},
    legend pos=north east,
    area style,
]

% Valid Distribution
\addplot [name path=valid, draw=blue, thick, fill=blue!20, fill opacity=0.5] table [x=x, y=y_valid] {{\mydistdata}};
\addlegendentry{{Valid ($N={n_v}$)}}

% Invalid Distribution
\addplot [name path=invalid, draw=red, thick, fill=red!20, fill opacity=0.5] table [x=x, y=y_invalid] {{\mydistdata}};
\addlegendentry{{Invalid ($N={n_i}$)}}

% Vertical Mean Lines (Optional)
\draw [blue, dashed, thick] (axis cs:{mu_v},0) -- (axis cs:{mu_v},\pgfkeysvalueof{{/pgfplots/ymax}});
\draw [red, dashed, thick] (axis cs:{mu_i},0) -- (axis cs:{mu_i},\pgfkeysvalueof{{/pgfplots/ymax}});

\end{{axis}}
\end{{tikzpicture}}
\end{{document}}
"""

if __name__ == "__main__":
    results_dir = "data/results"
    reclaimed_dir = "data/reclaimed"
    output_dir = "data/paper_figures"
    
    # Define mapping of Model -> (Result File, List B File)
    # We use substrings for matching
    models_to_process = [
        ("Llama-3.2-1B", "experiment_results_Llama-3.2-1B-Instruct.json", "1B_list_b_confident_invalid.json"),
        ("Llama-3.2-3B", "experiment_results_Llama-3.2-3B-Instruct.json", "3B_list_b_confident_invalid.json"),
        ("Meta-Llama-3.1-8B", "experiment_results_Meta-Llama-3.1-8B-Instruct.json", "8B_list_b_confident_invalid.json"),
        ("Qwen2.5-7B", "experiment_results_Qwen2.5-7B-Instruct.json", "Qwen7B_list_b_confident_invalid.json"),
        ("Qwen2.5-0.5B-Instruct", "experiment_results_Qwen2.5-0.5B-Instruct.json", "Qwen0.5B_list_b_confident_invalid.json"),
        ("Phi-3.5-mini", "experiment_results_Phi-3.5-mini-instruct.json", "Phi3.5_list_b_confident_invalid.json"),
        ("Mistral-7B-v0.1", "experiment_results_Mistral-7B-v0.1.json", "Mistral7B_list_b_confident_invalid.json"),
        ("Qwen1.5-MoE-A2.7B-Chat", "experiment_results_Exp1_Qwen-MoE.json", "Qwen0.5B_list_b_confident_invalid.json"),
    ]

    for model_name, res_file, lb_file in tqdm(models_to_process):
        res_path = os.path.join(results_dir, res_file)
        lb_path = os.path.join(reclaimed_dir, lb_file)
        
        if not os.path.exists(res_path):
            print(f"Skipping {model_name}: Result file not found.")
            continue
            
        try:
            process_model(model_name, res_path, lb_path, output_dir)
        except Exception as e:
            print(f"Failed to process {model_name}: {e}")
            import traceback
            traceback.print_exc()
