import json
import os
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, mannwhitneyu, ttest_ind

# Configuration
RESULTS_DIR = "data/results"
RECLAIMED_DIR = "data/reclaimed"
OUTPUT_DIR = "data/paper_figures/main_plots"

# Map friendly model names to filenames
MODEL_FILES = {
    "Llama-3.1-8B": ("experiment_results_Meta-Llama-3.1-8B-Instruct.json", "8B_list_b_confident_invalid.json"),
    "Phi-3.5-mini": ("experiment_results_Phi-3.5-mini-instruct.json", "Phi3.5_list_b_confident_invalid.json"),
    "Mistral-7B": ("experiment_results_Mistral-7B-v0.1.json", "Mistral7B_list_b_confident_invalid.json"),
    "Qwen2.5-0.5B": ("experiment_results_Qwen2.5-0.5B-Instruct.json", "Qwen0.5B_list_b_confident_invalid.json"),
    "Llama-1B": ("experiment_results_Llama-3.2-1B-Instruct.json", "1B_list_b_confident_invalid.json"), # For Fig 4 if needed, mostly hardcoded
    "Qwen-MoE": ("experiment_results_Exp1_Qwen-MoE.json", "Qwen0.5B_list_b_confident_invalid.json"), # Using Qwen0.5B reclaimed list as proxy for dataset split
}

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def load_and_relabel(model_key):
    res_file, lb_file = MODEL_FILES[model_key]
    res_path = os.path.join(RESULTS_DIR, res_file)
    lb_path = os.path.join(RECLAIMED_DIR, lb_file)
    
    with open(res_path, 'r') as f:
        data = json.load(f)
        
    reclaimed_files = set()
    if os.path.exists(lb_path):
        with open(lb_path, 'r') as f:
            raw = json.load(f)
            if raw and isinstance(raw[0], dict):
                reclaimed_files = set(item['file'] for item in raw)
            else:
                reclaimed_files = set(raw)
    
    valid_items = data['valid'][:]
    invalid_items = []
    
    for item in data['invalid']:
        if item['file'] in reclaimed_files:
            valid_items.append(item)
        else:
            invalid_items.append(item)
            
    return valid_items, invalid_items

def get_metric_data(items, layer, metric):
    vals = []
    for item in items:
        traj = item.get('trajectory', [])
        if layer < len(traj):
            v = traj[layer].get(metric)
            if v is not None:
                vals.append(v)
    return vals

def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

def generate_kde_csv(v_vals, i_vals, points=200):
    min_val = min(min(v_vals), min(i_vals))
    max_val = max(max(v_vals), max(i_vals))
    margin = (max_val - min_val) * 0.2
    x_grid = np.linspace(min_val - margin, max_val + margin, points)
    
    kde_v = gaussian_kde(v_vals)(x_grid)
    kde_i = gaussian_kde(i_vals)(x_grid)
    
    return pd.DataFrame({'x': x_grid, 'y_valid': kde_v, 'y_invalid': kde_i})

# -----------------------------------------------------------------------------
# Figure 2: The Shape of Truth (Hero)
# -----------------------------------------------------------------------------
def generate_figure_2():
    print("Generating Figure 2: The Shape of Truth...")
    configs = [
        ("Llama-3.1-8B", "hfer", 30, "(a) Llama-3.1-8B (HFER @ L30)", True), # True = Inverted (d negative) -> Valid on left usually means lower value for HFER? 
        # Wait, for HFER, valid is lower (d=-3.00 implies valid < invalid). 
        # For Smoothness, valid is higher (d positive).
        # We want to standardize visualization? Users prompt: valid [blue], invalid [red]. 
        # Just plot them as is, the color code handles identity.
        ("Phi-3.5-mini", "smoothness", 25, "(b) Phi-3.5-mini (Smoothness @ L25)", False),
        ("Mistral-7B", "smoothness", 26, "(c) Mistral-7B (Smoothness @ L26)", False),
        ("Qwen2.5-0.5B", "entropy", 0, "(d) Qwen2.5-0.5B (Entropy @ L0)", False)
    ]
    
    plots_code = ""
    
    for model, metric, layer, title, inverted in configs:
        valid_items, invalid_items = load_and_relabel(model)
        v_vals = get_metric_data(valid_items, layer, metric)
        i_vals = get_metric_data(invalid_items, layer, metric)
        
        d = cohen_d(v_vals, i_vals)
        stat, p_mw = mannwhitneyu(v_vals, i_vals)
        t_stat, p_t = ttest_ind(v_vals, i_vals, equal_var=False)
        
        df = generate_kde_csv(v_vals, i_vals)
        csv_str = df.to_csv(None, index=False, lineterminator='\n')
        
        macro_name = f"\\mydata{chr(65+list(MODEL_FILES.keys()).index(model) if model in MODEL_FILES else 65)}"
        # Actually simplest is just to make a safe slug
        safe_slug = "".join([c for c in model if c.isalpha()]) + "FigTwo"
        macro_name = f"\\mydata{safe_slug}"

        plots_code += fR"""
    \nextgroupplot[
        title={{{title}}},
        xlabel={{Value}},
        ylabel={{Density}},
        grid=major,
        grid style={{dashed, gray!30}}
    ]
    \pgfplotstableread[row sep=newline, col sep=comma]{{
{csv_str}
    }}{macro_name}
    
    \addplot [blue, fill=blue!10, area style] table [x=x, y=y_valid] {{{macro_name}}};
    \addplot [red, fill=red!10, area style] table [x=x, y=y_invalid] {{{macro_name}}};
    \node[anchor=north east, font=\tiny, align=right] at (rel axis cs: 0.95, 0.95) {{
        $d={d:.2f}$\\
        $p_{{MW}} < 10^{{{int(np.log10(p_mw))}}}$
    }};
"""

    tex = fR"""\documentclass[tikz,border=10pt]{{standalone}}
\usepackage{{pgfplots}}
\usepgfplotslibrary{{groupplots}}
\pgfplotsset{{compat=1.18}}

\begin{{document}}
\begin{{tikzpicture}}
\begin{{groupplot}}[
    group style={{
        group size=2 by 2,
        horizontal sep=2cm,
        vertical sep=2cm
    }},
    width=7cm, height=5cm,
    ymin=0
]
{plots_code}
\end{{groupplot}}
% Legend
\node[anchor=north] at ($(group c1r2.south)!0.5!(group c2r2.south) + (0,-1.5cm)$) {{
    \begin{{tikzpicture}}
        \draw[blue, fill=blue!10] (0,0) rectangle (0.5,0.3) node[right, black] at (0.5, 0.15) {{Valid}};
        \draw[red, fill=red!10] (1.5,0) rectangle (2.0,0.3) node[right, black] at (2.0, 0.15) {{Invalid}};
    \end{{tikzpicture}}
}};
\end{{tikzpicture}}
\end{{document}}
"""
    with open(os.path.join(OUTPUT_DIR, "Figure2_ShapeOfTruth.tex"), "w") as f:
        f.write(tex)

# -----------------------------------------------------------------------------
# Figure 3: Layer-wise Evolution
# -----------------------------------------------------------------------------
def generate_figure_3():
    print("Generating Figure 3: Layer Evolution...")
    model = "Llama-3.1-8B"
    metric = "hfer"
    
    valid_items, invalid_items = load_and_relabel(model)
    
    # Check max layer
    max_layer = 0
    if valid_items: max_layer = len(valid_items[0]['trajectory'])
    
    stats = []
    for l in range(max_layer):
        v = get_metric_data(valid_items, l, metric)
        i = get_metric_data(invalid_items, l, metric)
        if v and i:
            stats.append({
                'layer': l,
                'v_mean': np.mean(v), 'v_std': np.std(v),
                'i_mean': np.mean(i), 'i_std': np.std(i)
            })
    
    df = pd.DataFrame(stats)
    df['v_upper'] = df['v_mean'] + df['v_std']
    df['v_lower'] = df['v_mean'] - df['v_std']
    df['i_upper'] = df['i_mean'] + df['i_std']
    df['i_lower'] = df['i_mean'] - df['i_std']
    
    csv_str = df.to_csv(None, index=False, lineterminator='\n')
    
    tex = fR"""\documentclass[tikz,border=10pt]{{standalone}}
\usepackage{{pgfplots}}
\usepgfplotslibrary{{fillbetween}}
\pgfplotsset{{compat=1.18}}

\begin{{document}}
\begin{{tikzpicture}}
\pgfplotstableread[row sep=newline, col sep=comma]{{
{csv_str}
}}\mydata

\begin{{axis}}[
    width=10cm, height=6cm,
    xlabel={{Layer}},
    ylabel={{HFER}},
    title={{Llama-3.1-8B Spectral Evolution}},
    grid=major,
    grid style={{dashed, gray!30}}
]

\addplot [name path=v_upper, draw=none, forget plot] table [x=layer, y=v_upper] {{\mydata}};
\addplot [name path=v_lower, draw=none, forget plot] table [x=layer, y=v_lower] {{\mydata}};
\addplot [blue!20] fill between [of=v_upper and v_lower];

\addplot [name path=i_upper, draw=none, forget plot] table [x=layer, y=i_upper] {{\mydata}};
\addplot [name path=i_lower, draw=none, forget plot] table [x=layer, y=i_lower] {{\mydata}};
\addplot [red!20] fill between [of=i_upper and i_lower];

\addplot [blue, thick] table [x=layer, y=v_mean] {{\mydata}};
\addlegendentry{{Valid}}
\addplot [red, thick] table [x=layer, y=i_mean] {{\mydata}};
\addlegendentry{{Invalid}}

\end{{axis}}
\end{{tikzpicture}}
\end{{document}}
"""
    with open(os.path.join(OUTPUT_DIR, "Figure3_Evolution.tex"), "w") as f:
        f.write(tex)

# -----------------------------------------------------------------------------
# Figure 4: Cross-Architecture Bar Chart
# -----------------------------------------------------------------------------
def generate_figure_4():
    print("Generating Figure 4: Bar Chart...")
    # Data from recalc_full_stats.py findings
    data = [
        ("Phi-3.5-mini", 3.30),
        ("Llama-3.2-1B", 3.02), # Actually table said 3.02 for Fiedler, 3.00 for HFER. Let's strictly use HFER values or best ones.
        # Recalc table:
        # Llama-1B: HFER L9 d=-1.79... wait, recalculate says HFER L0 d=-3.00. Use 3.00.
        # User prompt says 3.02. I will stick to the user's provided numbers in the prompt for consistency with their request.
        ("Llama-3.1-8B", 3.00),
        ("Llama-3.2-3B", 2.97),
        ("Qwen2.5-0.5B", 2.93),
        ("Qwen-MoE", 2.73),
        ("Qwen2.5-7B", 2.43),
        ("Mistral-7B", 2.09)
    ]
    
    # CSV
    csv_str = "model,d\n" + "\n".join([f"{n},{v}" for n,v in data])
    
    tex = fR"""\documentclass[tikz,border=10pt]{{standalone}}
\usepackage{{pgfplots}}
\pgfplotsset{{compat=1.18}}

\begin{{document}}
\begin{{tikzpicture}}
\pgfplotstableread[row sep=newline, col sep=comma]{{
{csv_str}
}}\mydata

\begin{{axis}}[
    xbar,
    width=8cm, height=6cm,
    xlabel={{$|Cohen's d|$}},
    ytick=data,
    yticklabels from table={{\mydata}}{{model}},
    nodes near coords,
    nodes near coords align={{horizontal}},
    xmax=4.0,
    grid=major,
    grid style={{dashed, gray!30}}
]
\addplot [fill=gray!80] table [x=d, y expr=\coordindex] {{\mydata}};
\draw [red, dashed, thick] (axis cs: 0.8, -1) -- (axis cs: 0.8, 7);
\node [red, anchor=south, rotate=90] at (axis cs: 0.85, 3) {{Large Effect (0.8)}};
\end{{axis}}
\end{{tikzpicture}}
\end{{document}}
"""
    with open(os.path.join(OUTPUT_DIR, "Figure4_EffectSizes.tex"), "w") as f:
        f.write(tex)

# -----------------------------------------------------------------------------
# Figure 5: Mistral Comparison
# -----------------------------------------------------------------------------
def generate_figure_5():
    print("Generating Figure 5: Mistral Comparison...")
    # 2x2: Llama HFER, Mistral HFER / Llama Smoothness, Mistral Smoothness
    
    configs = [
        ("Llama-3.1-8B", "hfer", 30, "Llama-8B (HFER)"),
        ("Mistral-7B", "hfer", 11, "Mistral-7B (HFER)"), # Best HFER for Mistral is L11
        ("Llama-3.1-8B", "smoothness", 30, "Llama-8B (Smoothness)"), # Using same L30 for fair comparison? Or L9 where it was also good? Let's use L30 unless L9 is significantly better.
        # Actually user said "Smooth: d=+2.11" for Llama. Let's find max smoothness for Llama.
        # I'll just load Llama and find max smoothness layer dynamically or pick a representative deeper layer.
        # Let's assume L30 for simplicity or check prompt. "Smooth: d=+2.11".
        ("Mistral-7B", "smoothness", 26, "Mistral-7B (Smoothness)")
    ]
    
    plots_code = ""
    
    for i, (model, metric, layer, title) in enumerate(configs):
        valid_items, invalid_items = load_and_relabel(model)
        v_vals = get_metric_data(valid_items, layer, metric)
        i_vals = get_metric_data(invalid_items, layer, metric)
        
        d = cohen_d(v_vals, i_vals)
        
        df = generate_kde_csv(v_vals, i_vals)
        csv_str = df.to_csv(None, index=False, lineterminator='\n')
        
        safe_slug = "".join([c for c in model if c.isalpha()]) + f"FigFive{i}"
        macro_name = f"\\mydata{safe_slug}"
        
        plots_code += fR"""
    \nextgroupplot[
        title={{{title}}},
        xlabel={{Value}},
        ylabel={{Density}},
        grid=major,
        grid style={{dashed, gray!30}}
    ]
    \pgfplotstableread[row sep=newline, col sep=comma]{{
{csv_str}
    }}{macro_name}
    
    \addplot [blue, fill=blue!10, area style] table [x=x, y=y_valid] {{{macro_name}}};
    \addplot [red, fill=red!10, area style] table [x=x, y=y_invalid] {{{macro_name}}};
    \node[anchor=north east, font=\small] at (rel axis cs: 0.95, 0.95) {{
        $d={d:.2f}$
    }};
"""

    tex = fR"""\documentclass[tikz,border=10pt]{{standalone}}
\usepackage{{pgfplots}}
\usepgfplotslibrary{{groupplots}}
\pgfplotsset{{compat=1.18}}

\begin{{document}}
\begin{{tikzpicture}}
\begin{{groupplot}}[
    group style={{
        group size=2 by 2,
        horizontal sep=2cm,
        vertical sep=2cm
    }},
    width=6cm, height=4cm,
    ymin=0
]
{plots_code}
\end{{groupplot}}
\end{{tikzpicture}}
\end{{document}}
"""
    with open(os.path.join(OUTPUT_DIR, "Figure5_MistralEffect.tex"), "w") as f:
        f.write(tex)

# -----------------------------------------------------------------------------
# Figure 7: MoE Spectral Signature
# -----------------------------------------------------------------------------
def generate_figure_7():
    print("Generating Figure 7: MoE Spectral Signature...")
    model = "Qwen-MoE"
    metric = "smoothness"
    layer = 6
    
    valid_items, invalid_items = load_and_relabel(model)
    v_vals = get_metric_data(valid_items, layer, metric)
    i_vals = get_metric_data(invalid_items, layer, metric)
    
    # Filter none
    v_vals = [x for x in v_vals if x is not None]
    i_vals = [x for x in i_vals if x is not None]
    
    d = cohen_d(v_vals, i_vals)
    stat, p_mw = mannwhitneyu(v_vals, i_vals)
    
    df = generate_kde_csv(v_vals, i_vals)
    csv_str = df.to_csv(None, index=False, lineterminator='\n')
    
    safe_slug = "QwenMoEFigSeven"
    macro_name = f"\\mydata{safe_slug}"
    
    tex = fR"""\documentclass[tikz,border=10pt]{{standalone}}
\usepackage{{pgfplots}}
\pgfplotsset{{compat=1.18}}

\begin{{document}}
\begin{{tikzpicture}}
\pgfplotstableread[row sep=newline, col sep=comma]{{
{csv_str}
}}{macro_name}

\begin{{axis}}[
    width=8cm, height=6cm,
    title={{Qwen1.5-MoE-A2.7B (Smoothness @ L6)}},
    xlabel={{Smoothness}},
    ylabel={{Density}},
    grid=major,
    grid style={{dashed, gray!30}}
]

\addplot [blue, fill=blue!10, area style] table [x=x, y=y_valid] {{ {macro_name} }};
\addplot [red, fill=red!10, area style] table [x=x, y=y_invalid] {{ {macro_name} }};

\node[anchor=north east, font=\small] at (rel axis cs: 0.95, 0.95) {{
    $d={d:.2f}$\\
    $p_{{MW}} < 10^{{{int(np.log10(p_mw))}}}$
}};

\legend{{Valid, Invalid}}

\end{{axis}}
\end{{tikzpicture}}
\end{{document}}
"""
    with open(os.path.join(OUTPUT_DIR, "Figure7_MoE.tex"), "w") as f:
        f.write(tex)

if __name__ == "__main__":
    ensure_dir(OUTPUT_DIR)
    generate_figure_2()
    generate_figure_3()
    generate_figure_4()
    generate_figure_5()
    generate_figure_7()
    print("All main figures generated in " + OUTPUT_DIR)
