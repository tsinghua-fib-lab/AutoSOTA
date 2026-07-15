import json
import os
import numpy as np
import pandas as pd
from scipy.stats import sem

RESULTS_FILE = "data/results/experiment_results_Exp1_Qwen-MoE.json"
TAXONOMY_FILE = "data/experiment_ready/taxonomy.json"
OUTPUT_DIR = "data/paper_figures/main_plots"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "Figure6_Taxonomy.tex")

def cohen_d(x, y):
    if len(x) < 2 or len(y) < 2: return 0.0
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

def get_vals(items, layer, metric):
    vals = []
    for item in items:
        traj = item.get('trajectory', [])
        if layer < len(traj):
            v = traj[layer].get(metric)
            if v is not None: vals.append(v)
    return vals

def main():
    if not os.path.exists(RESULTS_FILE) or not os.path.exists(TAXONOMY_FILE):
        print("Results or Taxonomy file not found yet.")
        return

    with open(RESULTS_FILE, 'r') as f:
        data = json.load(f)
    with open(TAXONOMY_FILE, 'r') as f:
        taxonomy = json.load(f)

    valid_items = data['valid']
    invalid_items = data['invalid']

    logic_items = []
    calc_items = []

    for item in invalid_items:
        fname = os.path.basename(item['file'])
        cat = taxonomy.get(fname, "Logic")
        if cat == "Logic" or cat == "Logic_Incomplete":
            logic_items.append(item)
        elif cat == "Calc":
            calc_items.append(item)

    print(f"Valid: {len(valid_items)}")
    print(f"Logic: {len(logic_items)}")
    print(f"Calc: {len(calc_items)}")

    # Find best metric/layer for Logic separation
    best_config = None
    best_d = 0
    
    metrics = ["hfer", "fiedler_value", "smoothness", "entropy", "energy"]
    num_layers = len(valid_items[0]['trajectory']) if valid_items else 0

    for m in metrics:
        for l in range(num_layers):
            v = get_vals(valid_items, l, m)
            i_logic = get_vals(logic_items, l, m)
            
            if not v or not i_logic: continue
            
            d = abs(cohen_d(v, i_logic))
            if d > best_d:
                best_d = d
                best_config = (m, l)

    if not best_config:
        print("No valid config found.")
        return

    m, l = best_config
    print(f"Best Config: {m} @ L{l} (d={best_d:.2f})")
    
    # Calculate d for both groups at this config
    v = get_vals(valid_items, l, m)
    i_logic = get_vals(logic_items, l, m)
    i_calc = get_vals(calc_items, l, m)
    
    d_logic = abs(cohen_d(v, i_logic))
    d_calc = abs(cohen_d(v, i_calc))
    
    print(f"Logic d: {d_logic:.2f}")
    print(f"Calc d: {d_calc:.2f}")

    # Generate PGFPlots Bar Chart
    csv_str = f"category,d\nLogic Error,{d_logic:.2f}\nCalculation Error,{d_calc:.2f}"
    
    tex = fR"""\documentclass[tikz,border=10pt]{{standalone}}
\usepackage{{pgfplots}}
\pgfplotsset{{compat=1.18}}

\begin{{document}}
\begin{{tikzpicture}}
\pgfplotstableread[row sep=newline, col sep=comma]{{
{csv_str}
}}\mydata

\begin{{axis}}[
    ybar,
    bar width=1cm,
    width=6cm, height=5cm,
    ylabel={{$|Cohen's d|$ (Signal Strength)}},
    xtick=data,
    xticklabels from table={{\mydata}}{{category}},
    nodes near coords,
    ymin=0, ymax={max(d_logic, d_calc)*1.2:.2f},
    grid=major,
    grid style={{dashed, gray!30}},
    title={{The Spectral Signature of Hallucination}}
]
\addplot [fill=purple!60] table [x expr=\coordindex, y=d] {{\mydata}};
\end{{axis}}
\end{{tikzpicture}}
\end{{document}}
"""
    with open(OUTPUT_FILE, 'w') as f:
        f.write(tex)
    print(f"Generated {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
