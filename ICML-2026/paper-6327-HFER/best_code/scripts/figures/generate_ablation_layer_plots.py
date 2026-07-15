import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

RESULTS_FILE = "data/results/ablation_results_phi35_mini.json"
OUTPUT_DIR = "data/paper_figures/main_plots"
FILE_PREFIX = "Figure9_AblationLayers_Phi35Mini"

def main():
    if not os.path.exists(RESULTS_FILE):
        print("Ablation results not found.")
        return
        
    with open(RESULTS_FILE, 'r') as f:
        data = json.load(f)
        
    # data is list of dicts: {'k': int, 'trajectory': list of dicts per layer}
    keys = sorted([d['k'] for d in data])
    metrics = ["fiedler_value", "entropy", "hfer"]
    labels = {
        "fiedler_value": "Fiedler Value ($\\lambda_2$)",
        "entropy": "Spectral Entropy",
        "hfer": "HFER"
    }
    
    # We want one standalone plot per metric, or grouped?
    # Let's generate one combined file with groupplots
    
    # We need to construct CSVs for PGFPlots
    # Structure: layer, k0, k5, k10...
    
    csv_strs = {}
    
    max_layer = 0
    if data:
        max_layer = len(data[0]['trajectory'])
    
    for metric in metrics:
        rows = []
        for l in range(max_layer):
            r = {'layer': l}
            for entry in data:
                k = entry['k']
                # find value at layer l
                val = entry['trajectory'][l][metric]
                r[f"k{k}"] = val
            rows.append(r)
        
        df = pd.DataFrame(rows)
        csv_strs[metric] = df.to_csv(None, index=False, lineterminator='\n')
        
    # Generate LaTeX
    # Color cycle: Blue -> Red.
    # We have len(keys) steps.
    # We can hardcode colors or use a mapped color list.
    # Steps: 0, 5, 10, 15, 20, 25, 30 (7 steps).
    # blue, blue!80!red, blue!60!red, ..., red
    
    colors = []
    nk = len(keys)
    get_col = lambda i: f"blue!{int(100 * (1 - i/(nk-1)))}!red"
    for i in range(nk):
        colors.append(get_col(i))
        
    # Construct Axis for each metric
    plots_code = ""
    for i, metric in enumerate(metrics):
        plots_code += fR"""
    \nextgroupplot[
        title={{{labels[metric]}}},
        xlabel={{Layer}},
        grid=major,
        xmin=0, xmax={max_layer-1},
        transpose legend,
        legend columns=2,
        legend style={{at={{(0.5,-0.25)}},anchor=north}}
    ]
    \pgfplotstableread[row sep=newline, col sep=comma]{{
{csv_strs[metric]}
    }}\datatable

"""
        for j, k in enumerate(keys):
            col_name = f"k{k}"
            color = colors[j]
            # Only add legend to first plot or outside?
            # Let's add leg entries to first plot only? Or use specific legend strategy.
            # Groupplot single legend is tricky.
            # Let's repeat legend or hide it.
            # Actually, user wants to see the gradient.
            leg = f"\\addlegendentry{{$k={k}$}}" if i == 0 else ""
            plots_code += f"    \\addplot [thick, {color}] table [x=layer, y={col_name}] {{\\datatable}}; {leg}\n"

    tex = fR"""\documentclass[tikz,border=10pt]{{standalone}}
\usepackage{{pgfplots}}
\usepgfplotslibrary{{groupplots}}
\pgfplotsset{{compat=1.18}}

\begin{{document}}
\begin{{tikzpicture}}
\begin{{groupplot}}[
    group style={{
        group size=3 by 1,
        horizontal sep=1.8cm,
    }},
    width=6cm, height=5cm,
    ymin=0,
]
{plots_code}
\end{{groupplot}}
\end{{tikzpicture}}
\end{{document}}
"""
    
    outfile = os.path.join(OUTPUT_DIR, f"{FILE_PREFIX}.tex")
    with open(outfile, 'w') as f:
        f.write(tex)
    print(f"Generated {outfile}")

if __name__ == "__main__":
    main()
