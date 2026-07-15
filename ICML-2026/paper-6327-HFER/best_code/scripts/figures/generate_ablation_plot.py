import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

RESULTS_FILE = "data/results/ablation_results.json"
OUTPUT_DIR = "data/paper_figures/main_plots"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "Figure8_InductionAblation.tex")

def main():
    if not os.path.exists(RESULTS_FILE):
        print("Ablation results not found.")
        return
        
    with open(RESULTS_FILE, 'r') as f:
        data = json.load(f)
        
    # Extract data
    ks = [d['k'] for d in data]
    # user wants "Fiedler Value (Low = Good)".
    # If we ablate, Fiedler should spike "High Freq Noise".
    # So we plot Fiedler.
    fiedlers = [d['fiedler_avg'] for d in data]
    
    # Or specifically Fiedler of a relevant layer?
    # Let's peek at the layer_fiedlers in the first item to pick a representative one.
    # Usually L15-20 is where induction heads are.
    # Let's stick to Average Fiedler for robustness, or plot multiple lines?
    # User said "The Fiedler Value should spike". Simple 1D plot is best.
    
    # Create CSV for PGFPlots
    df = pd.DataFrame({'k': ks, 'fiedler': fiedlers})
    csv_str = df.to_csv(None, index=False, lineterminator='\n')
    
    tex = fR"""\documentclass[tikz,border=10pt]{{standalone}}
\usepackage{{pgfplots}}
\pgfplotsset{{compat=1.18}}

\begin{{document}}
\begin{{tikzpicture}}
\pgfplotstableread[row sep=newline, col sep=comma]{{
{csv_str}
}}\mydata

\begin{{axis}}[
    width=8cm, height=6cm,
    xlabel={{Number of Ablated Induction Heads}},
    ylabel={{Fiedler Value (Spectral Roughness)}},
    title={{Impact of Kill-Switch on Spectral Signature}},
    grid=major,
    grid style={{dashed, gray!30}},
    mark=*,
    mark options={{fill=red}},
    thick,
    red
]
\addplot table [x=k, y=fiedler] {{\mydata}};
\node[anchor=south west] at (axis cs: 5, {min(fiedlers)}) {{Baseline (0)}};
\end{{axis}}
\end{{tikzpicture}}
\end{{document}}
"""
    with open(OUTPUT_FILE, 'w') as f:
        f.write(tex)
    print(f"Generated {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
