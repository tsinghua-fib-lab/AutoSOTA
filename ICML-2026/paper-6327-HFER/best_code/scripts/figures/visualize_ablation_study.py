import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

RESULTS_FILE = "data/results/ablation_results_phi35_mini.json"
OUTPUT_DIR = "output"
OUTPUT_FILE_PNG = os.path.join(OUTPUT_DIR, "ablation_layer_plot_Phi-3.5-mini.png")
OUTPUT_FILE_PDF = os.path.join(OUTPUT_DIR, "ablation_layer_plot_Phi-3.5-mini.pdf")

def main():
    if not os.path.exists(RESULTS_FILE):
        print("Ablation results not found.")
        return
        
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    with open(RESULTS_FILE, 'r') as f:
        data = json.load(f)
        
    # data: list of {k, trajectory: [ {layer, fiedler_value...} ]}
    
    # 4 metrics
    metrics = ["fiedler_value", "entropy", "hfer", "smoothness"]
    titles = {
        "fiedler_value": "Fiedler Value (Structure)",
        "entropy": "Spectral Entropy (Complexity)",
        "hfer": "HFER (High Freq Energy)",
        "smoothness": "Smoothness (Proxy)"
    }
    
    # Setup plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Color palette
    ks = sorted([d['k'] for d in data])
    # Use a sequential palette
    palette = sns.color_palette("coolwarm", len(ks))
    
    max_layer = len(data[0]['trajectory'])
    layers = range(max_layer)
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        for j, entry in enumerate(data):
            k = entry['k']
            traj = entry['trajectory']
            
            # Extract y
            y = [t[metric] for t in traj]
            
            label = f"Ablated {k} Heads" if k > 0 else "Baseline (Valid)"
            style = "-" if k == 0 else "--"
            width = 3 if k == 0 else 1.5
            
            ax.plot(layers, y, label=label, color=palette[j], linestyle=style, linewidth=width)
            
        ax.set_title(titles[metric], fontsize=14, fontweight='bold')
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Metric Value")
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Add induction head zone shading? (L15-L25 typically)
        # ax.axvspan(15, 25, color='gray', alpha=0.1, label="Induction Zone")
    
    # Legend on first plot or outside?
    axes[0].legend(loc='upper left', fontsize='small', framealpha=0.9)
    
    plt.suptitle("Impact of Induction Head Ablation on Spectral Signature (Phi-3.5-mini)", fontsize=18)
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE_PNG, dpi=300)
    plt.savefig(OUTPUT_FILE_PDF)
    print(f"Saved {OUTPUT_FILE_PNG} and {OUTPUT_FILE_PDF}")

if __name__ == "__main__":
    main()
