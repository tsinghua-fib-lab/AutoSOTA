import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Mapping: Model Name (for Title/Filename) -> Result JSON
MODEL_MAP = {
    "Llama-3.1-8B": "data/results/ablation_results.json",
    "Llama-3.2-3B": "data/results/ablation_results_3b.json", 
    "Llama-3.2-1B": "data/results/ablation_results_llama3_1b.json",
    "Mistral-7B": "data/results/ablation_results_mistral7b.json",
    "Qwen-0.5B": "data/results/ablation_results_qwen0.5b.json",
    "Qwen-7B": "data/results/ablation_results_qwen2.5_7b.json", 
    "Qwen-MoE": "data/results/ablation_results_qwen_moe.json",
    "Phi-3.5": "data/results/ablation_results_phi35_mini.json"
}

OUTPUT_DIR = "output"

def generate_plot(model_name, json_path):
    if not os.path.exists(json_path):
        print(f"Skipping {model_name}: {json_path} not found.")
        return

    print(f"Generating plot for {model_name}...")
    
    with open(json_path, 'r') as f:
        raw_data = json.load(f)

    # Normalize Data to List of Dicts [{'k': step, 'trajectory': ...}]
    data = []
    if isinstance(raw_data, list):
        # Check uniqueness of format
        if 'k' in raw_data[0] and 'trajectory' in raw_data[0]:
            data = raw_data
        else:
            # Maybe list of single-entry dicts? merging
            merged = {}
            for item in raw_data:
                merged.update(item)
            # Now dict to list
            for k, v in merged.items():
                data.append({'k': int(k), 'trajectory': v})
    elif isinstance(raw_data, dict):
        for k, v in raw_data.items():
            if k.isdigit():
                 data.append({'k': int(k), 'trajectory': v})
    
    # Sort by k
    data.sort(key=lambda x: x['k'])

    if not data:
        print(f"Error: No valid data found for {model_name}")
        return

    # Metrics
    metrics = ["fiedler_value", "entropy", "hfer", "smoothness"]
    titles = {
        "fiedler_value": "Fiedler Value (Structure)",
        "entropy": "Spectral Entropy (Complexity)",
        "hfer": "HFER (High Freq Energy)",
        "smoothness": "Smoothness (Proxy)"
    }
    
    # Setup plot (High DPI, Seaborn Paper context for LaTeX-like look)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10)) # Slightly smaller, tight layout
    axes = axes.flatten()
    
    # Palette
    ks = [d['k'] for d in data]
    palette = sns.color_palette("viridis", len(ks)) # Viridis is scientific standard
    
    # Find max layer
    max_layer = len(data[0]['trajectory'])
    layers = range(max_layer)
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        for j, entry in enumerate(data):
            k = entry['k']
            traj = entry['trajectory']
            
            # Robustness check
            y = [t[metric] for t in traj]
            min_len = min(len(layers), len(y))
            
            label = f"Ablated {k} Heads" if k > 0 else "Baseline"
            # Highlight extreme values
            if k == 0:
                style = "-"
                width = 2.5
                alpha = 1.0
                color = 'black' # Baseline black
            elif k == max(ks):
                style = "-"
                width = 2.0
                alpha = 1.0
                color = 'red' # Max ablation red
            else:
                style = "-"
                width = 1.0 if len(ks) > 5 else 1.5
                alpha = 0.6
                color = palette[j]

            ax.plot(layers[:min_len], y[:min_len], label=label, color=color, linestyle=style, linewidth=width, alpha=alpha)
            
        ax.set_title(titles[metric], fontsize=14, fontweight='bold')
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Metric Value")
        ax.grid(True, linestyle=':', alpha=0.6)
    
    # Legend (only on first plot to avoid clutter, or outside?)
    # Let's put it on the HFER plot (usually cleaner) or just 1st
    axes[0].legend(loc='best', fontsize='x-small', framealpha=0.9, ncol=2)
    
    plt.suptitle(f"{model_name}: Induction Head Ablation", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0.0, 1, 0.96])
    
    # Save with LateX-matching filename
    # The user asked for "corresponding to these .tex", so we use the exact same base name.
    # The .tex are named: combined_plot_latex_{model}.tex
    
    filename_base = f"combined_plot_latex_{model_name}"
    png_path = os.path.join(OUTPUT_DIR, f"{filename_base}.png")
    pdf_path = os.path.join(OUTPUT_DIR, f"{filename_base}.pdf")
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"Saved {png_path} and {pdf_path}")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    for name, path in MODEL_MAP.items():
        try:
            generate_plot(name, path)
        except Exception as e:
            print(f"Failed to generate for {name}: {e}")

if __name__ == "__main__":
    main()
