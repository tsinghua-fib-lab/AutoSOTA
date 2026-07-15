
import json
import os
import subprocess
import numpy as np


# Map model names to their results files and output prefixes
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
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ... (Template remains same) ...

LATEX_TEMPLATE = r"""\documentclass[tikz,border=10pt]{standalone}
\usepackage{pgfplots}
\usepgfplotslibrary{groupplots}
\usepgfplotslibrary{fillbetween}
\pgfplotsset{compat=1.18}

\begin{document}
\begin{tikzpicture}
\begin{groupplot}[
    group style={
        group size=2 by 2,
        horizontal sep=2cm,
        vertical sep=2cm
    },
    width=8cm, height=6cm,
    xlabel={Layer},
    grid=major,
    legend pos=north west,
    legend style={font=\tiny, fill=none, draw=none},
    tick label style={font=\scriptsize},
    label style={font=\small},
    title style={font=\normalsize, yshift=-0.5ex}
]

    % Plot 1: Fiedler Value
    \nextgroupplot[title={Fiedler Value ($\lambda_2$)}, ylabel={$\lambda_2$}]
    \pgfplotstableread[row sep=newline, col sep=comma]{
DATA_FIEDLER
    }\datafiedler
    PLOTS_FIEDLER

    % Plot 2: Spectral Entropy
    \nextgroupplot[title={Spectral Entropy ($H$)}, ylabel={$H$}]
    \pgfplotstableread[row sep=newline, col sep=comma]{
DATA_ENTROPY
    }\dataentropy
    PLOTS_ENTROPY

    % Plot 3: HFER
    \nextgroupplot[title={HFER}, ylabel={Ratio}]
    \pgfplotstableread[row sep=newline, col sep=comma]{
DATA_HFER
    }\datahfer
    PLOTS_HFER

    % Plot 4: Smoothness
    \nextgroupplot[title={Smoothness ($\eta$)}, ylabel={$\eta$}, legend pos=south west]
    \pgfplotstableread[row sep=newline, col sep=comma]{
DATA_SMOOTHNESS
    }\datasmoothness
    PLOTS_SMOOTHNESS

\end{groupplot}
\end{tikzpicture}
\end{document}
"""

def generate_csv_block(results, metric_key):
    """Generates the CSV data block for pgfplots."""
    # Header: layer, k0, k5, k10, ...
    header = ["layer"]
    ablation_steps = sorted([int(k) for k in results.keys() if k.isdigit()])
    for k in ablation_steps:
        header.append(f"k{k}")
    
    csv_lines = [",".join(header)]
    
    # Assuming all steps have same number of layers
    num_layers = len(results[str(ablation_steps[0])])
    
    for l in range(num_layers):
        row = [str(l)]
        for k in ablation_steps:
            # Extract specific metric for layer l
            # Structure: results[k] is a list of layers, each layer dict has metrics
            layer_data = results[str(k)][l]
            val = layer_data.get(metric_key, 0.0)
            if val is None: val = 0.0
            row.append(str(val))
        csv_lines.append(",".join(row))
        
    return "\n".join(csv_lines)

def generate_plot_commands(results, data_macro_name):
    """Generates the \addplot commands."""
    commands = []
    ablation_steps = sorted([int(k) for k in results.keys() if k.isdigit()])
    
    # Define a color cycle or gradient manually if needed, or use cycle list
    # pgfplots has default cycle lists, but we can be explicit for clarity
    # Using 'viridis' style progression manually is hard in pure tex without defining colours.
    # Let's rely on standard color cycle but maybe dashed for higher k?
    
    for i, k in enumerate(ablation_steps):
        # We only plot a subset to avoid clutter if there are too many (e.g. 0, 10, 20, 30)
        # But user asked for "combined plot", usually implies showing the trend.
        # Let's plot 0, 5, 10, 20, 30 if available.
        
        # Color mixing calculation for gradient effect (Blue to Red)
        ratio = i / (len(ablation_steps) - 1) if len(ablation_steps) > 1 else 0
        red = int(ratio * 100)
        blue = int((1 - ratio) * 100)
        color_str = f"color=red!{red}!blue"
        
        entry = f"\\addplot+[no markers, {color_str}, thick] table [x=layer, y=k{k}] {{\\{data_macro_name}}};"
        entry += f" \\addlegendentry{{$k={k}$}}"
        commands.append(entry)
        
    return "\n    ".join(commands)

def main():
    print("Generating LaTeX ablation plots...")
    
    # Check for pdflatex
    try:
        subprocess.run(["pdflatex", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        has_pdflatex = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: pdflatex not found. Will generate .tex files only.")
        has_pdflatex = False

    for model_name, json_path in MODEL_MAP.items():
        if not os.path.exists(json_path):
            print(f"Skipping {model_name}: {json_path} not found.")
            continue
            
        print(f"Processing {model_name}...")
        with open(json_path, 'r') as f:
            raw_data = json.load(f)
            
        # Parse Data
        results = {}
        if isinstance(raw_data, list):
            for item in raw_data:
                # standard format: {'k': step, 'trajectory': [...]}
                if 'k' in item and 'trajectory' in item:
                    results[str(item['k'])] = item['trajectory']
                else:
                    # fallback format: {step: [...], ...}
                    for k, v in item.items():
                        results[str(k)] = v
        elif isinstance(raw_data, dict):
            # If it's a single dict of steps
            results = raw_data
            
        # Prepare Data Blocks
        csv_fiedler = generate_csv_block(results, "fiedler_value")
        csv_entropy = generate_csv_block(results, "spectral_entropy")
        csv_hfer = generate_csv_block(results, "hfer")
        csv_smooth = generate_csv_block(results, "smoothness") # Assuming key exists
        
        # Prepare Plot Commands
        plots_fiedler = generate_plot_commands(results, "datafiedler")
        plots_entropy = generate_plot_commands(results, "dataentropy")
        plots_hfer = generate_plot_commands(results, "datahfer")
        plots_smooth = generate_plot_commands(results, "datasmoothness")
        
        # Fill Template
        tex_content = LATEX_TEMPLATE.replace("DATA_FIEDLER", csv_fiedler)
        tex_content = tex_content.replace("PLOTS_FIEDLER", plots_fiedler)
        
        tex_content = tex_content.replace("DATA_ENTROPY", csv_entropy)
        tex_content = tex_content.replace("PLOTS_ENTROPY", plots_entropy)
        
        tex_content = tex_content.replace("DATA_HFER", csv_hfer)
        tex_content = tex_content.replace("PLOTS_HFER", plots_hfer)
        
        tex_content = tex_content.replace("DATA_SMOOTHNESS", csv_smooth)
        tex_content = tex_content.replace("PLOTS_SMOOTHNESS", plots_smooth)
        
        # Write .tex file
        tex_filename = f"combined_plot_latex_{model_name}.tex"
        tex_path = os.path.join(OUTPUT_DIR, tex_filename)
        with open(tex_path, 'w') as f:
            f.write(tex_content)
            
        print(f"Saved {tex_path}")
        
        # Compile if possible
        if has_pdflatex:
            try:
                # Run pdflatex in the output directory to keep clutter there
                subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", tex_filename], 
                    cwd=OUTPUT_DIR, 
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                pdf_filename = tex_filename.replace(".tex", ".pdf")
                print(f"Compiled {pdf_filename}")
                
            except subprocess.CalledProcessError as e:
                print(f"Error compiling {tex_filename}: {e}")

if __name__ == "__main__":
    main()
