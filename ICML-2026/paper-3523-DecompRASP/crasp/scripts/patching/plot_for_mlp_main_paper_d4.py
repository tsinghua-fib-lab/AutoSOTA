from dataclasses import dataclass
import tyro
import json
import yaml
from pathlib import Path
import shutil
import sys
import os
import torch
from patching_data import get_tokenizer_and_dataset_for_task
import re

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

def truncate_colormap(cmap_name, minval=0.0, maxval=1.0, n=100):
    """
    Returns a new colormap consisting of a slice of the original colormap name.
    
    Parameters:
    - cmap_name: string name of the original colormap (e.g., 'viridis')
    - minval: float, start of the range (0.0 to 1.0)
    - maxval: float, end of the range (0.0 to 1.0)
    - n: int, number of discrete colors to sample
    """
    # 1. Get the original colormap object
    cmap = plt.get_cmap(cmap_name)
    
    # 2. Create a list of colors by sampling the original map
    #    from minval to maxval
    new_colors = cmap(np.linspace(minval, maxval, n))
    
    # 3. Create a new colormap object from that list
    new_cmap = colors.LinearSegmentedColormap.from_list(
        f'trunc({cmap_name},{minval:.2f},{maxval:.2f})', 
        new_colors
    )
    return new_cmap

# Add the parent directory to the path to import convert_to_code
sys.path.append(os.path.abspath("."))
from convert_to_code import convert_config_to_code_qk_pruned


TOKEN_TO_TAKE_TOP = {
    "a": 4, 
    "b": 0,
    "<eos>": 0,
}

TOKEN_TO_TAKE_BOTTOM = {
    "a": 0, 
    "b": 0,
    "<eos>": 4,
}

TASK_TO_TOKENS_TO_SHOW = {
    "bce_D4": ["a", "b", "<eos>"],
    "sort": ["9", "10", "11"]
}

def escape_latex_special_chars(text: str) -> str:
    """Escape special LaTeX characters in text."""
    # Escape all LaTeX special characters
    text = text.replace('\\', r'\textbackslash{}')
    text = text.replace('&', r'\&')
    text = text.replace('%', r'\%')
    text = text.replace('$', r'\$')
    text = text.replace('#', r'\#')
    text = text.replace('_', r'\_')
    text = text.replace('{', r'\{')
    text = text.replace('}', r'\}')
    text = text.replace('~', r'\textasciitilde{}')
    text = text.replace('^', r'\textasciicircum{}')
    text = text.replace('@', r'\@')
    return text


VAL_TO_BE_NON_ZERO = 0.05

def get_mlp_input_output_latex_no_bins(path_to_model_outputs, task_name, dest_path):
    """Generate LaTeX code for MLP input-output distributions with heatmaps."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    latex_content = []

    tokenizer, _ = get_tokenizer_and_dataset_for_task(task_name, (0, 150), 150, {"period_for_data":3})
    
    mlp_file = Path(path_to_model_outputs) / "mlp_input_output.pt"
    if not mlp_file.exists():
        return []
    
    mlp_input_output = torch.load(mlp_file, map_location="cpu", weights_only=False)
    if mlp_input_output is None or len(mlp_input_output) == 0:
        return []
    
    # Create mlps folder for saving heatmaps
    if dest_path is not None:
        mlps_folder = dest_path / "mlps"
        mlps_folder.mkdir(parents=True, exist_ok=True)
    else:
        return []  # Can't save heatmaps without dest_path
    
    for selected_path in mlp_input_output:
        if type(selected_path) == str:    # end point is unembed
            # k, vocab, in_vocab
            inp_cache, out_cache = mlp_input_output[selected_path]
            tokens = [tokenizer.vocab_inv[i] for i in range(len(tokenizer))]
            
            tokens_to_show = TASK_TO_TOKENS_TO_SHOW[task_name]

            # Collect all samples from all bins and sort by output logit
            num_bins = len(inp_cache)

            # Collect all samples from all bins and sort by output logit
            all_samples_per_token = {sel_token: [] for sel_token in tokens_to_show}
            num_bins = len(inp_cache)
            for sel_token in tokens_to_show:
                sel_token_id = tokenizer.vocab[sel_token]
                for bin_idx in range(num_bins):
                    bin_inp = torch.tensor(inp_cache[bin_idx])
                    bin_out = out_cache[bin_idx]
                    
                    if bin_inp.numel() == 0:
                        continue
                    
                    num_samples_in_bin = bin_inp.size(0)
                    for sample_idx in range(num_samples_in_bin):
                        all_logit_vals = []
                        for other_sel_token in tokens_to_show:
                            other_sel_token_id = tokenizer.vocab[other_sel_token]
                            all_logit_vals.append(bin_out[sample_idx, sel_token_id, other_sel_token_id].item())
                        logit_val = bin_out[sample_idx, sel_token_id, sel_token_id].item()
                        inp_row = bin_inp[sample_idx, sel_token_id, :].clone()
                        all_samples_per_token[sel_token].append((logit_val, tuple(all_logit_vals), inp_row))
            
            if any([len(all_samples_per_token[t]) == 0 for t in all_samples_per_token]):
                continue

            for sel_token in tokens_to_show:
                filtered_sampels = []
                set_of_vals = {t: set() for t in tokens_to_show}
                set_of_vals["inp"] = set()
                for sample in all_samples_per_token[sel_token]:
                    add_sample = True
                    for t_i, t in enumerate(tokens_to_show):
                        val = f"{sample[1][t_i]:.2f}"
                        if val in set_of_vals[t]:
                            add_sample = False
                        set_of_vals[t].add(val)
                    inp = tuple([f"{v:.2f}" for v in sample[-1]])
                    if inp in set_of_vals["inp"]:
                        add_sample = False
                    set_of_vals["inp"].add(inp)
                    if add_sample:
                        filtered_sampels.append(sample)
                all_samples_per_token[sel_token] = filtered_sampels

            selected_samples = []
            for sel_token in tokens_to_show:
                sel_token_id = tokenizer.vocab[sel_token]
                all_samples_per_token[sel_token].sort(key=lambda x: x[0], reverse=True)
                if TOKEN_TO_TAKE_TOP[sel_token]:
                    top_items = all_samples_per_token[sel_token][:TOKEN_TO_TAKE_TOP[sel_token]]
                    selected_samples.extend([(item[1], item[2])for item in top_items])
                if TOKEN_TO_TAKE_BOTTOM[sel_token]:
                    top_items = all_samples_per_token[sel_token][-TOKEN_TO_TAKE_BOTTOM[sel_token]:]
                    selected_samples.extend([(item[1], item[2])for item in top_items])

            a_id = 0
            b_id = 1
            all_samples_per_two_tokens = all_samples_per_token["b"] + all_samples_per_token["a"]
            all_samples_per_two_tokens.sort(key=lambda x: x[1][b_id] - x[1][a_id], reverse=True)
            top_items = all_samples_per_two_tokens[:4]
            selected_samples.extend([(item[1], item[2])for item in top_items])

            filtered_sampels = []
            set_of_vals = set()
            for sample in selected_samples:
                val = tuple([f"{v:.2f}" for v in sample[0]])
                if val not in set_of_vals:
                    filtered_sampels.append(sample)
                    set_of_vals.add(val)
            selected_samples = filtered_sampels


            selected_samples.sort(key=lambda x: [(item > 0) - (item < 0) for item in x[0]], reverse=True)
                
            if len(selected_samples) == 0:
                continue
                
            # Check if input features are vocabulary (size matches token count)
            sample_inp_size = selected_samples[0][1].size(0)
            in_labels = tokens if sample_inp_size == len(tokens) else [str(i) for i in range(sample_inp_size)]
            
            # Identify non-zero columns across all selected samples
            non_zero_cols = []
            total_cols = sample_inp_size
            
            if total_cols > 15:
                # Remove columns where max value is at most 0.05
                for col_idx in range(total_cols):
                    max_val = 0.0
                    for _, inp_row in selected_samples:
                        max_val = max(max_val, abs(inp_row[col_idx].item()))
                    if max_val > VAL_TO_BE_NON_ZERO:
                        non_zero_cols.append(col_idx)
            else:
                # Keep all columns with any non-zero values
                for col_idx in range(total_cols):
                    has_non_zero = False
                    for _, inp_row in selected_samples:
                        if abs(inp_row[col_idx].item()) > 1e-6:
                            has_non_zero = True
                            break
                    if has_non_zero:
                        non_zero_cols.append(col_idx)
            
            if len(non_zero_cols) == 0:
                continue
            
            # Find contiguous blocks of non-zero columns
            blocks = []
            if non_zero_cols:
                block_start = non_zero_cols[0]
                block_end = non_zero_cols[0]
                for col in non_zero_cols[1:]:
                    if col == block_end + 1:
                        block_end = col
                    else:
                        blocks.append((block_start, block_end))
                        block_start = col
                        block_end = col
                blocks.append((block_start, block_end))
                
            # Build display columns with "..." separators
            display_cols = []
            display_labels = []
            for block_idx, (start, end) in enumerate(blocks):
                if block_idx > 0:
                    display_cols.append(None)  # Marker for "..."
                    display_labels.append("...")
                for col_idx in range(start, end + 1):
                    display_cols.append(col_idx)
                    display_labels.append(in_labels[col_idx])
            
            # Create heatmap data matrix with spacing between rows
            num_display_cols = len(display_cols)
            num_samples = len(selected_samples)
            # Add blank rows between each sample for visual separation
            num_display_rows = num_samples * 2 - 1  # Original rows + blank rows between them
            heatmap_data = np.full((num_display_rows, num_display_cols), np.nan)
            output_data = np.full((num_display_rows, len(tokens_to_show)), np.nan)
                
            for sample_idx, (logit_vals, inp_row) in enumerate(selected_samples):
                display_row = sample_idx * 2  # Every other row, leaving blanks in between
                for display_idx, col_idx in enumerate(display_cols):
                    if col_idx is not None:  # Not a "..." marker
                        heatmap_data[display_row, display_idx] = inp_row[col_idx].item()
                output_data[display_row, :] = np.array(logit_vals)
            
            # Create custom y-coordinates where blank rows are half the height of data rows
            y_coords = [0]
            for i in range(num_display_rows):
                if i % 2 == 0:  # Data row
                    y_coords.append(y_coords[-1] + 1.0)
                else:  # Blank row - half height
                    y_coords.append(y_coords[-1] + 0.5)
            y_coords = np.array(y_coords)
            
            # Create figure with three subplots: input heatmap, connection space, and output heatmap
            total_height = y_coords[-1]
            fig = plt.figure(figsize=(max(12, (num_display_cols) * 0.8), total_height * 0.15 + 1))
            # Create gridspec with custom width ratios
            gs = fig.add_gridspec(1, 3, width_ratios=[num_display_cols, 0.3, len(tokens_to_show) * 0.4], wspace=0.05)
            ax1 = fig.add_subplot(gs[0, 0])
            ax_conn = fig.add_subplot(gs[0, 1])
            ax2 = fig.add_subplot(gs[0, 2])
            
            # Input heatmap using pcolormesh with custom y-coordinates
            X, Y = np.meshgrid(np.arange(num_display_cols + 1), y_coords)
            im1 = ax1.pcolormesh(X, Y, heatmap_data, cmap=truncate_colormap('Blues', minval=0.15), shading='flat')
            ax1.set_xticks(np.arange(len(display_labels)) + 0.5)
            ax1.set_xticklabels(display_labels, rotation=45, ha='right', fontsize=12)
            # Remove y-axis ticks
            ax1.set_yticks([])
            ax1.set_yticklabels([])
            ax1.set_ylim(y_coords[0], y_coords[-1])
            ax1.invert_yaxis()
            ax1.set_title('Input Features', fontsize=12)
            plt.colorbar(im1, ax=ax1, location='left')
                
            # Add text annotations to input cells if there are 5 or fewer columns
            if num_display_cols <= 1e9:
                for i in range(num_samples):
                    display_row = i * 2
                    # Calculate y-position as the center of the row
                    y_pos = (y_coords[display_row] + y_coords[display_row + 1]) / 2
                    for j in range(num_display_cols):
                        if not np.isnan(heatmap_data[display_row, j]):
                            text_val = heatmap_data[display_row, j]
                            if text_val > VAL_TO_BE_NON_ZERO:
                                ax1.text(j + 0.5, y_pos, f'{text_val:.2f}', ha='center', va='center',
                                        color='white' if text_val > heatmap_data[~np.isnan(heatmap_data)].max() * 0.5 else 'black',
                                        fontsize=10)
            
            # Connection axis - draw arrows between heatmaps with custom y-coordinates
            ax_conn.set_xlim(0, 1)
            ax_conn.set_ylim(y_coords[0], y_coords[-1])
            ax_conn.axis('off')
            ax_conn.invert_yaxis()
            # Draw horizontal lines and arrows for each sample
            for i in range(num_samples):
                display_row = i * 2
                # Calculate y-position as the center of the row
                y_pos = (y_coords[display_row] + y_coords[display_row + 1]) / 2
                # Draw horizontal line with arrow
                ax_conn.annotate('', xy=(1, y_pos), xytext=(0, y_pos),
                               arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))
            
            # Output heatmap using pcolormesh with custom y-coordinates
            output_2d = output_data
            # Make blank spaces white by replacing NaN with 0 in the sign calculation
            output_2d_sign = np.where(np.isnan(output_2d), -0.333, (output_2d > 0).astype(int) - (output_2d < 0).astype(int))
            X_out, Y_out = np.meshgrid(np.arange(len(tokens_to_show) + 1), y_coords)
            im2 = ax2.pcolormesh(X_out, Y_out, output_2d_sign, cmap=truncate_colormap('RdGy_r', minval=0.3, maxval=0.9), shading='flat')
            ax2.set_xticks(np.arange(len(tokens_to_show)) + 0.5)
            ax2.set_xticklabels([f'{sel_token}' for sel_token in tokens_to_show], rotation=45, ha='right', fontsize=12)
            ax2.set_yticks([])
            ax2.set_yticklabels([])
            ax2.set_ylim(y_coords[0], y_coords[-1])
            ax2.invert_yaxis()
            ax2.set_title('Output', fontsize=12)
            plt.colorbar(im2, ax=ax2)

            for ax in [ax1, ax2]:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                
            # Add text annotations for output values
            if num_display_cols <= 5:
                for i in range(num_samples):
                    display_row = i * 2
                    # Calculate y-position as the center of the row
                    y_pos = (y_coords[display_row] + y_coords[display_row + 1]) / 2
                    for j in range(len(tokens_to_show)):
                        text_val = output_data[display_row, j].item()
                        if text_val > VAL_TO_BE_NON_ZERO:
                            ax2.text(j + 0.5, y_pos, f'{text_val:.2f}', ha='center', va='center', 
                                    color='white' if text_val > 0 else "black",
                                    fontsize=10)
            
            plt.tight_layout()
            
            # Save heatmap
            safe_path_name = selected_path.replace("/", "_").replace("\\", "_")
            safe_token_name = sel_token.replace("/", "_").replace("\\", "_")
            heatmap_filename = f"mlp_{safe_path_name}_{safe_token_name}_for_main_paper.pdf"
            heatmap_path = mlps_folder / heatmap_filename
            plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        else:  # end point is attention head
            pass
    
    return latex_content

if __name__ == "__main__":
    path_to_model_outputs = "PATH_TO_SPECIFIC_EXP_WITH_MODEL_OUTPUTS"
    task_name = "bce_D4"
    dest_path = "PATH_TO_WHERE_TO_PUT_THE_IMAGE"
    get_mlp_input_output_latex_no_bins(path_to_model_outputs, task_name, dest_path)