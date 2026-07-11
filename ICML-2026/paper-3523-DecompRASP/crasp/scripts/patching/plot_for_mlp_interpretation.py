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

# Add the parent directory to the path to import convert_to_code
sys.path.append(os.path.abspath("."))
from convert_to_code import convert_config_to_code_qk_pruned

# Hyperparameter: show every K_STEP-th input when sorted by logits
K_STEP = 10

K_STEPS_TOP = [1, 2]
TAKE_TOP = [10, 20]

TASK_TO_TOKENS_TO_SHOW = {
    "012_0_2": ["0", "1"],
    "bce_012_0_2": ["0", "1", "2", "<eos>"],
    "aaaastar": ["0", "1"],
    "bce_aaaastar": ["a", "<eos>"],
    "aastar": ["0", "1"],
    "bce_aastar": ["a", "<eos>"],
    "ab_d_bc": ["0", "1"],
    "bce_ab_d_bc": ["a", "b", "c", "d", "<eos>"],
    "ababstar": ["0", "1"],
    "bce_ababstar": ["a", "b", "<eos>"],
    "abcde": ["0", "1"],
    "bce_abcde": ["a", "b", "c", "d", "e", "<eos>"],
    "bin_majority": ["0", "1"],
    "bin_majority_interleave": ["0", "1", "<eos>"],
    "count": ["1", "10", "100"] + ['<eos>'], # [str(i) for i in range(150)] + ['<eos>'],
    "D2": ["0", "1"],
    "bce_D2": ["a", "b", "<eos>"],
    "D3": ["0", "1"],
    "bce_D3": ["a", "b", "<eos>"],
    "D4": ["0", "1"],
    "bce_D4": ["a", "b", "<eos>"],
    "D12": ["0", "1"],
    "bce_D12": ["a", "b", "<eos>"],
    "majority": ["a", "p"], # [chr(i + ord('a')) for i in range(ord('z') - ord('a') + 1)] + ['<eos>'],
    "sort": ["0", "10", "100"] + ['<eos>'], # [str(i) for i in range(150)] + ['<eos>'],
    "tomita1": ["0", "1"],
    "bce_tomita1": ["1", "<eos>"],
    "tomita2": ["0", "1"],
    "bce_tomita2": ["0", "1", "<eos>"],
    "tomita4": ["0", "1"],
    "bce_tomita4": ["0", "1", "<eos>"],
    "tomita7": ["0", "1"],
    "bce_tomita7": ["0", "1", "<eos>"],
    "unique_bigram_copy": [str(i) for i in range(16)] + ['<eos>'],
    "unique_copy": ["0", "10", "100"] + ['<eos>'], # [str(i) for i in range(150)] + ['<eos>'],
    "unique_reverse": ["0", "10", "100"] + ['<eos>'], # [str(i) for i in range(150)] + ['<eos>'],
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


VAL_TO_BE_NON_ZERO = 0.2 # 0.05

def get_mlp_input_output_latex_no_bins(path_to_model_outputs, task_name, config_to_code_line,
                               config_to_deleted_code_line, dest_path, sanitized_task_name,
                               base_path_prefix, deleted_config_lines):
    """Generate LaTeX code for MLP input-output distributions with heatmaps."""
    # Load converted_mlp if it exists
    converted_mlp = None
    converted_mlp_path = path_to_model_outputs / "converted_mlp.pt"
    if converted_mlp_path.exists():
        import torch
        converted_mlp = torch.load(converted_mlp_path, map_location="cpu", weights_only=False)

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
    
    latex_content.append("\\paragraph{MLP Input-Output Distributions}")
    latex_content.append("")
    latex_content.append("")

    print("deleted_config_lines", deleted_config_lines)
    print("mlp_input_output", mlp_input_output.keys())
    
    for selected_path in mlp_input_output:
        if type(selected_path) == str:    # end point is unembed
            # k, vocab, in_vocab
            inp_cache, out_cache = mlp_input_output[selected_path]
            tokens = [tokenizer.vocab_inv[i] for i in range(len(tokenizer))]
            
            mlp_subpaths = selected_path.split("mlp-")
            line_nums = []
            for mlp_subpath_i, mlp_subpath in enumerate(mlp_subpaths):
                subpath_name = "mlp-".join(mlp_subpaths[mlp_subpath_i:])
                if subpath_name.startswith("lm_head"):
                    subpath_name = subpath_name.replace("lm_head-", "")
                    subpath_name = "LOGITS." + subpath_name
                else:
                    subpath_name = "MLP.mlp-" + subpath_name
                if subpath_name in config_to_code_line:    
                    line_nums.append(config_to_code_line[subpath_name])
                elif subpath_name in config_to_deleted_code_line:
                    line_nums.append(config_to_deleted_code_line[subpath_name])
            
            if len(line_nums) < 2:
                continue

            line_nums = sorted(line_nums)
            
            latex_content.append(f"Explaining per-position operation in Line {line_nums[0]} via its effect on Output Logits in Line {line_nums[-1]}")
            # latex_content.append(f"\\subparagraph{{Explain a path through MLP spanning lines {', '.join(map(str, sorted(line_nums)))} through its effect on logits}}")
            latex_content.append("")
            latex_content.append("")
            
            tokens_to_show = TASK_TO_TOKENS_TO_SHOW[task_name]
            
            for sel_token in tokens_to_show:
                sel_token_id = tokenizer.vocab[sel_token]
                
                latex_content.append(f"\\textbf{{Output Token: {escape_latex_special_chars(sel_token)}}}")
                latex_content.append("")
                
                # Collect all samples from all bins and sort by output logit
                all_samples = []
                num_bins = len(inp_cache)
                for bin_idx in range(num_bins):
                    bin_inp = torch.tensor(inp_cache[bin_idx])
                    bin_out = out_cache[bin_idx]
                    
                    if bin_inp.numel() == 0:
                        continue
                    
                    num_samples_in_bin = bin_inp.size(0)
                    for sample_idx in range(num_samples_in_bin):
                        logit_val = bin_out[sample_idx, sel_token_id, sel_token_id].item()
                        inp_row = bin_inp[sample_idx, sel_token_id, :].clone()
                        all_samples.append((logit_val, inp_row, bin_out[sample_idx, sel_token_id, sel_token_id].item()))
                
                if len(all_samples) == 0:
                    continue
                
                # Sort by logit (descending)
                all_samples.sort(key=lambda x: x[0], reverse=True)

                set_of_vals = set()
                filtered_sampels = []
                for row in all_samples:
                    if row[-1] not in set_of_vals:
                        filtered_sampels.append(row)
                        set_of_vals.add(row[-1])
                all_samples = filtered_sampels
                
                # Select every K_STEP-th sample
                selected_samples = []
                for (k_step, take), prev_take in zip(zip(K_STEPS_TOP, TAKE_TOP), [0] + TAKE_TOP):
                    selected_samples.extend(all_samples[prev_take:take:k_step])
                selected_samples.extend(all_samples[max(TAKE_TOP):len(all_samples) - max(TAKE_BOTTOM):K_STEP])
                
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
                        for _, inp_row, _ in selected_samples:
                            max_val = max(max_val, abs(inp_row[col_idx].item()))
                        if max_val > VAL_TO_BE_NON_ZERO:
                            non_zero_cols.append(col_idx)
                else:
                    # Keep all columns with any non-zero values
                    for col_idx in range(total_cols):
                        has_non_zero = False
                        for _, inp_row, _ in selected_samples:
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
                
                # Create heatmap data matrix
                num_display_cols = len(display_cols)
                num_samples = len(selected_samples)
                heatmap_data = np.full((num_samples, num_display_cols), np.nan)
                output_data = np.zeros(num_samples)
                
                for sample_idx, (logit_val, inp_row, out_val) in enumerate(selected_samples):
                    for display_idx, col_idx in enumerate(display_cols):
                        if col_idx is not None:  # Not a "..." marker
                            heatmap_data[sample_idx, display_idx] = inp_row[col_idx].item()
                    output_data[sample_idx] = out_val
                
                # Create figure with two subplots: input heatmap and output heatmap
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(12, num_display_cols * 0.8), num_samples * 0.3 + 1), 
                                               gridspec_kw={'width_ratios': [num_display_cols, 1]})
                
                # Input heatmap (NaN values will appear as white/blank)
                im1 = ax1.imshow(heatmap_data, cmap='Blues', aspect='auto', vmin=0)
                ax1.set_xticks(range(len(display_labels)))
                ax1.set_xticklabels(display_labels, rotation=45, ha='right')
                ax1.set_yticks(range(num_samples))
                ax1.set_yticklabels([])
                ax1.set_title('Input Features')
                plt.colorbar(im1, ax=ax1, location='left')
                
                # Add text annotations to input cells if there are 5 or fewer columns
                if num_display_cols <= 5:
                    for i in range(num_samples):
                        for j in range(num_display_cols):
                            if not np.isnan(heatmap_data[i, j]):
                                text_val = heatmap_data[i, j]
                                ax1.text(j, i, f'{text_val:.2f}', ha='center', va='center',
                                        color='white' if text_val > heatmap_data[~np.isnan(heatmap_data)].max() * 0.5 else 'black',
                                        fontsize=8)
                
                # Output heatmap (single column)
                output_2d = output_data.reshape(-1, 1)
                im2 = ax2.imshow(output_2d, cmap='RdGy_r', aspect='auto')
                ax2.set_xticks([0])
                ax2.set_xticklabels([f'logit:{sel_token}'])
                ax2.set_yticks(range(num_samples))
                ax2.set_yticklabels([])
                ax2.set_title('Output')
                plt.colorbar(im2, ax=ax2)
                
                # Add text annotations for output values
                for i in range(num_samples):
                    text_val = output_data[i]
                    ax2.text(0, i, f'{text_val:.2f}', ha='center', va='center', 
                            color='white' if text_val >= output_2d.max() * 0.5 or text_val <= output_2d.min() * 0.5 else 'black',
                            fontsize=8)
                
                plt.tight_layout()
                
                # Save heatmap
                safe_path_name = selected_path.replace("/", "_").replace("\\", "_")
                safe_token_name = sel_token.replace("/", "_").replace("\\", "_")
                heatmap_filename = f"mlp_{safe_path_name}_{safe_token_name}_sorted.pdf"
                heatmap_path = mlps_folder / heatmap_filename
                plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                # Add to LaTeX content
                latex_content.append("\\begin{figure}[H]")
                latex_content.append("    \\centering")
                latex_content.append(f"    \\includegraphics[width=0.95\\linewidth]{{{base_path_prefix}/{sanitized_task_name}/mlps/{heatmap_filename}}}")
                latex_content.append(f"    \\caption{{MLP Input-Output for token: {escape_latex_special_chars(sel_token)} (sorted by logits)}}")
                latex_content.append("\\end{figure}")
                latex_content.append("")
        
        else:  # end point is attention head
            q_inp_cache, k_inp_cache, out_cache = mlp_input_output[selected_path]
            # k, num_q_cluster, in_vocab
            num_q_cluster = q_inp_cache[0].size(1)
            tokens = [tokenizer.vocab_inv[i] for i in range(len(tokenizer))]
            clusters = ["All"] + [f"cluster_{i}" for i in range(num_q_cluster-1)]
            
            q, k, l, h = selected_path
            
            mlp_subpaths_q = q.split("mlp-") if isinstance(q, str) else []
            mlp_subpaths_k = k.split("mlp-") if isinstance(k, str) else []
            if mlp_subpaths_q[0] == "":
                mlp_subpaths_q = mlp_subpaths_q[1:]
            if mlp_subpaths_k[0] == "":
                mlp_subpaths_k = mlp_subpaths_k[1:]
            line_nums_q, line_nums_k = [], []
            
            for mlp_subpath_i, mlp_subpath in enumerate(mlp_subpaths_q):
                subpath_name = "mlp-" + "mlp-".join(mlp_subpaths_q[mlp_subpath_i:])
                print("subpath_name q:", subpath_name)
                if subpath_name in converted_mlp:
                    continue
                subpath_name = "MLP." + subpath_name
                if subpath_name in config_to_code_line:    
                    line_nums_q.append(config_to_code_line[subpath_name])
                elif subpath_name in config_to_deleted_code_line:
                    line_nums_q.append(config_to_deleted_code_line[subpath_name])
            
            for mlp_subpath_i, mlp_subpath in enumerate(mlp_subpaths_k):
                subpath_name = "mlp-" + "mlp-".join(mlp_subpaths_k[mlp_subpath_i:])
                print("subpath_name k:", subpath_name)
                if subpath_name in converted_mlp:
                    continue
                subpath_name = "MLP." + subpath_name
                if subpath_name in config_to_code_line:    
                    line_nums_k.append(config_to_code_line[subpath_name])
                elif subpath_name in config_to_deleted_code_line:
                    line_nums_k.append(config_to_deleted_code_line[subpath_name])

            if len(line_nums_q) == 0 and len(line_nums_k) == 0:
                continue

            assert not (len(line_nums_q) > 0 and len(line_nums_k) > 0), (line_nums_q, line_nums_k, converted_mlp.keys(), q, k)
            line_nums_qk = sorted(line_nums_q + line_nums_k)

            output_what = "Query" if line_nums_q else "Key"

            attention_line = f"ATTENTION-{l}-{h}.{q}-{k}"
            if attention_line not in config_to_deleted_code_line:
                continue
            attention_code_line = config_to_deleted_code_line[attention_line]
            
            latex_content.append(f"\\subparagraph{{Explaining per-position operation in Line {line_nums_qk[0]} via its effect on Output {output_what} in Line {attention_code_line}}}")
            latex_content.append("")
            latex_content.append("")

            num_clusters_to_show = 1
            
            for cluster_idx in range(num_clusters_to_show):
                sel_cluster = clusters[cluster_idx]
                sel_cluster_id = cluster_idx
                
                
                # Collect all samples from all bins and sort by output logit
                all_samples = []
                num_bins = len(q_inp_cache)
                for bin_idx in range(num_bins):
                    bin_q_inp = q_inp_cache[bin_idx]
                    bin_k_inp = k_inp_cache[bin_idx]
                    bin_out = out_cache[bin_idx]
                    
                    num_samples_in_bin = bin_q_inp.size(0)
                    for sample_idx in range(num_samples_in_bin):
                        logit_val = bin_out[sample_idx, sel_cluster_id].item()
                        q_inp_row = bin_q_inp[sample_idx, sel_cluster_id, :].clone()
                        k_inp_row = bin_k_inp[sample_idx, sel_cluster_id, :].clone()
                        all_samples.append((logit_val, q_inp_row, k_inp_row, logit_val))
                
                if len(all_samples) == 0:
                    continue
                
                # Sort by logit (descending)
                all_samples.sort(key=lambda x: x[0], reverse=True)

                set_of_vals = set()
                filtered_sampels = []
                for row in all_samples:
                    if row[-1] not in set_of_vals:
                        filtered_sampels.append(row)
                        set_of_vals.add(row[-1])
                all_samples = filtered_sampels
                
                # Select every K_STEP-th sample
                selected_samples = []
                for (k_step, take), prev_take in zip(zip(K_STEPS_TOP, TAKE_TOP), [0] + TAKE_TOP):
                    selected_samples.extend(all_samples[prev_take:take:k_step])
                selected_samples.extend(all_samples[max(TAKE_TOP):len(all_samples) - max(TAKE_BOTTOM):K_STEP])
                # samples_bottom = []
                # for (k_step, take), prev_take in zip(zip(K_STEPS_BOTTOM, TAKE_BOTTOM), [0] + TAKE_BOTTOM):
                #     samples_bottom.extend(all_samples[-prev_take-1:-take-1:-k_step])
                # selected_samples.extend(reversed(samples_bottom))
                # selected_samples = all_samples[::K_STEP]
                
                if len(selected_samples) == 0:
                    continue
                
                # Check if Q and K inputs are vocabulary
                q_sample_size = selected_samples[0][1].size(0)
                k_sample_size = selected_samples[0][2].size(0)
                q_in_labels = tokens if q_sample_size == len(tokens) else [str(i) for i in range(q_sample_size)]
                k_in_labels = tokens if k_sample_size == len(tokens) else [str(i) for i in range(k_sample_size)]
                
                # Identify non-zero columns for Q and K
                non_zero_q_cols = []
                total_q_cols = q_sample_size
                
                if total_q_cols > 15:
                    for col_idx in range(total_q_cols):
                        max_val = 0.0
                        for _, q_inp_row, _, _ in selected_samples:
                            max_val = max(max_val, abs(q_inp_row[col_idx].item()))
                        if max_val > VAL_TO_BE_NON_ZERO:
                            non_zero_q_cols.append(col_idx)
                else:
                    for col_idx in range(total_q_cols):
                        has_non_zero = False
                        for _, q_inp_row, _, _ in selected_samples:
                            if abs(q_inp_row[col_idx].item()) > 1e-6:
                                has_non_zero = True
                                break
                        if has_non_zero:
                            non_zero_q_cols.append(col_idx)
                
                non_zero_k_cols = []
                total_k_cols = k_sample_size
                
                if total_k_cols > 15:
                    for col_idx in range(total_k_cols):
                        max_val = 0.0
                        for _, _, k_inp_row, _ in selected_samples:
                            max_val = max(max_val, abs(k_inp_row[col_idx].item()))
                        if max_val > VAL_TO_BE_NON_ZERO:
                            non_zero_k_cols.append(col_idx)
                else:
                    for col_idx in range(total_k_cols):
                        has_non_zero = False
                        for _, _, k_inp_row, _ in selected_samples:
                            if abs(k_inp_row[col_idx].item()) > 1e-6:
                                has_non_zero = True
                                break
                        if has_non_zero:
                            non_zero_k_cols.append(col_idx)
                
                if len(non_zero_q_cols) == 0 and len(non_zero_k_cols) == 0:
                    continue
                
                # Helper function to build display columns with "..." separators
                def build_display_columns(non_zero_cols, labels, prefix):
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
                    
                    display_cols = []
                    display_labels = []
                    for block_idx, (start, end) in enumerate(blocks):
                        if block_idx > 0:
                            display_cols.append(None)
                            display_labels.append("...")
                        for col_idx in range(start, end + 1):
                            display_cols.append(col_idx)
                            display_labels.append(f"{labels[col_idx]}")
                    
                    return display_cols, display_labels
                
                # Build Q and K display columns
                q_display_cols, q_display_labels = build_display_columns(non_zero_q_cols, q_in_labels, "Q")
                k_display_cols, k_display_labels = build_display_columns(non_zero_k_cols, k_in_labels, "K")
                
                # Create heatmap data
                num_samples = len(selected_samples)
                heatmap_data_q = np.full((num_samples, len(q_display_cols)), np.nan)
                heatmap_data_k = np.full((num_samples, len(k_display_cols)), np.nan)
                output_data = np.zeros(num_samples)
                
                # Fill Q data
                for sample_idx, (_, q_inp_row, k_inp_row, out_val) in enumerate(selected_samples):
                    for display_idx, col_idx in enumerate(q_display_cols):
                        if col_idx is not None:
                            heatmap_data_q[sample_idx, display_idx] = q_inp_row[col_idx].item()
                    
                    # Fill K data
                    for display_idx, col_idx in enumerate(k_display_cols):
                        if col_idx is not None:
                            heatmap_data_k[sample_idx, display_idx] = k_inp_row[col_idx].item()
                    
                    output_data[sample_idx] = out_val
                
                # Create figure with three subplots
                total_display_cols = len(q_display_cols) + len(k_display_cols)
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(max(12, total_display_cols * 0.8), num_samples * 0.3 + 1),
                                               gridspec_kw={'width_ratios': [len(q_display_cols), len(k_display_cols), 1]})
                
                # Query heatmap
                im1 = ax1.imshow(heatmap_data_q, cmap='Blues', aspect='auto', vmin=0)
                ax1.set_xticks(range(len(q_display_labels)))
                ax1.set_xticklabels(q_display_labels, rotation=45, ha='right')
                ax1.set_yticks(range(num_samples))
                ax1.set_yticklabels([])
                ax1.set_title('Input Features: Query')
                plt.colorbar(im1, ax=ax1, location='left')
                
                # Add text annotations to Query cells if there are 5 or fewer columns
                if len(q_display_cols) <= 5:
                    for i in range(num_samples):
                        for j in range(len(q_display_cols)):
                            if not np.isnan(heatmap_data_q[i, j]):
                                text_val = heatmap_data_q[i, j]
                                ax1.text(j, i, f'{text_val:.2f}', ha='center', va='center',
                                        color='white' if text_val > heatmap_data_q[~np.isnan(heatmap_data_q)].max() * 0.5 else 'black',
                                        fontsize=8)
                
                # Key heatmap
                im2 = ax2.imshow(heatmap_data_k, cmap='Blues', aspect='auto', vmin=0)
                ax2.set_xticks(range(len(k_display_labels)))
                ax2.set_xticklabels(k_display_labels, rotation=45, ha='right')
                ax2.set_yticks(range(num_samples))
                ax2.set_yticklabels([])
                ax2.set_title('Input Features: Key')
                plt.colorbar(im2, ax=ax2, location='left')
                
                # Add text annotations to Key cells if there are 5 or fewer columns
                if len(k_display_cols) <= 5:
                    for i in range(num_samples):
                        for j in range(len(k_display_cols)):
                            if not np.isnan(heatmap_data_k[i, j]):
                                text_val = heatmap_data_k[i, j]
                                ax2.text(j, i, f'{text_val:.2f}', ha='center', va='center',
                                        color='white' if text_val > heatmap_data_k[~np.isnan(heatmap_data_k)].max() * 0.5 else 'black',
                                        fontsize=8)
                
                # Output heatmap
                output_2d = output_data.reshape(-1, 1)
                im3 = ax3.imshow(output_2d, cmap='RdGy_r', aspect='auto')
                ax3.set_xticks([0])
                ax3.set_xticklabels(['Logits'])
                ax3.set_yticks(range(num_samples))
                ax3.set_yticklabels([])
                ax3.set_title('Output')
                plt.colorbar(im3, ax=ax3)
                
                # Add text annotations for output values
                for i in range(num_samples):
                    text_val = output_data[i]
                    ax3.text(0, i, f'{text_val:.2f}', ha='center', va='center', 
                            color='white' if text_val >= output_2d.max() * 0.5 or text_val <= output_2d.min() * 0.5 else 'black',
                            fontsize=8)
                
                plt.tight_layout()
                
                # Save heatmap
                heatmap_filename = f"mlp_attn_{l}_{h}_q{q}_k{k}_cluster{cluster_idx}_sorted.pdf"
                heatmap_path = mlps_folder / heatmap_filename
                plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                # Add to LaTeX content
                latex_content.append("\\begin{figure}[H]")
                latex_content.append("    \\centering")
                latex_content.append(f"    \\includegraphics[width=0.95\\linewidth]{{{base_path_prefix}/{sanitized_task_name}/mlps/{heatmap_filename}}}")
                latex_content.append(f"    \\caption{{MLP Input-Output (sorted by logits)}}")
                latex_content.append("\\end{figure}")
                latex_content.append("")
    
    return latex_content