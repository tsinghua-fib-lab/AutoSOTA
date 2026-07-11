import matplotlib.pyplot as plt
from pathlib import Path
import re
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import copy
from itertools import product

from show_heatmap import get_LN_matrix_for_node, get_OV_for_head, get_QK_for_head

def is_token_dim_activation(act_name, converted_mlp):
    pattern = r'attn_output-\d+-\d+|mlp-\d+|lm_head|wte|wpe'
    split_nodes = re.findall(pattern, act_name) # ['mlp-1', 'attn_output-1-1', 'wte']
    for i, node in enumerate(split_nodes): # outer to inner
        if node.startswith("attn_output"):
            continue
        elif node == "wte":
            return True
        elif node == "wpe":
            return False
        elif node.startswith("mlp"):
            converted_mlp_name = "-".join(split_nodes[i:])
            if converted_mlp_name in converted_mlp:
                mlp_primitive, _ = converted_mlp[converted_mlp_name]
                match mlp_primitive:
                    case "no_op":
                        continue
                    case ("sharpen", n):
                        continue
                    case "erase":
                        return False
                    case "harden":
                        continue
                    case ("exists", idx):   # can try different threshold
                        return False
                    case ("forall", threshold):   # expressivity > individual ones, e.g. x[:, idx]> threshold
                        return False
                    case ("equal", indices):
                        return False
                    case ("01balance", pow, center):   # 1/2..1/4
                        return False
                    case ("ABbalance", pow, center):
                        return False
                    case ("diff", idx1, idx2):
                        return False
                    case "erase":
                        return False
                    case ("keep_one", n):
                        return False
                    case "combine":
                        return False
                    case _:
                        raise NotImplementedError(mlp_primitive, "is not implemented")
            else:
                return False
        else:
            raise RuntimeError("node not recognized", node)
    raise RuntimeError("somehting went wrong", act_name, converted_mlp)

def is_cartesian_act(act_name, converted_mlp, current_config):
    pattern = r'attn_output-\d+-\d+|mlp-\d+|lm_head|wte|wpe'
    split_nodes = re.findall(pattern, act_name)
    closest_non_attn = 0
    for i in range(len(split_nodes)):
        if split_nodes[i].startswith("attn_output"):
            closest_non_attn += 1
        else:
            break
    split_nodes = split_nodes[closest_non_attn:]
    closest_non_attn_node = "-".join(split_nodes)
    if closest_non_attn_node in converted_mlp and converted_mlp[closest_non_attn_node][0] == "combine":
        mlp_match = re.match("mlp-(\d+)", closest_non_attn_node)
        assert mlp_match, (closest_non_attn_node, act_name)
        mlp_layer = int(mlp_match.group(1))
        acts_incoming_to_mlp = current_config[mlp_layer]["mlp"]
        is_cartesian_acts_incoming = [is_cartesian_act(act, converted_mlp, current_config) for act in acts_incoming_to_mlp]
        is_token_acts_incoming = [is_token_dim_activation(act, converted_mlp) for act in acts_incoming_to_mlp]
        if all([t or c[0] for t, c in zip(is_token_acts_incoming, is_cartesian_acts_incoming)]):
            sum_incoming_acts = sum([(1 if t else c[1]) for t, c in zip(is_token_acts_incoming, is_cartesian_acts_incoming)])
            return True, sum_incoming_acts
    return False, None

def get_num_dims(act_name, converted_mlp, hooked_model, tokenizer):
    pattern = r'attn_output-\d+-\d+|mlp-\d+|lm_head|wte|wpe'
    split_nodes = re.findall(pattern, act_name) # ['mlp-1', 'attn_output-1-1', 'wte']
    split_nodes = split_nodes[len(split_nodes)-1::-1]
    num_dims = 0
    primitives_on_the_way = []
    for i, node in enumerate(split_nodes): # outer to inner
        if node.startswith("attn_output"):
            continue
        elif node == "wte":
            num_dims = len(tokenizer.vocab)
        elif node == "wpe":
            num_dims = hooked_model.model.transformer.wpe.weight.shape[0]
        elif node.startswith("mlp"):
            converted_mlp_name = "-".join(split_nodes[i::-1])
            if converted_mlp_name in converted_mlp:
                mlp_primitive, _ = converted_mlp[converted_mlp_name]
                primitives_on_the_way.append(mlp_primitive)
                hooked_model.logger("primitive for", "-".join(split_nodes[i::-1]), "is", mlp_primitive)
                match mlp_primitive:
                    case "no_op":
                        continue
                    case ("sharpen", n):
                        continue
                    case "erase":
                        num_dims = 1
                    case "harden":
                        continue
                    case ("exists", idx):   # can try different threshold
                        num_dims = 2
                    case ("forall", threshold):   # expressivity > individual ones, e.g. x[:, idx]> threshold
                        num_dims = num_dims + 1
                    case ("equal", indices):
                        num_dims = 2
                    case ("01balance", pow, center):   # 1/2..1/4
                        num_dims = 3
                    case ("ABbalance", pow, center):
                        num_dims = 3
                    case ("diff", idx1, idx2):
                        return 12
                    case "erase":
                        num_dims = 1
                    case ("keep_one", n):
                        inps = []
                        for mlp_inp in hooked_model.config[int(node[4:])]["mlp"]:
                            inps.append(get_num_dims(mlp_inp, converted_mlp, hooked_model, tokenizer))
                        num_dims = inps[n]
                    case "combine":
                        inps_prod = 1
                        for mlp_inp in hooked_model.config[int(node[4:])]["mlp"]:
                            inps_prod *= get_num_dims(mlp_inp, converted_mlp, hooked_model, tokenizer).shape[-1]
                        num_dims = inps_prod
                    case _:
                        raise NotImplementedError(mlp_primitive, "is not implemented")
            else:
                primitives_on_the_way.append("not converted")
                model_config = hooked_model.model.config
                num_dims = model_config.n_embd
        else:
            raise RuntimeError("node not recognized", node)
    hooked_model.logger(f"num dims for {act_name} is {num_dims}; primitives_on_the_way={primitives_on_the_way}")
    return int(num_dims)

def plot_and_save_primitives_matrices(matrix, save_path, ticks_x=None, ticks_y=None, add_causal_mask=False):
    def simplify_product_labels(labels):
        """
        Detect if labels are from a Cartesian product (e.g., "A-B-C-D") and simplify
        to show only the leftmost two parts (e.g., "A-B").
        """
        if labels is None or len(labels) == 0:
            return labels
        
        # Check if labels look like Cartesian products (contain "-" separator)
        # and have consistent structure
        sample_splits = [str(label).split("-") for label in labels[:min(10, len(labels))]]
        
        # Detect if this looks like a product: all labels have "-", and most have same number of parts
        has_separator = all(len(parts) > 1 for parts in sample_splits)
        
        if has_separator:
            num_parts = [len(parts) for parts in sample_splits]
            assert all([nump == num_parts[0] for nump in num_parts])
            num_parts = num_parts[0]
            # If most labels have 3+ parts, it's likely a product
            # Simplify: take only leftmost 1 part
            simplified = []
            last_parts = None
            for label in labels:
                parts = str(label).split("-")

                first_two = "-".join(parts[:1])

                if last_parts != first_two:
                    last_parts = first_two
                    simplified.append(label)
                else:
                    simplified.append("")
            return simplified
        return labels
    
    # Simplify labels if they're from Cartesian products
    if ticks_x is not None:
        ticks_x = simplify_product_labels(ticks_x)
    if ticks_y is not None:
        ticks_y = simplify_product_labels(ticks_y)
    
    is_column = False
    original_shape = matrix.shape
    
    # Handle 1D tensors by converting to 2D
    if matrix.ndim < 2:
        matrix = matrix.unsqueeze(0)
        is_column = True
    
    matrix_np = matrix.detach().cpu().numpy()
    
    # Calculate adaptive figure sizing based on matrix dimensions and number of ticks
    num_x_ticks = matrix_np.shape[1] if ticks_x is None else len(ticks_x)
    num_y_ticks = matrix_np.shape[0] if ticks_y is None else len(ticks_y)
    
    if is_column or matrix_np.shape[0] == 1:
        # For single row matrices, scale width based on number of x-ticks
        width = max(6, min(num_x_ticks * 0.4, 20))
        height = 2.5
        figsize = (width, height)
    elif matrix_np.shape[1] == 1:
        # For single column matrices, scale height based on number of y-ticks
        width = 3
        height = max(4, min(num_y_ticks * 0.3, 16))
        figsize = (width, height)
    else:
        # For regular matrices, use proportional sizing with tick-aware scaling
        base_width = max(6, min(num_x_ticks * 0.25, 16))
        base_height = max(4, min(num_y_ticks * 0.25, 16))
        aspect_ratio = matrix_np.shape[1] / matrix_np.shape[0]
        
        if aspect_ratio > 2:
            figsize = (base_width, base_height * 0.7)
        elif aspect_ratio < 0.5:
            figsize = (base_width * 0.7, base_height)
        else:
            figsize = (base_width, base_height)

    fig, ax = plt.subplots(figsize=figsize)
    
    # Use nicer color schemes
    import matplotlib.colors as mcolors
    import numpy as np
    
    # Choose color scheme based on data characteristics
    data_range = np.max(matrix_np) - np.min(matrix_np)
    has_negative = np.min(matrix_np) < 0
    
    if has_negative:
        # Use diverging colormap for data with negative values
        base_cmap = plt.cm.get_cmap('Blues')
        colors = base_cmap(np.linspace(0, 1, 256))
        # Make it more pastel
        # pastel_colors = colors * 0.75 + np.array([1, 1, 1, 1]) * 0.25
        custom_cmap = mcolors.ListedColormap(colors)
    else:
        # Use sequential colormap for positive-only data
        # Choose between different pleasant color schemes
        color_schemes = ['viridis', 'plasma', 'cividis', 'magma']
        base_cmap_name = 'Blues'  # Default to viridis, but you can randomize or make it configurable
        
        base_cmap = plt.cm.get_cmap(base_cmap_name)
        colors = base_cmap(np.linspace(0, 1, 256))
        # Create softer, more pastel version
        # pastel_colors = colors * 0.65 + np.array([1, 1, 1, 1]) * 0.35
        custom_cmap = mcolors.ListedColormap(colors)
    
    im = ax.imshow(matrix_np, cmap=custom_cmap, aspect='auto')
    
    # Add causal mask overlay if requested
    if add_causal_mask and matrix_np.shape[0] == matrix_np.shape[1]:
        import numpy as np
        mask = np.triu(np.ones_like(matrix_np), k=1)
        mask = mask.astype(float)
        mask[mask == 0] = np.nan
        ax.imshow(mask, cmap='Greys', alpha=1., aspect='auto', vmin=0, vmax=2)
    
    # Style the colorbar with better proportions
    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.ax.tick_params(labelsize=14)
    
    # Helper function to calculate optimal font size
    def calculate_optimal_fontsize(num_ticks, available_space_inches, max_label_length, rotation_deg, dpi=300):
        """
        Calculate the maximum font size that prevents tick label overlap.
        
        Args:
            num_ticks: Number of tick labels
            available_space_inches: Available space in inches (width for x-axis, height for y-axis)
            max_label_length: Maximum character length of labels
            rotation_deg: Rotation angle in degrees
            dpi: Dots per inch for the figure
        """
        if num_ticks <= 1:
            return 14
        
        # Average space per tick in inches
        space_per_tick = available_space_inches / num_ticks
        
        # Estimate character width in points (1 point = 1/72 inch)
        # Typical character width is about 0.6 * font_size in points
        rotation_rad = np.deg2rad(rotation_deg)
        
        # For rotated text, calculate effective width
        # Width = char_width * cos(rotation) + char_height * sin(rotation)
        # Height = char_width * sin(rotation) + char_height * cos(rotation)
        
        # Character dimensions relative to font size
        char_width_ratio = 0.55  # Width of average char relative to font size
        char_height_ratio = 1.2  # Height including ascenders/descenders
        
        # Calculate maximum font size that fits
        # Projected width per label = (char_width * length * cos + height * sin) * fontsize/72
        if abs(rotation_deg) < 1:  # Horizontal text
            max_fontsize = (space_per_tick * 72) / (max_label_length * char_width_ratio + 0.2)
        else:
            # For rotated text, we care about the projected width
            projected_width_ratio = (max_label_length * char_width_ratio * abs(np.cos(rotation_rad)) + 
                                    char_height_ratio * abs(np.sin(rotation_rad)))
            max_fontsize = (space_per_tick * 72) / (projected_width_ratio + 0.2)
        
        # Clamp to smaller, less cluttered bounds
        return max(12, min(max_fontsize, 24))
    
    # Improved x-tick handling with optimal font sizing
    if matrix_np.shape[1] > 1 and ticks_x is not None:
        ax.set_xticks(range(matrix_np.shape[1]))
        
        # Calculate max label length
        max_label_len = max(len(str(label)) for label in ticks_x) if ticks_x else 1
        
        # Get available width for ticks (figure width minus margins)
        available_width = figsize[0] * 0.85  # Account for margins and colorbar
        
        # Determine optimal rotation and font size with more aggressive rotation
        if num_x_ticks <= 3:
            rotation = 0
            fontsize = calculate_optimal_fontsize(num_x_ticks, available_width, max_label_len, rotation)
        elif num_x_ticks <= 8:
            rotation = 45
            fontsize = calculate_optimal_fontsize(num_x_ticks, available_width, max_label_len, rotation)
        elif num_x_ticks <= 20:
            rotation = 60
            fontsize = calculate_optimal_fontsize(num_x_ticks, available_width, max_label_len, rotation)
        else:
            rotation = 80
            fontsize = calculate_optimal_fontsize(num_x_ticks, available_width, max_label_len, rotation)
        
        tick_indices = range(num_x_ticks)
        
        # Set the ticks and labels
        ax.set_xticks([i for i in tick_indices])
        ax.set_xticklabels([ticks_x[i] for i in tick_indices], 
                          fontsize=fontsize, rotation=rotation, ha='right' if rotation > 0 else 'center')
        
        # Add subtle grid lines aligned with ticks
        ax.set_xticks(range(matrix_np.shape[1]), minor=True)
    
    # Improved y-tick handling with optimal font sizing
    if matrix_np.shape[0] > 1 and not is_column and ticks_y is not None:
        # Calculate max label length
        max_label_len = 1.5 # max(len(str(label)) for label in ticks_y) if ticks_y else 1
        
        # Get available height for ticks
        available_height = figsize[1] * 0.9  # Account for margins
        
        # Y-ticks are typically not rotated
        fontsize = calculate_optimal_fontsize(num_y_ticks, available_height, max_label_len, rotation_deg=0)
        
        tick_indices = range(num_y_ticks)
            
        ax.set_yticks([i for i in tick_indices])
        ax.set_yticklabels([ticks_y[i] for i in tick_indices], fontsize=fontsize)
        
        # Add subtle grid lines for y-axis too
        ax.set_yticks(range(matrix_np.shape[0]), minor=True)
    
    # Remove ticks for single-row/column cases if no labels
    if is_column and ticks_y is None:
        ax.set_yticks([])
        ax.set_ylabel('')
    
    # Improved styling
    # ax.grid(True, which='major', alpha=0.2, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_color('#666666')
    ax.spines['left'].set_color('#666666')
    
    # Enhanced title with better formatting
    # title = f"Matrix Shape: {original_shape}"
    # ax.set_title(title, fontsize=11, pad=20, fontweight='normal', color='#333333')
    
    # Improve layout to prevent label cutoff
    plt.tight_layout()
    
    # Ensure directory exists and save with high quality
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.1)
    plt.close(fig)  # Close figure to free memory

def get_product_for_one_side_ignore_dep_prod(hooked_model, oa_vecs, converted_mlp, path):
    model = hooked_model.model
    d_model = model.config.hidden_size

    pattern = r'attn_output-\d+-\d+|mlp-\d+|lm_head|wte|wpe'
    split_nodes = re.findall(pattern, path) # ['attn_output-2-0', 'attn_output-1-1', 'wte']
    split_nodes = split_nodes[::-1].copy()
    indep_prod = None
    for i, node in enumerate(split_nodes):
        if node.startswith("attn_output"):
            _, layer, head = node.split("-")
            layer, head = int(layer), int(head)

            past_path = "-".join(split_nodes[i-1::-1])
            W_ln = get_LN_matrix_for_node(model, oa_vecs, layer, "v", head, past_path)
            W_v, W_o = get_OV_for_head(model, layer, head)
            indep_prod = indep_prod @ W_ln @ W_v @ W_o

        elif node == "wte":
            indep_prod = model.transformer.wte.weight.data     # vocab_size, d_model
        elif node == "wpe":
            indep_prod = model.transformer.wpe.weight.data

        elif node.startswith("mlp"):
            layer = int(node.split("-")[1])
            node_path = "-".join(split_nodes[i::-1])
            if node_path in converted_mlp:
                primitive, C = converted_mlp[node_path]
                indep_prod = C
            else:
                indep_prod = torch.eye(d_model).to(model.device)
        else:
            raise RuntimeError("node not recognized", node)

    return indep_prod

def get_product_for_one_side_for_unembed_ignore_dep_prod(hooked_model, oa_vecs, converted_mlp, path):
    
    if path != "vocab_bias":
        indep_prod = get_product_for_one_side_ignore_dep_prod(hooked_model, oa_vecs, converted_mlp, path)
    else:
        indep_prod = oa_vecs.output_vertex_oa.data[oa_vecs.to_out_oa_idx[("lm_head",)]].unsqueeze(0)
    
    W_ln = get_LN_matrix_for_node(hooked_model.model, oa_vecs, None, "lm_head", None)
    indep_prod = indep_prod @ W_ln @ hooked_model.model.lm_head.weight.data.T

    return indep_prod    # all already transposed


def get_product_for_one_side_for_head_ignore_dep_prod(hooked_model, oa_vecs, converted_mlp, attn_layer_idx, attn_head_idx, qk_type, path):
    
    indep_prod = get_product_for_one_side_ignore_dep_prod(hooked_model, oa_vecs, converted_mlp, path)
    
    W_ln = get_LN_matrix_for_node(hooked_model.model, oa_vecs, attn_layer_idx, qk_type, attn_head_idx)
    W_q_or_k = get_QK_for_head(hooked_model.model, attn_layer_idx, qk_type, attn_head_idx)
    indep_prod = indep_prod @ W_ln @ W_q_or_k

    return indep_prod    # all already transposed

def expand_grid(params):
    grid_keys = [k for k, v in params.items() if isinstance(v, list)]
    fixed_keys = [k for k, v in params.items() if not isinstance(v, list)]
    
    grid_values = [params[k] for k in grid_keys]
    
    # Generate all combinations
    combos = []
    for combo in product(*grid_values):
        conf = {k: v for k, v in zip(grid_keys, combo)}
        conf.update({k: params[k] for k in fixed_keys})
        combos.append(conf)
    return combos