import matplotlib.pyplot as plt
from pathlib import Path
import re
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import copy
from itertools import product

from show_heatmap import get_LN_matrix_for_node, get_OV_for_head, get_QK_for_head

def plot_and_save_primitives_matrices_for_main_paper(matrix, save_path, ticks_x=None, ticks_y=None, add_causal_mask=False):
    if matrix.dim() == 2:
        if ticks_x is None:
            ticks_x = list(range(matrix.shape[1]))
        if ticks_y is None:
            ticks_y = list(range(matrix.shape[0]))
    else:
        if ticks_x is None:
            ticks_x = list(range(matrix.shape[0]))
    
    is_column = False
    original_shape = matrix.shape
    
    if matrix.ndim == 1:
        is_column = True
    elif matrix.ndim == 2 and (matrix.shape[0] == 1 or matrix.shape[1] == 1):
        is_column = True
        matrix = matrix.squeeze()
    
    matrix_np = matrix.detach().cpu().numpy()
    
    import numpy as np
    MAX_SIZE = 10
    MAX_SIZE_COLUMN = 10
    BOTTOM_ROWS = 1
    RIGHT_COLS = 1
    DOTS_WIDTH = 1
    DOTS_HEIGHT = 1
    truncated = False

    TRUNCATE = True
    KEEP_TOP = False
    KEEP_BOTTOM = True
    

    dots_cols = []
    dots_rows = []

    if not is_column and matrix_np.shape[0] > 2:
        zero_rows = np.all(np.isclose(matrix_np, 0), axis=1)
        
        new_matrix_np = []
        new_ticks_y = [] if ticks_y is not None else None
        new_dots_rows = []
        
        i = 0
        original_row_idx = 0
        while i < len(zero_rows):
            if zero_rows[i] and (i >= MAX_SIZE or not KEEP_TOP) and (i < len(zero_rows) - BOTTOM_ROWS or not KEEP_BOTTOM):
                start = i
                while i < len(zero_rows) and zero_rows[i] and (i >= MAX_SIZE or not KEEP_TOP) and (i < len(zero_rows) - BOTTOM_ROWS or not KEEP_BOTTOM):
                    i += 1
                count = i - start
                
                if count > 2:
                    new_matrix_np.append(np.full((1, matrix_np.shape[1]), np.nan))
                    if new_ticks_y is not None:
                        new_ticks_y.append('...')
                    new_dots_rows.append(original_row_idx)
                    original_row_idx += 1
                else:
                    for row_idx in range(start, i):
                        new_matrix_np.append(matrix_np[row_idx:row_idx+1])
                    if new_ticks_y is not None:
                        new_ticks_y.extend(ticks_y[start:i])
                    original_row_idx += count
            else:
                new_matrix_np.append(matrix_np[i:i+1])
                if new_ticks_y is not None:
                    new_ticks_y.append(ticks_y[i])
                i += 1
                original_row_idx += 1
        
        if len(new_matrix_np) < matrix_np.shape[0]:
            matrix_np = np.vstack(new_matrix_np)
            if ticks_y is not None:
                ticks_y = new_ticks_y
            dots_rows.extend(new_dots_rows)
            dots_rows = sorted(list(set(dots_rows)))

    rows_dim = matrix_np.shape[0]
    cols_dim = matrix_np.shape[1] if len(matrix_np.shape) > 1 else 1

    should_truncate_rows = (rows_dim >= 15 and TRUNCATE)
    should_truncate_cols = (cols_dim >= 15 and TRUNCATE)

    if is_column and should_truncate_rows:
        if matrix_np.shape[0] > MAX_SIZE_COLUMN:
            truncated = True
            matrix_np = matrix_np[-MAX_SIZE_COLUMN:]
            if ticks_x is not None:
                ticks_x = ['...'] + list(ticks_x[-MAX_SIZE_COLUMN:])
                dots_col = np.full((DOTS_WIDTH,), np.nan)
                matrix_np = np.concatenate([dots_col, matrix_np], axis=0)
                dots_cols = list(range(DOTS_WIDTH))
        matrix_np = np.expand_dims(matrix_np, axis=0)
    elif (matrix_np.ndim == 2 and matrix_np.shape[0] == 1) and should_truncate_cols:
        if matrix_np.shape[1] > MAX_SIZE:
            truncated = True
            matrix_np = matrix_np[:, -MAX_SIZE:]
            if ticks_x is not None:
                ticks_x = ['...'] + list(ticks_x[-MAX_SIZE:])
                dots_col = np.full((matrix_np.shape[0], DOTS_WIDTH), np.nan)
                matrix_np = np.concatenate([dots_col, matrix_np], axis=1)
                dots_cols = list(range(DOTS_WIDTH))
    elif should_truncate_rows or should_truncate_cols:
        rows_to_keep = min(MAX_SIZE, matrix_np.shape[0]) if should_truncate_rows else matrix_np.shape[0]
        cols_to_keep = min(MAX_SIZE, matrix_np.shape[1]) if should_truncate_cols else matrix_np.shape[1]
        
        need_bottom_rows = should_truncate_rows and original_shape[0] > MAX_SIZE + BOTTOM_ROWS
        need_right_cols = should_truncate_cols and original_shape[1] > MAX_SIZE + RIGHT_COLS
        
        if need_bottom_rows or need_right_cols:
            truncated = True
            
            if need_bottom_rows and need_right_cols:
                top_left = matrix_np[:rows_to_keep, :cols_to_keep]
                top_right = matrix_np[:rows_to_keep, -RIGHT_COLS:]
                bottom_left = matrix_np[-BOTTOM_ROWS:, :cols_to_keep]
                bottom_right = matrix_np[-BOTTOM_ROWS:, -RIGHT_COLS:]
                
                v_dots = np.full((rows_to_keep, DOTS_WIDTH), np.nan)
                v_dots_bottom = np.full((BOTTOM_ROWS, DOTS_WIDTH), np.nan)
                
                h_dots = np.full((DOTS_HEIGHT, cols_to_keep + DOTS_WIDTH + RIGHT_COLS), np.nan)

                top_row = np.concatenate([top_left, v_dots, top_right], axis=1)
                bottom_row = np.concatenate([bottom_left, v_dots_bottom, bottom_right], axis=1)
                matrix_np = np.concatenate([top_row, h_dots, bottom_row], axis=0)

                dots_cols = list(range(cols_to_keep, cols_to_keep + DOTS_WIDTH))
                dots_rows = list(range(rows_to_keep, rows_to_keep + DOTS_HEIGHT))
                
                if ticks_y is not None:
                    ticks_y = list(ticks_y[:rows_to_keep]) + ['...'] * DOTS_HEIGHT + list(ticks_y[-BOTTOM_ROWS:])
                if ticks_x is not None:
                    ticks_x = list(ticks_x[:cols_to_keep]) + ['...'] * DOTS_WIDTH + list(ticks_x[-RIGHT_COLS:])
                    
            elif need_bottom_rows:
                top_part = matrix_np[:rows_to_keep, :]
                bottom_part = matrix_np[-BOTTOM_ROWS:, :]
                h_dots = np.full((DOTS_HEIGHT, matrix_np.shape[1]), np.nan)
                matrix_np = np.concatenate([top_part, h_dots, bottom_part], axis=0)
                
                dots_rows = list(range(rows_to_keep, rows_to_keep + DOTS_HEIGHT))
                
                if ticks_y is not None:
                    ticks_y = list(ticks_y[:rows_to_keep]) + ['...'] * DOTS_HEIGHT + list(ticks_y[-BOTTOM_ROWS:])
                    
            elif need_right_cols:
                left_part = matrix_np[:, :cols_to_keep]
                right_part = matrix_np[:, -RIGHT_COLS:]
                v_dots = np.full((matrix_np.shape[0], DOTS_WIDTH), np.nan)
                matrix_np = np.concatenate([left_part, v_dots, right_part], axis=1)

                dots_cols = list(range(cols_to_keep, cols_to_keep + DOTS_WIDTH))
                
                if ticks_x is not None:
                    ticks_x = list(ticks_x[:cols_to_keep]) + ['...'] * DOTS_WIDTH + list(ticks_x[-RIGHT_COLS:])
 
    if len(dots_rows) > 0 or len(dots_cols) > 0:
        rows_to_remove = []
        for i in range(matrix_np.shape[0] - 1):
            if np.all(np.isnan(matrix_np[i, :])) and np.all(np.isnan(matrix_np[i + 1, :])):
                rows_to_remove.append(i)
        if rows_to_remove:
            matrix_np = np.delete(matrix_np, rows_to_remove, axis=0)
            if ticks_y is not None:
                ticks_y = [t for i, t in enumerate(ticks_y) if i not in rows_to_remove]
            dots_rows = [r - sum(1 for removed in rows_to_remove if removed < r) for r in dots_rows if r not in rows_to_remove]
        
        cols_to_remove = []
        for j in range(matrix_np.shape[1] - 1):
            if np.all(np.isnan(matrix_np[:, j])) and np.all(np.isnan(matrix_np[:, j + 1])):
                cols_to_remove.append(j)
        if cols_to_remove:
            matrix_np = np.delete(matrix_np, cols_to_remove, axis=1)
            if ticks_x is not None:
                ticks_x = [t for i, t in enumerate(ticks_x) if i not in cols_to_remove]
            dots_cols = [c - sum(1 for removed in cols_to_remove if removed < c) for c in dots_cols if c not in cols_to_remove]

    num_x_ticks = matrix_np.shape[1] if ticks_x is None else len(ticks_x)
    num_y_ticks = matrix_np.shape[0] if ticks_y is None else len(ticks_y)
    
    if is_column or matrix_np.shape[0] == 1:
        width = max(6, min(num_x_ticks * 0.4, 20))
        height = 2.5
        figsize = (width, height)
    elif matrix_np.shape[1] == 1:
        width = 3
        height = max(4, min(num_y_ticks * 0.3, 16))
        figsize = (width, height)
    else:
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

    import matplotlib.colors as mcolors

    valid_data = matrix_np[~np.isnan(matrix_np)]
    data_range = np.max(valid_data) - np.min(valid_data) if len(valid_data) > 0 else 0
    has_negative = np.min(valid_data) < 0 if len(valid_data) > 0 else False
    
    base_cmap_name = 'Blues'
    
    base_cmap = plt.cm.get_cmap(base_cmap_name)
    colors = base_cmap(np.linspace(0.2, 1.0, 256))
    custom_cmap = mcolors.ListedColormap(colors)
    custom_cmap.set_bad(color=base_cmap(0.2))
    
    if matrix_np.ndim == 1:
        matrix_np = np.expand_dims(matrix_np, 0)
        dots_cols = dots_rows
        dots_rows = []
    im = ax.imshow(matrix_np, cmap=custom_cmap, aspect='auto')

    if add_causal_mask and matrix_np.shape[0] == matrix_np.shape[1]:
        import numpy as np
        mask = np.triu(np.ones_like(matrix_np), k=1)
        mask = mask.astype(float)
        mask[mask == 0] = np.nan
        ax.imshow(mask, cmap='Greys', alpha=1., aspect='auto', vmin=0, vmax=2)

    nan_mask = np.isnan(matrix_np)
    if np.any(nan_mask):
        for i in range(matrix_np.shape[0]):
            for j in range(matrix_np.shape[1]):
                if nan_mask[i, j]:
                    ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                                              fill=True,
                                              facecolor=base_cmap(0.2),
                                              edgecolor='gray',
                                              hatch='///', 
                                              linewidth=0.,
                                              alpha=0.3))

    for col in dots_cols:
        ax.add_patch(plt.Rectangle((col-0.5, -0.5), 1, matrix_np.shape[0], 
                                  fill=True,
                                  facecolor=base_cmap(0.2),
                                  edgecolor='gray',
                                  hatch='///', 
                                  linewidth=0.,
                                  alpha=0.3))
    for row in dots_rows:
        ax.add_patch(plt.Rectangle((-0.5, row-0.5), matrix_np.shape[1], 1, 
                                  fill=True,
                                  facecolor=base_cmap(0.2),
                                  edgecolor='gray',
                                  hatch='///', 
                                  linewidth=0.,
                                  alpha=0.3))
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.ax.tick_params(labelsize=14)
    
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
        char_width_ratio = 0.4  # Width of average char relative to font size
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
        return max(8, min(max_fontsize, 24))
    
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
        max_label_len = 3 # max(len(str(label)) for label in ticks_y) if ticks_y else 1
        
        # Get available height for ticks
        available_height = figsize[1] * 0.9  # Account for margins
        
        # Y-ticks are typically not rotated
        fontsize = calculate_optimal_fontsize(num_y_ticks, available_height, max_label_len, rotation_deg=0)
        
        tick_indices = range(num_y_ticks)
            
        ax.set_yticks([i for i in tick_indices])
        ax.set_yticklabels([ticks_y[i] for i in tick_indices], fontsize=fontsize)
        
        # Add subtle grid lines for y-axis too
        ax.set_yticks(range(matrix_np.shape[0]), minor=True)
    
    if is_column and ticks_y is None:
        ax.set_yticks([])
        ax.set_ylabel('')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_color('#666666')
    ax.spines['left'].set_color('#666666')

    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.1)
    plt.close(fig)