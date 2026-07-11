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
from plot_for_mlp_interpretation import get_mlp_input_output_latex_no_bins

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

TASK_TO_DESCRIPTION = {
    "012_0_2": r"""
    \begin{cases} 
        \langle\text{bos}\rangle\ s\ \langle\text{sep}\rangle\ 1\ & \text{if } s \in \{0, 1, 2\}^*02^* \\
        \langle\text{bos}\rangle\ s\ \langle\text{sep}\rangle\ 0\ & \text{otherwise}
    \end{cases}
    """,
    "bce_012_0_2": r"""
    \langle\text{bos}\rangle\ \{0, 1, 2\}^*02^* \langle\text{eos}\rangle
    """,
    "aaaastar": r"""
    \begin{cases} 
        \langle\text{bos}\rangle\ s\ \langle\text{sep}\rangle\ 1\ & \text{if } s \in (aaaa)^* \\
        \langle\text{bos}\rangle\ s\ \langle\text{sep}\rangle\ 0\ & \text{otherwise}
    \end{cases}
    """,
    "bce_aaaastar": r"""
    \langle\text{bos}\rangle\ (aaaa)^*\ \langle\text{eos}\rangle
    """,
    "aastar": r"""
    \begin{cases} 
        \langle\text{bos}\rangle\ s\ \langle\text{sep}\rangle\ 1\  & \text{if } s \in (aa)^* \\
        \langle\text{bos}\rangle\ s\ \langle\text{sep}\rangle\ 0\  & \text{otherwise}
    \end{cases}
    """,
    "bce_aastar": r"""
    \langle\text{bos}\rangle\ (aa)^*\ \langle\text{eos}\rangle
    """,
    "ab_d_bc": r"""
    \begin{cases} 
        \langle\text{bos}\rangle\ s\ \langle\text{sep}\rangle\ 1\ & \text{if } s \in \{a, b\}^*d\{b, c\}^* \\
        \langle\text{bos}\rangle\ s\ \langle\text{sep}\rangle\ 0\ & \text{otherwise}
    \end{cases}
    """,
    "bce_ab_d_bc": r"""
    \langle\text{bos}\rangle\ \{a, b\}^*d\{b, c\}^*\ \langle\text{eos}\rangle
    """,
    "ababstar": r"""
    \begin{cases} 
        \langle\text{bos}\rangle\ s\ \langle\text{sep}\rangle\ 1\ & \text{if } s \in (abab)^* \\
        \langle\text{bos}\rangle\ s\ \langle\text{sep}\rangle\ 0\ & \text{otherwise}
    \end{cases}
    """,
    "bce_ababstar": r"""
    \langle\text{bos}\rangle\ (abab)^*\ \langle\text{eos}\rangle
    """,
    "abcde": r"""
    \begin{cases} 
        \langle\text{bos}\rangle\ s\ \langle\text{sep}\rangle\ 1\ & \text{if } s \in aa^*bb^*cc^*dd^*ee^* \\
        \langle\text{bos}\rangle\ s\ \langle\text{sep}\rangle\ 0\ & \text{otherwise}
    \end{cases}
    """,
    "bce_abcde": r"""
    \langle\text{bos}\rangle\ aa^*bb^*cc^*dd^*ee^*\ \langle\text{eos}\rangle
    """,
    "bin_majority": r"""
    \langle\text{bos}\rangle\ s\ \langle\text{sep}\rangle\ \text{Maj}(s)\ \text{where } s \in \{0, 1\}^*
    """,
    "bin_majority_interleave": r"""
    \langle\text{bos}\rangle\ s\ \langle\text{sep}\rangle\ \text{Maj}(\{s_i| i \% 3\ =\ 0\})\text{Maj}(\{s_i| i \% 3\ =\ 1\})\text{Maj}(\{s_i| i \% 3\ =\ 2\})\ \langle\text{eos}\rangle \text{where } s \in \{0, 1\}^*
    """,
    "count": r"""
    \langle\text{bos}\rangle\ s_0, s_n\ \langle\text{sep}\rangle\ s_0 s_1 \dots s_n \ \langle\text{eos}\rangle \text{where } s_0, s_n \in \{0, 1, \dots, 150\}, s_n > s_0, s_{i + 1} = s_i + 1
    """,
    "D2": r"""
    \begin{cases} 
        \langle\text{bos}\rangle\ s\ \langle\text{sep}\rangle\ 1\ & \text{if } s \in (a(ab)^*b)^* \\
        \langle\text{bos}\rangle\ s\ \langle\text{sep}\rangle\ 0\ & \text{otherwise}
    \end{cases}
    """,
    "bce_D2": r"""
    \langle\text{bos}\rangle\ (a(ab)^*b)^*\ \langle\text{eos}\rangle
    """,
    "D3": r"""
    \begin{cases} 
        \langle\text{bos}\rangle\ s\ \langle\text{sep}\rangle\ 1\ & \text{if } s \in (a(a(ab)^*b)^*b)^* \\
        \langle\text{bos}\rangle\ s\ \langle\text{sep}\rangle\ 0\ & \text{otherwise}
    \end{cases}
    """,
    "bce_D3": r"""
    \langle\text{bos}\rangle\ (a(a(ab)^*b)^*b)^*\ \langle\text{eos}\rangle
    """,
    "D4": r"""
    \begin{cases} 
        \langle\text{bos}\rangle\ s\ \langle\text{sep}\rangle\ 1\ & \text{if } s \in (a(a(a(ab)^*b)^*b)^*b)^* \\
        \langle\text{bos}\rangle\ s\ \langle\text{sep}\rangle\ 0\ & \text{otherwise}
    \end{cases}
    """,
    "bce_D4": r"""
    \langle\text{bos}\rangle\ (a(a(a(ab)^*b)^*b)^*b)^*\ \langle\text{eos}\rangle
    """,
    "D12": r"""
    \begin{cases} 
        \langle\text{bos}\rangle\ s\ \langle\text{sep}\rangle\ 1\ & \text{if } s \in \underbrace{(a\dots a(a}_{12 times}\underbrace{b)^*)\dots b)^*}_{12 times} \\
        \langle\text{bos}\rangle\ s\ \langle\text{sep}\rangle\ 0\ & \text{otherwise}
    \end{cases}
    """,
    "bce_D12": r"""
    \langle\text{bos}\rangle\ \underbrace{(a\dots a(a}_{12 times}\underbrace{b)^*)\dots b)^*}_{12 times}\ \langle\text{eos}\rangle
    """,
    "majority": r"""
    \langle\text{bos}\rangle\ s\ \langle\text{sep}\rangle\ \text{Maj}(s)\ \text{where } s \in \{a, b, c, \dots , y, z\}^*
    """,
    "sort": r"""
    \langle\text{bos}\rangle\ s\ \langle\text{sep}\rangle\ s_{\sigma(0)}, s_{\sigma(0)} \dots s_{\sigma(n)}\ \langle\text{eos}\rangle \text{where } s_0 \in \{0, 1, \dots , 150\}, s_{i + 1} = s_i + 1, \sigma sorts s
    """,
    "tomita1": r"""
    \begin{cases} 
        \langle\text{bos}\rangle\ s\ \langle\text{sep}\rangle\ 1\  & \text{if } s \in 1^*
        \langle\text{bos}\rangle\ s\ \langle\text{sep}\rangle\ 0\ & \text{otherwise}
    \end{cases}
    """,
    "bce_tomita1": r"""
    \langle\text{bos}\rangle\ 1^*\ \langle\text{eos}\rangle
    """,
    "tomita2": r"""
    \begin{cases} 
        \langle\text{bos}\rangle\ s\ \langle\text{sep}\rangle\ 1\ & \text{if } s \in (10)^* \\
        \langle\text{bos}\rangle\ s\ \langle\text{sep}\rangle\ 0\ & \text{otherwise}
    \end{cases}
    """,
    "bce_tomita2": r"""
    \langle\text{bos}\rangle\ (10)^*\ \langle\text{eos}\rangle
    """,
    "tomita4": r"""
    \begin{cases} 
        \langle\text{bos}\rangle\ s\ \langle\text{sep}\rangle\ 1\ & \text{if } s \in \{0, 1\}*, 000 \notin s \\
        \langle\text{bos}\rangle\ s\ \langle\text{sep}\rangle\ 0\ & \text{otherwise}
    \end{cases}
    """,
    "bce_tomita4": r"""
    \langle\text{bos}\rangle\ s\ \langle\text{eos}\rangle,\ \text{where } s \in \{0, 1\}*, 000 \notin s
    """,
    "tomita7": r"""
    \begin{cases} 
        \langle\text{bos}\rangle\ s\ \langle\text{sep}\rangle\ 1\ & \text{if } s \in 0^*1^*0^*1^* \\
        \langle\text{bos}\rangle\ s\ \langle\text{sep}\rangle\ 0\ & \text{otherwise}
    \end{cases}
    """,
    "bce_tomita7": r"""
    \langle\text{bos}\rangle\ 0^*1^*0^*1^*\ \langle\text{eos}\rangle
    """,
    "unique_bigram_copy": r"""
    """,
    "unique_copy": r"""
    \langle\text{bos}\rangle\ s_0, s_1, \dots, s_n\ \langle\text{sep}\rangle\ s_0, s_1, \dots, s_n\ \langle\text{eos}\rangle \text{where } s_i \in \{0, 1, \dots , 150\}
    """,
    "unique_reverse": r"""
    \langle\text{bos}\rangle\ s_0, s_1, \dots, s_n\ \langle\text{sep}\rangle\ s_n, s_{n-1}, \dots, s_0\ \langle\text{eos}\rangle \text{where } s_i \in \{0, 1, \dots , 150\}
    """,
}

@dataclass
class Args:
    exp_path: str = ""  # Full path to the experiment folder (e.g., /scratch/abakalov/patching_runs_2/newprune-task-@arch-date/exp001)
    run_all: bool = False  # If True, process all models from good_models.json
    good_models_json: str = "good_models.json"  # Path to the good_models.json file

    results_base_dir: str = ""  # Base directory for results
    base_path: str = "PATH_TO_EXPERIMENTS"
    base_path_prefix: str = "matrices_primitive"  # Base path prefix for primary results in LaTeX
    
    copy_activations_first_base_path: bool = True

IS_ONLY_LARGE_SCALING = True

def extract_task_and_exp_name(exp_path: Path):
    """Extract task_name and exp_name from the path."""
    # The exp_path should be something like: .../newprune-task_name-@arch-date/exp_name
    exp_name = exp_path.name  # e.g., exp001
    parent_dir = exp_path.parent.name  # e.g., newprune-task_name-@arch-date
    
    # Extract task_name from parent directory (format: newprune-task_name-@...)
    parts = parent_dir.split("-")
    if len(parts) >= 2:
        task_name = f"{parts[1]}-{parts[2]}"  # Get the task name after "newprune-"
    else:
        raise ValueError(f"Cannot extract task_name from path: {parent_dir}")
    
    return task_name, exp_name

def copy_primitive_heatmaps(source_exp_path: Path, dest_path: Path, task_name: str):
    """Copy only the primitives-matrices subfolders from the experiment."""
    source_heatmaps = source_exp_path / "attention_primitives" / "heatmaps" / task_name.split("-")[0]
    
    if not source_heatmaps.exists():
        print(f"Warning: No heatmaps found at {source_heatmaps}")
        return False
    
    # Walk through the source directory and copy only primitives-matrices folders
    for root, dirs, files in os.walk(source_heatmaps):
        root_path = Path(root)
        
        # Check if we're in a primitives-matrices folder
        if root_path.name == "primitives-matrices":
            # Get relative path from source_heatmaps
            rel_path = root_path.relative_to(source_heatmaps)
            dest_folder = dest_path / rel_path
            dest_folder.mkdir(parents=True, exist_ok=True)
            
            # Copy all files in this primitives-matrices folder
            for file in files:
                if file.endswith('.pdf'):
                    src_file = root_path / file
                    dst_file = dest_folder / file
                    shutil.copy2(src_file, dst_file)
                    print(f"Copied: {src_file} -> {dst_file}")
    
    return True

def copy_activation_heatmaps(source_exp_path: Path, dest_path: Path, task_name: str):
    """Copy activation variables heatmaps from pretty_examples_positive subfolders."""
    source_heatmaps = source_exp_path / "attention_primitives" / "heatmaps" / "pretty_examples_positive" / task_name.split("-")[0]
    
    if not source_heatmaps.exists():
        print(f"Warning: No activation heatmaps found at {source_heatmaps}")
        return False
    
    # Walk through the source directory and copy files from primitives-example folders
    for root, dirs, files in os.walk(source_heatmaps):
        root_path = Path(root)
        
        # Check if we're in a primitives-example folder
        if root_path.name == "primitives-example":
            # Get relative path from source_heatmaps
            rel_path = root_path.relative_to(source_heatmaps)
            dest_folder = dest_path / rel_path
            dest_folder.mkdir(parents=True, exist_ok=True)
            
            # Copy all files in this primitives-example folder
            for file in files:
                if file.endswith('.pdf'):
                    src_file = root_path / file
                    dst_file = dest_folder / file
                    shutil.copy2(src_file, dst_file)
                    print(f"Copied activation heatmap: {src_file} -> {dst_file}")
    
    return True

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

def sanitize_for_latex_command(text: str) -> str:
    """Sanitize text for use in LaTeX commands (like color names, labels, etc)."""
    # Remove or replace characters that can't be in LaTeX command names
    text = text.replace('_', '')
    text = text.replace('-', '')
    text = text.replace('@', 'at')
    text = text.replace('#', 'hash')
    text = text.replace('^', 'caret')
    text = text.replace('&', 'and')
    text = text.replace('%', 'pct')
    text = text.replace('$', 'dollar')
    text = text.replace('[', '')
    text = text.replace(']', '')
    text = text.replace('{', '')
    text = text.replace('}', '')
    text = text.replace('~', '')
    text = text.replace('\\', '')
    return text

def get_display_task_name(task_name: str) -> str:
    """
    Convert task name to display format with suffix notation.
    
    Args:
        task_name: Full task name (e.g., "bce_tomita1-@1l1h16d3lr00drop")
    
    Returns:
        Display name with suffix (e.g., "bce_tomita1 : default")
    """
    # Extract just the task name without architecture for display
    # Split by @, ^, #, or % and take the first part (removing trailing -)
    display_task_name = re.split(r'[@^#%]', task_name)[0][:-1]
    # if display_task_name.endswith('-'):
    #     display_task_name = display_task_name[:-1]
    is_bce = False
    if display_task_name.startswith('bce_'):
        display_task_name = display_task_name[len('bce_'):]
        is_bce = True

    if display_task_name == "bin_majority":
        display_task_name = "Majority"
    elif display_task_name == "majority":
        display_task_name = "Most Frequent"
    elif display_task_name == "bin_majority_interleave":
        display_task_name = "Majority Interleave"
    elif display_task_name == "D2":
        display_task_name = "Dyck-2"
    elif display_task_name == "D4":
        display_task_name = "Dyck-4"
    elif display_task_name == "D12":
        display_task_name = "Dyck-12"
    elif not is_bce:
        display_task_name_parts = display_task_name.split("_")
        display_task_name_parts = [p.capitalize() for p in display_task_name_parts]
        display_task_name = " ".join(display_task_name_parts)

    if "@" in task_name:
        display_task_name = display_task_name
    elif "#" in task_name:
        display_task_name = display_task_name + " : different length generalization performance"
    elif "%" in task_name:
        display_task_name = display_task_name + " : different arch with same length generalization performance"
    elif "^" in task_name:
        display_task_name = display_task_name + " : different checkpoints"
    return display_task_name

def parse_model_architecture(task_name: str) -> dict:
    """
    Parse model architecture from task name.
    Format: task_name-@2l1h64d3lr01drop
    Where @ can be @, ^, or #
    Returns dict with: layers, heads, hidden_dim, learning_rate, dropout
    """
    # Split by separators @, ^, or #
    parts = re.split(r'[@^#]', task_name)
    if len(parts) < 2:
        return None
    
    arch_string = parts[1]  # e.g., "2l1h64d3lr01drop"
    
    # Parse architecture components
    arch_info = {}
    
    # Extract layers (e.g., "2l")
    layers_match = re.search(r'(\d+)l', arch_string)
    if layers_match:
        arch_info['layers'] = int(layers_match.group(1))
    
    # Extract heads (e.g., "1h")
    heads_match = re.search(r'(\d+)h', arch_string)
    if heads_match:
        arch_info['heads'] = int(heads_match.group(1))
    
    # Extract hidden dimension (e.g., "64d")
    dim_match = re.search(r'(\d+)d', arch_string)
    if dim_match:
        arch_info['hidden_dim'] = int(dim_match.group(1))
    
    # Extract learning rate (e.g., "3lr" -> 0.003)
    lr_match = re.search(r'(\d+)lr', arch_string)
    if lr_match:
        lr_value = int(lr_match.group(1))
        arch_info['learning_rate'] = 1 / (10**lr_value)
    
    # Extract dropout (e.g., "01drop" -> 0.1)
    dropout_match = re.search(r'(\d+)drop', arch_string)
    if dropout_match:
        dropout_value = int(dropout_match.group(1))
        assert dropout_match.group(1) in ["01", "00"], dropout_match.group(1)
        if dropout_match.group(1) == "01":
            arch_info['dropout'] = 0.1
        elif dropout_match.group(1) == "00":
            arch_info['dropout'] = 0
    
    return arch_info if arch_info else None

def get_color_palette():
    """Return a list of distinctive colors for variable highlighting."""
    # Using a diverse color palette (RGB values)
    colors = [
        (0.12, 0.47, 0.71),  # Blue
        (1.00, 0.50, 0.05),  # Orange
        (0.17, 0.63, 0.17),  # Green
        (0.84, 0.15, 0.16),  # Red
        (0.58, 0.40, 0.74),  # Purple
        (0.55, 0.34, 0.29),  # Brown
        (0.89, 0.47, 0.76),  # Pink
        (0.50, 0.50, 0.50),  # Gray
        (0.74, 0.74, 0.13),  # Olive
        (0.09, 0.75, 0.81),  # Cyan
        (0.00, 0.50, 0.50),  # Teal
        (0.90, 0.60, 0.00),  # Gold
        (0.60, 0.20, 0.80),  # Violet
        (0.95, 0.20, 0.20),  # Bright Red
        (0.20, 0.70, 0.30),  # Emerald
        (0.00, 0.40, 0.80),  # Navy
        (0.80, 0.40, 0.00),  # Dark Orange
        (0.40, 0.00, 0.40),  # Dark Purple
        (0.70, 0.70, 0.00),  # Yellow-Green
        (0.00, 0.60, 0.60),  # Dark Cyan
    ]
    return colors

def rgb_to_latex(rgb):
    """Convert RGB tuple to LaTeX color definition."""
    return f"{{rgb,1:red,{rgb[0]:.2f};green,{rgb[1]:.2f};blue,{rgb[2]:.2f}}}"

def colorize_variable_in_code(line: str, var_name: str, color_rgb: tuple) -> str:
    """Add color highlighting to a variable in a code line for LaTeX."""
    # Match the variable name at the beginning of assignment
    var_match = re.match(r'^([0-9]*\.\s*)('+  var_name + r')(\s*=.*)', line)
    if var_match:
        prefix = var_match.group(1)
        var = var_match.group(2)
        rest = var_match.group(3)
        # Use textcolor for inline coloring in verbatim-like environment
        # We'll need to use fancyvrb or listings package
        # For now, mark it with a special comment
        return f"{prefix}\\textcolor{rgb_to_latex(color_rgb)}{{{escape_latex_special_chars(var)}}}{rest}"
    return line

def write_standalone_figure_tex(dest_path: Path, filename: str, figure_content: list, task_name: str, base_path_prefix: str, sanitized_task_name: str, color_definitions):
    """
    Write a standalone LaTeX file for a figure that can be compiled to PDF separately.
    
    Args:
        dest_path: Destination directory for the .tex file
        filename: Name of the .tex file (without extension)
        figure_content: List of LaTeX lines containing the subfigure content (without figure environment)
        task_name: Original task name for caption
        base_path_prefix: Base path prefix for images
        sanitized_task_name: Sanitized task name for paths
        color_definitions: List of color definition commands
    """
    tex_file = dest_path / f"{filename}.tex"
    
    standalone_content = []
    standalone_content.append("\\documentclass{standalone}")
    standalone_content.append("\\usepackage{graphicx}")
    standalone_content.append("\\usepackage{subcaption}")
    standalone_content.append("\\usepackage{xcolor}")
    standalone_content.append("\\usepackage{tikz}")
    standalone_content.append("\\captionsetup[sub]{font=scriptsize}")
    standalone_content.append("\\newcommand{\\circled}[1]{\\tikz[baseline=(char.base)]{\\node[shape=circle,draw,inner sep=2pt] (char) {#1};}}")
    standalone_content.extend(color_definitions)
    standalone_content.append("")
    standalone_content.append("\\begin{document}")
    standalone_content.append("")
    
    # Wrap content in a minipage to control layout
    standalone_content.append("\\begin{minipage}{\\textwidth}")
    standalone_content.append("    \\centering")
    standalone_content.append("\\captionsetup{type=figure}")
    
    # Add the subfigure content directly
    standalone_content.extend(figure_content)
    
    standalone_content.append("\\end{minipage}")
    standalone_content.append("")
    standalone_content.append("\\end{document}")
    
    with open(tex_file, 'w') as f:
        f.write('\n'.join(standalone_content))
    
    print(f"Standalone figure TeX saved to: {tex_file}")

def generate_heatmap_figure_latex(heatmap_files, sanitized_task_name, config_to_heatmap, 
                                   var_to_color_name, var_name_to_code_line, var_to_op,
                                   task_name, sanitized_task_name_for_label,
                                   base_path_prefix, added_to_task_name, images_prefix):
    """
    Generate LaTeX code for a heatmap figure with multiple subfigures.
    
    Args:
        heatmap_files: List of heatmap file paths
        sanitized_task_name: Sanitized task name for file paths
        config_to_heatmap: Mapping from config keys to variable names
        var_to_color_name: Mapping from variable names to LaTeX color names
        var_name_to_code_line: Mapping from variable names to code line numbers
        escaped_task_name: Task name escaped for LaTeX display
        sanitized_task_name_for_label: Task name sanitized for LaTeX labels
        base_path_prefix: Base path prefix for images (e.g., "screenshots_3" or "screenshots_comparison")
    
    Returns:
        List of LaTeX lines for the figure
    """
    latex_content = []
    var_to_letter = {}
    last_letter = chr(ord('a') - 1)
    
    if not heatmap_files:
        return latex_content
    
    # Sort heatmaps by line number before plotting
    def get_line_number(heatmap_path):
        """Extract line number for a heatmap file."""
        filename = heatmap_path.stem
        if "primitives-matrices" not in str(heatmap_path):
            return 999999  # Put non-matrix heatmaps at the end
        
        file_start, file_end = str(heatmap_path).split("/primitives-matrices/")
        if file_start == "lm_head":
            prefix = "LOGITS."
        else:
            ah_match = re.match(r'(\d+)-(\d+)', file_start)
            if ah_match:
                l, h = ah_match.group(1), ah_match.group(2)
                prefix = f"ATTENTION-{l}-{h}."
            else:
                return 999999
        
        config_key = prefix + filename
        if config_key in config_to_heatmap:
            var_name = config_to_heatmap[config_key]
            if var_name in var_name_to_code_line:
                return var_name_to_code_line[var_name]
        return 999999
    
    # Sort heatmap files by line number
    heatmap_files = sorted(heatmap_files, key=get_line_number)

    tokenizer, _ = get_tokenizer_and_dataset_for_task(task_name.split("-")[0], (0, 150), 150, {"period_for_data":3})

    def is_tall(file_name):
        file_name = str(file_name)
        if file_name.count("wpe") == 1:
            return True
        if len(tokenizer.vocab) > 20 and file_name.count("wte") == 1 and file_name.count("wpe") == 0:
            return True
        if task_name.split("-")[0] == "unique_bigram_copy" and "mlp" in file_name:
            return True
        return False
    
    # Separate tall heatmaps
    normal_heatmaps = [p for p in heatmap_files if not is_tall(p) and not "bias" in str(p)]
    tall_heatmaps = [p for p in heatmap_files if is_tall(p) and not "bias" in str(p)]
    bias_heatmaps = [p for p in heatmap_files if "bias" in str(p)]

    print("len(normal_heatmaps), len(tall_heatmaps), len(bias_heatmaps)", len(normal_heatmaps), len(tall_heatmaps), len(bias_heatmaps))

    num_per_row = 3
    while True:
        num_normal_rows = len(normal_heatmaps) // num_per_row + int(len(normal_heatmaps) % num_per_row != 0)
        num_tall_rows = len(tall_heatmaps) // num_per_row + int(len(tall_heatmaps) % num_per_row != 0)
        num_bias_rows = len(bias_heatmaps) // 2 + int(len(bias_heatmaps) % 2 != 0)
        if ((num_normal_rows + num_bias_rows) <= 5 and num_tall_rows == 0):
            break
        if (num_normal_rows + num_bias_rows) <= 1 and num_tall_rows == 1:
            break
        if (num_normal_rows + num_bias_rows) <= 2 and num_per_row >= 4 and num_tall_rows == 1:
            break
        if (num_normal_rows + num_bias_rows) <= 3 and num_per_row >= 5 and num_tall_rows == 1:
            break
        if (num_normal_rows + num_bias_rows) <= 4 and num_per_row >= 6 and num_tall_rows == 1:
            break
        num_per_row += 1

    def generate_rows(files, is_tall_row, is_bias_row, last_letter, var_to_letter):
        if not files:
            return last_letter, var_to_letter

        # Calculate how many heatmaps per row (max 4 for normal, 1 for tall)
        num_heatmaps = len(files)
        if is_bias_row:
            heatmaps_per_row = min(2, num_heatmaps)
            width = 0.95 / 2
        else:
            heatmaps_per_row = min(num_per_row, num_heatmaps)
            if is_tall_row:
                width = 0.95 / max(heatmaps_per_row, 4)
            else:
                width = 0.95 / max(heatmaps_per_row, 3)
        
        for idx, heatmap_path in enumerate(files):
            # Convert path to use forward slashes for LaTeX
            latex_heatmap_path = str(heatmap_path).replace('\\', '/')
            
            # Extract the variable name from filename using config_to_heatmap
            filename = heatmap_path.stem
            if "primitives-matrices" not in str(heatmap_path):
                # this is an activation heatmap
                continue
            file_start, file_end = str(heatmap_path).split("/primitives-matrices/")
            if file_start == "lm_head":
                prefix = "LOGITS."
            else:
                ah_match = re.match(r'(\d+)-(\d+)', file_start)
                if ah_match:
                    l, h = ah_match.group(1), ah_match.group(2)
                    prefix = f"ATTENTION-{l}-{h}."
            var_name = config_to_heatmap[prefix + filename]

            last_letter = chr(ord(last_letter) + 1)
            var_to_letter[var_name] = last_letter
            
            # Create colored caption
            if var_name in var_to_color_name:
                color_name = var_to_color_name[var_name]
                colored_var = f"\\textcolor{{{color_name}}}{{{escape_latex_special_chars(var_name)}}}"
                op = var_to_op[var_name]
                if "circled" in op:
                    pattern = r"\\circled\{.\}"
                    replacement = r"\\circled{" + var_to_letter[var_name] + "}"
                    op = re.sub(pattern, replacement, op)
                    colored_op = f"\\textcolor{{{color_name}}}{{{op}}} in \\textcolor{{{color_name}}}{{{escape_latex_special_chars(var_name)}}}"
                else:
                    colored_op = f"\\textcolor{{{color_name}}}{{{escape_latex_special_chars(op)}}}"
            else:
                colored_var = escape_latex_special_chars(var_name)
                op = var_to_op[var_name]
                if "circled" in op:
                    colored_op = f"Operation in {escape_latex_special_chars(var_name)}"
                else:
                    colored_op = escape_latex_special_chars(escape_latex_special_chars(op))
            
            if var_name in var_name_to_code_line:
                caption = escape_latex_special_chars(f"Line {var_name_to_code_line[var_name]}: ") + colored_op
            else:
                caption = colored_op
            
            latex_content.append("    \\begin{subfigure}[b]{" + f"{width:.2f}" + "\\textwidth}")
            latex_content.append("        \\centering")
            latex_content.append(f"        \\includegraphics[width=\\linewidth]{{{images_prefix}{latex_heatmap_path}}}")
            latex_content.append(f"        \\caption{{{{{caption}}}}}")
            latex_content.append("    \\end{subfigure}")
            
            # Add hfill between subfigures (but not after the last one in a row)
            if (idx + 1) % heatmaps_per_row != 0 and idx < num_heatmaps - 1:
                latex_content.append("    \\hfill")
            elif (idx + 1) % heatmaps_per_row == 0 and idx < num_heatmaps - 1:
                # Add a line break for the next row, but use % to prevent paragraph breaks
                latex_content.append("    \\\\[1em]")
        return last_letter, var_to_letter

    if not bias_heatmaps and len(tall_heatmaps) < num_per_row and len(tall_heatmaps) + len(normal_heatmaps) % num_per_row <= num_per_row:
        tall_heatmaps = normal_heatmaps[-len(normal_heatmaps) % num_per_row:] + tall_heatmaps
        normal_heatmaps = normal_heatmaps[:-len(normal_heatmaps) % num_per_row]

    last_letter, var_to_letter = generate_rows(normal_heatmaps, is_tall_row=False, is_bias_row=False,
                                                              last_letter=last_letter, var_to_letter=var_to_letter)
    if normal_heatmaps and bias_heatmaps:
        latex_content.append("    \\\\[1em]") # Add space between normal and tall rows
    last_letter, var_to_letter = generate_rows(bias_heatmaps, is_tall_row=False, is_bias_row=True,
                                                              last_letter=last_letter, var_to_letter=var_to_letter)
    if (bias_heatmaps or normal_heatmaps) and tall_heatmaps:
        latex_content.append("    \\\\[1em]") # Add space between normal and tall rows
    last_letter, var_to_letter = generate_rows(tall_heatmaps, is_tall_row=True, is_bias_row=False,
                                                              last_letter=last_letter, var_to_letter=var_to_letter)

    return latex_content, var_to_letter

def get_var_to_op(code_lines, var_to_color_name):
    var_to_op = {}
    for line in code_lines:
        # Check if this line defines a variable that has a heatmap
        # Match: "1. varname = ..." or "1. varname[stuff][more] = ..."
        var_match = re.match(r'^(\d+)\.\s+([^\s=]+(?:\[[^\]]*\])*)\s*=', line)
        if var_match:
            var_name = var_match.group(2)
            
            if var_name in var_to_color_name:
                color_name = var_to_color_name[var_name]
                # Escape special characters for verbatim but keep color commands
                # Need to escape brackets, underscores, etc.
                escaped_var = var_name.replace('_', r'\_').replace('[', r'[').replace(']', r']')
                prefix = line[:line.find(var_name)]
                rest = line[line.find(var_name) + len(var_name):]
                
                # Also highlight operations in the rest of the line
                # Match patterns like: op=\circled{...}, op=(...), special_op=(...)
                op_pattern = r'((?:special_)?op=(?:\((?:[^)]+)\)|\\circled\{[^}]+\}),?)'
                
                # Find all operation matches in the rest of the line
                highlighted_rest = rest

                full_op_name = ""
                
                for op_match in re.finditer(op_pattern, rest):
                    op_text = op_match.group(1)
                    # Escape the operation text for verbatim
                    escaped_op = op_text.replace('_', r'\_')
                    # Wrap it in the same color as the variable
                    colored_op = f"\\textcolor{{{color_name}}}{{{escaped_op}}}"
                    highlighted_rest = highlighted_rest.replace(op_text, colored_op, 1)
                    if full_op_name:
                        full_op_name = full_op_name + " " + op_text
                    else:
                        full_op_name = full_op_name + op_text

                var_to_op[var_name] = full_op_name
    return var_to_op

def get_code_latex(code_lines, var_to_color_name, var_to_letter, is_bce):
    latex_content = []
    latex_content.append(f"\\paragraph{{Code}}")
    
    # Add code with color highlighting using fancyvrb
    latex_content.append("\\begin{Verbatim}[commandchars=\\\\\\{\\}]")
    var_to_op = {}
    for line in code_lines:
        if is_bce and "softmax" in line:
            line = re.sub("softmax", "sigmoid", line)
        # Check if this line defines a variable that has a heatmap
        # Match: "1. varname = ..." or "1. varname[stuff][more] = ..."
        var_match = re.match(r'^(\d+)\.\s+([^\s=]+(?:\[[^\]]*\])*)\s*=', line)
        if var_match:
            var_name = var_match.group(2)

            if var_name in var_to_letter and "circled" in line:
                pattern = r"\\circled\{.\}"
                replacement = r"\\circled{" + var_to_letter[var_name] + "}"
                line = re.sub(pattern, replacement, line)
            
            if var_name in var_to_color_name:
                color_name = var_to_color_name[var_name]
                # Escape special characters for verbatim but keep color commands
                # Need to escape brackets, underscores, etc.
                escaped_var = var_name.replace('_', r'\_').replace('[', r'[').replace(']', r']')
                prefix = line[:line.find(var_name)]
                rest = line[line.find(var_name) + len(var_name):]
                
                # Also highlight operations in the rest of the line
                # Match patterns like: op=\circled{...}, op=(...), special_op=(...)
                op_pattern = r'((?:special_)?op=(?:\((?:[^)]+)\)|\\circled\{[^}]+\}),?)'
                
                # Find all operation matches in the rest of the line
                highlighted_rest = rest

                full_op_name = ""
                
                for op_match in re.finditer(op_pattern, rest):
                    op_text = op_match.group(1)
                    # Escape the operation text for verbatim
                    escaped_op = op_text.replace('_', r'\_')
                    # Wrap it in the same color as the variable
                    colored_op = f"\\textcolor{{{color_name}}}{{{escaped_op}}}"
                    highlighted_rest = highlighted_rest.replace(op_text, colored_op, 1)
                    if full_op_name:
                        full_op_name = full_op_name + " " + op_text
                    else:
                        full_op_name = full_op_name + op_text

                var_to_op[var_name] = full_op_name
                
                colored_line = f"{prefix}\\textcolor{{{color_name}}}{{{escaped_var}}}{highlighted_rest}"
                latex_content.append(colored_line)
            else:
                latex_content.append(line)
        else:
            latex_content.append(line)
    latex_content.append("\\end{Verbatim}")
    return latex_content

def generate_latex_file(exp_path: Path, dest_path: Path, task_name: str, sanitized_task_name: str, code_lines: list, heatmap_to_config: dict, var_mapping: dict, base_path_prefix: str , deleted_config_lines, copy_activations_first_base_path: bool = False, primitives_accuracy=None, primitives_acc_match=None):
    """Generate a latex.tex file with formatted LaTeX code."""
    latex_path = dest_path / "latex.tex"
    
    # Find all heatmap files and organize them
    heatmap_files = []
    for root, dirs, files in os.walk(dest_path):
        if root.endswith("mlps"):
            continue
        for file in files:
            if file.endswith('.pdf'):
                rel_path = Path(root).relative_to(dest_path) / file
                heatmap_files.append(rel_path)
    
    # Sort heatmaps: first by layer-head, then by type (qk interactions, then k interactions, then lm_head)
    def sort_key(path):
        path_str = str(path)
        # Extract layer-head if present
        layer_head_match = re.search(r'(\d+)-(\d+)', path_str)
        if layer_head_match:
            layer = int(layer_head_match.group(1))
            head = int(layer_head_match.group(2))
            # Bias interactions come after regular interactions
            if 'bias-' in path_str:
                return (layer, head, 1, path_str)
            else:
                return (layer, head, 0, path_str)
        elif 'lm_head' in path_str:
            # lm_head comes last
            if 'bias' in path_str:
                return (999, 999, 1, path_str)
            else:
                return (999, 999, 0, path_str)
        else:
            return (9999, 9999, 9999, path_str)
    
    heatmap_files.sort(key=sort_key)
    
    # Parse code to extract variable names for captions
    # Match format: "number. variable_name = ..." where variable_name can include underscores
    var_name_to_code_line = {}
    for line_i, line in enumerate(code_lines):
        # Match: "1. varname = ..." or "1. varname_stuff_more = ..."
        # Capture everything between the line number and the equals sign (including underscores)
        var_match = re.match(r'^(\d+)\.\s+(\w+)\s*=', line)
        if var_match:
            line_number = int(var_match.group(1))
            var_name = var_match.group(2)
            assert var_name not in var_name_to_code_line, (var_name, line, var_name_to_code_line)
            var_name_to_code_line[var_name] = line_number
            print("var_match", line, var_name, line_number)
        else:
            print("not var_match", line)

    print("var_name_to_code_line", var_name_to_code_line)
    print("heatmap_to_config", heatmap_to_config)

    config_to_heatmap = {}
    for h, c in heatmap_to_config.items():
        if type(c) == str:
            if c == "vocab_bias":
                config_to_heatmap["LOGITS." + "bias"] = h
            else:
                config_to_heatmap["LOGITS." + c] = h
        elif type(c[-1]) == str:
            config_to_heatmap[f"ATTENTION-{c[0]}-{c[1]}." + "bias-" + c[-1]] = h
        else:
            config_to_heatmap[f"ATTENTION-{c[0]}-{c[1]}." + f"{c[-1][0]}-{c[-1][1]}"] = h

    config_to_deleted_heatmaps = {}
    for h, c in deleted_config_lines.items():
        if type(c) == str:
            if c == "vocab_bias":
                config_to_deleted_heatmaps["LOGITS." + "bias"] = h
            else:
                config_to_deleted_heatmaps["LOGITS." + c] = h
        elif type(c[-1]) == str:
            config_to_deleted_heatmaps[f"ATTENTION-{c[0]}-{c[1]}." + "bias-" + c[-1]] = h
        else:
            config_to_deleted_heatmaps[f"ATTENTION-{c[0]}-{c[1]}." + f"{c[-1][0]}-{c[-1][1]}"] = h

    print("config_to_heatmap", config_to_heatmap)
    print("config_to_deleted_heatmaps", config_to_deleted_heatmaps)
    
    print("heatmap_files", [str(f) for f in heatmap_files])
    print("var_mapping", var_mapping)

    var_name_to_code_line["token"] = 0
    var_name_to_code_line["pos"] = 0

    config_to_code_line = {
        c: var_name_to_code_line[h]
        for c, h in config_to_heatmap.items()
    }
    config_to_code_line.update({
        f"MLP." + v: var_name_to_code_line[p]
        for v, p in var_mapping.items() if v.startswith("mlp")
    })
    print("config_to_code_line", config_to_code_line)

    config_to_deleted_code_line = {
        c: var_name_to_code_line[h]
        for c, h in config_to_deleted_heatmaps.items()
    }
    print("config_to_deleted_code_line", config_to_deleted_code_line)

    new_heatmap_files = []
    for file in heatmap_files:
        if "primitives-matrices" not in str(file):
            # not an activation heatmap
            continue
        file_start, file_end = str(file).split("/primitives-matrices/")
        if file_start == "lm_head":
            if "LOGITS." + file.stem in config_to_heatmap:
                new_heatmap_files.append(file)
        else:
            ah_match = re.match(r'(\d+)-(\d+)', file_start)
            if ah_match:
                l, h = ah_match.group(1), ah_match.group(2)
                if f"ATTENTION-{l}-{h}." + file.stem in config_to_heatmap:
                    new_heatmap_files.append(file)
    heatmap_files = new_heatmap_files
    print("heatmap_files_after", [str(f) for f in heatmap_files])
    
    # Create color mapping: map each variable to a color
    # Map directly from heatmap_to_config which has the actual variable names
    color_palette = get_color_palette()
    var_to_color = {}
    var_to_color_name = {}
    
    # Assign colors to variables based on heatmap_to_config mapping
    color_idx = 0
    for var_name in config_to_heatmap.values():
        var_to_color[var_name] = color_palette[color_idx % len(color_palette)]
        # Create a unique color name for this variable (sanitize variable name and task name)
        sanitized_var_name = sanitize_for_latex_command(var_name)
        sanitized_task_name_cmd = sanitize_for_latex_command(task_name)
        var_to_color_name[var_name] = f"varcolor{sanitized_task_name_cmd}{sanitized_var_name}"
        color_idx += 1
    
    print("var_to_color", var_to_color)
    
    # Start building the LaTeX content
    latex_content = []
    
    # Add required packages for color support
    latex_content.append("% Add these packages to your preamble:")
    latex_content.append("% \\usepackage{xcolor}")
    latex_content.append("% \\usepackage{fancyvrb}")
    latex_content.append("% \\usepackage{subcaption}")
    latex_content.append("% \\usepackage{tabularx}")
    latex_content.append("% \\usepackage{booktabs}")
    # Define all colors used in this task
    latex_content.append("% Color definitions for this task")
    color_definitions = []
    for var_name, color_name in var_to_color_name.items():
        rgb = var_to_color[var_name]
        color_def = f"\\definecolor{{{color_name}}}{{rgb}}{{{rgb[0]:.2f},{rgb[1]:.2f},{rgb[2]:.2f}}}"
        latex_content.append(color_def)
        color_definitions.append(color_def)
    # Escape task_name for display in LaTeX
    escaped_task_name = escape_latex_special_chars(task_name)
    sanitized_task_name_for_label = sanitize_for_latex_command(task_name)
    
    # Extract just the task name without architecture for display
    escaped_display_task_name = escape_latex_special_chars(get_display_task_name(task_name))
    
    # Add subsection with task name
    latex_content.append(f"\\subsection{{{escaped_display_task_name}}}")
    latex_content.append(f"\\label{{{sanitized_task_name_for_label}section}}")
    
    # Parse model architecture information
    arch_info = parse_model_architecture(task_name)
    
    # Build the table with task description, model architecture, and accuracy
    
    # Build model architecture cell
    arch_line1 = ""
    arch_line2 = ""
    if arch_info:
        arch_parts_line1 = []
        arch_parts_line2 = []
        if 'layers' in arch_info:
            arch_parts_line1.append(f"Layers: {arch_info['layers']}")
        if 'heads' in arch_info:
            arch_parts_line1.append(f"Heads: {arch_info['heads']}")
        if 'hidden_dim' in arch_info:
            arch_parts_line1.append(f"Hidden Dim: {arch_info['hidden_dim']}")
        if 'learning_rate' in arch_info:
            arch_parts_line2.append(f"LR: {arch_info['learning_rate']}")
        if 'dropout' in arch_info:
            arch_parts_line2.append(f"Dropout: {arch_info['dropout']}")
        
        arch_line1 = " \\quad ".join(arch_parts_line1)
        arch_line2 = " \\quad ".join(arch_parts_line2)

    # 1. Build the Architecture String
    arch_parts = []
    if arch_line1:
        arch_parts.append(arch_line1)
    if arch_line2:
        arch_parts.append(arch_line2)
    # Join architecture lines with a space or a comma
    arch_text = " ".join(arch_parts)

    # 2. Build the Metrics String using Arrow Notation
    metric_parts = []
    
    if primitives_accuracy is not None:
        p_before = primitives_accuracy['before'] if isinstance(primitives_accuracy, dict) else 0.0
        p_after = primitives_accuracy['after'] if isinstance(primitives_accuracy, dict) else 0.0
        # Format: Task Accuracy: 0.85 -> 0.84
        metric_parts.append(f"Task Accuracy: ${p_before:.2f} \\to {p_after:.2f}$")

    if primitives_acc_match is not None:
        m_before = primitives_acc_match['before'] if isinstance(primitives_acc_match, dict) else 0.0
        m_after = primitives_acc_match['after'] if isinstance(primitives_acc_match, dict) else 0.0
        # Format: Match Accuracy: 0.70 -> 0.72
        metric_parts.append(f"Match Accuracy: ${m_before:.2f} \\to {m_after:.2f}$")

    # Join metrics with a separator (semicolon looks best in prose)
    metrics_text = "; ".join(metric_parts)

    latex_content.append("")
    latex_content.append("    % --- Text Summary Block ---")
    latex_content.append(r"    \par\vspace{0.5em}\noindent")
    latex_content.append(f"    \\textbf{{Task Description:}} $${TASK_TO_DESCRIPTION[task_name.split('-')[0]]}$$ \\\\")
    
    if arch_text:
        latex_content.append(f"    \\textbf{{Architecture:}} {arch_text} \\\\")
        
    if metrics_text:
        # We explicitly mention the direction of the arrow (Pruning -> Primitives)
        latex_content.append(f"    \\textbf{{Performance (w/Pruning $\\to$ w/Primitives):}} {metrics_text}")
    
    latex_content.append(r"    \vspace{0.5em}\par")
    latex_content.append("")
    
    var_to_op = get_var_to_op(code_lines, var_to_color_name)

    secondary_heatmap_latex, primary_heatmap_latex = [], []
    
    # Add first figure environment with heatmaps from dest_path
    var_to_letter = {}
    if heatmap_files:
        primary_heatmap_latex, var_to_letter = generate_heatmap_figure_latex(
            heatmap_files, sanitized_task_name, config_to_heatmap,
            var_to_color_name, var_name_to_code_line, var_to_op,
            task_name, sanitized_task_name_for_label,
            base_path_prefix=base_path_prefix, added_to_task_name="",
            images_prefix=""
        )
                
    print("var_to_letter", var_to_letter)
    code_latex = get_code_latex(code_lines, var_to_color_name, var_to_letter, is_bce=("bce" in task_name))

    latex_content.append("")
    latex_content.extend(code_latex)
    latex_content.append("")
    latex_content.append("")


    if primary_heatmap_latex:
        # Write standalone .tex file for primary heatmaps
        write_standalone_figure_tex(
            dest_path, "heatmaps-primary", primary_heatmap_latex,
            task_name, base_path_prefix, sanitized_task_name, color_definitions=color_definitions
        )
        
        # Add reference to the PDF in main latex file
        latex_content.append("")
        latex_content.append("\\begin{figure}[H]")
        latex_content.append("    \\centering")
        latex_content.append(f"% Compile {sanitized_task_name}/heatmaps-primary.tex separately to generate heatmaps-primary.pdf")
        latex_content.append(f"\\includegraphics[width=\\textwidth]{{{base_path_prefix}/{sanitized_task_name}/heatmaps-primary.pdf}}")
        latex_content.append(f"    \\caption{{Heatmaps supporting the program for {escape_latex_special_chars(get_display_task_name(task_name))} model.}}")
        latex_content.append(f"    \\label{{fig:{sanitized_task_name_for_label}}}")
        latex_content.append("\\end{figure}")
        latex_content.append("")

    # Add third figure from activation_base_dir if provided (activation variables heatmaps)
    if copy_activations_first_base_path:
        activation_base_path = Path(exp_path) / "attention_primitives" / "heatmaps" / "pretty_examples_positive" / task_name.split("-")[0]
        assert activation_base_path.exists(), activation_base_path
        if activation_base_path.exists():
            # Find heatmaps in the activation variables directory
            activation_heatmap_files = []
            for root, dirs, files in os.walk(activation_base_path):
                if root.endswith("mlps"):
                    continue
                for file in files:
                    if file.endswith('.pdf'):
                        rel_path = Path(root).relative_to(activation_base_path) / file
                        activation_heatmap_files.append(rel_path)
            
            activation_heatmap_files.sort(key=sort_key)
            
            # Filter to only matching heatmaps (same logic as comparison heatmaps)
            new_activation_heatmap_files = []
            for file in activation_heatmap_files:
                # Check if this is from primitives-example folder (activation vars are in primitives-example, not primitives-matrices)
                if "primitives-example" in str(file):
                    new_activation_heatmap_files.append(file)
            activation_heatmap_files = new_activation_heatmap_files
            
            if activation_heatmap_files:
                cleaned_activation_heatmap_files = []
                for idx, heatmap_path in enumerate(activation_heatmap_files):
                    filename = heatmap_path.stem
                    path_parts = str(heatmap_path).split("/primitives-example/")
                    if len(path_parts) != 2:
                        continue
                    file_start = path_parts[0]
                    if file_start == "lm_head":
                        prefix = "LOGITS."
                    else:
                        ah_match = re.match(r'(\d+)-(\d+)', file_start)
                        if ah_match:
                            l, h = ah_match.group(1), ah_match.group(2)
                            prefix = f"ATTENTION-{l}-{h}."
                        else:
                            # Fallback to just filename if pattern doesn't match
                            prefix = ""
                    
                    # Try to find the variable name in config_to_heatmap
                    config_key = prefix + filename
                    if config_key in config_to_heatmap:
                        print("config_key passed (config_to_heatmap)", config_key)
                        cleaned_activation_heatmap_files.append(heatmap_path)
                    elif config_key in config_to_deleted_heatmaps:
                        print("config_key passed (config_to_deleted_heatmaps)", config_key)
                        cleaned_activation_heatmap_files.append(heatmap_path)
                    elif filename.startswith("aggregation-"):
                        print("config_key passed (agg)", config_key)
                        agg_path = filename[len("aggregation-"):]
                        if agg_path in var_mapping:
                            cleaned_activation_heatmap_files.append(heatmap_path)

                activation_heatmap_files = cleaned_activation_heatmap_files

            def get_line_number(heatmap_path):
                """Extract line number for a heatmap file."""
                filename = heatmap_path.stem
                if "primitives-example" not in str(heatmap_path):
                    return 999999  # Put non-matrix heatmaps at the end
                
                file_start, file_end = str(heatmap_path).split("/primitives-example/")
                if file_start == "lm_head":
                    prefix = "LOGITS."
                else:
                    ah_match = re.match(r'(\d+)-(\d+)', file_start)
                    if ah_match:
                        l, h = ah_match.group(1), ah_match.group(2)
                        prefix = f"ATTENTION-{l}-{h}."
                    else:
                        return 999999
                
                config_key = prefix + filename
                if config_key in config_to_heatmap:
                    var_name = config_to_heatmap[config_key]
                    if var_name in var_name_to_code_line:
                        return var_name_to_code_line[var_name]
                elif config_key in config_to_deleted_heatmaps:
                    var_name = config_to_deleted_heatmaps[config_key]
                    if var_name in var_name_to_code_line:
                        return var_name_to_code_line[var_name]
                elif filename.startswith("aggregation-"):
                    agg_path = filename[len("aggregation-"):]
                    var_name = var_mapping[agg_path]
                    if var_name in var_name_to_code_line:
                        return var_name_to_code_line[var_name]
                return 999999
            
            # Sort heatmap files by line number
            activation_heatmap_files = sorted(activation_heatmap_files, key=get_line_number)
            print("activation_heatmap_files", activation_heatmap_files)

            if activation_heatmap_files:
                # Generate activation variables heatmap figure
                activation_latex = []
                
                tokenizer, _ = get_tokenizer_and_dataset_for_task(task_name.split("-")[0], (0, 150), 150, {"period_for_data":3})

                def is_tall(file_name):
                    file_name = str(file_name)
                    if file_name.count("wpe") >= 1 and ("aggregation" in file_name):
                        return True
                    if len(tokenizer.vocab) > 20 and ("aggregation" in file_name) and file_name.count("wte") >= 1:
                        return True
                    if len(tokenizer.vocab) > 20 and ("lm_head" in file_name):
                        return True
                    if task_name.split("-")[0] == "unique_bigram_copy" and "mlp" in file_name:
                        return True
                    return False

                normal_heatmaps = [p for p in activation_heatmap_files if not is_tall(p)]
                tall_heatmaps = [p for p in activation_heatmap_files if is_tall(p)]
                bias_heatmaps = []
                print("len(normal_heatmaps), len(tall_heatmaps), len(bias_heatmaps)", len(normal_heatmaps), len(tall_heatmaps), len(bias_heatmaps))

                
                num_per_row = 3
                while True:
                    num_normal_rows = len(normal_heatmaps) // num_per_row + int(len(normal_heatmaps) % num_per_row != 0)
                    num_tall_rows = len(tall_heatmaps) // num_per_row + int(len(tall_heatmaps) % num_per_row != 0)
                    num_bias_rows = len(bias_heatmaps) // 2 + int(len(bias_heatmaps) % 2 != 0)
                    if ((num_normal_rows + num_bias_rows) <= 5 and num_tall_rows == 0):
                        break
                    if (num_normal_rows + num_bias_rows) <= 1 and num_tall_rows == 1:
                        break
                    if (num_normal_rows + num_bias_rows) <= 2 and num_per_row >= 4 and num_tall_rows == 1:
                        break
                    if (num_normal_rows + num_bias_rows) <= 3 and num_per_row >= 5 and num_tall_rows == 1:
                        break
                    if (num_normal_rows + num_bias_rows) <= 4 and num_per_row >= 5 and num_tall_rows == 1:
                        break
                    num_per_row += 1

                if not bias_heatmaps and len(tall_heatmaps) < num_per_row and len(normal_heatmaps) % num_per_row > 0 and len(tall_heatmaps) + len(normal_heatmaps) % num_per_row <= num_per_row:
                    tall_heatmaps = normal_heatmaps[-(len(normal_heatmaps) % num_per_row):] + tall_heatmaps
                    normal_heatmaps = normal_heatmaps[:-(len(normal_heatmaps) % num_per_row)]

                print("after rearrange tall and normal: len(normal_heatmaps), len(tall_heatmaps), len(bias_heatmaps)", len(normal_heatmaps), len(tall_heatmaps), len(bias_heatmaps))

                
                def generate_rows(files, is_tall_row, is_bias_row):
                    if not files:
                        return
                    # Calculate how many heatmaps per row (max 4)
                    num_heatmaps = len(files)
                    if is_bias_row:
                        heatmaps_per_row = min(2, num_heatmaps)
                        width = 0.95 / 2
                    else:
                        heatmaps_per_row = min(num_per_row, num_heatmaps)
                        if is_tall_row:
                            width = 0.95 / max(heatmaps_per_row, 4)
                        else:
                            width = 0.95 / max(heatmaps_per_row, 3)
                    
                    for idx, heatmap_path in enumerate(files):
                        latex_heatmap_path = str(heatmap_path).replace('\\', '/')
                        
                        # Extract variable name from path using config_to_heatmap
                        filename = heatmap_path.stem
                        
                        # Parse the path to get the config key (similar to primitives-matrices logic)
                        path_parts = str(heatmap_path).split("/primitives-example/")
                        print("path_parts", path_parts)
                        if len(path_parts) == 2:
                            file_start = path_parts[0]
                            if file_start == "lm_head":
                                prefix = "LOGITS."
                            else:
                                ah_match = re.match(r'(\d+)-(\d+)', file_start)
                                if ah_match:
                                    l, h = ah_match.group(1), ah_match.group(2)
                                    prefix = f"ATTENTION-{l}-{h}."
                                else:
                                    # Fallback to just filename if pattern doesn't match
                                    prefix = ""
                            
                            # Try to find the variable name in config_to_heatmap
                            config_key = prefix + filename
                            print("config_key", config_key)
                            if config_key in config_to_heatmap or filename.startswith("aggregation-") or config_key in config_to_deleted_heatmaps:
                                if filename.startswith("aggregation-"):
                                    agg_path = filename[len("aggregation-"):]
                                    var_name = var_mapping[agg_path]
                                elif config_key in config_to_deleted_heatmaps:
                                    var_name = config_to_deleted_heatmaps[config_key]
                                else:
                                    var_name = config_to_heatmap[config_key]

                                print("var_name", var_name)
                            
                                # Create colored caption with line number
                                if var_name in var_to_color_name:
                                    color_name = var_to_color_name[var_name]
                                    colored_var = f"\\textcolor{{{color_name}}}{{{escape_latex_special_chars(var_name)}}}"
                                else:
                                    colored_var = escape_latex_special_chars(var_name)
                                
                                if var_name in var_name_to_code_line:
                                    caption = escape_latex_special_chars(f"Line {var_name_to_code_line[var_name]}: ") + colored_var
                                elif config_key in config_to_deleted_code_line:
                                    caption = escape_latex_special_chars(f"Line {config_to_deleted_code_line[config_key]}: ") + colored_var
                                else:
                                    caption = colored_var
                            else:
                                # Fallback to filename if not found in mapping
                                caption = escape_latex_special_chars(filename)
                        else:
                            # Fallback to filename if path parsing fails
                            caption = escape_latex_special_chars(filename)
                        
                        activation_latex.append("    \\begin{subfigure}[b]{" + f"{width:.2f}" + "\\textwidth}")
                        activation_latex.append("        \\centering")
                        # activation_latex.append(f"        \\includegraphics[width=\\linewidth]{{{base_path_prefix}/{sanitized_task_name}/{latex_heatmap_path}}}")
                        activation_latex.append(f"        \\includegraphics[width=\\linewidth]{{{latex_heatmap_path}}}")
                        activation_latex.append(f"        \\caption{{{{{caption}}}}}")
                        activation_latex.append("    \\end{subfigure}")
                    
                        if (idx + 1) % heatmaps_per_row != 0 and idx < num_heatmaps - 1:
                            activation_latex.append("    \\hfill")
                        elif (idx + 1) % heatmaps_per_row == 0 and idx < num_heatmaps - 1:
                            activation_latex.append("    \\\\[1em]")

                generate_rows(normal_heatmaps, is_tall_row=False, is_bias_row=False)
                if normal_heatmaps and bias_heatmaps:
                    activation_latex.append("    \\\\[1em]") # Add space between normal and tall rows
                generate_rows(bias_heatmaps, is_tall_row=False, is_bias_row=True)
                if (bias_heatmaps or normal_heatmaps) and tall_heatmaps:
                    activation_latex.append("    \\\\[1em]") # Add space between normal and tall rows
                generate_rows(tall_heatmaps, is_tall_row=True, is_bias_row=False)

                # activation_latex.append(f"    \\caption{{Variables Heatmaps for {escape_latex_special_chars(get_display_task_name(task_name))} model on an example input.}}")
                # activation_latex.append(f"    \\label{{fig:{sanitized_task_name_for_label}activation}}")
                # activation_latex.append("\\end{figure}")
                
                # Write standalone .tex file for activation heatmaps
                write_standalone_figure_tex(
                    dest_path, "activation-vars", activation_latex,
                    task_name, base_path_prefix, sanitized_task_name, color_definitions=color_definitions
                )
                
                # Add reference to the PDF in main latex file
                latex_content.append("")
                latex_content.append("\\begin{figure}[H]")
                latex_content.append("    \\centering")
                latex_content.append(f"% Compile {sanitized_task_name}/activation-vars.tex separately to generate activation-vars.pdf")
                latex_content.append(f"\\includegraphics[width=\\textwidth]{{{base_path_prefix}/{sanitized_task_name}/activation-vars.pdf}}")
                latex_content.append(f"    \\caption{{Variables Heatmaps for {escape_latex_special_chars(get_display_task_name(task_name))} model on an example input.}}")
                latex_content.append(f"    \\label{{fig:{sanitized_task_name_for_label}activation}}")
                latex_content.append("\\end{figure}")
                latex_content.append("")


    latex_content.extend(get_mlp_input_output_latex_no_bins(exp_path, task_name.split("-")[0], config_to_code_line, config_to_deleted_code_line, dest_path, sanitized_task_name, base_path_prefix, deleted_config_lines))

    # Write to file
    with open(latex_path, 'w') as f:
        f.write('\n'.join(latex_content))
    
    print(f"LaTeX file saved to: {latex_path}")
    return color_definitions  # Return color definitions for combining

def generate_code_file(exp_path: Path, dest_path: Path):
    """Generate code from the experiment config and save to a text file."""
    # Load output.json
    output_json_path = exp_path / "output.json"
    if not output_json_path.exists():
        print(f"Warning: output.json not found at {output_json_path}")
        return False
    
    with open(output_json_path, 'r') as f:
        output_data = json.load(f)
    
    # Check if we have the final config
    if "result_patching_config_global_iteration_2" not in output_data:
        print(f"Warning: result_patching_config_global_iteration_2 not found in output.json")
        return False
    
    config = output_data["result_patching_config_global_iteration_2"]
    
    # Load config.yaml to get split_mlps setting
    config_yaml_path = exp_path / "config.yaml"
    split_mlps = False
    if config_yaml_path.exists():
        with open(config_yaml_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
            split_mlps = yaml_config.get("split_mlps", False)
    
    # Load converted_mlp if it exists
    converted_mlp = None
    converted_mlp_path = exp_path / "converted_mlp.pt"
    if converted_mlp_path.exists():
        import torch
        converted_mlp = torch.load(converted_mlp_path, map_location="cpu", weights_only=False)
    
    # Check for primitives config
    primitives_config_path = exp_path / "attention_primitives" / "configs"
    replace_attn_with_primitives = False
    attn_primitives = None
    primitives_accuracy = None
    primitives_acc_match = None
    model_config_path = exp_path / "output.json"
    
    if primitives_config_path.exists():
        # Find the task-specific config file
        config_files = list(primitives_config_path.glob("*.json"))
        if config_files:
            with open(config_files[0], 'r') as f:
                primitives_data = json.load(f)
                replace_attn_with_primitives = True
                attn_primitives = primitives_data["primitives"]
                primitives_accuracy = primitives_data.get('accuracy', 'N/A')
                primitives_acc_match = primitives_data.get('acc_match', 'N/A')
                print(f"Found primitives config with accuracy: {primitives_accuracy}")

    if model_config_path.exists():
        with open(model_config_path, "r") as file:
            model_config = json.load(file)
            acc_match = model_config["acc_match"]
            acc_task = model_config["acc_task"]
            if primitives_accuracy:
                primitives_acc_match["before"] = acc_match
                primitives_accuracy["before"] = acc_task
    
    # Generate code using convert_config_to_code_qk_pruned (same as in app.py)
    show_logits_for_unconverted_mlp = False
    code, heatmap_to_config, var_mapping = convert_config_to_code_qk_pruned(
        config, 
        split_mlps, 
        converted_mlp,
        replace_attn_with_primitives, 
        attn_primitives,
        return_var_mapping=True,
        use_code_with_ops=IS_ONLY_LARGE_SCALING,
        show_logits_for_unconverted_mlp=show_logits_for_unconverted_mlp
    )

    all_vars = [v for v in heatmap_to_config.keys()]
    deleted_config_lines = {}
    if not show_logits_for_unconverted_mlp:
        for var_name in all_vars:
            inp = heatmap_to_config[var_name]
            if "mlp" in inp and inp[inp.find("mlp"):] not in converted_mlp:
                print("deleted from heatmap_to_config", var_name, inp, "not in converted_mlp", converted_mlp)
                deleted_config_lines[var_name] = heatmap_to_config[var_name]
                del heatmap_to_config[var_name]
            elif type(inp) == tuple and "mlp" in inp[-1][0] and inp[-1][0][inp[-1][0].find("mlp"):] not in converted_mlp:
                print("deleted from heatmap_to_config", var_name, inp, "not in converted_mlp", converted_mlp)
                deleted_config_lines[var_name] = heatmap_to_config[var_name]
                del heatmap_to_config[var_name]
            elif type(inp) == tuple and "mlp" in inp[-1][1] and inp[-1][1][inp[-1][1].find("mlp"):] not in converted_mlp:
                print("deleted from heatmap_to_config", var_name, inp, "not in converted_mlp", converted_mlp)
                deleted_config_lines[var_name] = heatmap_to_config[var_name]
                del heatmap_to_config[var_name]
            
    # code = "\n".join(code)
    # code = code.split("\n")
    code = [f"{i+1}. {line}" for i, line in enumerate(code)]
    # code = "\n".join(code)
    
    # Save code to text file
    code_file_path = dest_path / "generated_code.txt"
    with open(code_file_path, 'w') as f:
        f.write("\n".join(code))
    
    print(f"Generated code saved to: {code_file_path}")
    return code, heatmap_to_config, deleted_config_lines, var_mapping, (primitives_accuracy, primitives_acc_match)  # Return accuracy as well

def main(args: Args):
    if args.run_all:
        # Load good_models.json
        good_models_path = Path(args.good_models_json)
        if not good_models_path.exists():
            print(f"Error: good_models.json not found at {good_models_path}")
            return
        
        with open(good_models_path, 'r') as f:
            good_models = json.load(f)
        
        # Check if "chosen_exps" key exists
        if "chosen_exps" not in good_models:
            print(f"Error: 'chosen_exps' key not found in {good_models_path}")
            return
        
        chosen_exps = good_models["chosen_exps"]

        print("chosen_exps", chosen_exps)
        
        # Collect all color definitions
        all_color_definitions = set()
        
        
        # Process experiments from primary base_path to results_base_dir
        print(f"\n{'='*80}")
        print(f"Processing experiments from primary base path: {args.base_path}")
        print(f"Generating results in: {args.results_base_dir}")
        print(f"{'='*80}\n")
        
        for model_key, exp_info in chosen_exps.items():
            if not exp_info:
                continue
            
            exp_dir = f"{model_key}"
            exp_name = exp_info
            
            exp_path = Path(args.base_path) / exp_dir / exp_name
            print(f"\n[Primary Base Path] Processing: {model_key} / {exp_name}")
            print(f"Full path: {exp_path}")
            
            color_defs = process_experiment(
                exp_path, 
                args.results_base_dir, 
                args.base_path_prefix,
                args.copy_activations_first_base_path
            )
            
            if color_defs:
                all_color_definitions.update(color_defs)
        
        results_base = Path(args.results_base_dir)
        
        # Save color definitions to a separate file
        color_defs_content = []
        color_defs_content.append("% Color definitions for all tasks")
        color_defs_content.append("% Copy these to your LaTeX preamble")
        for color_def in sorted(all_color_definitions):
            color_defs_content.append(color_def)
        
        with open(results_base / "color_definitions.txt", "w") as file:
            file.write("\n".join(color_defs_content))
        
        # Create combined code file with reference to color definitions
        combined_content = []
        
        # Add header comments
        combined_content.append("% Combined LaTeX code for all tasks")
        combined_content.append("% Add these packages to your preamble:")
        combined_content.append("% \\usepackage{xcolor}")
        combined_content.append("% \\usepackage{fancyvrb}")
        combined_content.append("% \\usepackage{subcaption}")
        combined_content.append("% \\usepackage{float}")
        combined_content.append("% \\usepackage{tikz}")
        combined_content.append("% \\newcommand{\\circled}[1]{\\tikz[baseline=(char.base)]{\\node[shape=circle,draw,inner sep=2pt] (char) {#1};}}")
        combined_content.append("")
        combined_content.append("% ========================================")
        combined_content.append("% Color definitions are in color_definitions.txt")
        combined_content.append("% Copy those to your LaTeX preamble first")
        combined_content.append("% ========================================")
        
        # Append all individual latex files (without their color definitions)
        for folder in sorted(os.listdir(args.results_base_dir)):
            latex_file = results_base / folder / "latex.tex"
            latex_path = Path(args.base_path_prefix) / folder / "latex.tex"
            if latex_file.exists():
                combined_content.append(f"\\input{{{str(latex_path)}}}")
        
        # Write combined file
        with open(results_base / "combined_code.txt", "w") as file:
            file.write("\n".join(combined_content))
        
        print(f"\n{'='*80}")
        print(f"✓ Color definitions saved to: {results_base / 'color_definitions.txt'}")
        print(f"✓ Combined LaTeX code saved to: {results_base / 'combined_code.txt'}")
        print(f"  Copy color_definitions.txt to your LaTeX preamble first")
        print(f"{'='*80}")
        
    else:
        if not args.exp_path:
            print("Error: exp_path is required when run_all is False")
            return
        exp_path = Path(args.exp_path)
        process_experiment(
                exp_path, 
                args.results_base_dir, 
                args.base_path_prefix,
                args.copy_activations_first_base_path
        )

def process_experiment(exp_path: Path, results_base_dir: str, base_path_prefix, copy_activations_first_base_path: bool = False):
    """Process a single experiment: generate code, copy heatmaps, and create LaTeX."""
    # Validate the experiment path
    if not exp_path.exists():
        print(f"Error: Experiment path does not exist: {exp_path}")
        return None
    
    # Extract task_name and exp_name from the path
    try:
        task_name, exp_name = extract_task_and_exp_name(exp_path)
        print(f"Task name: {task_name}")
        print(f"Experiment name: {exp_name}")
    except ValueError as e:
        print(f"Error: {e}")
        return None
    
    # Sanitize task name for use in folder paths
    sanitized_task_name = sanitize_for_latex_command(task_name)
    
    # Create results directory structure: results/sanitized_task_name/
    results_base = Path(results_base_dir)
    dest_path = results_base / sanitized_task_name
    dest_path.mkdir(parents=True, exist_ok=True)
    print(f"Created results directory: {dest_path}")
    
    # Copy primitive heatmaps (only primitives-matrices subfolders)
    print("\nCopying primitive heatmaps...")
    copy_success = copy_primitive_heatmaps(exp_path, dest_path, task_name)
    
    # Copy activation heatmaps if activation_base_dir is provided
    if copy_activations_first_base_path:
        print("\nCopying activation heatmaps...")
        copy_activation_heatmaps(exp_path, dest_path, task_name)
    
    # Generate and save code
    print("\nGenerating code...")
    code_lines, heatmap_to_config, deleted_config_lines, var_mapping, (primitives_accuracy, primitives_acc_match) = generate_code_file(exp_path, dest_path)
    code_success = code_lines is not None
    
    # Generate LaTeX file
    color_definitions = None
    if code_success and copy_success:
        print("\nGenerating LaTeX file...")
        # Pass both original and sanitized task names
        color_definitions = generate_latex_file(exp_path, dest_path, task_name, sanitized_task_name, code_lines, heatmap_to_config, var_mapping,
                    base_path_prefix=base_path_prefix,
                    deleted_config_lines=deleted_config_lines, copy_activations_first_base_path=copy_activations_first_base_path,
                    primitives_accuracy=primitives_accuracy, primitives_acc_match=primitives_acc_match)
        latex_success = color_definitions is not None
    else:
        latex_success = False
    
    if copy_success and code_success and latex_success:
        print(f"\n✓ Successfully created results in: {dest_path}")
        print(f"  - generated_code.txt: Code representation")
        print(f"  - latex.tex: LaTeX code ready to copy-paste")
        print(f"  - Heatmap images in subdirectories")
    elif copy_success and code_success:
        print(f"\n⚠ Code and heatmaps created but LaTeX generation failed")
    elif copy_success:
        print(f"\n⚠ Heatmaps copied but code generation failed")
    elif code_success:
        print(f"\n⚠ Code generated but heatmap copying failed")
    else:
        print(f"\n✗ Failed to process experiment")
    
    return color_definitions

def process_experiment_from_base_path(base_path: str, exp_dir: str, exp_name: str, dest_dir: str, base_path_prefix, copy_activation=False):
    """
    Process an experiment from a base path and copy results to destination directory.
    
    Args:
        base_path: Base path where experiments are stored (e.g., /scratch/abakalov/patching_runs_2)
        exp_dir: Experiment directory name (e.g., "newprune-task-@arch")
        exp_name: Experiment name (e.g., "exp001")
        dest_dir: Destination directory for results
        copy_activation: Whether to copy activation heatmaps instead of primitives heatmaps
    
    Returns:
        tuple: (success, task_name, sanitized_task_name, color_definitions)
    """
    exp_path = Path(base_path) / exp_dir / exp_name
    
    # Validate the experiment path
    if not exp_path.exists():
        print(f"Warning: Experiment path does not exist: {exp_path}")
        return (False, None, None, None)
    
    # Extract task_name and exp_name from the path
    try:
        task_name, _ = extract_task_and_exp_name(exp_path)
        print(f"Task name: {task_name}")
        print(f"Experiment name: {exp_name}")
    except ValueError as e:
        print(f"Error: {e}")
        return (False, None, None, None)
    
    # Sanitize task name for use in folder paths
    sanitized_task_name = sanitize_for_latex_command(task_name)
    
    # Create results directory structure
    dest_base = Path(dest_dir)
    dest_path = dest_base / sanitized_task_name
    dest_path.mkdir(parents=True, exist_ok=True)
    print(f"Created results directory: {dest_path}")
    
    # Copy heatmaps based on the copy_activation flag
    if copy_activation:
        print("\nCopying activation heatmaps...")
        copy_success = copy_activation_heatmaps(exp_path, dest_path, task_name)
    else:
        print("\nCopying primitive heatmaps...")
        copy_success = copy_primitive_heatmaps(exp_path, dest_path, task_name)
    
    # Generate and save code (only if copying from primary base path to results_base_dir)
    code_lines = None
    heatmap_to_config = None
    var_mapping = None
    
    if copy_success:
        print(f"\n✓ Successfully copied heatmaps to: {dest_path}")
    else:
        print(f"\n✗ Failed to copy heatmaps")
    
    return (copy_success, task_name, sanitized_task_name, None)

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)