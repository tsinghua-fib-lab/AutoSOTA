#!/usr/bin/env python3

# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Consolidated plotting utilities for ProEval.

This module contains all plotting constants and helpers used across the codebase:
- Color palettes (COLOR_LIST, METHOD_COLORS)
- Figure sizing utilities (set_style, set_size)
- Common plot functions

Merged from: eval/utils.py + existing utils/plot.py
"""
import os
import datetime
import matplotlib as mpl

try:
    from matplotlib.backends import backend_pgf
    FigureCanvasPgf = backend_pgf.FigureCanvasPgf
    mpl.backend_bases.register_backend('pdf', FigureCanvasPgf)
except ImportError:
    pass

import matplotlib.pyplot as plt
import numpy as np


# Plotting Constants

# Figure width for LaTeX documents (in points)
WIDTH = 506.295

# Marker styles for line plots
MARKER_LIST = ['.', '*', '^', 'p', 'o', 'x']

# Hatch patterns for bar charts
HATCH_LIST = [None, '\\', 'x', '-', '/', '+']

# General color palette (for various types of plots)
COLOR_LIST = [
    '#DC143C',  # Crimson
    '#8B008B',  # Dark Magenta
    '#6495ED',  # Cornflower Blue
    '#3CB371',  # Medium Sea Green
    '#FFD700',  # Gold
    '#F08080',  # Light Coral
    '#FF8C00',  # Dark Orange
    '#008B8B',  # Dark Cyan
    '#7B68EE',  # Medium Slate Blue
]

# Standardized method colors for BQ experiments
# Qualitative palette: Gray, Purple, Teal, Gold, Red, Dark Blue
METHOD_COLORS = {
    'SS': '#59a89c',          # Teal - Active sampling, no-gen
    'Rand': '#cecece',        # Gray - Random sampling
    'SS-Gen': '#e02b35',      # Red - SS with generation
    'TSS': '#082a54',         # Dark Blue - Topic + SS + generation (best method)
    'Rand-T-Gen': '#f0c571',  # Gold - Random Topic + generation
    'Rand-Gen': '#a559aa',    # Purple - Pure random generation
}

# Preferred method order for legend display
PREFERRED_METHOD_ORDER = ['Rand', 'Rand-Gen', 'Rand-T-Gen', 'SS', 'SS-Gen', 'TSS']


# Figure Sizing Utilities

def set_size(width, fraction=1):
    """Set aesthetic figure dimensions to avoid scaling in LaTeX.
    
    Args:
        width: Width in pts (use WIDTH constant for full width)
        fraction: Fraction of the width for the figure to occupy
        
    Returns:
        tuple: (fig_width_in, fig_height_in) dimensions in inches
    """
    # Width of figure
    fig_width_pt = width * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27
    # Golden ratio for aesthetic figure height
    golden_ratio = (5**0.5 - 1) / 1.5
    # Figure dimensions in inches
    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio
    return (fig_width_in, fig_height_in)


def set_style():
    """Set matplotlib style for publication-quality figures.
    
    Configures:
    - Classic style base
    - Serif fonts
    - LaTeX rendering (if available)
    - Appropriate font sizes
    """
    import shutil
    
    plt.style.use('classic')
    
    # Check if LaTeX is available
    latex_available = shutil.which('latex') is not None
    
    nice_fonts = {
        'font.family': 'serif',
        'axes.labelsize': 19,
        'font.size': 16,
        'legend.fontsize': 12,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
    }
    
    if latex_available:
        nice_fonts['text.usetex'] = True
    else:
        nice_fonts['text.usetex'] = False
    
    try:
        mpl.rcParams.update(nice_fonts)
    except Exception as e:
        print(f"Warning: Could not update matplotlib rcParams: {e}")


# File I/O Utilities

def save_fig(fig, path, timestamp=True, **kwargs):
    """Save a matplotlib figure to a path.
    
    Args:
        fig: The matplotlib figure object
        path: The full path where to save the figure
        timestamp: Whether to append a timestamp to the filename
        **kwargs: Additional arguments to pass to fig.savefig()
    """
    # Append timestamp if requested
    if timestamp:
        name, ext = os.path.splitext(path)
        if isinstance(timestamp, str):
            ts = timestamp
        else:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"{name}_{ts}{ext}"

    # Ensure the directory exists
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f'Created directory: {directory}')
    
    # Save the figure
    try:
        fig.savefig(path, **kwargs)
        print(f'Figure saved to: {path}')
    except Exception as e:
        print(f"Error saving figure to {path}: {e}")


# Common Plot Helpers

def get_timestamp() -> str:
    """Get timestamp string for file naming (YYYYMMDD_HH format)."""
    import time
    return time.strftime("%Y%m%d_%H")


def get_method_color(method_name: str, fallback_index: int = 0) -> str:
    """Get consistent color for a method name.
    
    Args:
        method_name: Name of the method
        fallback_index: Index to use in COLOR_LIST if method not found
        
    Returns:
        Hex color string
    """
    if method_name in METHOD_COLORS:
        return METHOD_COLORS[method_name]
    return COLOR_LIST[fallback_index % len(COLOR_LIST)]


def sort_methods_by_preference(methods: list) -> list:
    """Sort method names by preferred ordering for legends.
    
    Args:
        methods: List of method names
        
    Returns:
        Sorted list of method names
    """
    def method_sort_key(m):
        if m in PREFERRED_METHOD_ORDER:
            return PREFERRED_METHOD_ORDER.index(m)
        return len(PREFERRED_METHOD_ORDER)  # Unknown methods go at end
    
    return sorted(methods, key=method_sort_key)
