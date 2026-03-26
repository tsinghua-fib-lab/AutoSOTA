"""
Helper functions used for saving and loading causal effects and qini results in the notebooks.
"""

import os
import re
from contextlib import contextmanager

import matplotlib.pyplot as plt
import matplotlib
from cycler import cycler
import typing as t

import numpy as np


def check_match(name, pattern):
    # Escape any special characters in the pattern except '*'
    pattern = re.escape(pattern)

    # Replace '*' with '.*' (the regex pattern for matching any characters)
    pattern = pattern.replace(r"\*", ".*")

    # Match the pattern with the name using full-string match
    return bool(re.fullmatch(pattern, name))


def save_results(result: dict, dataset_name: str, method_name: str, idx: int, artifact_dir: str):
    save_path = os.path.join(artifact_dir, f"{dataset_name}_{method_name}[{idx}].npz")
    np.savez(save_path, **result)


@contextmanager
def result_saver(
    dataset_name: str,
    method_name: str,
    all_method_patterns: list,
    all_datasets_patterns: list,
    idx: int,
    artifact_dir: str,
    replace: bool = False,
):
    save_path = os.path.join(artifact_dir, f"{dataset_name}_{method_name}[{idx}].npz")
    dset_match = False
    for dset_pattern in all_datasets_patterns:
        if check_match(dataset_name, dset_pattern):
            dset_match = True
            break
    method_match = False
    for method_pattern in all_method_patterns:
        if check_match(method_name, method_pattern):
            method_match = True
            break
    if not method_match or not dset_match:
        yield None  # Skip code execution if method is not in run_methods or dataset is not in run_datasets
    elif os.path.exists(save_path) and not replace:
        yield None  # Skip code execution
    else:
        result = {}
        yield result  # Let user fill this with results inside the context
        save_results(result, dataset_name, method_name, idx, artifact_dir)


def load_results(dataset_name: str, method_name: str, idx: int, artifact_dir: str) -> dict:
    load_path = os.path.join(artifact_dir, f"{dataset_name}_{method_name}[{idx}].npz")
    if os.path.exists(load_path):
        data = np.load(load_path)
        return data
    else:
        return None


def load_all_results(dataset_name: str, artifact_dir: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    results = {}
    # go through all the files in the path
    for file in os.listdir(artifact_dir):
        if file.startswith(dataset_name) and file.endswith(".npz"):
            method_name = file.split("_")[-1].split("[")[0]
            idx = int(file.split("[")[1].split("]")[0]) if "[" in file else 0
            loaded_dict = load_results(dataset_name, method_name, idx, artifact_dir)
            if method_name not in results:
                results[method_name] = {}
            for key, value in loaded_dict.items():
                if key not in results[method_name]:
                    results[method_name][key] = [value]
                else:
                    results[method_name][key].append(value)

    return results


# plotting utility

### Based on the code from https://github.com/alexcapstick/llm-elicited-priors/main/llm_elicited_priors/plotting.py

NEURIPS_COL_SIZE = 5.5


def save_fig(fig: plt.figure, file_name: str, **kwargs) -> None:
    """
    This function saves a pdf, png, and svg of the figure,
    with :code:`dpi=300`.


    Arguments
    ---------

    - fig: plt.figure:
        The figure to save.

    - file_name: str:
        The file name, including path, to save the figure at.
        This should not include the extension, which will
        be added when each file is saved.

    """

    fig.savefig(f"{file_name}.pdf", **kwargs)
    fig.savefig(f"{file_name}.png", dpi=300, **kwargs)
    fig.savefig(f"{file_name}.svg", **kwargs)


# colours
tol_muted = [
    "#332288",
    "#88CCEE",
    "#44AA99",
    "#117733",
    "#999933",
    "#DDCC77",
    "#CC6677",
    "#882255",
    "#AA4499",
]

ibm = {
    "blue": "#648fff",
    "orange": "#fe6100",
    "pink": "#dc267f",
    "purple": "#785ef0",
    "amber": "#ffb000",
    "black": "#000000",
}


@contextmanager
def graph_theme(colours: t.List[str] = ibm.values(), **kwargs):
    """
    Temporarily sets the default theme for all plots.


    Examples
    ---------

    .. code-block::

        >>> with graph_theme():
        ...     plt.plot(x,y)


    Arguments
    ---------

    - colours: t.List[str], optional:
        Any acceptable list to :code:`cycler`.
        Defaults to :code:`ibm`.


    """
    with matplotlib.rc_context():
        # sns.set(context="paper", style="whitegrid")
        custom_params = {
            "axes.spines.right": True,
            "axes.spines.top": True,
            "axes.edgecolor": "black",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.prop_cycle": cycler(color=colours),
            "grid.alpha": 0.5,
            "grid.color": "#b0b0b0",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            # following requires latex
            "font.family": "serif",
            "font.serif": "Computer Modern",
            "font.size": 7,
            "figure.titlesize": 7,
            "text.usetex": True,
            "text.latex.preamble": "\\usepackage{amsmath,amsfonts,amssymb,amsthm, mathtools,times}",
            # end requiring latex
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.labelsize": 7,  # Smaller tick labels
            "ytick.labelsize": 7,
            # Figure settings
            "figure.dpi": 300,  # High resolution for print
            "figure.constrained_layout.use": True,  # Better layout management
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            # Legend settings
            "legend.fontsize": 6,
            "legend.frameon": True,
            "legend.handlelength": 1.5,
            # Marker settings for scatter plots
            "lines.markersize": 4,
            "lines.linewidth": 1.2,
            "axes.labelsize": 7,
        }

        matplotlib.rcParams.update(**custom_params)
        matplotlib.rcParams.update(**kwargs)

        yield
