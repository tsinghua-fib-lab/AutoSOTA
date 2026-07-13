import os
import numpy as np
import matplotlib.pyplot as plt


def init_plot_font():
    pass


def describe_statistics(numbers, return_dict=False):
    # Convert the list to a NumPy array
    numbers_array = np.array(numbers)
    
    # Calculate statistics
    avg = np.mean(numbers_array)
    std_dev = np.std(numbers_array)
    quartiles = np.percentile(numbers_array, [25, 50, 75])
    min_val = np.min(numbers_array)
    max_val = np.max(numbers_array)
    
    if return_dict:
        return {
            "avg": avg,
            "std": std_dev,
            "quartiles": list(quartiles),
            "min": min_val,
            "max": max_val,
            "n": len(numbers),
        }
    else:
        # Format the statistics as a string
        output = f"Average: {avg}\n"
        output += f"Standard Deviation: {std_dev}\n"
        output += f"Quartiles: {quartiles}\n"
        output += f"Min: {min_val}\n"
        output += f"Max: {max_val}\n"
        return output


def tictoc_histogram(results, show=False, save_to=None, figsize=None, logscale=True):
    num_subplots = len(results)
    if figsize is None:
        figsize = (num_subplots * 5, 5)
    figure, axes = plt.subplots(1, num_subplots, figsize=figsize)
    if num_subplots == 1:
        axes = [axes]
    for idx, (name, vals) in enumerate(results.items()):
        axes[idx].hist(vals)
        axes[idx].set_xlabel('Time (s)')
        axes[idx].title.set_text(name)
        if logscale:
            axes[idx].set_yscale('log', nonpositive='clip')
    if save_to is not None:
        figure.savefig(save_to, bbox_inches="tight", dpi=192)
    if show:
        plt.show()
    else:
        plt.close(figure)