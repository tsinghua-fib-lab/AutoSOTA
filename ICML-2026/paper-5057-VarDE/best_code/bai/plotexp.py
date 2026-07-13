import matplotlib.pyplot as plt
import numpy as np
import os

def plot_experiment_results(rec, true_best, expnum):
    # Create the results directory if it doesn't exist
    output_dir = 'results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Calculate error probabilities (skip empty entries and align lengths)
    labels = []
    error_probs = []
    for label, runs in rec.items():
        if len(runs) == 0:
            print(f"Skipping '{label}' (no runs recorded).")
            continue

        arrays = []
        for run in runs:
            arr = np.asarray(run)
            if arr.ndim == 0:
                arr = arr.reshape(1)
            if arr.size == 0:
                continue
            arrays.append(arr)

        if len(arrays) == 0:
            print(f"Skipping '{label}' (all runs empty).")
            continue

        min_len = min(a.shape[0] for a in arrays)
        if min_len == 0:
            print(f"Skipping '{label}' (zero-length runs).")
            continue

        stack = np.stack([a[:min_len] for a in arrays], axis=0)
        labels.append(label)
        error_probs.append(np.mean(stack != true_best, axis=0))

    if len(labels) == 0:
        raise ValueError("No non-empty results found in rec; nothing to plot.")
    
    # --- Line plot ---
    plt.figure(figsize=(10, 6))
    for i, label in enumerate(labels):
        plt.plot(error_probs[i], label=label)
    plt.xlabel('Time Steps')
    plt.ylabel('Error Probability')
    plt.title(f'Experiment {expnum} - Error Probability over Time')
    plt.legend()
    plt.tight_layout()
    
    line_plot_path = os.path.join(output_dir, f'exp{expnum}_line.png')
    plt.savefig(line_plot_path, bbox_inches='tight')
    plt.close() # Good practice to close plots to save memory
    print(f"Line plot saved as '{line_plot_path}'")

    # --- Save error probabilities to a text file ---
    txt_path = os.path.join(output_dir, f'exp{expnum}_results.txt')
    with open(txt_path, 'w') as f:
        for i, label in enumerate(labels):
            f.write(f'{label}: ' + ','.join(map(str, error_probs[i])) + '\n')
    print(f"Error probabilities over time saved as '{txt_path}'")

    # --- Bar plot at final time step ---
    final_errors = [ep[-1] for ep in error_probs]
    plt.figure(figsize=(10, 6))
    plt.bar(labels, final_errors)
    plt.ylabel('Error Probability')
    plt.title(f'Experiment {expnum} - Error Probability at Final Time Step')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    bar_plot_path = os.path.join(output_dir, f'exp{expnum}_bar.png')
    plt.savefig(bar_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Bar plot saved as '{bar_plot_path}'")
