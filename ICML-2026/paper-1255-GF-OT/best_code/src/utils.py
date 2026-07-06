import os
import numpy as np


def kl_divergence(p, q, eps=1e-4):
    """Compute the KL divergence between two distributions with clipping for
    numerical stability."""
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    return np.sum(p * np.log(p / q))


def save_config(results_dir, experiment_id, config):
    config_path = os.path.join(results_dir, f"config_{experiment_id}.py")
    with open(config_path, "w") as f:
        f.write("config = ")
        f.write(repr(config))
