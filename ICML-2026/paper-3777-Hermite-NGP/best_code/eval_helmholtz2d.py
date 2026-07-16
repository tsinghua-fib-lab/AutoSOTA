"""Evaluation script for Helmholtz 2D compact model.
Loads saved results and evaluates the model on a fine grid.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MPLBACKEND"] = "Agg"

import torch
import numpy as np
import argparse

PI = np.pi

def exact_solution(x, y, a1, a2):
    return torch.sin(a1 * PI * x) * torch.sin(a2 * PI * y)

def evaluate_from_npz(npz_path, resolution=200):
    """Load model from NPZ and evaluate L2 error."""
    data = np.load(npz_path, allow_pickle=True)

    a1 = float(data["config_a1"])
    a2 = float(data["config_a2"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Reconstruct model architecture
    from examples.helmholtz2d import HermiteNGP_PINN
    config = {
        "n_levels": int(data["config_n_levels"]),
        "log2_hashmap_size_1": int(data["config_log2_hashmap_size_1"]),
        "log2_hashmap_size_2": int(data["config_log2_hashmap_size_2"]),
        "log2_hashmap_size_3": int(data["config_log2_hashmap_size_3"]),
        "hidden_dim": int(data["config_hidden_dim"]),
        "n_layers": int(data["config_n_layers"]),
        "omega": float(data["config_omega"]),
        "per_level_scale": float(data["config_per_level_scale"]),
    }

    model = HermiteNGP_PINN(config)

    # Load saved parameters
    with torch.no_grad():
        for name, param in model.named_parameters():
            key = "param_" + name.replace(".", "_")
            if key in data:
                param.copy_(torch.from_numpy(data[key]))

    model.eval()

    # Evaluate on uniform grid
    with torch.no_grad():
        g = torch.linspace(0, 1, resolution, device=device)
        X, Y = torch.meshgrid(g, g, indexing="ij")
        pts = torch.stack([X.flatten(), Y.flatten()], dim=1)
        u_pred = model.forward(pts).reshape(resolution, resolution)
        u_exact = exact_solution(X, Y, a1, a2)
        l2_error = (torch.sqrt(((u_pred - u_exact)**2).sum()) / torch.sqrt((u_exact**2).sum())).item()

    return l2_error

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=200)
    args = parser.parse_args()

    l2 = evaluate_from_npz(args.npz, args.resolution)
    print("Relative L2 Error: {:.6e} (resolution={}x{})".format(l2, args.resolution, args.resolution))
