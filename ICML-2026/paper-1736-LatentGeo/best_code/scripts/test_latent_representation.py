"""Executable test for training and visualizing the learned point-cloud representation."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from latentgeodesics import LatentPointCloud, MultiIndexDataLoader, setup_latent_network, train_latent_representation

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = REPO_ROOT / "examples" / "Point_Cloud_example" / "data" / "samples.npy"
DEFAULT_CONFIG_PATH = REPO_ROOT / "examples" / "Point_Cloud_example" / "saved_models" / "args_latentrep.json"
DEFAULT_RUNS_DIR = REPO_ROOT / "runs"


def build_args(config_path, data_path, epochs, device):
    with open(config_path, "r", encoding="utf-8") as handle:
        args = json.load(handle)

    args["train"]["epochs"] = epochs
    args["train"]["device"] = device
    args["data"]["load_path"] = str(data_path)
    args["test"]["data"] = None
    return args


def train_representation(args, run_dir, device):
    dataset = LatentPointCloud(Path(args["data"]["load_path"]), reach=args["data"].get("reach", None), device=str(device))
    dataloader = MultiIndexDataLoader(
        dataset,
        batch_size=args["train"]["batch_size"],
        num_samples=args["data"].get("num_samples", len(dataset)),
        shuffle=True,
    )

    network = train_latent_representation(
        dataloader,
        run_dir,
        args=args,
        eval_every=10,
        save_every=50,
        device=device,
    )

    best_state_path = run_dir / "state_dict_best.pth"
    final_state_path = run_dir / "state_dict_final.pth"
    state_path = best_state_path if best_state_path.exists() else final_state_path

    reloaded_network = setup_latent_network(args["architecture"]).to(device)
    reloaded_network.load_state_dict(torch.load(state_path, map_location=device))
    return reloaded_network, state_path

def _batched_tensor_eval(points, fn, batch_size: int = 2048):
    chunks = torch.split(points, batch_size)
    with torch.no_grad():
        values = [fn(chunk) for chunk in chunks]
    return torch.cat(values, dim=0)

def visualize_representation(network, data_samples, figure_path) -> None:
    device = network.device()
    latent_dim = network.in_out_dim

    assert data_samples.ndim == 2 and data_samples.shape[1] == latent_dim, data_samples.shape

    x_min, x_max = data_samples[:, 0].min(), data_samples[:, 0].max()
    y_min, y_max = data_samples[:, 1].min(), data_samples[:, 1].max()
    offset = 0.1

    fine_resolution = 128
    coarse_resolution = 24

    x_values = np.linspace(x_min - offset, x_max + offset, fine_resolution, dtype="float32")
    y_values = np.linspace(y_min - offset, y_max + offset, fine_resolution, dtype="float32")
    xx, yy = np.meshgrid(x_values, y_values)
    coords_flat = np.column_stack([xx.ravel(), yy.ravel()])
    grid_points = torch.tensor(coords_flat, dtype=torch.float32, device=device)

    coarse_x = np.linspace(x_min - offset, x_max + offset, coarse_resolution, dtype="float32")
    coarse_y = np.linspace(y_min - offset, y_max + offset, coarse_resolution, dtype="float32")
    cxx, cyy = np.meshgrid(coarse_x, coarse_y)
    coarse_coords = np.column_stack([cxx.ravel(), cyy.ravel()])
    coarse_grid_points = torch.tensor(coarse_coords, dtype=torch.float32, device=device)

    def levels(points):
        return torch.norm(network.latent_space_deviation(points), dim=1)

    def projection(points):
        return network.latent_space_deviation(points)

    values = _batched_tensor_eval(grid_points, levels).cpu().numpy().reshape(fine_resolution, fine_resolution)
    projected = 0.1 * _batched_tensor_eval(coarse_grid_points, projection).cpu().numpy()

    fig, ax = plt.subplots(figsize=(7.5, 5))
    image = ax.imshow(
        values,
        extent=[x_min - offset, x_max + offset, y_min - offset, y_max + offset],
        cmap="viridis",
        origin="lower",
        aspect="equal",
    )
    ax.scatter(data_samples[:, 0], data_samples[:, 1], s=1, marker="o", c="white", alpha=0.6, linewidths=0)
    ax.quiver(
        coarse_coords[:, 0],
        coarse_coords[:, 1],
        projected[:, 0],
        projected[:, 1],
        width=0.005,
        scale=0.3,
        headwidth=3,
        headlength=3,
        headaxislength=3,
        color="white",
        alpha=0.9,
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("levelset")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Learned point cloud representation")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)

    assert figure_path.exists(), f"failed to save figure to {figure_path}"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to the point-cloud config JSON.")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH), help="Path to the point-cloud samples.npy file.")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs.")
    parser.add_argument("--runs-dir", type=str, default=str(DEFAULT_RUNS_DIR), help="Directory where the run folder is created.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="auto", help="Training device: auto, cpu, or cuda.")
    return parser.parse_args()

def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)

def main():
    args = parse_args()
    device = resolve_device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config_path = Path(args.config)
    data_path = Path(args.data)
    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    args = build_args(config_path, data_path, args.epochs, str(device))

    timestr = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = runs_dir / f"test_latent_representation_{timestr}"
    run_dir.mkdir(parents=True, exist_ok=False)

    print(f"test training the latent representation for {args['train']['epochs']} epochs with config {config_path}")
    print("____________________________________")
    network, state_path = train_representation(args, run_dir, device)
    data_samples = np.load(data_path)

    figure_path = run_dir / "representation_test.png"
    visualize_representation(network, data_samples, figure_path)
    print("____________________________________")
    print("representation test passed, saved a result figure and the trained weights")
    print("____________________________________")
    print(f"run_dir={run_dir}")
    print(f"weights={state_path}")
    print(f"figure={figure_path}")


if __name__ == "__main__":
    main()