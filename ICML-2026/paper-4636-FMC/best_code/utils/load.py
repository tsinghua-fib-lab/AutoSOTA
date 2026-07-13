import copy
import mlxp

from pathlib import Path
from typing import Dict, List, Tuple

import torch
from mlxp.logger import Logger
from torch import Tensor

from posteriors import Estimator
from baselines.trainers import train_npe
from simulator import generate_calibration_dataset, generate_simulation_dataset
from simulator.base import Simulator
from utils.timing import HierarchicalTimer


def setup_paths_and_device(
    ctx: mlxp.Context, cfg: Dict
) -> Tuple[Path, Path, Path, Path, torch.device]:
    """Initialize directory paths and compute device.

    Args:
        ctx: MLXP context
        cfg: Configuration dictionary

    Returns:
        Tuple of (log_path, data_path, model_path, timing_path, device)
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Found device: {device}")

    log_path = Path(ctx.mlxp.logger.parent_log_dir)
    task_name = cfg["task"]["name"]

    data_path = log_path / "data" / task_name
    data_path.mkdir(parents=True, exist_ok=True)

    model_path = log_path / "models" / task_name / f"seed_{cfg['seed']}"
    model_path.mkdir(parents=True, exist_ok=True)

    timing_path = model_path / "timing"
    timing_path.mkdir(parents=True, exist_ok=True)

    return log_path, data_path, model_path, timing_path, device


def load_or_generate_datasets(
    cfg: Dict,
    simulator: Simulator,
    data_path: Path,
    num_samples: int,
    num_cal_max: int,
    task_config: Dict,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Load or generate simulation and calibration datasets.

    Args:
        cfg: Configuration dictionary
        simulator: Simulator instance
        data_path: Path to save/load data
        num_samples: Number of simulation samples
        num_cal_max: Maximum number of calibration samples
        task_config: Task configuration

    Returns:
        Tuple of (theta_sim, x_sim, theta_cal, x_cal, y_cal)
    """
    # Simulation Data
    if cfg["load_data"]:
        try:
            data = torch.load(data_path / "simulations.pt")
            theta_sim = data["theta"]
            x_sim = data["x"]
        except FileNotFoundError:
            raise FileNotFoundError("Simulation data not found. Please generate it first.")
    else:
        theta_sim, x_sim = generate_simulation_dataset(simulator, num_samples)
        torch.save(
            {"theta": theta_sim, "x": x_sim},
            data_path / "simulations.pt",
        )

    # Calibration Data
    if cfg["load_data"]:
        try:
            data = torch.load(data_path / "calibrations.pt")
            theta_cal = data["theta"]
            x_cal = data["x"]
            y_cal = data["y"]
        except FileNotFoundError:
            raise FileNotFoundError("Calibration data not found. Please generate it first.")
    else:
        theta_cal, x_cal, y_cal = generate_calibration_dataset(
            simulator, num_cal_max, task_config["generation"]
        )
        torch.save({"theta": theta_cal, "x": x_cal, "y": y_cal}, data_path / "calibrations.pt")

    return theta_sim, x_sim, theta_cal, x_cal, y_cal


def save_embedding_networks(
    task: Dict,
    theta_cal: Tensor,
    y_cal: Tensor,
    cal_values: List[int],
    device: torch.device,
    model_path: Path,
    timing_path: Path,
    logger: Logger,
    npe: Estimator,
) -> None:
    """Train DPE ref and save embedding networks for x and y.

    Args:
        task: Task configuration
        theta_cal: Calibration parameters
        y_cal: Calibration observations
        cal_values: List of calibration dataset sizes
        device: Compute device
        model_path: Path to save models
        timing_path: Path for timing data
        logger: MLXP logger
        npe: NPE model (for x embedding)
    """
    print("\n" + "=" * 60)
    print("SAVING EMBEDDING NETWORKS")
    print("=" * 60)

    # Train DPE on reference calibration data to get y embedding
    dpe_ref_timer = HierarchicalTimer()
    dpe_ref = train_npe(
        task=task,
        theta=theta_cal[: cal_values[-1]],  # Removed naugment multiplication
        x=y_cal[: cal_values[-1]],
        device=device,
        model_path=model_path,
        load=True,
        save=True,
        logger=logger,
        logname="dpe_ref",
        timer=dpe_ref_timer,
        timer_operation_name="dpe_ref",
    )
    dpe_ref_timer.save(timing_path / "timing_dpe_ref.json")
    dpe_ref.cpu()

    # Save x and y embeddings
    embedding_nets = {
        "x": copy.deepcopy(npe.embedding_net),
        "y": copy.deepcopy(dpe_ref.embedding_net),
    }

    logger.log_artifacts(
        embedding_nets,
        artifact_name="embedding_networks",
        artifact_type="pickle",
    )

    print("Embedding networks saved")
    print("=" * 60 + "\n")
