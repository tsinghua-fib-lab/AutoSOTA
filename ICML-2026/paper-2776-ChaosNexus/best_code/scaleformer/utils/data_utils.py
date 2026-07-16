import logging
import os
import time
from datetime import datetime
from functools import wraps
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Callable, List, Literal, Union

import numpy as np
from dysts.systems import get_attractor_list
from gluonts.dataset.arrow import ArrowWriter
from gluonts.dataset.common import FileDataset


def get_dim_from_dataset(dataset: FileDataset) -> int:  # type: ignore
    """
    helper function to get system dimension from file dataset
    """
    return next(iter(dataset))["target"].shape[0]


def safe_standardize(
    arr: np.ndarray,
    epsilon: float = 1e-10,
    axis: int = -1,
    context: np.ndarray | None = None,
    denormalize: bool = False,
) -> np.ndarray:
    """
    Standardize the trajectories by subtracting the mean and dividing by the standard deviation

    Args:
        arr: The array to standardize
        epsilon: A small value to prevent division by zero
        axis: The axis to standardize along
        context: The context to use for standardization. If provided, use the context to standardize the array.
        denormalize: If True, denormalize the array using the mean and standard deviation of the context.

    Returns:
        The standardized array
    """
    # if no context is provided, use the array itself
    context = arr if context is None else context

    assert arr.ndim == context.ndim, (
        "arr and context must have the same num dims if context is provided"
    )
    assert axis < arr.ndim and axis >= -arr.ndim, (
        "invalid axis specified for standardization"
    )
    mean = np.nanmean(context, axis=axis, keepdims=True)
    std = np.nanstd(context, axis=axis, keepdims=True)
    std = np.where(std < epsilon, epsilon, std)
    if denormalize:
        return arr * std + mean
    return (arr - mean) / std


def demote_from_numpy(param: float | np.ndarray) -> float | list[float]:
    """
    Demote a float or numpy array to a float or list of floats
    Used for serializing parameters to json
    """
    if isinstance(param, np.ndarray):
        return param.tolist()
    return param


def dict_demote_from_numpy(
    param_dict: dict[str, float | np.ndarray],
) -> dict[str, float | list[float]]:
    """
    Demote a dictionary of parameters to a dictionary of floats or list of floats
    """
    return {k: demote_from_numpy(v) for k, v in param_dict.items()}


def timeit(logger: logging.Logger | None = None) -> Callable:
    """Decorator that measures and logs execution time of a function.

    Args:
        logger: Optional logger to use for timing output. If None, prints to stdout.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start

            if elapsed < 60:
                time_str = f"{elapsed:.2f} seconds"
            elif elapsed < 3600:
                time_str = f"{elapsed / 60:.2f} minutes"
            else:
                time_str = f"{elapsed / 3600:.2f} hours"

            msg = f"{func.__name__} took {time_str}"
            if logger:
                logger.info(msg)
            else:
                print(msg)

            return result

        return wrapper

    return decorator


def split_systems(
    prop: float,
    seed: int,
    sys_class: str = "continuous",
    excluded_systems: list[str] = [],
) -> tuple[list[str], list[str]]:
    """
    Split the list of attractors into training and testing sets.
    if exclude_systems is provided, the systems in the list will be excluded
    """
    np.random.seed(seed)
    systems = get_attractor_list(sys_class=sys_class, exclude=excluded_systems)
    np.random.default_rng(seed).shuffle(systems)
    split = int(len(systems) * prop)
    return systems[:split], systems[split:]


def convert_to_arrow(
    path: Union[str, Path],
    time_series: Union[List[np.ndarray], np.ndarray],
    compression: Literal["lz4", "zstd"] = "lz4",
    split_coords: bool = False,
):
    """
    Store a given set of series into Arrow format at the specified path.

    Input data can be either a list of 1D numpy arrays, or a single 2D
    numpy array of shape (num_series, time_length).
    """
    assert isinstance(time_series, list) or (
        isinstance(time_series, np.ndarray) and time_series.ndim == 2
    ), "time_series must be a list of 1D numpy arrays or a 2D numpy array"

    # GluonTS requires this datetime format for reading arrow file
    start = datetime.now().strftime("%Y-%m-%d %H:%M")

    if split_coords:
        dataset = [{"start": start, "target": ts} for ts in time_series]
    else:
        dataset = [{"start": start, "target": time_series}]

    ArrowWriter(compression=compression).write_to_file(
        dataset,
        path=Path(path),
    )


def process_trajs(
    base_dir: str,
    timeseries: dict[str, np.ndarray],
    split_coords: bool = False,
    overwrite: bool = False,
    verbose: bool = False,
    base_sample_idx: int = -1,
) -> None:
    """
    Saves each trajectory in timeseries ensemble to a separate directory

    Args:
        base_dir: The base directory to save the trajectories to
        timeseries: A dictionary mapping system names to numpy arrays of shape
            (num_eval_windows * num_datasets, num_channels, T) where T is the prediction
            length or context length.
        split_coords: Whether to split the coordinates by dimension
        overwrite: Whether to overwrite existing trajectories
        verbose: Whether to print verbose output
        base_sample_idx: The base sample index to use for the trajectories
    """
    for sys_name, trajectories in timeseries.items():
        if verbose:
            print(
                f"Processing trajectories of shape {trajectories.shape} for system {sys_name}"
            )

        system_folder = os.path.join(base_dir, sys_name)
        os.makedirs(system_folder, exist_ok=True)

        if not overwrite:
            for filename in os.listdir(system_folder):
                if filename.endswith(".arrow"):
                    sample_idx = int(filename.split("_")[0])
                    base_sample_idx = max(base_sample_idx, sample_idx)

        for i, trajectory in enumerate(trajectories):
            # very hacky, if there is only one trajectory, we can just use the base_sample_idx
            curr_sample_idx = base_sample_idx + i + 1

            if trajectory.ndim == 1:
                trajectory = np.expand_dims(trajectory, axis=0)
            if verbose:
                print(
                    f"Saving {sys_name} trajectory {curr_sample_idx} with shape {trajectory.shape}"
                )

            path = os.path.join(
                system_folder, f"{curr_sample_idx}_T-{trajectory.shape[-1]}.arrow"
            )

            convert_to_arrow(path, trajectory, split_coords=split_coords)


def load_trajectory_from_arrow(
    filepath: Path | str, one_dim_target: bool = False
) -> tuple[np.ndarray, tuple[Any]]:
    """Load a trajectory and metadata from an arrow file

    Args:
        filepath: The path to the arrow file
        one_dim_target: Whether the target is one-dimensional

    Returns:
        A tuple containing the trajectory and metadata
    """
    assert Path(filepath).exists(), f"Filepath {filepath} does not exist"
    dataset = FileDataset(path=Path(filepath), freq="h", one_dim_target=one_dim_target)
    coords, metadata = zip(*[(coord["target"], coord["start"]) for coord in dataset])
    coordinates = np.stack(coords)
    if coordinates.ndim > 2:  # if not one_dim_target:
        coordinates = coordinates.squeeze()
    return coordinates, metadata


def get_system_filepaths(
    system_name: str, base_dir: Union[str, Path], split: str = "train"
) -> List[Path]:
    """
    Retrieve sorted filepaths for a given dynamical system.

    This function finds and sorts all Arrow files for a specified dynamical system
    within a given directory structure.

    Args:
        system_name (str): The name of the dynamical system.
        base_dir (Union[str, Path]): The base directory containing the data.
        split (str, optional): The data split to use (e.g., "train", "test"). Defaults to "train".

    Returns:
        List[Path]: A sorted list of Path objects for the Arrow files of the specified system.

    Raises:
        Exception: If the directory for the specified system does not exist.

    Note:
        The function assumes that the Arrow files are named with a numeric prefix
        (e.g., "1_T-1024.arrow") and sorts them based on this prefix.
    """
    dyst_dir = os.path.join(base_dir, split, system_name)
    if not os.path.exists(dyst_dir):
        raise Exception(f"Directory {dyst_dir} does not exist.")

    # NOTE: sorting by numerical order wrt sample index is very important for consistency
    filepaths = sorted(
        list(Path(dyst_dir).glob("*.arrow")), key=lambda x: int(x.stem.split("_")[0])
    )
    return filepaths


def load_dyst_samples(
    dyst_name: str,
    base_dir: str,
    split: str,
    one_dim_target: bool,
    num_samples: int | None = None,
) -> np.ndarray:
    """
    Load a set of sample trajectories of a given dynamical system from an arrow file

    Args:
        dyst_name: The name of the dynamical system
        base_dir: The base directory containing the data
        split: The data split to use (e.g., "train", "test")
        num_samples: The number of samples to load

    Returns:
        A (num_samples, num_features, num_timesteps) numpy array containing the trajectories
    """
    filepaths = get_system_filepaths(dyst_name, base_dir, split)
    dyst_coords_samples = []
    for filepath in filepaths[:num_samples]:
        dyst_coords, _ = load_trajectory_from_arrow(filepath, one_dim_target)
        dyst_coords_samples.append(dyst_coords)

    dyst_coords_samples = np.array(dyst_coords_samples)  # type: ignore
    return dyst_coords_samples


def make_ensemble_from_arrow_dir(
    base_dir: str,
    split: str,
    dyst_names_lst: list[str] | None = None,
    num_samples: int | None = None,
    num_systems: int | None = None,
    one_dim_target: bool = False,
    num_processes: int = cpu_count(),
) -> dict[str, np.ndarray]:
    ensemble: dict[str, np.ndarray] = {}
    if dyst_names_lst is None:
        data_dir = os.path.join(base_dir, split)
        dyst_names_lst = sorted(
            [folder.name for folder in Path(data_dir).iterdir() if folder.is_dir()]
        )

    if num_systems is not None:
        assert num_systems <= len(dyst_names_lst), (
            f"num_systems {num_systems} must be less than or equal to the number of systems in the directory {len(dyst_names_lst)}"
        )
        dyst_names_lst = dyst_names_lst[:num_systems]

    # Prepare arguments for multiprocessing
    args = [
        (dyst_name, base_dir, split, one_dim_target, num_samples)
        for dyst_name in dyst_names_lst
    ]

    # Use multiprocessing to process each dyst_name
    with Pool(num_processes) as pool:
        results = pool.starmap(load_dyst_samples, args)

    # Collect results into the ensemble dictionary
    for dyst_name, dyst_coords_samples in zip(dyst_names_lst, results):
        ensemble[dyst_name] = dyst_coords_samples

    return ensemble
