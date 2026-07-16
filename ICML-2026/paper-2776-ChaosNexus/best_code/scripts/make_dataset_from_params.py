"""
Script to generate and save trajectory ensembles for a given set of dynamical systems.
"""

import json
import logging
from functools import partial
from typing import Callable

import hydra
import numpy as np

from scaleformer.attractor import (
    check_boundedness,
    check_lyapunov_exponent,
    check_not_fixed_point,
    check_not_limit_cycle,
    check_not_linear,
    check_power_spectrum,
    check_stationarity,
    check_zero_one_test,
)
from scaleformer.dyst_data import DynSysSamplerRestartIC
from scaleformer.events import InstabilityEvent, TimeLimitEvent, TimeStepEvent
from scaleformer.utils import init_skew_system_from_params


def default_attractor_tests(tests_to_use: list[str]) -> list[Callable]:
    """Builds default attractor tests to check for each trajectory ensemble"""
    default_tests = [
        partial(check_not_linear, r2_threshold=0.99, eps=1e-10),  # pretty lenient
        partial(check_boundedness, threshold=1e4, max_zscore=15),
        partial(check_not_fixed_point, atol=1e-3, tail_prop=0.1),
        partial(check_zero_one_test, threshold=0.2, strategy="score"),
        partial(
            check_not_limit_cycle,
            tolerance=1e-3,
            min_prop_recurrences=0.1,
            min_counts_per_rtime=200,
            min_block_length=50,
            enforce_endpoint_recurrence=True,
        ),
        partial(
            check_power_spectrum, rel_peak_height=1e-5, rel_prominence=1e-5, min_peaks=4
        ),
        partial(check_lyapunov_exponent, traj_len=200),
        partial(check_stationarity, p_value=0.05),
    ]
    filtered_tests = [
        test for test in default_tests if test.func.__name__ in tests_to_use
    ]
    return filtered_tests


def create_sample_idx_mapping(
    subdir_sample_counts_dict: dict[str, int],
) -> Callable[[np.ndarray | list[int]], tuple[np.ndarray, np.ndarray]]:
    # Create arrays for fast lookup
    system_names = []
    boundaries = [0]  # Start with 0

    tot_systems = sum(subdir_sample_counts_dict.values())
    # Build the boundaries and system names arrays
    for system_name, count in subdir_sample_counts_dict.items():
        system_names.append(system_name)
        boundaries.append(boundaries[-1] + count)

    # Convert to numpy arrays for faster operations
    boundaries = np.array(boundaries)
    system_names = np.array(system_names)

    def get_system_names_and_positions(
        sample_idxs: np.ndarray | list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        # Validate input
        sample_idxs = np.asarray(sample_idxs)
        if np.any((sample_idxs < 0) | (sample_idxs >= tot_systems)):
            raise ValueError(f"All sample_idxs must be between 0 and {tot_systems - 1}")

        # Find the index where each sample_idx would be inserted in boundaries
        # Subtract 1 to get the correct system index
        system_indices = np.searchsorted(boundaries, sample_idxs, side="right") - 1

        # Calculate relative positions within each system
        relative_positions = sample_idxs - boundaries[system_indices]

        # Return both the system names and relative positions
        return system_names[system_indices], relative_positions

    return get_system_names_and_positions


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    time_limit_event = partial(
        TimeLimitEvent,
        max_duration=cfg.events.max_duration,
        verbose=cfg.events.verbose,
    )
    instability_event = partial(
        InstabilityEvent,
        threshold=cfg.events.instability_threshold,
        verbose=cfg.events.verbose,
    )
    time_step_event = partial(
        TimeStepEvent,
        min_step=cfg.events.min_step,
        verbose=cfg.events.verbose,
    )
    event_fns = [time_limit_event, time_step_event, instability_event]

    num_periods_lst = np.arange(
        cfg.sampling.num_periods_min, cfg.sampling.num_periods_max + 1
    ).tolist()

    sys_sampler = DynSysSamplerRestartIC(
        rseed=cfg.sampling.rseed,
        num_periods=num_periods_lst,
        num_points=cfg.sampling.num_points,
        num_ics=cfg.sampling.num_ics,
        events=event_fns,
        split_coords=cfg.sampling.split_coords,
        attractor_tests=default_attractor_tests(cfg.validator.attractor_tests),
        validator_transient_frac=cfg.validator.transient_time_frac,
    )

    # load params dict from json file
    with open(cfg.restart_sampling.params_json_path, "r") as f:
        params_dicts = json.load(f)

    tot_systems = sum(len(params_dict) for params_dict in params_dicts.values())
    logger.info(f"Total systems: {tot_systems}")

    subdir_sample_counts_dict = {}
    for system_name, system_params in params_dicts.items():
        n_samples = len(system_params)
        subdir_sample_counts_dict[system_name] = n_samples

    logger.info(f"Subdir sample counts: {subdir_sample_counts_dict}")
    # Create the mapping function
    get_system_names_and_positions = create_sample_idx_mapping(
        subdir_sample_counts_dict
    )

    # chunk the params dict to process only a subset of systems_batch_size
    systems_batch_size = cfg.restart_sampling.systems_batch_size
    n_batches = (
        tot_systems + systems_batch_size - 1
    ) // systems_batch_size  # Ceiling division\

    logger.info(
        f"Total batches: {n_batches} with systems batch size {systems_batch_size}"
    )

    bi_low = cfg.restart_sampling.batch_idx_low
    bi_high = cfg.restart_sampling.batch_idx_high
    if bi_low is not None and bi_high is not None:
        if bi_low < 0 or bi_high >= n_batches:
            raise ValueError("invalid batch index!")
        batch_indices = np.arange(bi_low, bi_high)
    else:
        batch_indices = np.arange(0, n_batches)

    logger.info(f"batch indices: {batch_indices}")
    for batch_idx in batch_indices:
        start_idx = batch_idx * systems_batch_size
        end_idx = min((batch_idx + 1) * systems_batch_size, tot_systems)
        sample_idx_lst = np.arange(start_idx, end_idx)
        logger.info(f"Processing batch {batch_idx} with {len(sample_idx_lst)} systems")
        # positions is list of param_pert index e.g. Lorenz-pp0, Lorenz-pp1, etc.
        system_names, positions = get_system_names_and_positions(sample_idx_lst)

        # Initialize systems, assuming skew systems
        systems = []
        for name, pos in zip(system_names, positions):
            driver_name, response_name = name.split("_")
            system_params = params_dicts[name]
            params = system_params[pos]
            pp_idx = params["sample_idx"]
            name_with_pp_idx = f"{name}-pp{pp_idx}"
            sys = init_skew_system_from_params(driver_name, response_name, params)
            sys.name = name_with_pp_idx
            systems.append(sys)

        sys_sampler.sample_ensembles(
            systems,
            save_dir=cfg.sampling.data_dir,
            split="train",
            samples_process_interval=1,
            starting_sample_idx=cfg.restart_sampling.starting_sample_idx,
            save_first_sample=cfg.restart_sampling.save_first_sample,
            standardize=cfg.sampling.standardize,
            use_multiprocessing=cfg.sampling.multiprocessing,
            silent_errors=cfg.sampling.silence_integration_errors,
            atol=cfg.sampling.atol,
            rtol=cfg.sampling.rtol,
            use_tqdm=False,
        )


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    main()
