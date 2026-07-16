"""
Script to generate and save trajectory ensembles for a given set of dynamical systems.
"""

import logging
import os
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
from scaleformer.dyst_data import DynSysSampler
from scaleformer.events import InstabilityEvent, TimeLimitEvent, TimeStepEvent
from scaleformer.sampling import OnAttractorInitCondSampler, SignedGaussianParamSampler
from scaleformer.utils import split_systems


def default_attractor_tests(tests_to_use: list[str]) -> list[Callable]:
    """Builds default attractor tests to check for each trajectory ensemble"""
    default_tests = [
        partial(check_not_linear, r2_threshold=0.99, eps=1e-10),  # pretty lenient
        partial(check_boundedness, threshold=1e4, max_zscore=15),
        partial(check_not_fixed_point, atol=1e-3, tail_prop=0.1),
        # for STRICT MODE (strict criteria for detecting limit cycles), try:
        # min_prop_recurrences = 0.1, min_counts_per_rtime = 100, min_block_length=50, min_recurrence_time = 10, enforce_endpoint_recurrence = True,
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


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    test_systems, train_systems = split_systems(
        cfg.sampling.test_split,
        seed=cfg.sampling.rseed,
        sys_class=cfg.sampling.sys_class,
    )

    # solve_ivp events
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

    param_sampler = SignedGaussianParamSampler(
        random_seed=cfg.sampling.rseed,
        scale=cfg.sampling.param_scale,
        sign_match_probability=cfg.sampling.sign_match_probability,
        ignore_probability=cfg.sampling.ignore_probability,
        verbose=cfg.sampling.verbose,
    )
    ic_sampler = OnAttractorInitCondSampler(
        reference_traj_length=cfg.sampling.reference_traj.length,
        reference_traj_transient=cfg.sampling.reference_traj.transient,
        reference_traj_n_periods=cfg.sampling.reference_traj.n_periods,
        reference_traj_atol=cfg.sampling.reference_traj.atol,
        reference_traj_rtol=cfg.sampling.reference_traj.rtol,
        recompute_standardization=cfg.sampling.standardize,  # Important!
        events=event_fns,
        random_seed=cfg.sampling.rseed,
        silence_integration_errors=cfg.sampling.silence_integration_errors,
        verbose=int(cfg.sampling.verbose),
    )

    num_periods_lst = np.arange(
        cfg.sampling.num_periods_min, cfg.sampling.num_periods_max + 1
    ).tolist()

    sys_sampler = DynSysSampler(
        rseed=cfg.sampling.rseed,
        num_periods=num_periods_lst,
        num_points=cfg.sampling.num_points,
        num_ics=cfg.sampling.num_ics,
        num_param_perturbations=cfg.sampling.num_param_perturbations,
        param_sampler=param_sampler,
        ic_sampler=ic_sampler,
        events=event_fns,
        verbose=cfg.sampling.verbose,
        split_coords=cfg.sampling.split_coords,
        attractor_tests=default_attractor_tests(cfg.validator.attractor_tests),
        validator_transient_frac=cfg.validator.transient_time_frac,
        save_failed_trajs=cfg.validator.save_failed_trajs,
    )

    param_dir = (
        os.path.join(cfg.sampling.data_dir, "parameters")
        if cfg.sampling.save_params
        else None
    )
    traj_stats_dir = (
        os.path.join(cfg.sampling.data_dir, "trajectory_stats")
        if cfg.sampling.save_traj_stats
        else None
    )

    split_prefix = cfg.sampling.split_prefix + "_" if cfg.sampling.split_prefix else ""
    run_name = cfg.run_name + "_" if cfg.run_name else ""
    for split, systems in [("train", train_systems), ("test", test_systems)]:
        split_name = f"{split_prefix}{split}"
        sys_sampler.sample_ensembles(
            systems=systems,
            save_dir=cfg.sampling.data_dir,
            split=split_name,
            split_failures=f"{split_prefix}failed_attractors_{split}",
            samples_process_interval=1,
            save_params_dir=f"{param_dir}/{split_name}" if param_dir else None,
            save_traj_stats_dir=f"{traj_stats_dir}/{split_name}"
            if traj_stats_dir
            else None,
            standardize=cfg.sampling.standardize,
            use_multiprocessing=cfg.sampling.multiprocessing,
            reset_attractor_validator=True,
            silent_errors=cfg.sampling.silence_integration_errors,
            atol=cfg.sampling.atol,
            rtol=cfg.sampling.rtol,
            use_tqdm=False,
        )
        sys_sampler.save_summary(
            os.path.join(
                "outputs",
                f"{run_name}{split_prefix}{split}_attractor_checks.json",
            ),
        )


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    main()
