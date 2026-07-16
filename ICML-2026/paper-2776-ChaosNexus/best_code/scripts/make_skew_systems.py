"""
Search for valid skew-product dynamical sytems and generate trajectory datasets
"""

import logging
import os
from functools import partial
from itertools import permutations
from multiprocessing import Pool
from typing import Callable

import dysts.flows as flows
import hydra
import numpy as np
from dysts.systems import DynSys, get_attractor_list
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
from scaleformer.coupling_maps import (
    RandomActivatedCouplingMap,
    RandomAdditiveCouplingMap,
)
from scaleformer.dyst_data import DynSysSampler
from scaleformer.events import InstabilityEvent, TimeLimitEvent, TimeStepEvent
from scaleformer.sampling import OnAttractorInitCondSampler, SignedGaussianParamSampler
from scaleformer.skew_system import SkewProduct

import wandb


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


def additive_coupling_map_factory(
    driver_name: str,
    response_name: str,
    stats_cache: dict[str, dict[str, np.ndarray]],
    transform_scales: bool = True,
    randomize_driver_indices: bool = True,
    normalization_strategy: str = "flow_rms",
    random_seed: int = 0,
) -> Callable[[int, int], RandomAdditiveCouplingMap]:
    """
    Initialize a random additive coupling map for a skew-product dynamical system
    """
    driver_stats = stats_cache[driver_name]
    response_stats = stats_cache[response_name]
    if normalization_strategy == "mean_amp_response":
        # NOTE: response_stats actually returns reciprocals of stats i.e. stiffness / amplitude
        mean_amp_response = 1 / response_stats.get("mean_amp", 1.0)
        inv_mean_amp_driver = driver_stats.get("mean_amp", 1.0)
        driver_scale = mean_amp_response * inv_mean_amp_driver
        response_scale = 1.0
    else:
        driver_scale = driver_stats.get(normalization_strategy, 1.0)
        response_scale = response_stats.get(normalization_strategy, 1.0)

    return partial(
        RandomAdditiveCouplingMap,
        driver_scale=driver_scale,
        response_scale=response_scale,
        transform_scales=transform_scales,
        randomize_driver_indices=randomize_driver_indices,
        random_seed=random_seed,
    )


def lowrank_response_matrix(
    n: int, m: int, rng: np.random.Generator, rank: int = 1
) -> np.ndarray:
    """Initialize a low-rank perturbed response matrix"""
    v = rng.normal(0, 1, size=(n, rank))
    v /= np.linalg.norm(v, axis=0)
    svs = rng.random(size=rank)
    response_matrix = v @ np.diag(svs) @ v.T
    driving_matrix = np.eye(max(n, m - n))[:n, : m - n]
    return np.hstack([driving_matrix, np.eye(n) + response_matrix])


def activated_coupling_map_factory(
    driver_name: str,
    response_name: str,
    stats_cache: dict[str, dict[str, np.ndarray]],
    random_seed: int = 0,
    **kwargs,
) -> Callable[[int, int], RandomActivatedCouplingMap]:
    """Initialize a random activated coupling map with an ill-conditioned driving matrix"""
    driver_stats = stats_cache[driver_name]
    response_stats = stats_cache[response_name]
    driver_scale = driver_stats.get("flow_rms", 1.0)
    response_scale = response_stats.get("flow_rms", 1.0)
    return partial(
        RandomActivatedCouplingMap,
        random_seed=random_seed,
        matrix_init_fn=lowrank_response_matrix,
        driver_scale=driver_scale,
        response_scale=response_scale,
    )


def init_skew_system(
    driver_name: str, response_name: str, coupling_map_fn: Callable, **kwargs
) -> DynSys:
    """
    Initialize a skew-product dynamical system with a driver and response system

    Args:
        driver_name: name of the driver system
        response_name: name of the response system
        coupling_map_fn: function for initializing the coupling map
        kwargs: additional arguments for the SkewProduct constructor
    """
    driver = getattr(flows, driver_name)()
    response = getattr(flows, response_name)()
    coupling_map = coupling_map_fn(driver.dimension, response.dimension)
    return SkewProduct(
        driver=driver, response=response, coupling_map=coupling_map, **kwargs
    )


def sample_skew_systems(
    systems: list[str], num_pairs: int, random_seed: int = 0
) -> list[tuple[str, str]]:
    system_pairs = list(permutations(systems, 2))
    rng = np.random.default_rng(random_seed)
    sampled_pairs = rng.choice(
        len(system_pairs), size=min(num_pairs, len(system_pairs)), replace=False
    )
    return [system_pairs[i] for i in sampled_pairs]


def filter_and_split_skew_systems(
    skew_pairs: list[tuple[str, str]],
    test_split: float = 0.2,
    train_systems: list[str] | None = None,
    test_systems: list[str] | None = None,
    coupling_map_type: str = "additive",
    coupling_map_kwargs: dict | None = None,
    skew_system_kwargs: dict | None = None,
) -> tuple[list[str], list[str]]:
    """Sample skew systems from all pairs of non-skew systems and split into train/test

    Args:
        skew_pairs: List of skew system pairs to sample from
        test_split: Fraction of systems to use for testing
        random_seed: Random seed for reproducibility
        train_systems: Optional list of system names to use for training
        test_systems: Optional list of system names to use for testing

    Returns:
        Tuple of (train_systems, test_systems) where each is a list of initialized
        skew product systems
    """
    coupling_map_kwargs = coupling_map_kwargs or {}
    skew_system_kwargs = skew_system_kwargs or {}

    split_idx = int(len(skew_pairs) * (1 - test_split))
    train_pairs, test_pairs = skew_pairs[:split_idx], skew_pairs[split_idx:]

    # if provided, filter out pairs from train and test pairs that contain systems
    # that are not in the train or test sets, then recombine to update valid train/test pairs
    def is_valid_pair(pair: tuple[str, str], filter_list: list[str] | None) -> bool:
        return (
            True
            if filter_list is None
            else all(system in filter_list for system in pair)
        )

    valid_train_pairs = filter(
        lambda pair: is_valid_pair(pair, train_systems), train_pairs
    )
    valid_test_pairs = filter(
        lambda pair: is_valid_pair(pair, test_systems), test_pairs
    )
    invalid_train_pairs = filter(
        lambda pair: not is_valid_pair(pair, train_systems), train_pairs
    )
    invalid_test_pairs = filter(
        lambda pair: not is_valid_pair(pair, test_systems), test_pairs
    )
    train_pairs = list(valid_train_pairs) + list(invalid_test_pairs)
    test_pairs = list(valid_test_pairs) + list(invalid_train_pairs)

    coupling_map_factory = {
        "additive": additive_coupling_map_factory,
        "activated": activated_coupling_map_factory,
    }[coupling_map_type]

    systems = {}
    for split, skew_pairs in [("train", train_pairs), ("test", test_pairs)]:
        systems[split] = [
            init_skew_system(
                driver,
                response,
                coupling_map_fn=coupling_map_factory(
                    driver, response, **coupling_map_kwargs
                ),
                **skew_system_kwargs,
            )
            for driver, response in skew_pairs
        ]

    return systems["train"], systems["test"]


def _compute_system_stats(
    system: str,
    n: int,
    num_periods: int,
    transient: int,
    atol: float,
    rtol: float,
    stiffness: float = 1.0,
) -> tuple[str, dict[str, np.ndarray]]:
    """
    Compute RMS scale and amplitude for a single system's trajectory.
    Returns the reciprocals of the computed stats, with a stiffness factor applied
    """
    sys = getattr(flows, system)()
    ts, traj = sys.make_trajectory(
        n, pts_per_period=n // num_periods, return_times=True, atol=atol, rtol=rtol
    )
    assert traj is not None, f"{system} should be integrable"
    ts, traj = ts[transient:], traj[transient:]
    flow_rms = np.sqrt(
        np.mean([np.asarray(sys(x, t)) ** 2 for x, t in zip(traj, ts)], axis=0)
    )
    mean_amp = np.mean(np.abs(traj), axis=0)
    system_stats = {
        "flow_rms": stiffness / flow_rms,
        "mean_amp": stiffness / mean_amp,
    }
    return system, system_stats


def init_trajectory_stats_cache(
    systems: list[str],
    traj_length: int,
    num_periods: int,
    traj_transient: float,
    atol: float,
    rtol: float,
    multiprocess_kwargs: dict = {},
) -> dict[str, dict[str, np.ndarray]]:
    """Initialize a cache of vector field RMS scales and amplitudes for each system using multiprocessing"""
    _compute_stats_worker = partial(
        _compute_system_stats,
        n=traj_length,
        num_periods=num_periods,
        transient=int(traj_length * traj_transient),
        atol=atol,
        rtol=rtol,
    )
    with Pool(**multiprocess_kwargs) as pool:
        results = pool.map(_compute_stats_worker, systems)
    return dict(results)


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    # set up wandb project and logging if enabled
    if cfg.wandb.log:
        run = wandb.init(
            project=cfg.wandb.project_name,
            entity=cfg.wandb.entity,
            name=cfg.run_name,
            config=dict(cfg),
            sync_tensorboard=False,  # auto-upload tensorboard metrics
            group=cfg.wandb.group_name,
            resume=cfg.wandb.resume,
        )
        logger.info(f"Wandb initialized: {run.id}")

        wandb.define_metric("sample_idx")
        wandb.define_metric("num_systems_valid", step_metric="sample_idx")
        wandb.define_metric("num_systems_integrated", step_metric="sample_idx")
        for attractor_test_name in cfg.validator.attractor_tests:
            wandb.define_metric(
                f"failed_{attractor_test_name}", step_metric="sample_idx"
            )

    systems = get_attractor_list(sys_class=cfg.sampling.sys_class)

    # events for solve_ivp
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
    event_fns = [time_limit_event, instability_event, time_step_event]

    # initialize samplers for perturbing systems
    param_sampler = SignedGaussianParamSampler(
        random_seed=cfg.sampling.rseed,
        scale=cfg.sampling.param_scale,
        verbose=cfg.sampling.verbose,
    )
    ic_sampler = OnAttractorInitCondSampler(
        reference_traj_length=cfg.sampling.reference_traj.length,
        reference_traj_n_periods=cfg.sampling.reference_traj.n_periods,
        reference_traj_transient=cfg.sampling.reference_traj.transient,
        reference_traj_atol=cfg.sampling.reference_traj.atol,
        reference_traj_rtol=cfg.sampling.reference_traj.rtol,
        recompute_standardization=cfg.sampling.standardize,  # Important (if standardize=True)
        random_seed=cfg.sampling.ic_rseed,
        events=event_fns,
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
        wandb_run=run if cfg.wandb.log else None,
        multiprocess_kwargs=dict(cfg.multiprocess_kwargs),
    )

    # sample skew system train/test splits
    skew_pairs = sample_skew_systems(
        systems, cfg.skew.num_pairs, random_seed=cfg.skew.pairs_rseed
    )
    # optionally make only a subset of the skew pairs, for memory-constrained runs and for power-law dataset generation
    skew_pairs = skew_pairs[cfg.skew.sys_idx_low : cfg.skew.sys_idx_high]
    logger.info(f"Making {len(skew_pairs)} skew pairs: {skew_pairs}")

    logger.info(
        f"Sampled {len(skew_pairs)}/{len(systems) * (len(systems) - 1)} system pair candidates"
    )

    base_systems = set(sys for pair in skew_pairs for sys in pair)
    logger.info(f"Initializing trajectory scale cache for {len(base_systems)} systems")
    stats_cache = init_trajectory_stats_cache(
        list(base_systems),
        cfg.sampling.num_points,
        cfg.sampling.num_periods,
        cfg.sampling.reference_traj.transient,
        atol=cfg.sampling.reference_traj.atol,
        rtol=cfg.sampling.reference_traj.rtol,
        multiprocess_kwargs=dict(cfg.multiprocess_kwargs),
    )

    logger.info(
        f"Attempting to split {len(skew_pairs)} skew pairs into a "
        f"{1 - cfg.sampling.test_split:.2f}/{cfg.sampling.test_split:.2f} train/test split"
    )
    train_systems, test_systems = filter_and_split_skew_systems(
        skew_pairs,
        test_split=cfg.sampling.test_split,
        coupling_map_type=cfg.skew.coupling_map_type,
        coupling_map_kwargs={  # add the stats cache to the coupling map kwargs
            "stats_cache": stats_cache,
            **cfg.skew.coupling_map,
        },
    )
    train_prop = len(train_systems) / len(skew_pairs)
    test_prop = len(test_systems) / len(skew_pairs)
    logger.info(
        f"Achieved {len(train_systems)}/{len(test_systems)} = {train_prop:.2f}/{test_prop:.2f}"
        " train/test split after filtering"
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

    concise_comb_summary_dict = {}
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
        summary_json_path = os.path.join(
            "outputs",
            f"{run_name}{split_prefix}{split}_attractor_checks.json",
        )
        summary_dict = sys_sampler.save_summary(summary_json_path, return_dict=True)
        # now log summary_dict to wandb
        if summary_dict is not None:
            artifact = wandb.Artifact(
                name=f"{split_prefix}{split}_attractor_checks",
                type="attractor_summary",
            )
            artifact.add_file(summary_json_path)
            run.log_artifact(artifact)

            concise_comb_summary_dict[split] = {
                k: summary_dict[k]
                for k in [
                    "num_parameter_successes",
                    "num_total_candidates",
                    "valid_dyst_counts",
                    "valid_samples",
                ]
            }
            run.summary.update(concise_comb_summary_dict)
            print(f"updated summary for wandb run {run.id}")

            # # NOTE: if we want to actually log the summary dict and make it queryable:
            # run.log({f"{split_prefix}{split}_attractor_checks": summary_dict})

    # terminate wandb run after data run
    if cfg.wandb.log:
        wandb.finish(exit_code=0)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    main()
