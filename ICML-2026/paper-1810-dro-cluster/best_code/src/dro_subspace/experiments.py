"""
FILE: experiments.py
INPUT: Experiment configuration, command-line choices, and optional existing CSV outputs.
OUTPUT: Reproducible ablation result DataFrames and CSV files.
POS: Experiment orchestration layer for the ICML release artifact.
NOTE: Update this header and the folder's _MANIFEST.md explicitly if logic changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tqdm.auto import tqdm

from dro_subspace.dro import compute_dro_coefficients
from dro_subspace.metrics import clustering_score, spectral_clustering
from dro_subspace.synthetic import make_cluster_labels, normalize_design, sample_cluster_sizes, sample_random_subspace

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]
RecordValue = int | float

DEFAULT_RESULTS_DIR = Path("results") / "ablation_results"
DEFAULT_N_SAMPLES = 250
DEFAULT_N_VARIABLES = 500
DEFAULT_N_CLUSTERS = 25
DEFAULT_N_EXPERIMENTS = 10
DEFAULT_SEED_START = 2021
DEFAULT_NOISE_RANGE = (0.0, 0.5)
DEFAULT_RHO_RANGE = (0.0, 0.5)
DEFAULT_N_SIMULATIONS = 1000
DEFAULT_MAX_OUTER_ITERATIONS = 5000

ADMM_RHO_VALUES = (0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
MISSPECIFIED_K_VALUES = (10, 15, 20, 23, 25, 27, 30, 35, 40)
ALPHA_VALUES = (0.001, 0.01, 0.05, 0.1, 0.2)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExperimentConfig:
    n_samples: int = DEFAULT_N_SAMPLES
    n_variables: int = DEFAULT_N_VARIABLES
    n_clusters: int = DEFAULT_N_CLUSTERS
    noise_range: tuple[float, float] = DEFAULT_NOISE_RANGE
    rho_range: tuple[float, float] = DEFAULT_RHO_RANGE
    n_simulations: int = DEFAULT_N_SIMULATIONS
    max_outer_iterations: int = DEFAULT_MAX_OUTER_ITERATIONS


def _load_records(path: Path) -> list[dict[str, RecordValue]]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    return [dict(row) for row in frame.to_dict("records")]


def _save_records(records: list[dict[str, RecordValue]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False)


def _trial_seed(seed: int, offset: int) -> int:
    return seed * 1_000_003 + offset


def generate_trial(config: ExperimentConfig, seed: int) -> tuple[FloatArray, IntArray]:
    """Generate one normalized synthetic trial and its true labels."""
    cluster_sizes = sample_cluster_sizes(config.n_variables, config.n_clusters, seed)
    samples, _, _, _, _ = sample_random_subspace(
        n_samples=config.n_samples,
        n_variables=config.n_variables,
        cluster_sizes=cluster_sizes,
        n_group_factors=None,
        n_group_factors_range=None,
        rho_range=config.rho_range,
        noise_range=config.noise_range,
        seed=seed,
        positive_loadings_only=False,
    )
    return normalize_design(samples, n_removed_pcs=0), make_cluster_labels(cluster_sizes)


def compute_trial_scores(
    samples: FloatArray,
    labels: IntArray,
    config: ExperimentConfig,
    alpha: float,
    admm_rho: float,
    clustering_k: int,
    seed: int,
) -> tuple[float, float]:
    coefs = compute_dro_coefficients(
        samples,
        alpha=alpha,
        n_simulations=config.n_simulations,
        seed=seed,
        admm_rho=admm_rho,
        max_outer_iterations=config.max_outer_iterations,
    )
    predicted = spectral_clustering(coefs, clustering_k, config.n_samples)
    score = clustering_score(labels, predicted)
    return score.accuracy, score.adjusted_mutual_information


def run_admm_rho_ablation(config: ExperimentConfig, seeds: list[int], output_dir: Path) -> pd.DataFrame:
    logger.info("Running ADMM rho ablation.")
    out_path = output_dir / "ablation_admm_rho.csv"
    records = _load_records(out_path)
    done = {(int(row["seed"]), float(row["rho_init"])) for row in records}

    for seed in tqdm(seeds, desc="ADMM rho seeds"):
        samples, labels = generate_trial(config, seed)
        for offset, rho_init in enumerate(tqdm(ADMM_RHO_VALUES, desc=f"rho seed={seed}", leave=False)):
            if (seed, float(rho_init)) in done:
                continue
            accuracy, ami = compute_trial_scores(
                samples,
                labels,
                config,
                alpha=0.05,
                admm_rho=float(rho_init),
                clustering_k=config.n_clusters,
                seed=_trial_seed(seed, 100 + offset),
            )
            records.append({"seed": seed, "rho_init": float(rho_init), "accuracy": accuracy, "ami": ami})
            _save_records(records, out_path)
    return pd.DataFrame(records)


def run_misspecified_k_ablation(config: ExperimentConfig, seeds: list[int], output_dir: Path) -> pd.DataFrame:
    logger.info("Running misspecified K ablation.")
    out_path = output_dir / "ablation_misspecified_K.csv"
    records = _load_records(out_path)
    done = {(int(row["seed"]), int(row["K_test"])) for row in records}

    for seed in tqdm(seeds, desc="Misspecified K seeds"):
        samples, labels = generate_trial(config, seed)
        coefs = compute_dro_coefficients(
            samples,
            alpha=0.05,
            n_simulations=config.n_simulations,
            seed=_trial_seed(seed, 200),
            admm_rho=1.0,
            max_outer_iterations=config.max_outer_iterations,
        )
        for k_test in tqdm(MISSPECIFIED_K_VALUES, desc=f"K seed={seed}", leave=False):
            if (seed, int(k_test)) in done:
                continue
            predicted = spectral_clustering(coefs, int(k_test), config.n_samples)
            score = clustering_score(labels, predicted)
            records.append(
                {
                    "seed": seed,
                    "K_test": int(k_test),
                    "K_true": config.n_clusters,
                    "accuracy": score.accuracy,
                    "ami": score.adjusted_mutual_information,
                }
            )
            _save_records(records, out_path)
    return pd.DataFrame(records)


def run_alpha_ablation(config: ExperimentConfig, seeds: list[int], output_dir: Path) -> pd.DataFrame:
    logger.info("Running alpha ablation.")
    out_path = output_dir / "ablation_alpha.csv"
    records = _load_records(out_path)
    done = {(int(row["seed"]), float(row["alpha"])) for row in records}

    for seed in tqdm(seeds, desc="Alpha seeds"):
        samples, labels = generate_trial(config, seed)
        for offset, alpha in enumerate(tqdm(ALPHA_VALUES, desc=f"alpha seed={seed}", leave=False)):
            if (seed, float(alpha)) in done:
                continue
            accuracy, ami = compute_trial_scores(
                samples,
                labels,
                config,
                alpha=float(alpha),
                admm_rho=1.0,
                clustering_k=config.n_clusters,
                seed=_trial_seed(seed, 300 + offset),
            )
            records.append({"seed": seed, "alpha": float(alpha), "accuracy": accuracy, "ami": ami})
            _save_records(records, out_path)
    return pd.DataFrame(records)


def run_ablation_suite(
    ablation: str,
    n_experiments: int = DEFAULT_N_EXPERIMENTS,
    seed_start: int = DEFAULT_SEED_START,
    output_dir: Path = DEFAULT_RESULTS_DIR,
    n_simulations: int = DEFAULT_N_SIMULATIONS,
    max_outer_iterations: int = DEFAULT_MAX_OUTER_ITERATIONS,
) -> dict[str, pd.DataFrame]:
    """Run selected ablations and return result tables keyed by ablation name."""
    if ablation not in {"rho", "K", "alpha", "all"}:
        raise ValueError("ablation must be one of: rho, K, alpha, all.")
    if n_experiments <= 0:
        raise ValueError("n_experiments must be positive.")

    config = ExperimentConfig(n_simulations=n_simulations, max_outer_iterations=max_outer_iterations)
    seeds = list(range(seed_start, seed_start + n_experiments))
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, pd.DataFrame] = {}

    if ablation in {"rho", "all"}:
        results["rho"] = run_admm_rho_ablation(config, seeds, output_dir)
    if ablation in {"K", "all"}:
        results["K"] = run_misspecified_k_ablation(config, seeds, output_dir)
    if ablation in {"alpha", "all"}:
        results["alpha"] = run_alpha_ablation(config, seeds, output_dir)
    return results
