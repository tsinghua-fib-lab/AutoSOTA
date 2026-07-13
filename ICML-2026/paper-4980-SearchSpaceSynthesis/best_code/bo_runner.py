from typing import Callable, Sequence, Any
import logging
import random
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from cosy.core.types import Type
from cosy.core import Synthesizer
from cosy.core.solution_space import SolutionSpace
from cosy.core.tree import Tree
from bayesian_optimization.examples.damg_nas.damg_repo_algebras import pytorch_function_algebra, pretty_term_algebra
from bayesian_optimization.examples.damg_nas.damg_repo import DAMGrepository
from utils import (
    create_and_save_dataset,
    load_dataset,
    DEFAULT_DATASET_PATH,
    get_or_create_pre_samples,
)
from cosy.evolutionary_algorithms import (
    SimpleGeneticProgramming,
    RandomLimitedDepthFirstInitialization,
    ResolutionMutation,
    Crossover,
    ScalarFitnessComparator,
    FitnessProportionalSelection,
    AgeBasedReplacement,
)

from bayesian_optimization.bo import (
    BayesianOptimization,
    Suggestion,
)

from bayesian_optimization.examples.damg_nas.damg_kernels import (
    noisy_combined_hierarchical_kernel,
    noisy_hierarchical_damg_kernel,
    noisy_hierarchical_tree_kernel,
    noisy_hierarchical_wl_kernel,
    noisy_hierarchical_damg_kernel_13,
    tree_kernel_1, tree_kernel_2, tree_kernel_3,
    wl_kernel_1, wl_kernel_2, wl_kernel_3,
    damg_kernel_1, damg_kernel_2, damg_kernel_3,
)

from bayesian_optimization.kernels.graph_kernel import WeisfeilerLehmanKernel
from bayesian_optimization.kernels.tree_kernel import OrderedRootedSubtreeKernel

from bayesian_optimization.examples.damg_nas.damg_targets import target_to_name
from bo_experiment_config import (
    EVAL_BUDGET,
    INITIAL_SAMPLE_SIZE,
    RANKING_POOL_SIZE,
    DEFAULT_DISTANCE_KERNEL,
    DISTANCE_KERNEL_OPTIONS,
)
from bo_metrics import compute_ranking_metrics
from bo_distance_analysis import compute_distance_to_best, compute_distance_to_topk

logger = logging.getLogger(__name__)


def _progress_iterable(iterable, *, total: int, desc: str, enabled: bool):
    """Wrap ``iterable`` in a tqdm progress bar when enabled and tqdm is importable.

    The wrapping is intentionally local to the CLI/runner: library callers leave
    ``enabled=False`` and get the plain iterable back. tqdm itself is imported
    defensively so that environments without tqdm fall back to the plain
    iterator without raising.
    """
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=False)


def _resolve_distance_kernel(name: str | None):
    """Resolve a CLI distance-kernel name to an instantiated kernel.

    Defaults to ``"wl"`` when ``name`` is None. Only the structural kernels
    listed in ``DISTANCE_KERNEL_OPTIONS`` are accepted: ``"wl"`` →
    :class:`WeisfeilerLehmanKernel`, ``"tree"`` →
    :class:`OrderedRootedSubtreeKernel`. The surrogate (BO) kernel is no
    longer used as a distance kernel — this gives Plot 5/6 a stable structural
    metric across BO kernel configurations.
    """
    n = (name or DEFAULT_DISTANCE_KERNEL or "wl").lower()
    if n == "wl":
        return WeisfeilerLehmanKernel(n_iter=1, normalize=True)
    if n == "tree":
        return OrderedRootedSubtreeKernel(max_height=None, normalize=True)
    raise ValueError(
        f"Unknown distance kernel: {name!r}; allowed values are {DISTANCE_KERNEL_OPTIONS}."
    )

# =============================================================================
# BAYESIAN OPTIMIZATION RUNNER (COMPANION CODE FOR ICML 2026)
# =============================================================================
#
# Core optimization loop and instrumentation used by the ICML 2026
# experiments (see poster: https://icml.cc/virtual/2026/poster/62530). The
# runner executes a single experimental configuration (target × kernel × seed)
# and performs the following main steps:
#
# 1. construct the search space for the requested target
# 2. load presamples used to initialize the Ask/Tell optimizer
# 3. run the Ask/Tell BO loop (suggest → evaluate → observe)
# 4. record runtime instrumentation and ranking diagnostics
# 5. write per-iteration traces and ranking metrics to CSV
#
# The BO code uses an Ask/Tell-style optimizer API (initialize, suggest,
# observe) to remain interoperable with the refactored cosy-examples
# implementations.
#
# =============================================================================
# IMPORTED COMPONENTS
# =============================================================================
#
# Use the following functions from the repository:
#
# generate_search_space()
#
# get_or_create_pre_samples()
#
# f_obj(tree)
#
#
# Bayesian optimizers:
#
# BayesianOptimization (Ask/Tell interface)
#
#
# Random baseline:
#
# random_search()
#
#
# =============================================================================
# RUNTIME MEASUREMENTS
# =============================================================================
#
# Each BO iteration must measure runtime using time.perf_counter().
#
# For each iteration perform:
#
# t0 = time.perf_counter()
#
# x_next = optimizer.suggest()
#
# t1 = time.perf_counter()
#
# y_next = f_obj(x_next)
#
# t2 = time.perf_counter()
#
# optimizer.observe(x_next, y_next)
#
# t3 = time.perf_counter()
#
#
# Compute:
#
# acquisition_time = t1 - t0
#
# objective_eval_time = t2 - t1
#
# update_time = t3 - t2
#
# bo_overhead_time = acquisition_time + update_time
#
#
# Maintain cumulative wallclock time.
#
#
# =============================================================================
# TRACE DATA
# =============================================================================
#
# During optimization store the following values after each iteration:
#
# experiment_id
# method
# target_name
# kernel_name
# seed
# evaluation
# objective_value
# best_objective_value
# bo_overhead_time
# objective_eval_time
# cumulative_wallclock_time
#
#
# These rows must be appended to a pandas DataFrame.
#
#
# =============================================================================
# RANKING METRICS
# =============================================================================
#
# Ranking metrics are computed only for BO methods.
#
# Use a candidate pool generated outside the optimizer.
#
#
# predictions = surrogate.predict(candidate_pool)
#
#
# Compute metrics using functions from bo_metrics.py:
#
# kendall_tau
# spearman_rho
#
#
# Store ranking metrics in a second DataFrame with columns:
#
# experiment_id
# method
# target_name
# kernel_name
# seed
# evaluation
# kendall_tau
# spearman_rho
#
#
# =============================================================================
# FUNCTIONS TO IMPLEMENT
# =============================================================================
#
# run_single_optimization(...)
#
# Runs one optimizer instance.
#
#
# run_experiment(...)
#
# Runs all methods for a given:
#
# target
# kernel
# seeds
#
#
# This function should:
#
# - generate ranking pool
# - call run_single_optimization()
# - concatenate results
#
#
# Return:
#
# optimization_trace_df
# ranking_metrics_df
#
#
# =============================================================================


MAX_TREE_DEPTH = 10000
POPULATION_SIZE = 100
MUTATION_RATE = 0.0
RECOMBINATION_RATE = 0.98
NUMBER_OF_GENERATIONS = 100

FEATURE_DIMENSIONS = [1, 2, 3, 4, 5,] # 6, 7, 8,]
CONSTANT_VALUES = [0, 1, -1]
LEARNING_RATES = [1e-2,]


def random_search(search_space: SolutionSpace, target: Type, f_obj, budget, start_x, start_y, greater_is_better,
                  rng: random.Random):
    xp = np.asarray(start_x)
    yp = np.asarray(start_y)
    if greater_is_better:
        best_x: Tree = xp[np.argmax(yp)]
        best_y = np.max(yp)
    else:
        best_x = xp[np.argmin(yp)]
        best_y = np.min(yp)
    for i in range(budget):
        next_x = search_space.sample_tree(target, MAX_TREE_DEPTH, rng=rng)
        next_y = f_obj(next_x)
        if greater_is_better:
            if next_y > best_y:
                best_x = next_x
                best_y = next_y
        else:
            if next_y < best_y:
                best_x = next_x
                best_y = next_y
        xp = np.append(xp, next_x)
        yp = np.append(yp, next_y)
    return best_x, best_y, xp, yp


class RandomBaselineOptimizer:
    """Tiny optimizer wrapper implementing the Ask/Tell interface for a random baseline.

    This allows using the same instrumentation in `run_single_optimization` as for BO.
    """
    def __init__(self, search_space: SolutionSpace, target: Type, rng: random.Random | None = None, max_depth: int = MAX_TREE_DEPTH):
        self.search_space = search_space
        self.target = target
        self.rng = rng or random.Random()
        self.max_depth = int(max_depth)
        self._x_list: list = []
        self._y_list: list = []

    def initialize(self, obj_fun=None, x0=None, y0=None, n_pre_samples: int = 0, **kwargs):
        self._x_list = list(x0) if x0 is not None else []
        self._y_list = list(y0) if y0 is not None else []

    def suggest(self, **kwargs) -> Suggestion:
        # sample one candidate uniformly at random from the search space
        candidate = self.search_space.sample_tree(self.target, self.max_depth, rng=self.rng)
        return Suggestion(candidate=candidate, acquisition_value=None, diagnostics=None)

    def observe(self, x, y, **kwargs):
        self._x_list.append(x)
        self._y_list.append(y)

    def get_state_snapshot(self):
        return {"x_list": list(self._x_list), "y_list": list(self._y_list)}



def generate_search_space(request: Type, linear_feature_dimensions: list[int], constant_values: list[int],
                          learning_rate_values: list[float], epochs: int) -> tuple[SolutionSpace, float]:
    repo = DAMGrepository(linear_feature_dimensions=linear_feature_dimensions, constant_values=constant_values,
                         learning_rate_values=learning_rate_values,
                         n_epoch_values=[epochs])

    synthesizer = Synthesizer(repo.specification(), {})

    start_time = time.time()
    search_space = synthesizer.construct_solution_space(request).prune()
    end_time = time.time()
    construction_time = end_time - start_time
    return search_space, construction_time

try:
    x, y, x_test, y_test = load_dataset(DEFAULT_DATASET_PATH)
except (FileNotFoundError, ValueError) as e:
    print(f"Dataset not found or invalid at {DEFAULT_DATASET_PATH}. Generating new dataset.")
    x, y, x_test, y_test, _ = create_and_save_dataset(dataset_path=DEFAULT_DATASET_PATH)

def f_obj(t: Tree, seed: int | None = None) -> float:
    """Objective function: evaluate a Tree (neural network architecture) on the ODE regression task.

    This function is used ONLY when evaluating kernels as GP surrogates (evaluate_kernels_as_surrogates).
    Domain-based kernel analysis (analyze_kernel_domain_properties) does NOT need this function.

    Args:
        t: A Tree structure representing a neural network architecture
        seed: Random seed for reproducibility. If None, no seed is set.

    Returns:
        float: Test loss (lower is better)
    """
    if seed is not None:
        import random
        #import numpy as np
        import torch

        random.seed(seed)
        np.random.seed(seed)

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    learner = t.interpret(pytorch_function_algebra())
    return learner(x, y, x_test, y_test)



# =============================================================================
# DATA STRUCTURES
# =============================================================================

TRACE_COLUMNS = [
    "experiment_id",
    "method",
    "target_name",
    "kernel_name",
    "seed",
    "evaluation",
    "candidate",
    "best_candidate",
    "objective_value",
    "best_objective_value",
    "bo_overhead_time",
    "objective_eval_time",
    "cumulative_wallclock_time",
    "distance_to_best",
    "distance_to_topk",
    # EI sanity-check diagnostics (added columns; existing logging schema unchanged)
    "ei_optimizer",
    "ei_random_best",
    "ei_ratio",
    "ei_random_mean",
    "sigma_random_mean",
    "sigma_random_max",
    # GP ranking quality diagnostics
    "gp_rank_spearman",
    "gp_rank_kendall",
    "gp_rank_pearson",
]


RANKING_COLUMNS = [
    "experiment_id",
    "method",
    "target_name",
    "kernel_name",
    "seed",
    "evaluation",
    "kendall_tau",
    "spearman_rho",
]


def _attach_distance_columns(
    trace_df: pd.DataFrame,
    trees: Sequence,
    objective_values: Sequence[float],
    distance_kernel,
) -> pd.DataFrame:
    """Attach distance_to_best/distance_to_topk columns computed from collected trees.

    Trees themselves are NOT persisted in the returned DataFrame (callers write CSV
    directly from trace_df). When the trees list is empty or the kernel is missing
    the columns are filled with NaN so downstream consumers always see the schema.
    """
    n_rows = len(trace_df)
    if n_rows == 0:
        trace_df["distance_to_best"] = pd.Series(dtype=float)
        trace_df["distance_to_topk"] = pd.Series(dtype=float)
        return trace_df

    if not trees or distance_kernel is None or len(trees) != n_rows:
        trace_df["distance_to_best"] = float("nan")
        trace_df["distance_to_topk"] = float("nan")
        return trace_df

    interim = pd.DataFrame({"tree": list(trees), "objective_value": list(objective_values)})
    try:
        interim = compute_distance_to_best(interim, distance_kernel, tree_column="tree")
        interim = compute_distance_to_topk(interim, distance_kernel, tree_column="tree")
        trace_df["distance_to_best"] = interim["distance_to_best"].to_numpy()
        trace_df["distance_to_topk"] = interim["distance_to_topk"].to_numpy()
    except Exception:
        logger.exception("Distance analysis failed; filling NaN columns")
        trace_df["distance_to_best"] = float("nan")
        trace_df["distance_to_topk"] = float("nan")
    return trace_df


def compute_ei_sanity_check(
    optimizer,
    suggestion,
    iteration: int,
    n_sanity: int = 1000,
    debug_dir: str | Path | None = None,
    save_debug_arrays: bool = False,
):
    """
    Compute EI sanity diagnostics without evaluating the expensive objective.

    This helper samples `n_sanity` random candidates from the optimizer's
    search space (respecting the same target/request), uses the optimizer's
    fitted GP (`_model`) to predict mean/std, and evaluates Expected
    Improvement (EI) for the sampled points using the same acquisition
    implementation as the BO optimizer.

    Returns a dict with keys: ei_optimizer, ei_random_best, ei_ratio,
    ei_random_mean, sigma_random_mean, sigma_random_max. Values are floats
    or np.nan when diagnostics are unavailable.

    Notes:
    - Must not call the expensive objective function.
    - Uses the same EI implementation (bayesian_optimization.acquisition_function.ExpectedImprovement).
    """
    import math
    from pathlib import Path as _Path

    # Default NaN output
    nan = float("nan")
    out = {
        "ei_optimizer": nan,
        "ei_random_best": nan,
        "ei_ratio": nan,
        "ei_random_mean": nan,
        "sigma_random_mean": nan,
        "sigma_random_max": nan,
    }

    # Obtain GP model
    gp = getattr(optimizer, "_model", None)
    if gp is None:
        return out

    # Try to locate a search_space and a target/request to sample random trees
    search_space = getattr(optimizer, "search_space", None) or getattr(optimizer, "_search_space", None)
    target = getattr(optimizer, "request", None) or getattr(optimizer, "target", None)
    max_depth = getattr(optimizer, "max_depth", MAX_TREE_DEPTH)

    if search_space is None or target is None:
        return out

    # Sample random candidate trees (respect space bounds). Use a local RNG for reproducibility.
    rng = random.Random()
    samples: list = []
    try:
        for _ in range(int(n_sanity)):
            tree = search_space.sample_tree(target, max_depth=max_depth, rng=rng)
            if tree is not None: # This could be handled better to ensure len(sample) == n_sanity,
                # but n_sanity is usually really big and the correct size isn't that important...
                samples.append(tree)
    except Exception:
        # Sampling failed; return NaNs
        return out

    # Use the same EI implementation used by the BO algorithm
    try:
        from bayesian_optimization.acquisition_function import ExpectedImprovement

        greater_is_better = getattr(optimizer, "_greater_is_better", False)
        known_points = getattr(optimizer, "_x_set", None)

        # Prefer the incumbent stored with the suggestion (log1p space); fall
        # back to None so ExpectedImprovement recomputes from gp.y_train_,
        # which is already in the transformed space.
        incumbent_used = None
        if suggestion is not None and getattr(suggestion, "diagnostics", None):
            diag = suggestion.diagnostics
            if isinstance(diag, dict) and "incumbent" in diag:
                try:
                    incumbent_used = float(diag["incumbent"])
                except Exception:
                    incumbent_used = None

        ei_fn = ExpectedImprovement(
            gp, greater_is_better,
            known_points=known_points,
            incumbent=incumbent_used,
        )

        # Predict mu/sigma for the random samples (does not evaluate objective)
        mu_rand, sigma_rand = gp.predict(samples, return_std=True)
        mu_rand = np.asarray(mu_rand, dtype=float).reshape(-1)
        sigma_rand = np.asarray(sigma_rand, dtype=float).reshape(-1)

        # Determine xi used by the BO acquisition (try several fallbacks).
        xi = 0.01
        try:
            # 1) suggestion diagnostics may include xi (preferred)
            if suggestion is not None and getattr(suggestion, "diagnostics", None):
                diag = suggestion.diagnostics
                if isinstance(diag, dict):
                    xi = float(diag.get("xi", diag.get("ei_xi", xi)))
        except Exception:
            pass
        # Evaluate EI on the random set using the determined xi.
        ei_dict = ei_fn.evaluate_batch(samples, xi=xi)
        ei_random = np.asarray([float(ei_dict.get(t, 0.0)) for t in samples], dtype=float)

        # Compute stats
        best_idx = int(np.argmax(ei_random)) if ei_random.size > 0 else -1
        ei_random_best = float(ei_random[best_idx]) if best_idx >= 0 else nan
        ei_random_mean = float(np.mean(ei_random)) if ei_random.size > 0 else nan

        sigma_mean = float(np.mean(sigma_rand)) if sigma_rand.size > 0 else nan
        sigma_max = float(np.max(sigma_rand)) if sigma_rand.size > 0 else nan

        # EI at the optimizer's suggested point. Prefer the acquisition_value if present
        ei_optimizer = nan
        if suggestion is not None and getattr(suggestion, "acquisition_value", None) is not None:
            try:
                ei_optimizer = float(suggestion.acquisition_value)
            except Exception:
                ei_optimizer = float(ei_fn(suggestion.candidate, xi=xi))
        else:
            # Fallback: compute directly
            try:
                ei_optimizer = float(ei_fn(suggestion.candidate, xi=xi))
            except Exception:
                ei_optimizer = nan

        ei_ratio = float(ei_optimizer / ei_random_best) if (not math.isnan(ei_random_best) and ei_random_best != 0.0) else nan

        out.update(
            {
                "ei_optimizer": ei_optimizer,
                "ei_random_best": ei_random_best,
                "ei_ratio": ei_ratio,
                "ei_random_mean": ei_random_mean,
                "sigma_random_mean": sigma_mean,
                "sigma_random_max": sigma_max,
            }
        )

        # Optionally persist the raw EI array for a few iterations to aid histogram plotting
        if save_debug_arrays and debug_dir is not None:
            try:
                p = _Path(debug_dir)
                p.mkdir(parents=True, exist_ok=True)
                np.save(p / f"ei_random_iter_{iteration}.npy", ei_random)
            except Exception:
                pass

    except Exception:
        logger.exception("EI sanity check failed at iteration %s", iteration)

    return out


def compute_gp_ranking_diagnostics(
    gp,
    X_obs,
    y_obs,
    iteration: int | None = None,
    debug_dir: str | Path | None = None,
    save_debug_arrays: bool = False,
):
    """
    Compute ranking correlations between GP predictions and observed objectives.

    Returns a dict with keys: gp_rank_spearman, gp_rank_kendall, gp_rank_pearson.

    Optionally saves the raw predicted and true arrays to `debug_dir` when
    `save_debug_arrays` is True (filenames include iteration index).
    """
    nan = float("nan")
    out = {
        "gp_rank_spearman": nan,
        "gp_rank_kendall": nan,
        "gp_rank_pearson": nan,
    }

    try:
        from scipy.stats import spearmanr, kendalltau, pearsonr
        import numpy as _np
        from pathlib import Path as _Path

        if gp is None or X_obs is None or y_obs is None:
            return out

        # Ensure arrays
        X_arr = _np.asarray(X_obs, dtype=object)
        y_arr = _np.asarray(y_obs, dtype=float).reshape(-1)

        if X_arr.size == 0 or y_arr.size == 0:
            return out

        # GP predict on observed inputs (must not evaluate objective)
        if not hasattr(gp, "predict"):
            return out

        mu_obs = gp.predict(list(X_arr))
        mu_obs = _np.asarray(mu_obs, dtype=float).reshape(-1)

        # Only compute correlations when we have at least 2 observations
        if mu_obs.size < 2 or y_arr.size < 2:
            return out

        # GP predictions are in log1p-space by default; compare ranks against
        # raw y_arr (rank correlations are invariant), but compute Pearson on
        # the same transformed scale to avoid a spurious non-linearity.
        y_arr_gp = _np.log1p(_np.clip(y_arr, a_min=0.0, a_max=None))

        spearman_corr = float(spearmanr(mu_obs, y_arr).correlation)
        kendall_corr = float(kendalltau(mu_obs, y_arr).correlation)
        try:
            pearson_corr = float(pearsonr(mu_obs, y_arr_gp)[0])
        except Exception:
            pearson_corr = nan

        out["gp_rank_spearman"] = spearman_corr
        out["gp_rank_kendall"] = kendall_corr
        out["gp_rank_pearson"] = pearson_corr

        # Optionally persist arrays for later plotting (scatter/residual)
        if save_debug_arrays and debug_dir is not None and iteration is not None:
            try:
                p = _Path(debug_dir)
                p.mkdir(parents=True, exist_ok=True)
                _np.save(p / f"gp_pred_iter_{int(iteration)}.npy", mu_obs)
                _np.save(p / f"gp_true_iter_{int(iteration)}.npy", y_arr)
            except Exception:
                pass

    except Exception:
        logger.exception("GP ranking diagnostics failed at iteration %s", iteration)

    return out

def plot_objective_distribution(
    y_true,
    out_path: str | Path,
    *,
    bins: int = 50,
    title_suffix: str = "",
):
    """Side-by-side histogram of y_true and log1p(y_true).

    Used as a quick visual check that the GP transformation actually
    compresses the heavy tail. Saved as a PNG to ``out_path``.
    """
    try:
        import matplotlib.pyplot as plt
        from pathlib import Path as _Path

        y_arr = np.asarray(list(y_true), dtype=float)
        if y_arr.size == 0:
            return
        y_arr = y_arr[np.isfinite(y_arr)]
        if y_arr.size == 0:
            return
        y_log = np.log1p(np.clip(y_arr, a_min=0.0, a_max=None))

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(y_arr, bins=bins, color="steelblue")
        axes[0].set_title(f"y_true {title_suffix}".strip())
        axes[0].set_xlabel("y_true")
        axes[0].set_ylabel("count")
        axes[0].grid(True, alpha=0.3)

        axes[1].hist(y_log, bins=bins, color="seagreen")
        axes[1].set_title(f"log1p(y_true) {title_suffix}".strip())
        axes[1].set_xlabel("log1p(y_true)")
        axes[1].set_ylabel("count")
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        out_p = _Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p)
        plt.close(fig)
    except Exception:
        logger.exception("Failed to plot objective distribution to %s", out_path)

def generate_debug_plots(trace_df: pd.DataFrame, out_dir: str | Path):
    """
    Generate and save debugging plots for EI sanity diagnostics.

    Produces:
    - EI optimizer vs EI random best per iteration
    - EI ratio per iteration (with reference line y=1)
    - GP uncertainty diagnostics (sigma mean, sigma max)
    - Optional histograms (if raw EI arrays are saved in out_dir)
    """
    try:
        import matplotlib.pyplot as plt
        import os
        from pathlib import Path as _Path

        p = _Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)

        # Filter only BO method rows (they contain the diagnostics)
        df = trace_df.copy()
        if df.empty:
            return

        x = df["evaluation"].to_numpy()

        # Plot 1: EI optimizer vs EI_random_best
        plt.figure()
        plt.plot(x, df["ei_optimizer"].to_numpy(), label="EI_optimizer")
        plt.plot(x, df["ei_random_best"].to_numpy(), label="EI_random_best")
        plt.xlabel("iteration")
        plt.ylabel("EI")
        plt.title("EI: optimizer vs random max")
        plt.legend()
        plt.grid(True)
        plt.savefig(p / "ei_optimizer_vs_random_best.png")
        plt.close()

        # Plot 2: EI ratio
        plt.figure()
        plt.plot(x, df["ei_ratio"].to_numpy(), label="EI_ratio")
        plt.axhline(1.0, color="k", linestyle="--", label="reference=1")
        plt.xlabel("iteration")
        plt.ylabel("EI_ratio")
        plt.title("EI optimizer / EI random max")
        plt.legend()
        plt.grid(True)
        plt.savefig(p / "ei_ratio.png")
        plt.close()

        # Plot 3: GP uncertainty diagnostics
        plt.figure()
        plt.plot(x, df["sigma_random_mean"].to_numpy(), label="sigma_random_mean")
        plt.plot(x, df["sigma_random_max"].to_numpy(), label="sigma_random_max")
        plt.xlabel("iteration")
        plt.ylabel("sigma")
        plt.title("GP uncertainty diagnostics")
        plt.legend()
        plt.grid(True)
        plt.savefig(p / "sigma_diagnostics.png")
        plt.close()

        # Plot 5: GP ranking correlations over iterations (Spearman/Kendall/Pearson)
        try:
            plt.figure()
            if "gp_rank_spearman" in df.columns:
                plt.plot(x, df["gp_rank_spearman"].to_numpy(), label="spearman")
            if "gp_rank_kendall" in df.columns:
                plt.plot(x, df["gp_rank_kendall"].to_numpy(), label="kendall")
            if "gp_rank_pearson" in df.columns:
                plt.plot(x, df["gp_rank_pearson"].to_numpy(), label="pearson")
            plt.xlabel("iteration")
            plt.ylabel("correlation")
            plt.title("GP ranking correlations over iterations")
            plt.legend()
            plt.grid(True)
            plt.savefig(p / "gp_ranking_correlations.png")
            plt.close()
        except Exception:
            pass

        # Plot 6: Predicted vs true objective values for selected saved iterations
        try:
            files_pred = sorted([f for f in os.listdir(p) if f.startswith("gp_pred_iter_") and f.endswith(".npy")])
            files_true = sorted([f for f in os.listdir(p) if f.startswith("gp_true_iter_") and f.endswith(".npy")])
            common = [f.replace("gp_pred_iter_", "").replace(".npy", "") for f in files_pred]
            if common:
                # pick last saved iteration for scatter/residual
                chosen = common[-1]
                pred = np.load(p / f"gp_pred_iter_{chosen}.npy")
                true = np.load(p / f"gp_true_iter_{chosen}.npy")

                # Scatter: predicted vs true
                plt.figure()
                plt.scatter(true, pred, alpha=0.6)
                mn = min(np.min(true), np.min(pred))
                mx = max(np.max(true), np.max(pred))
                plt.plot([mn, mx], [mn, mx], color="k", linestyle="--")
                plt.xlabel("true objective (y_obs)")
                plt.ylabel("GP prediction (mu_obs)")
                plt.title(f"Predicted vs True (iter {chosen})")
                plt.grid(True)
                plt.savefig(p / f"gp_pred_vs_true_iter_{chosen}.png")
                plt.close()

                # Residual plot
                residual = pred - true
                plt.figure()
                plt.scatter(true, residual, alpha=0.6)
                plt.axhline(0.0, color="k", linestyle="--")
                plt.xlabel("true objective (y_obs)")
                plt.ylabel("residual = mu_obs - y_obs")
                plt.title(f"Residuals (iter {chosen})")
                plt.grid(True)
                plt.savefig(p / f"gp_residuals_iter_{chosen}.png")
                plt.close()
        except Exception:
            # non-critical
            pass

        # Plot 4: Histograms for selected iterations if raw EI arrays available
        try:
            files = sorted([f for f in os.listdir(p) if f.startswith("ei_random_iter_") and f.endswith(".npy")])
            if files:
                # choose up to three files: first, middle, last
                chosen = [files[0]]
                if len(files) > 2:
                    chosen.append(files[len(files) // 2])
                if len(files) > 1:
                    chosen.append(files[-1])

                for fname in chosen:
                    arr = np.load(p / fname)
                    plt.figure()
                    plt.hist(arr, bins=50)
                    plt.xlabel("EI")
                    plt.ylabel("count")
                    plt.title(f"Histogram of EI_random ({fname})")
                    plt.grid(True)
                    plt.savefig(p / (fname.replace('.npy', '.png')))
                    plt.close()
        except Exception:
            # non-critical; continue
            pass

    except Exception:
        logger.exception("Failed to generate debug plots in %s", out_dir)

def build_default_kernels() -> list[tuple[str, Any]]:
    return [
        ("noisy_hierarchical_tree_kernel", noisy_hierarchical_tree_kernel),
        ("noisy_hierarchical_wl_kernel", noisy_hierarchical_wl_kernel),
        ("noisy_hierarchical_damg_kernel", noisy_hierarchical_damg_kernel),
        ("noisy_combined_hierarchical_kernel", noisy_combined_hierarchical_kernel),
        ("noisy_hierarchical_damg_kernel_13", noisy_hierarchical_damg_kernel_13),
        ("tree_kernel_1", tree_kernel_1),
        ("tree_kernel_2", tree_kernel_2),
        ("tree_kernel_3", tree_kernel_3),
        ("wl_kernel_1", wl_kernel_1),
        ("wl_kernel_2", wl_kernel_2),
        ("wl_kernel_3", wl_kernel_3),
        ("damg_kernel_1", damg_kernel_1),
        ("damg_kernel_2", damg_kernel_2),
        ("damg_kernel_3", damg_kernel_3),
    ]

def _resolve_kernel_from_name(kernel_name: str):
    """Resolve a kernel-name string (CLI/config form) to the matching kernel object."""
    kernels = dict(build_default_kernels())
    if kernel_name in ("wl", "tree", "noisy_hierarchical_wl_kernel"):
        return noisy_hierarchical_wl_kernel
    if kernel_name in ("damg", "noisy_hierarchical_damg_kernel"):
        return noisy_hierarchical_damg_kernel
    if kernel_name in ("combined", "noisy_combined_hierarchical_kernel"):
        return noisy_combined_hierarchical_kernel
    if kernel_name in kernels.keys():
        return kernels[kernel_name]
    raise ValueError(f"Unknown kernel_name: {kernel_name}")


# =============================================================================
# CORE FUNCTIONS
# =============================================================================


def compute_total_objective_evaluations(eval_budget: int, n_pre_samples: int) -> int:
    """Return the total number of objective evaluations in one BO run."""
    return eval_budget + n_pre_samples

def run_single_optimization(
    method_name: str,
    optimizer,
    start_x,
    start_y,
    candidate_pool,
    candidate_values,
    eval_budget: int,
    seed: int,
    *,
    f_obj: Callable[[Tree, int], float] | None = None,
    distance_kernel=None,
    experiment_id: str | None = None,
    target_name: str | None = None,
    kernel_name: str | None = None,
    greater_is_better: bool = False,
    show_progress: bool = False,
    # Sanity-check configuration: number of random candidates to sample and
    # optional debug plotting/array persistence destination.
    sanity_n_sanity: int = 1000,
    save_debug_plots: bool = False,
    debug_plots_dir: str | Path | None = None,
):
    """
    Execute a single optimization run with shared instrumentation.

    Parameters
    ----------
    method_name : str
        Method label written to the CSV ("random", "bo").
    optimizer : object
        Ask-tell optimizer with ``suggest()``/``observe()``; ``_model`` attribute
        is used (when present) to compute ranking metrics against the candidate pool.
    start_x, start_y : list
        Initial (presample) programs and their objective values. Not written to
        the trace but used to initialize the optimizer and the best-so-far value.
    candidate_pool, candidate_values : list
        Fixed reference set + precomputed objective values for ranking metrics
        (Kendall τ, Spearman ρ) only. NEVER used as a substitute for ``f_obj``.
    eval_budget : int
        Number of optimization iterations after the presamples.
    seed : int
        Random seed (recorded in the trace; not consumed by this function directly).
    f_obj : callable
        REAL objective function applied to every suggestion. Required.
    distance_kernel : callable | None
        Kernel used to compute distance_to_best / distance_to_topk post-loop.
        When None, the columns are filled with NaN (still present for schema
        consistency with downstream plotting).
    experiment_id, target_name, kernel_name : str
        Metadata fields written into each row of trace_df / ranking_df.
    greater_is_better : bool
        Maximisation flag. Default False (minimisation).

    Returns
    -------
    trace_df, ranking_df : pandas.DataFrame
        ``trace_df`` always has ``TRACE_COLUMNS`` (incl. distance columns).
        ``ranking_df`` has ``RANKING_COLUMNS``; values are ``NaN`` whenever no
        surrogate is available (e.g. random baseline).
    """

    if f_obj is None:
        raise ValueError("f_obj is required: every suggestion must be evaluated by the true objective.")

    # Best-so-far tracker (initialised from presamples, NOT from len(str(...))!).
    if len(start_y) == 0:
        best_y: float | None = None
        best_x = None
    else:
        ys = np.asarray(start_y, dtype=float)
        best_y = float(np.max(ys) if greater_is_better else np.min(ys))
        xs = np.asarray(start_x, dtype=Tree)

        if greater_is_better:
            best_x = xs[int(np.argmax(ys))]
        else:
            best_x = xs[int(np.argmin(ys))]

    # If the optimizer exposes the Ask/Tell initialization API and has not been
    # initialized yet, hand over the presamples here.
    if hasattr(optimizer, "initialize"):
        already_initialized = False
        snapshot_fn = getattr(optimizer, "get_state_snapshot", None)
        if callable(snapshot_fn):
            try:
                already_initialized = snapshot_fn().get("state") != "UNINITIALIZED"
            except Exception:
                already_initialized = False
        elif getattr(optimizer, "_bo_state", None) is not None:
            already_initialized = getattr(optimizer, "_bo_state").name != "UNINITIALIZED"
        if not already_initialized:
            optimizer.initialize(obj_fun=None, x0=list(start_x), y0=list(start_y), n_pre_samples=len(start_x))

    trace_rows: list[dict] = []
    ranking_rows: list[dict] = []
    suggested_trees: list = []
    suggested_values: list[float] = []

    cumulative_time = 0.0

    # Precompute true_arr once; reused per iteration when a surrogate is present.
    has_pool = candidate_pool is not None and candidate_values is not None and len(candidate_pool) > 1
    true_arr = np.asarray(candidate_values, dtype=float) if has_pool else None
    X_pool = np.asarray(candidate_pool, dtype=object) if has_pool else None

    desc = f"{method_name} {experiment_id or ''}".strip()
    iterations = _progress_iterable(
        range(int(eval_budget)),
        total=int(eval_budget),
        desc=desc,
        enabled=show_progress,
    )
    for i in iterations:
        t0 = time.perf_counter()
        suggestion = optimizer.suggest()  # (verbose=True)
        t1 = time.perf_counter()
        candidate = suggestion.candidate

        # --- EI sanity check (diagnostic only; must NOT call f_obj) ---
        sanity_metrics = {
            "ei_optimizer": float("nan"),
            "ei_random_best": float("nan"),
            "ei_ratio": float("nan"),
            "ei_random_mean": float("nan"),
            "sigma_random_mean": float("nan"),
            "sigma_random_max": float("nan"),
        }
        if save_debug_plots: # only call sanity check, if debug is true.
            try:
                sanity_metrics = compute_ei_sanity_check(
                    optimizer,
                    suggestion,
                    iteration=i + 1,
                    n_sanity=sanity_n_sanity,
                    debug_dir=(Path(debug_plots_dir) if debug_plots_dir is not None else None),
                    save_debug_arrays=save_debug_plots,
                )
            except Exception:
                logger.exception("EI sanity check failed at iteration %d", i + 1)

        eval_start = time.perf_counter()
        y_next = float(f_obj(candidate, seed))
        eval_end = time.perf_counter()

        obs_start = time.perf_counter()
        optimizer.observe(candidate, y_next)
        obs_end = time.perf_counter()

        acquisition_time = t1 - t0
        objective_eval_time = eval_end - eval_start
        update_time = obs_end - obs_start
        bo_overhead_time = acquisition_time + update_time

        cumulative_time += acquisition_time + objective_eval_time + update_time

        # update best
        if best_y is None or (not greater_is_better and y_next < best_y) or (greater_is_better and y_next > best_y):
            best_y = float(y_next)
            best_x = candidate

        suggested_trees.append(candidate)
        suggested_values.append(y_next)

        # GP ranking diagnostics: compute correlations between GP predictions and
        # the observed objective values (only uses already observed data; does
        # not evaluate f_obj on new points).
        try:
            snap_fn = getattr(optimizer, "get_state_snapshot", None)
            if snap_fn is not None:
                snap = snap_fn()
                X_obs = snap.get("x_list")
                y_obs = snap.get("y_list")
            else:
                # Fallback to internal attributes when available
                X_obs = getattr(optimizer, "_x_list", None)
                y_obs = getattr(optimizer, "_y_list", None)

            model = getattr(optimizer, "_model", None)
            gp_rank_metrics = compute_gp_ranking_diagnostics(
                model,
                X_obs,
                y_obs,
                iteration=i + 1,
                debug_dir=(Path(debug_plots_dir) if debug_plots_dir is not None else None),
                save_debug_arrays=save_debug_plots,
            )
        except Exception:
            logger.exception("GP ranking diagnostics failed at iteration %d", i + 1)
            gp_rank_metrics = {"gp_rank_spearman": float("nan"), "gp_rank_kendall": float("nan"), "gp_rank_pearson": float("nan")}

        trace_rows.append(
            {
                "experiment_id": experiment_id,
                "method": method_name,
                "target_name": target_name,
                "kernel_name": kernel_name,
                "seed": seed,
                "evaluation": i + 1,
                "candidate": candidate.interpret(pretty_term_algebra()),
                "best_candidate": best_x.interpret(pretty_term_algebra()) if best_x is not None else None,
                "objective_value": float(y_next),
                "best_objective_value": float(best_y),
                "bo_overhead_time": float(bo_overhead_time),
                "objective_eval_time": float(objective_eval_time),
                "cumulative_wallclock_time": float(cumulative_time),
                # EI sanity-check diagnostics (computed before objective evaluation)
                "ei_optimizer": float(sanity_metrics.get("ei_optimizer", float("nan"))),
                "ei_random_best": float(sanity_metrics.get("ei_random_best", float("nan"))),
                "ei_ratio": float(sanity_metrics.get("ei_ratio", float("nan"))),
                "ei_random_mean": float(sanity_metrics.get("ei_random_mean", float("nan"))),
                "sigma_random_mean": float(sanity_metrics.get("sigma_random_mean", float("nan"))),
                "sigma_random_max": float(sanity_metrics.get("sigma_random_max", float("nan"))),
                # GP ranking diagnostics
                "gp_rank_spearman": float(gp_rank_metrics.get("gp_rank_spearman", float("nan"))),
                "gp_rank_kendall": float(gp_rank_metrics.get("gp_rank_kendall", float("nan"))),
                "gp_rank_pearson": float(gp_rank_metrics.get("gp_rank_pearson", float("nan"))),
                # distance columns filled by _attach_distance_columns
            }
        )

        # Ranking metrics: only meaningful when a surrogate is available.
        kendall_val = float("nan")
        spearman_val = float("nan")
        model = getattr(optimizer, "_model", None)
        if has_pool and model is not None and hasattr(model, "predict"):
            try:
                preds = model.predict(X_pool)
                metrics = compute_ranking_metrics(preds, true_arr)
                kendall_val = metrics["kendall_tau"]
                spearman_val = metrics["spearman_rho"]
            except Exception:
                logger.exception("Ranking metrics computation failed at iteration %d", i + 1)

        ranking_rows.append({
            "experiment_id": experiment_id,
            "method": method_name,
            "target_name": target_name,
            "kernel_name": kernel_name,
            "seed": seed,
            "evaluation": i + 1,
            "kendall_tau": kendall_val,
            "spearman_rho": spearman_val,
        })

    trace_df = pd.DataFrame(trace_rows, columns=TRACE_COLUMNS)
    ranking_df = pd.DataFrame(ranking_rows, columns=RANKING_COLUMNS)

    trace_df = _attach_distance_columns(trace_df, suggested_trees, suggested_values, distance_kernel)

    return trace_df, ranking_df


def run_experiment(
    target,
    kernel_name: str,
    seeds: list[int],
    *,
    eval_budget: int = EVAL_BUDGET,
    initial_sample_size: int = INITIAL_SAMPLE_SIZE,
    ranking_pool_size: int = RANKING_POOL_SIZE,
    results_root: str | Path = "results",
    kernel_optimizer: str = "fmin_l_bfgs_b",
    n_restarts_kernel_optimizer: int = 20,
    optimizer_population_size: int = POPULATION_SIZE,
    optimizer_mutation_rate: float = MUTATION_RATE,
    optimizer_recombination_rate: float = RECOMBINATION_RATE,
    max_depth: int = MAX_TREE_DEPTH,
    distance_kernel_name: str | None = None,
    show_progress: bool = False,
    # EI sanity-check configuration (number of random candidates sampled per iteration)
    sanity_n_sanity: int = 1000,
    # Whether to save debug plots/arrays for each BO seed run (default False)
    save_debug_plots: bool = False,
    # Optional directory to save debug plots; when None plots are written next to CSVs
    debug_plots_dir: str | Path | None = None,
):
    """
    Run experiment for one target and kernel.

    Steps
    -----

    1 generate search space
    2 load presamples
    3 generate ranking pool
    4 run all optimizers
    5 concatenate results

    Returns
    -------
    trace_df
    ranking_df
    export_paths
        Dict with per-experiment CSV artifact paths.
    """

    total_evaluations = compute_total_objective_evaluations(eval_budget, initial_sample_size)

    print(f"[run_experiment] EVAL_BUDGET={eval_budget}, INITIAL_SAMPLE_SIZE={initial_sample_size}")
    print(f"[run_experiment] total evaluations per run = {total_evaluations}")

    all_trace_dfs = []
    all_ranking_dfs = []
    export_paths: dict[str, dict[str, str]] = {}
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    results_root_path = Path(results_root).resolve()
    results_root_path.mkdir(parents=True, exist_ok=True)

    def write_results_csv(df: pd.DataFrame, target_obj, kernel_label: str, seed_value: int, filename: str):
        """Best-effort CSV write with timestamped experiment folder to avoid overwrites."""
        experiment_key = f"{target_to_name(target_obj)}_{kernel_label}_{seed_value}"
        try:
            exp_folder = results_root_path / f"{experiment_key}__{run_timestamp}"
            exp_folder.mkdir(parents=True, exist_ok=True)
            out_path = exp_folder / filename
            df.to_csv(out_path, index=False)
            if experiment_key not in export_paths:
                export_paths[experiment_key] = {}
            export_paths[experiment_key][filename] = str(out_path)
            return str(out_path)
        except Exception:
            return None

    kernel = _resolve_kernel_from_name(kernel_name)
    distance_kernel = _resolve_distance_kernel(distance_kernel_name)

    # Epochs used for search space construction (kept consistent with examples)
    epochs = 2000

    for seed in seeds:
        print(f"[run_experiment] seed={seed}")
        target_name = target_to_name(target)
        experiment_id = f"{target_name}_{kernel_name}_{seed}"

        # 1) construct search space
        search_space, construction_time = generate_search_space(
            target, FEATURE_DIMENSIONS, CONSTANT_VALUES, LEARNING_RATES, epochs
        )

        # 2) load presamples (initial training data for BO)
        presamples_x, _ = get_or_create_pre_samples(search_space, target, size=initial_sample_size, cache_tag="initial")
        start_x = list(presamples_x)
        start_y = [float(f_obj(t, seed)) for t in start_x]

        # 3) build ranking pool (used only for ranking metrics / distance reference)
        ranking_pool_x, _ = get_or_create_pre_samples(search_space, target, size=ranking_pool_size, cache_tag="ranking")
        candidate_pool = list(ranking_pool_x)
        candidate_values = [float(f_obj(t, seed)) for t in candidate_pool]

        # 4) prepare evolutionary optimizer components for acquisition optimization
        termination = lambda state: state.generation >= NUMBER_OF_GENERATIONS
        initialization = RandomLimitedDepthFirstInitialization(search_space, target, max_depth=max_depth)
        mutation = ResolutionMutation(search_space, target, max_depth=max_depth)
        recombination = Crossover(search_space, target, max_depth=max_depth)
        parent_selection = FitnessProportionalSelection()
        survivor_selection = AgeBasedReplacement()
        fitness_comparator = ScalarFitnessComparator(True)  # acquisition is maximized

        evo_alg = SimpleGeneticProgramming(
            search_space,
            target,
            termination,
            initialization,
            mutation,
            recombination,
            parent_selection,
            survivor_selection,
            fitness_comparator,
            rng=random.Random(seed)
        )

        common_run_kwargs = dict(
            start_x=start_x,
            start_y=start_y,
            candidate_pool=candidate_pool,
            candidate_values=candidate_values,
            eval_budget=eval_budget,
            seed=seed,
            f_obj=f_obj,
            distance_kernel=distance_kernel,
            experiment_id=experiment_id,
            target_name=target_name,
            kernel_name=kernel_name,
            show_progress=show_progress,
            sanity_n_sanity=sanity_n_sanity,
            save_debug_plots=save_debug_plots,
            debug_plots_dir=debug_plots_dir,
        )

        # 5) Random baseline
        random_optimizer = RandomBaselineOptimizer(search_space, target, rng=random.Random(seed), max_depth=max_depth)
        trace_random, ranking_random = run_single_optimization(
            method_name="random",
            optimizer=random_optimizer,
            **common_run_kwargs,
        )
        all_trace_dfs.append(trace_random)
        write_results_csv(trace_random, target, kernel_name, seed, "random_trace.csv")
        if not ranking_random.empty:
            all_ranking_dfs.append(ranking_random)
            write_results_csv(ranking_random, target, kernel_name, seed, "random_ranking.csv")

        # 6) Bayesian Optimization
        bo = BayesianOptimization(
            search_space,
            target,
            kernel=kernel,
            kernel_optimizer=kernel_optimizer,
            n_restarts_kernel_optimizer=n_restarts_kernel_optimizer,
            optimizer=evo_alg,
            optimizer_population_size=optimizer_population_size,
            optimizer_mutation_rate=optimizer_mutation_rate,
            optimizer_recombination_rate=optimizer_recombination_rate,
            seed=seed,
            max_depth=max_depth,
        )

        trace_bo, ranking_bo = run_single_optimization(
            method_name="bo",
            optimizer=bo,
            **common_run_kwargs,
        )

        all_trace_dfs.append(trace_bo)
        all_ranking_dfs.append(ranking_bo)
        write_results_csv(trace_bo, target, kernel_name, seed, "bo_trace.csv")
        write_results_csv(ranking_bo, target, kernel_name, seed, "bo_ranking.csv")
        # Optionally generate debug plots (saved next to CSVs by default)
        if save_debug_plots:
            try:
                # reconstruct the per-experiment folder used by write_results_csv
                exp_folder = results_root_path / f"{target_name}_{kernel_name}_{seed}__{run_timestamp}"
                debug_dir = Path(debug_plots_dir) if debug_plots_dir is not None else (exp_folder / "debug_plots")
                generate_debug_plots(trace_bo, debug_dir)
                plot_objective_distribution(
                    trace_bo["objective_value"].to_numpy(),
                    debug_dir / "objective_distribution.png",
                    title_suffix=f"({experiment_id})",
                )
            except Exception:
                logger.exception("Failed to generate debug plots for %s", experiment_id)

    trace_df = pd.concat(all_trace_dfs, ignore_index=True) if all_trace_dfs else pd.DataFrame(columns=TRACE_COLUMNS)
    ranking_df = pd.concat(all_ranking_dfs, ignore_index=True) if all_ranking_dfs else pd.DataFrame(columns=RANKING_COLUMNS)

    return trace_df, ranking_df, export_paths
