"""Utilities for pass-rate/cost curves and arbitrage analysis.

This module uses NumPy for scalar/bootstrap statistics and JAX for batched curve
composition and interpolation.
"""

from __future__ import annotations

from tqdm import tqdm
from typing import Any, Mapping, Sequence, Callable

import jax
import jax.numpy as jnp
import numpy as np


def pass_at_k(num_samples: int, num_correct: int, k: float) -> float:
    """Estimate pass@k from `num_samples` draws with `num_correct` successes.

    Uses the standard unbiased estimator:
    `1 - C(n-c, k) / C(n, k)`.
    """
    # Budgets map to a real-valued "attempt count"; estimator expects integer k.
    k_int = int(k)
    if k_int < 1 or num_samples <= 0:
        return 0.0
    if k_int >= num_samples or num_samples - num_correct < k_int:
        return 1.0 if num_correct > 0 else 0.0

    # Product form of C(n-c, k) / C(n, k): probability all k draws are failures.
    failure_terms = 1.0 - k_int / np.arange(
        num_samples - num_correct + 1,
        num_samples + 1,
    )
    return float(1.0 - np.prod(failure_terms))


def pass_at_budget(budgets, results_row, cost_per_problem_override=None):
    """
    Compute pass probability vs budget for a single problem.
    
    Args:
        budgets: Budget grid with shape `(B,)`.
        results_row: A dictionary containing 'attempts' (a list of booleans indicating
                     whether each attempt was correct) and 'mean_cost' (the average cost per attempt).
        cost_per_problem_override: If provided, use this fixed cost for the problem.

    Returns:
        Array with shape `(B,)`, representing the pass probability for each budget.
    """
    attempts = results_row['attempts']
    if cost_per_problem_override:
        sample_cost = cost_per_problem_override
    else:
        sample_cost = results_row['mean_cost']

    # Convert dollar budget to number of attempts this model can afford.
    attempts_per_budget = budgets / sample_cost
    num_attempts = len(attempts)
    num_correct = int(sum(attempts))
    pass_curve = np.array(
        [
            pass_at_k(num_attempts, num_correct, k)
            for k in attempts_per_budget
        ],
        dtype=float,
    )
    return pass_curve

def load_pass_curves(
    results: Mapping[Any, Sequence[bool]],
    model_names: Sequence[str],
    budgets: np.ndarray,
    filter_problems: Callable[[Any], bool] = None,
    cost_per_problem_override: float = None,
) -> np.ndarray:
    """Load provider pass curves for a given set of models and budgets.
    
    Returns an array of shape (len(problems), len(model_names), len(budgets))
    """
    results_dict = {m: {row['id']: row for row in results[m]} for m in model_names}

    # Find intersection of all problems across models
    all_problems = [set(results_dict[m].keys()) for m in model_names]
    problems = list(set.intersection(*all_problems))
    if filter_problems is not None:
        problems = [p for p in problems if filter_problems(p)]

    pass_curves = np.empty((len(problems), len(model_names), len(budgets)), dtype=float)
    for i, m in enumerate(model_names):
        for j, pid in enumerate(problems):
            pass_curves[j, i] = pass_at_budget(
                budgets,
                results_dict[m][pid],
                cost_per_problem_override,
            )

    return pass_curves

def integrate_log(
    x: jnp.ndarray,
    y: jnp.ndarray,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """Cumulative integral of `y(x)` assuming linearity in `log(x)` per segment.

    The function treats adjacent samples as piecewise-linear in
    `t = log(x)`, integrates each segment in closed form, and returns the
    cumulative sum of segment integrals.
    """
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    # Guard log-space math against zeros on the budget axis.
    x = jnp.clip(x, a_min=eps)

    t = jnp.log(x)
    t0, t1 = t[:-1], t[1:]
    y0, y1 = y[:-1], y[1:]
    x0, x1 = x[:-1], x[1:]

    dt = t1 - t0
    # Linear model in log-space: y(t) = slope * t + intercept.
    slope = jnp.where(dt != 0, (y1 - y0) / dt, 0.0)
    intercept = y0 - slope * t0

    # Closed-form integral of (slope * log(x) + intercept) dx per segment.
    segment_area = (
        slope * ((t1 - 1.0) * x1 - (t0 - 1.0) * x0)
        + intercept * (x1 - x0)
    )
    return jnp.cumsum(segment_area)


def interp_logx(
    x: jnp.ndarray,
    y: jnp.ndarray,
    x_new: jnp.ndarray,
    left: float = 0.0,
) -> jnp.ndarray:
    """Interpolate `log(x) -> y` and evaluate at `x_new`."""
    return jnp.interp(jnp.log(x_new), jnp.log(x), y, left=left)


def interp_logy(
    x: jnp.ndarray,
    y: jnp.ndarray,
    x_new: jnp.ndarray,
    left: float = 0.0,
    right: float | None = None,
) -> jnp.ndarray:
    """Interpolate `x -> log(y)` and evaluate on the original y-scale."""
    return jnp.exp(jnp.interp(x_new, x, jnp.log(y), left=left, right=right))


def cascade_problem(
    budget_grid: jnp.ndarray,
    pass_curves: jnp.ndarray,
    thresholds: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compose model pass curves under sequential threshold gating.

    Args:
        budget_grid: Cost grid `(B,)`.
        pass_curves: Per-model pass curves `(M, B)`.
        thresholds: Per-model threshold budget `(M,)`.

    Returns:
        Tuple:
            - pass_curve: Array `(B,)`, cumulative pass probability at each budget.
            - expected_spend: Array `(M, B)`, expected spend allocated to each
              model at each budget.
    """
    log_budget = jnp.log(budget_grid)

    def scan_step(carry, model_inputs):
        spent_budget, combined_pass_so_far = carry
        model_pass_curve, model_threshold = model_inputs

        # Pass probability achieved by this model before escalating onward.
        pass_at_threshold = jnp.interp(
            jnp.log(model_threshold),
            log_budget,
            model_pass_curve,
        )
        # Before threshold, this model contributes incremental pass mass;
        # after threshold, it is capped and remaining mass goes to later models.
        updated_pass_curve = (
            jnp.minimum(model_pass_curve, pass_at_threshold)
            * (1.0 - combined_pass_so_far)
            + combined_pass_so_far
        )
        new_combined_pass = (
            pass_at_threshold * (1.0 - combined_pass_so_far)
            + combined_pass_so_far
        )
        updated_budget_axis = jnp.minimum(budget_grid, model_threshold) + spent_budget

        # Expected spend this model receives before handoff. For each global
        # budget, map remaining budget to local model budget and weight by the
        # probability mass that reaches this model.
        model_expected_spend_curve = cost_from_pass(budget_grid, model_pass_curve)
        local_budget = jnp.minimum(
            jnp.maximum(budget_grid - spent_budget, 0.0),
            model_threshold,
        )
        local_budget_positive = local_budget > 0.0
        local_spend = jnp.where(
            local_budget_positive,
            jnp.interp(
                jnp.log(jnp.maximum(local_budget, 1e-12)),
                log_budget,
                model_expected_spend_curve,
                left=0.0,
            ),
            0.0,
        )
        model_spend_curve = (1.0 - combined_pass_so_far) * local_spend

        new_spent_budget = spent_budget + model_threshold
        return (new_spent_budget, new_combined_pass), (
            updated_budget_axis,
            updated_pass_curve,
            model_spend_curve,
        )

    _, (stacked_budget, stacked_pass, stacked_spend) = jax.lax.scan(
        scan_step,
        init=(0.0, 0.0),
        xs=(pass_curves, thresholds),
    )

    flat_budget = jnp.reshape(stacked_budget, (-1,))
    flat_pass = jnp.reshape(stacked_pass, (-1,))

    # Stitch piecewise segments from each cascade stage back onto base grid.
    pass_curve = jnp.interp(jnp.log(budget_grid), jnp.log(flat_budget), flat_pass)
    return pass_curve, stacked_spend


cascade_batch = jax.vmap(
    cascade_problem,
    in_axes=(None, 0, None),
)


def cascade(
    budget_grid: jnp.ndarray,
    pass_curves: jnp.ndarray,
    thresholds: jnp.ndarray,
) -> jnp.ndarray:
    """Compose model pass curves under sequential threshold gating."""
    pass_curves_batch, expenditure_curves = cascade_batch(budget_grid, pass_curves, thresholds)
    return pass_curves_batch.mean(axis=0), expenditure_curves.mean(axis=0)


def get_arbitrage_prices(
    budget_grid: jnp.ndarray,
    pass_curves: jnp.ndarray,
    thresholds: jnp.ndarray,
    performance_grid: jnp.ndarray,
) -> jnp.ndarray:
    arbitrage_perf, arbitrage_expend = cascade(budget_grid, pass_curves, thresholds)
    arbitrage_cost = cost_for_pass(budget_grid, arbitrage_perf, performance_grid)

    # interpolate expenditure along the grid
    arbitrage_expend = jax.vmap(interp_logy, (None, 0, None))(
        arbitrage_perf,
        arbitrage_expend,
        performance_grid,
    )
    # replace nan with 0
    arbitrage_expend = jnp.where(jnp.isnan(arbitrage_expend), 0, arbitrage_expend)
    return arbitrage_cost, arbitrage_expend


def cost_from_pass(
    budget_grid: jnp.ndarray,
    pass_curve: jnp.ndarray,
) -> jnp.ndarray:
    """Return expected spend at each budget using survival integration."""
    # E[cost up to b] = integral_0^b P(not solved by cost x) dx.
    failure_curve = 1.0 - pass_curve
    budget_with_zero = jnp.concatenate([jnp.zeros(1), budget_grid])
    failure_with_one = jnp.concatenate([jnp.ones(1), failure_curve])
    return integrate_log(budget_with_zero, failure_with_one)


cost_from_pass_batch = jax.vmap(
    cost_from_pass,
    in_axes=(None, 0),
)


def cost_for_pass(
    budget_grid: jnp.ndarray,
    pass_curve: jnp.ndarray,
    target_pass_grid: jnp.ndarray,
) -> jnp.ndarray:
    """Map target pass levels to expected cost via log-space interpolation."""
    expected_cost_curve = cost_from_pass(budget_grid, pass_curve)
    # Invert pass->cost curve by interpolating in log-cost space for stability.
    return interp_logy(
        pass_curve,
        expected_cost_curve,
        target_pass_grid,
        right=np.inf,
    )


cost_for_pass_batch = jax.vmap(
    cost_for_pass,
    in_axes=(None, 0, None),
)


def arbitrage_profit_curve(
    budget_grid: jnp.ndarray,
    market_pass_curves: jnp.ndarray,
    arbitrage_pass_curve: jnp.ndarray,
    target_pass_grid: jnp.ndarray,
) -> jnp.ndarray:
    """Compute non-negative arbitrage profit across target pass levels.

    Args:
        budget_grid: Budget axis `(B,)`.
        market_pass_curves: Provider curves `(N, B)`.
        arbitrage_pass_curve: Arbitrage strategy curve `(B,)`.
        target_pass_grid: Target pass levels `(G,)`.
    """
    provider_prices = cost_for_pass_batch(
        budget_grid,
        market_pass_curves,
        target_pass_grid,
    )
    # Best available market price at each target pass level.
    market_frontier_price = jnp.min(provider_prices, axis=0)
    arbitrage_price = cost_for_pass(
        budget_grid,
        arbitrage_pass_curve,
        target_pass_grid,
    )
    # Profit cannot be negative (no demand if price is too high...)
    profit = jnp.maximum(market_frontier_price - arbitrage_price, 0.0)
    profit_margin = profit / market_frontier_price
    return profit_margin


def threshold_to_profit(
    budget_grid: jnp.ndarray,
    market_pass_curves: jnp.ndarray,
    threshold_vector: jnp.ndarray,
    target_pass_grid: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate average arbitrage profit for a threshold vector."""
    # Build arbitrage curve by cascading each market curve with same thresholds.
    arbitrage_pass_curve, _ = cascade(
        budget_grid,
        market_pass_curves,
        threshold_vector,
    )
    return arbitrage_profit_curve(
        budget_grid,
        market_pass_curves.mean(axis=0),
        arbitrage_pass_curve,
        target_pass_grid,
    )

threshold_to_profit_batch = jax.vmap(
    threshold_to_profit,
    in_axes=(None, None, 0, None),
)

# # ---------------------------------------------------------------------------
# # Backward-compatible aliases for existing notebooks/scripts.
# # ---------------------------------------------------------------------------
# pass_at_k = pass_at_k
# get_pass_cost = pass_at_budget
# integrate_log = integrate_log
# log_interpolate_jax = interp_logx
# ylog_interpolate_jax = interp_logy
# combine_pass_many_jax = cascade_problem
# combine_many = cascade
# get_avg_cost = cost_from_pass
# get_avg_cost_vmap = cost_from_pass_batch
# get_cost_grid = cost_for_pass
# cost_grid_vmap = cost_for_pass_batch
# threshold_to_profit = threshold_to_profit
# threshold_to_profit_vmap = threshold_to_profit_batch

def sample_from_cdf(key, x, p, max_budget):
    """
    Sample from a CDF given a maximum budget.

    The CDF is p(x), where x is cost.
    The maximum budget changes the CDF such that p'(x) = 0 for x > max_budget.

    Args:
        key: JAX PRNG key.
        x: Array of x values of shape (N,).
        p: Array of p values of shape (N,).
        max_budget: Maximum budget.

    Returns:
        Sample from the CDF.
    """
    key_u, key_accept = jax.random.split(key)

    # find last valid index
    to_keep = x <= max_budget
    last_idx = jnp.max(jnp.where(to_keep, jnp.arange(x.shape[0]), 0))

    # check whether we exhaust max budget
    p_final = p[last_idx]
    accept = jax.random.uniform(key_accept) < p_final

    def sample_fn(_):
        # normalize CDF
        p_norm = p / p_final
        u = jax.random.uniform(key_u)
        logx = jnp.log(x)
        return jnp.exp(jnp.interp(u, p_norm, logx))

    def max_budget_fn(_):
        return jnp.float32(max_budget)

    return jax.lax.cond(accept, sample_fn, max_budget_fn, operand=None)

sample_from_cdf_batch = jax.vmap(sample_from_cdf, in_axes=(None, None, 0, None))
sample_from_cdf_batch = jax.vmap(sample_from_cdf_batch, in_axes=(0, None, None, None))
sample_from_cdf_batch = jax.vmap(sample_from_cdf_batch, in_axes=(None, None, 0, None))

def eval_ood_profitability(
    budget_grid,
    performances_fit,
    performances_eval,
    thresholds,
    fit_grid,
    eval_grid,
):
    # Find the optimal threshold for the fit data
    mean_profits = threshold_to_profit_batch(
        budget_grid,
        performances_fit,
        thresholds,
        fit_grid,
    ).mean(axis=-1)  # mean across the performance grid
    optimal_threshold = thresholds[jnp.argmax(mean_profits)]

    # Evaluate the profitability on the eval data
    test_profit = threshold_to_profit_batch(
        budget_grid,
        performances_eval,
        optimal_threshold[None],
        eval_grid,
    )

    # replace nan or inf with 0
    test_profit = jnp.where(jnp.isnan(test_profit), 0, test_profit)
    test_profit = jnp.where(jnp.isinf(test_profit), 0, test_profit)
    return jnp.mean(test_profit)


def group_by_budget(costs, max_search_budget):
    """Yield contiguous sample groups whose summed cost stays under budget."""
    cost_so_far = 0.0
    group_idxs = []
    for i, cost in enumerate(costs):
        if cost_so_far + cost > max_search_budget:
            if group_idxs:
                yield np.asarray(group_idxs, dtype=np.int32)
            group_idxs = []
            cost_so_far = 0.0

        group_idxs.append(i)
        cost_so_far += float(cost)

    if group_idxs:
        yield np.asarray(group_idxs, dtype=np.int32)


def _coarsen_groups_with_trim(groups, max_trim_fraction=0.05):
    """Cluster nearby group lengths and trim to a shared target length.

    Groups in the same cluster are trimmed to the shortest length in that
    cluster, bounded by `max_trim_fraction` relative loss per group.
    """
    if not groups:
        return []

    groups_by_len = {}
    for pos, idx in enumerate(groups):
        groups_by_len.setdefault(len(idx), []).append((pos, idx))

    pending_lengths = sorted(groups_by_len.keys(), reverse=True)
    clusters = []

    while pending_lengths:
        l_max = pending_lengths[0]
        l_min_allowed = int(np.ceil((1.0 - max_trim_fraction) * l_max))

        in_cluster = [length for length in pending_lengths if length >= l_min_allowed]
        target_len = min(in_cluster)

        cluster_items = []
        for length in in_cluster:
            for pos, idx in groups_by_len[length]:
                cluster_items.append((pos, idx[:target_len]))

        clusters.append((target_len, cluster_items))
        pending_lengths = [length for length in pending_lengths if length not in in_cluster]

    return clusters


def search_budget_experiments(
    search_budgets,
    budget_grid,
    perf_sample,
    perf_eval,
    thresholds,
    fit_grid,
    eval_grid,
    N_samples=100,
    max_budget_search=0.5,
    n_bootstraps=100,
    show_progress=True,
    trim_fraction=0.05,
):
    seed = 0
    keys = jax.random.split(jax.random.PRNGKey(seed), N_samples)

    if isinstance(n_bootstraps, int):
        n_bootstraps = [n_bootstraps] * len(search_budgets)

    budget_grid_jnp = jnp.asarray(budget_grid)
    perf_eval_jnp = jnp.asarray(perf_eval)
    thresholds_jnp = jnp.asarray(thresholds)
    fit_grid_jnp = jnp.asarray(fit_grid)
    eval_grid_jnp = jnp.asarray(eval_grid)

    # Compile and cache the expensive sampling kernel once.
    sample_from_cdf_batch_jit = jax.jit(sample_from_cdf_batch)

    if show_progress:
        print("[get_curve_data] sampling cost trajectories...")

    # sample costs from problems --> models x (samples x problems)
    perf_sample_swapped = jnp.asarray(perf_sample.swapaxes(0, 1))
    cost_samples = sample_from_cdf_batch_jit(
        keys,
        budget_grid_jnp,
        perf_sample_swapped,
        max_budget_search,
    )
    cost_samples = np.asarray(cost_samples).reshape(cost_samples.shape[0], -1)
    # cost_of_samples = cost_samples[0]  # one could also use the sum
    cost_of_samples = cost_samples[1]  # one could also use the sum

    # Precompute all empirical pass curves once: (models, samples, budget_grid).
    perf_lookup = (budget_grid[None, None] >= cost_samples[:, :, None]).astype(np.float32)

    def _eval_single(perf_fit):
        return eval_ood_profitability(
            budget_grid_jnp,
            perf_fit,
            perf_eval_jnp,
            thresholds_jnp,
            jnp.linspace(0.1, perf_fit.mean(axis=0).max(), 100),
            eval_grid_jnp,
        )

    # Compile once per (trimmed) length, while allowing variable batch sizes.
    eval_single_jit = jax.jit(_eval_single)

    budget_results = []
    budget_pairs = list(zip(search_budgets, n_bootstraps))
    budget_iter = budget_pairs

    for budget, n_boot in budget_iter:
        groups = []
        for i, idx in enumerate(group_by_budget(cost_of_samples, budget)):
            if i >= n_boot:
                break
            groups.append(idx)

        if not groups:
            budget_results.append([])
            continue

        # Coarsen near-equal lengths and trim <= trim_fraction to reduce
        # recompilations from shape proliferation.
        clusters = _coarsen_groups_with_trim(groups, max_trim_fraction=trim_fraction)

        results = np.empty(len(groups), dtype=np.float32)

        boot_pbar = None
        if show_progress:
            boot_pbar = tqdm(
                total=len(groups),
                desc=f"budget={budget} bootstraps",
                unit="boot",
                leave=True,
                position=1,
            )

        for _, items in clusters:
            positions = [pos for pos, _ in items]
            idx_batch = np.stack([idx for _, idx in items], axis=0)  # (K, L)

            # perf_batch: (K, L, models, budget_grid)
            perf_batch = perf_lookup[:, idx_batch, :].transpose(1, 2, 0, 3)
            # vmap over batch only; JIT cache is keyed mainly by trimmed length L.
            batch_out = jax.vmap(eval_single_jit, in_axes=0)(jnp.asarray(perf_batch))
            results[np.asarray(positions)] = np.asarray(batch_out)

            if boot_pbar is not None:
                boot_pbar.update(len(items))

        if boot_pbar is not None:
            boot_pbar.close()

        budget_results.append(results.tolist())

    return budget_results
