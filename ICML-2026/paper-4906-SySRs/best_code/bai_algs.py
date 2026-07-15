"""
Best-Arm Identification algorithms for LLM benchmarking.

Supported algorithms:
  - naive_baseline        : smart uniform sampling (same tasks per run, across all arms)
  - uniform_pulls         : uniform sampling (independent tasks per arm per run)
  - ucb_e                 : UCB-E with per-arm independent WOR task sampling
  - smart_ucb_e           : UCB-E with shared WOR task sampling across arms per run
  - successive_rejects_wo_replacement_no_budget_limit  : SR with budget reallocation
  - smart_successive_rejects_wo_replacement_no_budget_limit : Smart SR with budget reallocation

Global variables (must be set before calling any algorithm):
  - model_accuracies    : np.ndarray of shape (n_arms, n_tasks), pre-computed per-task accuracy

total_n_arms and total_n_tasks are derived automatically from model_accuracies.shape.
"""

import numpy as np
import random


# ============================================================================
# Global state (set before calling any algorithm)
# ============================================================================

model_accuracies = None   # shape: (n_arms, n_tasks), pre-computed per-task accuracy


# ============================================================================
# Shared utility functions
# ============================================================================

def get_results(models, tasks, cross=False):
    """Evaluate models on tasks and return rewards.

    Args:
        models: array of model indices
        tasks:  array of task indices or a single task index
        cross:  if True, evaluate each model on all tasks in tasks;
                if False, evaluate model[i] on task[i].

    Returns:
        Array of rewards.
        If cross=True : shape (#models, #tasks)
        If cross=False: shape (#models,)
    """
    if not cross:
        return model_accuracies[models, tasks]
    else:
        return model_accuracies[models][:, tasks]


def random_argmax(a, axis=None):
    """Argmax with uniform random tie-breaking.

    Args:
        a:    np.ndarray
        axis: axis along which to compute argmax

    Returns:
        Index (or array of indices) of a random maximum element.
    """
    a = np.asarray(a)
    if axis is None:
        m = a.max()
        return int(np.random.choice(np.flatnonzero(a.ravel() == m)))
    r = np.random.random(a.shape)
    r[~(a == np.nanmax(a, axis=axis, keepdims=True))] = -1.0
    return np.nanargmax(r, axis=axis)


def acc(results):
    """Fraction of runs that selected arm 0 (the designated best arm)."""
    return (results == 0).mean()


def calculate_budgets_from_percentage(total_tasks, total_models, percentages):
    """Convert percentage values to absolute budget integers.

    Each returned budget is divisible by total_models and at least total_models.

    Args:
        total_tasks:   int
        total_models:  int
        percentages:   list of floats, e.g. [0.01, 0.05, 0.10]

    Returns:
        Sorted list of unique budget integers.
    """
    total_entries = total_tasks * total_models
    budgets = []
    for pct in percentages:
        raw_budget = int(total_entries * pct)
        budget = (raw_budget // total_models) * total_models
        if budget < total_models:
            budget = total_models
        budgets.append(budget)
    return sorted(list(set(budgets)))


def cap_and_adjust_budgets(budgets, total_tasks, total_models):
    """Cap budgets to the maximum safe budget for SR algorithms.

    The maximum safe budget is computed from the SR schedule so that
    no phase requires more tasks than are available.

    Args:
        budgets:       list of ints
        total_tasks:   int
        total_models:  int

    Returns:
        (adjusted_budgets, capped, max_budget)
        - adjusted_budgets: sorted list of unique ints
        - capped:           True if any budget was capped
        - max_budget:       the cap value
    """
    n_arms = total_models
    logbar = 0.5 + (1 / np.arange(2, n_arms + 1)).sum() if n_arms >= 2 else 0.5
    max_theoretical_budget_raw = int((total_tasks - 1) * logbar + n_arms)
    max_budget = (max_theoretical_budget_raw // n_arms) * n_arms

    capped = False
    adjusted_budgets = []
    for budget in budgets:
        if budget > max_budget:
            adjusted_budgets.append(max_budget)
            capped = True
        else:
            adjusted = (budget // n_arms) * n_arms
            adjusted_budgets.append(adjusted)

    return sorted(list(set(adjusted_budgets))), capped, max_budget


# ============================================================================
# Uniform-sampling baselines
# ============================================================================

def naive_baseline(n_items, k=1000, track_means=False):
    """Smart uniform sampling: same tasks used for all arms within each run.

    Args:
        n_items:     total pull budget (must be divisible by n_arms)
        k:           number of independent runs
        track_means: if True, also return empirical means per arm per run

    Returns:
        selected_arms: np.ndarray of shape (k,) — 0-based best arm per run
        (if track_means) tuple (selected_arms, None, empirical_means)
            empirical_means: shape (n_arms, k)
    """
    random.seed(42)
    np.random.seed(42)
    n_arms = model_accuracies.shape[0]
    assert n_items % n_arms == 0
    n_items_per_arm = int(n_items / n_arms)
    assert n_items_per_arm > 0, (
        f"Budget too small: n_items={n_items}, n_arms={n_arms}"
    )
    assert model_accuracies.shape[1] >= n_items_per_arm

    # For each run, sample n_items_per_arm tasks without replacement
    subset = np.array([
        np.random.choice(np.arange(model_accuracies.shape[1]), n_items_per_arm, replace=False)
        for _ in range(k)
    ])  # (k, n_items_per_arm)
    subset = subset.flatten()  # (k * n_items_per_arm)

    outputs = (
        get_results(np.arange(n_arms), subset, cross=True)
        .reshape(n_arms, k, n_items_per_arm)
        .mean(2)
    )  # (n_arms, k)

    selected_arms = random_argmax(outputs, axis=0)

    if track_means:
        return selected_arms, None, outputs
    return selected_arms


def uniform_pulls(n_items, k=1000, track_means=False):
    """Non-smart uniform sampling: each (arm, run) pair draws independent tasks.

    Args:
        n_items:     total pull budget (must be divisible by n_arms)
        k:           number of independent runs
        track_means: if True, also return empirical means per arm per run

    Returns:
        selected_arms: np.ndarray of shape (k,)
        (if track_means) tuple (selected_arms, None, empirical_means)
            empirical_means: shape (n_arms, k)
    """
    random.seed(42)
    np.random.seed(42)
    n_arms = model_accuracies.shape[0]
    assert n_items % n_arms == 0
    n_items_per_arm = int(n_items / n_arms)
    assert n_items_per_arm > 0, (
        f"Budget too small: n_items={n_items}, n_arms={n_arms}"
    )

    # Each (arm, run) pair gets its own n_items_per_arm tasks
    subset = np.array([
        np.random.choice(np.arange(model_accuracies.shape[1]), n_items_per_arm, replace=False)
        for _ in range(n_arms * k)
    ]).flatten()  # (n_arms * k * n_items_per_arm)

    outputs = (
        get_results(np.repeat(np.arange(n_arms), n_items_per_arm * k), subset)
        .reshape(n_arms, k, n_items_per_arm)
        .mean(2)
    )  # (n_arms, k)

    selected_arms = random_argmax(outputs, axis=0)

    if track_means:
        return selected_arms, None, outputs
    return selected_arms


# ============================================================================
# UCB-E variants
# ============================================================================

def ucb_e(n, a=1, k=1000, checkpoint_budgets=None, track_arm_selection=False, track_means=False):
    """UCB-E with per-arm independent WOR task sampling.

    Each arm in each run draws tasks independently without replacement.

    Args:
        n:                  total pull budget
        a:                  exploration parameter
        k:                  number of independent runs
        checkpoint_budgets: if not None, list of intermediate budgets at which to
                            record the current best arm (returned as a dict)
        track_arm_selection: if True, log arm pulled at each iteration
        track_means:        if True, return empirical means per arm per run

    Returns:
        If checkpoint_budgets is None and no tracking:
            np.ndarray of shape (k,) — selected arms
        If checkpoint_budgets is None and tracking enabled:
            (selected_arms, arm_selections, empirical_means, counts)
            arm_selections shape: (n - n_arms, k) or None
            empirical_means shape: (n_arms, k) or None
            counts shape: (n_arms, k)
        If checkpoint_budgets is not None:
            dict mapping budget -> results (same format as above without tracking)
    """
    random.seed(42)
    np.random.seed(42)

    checkpoint_results = {}
    if checkpoint_budgets is not None:
        checkpoint_budgets_sorted = sorted(checkpoint_budgets)
        checkpoint_idx = 0
        print(f"Running UCB-E to budget {n} with checkpoints at: {checkpoint_budgets_sorted}")

    n_arms = model_accuracies.shape[0]
    assert n_arms >= 2 and n >= n_arms

    sums = np.zeros((n_arms, k))
    counts = np.zeros((n_arms, k), dtype=int)

    max_pulls_per_arm = min(n, model_accuracies.shape[1])
    print(f"Maximum pulls per arm: {max_pulls_per_arm} (n={n}, tasks={model_accuracies.shape[1]})")

    # Pre-sample task order for each (arm, run)
    task_indices = np.zeros((n_arms, k, max_pulls_per_arm), dtype=int)
    for arm in range(n_arms):
        for run in range(k):
            task_indices[arm, run, :] = np.random.choice(
                np.arange(model_accuracies.shape[1]), size=max_pulls_per_arm, replace=False
            )

    task_positions = np.zeros((n_arms, k), dtype=int)

    arm_selections = None
    if track_arm_selection:
        arm_selections = np.zeros((n - n_arms, k), dtype=int)

    # Warm-up: one pull per arm
    t = 0
    for arm in range(n_arms):
        indices = task_indices[arm, np.arange(k), task_positions[arm, :]]
        results = get_results(np.full(k, arm), indices, cross=False)
        sums[arm, :] += results
        counts[arm, :] += 1
        task_positions[arm, :] += 1
        t += 1

        if checkpoint_budgets is not None:
            while (checkpoint_idx < len(checkpoint_budgets_sorted)
                   and t >= checkpoint_budgets_sorted[checkpoint_idx]):
                checkpoint_results[checkpoint_budgets_sorted[checkpoint_idx]] = (
                    random_argmax(sums / counts, 0)
                )
                print(f"  UCB-E checkpoint at budget {checkpoint_budgets_sorted[checkpoint_idx]}, t={t}")
                checkpoint_idx += 1

    # Main loop
    for iteration in range(n - n_arms):
        means = sums / counts
        ucb = means + np.sqrt(a / counts)

        available_mask = task_positions < max_pulls_per_arm
        masked_ucb = ucb.copy()
        masked_ucb[~available_mask] = -np.inf
        to_pull = random_argmax(masked_ucb, axis=0)

        if track_arm_selection:
            arm_selections[iteration, :] = to_pull

        current_positions = task_positions[to_pull, np.arange(k)]
        mask = current_positions >= max_pulls_per_arm

        safe_positions = np.minimum(current_positions, max_pulls_per_arm - 1)
        indices = task_indices[to_pull, np.arange(k), safe_positions]
        indices[mask] = 0

        results = get_results(to_pull, indices, cross=False)
        results[mask] = 0.0

        sums[to_pull, np.arange(k)] += results
        counts[to_pull, np.arange(k)] += (~mask).astype(int)
        task_positions[to_pull, np.arange(k)] += (~mask).astype(int)
        t += 1

        if checkpoint_budgets is not None:
            while (checkpoint_idx < len(checkpoint_budgets_sorted)
                   and t >= checkpoint_budgets_sorted[checkpoint_idx]):
                checkpoint_results[checkpoint_budgets_sorted[checkpoint_idx]] = (
                    random_argmax(sums / counts, 0)
                )
                print(f"  UCB-E checkpoint at budget {checkpoint_budgets_sorted[checkpoint_idx]}, t={t}")
                checkpoint_idx += 1

    final_result = random_argmax(sums / counts, 0)
    final_means = sums / counts
    empirical_means = final_means if track_means else None

    if checkpoint_budgets is not None:
        if n not in checkpoint_results:
            checkpoint_results[n] = final_result
            print(f"  UCB-E final checkpoint at budget {n}")
        return checkpoint_results
    else:
        if track_arm_selection or track_means:
            return (final_result, arm_selections, empirical_means, counts)
        return final_result


def smart_ucb_e(n, a=1, k=1000, checkpoint_budgets=None, track_arm_selection=False, track_means=False):
    """UCB-E with shared WOR task sampling across all arms within each run.

    All arms in a given run pull from the same shuffled task sequence, so each
    pull step reveals a new task to exactly one arm.

    Args:
        n:                  total pull budget
        a:                  exploration parameter
        k:                  number of independent runs
        checkpoint_budgets: optional list of intermediate budgets
        track_arm_selection: if True, log arm pulled at each iteration
        track_means:        if True, return empirical means per arm per run

    Returns:
        Same format as ucb_e.
    """
    random.seed(42)
    np.random.seed(42)

    checkpoint_results = {}
    if checkpoint_budgets is not None:
        checkpoint_budgets_sorted = sorted(checkpoint_budgets)
        checkpoint_idx = 0
        print(f"Running Smart-UCB-E to budget {n} with checkpoints at: {checkpoint_budgets_sorted}")

    n_arms = model_accuracies.shape[0]
    assert n_arms >= 2 and n >= n_arms

    sums = np.zeros((n_arms, k))
    counts = np.zeros((n_arms, k), dtype=int)

    # Pre-sample a single shared task sequence per run
    max_tasks_needed = n - n_arms + 1
    if model_accuracies.shape[1] < max_tasks_needed:
        print("Warning: Not enough tasks for full WOR sampling; capping task pool.")
        index_cache = np.array([
            np.random.choice(np.arange(model_accuracies.shape[1]), model_accuracies.shape[1], replace=False)
            for _ in range(k)
        ]).T  # (model_accuracies.shape[1], k)
    else:
        index_cache = np.array([
            np.random.choice(np.arange(model_accuracies.shape[1]), max_tasks_needed, replace=False)
            for _ in range(k)
        ]).T  # (max_tasks_needed, k)

    # Warm-up: one pull per arm using the first cached task in each run
    t = 0
    sums += get_results(np.arange(n_arms), index_cache[0], cross=True)
    counts += 1
    t += n_arms

    arm_selections = None
    if track_arm_selection:
        arm_selections = np.zeros((n - n_arms, k), dtype=int)

    if checkpoint_budgets is not None:
        while (checkpoint_idx < len(checkpoint_budgets_sorted)
               and t >= checkpoint_budgets_sorted[checkpoint_idx]):
            checkpoint_results[checkpoint_budgets_sorted[checkpoint_idx]] = (
                random_argmax(sums / counts, 0)
            )
            print(f"  Smart-UCB-E checkpoint at budget {checkpoint_budgets_sorted[checkpoint_idx]}, t={t}")
            checkpoint_idx += 1

    # Main loop
    for iteration in range(n - n_arms):
        means = sums / counts
        ucb = means + np.sqrt(a / counts)

        available_mask = counts < index_cache.shape[0]
        masked_ucb = ucb.copy()
        masked_ucb[~available_mask] = -np.inf
        to_pull = random_argmax(masked_ucb, axis=0)

        if track_arm_selection:
            arm_selections[iteration, :] = to_pull

        per_arm_round = counts[to_pull, np.arange(k)]
        mask = per_arm_round >= index_cache.shape[0]
        per_arm_round[mask] = 0

        indices = index_cache[per_arm_round, np.arange(k)]
        results = get_results(to_pull, indices, cross=False)
        results[mask] = 0.0

        sums[to_pull, np.arange(k)] += results
        counts[to_pull, np.arange(k)] += (~mask).astype(int)
        t += 1

        if checkpoint_budgets is not None:
            while (checkpoint_idx < len(checkpoint_budgets_sorted)
                   and t >= checkpoint_budgets_sorted[checkpoint_idx]):
                checkpoint_results[checkpoint_budgets_sorted[checkpoint_idx]] = (
                    random_argmax(sums / counts, 0)
                )
                print(f"  Smart-UCB-E checkpoint at budget {checkpoint_budgets_sorted[checkpoint_idx]}, t={t}")
                checkpoint_idx += 1

    final_result = random_argmax(sums / counts, 0)
    final_means = sums / counts
    empirical_means = final_means if track_means else None

    if checkpoint_budgets is not None:
        if n not in checkpoint_results:
            checkpoint_results[n] = final_result
            print(f"  Smart-UCB-E final checkpoint at budget {n}")
        return checkpoint_results
    else:
        if track_arm_selection or track_means:
            return (final_result, arm_selections, empirical_means, counts)
        return final_result


# ============================================================================
# Budget reallocation helper for SR
# ============================================================================

def _reallocate_budget_across_rounds(n_items, nk_original, total_n_tasks, n_arms, verbose=True):
    """Redistribute budget from task-capped SR rounds to earlier rounds.

    When some phases of SR require more tasks than available (nk[t] > total_n_tasks),
    the unused budget is redistributed proportionally to uncapped phases so that
    the total budget is preserved.

    Args:
        n_items:       total budget (int)
        nk_original:   cumulative task schedule from standard SR, shape (n_arms,)
        total_n_tasks: number of available tasks
        n_arms:        number of arms
        verbose:       print reallocation details

    Returns:
        (nk_reallocated, allocation_info)
        nk_reallocated: adjusted cumulative schedule (np.ndarray, int)
        allocation_info: dict with reallocation metadata
    """
    nk_current = nk_original.copy().astype(float)
    num_rounds = len(nk_original)

    total_budget_original = 0
    for t in range(1, num_rounds):
        n_active_arms = n_arms - t + 1
        total_budget_original += (nk_original[t] - nk_original[t - 1]) * n_active_arms

    budget_difference = n_items - total_budget_original
    target_budget = n_items

    if verbose:
        print(f"\n=== Budget Reallocation ===")
        print(f"Original schedule nk: {nk_original}")
        print(f"Total tasks available: {total_n_tasks}")
        print(f"Original total budget: {total_budget_original}")
        print(f"Requested budget (n_items): {n_items}")
        print(f"Budget gap to reallocate: {budget_difference}")

    saturated_rounds = set()
    unused_budget_accumulated = 0

    iteration = 0
    max_iterations = num_rounds
    alpha = 1.0

    while iteration < max_iterations:
        iteration += 1

        newly_saturated = set()
        for t in range(1, num_rounds):
            if t not in saturated_rounds and nk_current[t] > total_n_tasks:
                newly_saturated.add(t)

        if not newly_saturated:
            break

        saturated_rounds.update(newly_saturated)

        if verbose:
            print(f"\nIteration {iteration}: Found newly saturated rounds: {newly_saturated}")

        unused_budget_this_iter = 0
        for t in sorted(newly_saturated):
            n_active_arms = n_arms - t + 1
            if t == min(saturated_rounds):
                unused_budget_this_iter += (nk_current[t] - total_n_tasks) * n_active_arms
            else:
                unused_budget_this_iter += (nk_current[t] - nk_current[t - 1]) * n_active_arms

        unused_budget_accumulated += unused_budget_this_iter

        total_weighted_budget = 0
        boundary_budget_constant = 0

        for t in range(1, num_rounds):
            if t not in saturated_rounds:
                n_active_arms = n_arms - t + 1
                total_weighted_budget += (nk_original[t] - nk_original[t - 1]) * n_active_arms

        for t in sorted(saturated_rounds):
            if t > 1 and (t - 1) not in saturated_rounds:
                n_active_arms = n_arms - t + 1
                boundary_budget_constant += total_n_tasks * n_active_arms
                total_weighted_budget -= nk_original[t - 1] * n_active_arms

        budget_consecutive_saturated = 0
        saturated_sorted = sorted(saturated_rounds)
        for i, t in enumerate(saturated_sorted):
            n_active_arms = n_arms - t + 1
            if i == 0 and t == 1:
                budget_consecutive_saturated += total_n_tasks * n_active_arms

        if total_weighted_budget == 0:
            if verbose:
                print("All rounds saturated, capping all to total_n_tasks")
            break

        alpha = (
            (target_budget - budget_consecutive_saturated - boundary_budget_constant)
            / total_weighted_budget
        )

        if verbose:
            print(f"  Alpha: {alpha:.6f}")

        for t in range(1, num_rounds):
            if t not in saturated_rounds:
                nk_current[t] = nk_original[t] * alpha

    for t in saturated_rounds:
        nk_current[t] = total_n_tasks

    nk_reallocated = np.ceil(nk_current).astype(int)
    nk_reallocated = np.maximum.accumulate(nk_reallocated)

    total_budget_final = 0
    for t in range(1, num_rounds):
        n_active_arms = n_arms - t + 1
        total_budget_final += (nk_reallocated[t] - nk_reallocated[t - 1]) * n_active_arms

    if verbose:
        print(f"\nFinal schedule nk: {nk_reallocated}")
        print(f"Final total budget: {total_budget_final}")
        print(f"Saturated rounds: {saturated_rounds}")
        print("=== End Reallocation ===\n")

    alpha_per_round = {}
    for t in range(1, num_rounds):
        if t in saturated_rounds:
            alpha_per_round[t] = None
        else:
            alpha_per_round[t] = (
                float(nk_reallocated[t] / nk_original[t]) if nk_original[t] > 0 else 1.0
            )

    allocation_info = {
        'nk_original': nk_original,
        'nk_reallocated': nk_reallocated,
        'total_budget_original': total_budget_original,
        'target_budget': target_budget,
        'total_budget_final': total_budget_final,
        'budget_gap_original': budget_difference,
        'saturated_rounds': saturated_rounds,
        'alpha_final': alpha,
        'alpha_per_round': alpha_per_round,
        'iterations': iteration,
        'budget_reallocated': len(saturated_rounds) > 0,
    }

    return nk_reallocated, allocation_info


# ============================================================================
# Successive Rejects variants
# ============================================================================

def successive_rejects_wo_replacement_no_budget_limit(
    n_items,
    k=1000,
    return_allocation=False,
    verbose=True,
    track_arm_selection=False,
    track_means=False,
    track_elimination=False,
):
    """Successive Rejects with per-arm independent WOR sampling and budget reallocation.

    Handles budgets that would require more tasks than available by proportionally
    redistributing unused budget from later (task-saturated) rounds to earlier rounds.

    Args:
        n_items:          total budget
        k:                number of independent runs
        return_allocation: if True and no tracking, return (result, allocation_info)
        verbose:          print reallocation scheme
        track_arm_selection: unused (SR does not select per-step; kept for API parity)
        track_means:      if True, return empirical means per arm per run

    Returns:
        No tracking, no return_allocation:
            np.ndarray of shape (k,) — selected arms
        No tracking, return_allocation=True:
            (selected_arms, allocation_info)
        Tracking enabled:
            (selected_arms, None, empirical_means, counts)
            empirical_means: shape (n_arms, k) or None
            counts: shape (n_arms, k)
    """
    random.seed(42)
    np.random.seed(42)
    n_arms = model_accuracies.shape[0]
    assert n_arms >= 2 and n_items >= n_arms

    logbar = 0.5 + (1 / np.arange(2, n_arms + 1)).sum()
    phases = np.arange(1, n_arms)
    nk_original = np.ceil(
        (n_items - n_arms) / logbar / (n_arms + 1 - phases)
    ).astype(int)
    nk_original = np.r_[0, np.maximum.accumulate(nk_original)]

    nk, allocation_info = _reallocate_budget_across_rounds(
        n_items, nk_original, model_accuracies.shape[1], n_arms, verbose
    )

    max_tasks_available = min(model_accuracies.shape[1], nk[-1])

    # Pre-sample task order for each (arm, run)
    task_indices = np.zeros((n_arms, k, max_tasks_available), dtype=int)
    for arm in range(n_arms):
        for run in range(k):
            task_indices[arm, run] = np.random.choice(
                np.arange(model_accuracies.shape[1]), max_tasks_available, replace=False
            )

    sums = np.zeros((n_arms, k))
    counts = np.zeros((n_arms, k), dtype=int)
    A = np.ones((n_arms, k))
    # elim_rank[arm, run] = phase at which arm was eliminated (1=first/worst);
    # survivor initialised to n_arms (best).  Only allocated when requested.
    elim_rank = np.full((n_arms, k), n_arms, dtype=int) if track_elimination else None

    task_idx = 0
    for t in range(1, n_arms):
        extra = nk[t] - nk[t - 1]
        if extra and task_idx + extra <= max_tasks_available:
            phase_tasks = task_indices[:, :, task_idx:task_idx + extra]
            phase_tasks_flat = phase_tasks.reshape(n_arms * k * extra)

            active_mask = ~np.isnan(A)
            results_all = get_results(
                np.repeat(np.arange(n_arms), extra * k),
                phase_tasks_flat,
                cross=False,
            ).reshape(n_arms, k, extra)

            for arm in range(n_arms):
                for run in range(k):
                    if active_mask[arm, run]:
                        sums[arm, run] += results_all[arm, run, :].sum()
                        counts[arm, run] += extra

            task_idx += extra

        means = np.divide(sums, counts, where=counts > 0)
        worst = random_argmax(-means * A, axis=0)
        if track_elimination:
            elim_rank[worst, np.arange(k)] = t
        A[worst, np.arange(k)] = np.nan

    result = random_argmax(A, 0)
    final_means = np.divide(sums, counts, where=counts > 0)
    empirical_means = final_means if track_means else None

    if track_arm_selection or track_means:
        if track_elimination:
            return result, None, empirical_means, counts, elim_rank
        return result, None, empirical_means, counts

    if return_allocation:
        return result, allocation_info
    return result


def smart_successive_rejects_wo_replacement_no_budget_limit(
    n_items,
    k=1000,
    return_allocation=False,
    verbose=True,
    track_arm_selection=False,
    track_means=False,
    track_elimination=False,
):
    """Successive Rejects with shared WOR sampling across arms and budget reallocation.

    Uses cross evaluation: within each run, all arms share the same task sequence,
    so each phase step reveals new tasks to all active arms simultaneously.

    Args:
        n_items:          total budget
        k:                number of independent runs
        return_allocation: if True and no tracking, return (result, allocation_info)
        verbose:          print reallocation scheme
        track_arm_selection: unused (SR does not select per-step; kept for API parity)
        track_means:      if True, return empirical means per arm per run

    Returns:
        Same format as successive_rejects_wo_replacement_no_budget_limit.
    """
    random.seed(42)
    np.random.seed(42)
    n_arms = model_accuracies.shape[0]
    assert n_arms >= 2 and n_items >= n_arms

    logbar = 0.5 + (1 / np.arange(2, n_arms + 1)).sum()
    phases = np.arange(1, n_arms)
    nk_original = np.ceil(
        (n_items - n_arms) / logbar / (n_arms + 1 - phases)
    ).astype(int)
    nk_original = np.r_[0, np.maximum.accumulate(nk_original)]

    nk, allocation_info = _reallocate_budget_across_rounds(
        n_items, nk_original, model_accuracies.shape[1], n_arms, verbose
    )

    max_tasks_available = min(model_accuracies.shape[1], nk[-1])

    # Shared task sequence per run (cross evaluation)
    task_indices = np.array([
        np.random.choice(np.arange(model_accuracies.shape[1]), max_tasks_available, replace=False)
        for _ in range(k)
    ]).T  # (max_tasks_available, k)

    sums = np.zeros((n_arms, k))
    counts = np.zeros((n_arms, k), dtype=int)
    A = np.ones((n_arms, k))
    # elim_rank[arm, run] = phase at which arm was eliminated (1=first/worst);
    # survivor initialised to n_arms (best).  Only allocated when requested.
    elim_rank = np.full((n_arms, k), n_arms, dtype=int) if track_elimination else None

    task_idx = 0
    for t in range(1, n_arms):
        extra = nk[t] - nk[t - 1]
        if extra and task_idx + extra <= max_tasks_available:
            phase_tasks = task_indices[task_idx:task_idx + extra]  # (extra, k)
            active_mask = ~np.isnan(A)

            for run in range(k):
                run_tasks = phase_tasks[:, run]
                run_results = get_results(
                    np.arange(n_arms), run_tasks, cross=True
                )  # (n_arms, extra)
                for arm in range(n_arms):
                    if active_mask[arm, run]:
                        sums[arm, run] += run_results[arm, :].sum()
                        counts[arm, run] += extra

            task_idx += extra

        means = np.divide(sums, counts, where=counts > 0)
        worst = random_argmax(-means * A, axis=0)
        if track_elimination:
            elim_rank[worst, np.arange(k)] = t
        A[worst, np.arange(k)] = np.nan

    result = random_argmax(A, 0)
    final_means = np.divide(sums, counts, where=counts > 0)
    empirical_means = final_means if track_means else None

    if track_arm_selection or track_means:
        if track_elimination:
            return result, None, empirical_means, counts, elim_rank
        return result, None, empirical_means, counts

    if return_allocation:
        return result, allocation_info
    return result


# ============================================================================
# Sequential Halving + Synchronization (SySH) – v2 with SR-merged budget
# ============================================================================

def smart_sequential_halving(
    n_items,
    k=1000,
    return_allocation=False,
    verbose=True,
    track_means=False,
    track_elimination=False,
):
    """Smart Sequential Halving: log2(K) phase elimination with shared tasks.

    Replaces the K-1 phase one-at-a-time elimination of SySRs with a
    ceil(log2(K)) phase halving schedule. Uses SR-style budget allocation
    merged into the halving phases, so early phases get proportionally
    more budget to avoid sample starvation.

    Args:
        n_items:          total budget
        k:                number of independent runs
        return_allocation: if True, return (result, allocation_info)
        verbose:          print allocation details
        track_means:      if True, return empirical means per arm per run
        track_elimination: if True, return elimination ranks

    Returns:
        Same format as smart_successive_rejects_wo_replacement_no_budget_limit.
    """
    random.seed(42)
    np.random.seed(42)
    n_arms = model_accuracies.shape[0]
    n_tasks = model_accuracies.shape[1]
    assert n_arms >= 2 and n_items >= n_arms

    # Surviving arm schedule: halving each phase
    surviving = [n_arms]
    while surviving[-1] > 1:
        surviving.append(max(1, int(np.ceil(surviving[-1] / 2))))
    n_phases = len(surviving) - 1

    if verbose:
        print(f"SySH: n_arms={n_arms}, n_phases={n_phases}, "
              f"schedule={surviving}")

    # Budget allocation: compute SR schedule, then merge into SH phases.
    # SR eliminates 1 arm per phase; SH eliminates many per phase.
    # Map: after SH phase p, S[p+1] arms survive = SR phases 1..(K-S[p+1]) done
    logbar = 0.5 + (1 / np.arange(2, n_arms + 1)).sum()
    sr_phases = np.arange(1, n_arms)
    sr_nk = np.ceil(
        (n_items - n_arms) / logbar / (n_arms + 1 - sr_phases)
    ).astype(int)
    sr_nk = np.r_[0, np.maximum.accumulate(sr_nk)]

    # For each SH phase, get the cumulative nk from the equivalent SR phase
    nk = [0]
    for p in range(n_phases):
        # After SH phase p, surviving[p+1] arms remain.
        # This equals SR after (n_arms - surviving[p+1]) phases.
        sr_idx = n_arms - surviving[p + 1]
        sr_idx = min(sr_idx, len(sr_nk) - 1)
        nk.append(int(sr_nk[sr_idx]))

    nk = np.array(nk, dtype=int)
    nk = np.maximum.accumulate(nk)

    # Ensure each phase has at least 1 task per arm
    for p in range(1, len(nk)):
        if nk[p] <= nk[p - 1]:
            nk[p] = nk[p - 1] + 1

    # Cap at available tasks
    if nk[-1] > n_tasks:
        nk = np.maximum.accumulate(np.ceil(
            nk.astype(float) * n_tasks / nk[-1]
        ).astype(int))
        nk = np.clip(nk, 0, n_tasks)
        # Ensure monotonic with at least +1 per phase
        for p in range(1, len(nk)):
            if nk[p] <= nk[p - 1]:
                nk[p] = nk[p - 1] + 1

    if verbose:
        print(f"SySH nk schedule: {nk}")

    max_available = min(n_tasks, nk[-1])

    # Shared task sequence per run (cross evaluation)
    task_indices = np.array([
        np.random.choice(np.arange(n_tasks), max_available, replace=False)
        for _ in range(k)
    ]).T  # (max_available, k)

    sums = np.zeros((n_arms, k))
    counts = np.zeros((n_arms, k), dtype=int)
    A = np.ones((n_arms, k))
    elim_rank = np.full((n_arms, k), n_arms, dtype=int) if track_elimination else None

    task_idx = 0
    for p in range(n_phases):
        extra = int(nk[p + 1] - nk[p])
        if extra <= 0 or task_idx + extra > max_available:
            continue

        phase_tasks = task_indices[task_idx:task_idx + extra]  # (extra, k)
        active_mask = ~np.isnan(A)

        for run in range(k):
            run_tasks = phase_tasks[:, run]
            run_results = get_results(
                np.arange(n_arms), run_tasks, cross=True
            )  # (n_arms, extra)
            for arm in range(n_arms):
                if active_mask[arm, run]:
                    sums[arm, run] += run_results[arm, :].sum()
                    counts[arm, run] += extra

        task_idx += extra

        means = np.divide(sums, counts, where=counts > 0)

        # Eliminate the bottom half of surviving arms
        current_surviving = surviving[p]
        next_surviving = surviving[p + 1]
        n_eliminate = current_surviving - next_surviving

        if n_eliminate > 0:
            for run in range(k):
                run_means = means[:, run].copy()
                run_means[np.isnan(A[:, run])] = np.inf
                worst_indices = np.argsort(run_means)[:n_eliminate]
                if track_elimination:
                    elim_rank[worst_indices, run] = p + 1
                A[worst_indices, run] = np.nan

    result = random_argmax(A, 0)
    final_means = np.divide(sums, counts, where=counts > 0)
    empirical_means = final_means if track_means else None

    if track_means:
        if track_elimination:
            return result, None, empirical_means, counts, elim_rank
        return result, None, empirical_means, counts

    if return_allocation:
        return result, {'nk': nk, 'surviving': surviving, 'n_phases': n_phases}
    return result
