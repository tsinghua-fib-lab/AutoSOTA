"""
Vectorized UCB-LRF implementation for k independent runs.

Maintains k separate observation matrices and makes k independent decisions at
each timestep, enabling efficient parallel execution via fully batched operations.

Key design points:
- BatchedFactorization: handles k runs with no Python loops over k; all ALS
  iterations are fully vectorized.
- Vectorized random_argmax_batched: random tie-breaking for all k runs at once.
- Vectorized UCB computation: bounds, selections, and item sampling use no loops.
- Vectorized observation updates: single batched get_results_torch call.
- Uniform and uncertainty-based item sampling both fully vectorized.
- No file I/O: all results kept in memory.
"""

import torch
import numpy as np
import pickle
import os
import sys
import time
import psutil


# ============================================================================
# Oracle: evaluate models on tasks
# ============================================================================

def get_results_torch(models, tasks, cross=False):
    """Evaluate models on tasks and return rewards as a tensor.

    Args:
        models: tensor/array of model indices
        tasks:  tensor/array of task indices
        cross:  if True, each model is evaluated on every task in tasks

    Returns:
        Tensor of rewards.
        cross=False: shape (#models,)
        cross=True:  shape (#models, #tasks)
    """
    if not cross:
        return model_accuracies[models, tasks]
    else:
        return model_accuracies[models][:, tasks]


# ============================================================================
# Utility functions
# ============================================================================

def random_argmax_torch(a, generator=None):
    """Random argmax for a 1-D tensor.

    Args:
        a:         1-D tensor
        generator: torch.Generator for reproducibility

    Returns:
        int, index of a randomly chosen maximum element
    """
    max_val = a.max()
    mask = (a == max_val)
    candidates = torch.nonzero(mask, as_tuple=False).squeeze()
    if candidates.dim() == 0:
        return candidates.item()
    pick = torch.randint(0, candidates.shape[0], (1,), generator=generator).item()
    return int(candidates[pick].item())


def acc(results):
    """Fraction of runs that selected arm 0 (the designated best arm)."""
    if isinstance(results, torch.Tensor):
        return ((results == 0).float().mean()).item()
    return (results == 0).mean()


def calculate_budgets_from_percentage(total_tasks, total_models, percentages):
    """Convert percentage values to absolute budget integers.

    Args:
        total_tasks:   int
        total_models:  int
        percentages:   list of floats, e.g. [0.01, 0.05, 0.10]

    Returns:
        list of ints
    """
    total_entries = total_tasks * total_models
    return [int(total_entries * pct) for pct in percentages]


def cap_and_adjust_budgets(budgets, n_tasks, n_arms, with_replacement=True):
    """Ensure all budgets are divisible by n_arms (no cap for UCB-LRF).

    Args:
        budgets:          list of budget values
        n_tasks:          number of tasks
        n_arms:           number of arms
        with_replacement: always True for UCB-LRF (kept for API compatibility)

    Returns:
        (adjusted_budgets, budgets_capped, max_theoretical_budget)
    """
    budgets = list(budgets)
    for i in range(len(budgets)):
        if budgets[i] % n_arms != 0:
            budgets[i] = (budgets[i] // n_arms) * n_arms
    return sorted(list(set(budgets))), False, None


def random_argmax_torch_batched(a: torch.Tensor, generator: torch.Generator = None) -> torch.Tensor:
    """Vectorized random argmax for a batch of rows.

    Args:
        a:         Tensor of shape (k, n)
        generator: torch.Generator for reproducibility

    Returns:
        Tensor of shape (k,) with a randomly chosen argmax index per row.
    """
    max_vals = a.max(dim=1, keepdim=True).values  # (k, 1)
    masks = (a == max_vals)  # (k, n)
    random_tiebreaker = torch.rand(a.shape, generator=generator, device=a.device)
    random_tiebreaker = torch.where(
        masks, random_tiebreaker, torch.tensor(-1.0, device=a.device)
    )
    return random_tiebreaker.argmax(dim=1)


# ============================================================================
# Global state (set by run_lrf_experiment_vectorized)
# ============================================================================

model_accuracies = None   # shape: (n_arms, n_tasks); set this before calling algorithms
# total_n_arms and total_n_tasks are derived from model_accuracies.shape


# ============================================================================
# Batched low-rank factorization (ALS)
# ============================================================================

class BatchedFactorization(torch.nn.Module):
    """Ensemble of low-rank matrix factorizations for k independent runs.

    Factorizes a (k, m_methods, n_examples) observation matrix using ALS.
    Each of the k runs maintains its own ensemble of ``ensemble_size`` factors.

    Shape conventions:
        U : (k, ensemble_size, m_methods,  rank)
        V : (k, ensemble_size, n_examples, rank)
        X : (k, m_methods, n_examples)          — input observation matrix
    """

    def __init__(
        self,
        m_methods: int,
        n_examples: int,
        rank: int,
        ensemble_size: int,
        k: int,
        regularizer_weight: float = 0.0,
        drop_probability: float = 0.05,
        device: str = "cpu",
        use_half: bool = False,
        generator: torch.Generator = None,
    ) -> None:
        super().__init__()

        self.device = torch.device(device)
        self.dtype = torch.float16 if use_half else torch.float32
        self.generator = generator

        U = torch.randn(
            k, ensemble_size, m_methods, rank,
            device=self.device, dtype=self.dtype, generator=generator,
        )
        V = torch.randn(
            k, ensemble_size, n_examples, rank,
            device=self.device, dtype=self.dtype, generator=generator,
        )

        self.register_buffer("U", U)
        self.register_buffer("V", V)
        self.register_buffer(
            "L",
            regularizer_weight * torch.eye(rank, device=self.device, dtype=self.dtype),
        )

        self.k = k
        self.m_methods = m_methods
        self.n_examples = n_examples
        self.rank = rank
        self.ensemble_size = ensemble_size
        self.regularizer_weight = regularizer_weight
        self.drop_probability = drop_probability

    def forward(self) -> torch.Tensor:
        """Return reconstructed matrices of shape (k, ensemble_size, m_methods, n_examples)."""
        k_ens = self.k * self.ensemble_size
        U_flat = self.U.reshape(k_ens, self.m_methods, self.rank)
        V_flat = self.V.reshape(k_ens, self.n_examples, self.rank)
        result = torch.bmm(U_flat, V_flat.transpose(1, 2))
        return result.reshape(self.k, self.ensemble_size, self.m_methods, self.n_examples)

    def _als_step_optimized_batched(
        self,
        data_matrix: torch.Tensor,
        data_filled: torch.Tensor,
        non_zero_mask: torch.Tensor,
        fixed_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Single ALS update step (vmapped over the batch dimension)."""
        y = fixed_matrix.unsqueeze(2)
        y_t = y.transpose(1, 2)

        A = (non_zero_mask.unsqueeze(2) * torch.bmm(y, y_t)).sum(0)
        A = A + self.L

        eps = 1e-6 if self.dtype == torch.float32 else 1e-3
        A = A + eps * torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)

        b = (data_filled * non_zero_mask * y.squeeze(2)).sum(0)

        try:
            return torch.linalg.solve(A, b)
        except RuntimeError:
            return torch.linalg.lstsq(A, b).solution

    def fit(self, X: torch.Tensor, iterations: int = 10) -> None:
        """Fit k independent factorizations simultaneously.

        Args:
            X:          Tensor of shape (k, m_methods, n_examples); NaN = unobserved
            iterations: number of ALS iterations
        """
        # Replicate across ensemble dimension: (k, m, n) -> (k, ens, m, n)
        X = X.unsqueeze(1).repeat(1, self.ensemble_size, 1, 1)

        if self.drop_probability > 0:
            n_total = self.m_methods * self.n_examples
            n_to_drop = int(self.drop_probability * n_total)
            random_vals = torch.rand(
                (self.k, self.ensemble_size, n_total),
                device=X.device, dtype=X.dtype, generator=self.generator,
            )
            _, indices_to_drop = torch.topk(random_vals, n_to_drop, dim=2, largest=False)
            X_flat = X.reshape(self.k, self.ensemble_size, n_total)
            X_flat.scatter_(2, indices_to_drop, float('nan'))
            X = X_flat.reshape(self.k, self.ensemble_size, self.m_methods, self.n_examples)

        non_zero_mask = (~torch.isnan(X)).float()
        X_filled = torch.where(torch.isnan(X), torch.zeros_like(X), X)

        k_ens = self.k * self.ensemble_size

        X_u = X.reshape(k_ens * self.m_methods, self.n_examples, 1)
        X_u_filled = X_filled.reshape(k_ens * self.m_methods, self.n_examples, 1)
        mask_u = non_zero_mask.reshape(k_ens * self.m_methods, self.n_examples, 1)

        X_v = X.transpose(2, 3).reshape(k_ens * self.n_examples, self.m_methods, 1)
        X_v_filled = X_filled.transpose(2, 3).reshape(
            k_ens * self.n_examples, self.m_methods, 1
        )
        mask_v = non_zero_mask.transpose(2, 3).reshape(
            k_ens * self.n_examples, self.m_methods, 1
        )

        vmap_als_step = torch.vmap(self._als_step_optimized_batched, in_dims=(0, 0, 0, 0))

        for _ in range(iterations):
            self.V.data = (
                vmap_als_step(
                    X_v, X_v_filled, mask_v,
                    self.U.repeat(1, self.n_examples, 1, 1).reshape(
                        k_ens * self.n_examples, self.m_methods, self.rank
                    ),
                )
            ).reshape(self.k, self.ensemble_size, self.n_examples, self.rank)

            self.U.data = (
                vmap_als_step(
                    X_u, X_u_filled, mask_u,
                    self.V.repeat(1, self.m_methods, 1, 1).reshape(
                        k_ens * self.m_methods, self.n_examples, self.rank
                    ),
                )
            ).reshape(self.k, self.ensemble_size, self.m_methods, self.rank)


# ============================================================================
# UCB-LRF exploration step (batched over k runs)
# ============================================================================

def upper_confidence_bound_exploration_low_rank_factorization_batched(
    observed_matrices: torch.Tensor,
    batch_size: int = 1,
    rank: int = 1,
    ensemble_size: int = 64,
    warmup_percentage: float = 0.05,
    regularizer_weight: float = 0.1,
    drop_probability: float = 0.05,
    iterations: int = 10,
    eta: float = 5,
    device: str = "cpu",
    use_half: bool = False,
    use_uncertainty_arm: bool = True,
    use_mean_arm: bool = True,
    uniform_item_sampling: bool = False,
    generator: torch.Generator = None,
    use_real_accuracy: bool = False,
    real_accuracy_matrix: torch.Tensor = None,
):
    """Compute UCB-LRF arm/item selections for all k runs in one batched pass.

    Args:
        observed_matrices: Tensor of shape (k, n_arms, n_examples); NaN = unobserved

    Returns:
        (result_batches, fully_explored, valid_counts, entry_mus, entry_stds)
        result_batches:  (k, 2, max_items) — [arm_idx, item_idx] for each run
        fully_explored:  (k,) bool — True if run has no remaining unobserved entries
        valid_counts:    (k,) int  — actual number of valid items in this batch
        entry_mus:       (k, n_arms, n_examples) — imputed means from ensemble
        entry_stds:      (k, n_arms, n_examples) — uncertainties from ensemble
    """
    k, m_methods, n_examples = observed_matrices.shape
    dtype = observed_matrices.dtype

    needs_uncertainty = use_uncertainty_arm or not uniform_item_sampling
    effective_ensemble_size = ensemble_size if needs_uncertainty else 1
    effective_drop_probability = drop_probability if needs_uncertainty else 0.0

    if use_real_accuracy:
        entry_mus = (
            real_accuracy_matrix.unsqueeze(0)
            .expand(k, -1, -1)
            .to(dtype=observed_matrices.dtype)
        )
        entry_stds = torch.zeros_like(entry_mus)
    else:
        factorization = BatchedFactorization(
            m_methods, n_examples, rank, effective_ensemble_size, k,
            regularizer_weight=regularizer_weight,
            drop_probability=effective_drop_probability,
            device=device,
            use_half=use_half,
            generator=generator,
        )
        factorization.fit(observed_matrices, iterations=iterations)
        matrix_approximations = factorization()  # (k, ens, n_arms, n_examples)

        entry_mus = matrix_approximations.mean(1)  # (k, n_arms, n_examples)
        entry_stds = (
            matrix_approximations.std(1)
            if effective_ensemble_size > 1
            else torch.zeros_like(entry_mus)
        )

    # Build UCB scores
    if use_mean_arm and use_uncertainty_arm:
        entry_ucb = entry_mus + eta * entry_stds
    elif use_mean_arm:
        entry_ucb = entry_mus.clone()
    elif use_uncertainty_arm:
        entry_ucb = eta * entry_stds
    else:
        entry_ucb = torch.zeros_like(entry_mus)

    # Override with observed values
    observed_mask = ~observed_matrices.isnan()  # (k, n_arms, n_examples)
    entry_ucb[observed_mask] = observed_matrices[observed_mask].to(entry_ucb.dtype)

    ucb = entry_ucb.mean(2)   # (k, n_arms) — arm-level UCB
    counts = observed_mask.sum(2)  # (k, n_arms)

    bounds = torch.full((k, m_methods), torch.inf, device=device, dtype=dtype)
    mask = counts > 0
    bounds[mask] = ucb[mask].to(bounds.dtype)

    completely_sensed_mask = (counts == n_examples)
    bounds[completely_sensed_mask] = -torch.inf

    fully_explored = (completely_sensed_mask.sum(1) == m_methods)  # (k,)
    best_method_indices = torch.argmax(bounds, dim=1)  # (k,)

    if uniform_item_sampling:
        best_arm_masks = observed_mask[torch.arange(k, device=device), best_method_indices]
        random_vals = torch.rand((k, n_examples), device=device, generator=generator)
        random_vals[best_arm_masks] = 2.0

        sorted_indices = torch.argsort(random_vals, dim=1)
        unobserved_counts = (~best_arm_masks).sum(dim=1)
        actual_sizes = torch.minimum(
            torch.full_like(unobserved_counts, batch_size), unobserved_counts
        )

        max_items = min(batch_size, n_examples)
        selected_items = sorted_indices[:, :max_items]
        valid_counts = actual_sizes
    else:
        best_arm_stds = entry_stds[torch.arange(k, device=device), best_method_indices]
        best_arm_masks = observed_mask[torch.arange(k, device=device), best_method_indices]

        best_arm_stds_masked = best_arm_stds.clone()
        best_arm_stds_masked[best_arm_masks] = -1

        unobserved_counts = (~best_arm_masks).sum(dim=1)
        actual_sizes = torch.minimum(
            torch.full_like(unobserved_counts, batch_size), unobserved_counts
        )

        max_items = min(batch_size, n_examples)
        _, topk_indices = torch.topk(best_arm_stds_masked, max_items, dim=1, largest=True)

        selected_items = topk_indices
        valid_counts = actual_sizes

    max_items = min(batch_size, n_examples)
    result_batches = torch.zeros((k, 2, max_items), dtype=torch.long, device=device)
    result_batches[:, 0, :] = best_method_indices.unsqueeze(1).expand(-1, max_items)
    result_batches[:, 1, :] = selected_items[:, :max_items]

    return result_batches, fully_explored, valid_counts, entry_mus, entry_stds


# ============================================================================
# Main UCB-LRF algorithm (vectorized over k runs)
# ============================================================================

def ucb_e_lrf_vectorized(
    budget,
    k=1000,
    r=1,
    C=64,
    T0=None,
    eta=5.0,
    drop_rate=0.2,
    lam=1e-3,
    max_iters=20,
    seed=0,
    device="cpu",
    use_half=False,
    batch_size=1,
    checkpoint_budgets=None,
    # Ablation study parameters
    use_uncertainty_arm=True,
    use_mean_arm=True,
    uniform_item_sampling=False,
    # Sanity check parameters
    use_real_accuracy=False,
    real_accuracy_matrix=None,
):
    """Vectorized UCB-LRF: k independent runs processed in parallel.

    Args:
        budget:             maximum budget per run
        k:                  number of independent runs
        r:                  matrix rank for factorization
        C:                  ensemble size
        T0:                 warm-up budget (random pulls before UCB phase);
                            defaults to total_n_arms if None
        eta:                UCB exploration coefficient
        drop_rate:          dropout probability for ensemble diversity
        lam:                ALS regularization weight
        max_iters:          number of ALS iterations per step
        seed:               random seed
        device:             'cpu' or 'cuda'
        use_half:           use float16 instead of float32
        batch_size:         items to query per step per run
        checkpoint_budgets: list of intermediate budgets at which to record
                            the current best arm selection; if None only the
                            final result is returned
        use_uncertainty_arm: include uncertainty in UCB score
        use_mean_arm:        include imputed mean in UCB score
        uniform_item_sampling: sample items uniformly instead of by uncertainty
        use_real_accuracy:   sanity-check mode; use ground-truth accuracy matrix
        real_accuracy_matrix: Tensor of shape (n_arms, n_examples) for sanity check

    Returns:
        If checkpoint_budgets is None:
            np.ndarray of shape (k,) — selected arms at final budget
        If checkpoint_budgets is not None:
            dict mapping budget (int) -> np.ndarray of shape (k,) — selected arms
    """
    start_time = time.time()
    start_cpu_mem = psutil.Process().memory_info().rss / 1024 ** 2

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    dev = torch.device(device)
    dtype = torch.float16 if use_half else torch.float32
    is_cuda = dev.type == 'cuda'

    if is_cuda:
        torch.cuda.reset_peak_memory_stats(dev)
        torch.cuda.synchronize(dev)
        start_gpu_mem = torch.cuda.memory_allocated(dev) / 1024 ** 2

    n_arms = model_accuracies.shape[0]
    n_examples = model_accuracies.shape[1]

    gen_cpu = torch.Generator(device='cpu').manual_seed(int(seed))
    gen = torch.Generator(device=dev).manual_seed(int(seed)) if is_cuda else gen_cpu

    print(f"\n{'=' * 60}")
    print(f"Starting UCB-LRF Vectorized (k={k}, device={device})")
    print(f"Dataset: {n_arms} arms × {n_examples} tasks, budget={budget}")
    print(f"{'=' * 60}")

    # Observation matrices: shape (k, n_arms, n_examples)
    S_obs = torch.full((k, n_arms, n_examples), float('nan'), dtype=dtype, device=dev)
    O = torch.zeros((k, n_arms, n_examples), dtype=torch.bool, device=dev)

    if T0 is None:
        T0 = n_arms
    T0 = min(int(T0), budget)

    total_observations = 0

    # Initialize checkpoint tracking
    if checkpoint_budgets is not None:
        checkpoint_budgets_sorted = sorted(checkpoint_budgets)
        checkpoint_results = {}
        checkpoint_idx = 0
        print(
            f"Running vectorized UCB-LRF to budget {budget} with {k} runs, "
            f"checkpoints at: {checkpoint_budgets_sorted}"
        )

    def get_current_selections():
        """Return best arm for all k runs using a final factorization pass."""
        factorization_batch = BatchedFactorization(
            n_arms, n_examples, r, 1, k,
            regularizer_weight=lam,
            drop_probability=0.0,
            device=device,
            use_half=use_half,
            generator=gen,
        )
        factorization_batch.fit(S_obs, iterations=20)
        S_hat_batch = factorization_batch().squeeze(1)  # (k, n_arms, n_examples)
        S_combined_batch = torch.where(O, S_obs, S_hat_batch)
        S_means_batch = torch.nanmean(S_combined_batch, dim=2)  # (k, n_arms)
        return random_argmax_torch_batched(S_means_batch, generator=gen)

    # ---- Warm-up phase: random sampling ----
    warmup_start = time.time()
    warmup_attempts = 0
    max_warmup_attempts = T0 * 10

    while total_observations < T0 * k and warmup_attempts < max_warmup_attempts:
        warmup_attempts += 1

        run_indices = torch.arange(k, device=dev)
        arm_choices = torch.randint(0, n_arms, (k,), generator=gen_cpu).to(dev)
        task_choices = torch.randint(0, n_examples, (k,), generator=gen_cpu).to(dev)

        mask_new = ~O[run_indices, arm_choices, task_choices]
        if mask_new.sum() > 0:
            valid_runs = run_indices[mask_new]
            valid_arms = arm_choices[mask_new]
            valid_tasks = task_choices[mask_new]

            vals = get_results_torch(valid_arms.to(dev), valid_tasks.to(dev), cross=False)
            S_obs[valid_runs, valid_arms, valid_tasks] = vals.to(dtype)
            O[valid_runs, valid_arms, valid_tasks] = True

            n_new = mask_new.sum().item()
            total_observations += n_new

            if checkpoint_budgets is not None:
                while (
                    checkpoint_idx < len(checkpoint_budgets_sorted)
                    and total_observations >= checkpoint_budgets_sorted[checkpoint_idx] * k
                ):
                    selections = get_current_selections()
                    selections_np = selections.cpu().numpy()
                    checkpoint_results[checkpoint_budgets_sorted[checkpoint_idx]] = selections_np
                    checkpoint_idx += 1

        if total_observations >= budget * k:
            break

    warmup_time = time.time() - warmup_start
    print(f"Warmup completed: {total_observations} observations in {warmup_time:.2f}s")

    # ---- Main UCB-LRF loop ----
    main_loop_start = time.time()
    while total_observations < budget * k:
        warmup_pct = T0 / float(max(1, n_arms * n_examples))

        (
            result_batches,
            fully_explored_mask,
            valid_counts,
            entry_mus,
            entry_stds,
        ) = upper_confidence_bound_exploration_low_rank_factorization_batched(
            S_obs,
            batch_size=batch_size,
            rank=r,
            ensemble_size=C,
            warmup_percentage=warmup_pct,
            regularizer_weight=lam,
            drop_probability=drop_rate,
            iterations=max(1, int(max_iters)),
            eta=eta,
            device=device,
            use_half=use_half,
            use_uncertainty_arm=use_uncertainty_arm,
            use_mean_arm=use_mean_arm,
            uniform_item_sampling=uniform_item_sampling,
            generator=gen,
            use_real_accuracy=use_real_accuracy,
            real_accuracy_matrix=real_accuracy_matrix,
        )

        best_arms = result_batches[:, 0, 0]           # (k,)
        all_selected_items = result_batches[:, 1, :]   # (k, max_items)
        item_counts = valid_counts                      # (k,)
        max_batch_items = result_batches.shape[2]

        active_mask = ~fully_explored_mask  # (k,)

        if active_mask.any():
            active_runs_idx = torch.nonzero(active_mask, as_tuple=False).squeeze()
            if active_runs_idx.dim() == 0:
                active_runs_idx = active_runs_idx.unsqueeze(0)

            item_range = torch.arange(max_batch_items, device=dev).unsqueeze(0).expand(k, -1)
            valid_items_mask = item_range < item_counts.unsqueeze(1)  # (k, max_items)
            active_valid_mask = active_mask.unsqueeze(1) & valid_items_mask

            flat_indices = torch.nonzero(active_valid_mask, as_tuple=False)  # (N, 2)
            if flat_indices.shape[0] > 0:
                run_indices = flat_indices[:, 0]
                item_indices = flat_indices[:, 1]

                arm_values = best_arms[run_indices]
                item_values = all_selected_items[run_indices, item_indices]

                mask_new = ~O[run_indices, arm_values, item_values]

                n_already_observed = (~mask_new).sum().item()
                if n_already_observed > 0:
                    print(
                        f"WARNING: {n_already_observed} items already observed (masking bug?)"
                    )

                if mask_new.sum() > 0:
                    new_run_indices = run_indices[mask_new]
                    new_arm_values = arm_values[mask_new]
                    new_item_values = item_values[mask_new]

                    vals = get_results_torch(new_arm_values, new_item_values, cross=False)

                    S_obs[new_run_indices, new_arm_values, new_item_values] = vals.to(dtype)
                    O[new_run_indices, new_arm_values, new_item_values] = True

                    total_observations += mask_new.sum().item()

                    if checkpoint_budgets is not None:
                        while (
                            checkpoint_idx < len(checkpoint_budgets_sorted)
                            and total_observations
                            >= checkpoint_budgets_sorted[checkpoint_idx] * k
                        ):
                            selections = get_current_selections()
                            selections_np = selections.cpu().numpy()
                            checkpoint_results[
                                checkpoint_budgets_sorted[checkpoint_idx]
                            ] = selections_np
                            checkpoint_idx += 1
                else:
                    break
            else:
                break
        else:
            break

    # ---- Final selection ----
    final_selections = get_current_selections()

    if is_cuda:
        torch.cuda.synchronize(dev)

    total_time = time.time() - start_time
    main_loop_time = time.time() - main_loop_start
    end_cpu_mem = psutil.Process().memory_info().rss / 1024 ** 2
    cpu_mem_used = end_cpu_mem - start_cpu_mem

    print(f"\n{'=' * 60}")
    print(f"Profiling Results")
    print(f"{'=' * 60}")
    print(f"Total time:        {total_time:.2f}s")
    print(f"  Warmup time:     {warmup_time:.2f}s ({warmup_time / total_time * 100:.1f}%)")
    print(f"  Main loop time:  {main_loop_time:.2f}s ({main_loop_time / total_time * 100:.1f}%)")
    print(f"CPU memory:        {cpu_mem_used:+.1f} MB (peak RSS: {end_cpu_mem:.1f} MB)")

    if is_cuda:
        peak_gpu_mem = torch.cuda.max_memory_allocated(dev) / 1024 ** 2
        current_gpu_mem = torch.cuda.memory_allocated(dev) / 1024 ** 2
        print(f"GPU memory:        {current_gpu_mem:.1f} MB (peak: {peak_gpu_mem:.1f} MB)")

    obs_per_sec = total_observations / total_time
    print(f"Throughput:        {obs_per_sec:.1f} observations/sec")
    print(f"                   {obs_per_sec / k:.1f} obs/sec per run")
    print(f"Total observations: {total_observations}")
    print(f"{'=' * 60}\n")

    if checkpoint_budgets is not None:
        if budget not in checkpoint_results:
            checkpoint_results[budget] = final_selections.cpu().numpy()
        return checkpoint_results
    else:
        return final_selections.cpu().numpy()


# ============================================================================
# High-level experiment runner
# ============================================================================

def run_lrf_experiment_vectorized(
    dataset="mmlu",
    budgets=None,
    model_indices=None,
    k=1000,
    seed=42,
    rank=1,
    ensemble=10,
    eta=5.0,
    drop=0.05,
    lam=1e-1,
    iters=5,
    batch_size=32,
    T0_percentage=0.05,
    device='cpu',
    use_half=False,
    use_uncertainty_arm=True,
    use_mean_arm=True,
    uniform_item_sampling=False,
    use_real_accuracy=False,
):
    """Load data, configure globals, and run vectorized UCB-LRF.

    Loads ``model_accuracies_filtered.pkl`` from ``<script_dir>/../<dataset>/``.
    Sets the module-level ``model_accuracies`` global; ``total_n_arms`` and
    ``total_n_tasks`` are derived automatically from its shape.

    Args:
        dataset:       name of the dataset folder
        budgets:       int or list of ints; multiple budgets → checkpoint mode
        model_indices: list of model indices to evaluate, or None for all
        k:             number of independent runs
        seed:          random seed
        rank:          matrix rank for factorization
        ensemble:      ensemble size
        eta:           UCB exploration coefficient
        drop:          dropout probability for ensemble diversity
        lam:           ALS regularization weight
        iters:         number of ALS iterations per step
        batch_size:    items to query per step per run
        T0_percentage: warm-up budget as fraction of total entries
        device:        'cpu' or 'cuda'
        use_half:      use float16 instead of float32
        use_uncertainty_arm: include uncertainty in UCB score
        use_mean_arm:  include imputed mean in UCB score
        uniform_item_sampling: sample items uniformly instead of by uncertainty
        use_real_accuracy: sanity-check mode using ground-truth accuracy matrix

    Returns:
        If budgets is a scalar: np.ndarray of shape (k,) — selected arms
        If budgets is a list:   dict mapping budget (int) -> np.ndarray of shape (k,)
    """
    dtype = torch.float16 if use_half else torch.float32

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, 'datasets', dataset)

    global model_accuracies

    with open(os.path.join(dataset_dir, 'model_accuracies_filtered.pkl'), 'rb') as f:
        model_accuracies = pickle.load(f)
    model_accuracies = torch.from_numpy(model_accuracies).to(device=device, dtype=dtype)

    if model_indices is not None:
        model_indices_arr = np.array(model_indices)
        assert np.all(model_indices_arr < model_accuracies.shape[0]), (
            f"Some model indices out of bounds. Max: {model_accuracies.shape[0] - 1}"
        )
        model_accuracies = model_accuracies[model_indices_arr, :]

    n_arms = model_accuracies.shape[0]
    n_tasks = model_accuracies.shape[1]
    print(f"Dataset: {dataset}, n_arms={n_arms}, n_tasks={n_tasks}")

    # Load real accuracy matrix for sanity-check mode
    real_accuracy_tensor = None
    if use_real_accuracy:
        accuracy_path = os.path.join(dataset_dir, 'model_accuracies_filtered.pkl')
        if not os.path.exists(accuracy_path):
            raise FileNotFoundError(f"Real accuracy file not found: {accuracy_path}")
        with open(accuracy_path, 'rb') as f:
            real_accuracy = pickle.load(f)
        if model_indices is not None:
            real_accuracy = real_accuracy[model_indices, :]
        real_accuracy_tensor = torch.from_numpy(real_accuracy).to(device=device, dtype=dtype)
        print(f"Sanity-check mode: using real accuracy matrix {real_accuracy_tensor.shape}")

    # Normalise budget argument
    if isinstance(budgets, (list, tuple)):
        budgets_list = sorted(budgets)
    else:
        budgets_list = [budgets]

    max_budget = max(budgets_list)

    T0 = int(T0_percentage * n_arms * n_tasks)
    print(
        f"Warm-up budget T0: {T0} "
        f"({T0_percentage * 100:.1f}% of {n_arms * n_tasks} total entries)"
    )

    results = ucb_e_lrf_vectorized(
        budget=max_budget,
        k=k,
        r=rank,
        C=ensemble,
        T0=T0,
        eta=eta,
        drop_rate=drop,
        lam=lam,
        max_iters=iters,
        seed=seed,
        device=device,
        use_half=use_half,
        batch_size=batch_size,
        checkpoint_budgets=budgets_list if len(budgets_list) > 1 else None,
        use_uncertainty_arm=use_uncertainty_arm,
        use_mean_arm=use_mean_arm,
        uniform_item_sampling=uniform_item_sampling,
        use_real_accuracy=use_real_accuracy,
        real_accuracy_matrix=real_accuracy_tensor,
    )

    return results
