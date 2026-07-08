"""Robust GP Phased Elimination (RGP-PE) algorithm.

Implementation of Algorithm 1 from:
Bogunovic et al. (2022), "A Robust Phased Elimination Algorithm for
Corruption-Tolerant Gaussian Process Bandits", ICML 2022.

Key ideas:
- Epoch-based elimination: the active set of candidate actions shrinks over epochs.
- Rare switching: posterior variance is only recomputed when the information gain
  (measured via the kernel matrix determinant) increases by a factor eta.
  This ensures the same action is selected multiple times, making averaged
  observations harder to corrupt.
- Robust mean estimator (Eq. 7): averages corrupted observations for identical
  actions before computing the GP posterior mean.
- Enlarged confidence bounds for elimination that account for the corruption budget C.
"""

import math
import torch
import gpytorch
from typing import List, Dict, Any, Optional

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

from bo_framework.base.evaluation_result import EvaluationResult
from utilities.regret_analysis import find_best_points


class RobustGPPhasedElimination:
    """RGP-PE: Robust GP Phased Elimination (Bogunovic et al., 2022).

    This algorithm discretises the domain into a finite set of candidate points
    and iteratively eliminates sub-optimal actions via phased confidence-bound
    comparisons.  Within each epoch the algorithm:

    1. **Explores** by selecting the action with maximum posterior variance
       (with rare switching to encourage repeated selection of the same action).
    2. **Resamples** each selected action a minimum number of times controlled
       by the truncation parameter psi.
    3. **Eliminates** actions whose robust UCB falls below the best robust LCB.

    Args:
        domain_points: Tensor [N, d] of candidate actions (discretised domain).
        eta: Switching parameter (eta > 1). The posterior variance is only
            recomputed when ``log det`` of the Gram matrix increases by
            ``log(eta)``.
        psi: Truncation parameter (psi > 0). Each selected action is replayed
            at least ``ceil(steps_per_epoch * psi)`` times during resampling.
        beta: Confidence bound parameter (constant across epochs).
        lambda_reg: Regularisation / noise variance used in the GP posterior.
        b: Practical scaling for the corruption-dependent confidence width.
            Replaces the theory term ``C sqrt(u_h) / (l_h psi lambda)`` with
            ``b * C / sqrt(u_h)`` as suggested in Section 4.1 of the paper.
        corruption_budget: Known total corruption budget C.
        steps_per_epoch: Number of *exploration* evaluations per epoch
            (replaces the paper's exponentially-doubling ``l_h = 2^{h+1}``).
    """

    def __init__(
        self,
        domain_points: torch.Tensor,
        eta: float = 2.0,
        psi: float = 0.5,
        beta: float = 4.0,
        lambda_reg: float = 1.0,
        b: float = 0.1,
        corruption_budget: float = 0.0,
        steps_per_epoch: int = 10,
    ):
        self.domain_points = domain_points.double()
        self.n_domain = len(domain_points)
        self.d = domain_points.shape[1]
        self.eta = eta
        self.psi = psi
        self.beta = beta
        self.lambda_reg = lambda_reg
        self.b = b
        self.C = corruption_budget
        self.steps_per_epoch = steps_per_epoch

    # ------------------------------------------------------------------
    # GP helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fit_gp(X: torch.Tensor, Y: torch.Tensor) -> SingleTaskGP:
        """Fit a SingleTaskGP (same configuration as the GP baseline).

        No outcome standardisation is applied.
        """
        X = X.double()
        Y = Y.double().unsqueeze(-1) if Y.dim() == 1 else Y.double()
        model = SingleTaskGP(X, Y, outcome_transform=None)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        try:
            fit_gpytorch_mll(mll)
        except Exception:
            pass  # keep default hypers if optimisation fails
        model.eval()
        return model

    @staticmethod
    def _posterior_variance(model: SingleTaskGP, X_query: torch.Tensor) -> torch.Tensor:
        """Return posterior standard deviation at *X_query*."""
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            post = model.posterior(X_query.double())
            return post.variance.squeeze(-1).clamp(min=1e-10).sqrt()

    # ------------------------------------------------------------------
    # Kernel-matrix helpers for the rare-switching condition
    # ------------------------------------------------------------------

    @staticmethod
    def _log_det_gram(kernel, X: torch.Tensor, lambda_reg: float) -> float:
        """Compute ``log det(I + lambda^{-1} K)`` for the switching test.

        Uses the kernel extracted from a fitted GP so that length-scale /
        output-scale are consistent with the current model.
        """
        with torch.no_grad():
            K = kernel(X.double()).evaluate()
            n = K.shape[0]
            M = torch.eye(n, dtype=torch.double) + K / lambda_reg
            # Use slogdet for numerical stability
            sign, logdet = torch.linalg.slogdet(M)
            return logdet.item()

    @staticmethod
    def _variance_from_kernel(
        kernel, X_obs: torch.Tensor, X_query: torch.Tensor, lambda_reg: float
    ) -> torch.Tensor:
        """Posterior variance using only *kernel* and observation locations.

        sigma^2(x) = k(x,x) - k_*(x)^T (K + lambda I)^{-1} k_*(x)

        This does not depend on observations, only on input locations.
        """
        with torch.no_grad():
            K = kernel(X_obs).evaluate()
            n = K.shape[0]
            L = K + lambda_reg * torch.eye(n, dtype=torch.double)
            k_star = kernel(X_query, X_obs).evaluate()  # [N_query, n]
            # k_diag = diag(K(X_query, X_query))
            k_diag = kernel(X_query).evaluate().diag()  # [N_query]
            alpha = torch.linalg.solve(L, k_star.T)  # [n, N_query]
            var = k_diag - (k_star * alpha.T).sum(dim=-1)
            return var.clamp(min=1e-10)

    # ------------------------------------------------------------------
    # Robust posterior (Eq. 7 + Eq. 6)
    # ------------------------------------------------------------------

    def _compute_robust_posterior(
        self,
        X_epoch: torch.Tensor,
        Y_epoch: torch.Tensor,
        X_query: torch.Tensor,
    ):
        """Compute robust posterior mean and standard deviation.

        * **Robust mean** (Eq. 7): for each unique action, the corrupted
          observations are averaged first; the GP posterior mean is then
          computed on the (unique-action, averaged-Y) pairs.
        * **Posterior variance** (Eq. 6): computed with all ``u_h`` evaluation
          locations (variance does not depend on observations).

        Returns:
            (mu_robust, sigma) tensors evaluated at *X_query*.
        """
        # --- Average corrupted observations per unique action ---------------
        unique_X, inverse = torch.unique(X_epoch, dim=0, return_inverse=True)
        avg_Y = torch.zeros(len(unique_X), dtype=torch.double)
        counts = torch.zeros(len(unique_X), dtype=torch.double)
        for i in range(len(X_epoch)):
            avg_Y[inverse[i]] += Y_epoch[i]
            counts[inverse[i]] += 1.0
        avg_Y /= counts

        # --- Robust mean from averaged GP -----------------------------------
        if len(unique_X) >= 2:
            gp_mean = self._fit_gp(unique_X, avg_Y)
            with torch.no_grad():
                mu_robust = gp_mean.posterior(X_query.double()).mean.squeeze(-1)
        else:
            mu_robust = avg_Y[0].expand(len(X_query))

        # --- Variance from full epoch data ----------------------------------
        if len(X_epoch) >= 2:
            gp_var = self._fit_gp(X_epoch, Y_epoch)
            sigma = self._posterior_variance(gp_var, X_query)
        else:
            sigma = torch.ones(len(X_query), dtype=torch.double)

        return mu_robust, sigma

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(
        self,
        evaluator,
        search_space,
        total_budget: int,
        initial_results: Optional[List[EvaluationResult]] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Execute RGP-PE for *total_budget* evaluations after the initial points.

        Args:
            evaluator: A ``CorruptedEvaluator`` (or compatible) that returns
                ``EvaluationResult`` objects.
            search_space: ``SearchSpace`` used to decode tensor points into
                parameter dictionaries for the evaluator.
            total_budget: Total number of function evaluations to spend
                (corresponds to ``N_ITERATIONS`` in the experiment scripts).
            initial_results: Evaluation results for the shared initial points.
            verbose: Print epoch-level progress.

        Returns:
            Dictionary in the same format as ``ExperimentRunner.run()`` so
            that existing analysis utilities work unchanged.
        """
        all_results: List[EvaluationResult] = list(initial_results or [])

        # Accumulated data (initial + all epochs)
        if initial_results:
            X_all = torch.stack(
                [
                    search_space.encode_point(r.x) if isinstance(r.x, dict) else r.x
                    for r in initial_results
                ]
            ).double()
            Y_all = torch.tensor(
                [r.y_observed for r in initial_results], dtype=torch.double
            )
        else:
            X_all = torch.empty(0, self.d, dtype=torch.double)
            Y_all = torch.empty(0, dtype=torch.double)

        active_indices = list(range(self.n_domain))
        budget_used = 0
        epoch = 0

        while budget_used < total_budget and len(active_indices) > 1:
            lh = self.steps_per_epoch
            if verbose:
                print(
                    f"  [RGP-PE] Epoch {epoch}: "
                    f"{len(active_indices)} active actions, "
                    f"budget {budget_used}/{total_budget}"
                )

            # ----- fit GP on accumulated data for exploration ---------------
            if len(X_all) >= 2:
                gp_explore = self._fit_gp(X_all, Y_all)
                explore_kernel = gp_explore.covar_module
            else:
                gp_explore = None
                explore_kernel = None

            # Collect exploration-phase observations
            S_h: Dict[int, int] = {}  # domain_index -> selection count
            epoch_explore_X: List[torch.Tensor] = []

            # Rare-switching state
            log_det_at_switch: Optional[float] = None
            sigma_cache: Optional[torch.Tensor] = None

            for t in range(lh):
                if budget_used >= total_budget:
                    break

                active_pts = self.domain_points[active_indices]

                # --- compute / reuse posterior std --------------------------
                if sigma_cache is None:
                    if gp_explore is not None:
                        sigma_cache = self._posterior_variance(gp_explore, active_pts)
                    else:
                        # Prior: uniform variance
                        sigma_cache = torch.ones(len(active_indices), dtype=torch.double)

                # Select action with maximum posterior std
                local_idx = sigma_cache.argmax().item()
                domain_idx = active_indices[local_idx]
                x_sel = self.domain_points[domain_idx : domain_idx + 1]

                # Evaluate
                params = search_space.decode_point(x_sel.squeeze(0))
                result = evaluator.evaluate(params)
                all_results.append(result)
                budget_used += 1

                # Update bookkeeping
                S_h[domain_idx] = S_h.get(domain_idx, 0) + 1
                epoch_explore_X.append(x_sel.squeeze(0))
                X_all = torch.cat([X_all, x_sel])
                Y_all = torch.cat(
                    [Y_all, torch.tensor([result.y_observed], dtype=torch.double)]
                )

                # --- rare-switching condition (Line 6) ----------------------
                if explore_kernel is not None and len(epoch_explore_X) >= 2:
                    X_ep = torch.stack(epoch_explore_X)
                    cur_log_det = self._log_det_gram(
                        explore_kernel, X_ep, self.lambda_reg
                    )
                    if log_det_at_switch is None:
                        log_det_at_switch = cur_log_det
                        sigma_cache = self._variance_from_kernel(
                            explore_kernel, X_ep, active_pts, self.lambda_reg
                        ).sqrt()
                    elif cur_log_det > math.log(self.eta) + log_det_at_switch:
                        log_det_at_switch = cur_log_det
                        sigma_cache = self._variance_from_kernel(
                            explore_kernel, X_ep, active_pts, self.lambda_reg
                        ).sqrt()
                    # else: keep sigma_cache (same action will be picked again)

            if budget_used >= total_budget:
                break
            if not S_h:
                break

            # ----- resampling phase (Lines 11-13) ---------------------------
            resample_X_list: List[torch.Tensor] = []
            resample_Y_list: List[float] = []

            for domain_idx, count in S_h.items():
                xi_h = count / lh
                u_h_x = math.ceil(lh * max(xi_h, self.psi))
                x_pt = self.domain_points[domain_idx : domain_idx + 1]

                for _ in range(u_h_x):
                    if budget_used >= total_budget:
                        break
                    params = search_space.decode_point(x_pt.squeeze(0))
                    result = evaluator.evaluate(params)
                    all_results.append(result)
                    budget_used += 1
                    resample_X_list.append(x_pt.squeeze(0))
                    resample_Y_list.append(result.y_observed)
                if budget_used >= total_budget:
                    break

            if not resample_X_list:
                break

            resample_X = torch.stack(resample_X_list)
            resample_Y = torch.tensor(resample_Y_list, dtype=torch.double)

            # Add resampling data to accumulated set
            X_all = torch.cat([X_all, resample_X])
            Y_all = torch.cat([Y_all, resample_Y])

            # ----- elimination (Line 15) ------------------------------------
            uh = len(resample_Y)
            # Practical confidence width (Section 4.1)
            w = self.beta + self.b * self.C / math.sqrt(max(uh, 1))

            active_pts = self.domain_points[active_indices]
            if len(resample_X) >= 2:
                mu_robust, sigma = self._compute_robust_posterior(
                    resample_X, resample_Y, active_pts
                )
                ucb = mu_robust + w * sigma
                lcb = mu_robust - w * sigma
                max_lcb = lcb.max().item()

                surviving = [
                    idx
                    for i, idx in enumerate(active_indices)
                    if ucb[i].item() >= max_lcb
                ]
                if verbose:
                    n_elim = len(active_indices) - len(surviving)
                    print(
                        f"           Eliminated {n_elim} actions "
                        f"({len(surviving)} remaining), w={w:.3f}"
                    )
                active_indices = surviving if surviving else active_indices

            epoch += 1

        # ----- exploit remaining budget with best surviving action ----------
        if budget_used < total_budget and active_indices:
            if len(active_indices) == 1:
                best_idx = active_indices[0]
            else:
                active_pts = self.domain_points[active_indices]
                if len(X_all) >= 2:
                    gp_final = self._fit_gp(X_all, Y_all)
                    with torch.no_grad():
                        means = gp_final.posterior(active_pts).mean.squeeze(-1)
                    best_idx = active_indices[means.argmax().item()]
                else:
                    best_idx = active_indices[0]

            x_best = self.domain_points[best_idx : best_idx + 1]
            while budget_used < total_budget:
                params = search_space.decode_point(x_best.squeeze(0))
                result = evaluator.evaluate(params)
                all_results.append(result)
                budget_used += 1

        if verbose:
            print(
                f"  [RGP-PE] Done: {budget_used} evaluations, "
                f"{len(active_indices)} surviving actions"
            )

        # ----- package results in ExperimentRunner-compatible format --------
        return self._package_results(
            all_results,
            search_space,
            n_initial=len(initial_results) if initial_results else 0,
            total_budget=total_budget,
        )

    # ------------------------------------------------------------------
    # Result packaging
    # ------------------------------------------------------------------

    @staticmethod
    def _package_results(
        all_results: List[EvaluationResult],
        search_space,
        n_initial: int,
        total_budget: int,
    ) -> Dict[str, Any]:
        """Build a result dict compatible with ``ExperimentRunner.run()``."""
        X = torch.stack(
            [
                search_space.encode_point(r.x) if isinstance(r.x, dict) else r.x
                for r in all_results
            ]
        ).double()
        Y_observed = torch.tensor(
            [r.y_observed for r in all_results], dtype=torch.double
        )
        Y_true = torch.tensor([r.y_true for r in all_results], dtype=torch.double)
        Y_noisy = torch.tensor([r.y_noisy for r in all_results], dtype=torch.double)
        corruption_levels = torch.tensor(
            [r.corruption for r in all_results], dtype=torch.double
        )

        best_obs_info, best_true_info = find_best_points(all_results)

        return {
            "all_results": all_results,
            "X": X,
            "Y_observed": Y_observed,
            "Y_true": Y_true,
            "Y_noisy": Y_noisy,
            "corruption_levels": corruption_levels,
            "final_model": None,  # RGP-PE doesn't keep a persistent model
            "final_acquisition": None,
            "best_observed_value": best_obs_info["value"],
            "best_observed_point": X[best_obs_info["index"]],
            "best_observed_params": best_obs_info["params"],
            "best_true_value": best_true_info["value"],
            "best_true_point": X[best_true_info["index"]],
            "best_true_params": best_true_info["params"],
            "n_iterations": total_budget,
            "n_initial": n_initial,
            "seed": None,
        }
