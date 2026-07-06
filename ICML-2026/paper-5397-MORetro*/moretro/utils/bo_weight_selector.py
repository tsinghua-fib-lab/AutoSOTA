import logging
from typing import Any

import gin
import numpy as np
import torch
from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf_discrete
from gpytorch.constraints import Interval
from gpytorch.mlls import ExactMarginalLogLikelihood
from pymoo.indicators.hv import Hypervolume
from scipy.spatial.distance import cdist

from moretro.utils.base_paths import LOG_DIR

# optional plotting
plt = None
Axes3D = None
try:
    import matplotlib.pyplot as plt  # type: ignore
    from mpl_toolkits.mplot3d import Axes3D  # type: ignore  # noqa: F401

    PLOTTING_AVAILABLE = True
except Exception:
    PLOTTING_AVAILABLE = False

logger = logging.getLogger(__name__)


@gin.configurable
class BOWeightSelector:
    """
    Bayesian Optimization for weight selection in Multi-Objective Search.

    Attributes
    ----------
    n_obj : int
        Number of objectives.
    seed : int
        Random seed for reproducibility.
    kappa : float
        Exploration-exploitation trade-off parameter for UCB (used in visualization).
    n_warmup : int
        Number of initial samples before starting BO.
    decay_factor : float
        Decay factor for utility of older weights to encourage exploration.
    max_age : int
        Maximum age of weights to consider for utility decay.
    """

    def __init__(
        self,
        n_obj: int,
        seed: int = 42,
        kappa: float = 2.0,
        n_warmup: int = 15,
        decay_factor: float = 0.75,
        max_age: int = 2,
    ):
        self.n_obj = n_obj
        self.rng = np.random.default_rng(seed=seed)
        torch.manual_seed(seed)
        self.kappa = kappa
        self.n_warmup = n_warmup
        self.decay_factor = decay_factor
        self.max_age = max_age
        self.weights_history: list[np.ndarray] = []
        self.utilities_history: list[float] = []
        self.batch_ids: list[int] = []
        self.current_batch_id: int = 0

    def add_batch(self, weights: np.ndarray) -> None:
        """Add a new batch of weights to the history."""
        if len(self.weights_history) < self.n_warmup:
            # Warmup phase: all weights get the same batch ID (0)
            batch_id = 0
        else:
            self.current_batch_id += 1
            batch_id = self.current_batch_id

        # reversing weights for correct utility assignment order
        for w in weights[::-1]:
            self.weights_history.insert(0, np.array(w))
            self.utilities_history.insert(0, 0.0)
            self.batch_ids.insert(0, batch_id)

    def compute_hypervolume(
        self, front: np.ndarray, ref_point: np.ndarray | None = None
    ) -> float:
        """
        Compute hypervolume for a minimization problem using pymoo.

        front: array shape (n_points, n_obj). Assumes smaller is better.
        ref_point: Optional reference point. If None, calculated from front.
        """
        if front.size == 0:
            return 0.0
        # Convert to 2D
        front = np.array(front, dtype=float)
        # remove duplicates
        front = np.unique(front, axis=0)

        if ref_point is None:
            # ref point: slightly worse than worst observed per objective
            ref = np.max(front, axis=0) + 1.0
        else:
            ref = ref_point

        front = front[np.all(front <= ref, axis=1)]

        ind = Hypervolume(ref_point=ref)
        hv_value = ind(front)
        if hv_value is None:
            return 0.0
        return float(hv_value)

    def select_next_weights(
        self, weights_open: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Select next k weights from weights_open using a GP surrogate trained
        on (self.weights_history, self.utilities_history).

        Returns:
            selected_weights: The selected weight vectors.
            remaining_weights_open: The updated pool of open weights (with selected removed).
        """
        # If there are fewer remaining weights than k, just take what's available
        if len(weights_open) <= k:
            return weights_open, np.empty((0, weights_open.shape[1]))

        # Warm-up phase: if we haven't collected enough samples, stick to the initial order (Queue/Grid)
        if len(self.weights_history) < self.n_warmup:
            selected = weights_open[:k]
            remaining = weights_open[k:]
            return selected, remaining

        train_X, train_Y = self._prepare_gp_data()

        # Filter candidates: remove those already in history
        dists_to_hist = cdist(weights_open, train_X.numpy(), metric="euclidean")
        min_dist_to_hist = dists_to_hist.min(axis=1)

        # Mask candidates that are too close to history (effectively already sampled)
        valid_mask = min_dist_to_hist > 1e-3

        if valid_mask.sum() == 0:
            logger.warning("All candidates are in history. Re-evaluating duplicates.")
            valid_candidates = weights_open
        else:
            valid_candidates = weights_open[valid_mask]

        try:
            # Fit GP model
            gp = SingleTaskGP(train_X, train_Y)

            if hasattr(gp.covar_module, "base_kernel"):
                gp.covar_module.base_kernel.lengthscale_constraint = Interval(0.05, 0.5)
            else:
                gp.covar_module.lengthscale_constraint = Interval(0.05, 0.5)

            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)

            # Optimize over discrete choices
            choices = torch.tensor(valid_candidates, dtype=torch.double, device="cpu")

            # Use GIBBON (qLowerBoundMaxValueEntropy) for diversity
            # It requires a candidate set to estimate the max value distribution
            qGIBBON = qLowerBoundMaxValueEntropy(model=gp, candidate_set=choices)

            # Select k candidates
            q_batch = min(k, len(valid_candidates))

            candidates, _ = optimize_acqf_discrete(
                acq_function=qGIBBON,
                q=q_batch,
                choices=choices,
                unique=True,  # Ensure we don't pick the same point twice
            )

            selected = candidates.detach().cpu().numpy()

            selected_indices = []
            for sel_w in selected:
                dists = np.linalg.norm(weights_open - sel_w, axis=1)
                idx = np.argmin(dists)
                selected_indices.append(idx)

            selected_indices = np.array(selected_indices)

            mask_selected = np.zeros(len(weights_open), dtype=bool)
            mask_selected[selected_indices] = True

            selected = weights_open[mask_selected]
            remaining = weights_open[~mask_selected]

            # --- Visualization ---
            # Compute acquisition function values for visualization
            if self.n_obj == 3 and PLOTTING_AVAILABLE:
                with torch.no_grad():
                    acq_values = np.zeros(len(choices))
                    for i, x in enumerate(choices):
                        acq_values[i] = qGIBBON(x.unsqueeze(0)).item()

                self._plot_selection(valid_candidates, acq_values, selected)

            return selected, remaining

        except (RuntimeError, ValueError) as e:
            logger.warning(
                f"BO weight selection failed ({e}), falling back to simple pop."
            )
            return weights_open[:k], weights_open[k:]
        except Exception as e:
            logger.exception("Unexpected error during BO weight selection")
            raise e

    def _plot_selection(
        self, candidates: np.ndarray, acq_values: np.ndarray, selected: np.ndarray
    ) -> None:
        """
        Plot the BO selection process: history, candidates (colored by acquisition function), and selected weights.
        Saves the plot to logs/bo_select_plot_{N}.png.
        """
        if not PLOTTING_AVAILABLE or self.n_obj != 3 or plt is None or Axes3D is None:
            return

        try:
            X_hist = np.array(self.weights_history)
            y_hist = np.array(self.utilities_history)

            # Apply decay to y_hist for visualization
            if self.batch_ids and len(self.batch_ids) == len(y_hist):
                batch_ids = np.array(self.batch_ids)
                ages = self.current_batch_id - batch_ids
                y_hist = np.where(ages > self.max_age, 0.0, y_hist)
                y_hist *= self.decay_factor**ages

            fig = plt.figure(figsize=(12, 5))
            ax: Any = fig.add_subplot(1, 1, 1, projection="3d")

            # history points colored by observed (raw) utility
            if len(X_hist):
                sc_hist = ax.scatter(
                    X_hist[:, 0],
                    X_hist[:, 1],
                    X_hist[:, 2],
                    s=80,  # marker size
                    c=y_hist,
                    cmap="viridis",
                    edgecolor="k",
                    label="history (IG)",
                )
                fig.colorbar(sc_hist, ax=ax, shrink=0.6, label="Observed IG")

            # open candidates colored by acquisition function value
            ax.scatter(
                candidates[:, 0],
                candidates[:, 1],
                candidates[:, 2],
                s=20,
                c=acq_values,
                cmap="viridis",
                alpha=0.7,
            )

            # selected weights overlay
            if len(selected):
                sel = np.array(selected)
                ax.scatter(
                    sel[:, 0],
                    sel[:, 1],
                    sel[:, 2],
                    s=250,
                    facecolors="none",
                    edgecolors="red",
                    linewidths=2.5,
                    marker="o",
                    label="selected",
                )

            ax.set_xlabel("w1")
            ax.set_ylabel("w2")
            ax.set_zlabel("w3")
            ax.view_init(elev=30, azim=45)
            out_dir = LOG_DIR / "weight_selection"
            out_dir.mkdir(exist_ok=True, parents=True)
            out_file = out_dir / f"bo_weights_{len(self.weights_history)}.png"
            fig.tight_layout()
            fig.savefig(out_file, dpi=150)
            plt.close(fig)
        except Exception:
            logger.debug("Plotting selection results failed")

    def process_pareto_update(
        self,
        old_pareto_costs: np.ndarray,
        new_pareto_costs: np.ndarray,
        contributing_indices: set[int],
    ) -> None:
        """
        Calculate Hypervolume improvement and distribute it to contributing weights.
        Updates utilities_history in-place.
        """
        try:
            # Determine a common reference point for valid comparison
            if old_pareto_costs.size > 0 and new_pareto_costs.size > 0:
                all_costs = np.vstack([old_pareto_costs, new_pareto_costs])
                ref = np.max(all_costs, axis=0) + 1.0
            elif new_pareto_costs.size > 0:
                ref = np.max(new_pareto_costs, axis=0) + 1.0
            else:
                # No data
                return

            hv_old = (
                self.compute_hypervolume(old_pareto_costs, ref_point=ref)
                if old_pareto_costs.size
                else 0.0
            )
            hv_new = (
                self.compute_hypervolume(new_pareto_costs, ref_point=ref)
                if new_pareto_costs.size
                else 0.0
            )

            if hv_old == 0.0:
                delta_hv = min(hv_new, 1.0)  # cap initial HV gain
            else:
                delta_hv = max(0.0, hv_new - hv_old)
        except Exception as e:
            logger.warning(f"Hypervolume computation failed: {e}. Setting delta_hv=0.")
            return

        if delta_hv > 0 and contributing_indices:
            share = delta_hv / len(contributing_indices)
            for idx in contributing_indices:
                if idx < len(self.utilities_history):
                    self.utilities_history[idx] += share

    def get_max_ucb(self, weights_open: np.ndarray) -> float:
        """
        Calculate the maximum UCB value among the candidate weights.
        Returns a high value if history is empty (infinite exploration).
        """
        if len(self.weights_history) < self.n_warmup:
            return float("inf")

        if len(weights_open) == 0:
            return -float("inf")

        dtype = torch.double
        device = torch.device("cpu")

        train_X, train_Y = self._prepare_gp_data()

        try:
            gp = SingleTaskGP(train_X, train_Y)

            if hasattr(gp.covar_module, "base_kernel"):
                gp.covar_module.base_kernel.lengthscale_constraint = Interval(0.05, 0.5)
            else:
                gp.covar_module.lengthscale_constraint = Interval(0.05, 0.5)

            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)

            candidates = torch.tensor(weights_open, dtype=dtype, device=device)

            with torch.no_grad():
                posterior = gp.posterior(candidates)
                mean = posterior.mean.squeeze()
                std = posterior.variance.sqrt().squeeze()
                ucb = mean + self.kappa * std
                return float(ucb.max().item())

        except Exception as e:
            logger.warning(f"Failed to compute max UCB: {e}")
            return float("inf")

    def _prepare_gp_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare training data for the GP model, including applying time decay to utilities.
        """
        dtype = torch.double
        device = torch.device("cpu")

        X_np = np.array(self.weights_history)
        y_np = np.array(self.utilities_history)

        # Apply time decay to utilities
        if self.batch_ids and len(self.batch_ids) == len(y_np):
            batch_ids = np.array(self.batch_ids)
            ages = self.current_batch_id - batch_ids
            # if ages > max_age, set y_np to zero
            y_np = np.where(ages > self.max_age, 0.0, y_np)
            # Apply decay to raw utilities
            y_np *= self.decay_factor**ages

        # Log-transform utilities
        y_np = np.log1p(y_np)

        train_X = torch.tensor(X_np, dtype=dtype, device=device)
        train_Y = torch.tensor(y_np, dtype=dtype, device=device).unsqueeze(-1)

        return train_X, train_Y
