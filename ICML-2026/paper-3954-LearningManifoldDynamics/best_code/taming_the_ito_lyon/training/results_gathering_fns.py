import matplotlib

matplotlib.use("Agg")
from taming_the_ito_lyon.utils.mpl_style import apply_mpl_style
from taming_the_ito_lyon.training.results_plotting import (
    save_rough_volatility_two_panel_plot,
    save_rough_volatility_fan_plot,
    save_sg_so3_sphere_plot,
    save_spd_covariance_eigenvalue_trajectory_single_plot,
    save_spd_covariance_eigenvalue_fan_single_plot,
)
from taming_the_ito_lyon.training.losses import (
    frobenius_loss,
    rotational_geodesic_loss,
)
from taming_the_ito_lyon.config.config import Config
from taming_the_ito_lyon.config.config_options import LossType, Datasets
import jax
import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


apply_mpl_style()
DEFAULT_RESULTS_TIMES = (128, 256, 384, 512)


def _extract_first_channel_paths(x: jax.Array) -> np.ndarray:
    """Extract (B, T) array from a model output/target tensor.

    We flatten any trailing dims and keep channel 0. This matches the rough-vol
    dataset behavior where the training pipeline keeps only the first channel.
    """
    x_np = np.array(jax.device_get(x))
    x_flat = x_np.reshape(x_np.shape[0], x_np.shape[1], -1)  # (B, T, Cflat)
    return x_flat[:, :, 0]  # (B, T)


def _valid_times_to_save(
    times_to_save: list[int] | None,
    *batch_groups: list[jax.Array],
) -> list[int]:
    if times_to_save is None:
        return []
    common_horizon = min(
        int(np.array(jax.device_get(batch)).shape[1])
        for batches in batch_groups
        for batch in batches
    )
    return [int(t) for t in times_to_save if int(t) < common_horizon]


def _ito_level12_feature_vectors(
    *,
    w_paths: np.ndarray,
    x_paths: np.ndarray,
    include_level1: bool = True,
) -> np.ndarray:
    """Compute per-path Itô/branched level-1/2 features from (W, X).

    Args:
        w_paths: (B, T) latent/driver BM paths.
        x_paths: (B, T) output log-price paths.
        include_level1: If True, include sum(dW) and sum(dX) as features.

    Returns:
        (B, D) feature matrix with columns:
            [qvW, qvX, covWX, (optional: sum_dW, sum_dX)]
    """
    if w_paths.ndim != 2 or x_paths.ndim != 2:
        raise ValueError(
            f"Expected w_paths/x_paths shaped (B, T), got {w_paths.shape} and {x_paths.shape}"
        )
    if w_paths.shape != x_paths.shape:
        raise ValueError(
            f"w_paths and x_paths must have the same shape, got {w_paths.shape} and {x_paths.shape}"
        )
    if int(w_paths.shape[1]) < 2:
        b = int(w_paths.shape[0])
        d = 5 if include_level1 else 3
        return np.zeros((b, d), dtype=np.float32)

    dW = np.diff(w_paths, axis=1)  # (B, T-1)
    dX = np.diff(x_paths, axis=1)  # (B, T-1)

    qvW = np.sum(dW * dW, axis=1, dtype=np.float64)  # (B,)
    qvX = np.sum(dX * dX, axis=1, dtype=np.float64)  # (B,)
    covWX = np.sum(dW * dX, axis=1, dtype=np.float64)  # (B,)

    if include_level1:
        sum_dW = np.sum(dW, axis=1, dtype=np.float64)
        sum_dX = np.sum(dX, axis=1, dtype=np.float64)
        feat = np.stack([qvW, qvX, covWX, sum_dW, sum_dX], axis=1)
    else:
        feat = np.stack([qvW, qvX, covWX], axis=1)

    return feat.astype(np.float32)


def _cumulative_ito_signature_feature_vectors(
    *,
    w_paths: np.ndarray,
    x_paths: np.ndarray,
    num_eval_times: int = 4,
) -> np.ndarray:
    """Compute joint (W, X) Itô features at several cumulative times.

    The final coordinate at each time is X_t + 0.5 [X]_t. For simple Bergomi
    log-prices this is the drift-corrected Itô martingale coordinate.
    """
    if w_paths.ndim != 2 or x_paths.ndim != 2:
        raise ValueError(
            f"Expected w_paths/x_paths shaped (B, T), got {w_paths.shape} and {x_paths.shape}"
        )
    if w_paths.shape != x_paths.shape:
        raise ValueError(
            f"w_paths and x_paths must have the same shape, got {w_paths.shape} and {x_paths.shape}"
        )
    b, t = int(w_paths.shape[0]), int(w_paths.shape[1])
    if t < 2:
        return np.zeros((b, 0), dtype=np.float32)

    dW = np.diff(w_paths, axis=1)
    dX = np.diff(x_paths, axis=1)
    qvW = np.cumsum(dW * dW, axis=1, dtype=np.float64)
    qvX = np.cumsum(dX * dX, axis=1, dtype=np.float64)
    covWX = np.cumsum(dW * dX, axis=1, dtype=np.float64)
    w_rel = w_paths[:, 1:] - w_paths[:, :1]
    x_rel = x_paths[:, 1:] - x_paths[:, :1]
    ito_mart = x_rel + 0.5 * qvX

    n_steps = int(dW.shape[1])
    n_eval = min(max(int(num_eval_times), 1), n_steps)
    eval_idx = np.unique(np.linspace(0, n_steps - 1, n_eval, dtype=np.int64))
    blocks = [
        w_rel[:, eval_idx],
        x_rel[:, eval_idx],
        qvW[:, eval_idx],
        qvX[:, eval_idx],
        covWX[:, eval_idx],
        ito_mart[:, eval_idx],
    ]
    return np.concatenate(blocks, axis=1).astype(np.float32)


def _ito_log_martingale_mean_l2(
    *,
    x_paths: np.ndarray,
    num_eval_times: int = 16,
) -> float:
    """L2 size of E[X_t + 0.5 [X]_t] across evaluation times."""
    if x_paths.ndim != 2:
        raise ValueError(f"Expected x_paths shaped (B, T), got {x_paths.shape}")
    if int(x_paths.shape[1]) < 2:
        return 0.0

    dX = np.diff(x_paths, axis=1)
    qvX = np.cumsum(dX * dX, axis=1, dtype=np.float64)
    x_rel = x_paths[:, 1:] - x_paths[:, :1]
    ito_mart = x_rel + 0.5 * qvX
    n_steps = int(dX.shape[1])
    n_eval = min(max(int(num_eval_times), 1), n_steps)
    eval_idx = np.unique(np.linspace(0, n_steps - 1, n_eval, dtype=np.int64))
    mean_curve = np.mean(ito_mart[:, eval_idx], axis=0, dtype=np.float64)
    return float(np.sqrt(np.mean(mean_curve * mean_curve)))


def _ito_driver_output_summary(
    *,
    w_paths: np.ndarray,
    x_paths: np.ndarray,
) -> dict[str, float]:
    """Interpretable realized Itô statistics for a batch of (W, X) paths."""
    if w_paths.ndim != 2 or x_paths.ndim != 2:
        raise ValueError(
            f"Expected w_paths/x_paths shaped (B, T), got {w_paths.shape} and {x_paths.shape}"
        )
    if w_paths.shape != x_paths.shape:
        raise ValueError(
            f"w_paths and x_paths must have the same shape, got {w_paths.shape} and {x_paths.shape}"
        )
    if int(w_paths.shape[1]) < 2:
        return {
            "qv_w_mean": 0.0,
            "qv_x_mean": 0.0,
            "cov_wx_mean": 0.0,
            "beta_mean": 0.0,
            "driver_corr": 0.0,
        }

    dW = np.diff(w_paths, axis=1)
    dX = np.diff(x_paths, axis=1)
    qv_w = np.sum(dW * dW, axis=1, dtype=np.float64)
    qv_x = np.sum(dX * dX, axis=1, dtype=np.float64)
    cov_wx = np.sum(dW * dX, axis=1, dtype=np.float64)
    beta = cov_wx / (qv_w + 1e-12)
    dW_flat = dW.reshape(-1).astype(np.float64, copy=False)
    dX_flat = dX.reshape(-1).astype(np.float64, copy=False)
    if float(np.std(dW_flat)) <= 0.0 or float(np.std(dX_flat)) <= 0.0:
        corr = 0.0
    else:
        corr = float(np.corrcoef(dW_flat, dX_flat)[0, 1])

    return {
        "qv_w_mean": float(np.mean(qv_w)),
        "qv_x_mean": float(np.mean(qv_x)),
        "cov_wx_mean": float(np.mean(cov_wx)),
        "beta_mean": float(np.mean(beta)),
        "driver_corr": corr,
    }


def _local_ito_projection_feature_vectors(
    *,
    w_paths: np.ndarray,
    x_paths: np.ndarray,
    block_size: int = 16,
    eps: float = 1e-12,
) -> np.ndarray:
    """Local drift-corrected residual after projecting onto the Brownian driver.

    For each block, fit the local Itô integrand through realized covariation,
    beta = Δ[X,W] / Δ[W], and record the residual of
    ΔX + 0.5 Δ[X] ≈ beta ΔW.
    """
    if w_paths.ndim != 2 or x_paths.ndim != 2:
        raise ValueError(
            f"Expected w_paths/x_paths shaped (B, T), got {w_paths.shape} and {x_paths.shape}"
        )
    if w_paths.shape != x_paths.shape:
        raise ValueError(
            f"w_paths and x_paths must have the same shape, got {w_paths.shape} and {x_paths.shape}"
        )
    b, t = int(w_paths.shape[0]), int(w_paths.shape[1])
    if t < 2:
        return np.zeros((b, 5), dtype=np.float32)

    dW = np.diff(w_paths, axis=1)
    dX = np.diff(x_paths, axis=1)
    block = max(int(block_size), 1)
    n_blocks = int(dW.shape[1]) // block
    if n_blocks < 1:
        block = int(dW.shape[1])
        n_blocks = 1
    n = n_blocks * block
    dWb = dW[:, :n].reshape(b, n_blocks, block)
    dXb = dX[:, :n].reshape(b, n_blocks, block)

    dW_win = np.sum(dWb, axis=2, dtype=np.float64)
    dX_win = np.sum(dXb, axis=2, dtype=np.float64)
    qvW_win = np.sum(dWb * dWb, axis=2, dtype=np.float64)
    qvX_win = np.sum(dXb * dXb, axis=2, dtype=np.float64)
    covWX_win = np.sum(dWb * dXb, axis=2, dtype=np.float64)
    beta = covWX_win / (qvW_win + float(eps))
    residual = dX_win + 0.5 * qvX_win - beta * dW_win
    normalized = residual / np.sqrt(qvX_win + float(eps))

    feat = np.stack(
        [
            np.mean(residual, axis=1, dtype=np.float64),
            np.sqrt(np.mean(residual * residual, axis=1, dtype=np.float64)),
            np.sum(residual, axis=1, dtype=np.float64),
            np.mean(np.abs(normalized), axis=1, dtype=np.float64),
            np.sqrt(np.mean(normalized * normalized, axis=1, dtype=np.float64)),
        ],
        axis=1,
    )
    return feat.astype(np.float32)


def _mmd_rbf_median_heuristic(
    x: np.ndarray,
    y: np.ndarray,
    *,
    eps: float = 1e-12,
    max_bandwidth_points: int = 1024,
) -> float:
    """Biased RBF-kernel MMD^2 with median-heuristic bandwidth.

    MMD^2 = mean(Kxx) + mean(Kyy) - 2*mean(Kxy)
    where K(u,v) = exp(-||u-v||^2 / (2*h^2)).
    """
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"Expected 2D feature matrices, got {x.shape} and {y.shape}")
    if int(x.shape[1]) != int(y.shape[1]):
        raise ValueError(
            f"x and y must have the same feature dimension, got {x.shape} and {y.shape}"
        )
    nx = int(x.shape[0])
    ny = int(y.shape[0])
    if nx == 0 or ny == 0:
        return 0.0

    x64 = x.astype(np.float64, copy=False)
    y64 = y.astype(np.float64, copy=False)

    pooled = np.concatenate([x64, y64], axis=0)
    m = min(int(max_bandwidth_points), int(pooled.shape[0]))
    pooled_m = pooled[:m]

    diffs = pooled_m[:, None, :] - pooled_m[None, :, :]
    dist2 = np.sum(diffs * diffs, axis=-1)  # (m, m)
    off = dist2[~np.eye(m, dtype=bool)]
    med = float(np.median(off)) if off.size > 0 else float(np.median(dist2))
    if not np.isfinite(med) or med <= 0.0:
        med = 1.0
    denom = 2.0 * med + float(eps)

    def k(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d = a[:, None, :] - b[None, :, :]
        d2 = np.sum(d * d, axis=-1)
        return np.exp(-d2 / denom)

    k_xx = k(x64, x64)
    k_yy = k(y64, y64)
    k_xy = k(x64, y64)
    mmd2 = float(np.mean(k_xx) + np.mean(k_yy) - 2.0 * np.mean(k_xy))
    return mmd2


def _make_branched_ito_signature_moment_gap_metrics(
    *,
    depth: int = 2,
) -> Callable[..., dict[str, float]]:
    """Return a metric fn that reuses a cached GL Hopf algebra."""
    from stochastax.hopf_algebras import GLHopfAlgebra

    hopf = GLHopfAlgebra.build(ambient_dim=2, depth=int(depth))
    if hopf.degree2_chain_indices is None:
        raise ValueError("GLHopfAlgebra missing degree-2 chain indices.")

    def _branched_ito_signature_moment_gap_metrics(
        *,
        pred_paths: np.ndarray,
        target_paths: np.ndarray,
        max_paths: int = 16,
    ) -> dict[str, float]:
        """Compute low-order branched Itô signature moment gaps via true signatures.

        We compute the depth-2 nonplanar (GL/BCK) branched Itô signature on the
        time-augmented path (t, y), then compare mean level-1, level-2, and the
        degree-2 chain terms across predicted vs. target paths.
        """

        if pred_paths.ndim != 2 or target_paths.ndim != 2:
            raise ValueError(
                f"Expected (B, T) pred/target paths, got {pred_paths.shape} and {target_paths.shape}"
            )
        if pred_paths.shape[1] != target_paths.shape[1]:
            raise ValueError(
                f"Pred/target must have same length, got {pred_paths.shape} and {target_paths.shape}"
            )

        Bp, T = pred_paths.shape
        Bt, _ = target_paths.shape
        if T < 2 or Bp < 1 or Bt < 1:
            return {
                "branched_sig_gap_lvl1_l2": 0.0,
                "branched_sig_gap_chain_l2": 0.0,
                "branched_sig_gap_lvl2_full_l2": 0.0,
            }

        n_pred = min(int(max_paths), int(Bp))
        n_target = min(int(max_paths), int(Bt))

        pred = jax.device_put(pred_paths[:n_pred].astype(np.float32))
        target = jax.device_put(target_paths[:n_target].astype(np.float32))

        from stochastax.control_lifts import compute_nonplanar_branched_signature
        import jax.numpy as jnp

        t = jnp.linspace(0.0, 1.0, int(T), dtype=pred.dtype)  # (T,)

        def _signature_stats(y: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
            path = jnp.stack([t, y], axis=1)  # (T, 2)
            increments = path[1:, :] - path[:-1, :]
            dy = increments[:, 1]
            cov = jnp.zeros((T - 1, 2, 2), dtype=path.dtype)
            cov = cov.at[:, 1, 1].set(dy**2)
            sig = compute_nonplanar_branched_signature(
                path=path,
                depth=int(depth),
                hopf=hopf,
                mode="full",
                cov_increments=cov,
            )
            lvl1 = sig.coeffs[0]
            lvl2 = sig.coeffs[1]
            chain = lvl2[hopf.degree2_chain_indices]
            return lvl1, lvl2, chain

        pred_lvl1, pred_lvl2, pred_chain = jax.vmap(_signature_stats)(pred)
        target_lvl1, target_lvl2, target_chain = jax.vmap(_signature_stats)(target)

        def _gap(a: jax.Array, b: jax.Array) -> float:
            return float(jnp.linalg.norm(jnp.mean(a, axis=0) - jnp.mean(b, axis=0)))

        lvl1_gap = _gap(pred_lvl1, target_lvl1)
        chain_gap = _gap(pred_chain, target_chain)
        lvl2_full_gap = _gap(pred_lvl2, target_lvl2)

        return {
            "branched_sig_gap_lvl1_l2": lvl1_gap,
            "branched_sig_gap_chain_l2": chain_gap,
            "branched_sig_gap_lvl2_full_l2": lvl2_full_gap,
        }

    return _branched_ito_signature_moment_gap_metrics


_branched_ito_signature_moment_gap_metrics = (
    _make_branched_ito_signature_moment_gap_metrics(depth=2)
)


def _driver_output_consistency_metrics(
    *,
    control_values: np.ndarray,
    price_paths: np.ndarray,
    eps: float = 1e-8,
    ridge: float = 1e-6,
    max_paths: int = 1028,
) -> dict[str, float]:
    """Driver→output consistency diagnostic for unconditional generation.

    Uses the sampled control (time + driver channels) that generated `price_paths`
    and checks the Itô drift relation implied by regressing Δlog S onto ΔX.

    This is *pred-only* by design: in unconditional mode the dataset targets are
    not paired with the sampled driver used for generation.
    """
    if control_values.ndim != 3:
        raise ValueError(
            f"Expected control_values shaped (B, T, C), got {control_values.shape}"
        )
    if price_paths.ndim != 2:
        raise ValueError(f"Expected price_paths shaped (B, T), got {price_paths.shape}")

    Bc, T, C = control_values.shape
    Bp, Tp = price_paths.shape
    if T != Tp or Bc != Bp:
        raise ValueError(
            f"control_values and price_paths must align, got {control_values.shape} and {price_paths.shape}"
        )
    if T < 2 or C < 2:
        return {
            "driver_consistency_nonpositive_frac": float(np.mean(price_paths <= 0.0)),
            "driver_consistency_drift_defect_mean_abs": 0.0,
            "driver_consistency_beta_rms": 0.0,
        }

    n = min(int(max_paths), int(Bc))
    x = control_values[:n]
    s = price_paths[:n]

    nonpos_frac = float(np.mean(s <= 0.0))
    s_safe = np.clip(s, eps, None)
    log_s = np.log(s_safe)  # (n, T)
    dlog_s = np.diff(log_s, axis=1)  # (n, T-1)

    t = x[0, :, 0]
    dt = float((t[-1] - t[0]) / float(T - 1))
    if dt <= 0.0:
        dt = 1.0 / float(T - 1)

    dX = np.diff(x[:, :, -1:], axis=1)  # (n, T-1, D=1)
    D = int(dX.shape[-1])

    drift_defects: list[float] = []
    betas: list[float] = []
    eye = np.eye(D, dtype=np.float32)
    for k in range(T - 1):
        dx = dX[:, k, :]  # (n, D)
        dy = dlog_s[:, k]  # (n,)
        cov = (dx.T @ dx) / float(n)  # (D, D)
        cross = (dx.T @ dy) / float(n)  # (D,)
        beta = np.linalg.solve(cov + float(ridge) * eye, cross)  # (D,)
        drift = float(np.mean(dy)) / float(dt)
        defect = drift + 0.5 * float(np.sum(beta**2))
        drift_defects.append(abs(defect))
        betas.append(float(np.sqrt(np.mean(beta**2))))

    return {
        "driver_consistency_nonpositive_frac": nonpos_frac,
        "driver_consistency_drift_defect_mean_abs": float(np.mean(drift_defects)),
        "driver_consistency_beta_rms": float(np.mean(betas)),
    }


@dataclass
class ResultsDict:
    eval_metric: float | None
    results_times: list[float]
    results: list[float]
    extra_metrics: dict[str, list[float]] | None = None
    extra_scalar_metrics: dict[str, float] | None = None


class ResultsGatheringFn(Protocol):
    def __call__(
        self,
        preds: list[jax.Array] | jax.Array,
        targets: list[jax.Array] | jax.Array,
        controls: list[jax.Array] | jax.Array | None,
        epoch_idx: int,
        model_name: str,
        times_to_save: list[int] | None = None,
        n_plot: int | None = None,
        save_plot_every: int = 1,
        config: Config | None = None,
    ) -> ResultsDict: ...


def get_ppg_dalia_results(
    preds: list[jax.Array] | jax.Array,
    targets: list[jax.Array] | jax.Array,
    controls: list[jax.Array] | jax.Array | None,
    epoch_idx: int,
    model_name: str,
    times_to_save: list[int] | None = None,
    n_plot: int | None = None,
    save_plot_every: int = 1,
    config: Config | None = None,
) -> ResultsDict:
    """PPG-DaLiA uses loss-space MSE for model selection and reporting.

    Returning ``eval_metric=None`` ensures the training loop falls back to the
    already-computed validation/test loss instead of injecting an auxiliary
    dataset-specific metric.
    """
    return ResultsDict(eval_metric=None, results_times=[], results=[])


def get_generic_path_results(
    preds: list[jax.Array] | jax.Array,
    targets: list[jax.Array] | jax.Array,
    controls: list[jax.Array] | jax.Array | None,
    epoch_idx: int,
    model_name: str,
    times_to_save: list[int] | None = None,
    n_plot: int | None = None,
    save_plot_every: int = 1,
    config: Config | None = None,
) -> ResultsDict:
    return ResultsDict(eval_metric=None, results_times=[], results=[])


def get_rough_volatility_results(
    preds: list[jax.Array] | jax.Array,
    targets: list[jax.Array] | jax.Array,
    controls: list[jax.Array] | jax.Array | None,
    epoch_idx: int,
    model_name: str,
    times_to_save: list[int] | None = list(DEFAULT_RESULTS_TIMES),
    n_plot: int | None = None,
    save_plot_every: int = 1,
    config: Config | None = None,
) -> ResultsDict:
    preds_batches = preds if isinstance(preds, list) else [preds]
    targets_batches = targets if isinstance(targets, list) else [targets]
    controls_batches = (
        None
        if controls is None
        else (controls if isinstance(controls, list) else [controls])
    )
    preds0 = np.array(jax.device_get(preds_batches[0]))
    targets0 = np.array(jax.device_get(targets_batches[0]))
    preds0_flat = preds0.reshape(preds0.shape[0], preds0.shape[1], -1)
    targets0_flat = targets0.reshape(targets0.shape[0], targets0.shape[1], -1)
    if n_plot is not None and epoch_idx % save_plot_every == 0:
        n_plot0 = min(int(n_plot), int(targets0_flat.shape[0]))
        out_dir = os.path.join(
            "z_paper_content", "rough_volatility_samples_by_epoch", model_name
        )
        os.makedirs(out_dir, exist_ok=True)
        save_rough_volatility_two_panel_plot(
            left=targets0_flat,
            right=preds0_flat,
            out_file=os.path.join(out_dir, f"batch_plot_epoch_{epoch_idx}.png"),
            n_plot=n_plot0,
            left_title="Targets (one batch)",
            right_title="Preds (one batch)",
        )
        fan_base_dir = os.environ.get(
            "ROUGH_VOL_FAN_PLOT_DIR", "z_paper_content/rough_volatility_fan_by_epoch"
        )
        fan_out_dir = os.path.join(fan_base_dir, model_name)
        os.makedirs(fan_out_dir, exist_ok=True)
        max_paths_env = os.environ.get("ROUGH_VOL_FAN_MAX_PATHS", "")
        max_paths = int(max_paths_env) if max_paths_env.isdigit() else None
        preds_label = os.environ.get("ROUGH_VOL_FAN_PREDS_LABEL", "Preds")
        save_rough_volatility_fan_plot(
            targets=targets0_flat,
            preds=preds0_flat,
            out_file=os.path.join(fan_out_dir, f"fan_epoch_{epoch_idx:05d}.png"),
            max_paths=max_paths,
            targets_label="Targets",
            preds_label=preds_label,
            targets_color="0.25",
            preds_color="tab:orange",
        )

    valid_times = _valid_times_to_save(times_to_save, preds_batches, targets_batches)
    if len(valid_times) == 0:
        return ResultsDict(eval_metric=None, results_times=[], results=[])

    ks_stats: list[float] = []
    for t in valid_times:
        ks_per_batch = []
        for pred_batch, target_batch in zip(preds_batches, targets_batches):
            pred_samples = np.array(jax.device_get(pred_batch))[:, t, ...].ravel()
            target_samples = np.array(jax.device_get(target_batch))[:, t, ...].ravel()
            ks_res, _ = ks_2samp(pred_samples, target_samples)  # type: ignore[arg-type]
            ks_per_batch.append(float(ks_res))
        ks_stats.append(float(np.mean(ks_per_batch)))

    if (
        config is not None
        and config.experiment_config.dataset_name == Datasets.SIMPLE_RBERGOMI
        and controls_batches is not None
    ):
        x_model = _extract_first_channel_paths(preds_batches[0])
        control_np0 = np.array(jax.device_get(controls_batches[0]))
        if control_np0.ndim != 3:
            raise ValueError(
                f"Expected model controls shaped (B, T, C), got {control_np0.shape}"
            )
        w_model = (
            control_np0[:, :, 1]
            if int(control_np0.shape[-1]) >= 2
            else control_np0[:, :, 0]
        )
        data = np.load(config.experiment_config.dataset_name.value)
        w_gt_raw = np.asarray(data["driver"], dtype=np.float32)
        x_gt = np.asarray(data["log_price"], dtype=np.float32)
        if w_gt_raw.ndim == 3:
            w_gt = w_gt_raw[:, :, 0]
        elif w_gt_raw.ndim == 2:
            w_gt = w_gt_raw
        else:
            raise ValueError(f"Unexpected gt driver shape {w_gt_raw.shape}")
        t_len = min(
            int(w_model.shape[1]),
            int(x_model.shape[1]),
            int(w_gt.shape[1]),
            int(x_gt.shape[1]),
        )
        n = min(int(w_model.shape[0]), int(w_gt.shape[0]), 2048)
        feat_model = _ito_level12_feature_vectors(
            w_paths=w_model[:n, :t_len],
            x_paths=x_model[:n, :t_len],
            include_level1=True,
        )
        feat_gt = _ito_level12_feature_vectors(
            w_paths=w_gt[:n, :t_len],
            x_paths=x_gt[:n, :t_len],
            include_level1=True,
        )
        joint_feat_model = _cumulative_ito_signature_feature_vectors(
            w_paths=w_model[:n, :t_len],
            x_paths=x_model[:n, :t_len],
            num_eval_times=4,
        )
        joint_feat_gt = _cumulative_ito_signature_feature_vectors(
            w_paths=w_gt[:n, :t_len],
            x_paths=x_gt[:n, :t_len],
            num_eval_times=4,
        )
        projection_feat_model = _local_ito_projection_feature_vectors(
            w_paths=w_model[:n, :t_len],
            x_paths=x_model[:n, :t_len],
            block_size=16,
        )
        projection_feat_gt = _local_ito_projection_feature_vectors(
            w_paths=w_gt[:n, :t_len],
            x_paths=x_gt[:n, :t_len],
            block_size=16,
        )
        mart_model = _ito_log_martingale_mean_l2(x_paths=x_model[:n, :t_len])
        mart_gt = _ito_log_martingale_mean_l2(x_paths=x_gt[:n, :t_len])
        ito_summary_model = _ito_driver_output_summary(
            w_paths=w_model[:n, :t_len],
            x_paths=x_model[:n, :t_len],
        )
        ito_summary_gt = _ito_driver_output_summary(
            w_paths=w_gt[:n, :t_len],
            x_paths=x_gt[:n, :t_len],
        )
        extra_scalar_metrics = {
            "ito_level2_mmd2": _mmd_rbf_median_heuristic(feat_model, feat_gt),
            "ito_joint_signature_mmd2": _mmd_rbf_median_heuristic(
                joint_feat_model, joint_feat_gt
            ),
            "ito_log_martingale_mean_l2": mart_model,
            "ito_log_martingale_mean_gap_l2": abs(mart_model - mart_gt),
            "ito_local_projection_mmd2": _mmd_rbf_median_heuristic(
                projection_feat_model, projection_feat_gt
            ),
            "ito_qv_x_mean": ito_summary_model["qv_x_mean"],
            "ito_qv_x_mean_gap_abs": abs(
                ito_summary_model["qv_x_mean"] - ito_summary_gt["qv_x_mean"]
            ),
            "ito_cov_wx_mean": ito_summary_model["cov_wx_mean"],
            "ito_cov_wx_mean_gap_abs": abs(
                ito_summary_model["cov_wx_mean"] - ito_summary_gt["cov_wx_mean"]
            ),
            "ito_beta_mean": ito_summary_model["beta_mean"],
            "ito_beta_mean_gap_abs": abs(
                ito_summary_model["beta_mean"] - ito_summary_gt["beta_mean"]
            ),
            "ito_driver_corr": ito_summary_model["driver_corr"],
            "ito_driver_corr_gap_abs": abs(
                ito_summary_model["driver_corr"] - ito_summary_gt["driver_corr"]
            ),
        }
    else:
        pred_price0 = _extract_first_channel_paths(preds_batches[0])
        target_price0 = _extract_first_channel_paths(targets_batches[0])
        extra_scalar_metrics = _branched_ito_signature_moment_gap_metrics(
            pred_paths=pred_price0,
            target_paths=target_price0,
        )
        if controls_batches is not None:
            extra_scalar_metrics.update(
                _driver_output_consistency_metrics(
                    control_values=np.array(jax.device_get(controls_batches[0])),
                    price_paths=pred_price0,
                )
            )

    return ResultsDict(
        eval_metric=float(np.median(ks_stats)),
        results_times=[float(t) for t in valid_times],
        results=[float(s) for s in ks_stats],
        extra_scalar_metrics=extra_scalar_metrics if extra_scalar_metrics else None,
    )


def get_sg_so3_simulation_results(
    preds: list[jax.Array] | jax.Array,
    targets: list[jax.Array] | jax.Array,
    controls: list[jax.Array] | jax.Array | None,
    epoch_idx: int,
    model_name: str,
    times_to_save: list[int] | None = None,
    n_plot: int | None = None,
    save_plot_every: int = 1,
    config: Config | None = None,
) -> ResultsDict:
    """Visualize SO(3) predictions on a sphere.

    Args:
        preds: Predicted rotation matrices (B, T, 3, 3) (or list of batches)
        targets: Target rotation matrices (B, T, 3, 3) (or list of batches)
        controls: Unused (for compatibility; used for rough-volatility diagnostics)
        epoch_idx: Current epoch number
        times_to_save: Unused (for compatibility)
        n_plot: Number of trajectories to plot (default: 1)

    Returns:
        Empty ResultsDict (plots saved to disk)
    """
    del controls
    preds0 = preds[0] if isinstance(preds, list) else preds
    targets0 = targets[0] if isinstance(targets, list) else targets
    preds_np = np.array(jax.device_get(preds0))
    targets_np = np.array(jax.device_get(targets0))

    if n_plot is not None and epoch_idx % save_plot_every == 0:
        base_out_dir = os.environ.get(
            "SG_SO3_SPHERE_PLOT_DIR", "z_paper_content/sg_so3_sphere_by_epoch"
        )
        out_dir = os.path.join(base_out_dir, model_name)
        os.makedirs(out_dir, exist_ok=True)
        np.savez(
            os.path.join(out_dir, "sg_so3_batch.npz"),
            preds=preds_np,
            targets=targets_np,
        )
        if os.environ.get("SG_SO3_SUPPRESS_INDIVIDUAL", "0") == "1":
            n_plot = None
        num_plots = int(os.environ.get("SG_SO3_NUM_PLOTS", "50"))
        trajs_per_plot = int(os.environ.get("SG_SO3_TRAJECTORIES_PER_PLOT", "1"))
        n_plot0 = min(int(trajs_per_plot), int(targets_np.shape[0]))
        rng = np.random.default_rng(int(epoch_idx))
        for plot_idx in range(num_plots):
            if int(targets_np.shape[0]) > n_plot0:
                sample_idx = rng.choice(
                    int(targets_np.shape[0]), size=n_plot0, replace=False
                )
                sample_preds = preds_np[sample_idx]
                sample_targets = targets_np[sample_idx]
                sample_labels = [model_name for _ in np.asarray(sample_idx)]
            else:
                sample_preds = preds_np
                sample_targets = targets_np
                sample_labels = [model_name for _ in range(n_plot0)]

            filename = f"sphere_epoch_{epoch_idx:05d}_sample_{plot_idx:02d}.png"
            if num_plots == 1:
                filename = f"sphere_epoch_{epoch_idx:05d}.png"
            save_sg_so3_sphere_plot(
                preds=sample_preds,
                targets=sample_targets,
                out_file=os.path.join(out_dir, filename),
                n_plot=n_plot0,
                labels=sample_labels,
            )

    if config is None:
        raise ValueError("config must be provided to compute RGE for SG_SO3 results.")

    rge_fn = rotational_geodesic_loss(config)
    rge_value = float(jax.device_get(rge_fn(preds0, targets0)))
    eval_metric: float | None = None
    if config.experiment_config.loss == LossType.RGE:
        eval_metric = rge_value
    elif config.experiment_config.loss == LossType.FROBENIUS:
        frob_fn = frobenius_loss(config)
        eval_metric = float(jax.device_get(frob_fn(preds0, targets0)))

    return ResultsDict(
        eval_metric=eval_metric,
        results_times=[],
        results=[],
        extra_scalar_metrics={"rge": rge_value},
    )


def get_spd_covariance_results(
    preds: list[jax.Array] | jax.Array,
    targets: list[jax.Array] | jax.Array,
    controls: list[jax.Array] | jax.Array | None,
    epoch_idx: int,
    model_name: str,
    times_to_save: list[int] | None = list(DEFAULT_RESULTS_TIMES),
    n_plot: int | None = None,
    save_plot_every: int = 1,
    config: Config | None = None,
) -> ResultsDict:
    """Plot eigenvalue trajectories for SPD covariance (targets vs preds).

    Note: `epoch_idx` is 0-based (as used throughout the training loop). For plot
    saving cadence and filenames we use the human-friendly 1-based epoch number
    `epoch_idx + 1`, so "10th epoch" corresponds to epoch_idx==9.
    """
    del controls, config

    preds0 = preds[0] if isinstance(preds, list) else preds
    targets0 = targets[0] if isinstance(targets, list) else targets
    preds_np = np.array(jax.device_get(preds0))
    targets_np = np.array(jax.device_get(targets0))

    epoch_number = int(epoch_idx) + 1
    save_every = max(1, int(save_plot_every))
    if n_plot is not None and epoch_number % save_every == 0:
        n_plot0 = min(int(n_plot), int(targets_np.shape[0]))
        base_out_dir = os.environ.get(
            "SPD_COV_EIG_PLOT_DIR", "z_paper_content/spd_covariance_eigs_by_epoch"
        )
        out_dir = os.path.join(base_out_dir, model_name)
        os.makedirs(out_dir, exist_ok=True)
        save_spd_covariance_eigenvalue_trajectory_single_plot(
            paths=targets_np,
            out_file=os.path.join(out_dir, f"eig_targets_epoch_{epoch_number:05d}.png"),
            n_plot=n_plot0,
        )
        save_spd_covariance_eigenvalue_trajectory_single_plot(
            paths=preds_np,
            out_file=os.path.join(out_dir, f"eig_preds_epoch_{epoch_number:05d}.png"),
            n_plot=n_plot0,
        )
        fan_out_dir = os.environ.get(
            "SPD_COV_EIG_FAN_PLOT_DIR", "z_paper_content/spd_covariance_fan_by_epoch"
        )
        fan_out_dir = os.path.join(fan_out_dir, model_name)
        os.makedirs(fan_out_dir, exist_ok=True)
        max_paths_env = os.environ.get("SPD_COV_EIG_FAN_MAX_PATHS", "")
        max_paths = int(max_paths_env) if max_paths_env.isdigit() else None
        save_spd_covariance_eigenvalue_fan_single_plot(
            paths=targets_np,
            out_file=os.path.join(
                fan_out_dir, f"fan_targets_epoch_{epoch_number:05d}.png"
            ),
            max_paths=max_paths,
        )
        save_spd_covariance_eigenvalue_fan_single_plot(
            paths=preds_np,
            out_file=os.path.join(
                fan_out_dir, f"fan_preds_epoch_{epoch_number:05d}.png"
            ),
            max_paths=max_paths,
        )

    preds_batches = preds if isinstance(preds, list) else [preds]
    targets_batches = targets if isinstance(targets, list) else [targets]

    def _as_spd_mats_at_time(x: jax.Array, t: int) -> np.ndarray:
        """Extract SPD matrices at time t as (B,3,3) NumPy array.

        Accepts:
        - (B, T, 3, 3) explicit matrices
        - (B, T, 6) vech representation
        """
        x_np = np.array(jax.device_get(x))
        if x_np.ndim == 4 and x_np.shape[-2:] == (3, 3):
            return np.asarray(x_np[:, t, :, :], dtype=np.float64)
        if x_np.ndim == 3 and int(x_np.shape[-1]) == 6:
            xt = x_np[:, t, :]  # (B, 6)
            from stochastax.manifolds.spd import SPDManifold

            mats = SPDManifold.unvech(jax.device_put(xt.astype(np.float32)))
            return np.asarray(jax.device_get(mats), dtype=np.float64)
        raise ValueError(
            f"Expected preds/targets shaped (B,T,6) or (B,T,3,3); got {x_np.shape}"
        )

    def _eigvals_at_time(x: jax.Array, t: int) -> np.ndarray:
        mats = _as_spd_mats_at_time(x, t)
        mats = 0.5 * (mats + np.swapaxes(mats, -1, -2))
        eigs = np.linalg.eigvalsh(mats)  # (B, 3)
        eps = 1e-12
        eigs = np.clip(eigs, eps, None)
        if os.environ.get("SPD_COV_W1_LOGEIG", "0") == "1":
            eigs = np.log(eigs)
        return eigs

    valid_times = _valid_times_to_save(times_to_save, preds_batches, targets_batches)
    if len(valid_times) == 0:
        return ResultsDict(eval_metric=None, results_times=[], results=[])

    w1_stats: list[float] = []
    for t in valid_times:
        w1_per_batch: list[float] = []
        for pb, tb in zip(preds_batches, targets_batches):
            pred_eigs = _eigvals_at_time(pb, int(t))  # (B, 3)
            target_eigs = _eigvals_at_time(tb, int(t))  # (B, 3)
            w1_eigs = []
            for k in range(3):
                w1_eigs.append(
                    float(wasserstein_distance(pred_eigs[:, k], target_eigs[:, k]))
                )
            w1_per_batch.append(float(np.mean(w1_eigs)))
        w1_stats.append(float(np.mean(w1_per_batch)))

    return ResultsDict(
        eval_metric=float(np.median(w1_stats)),
        results_times=[float(t) for t in valid_times],
        results=[float(s) for s in w1_stats],
    )
