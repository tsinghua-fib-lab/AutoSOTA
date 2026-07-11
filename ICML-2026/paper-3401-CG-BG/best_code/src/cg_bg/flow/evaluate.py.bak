"""Reusable alanine evaluation helpers.

This module groups the shared alanine-peptide logic used by the notebook and the
metric scripts:
- alanine-specific atom indices
- .npz loading and weight resampling
- split metric functions plus bootstrap helpers
- reusable plotting helpers for phi/psi and free-energy panels

The implementation uses JAX for the compute-heavy parts and Matplotlib for
plotting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from chemtrain.quantity import kb
from jax.scipy.special import rel_entr
from jax.scipy.stats import gaussian_kde
from ott.geometry import costs, pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from rich.table import Table

from cg_bg.utils import get_console

try:
    import warnings

    import tqdm.auto as tqdm_auto
    import tqdm.rich as tqdm_rich
    import tqdm.std as tqdm_std
    from jax_tqdm import scan_tqdm
    from tqdm import TqdmExperimentalWarning

    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
    tqdm_auto.tqdm = tqdm_rich.tqdm
    tqdm_std.tqdm = tqdm_rich.tqdm
except Exception:
    pass


@dataclass(frozen=True)
class AlanineIndices:
    """Atom indices for phi/psi dihedrals."""

    phi: tuple[int, int, int, int]
    psi: tuple[int, int, int, int]


@dataclass(frozen=True)
class AlanineSpec:
    """Configuration for one alanine peptide family member."""

    variant: str
    indices: AlanineIndices
    label: str = "alanine"


@dataclass(frozen=True)
class AlanineDataset:
    """Coordinates, dihedrals, and optional weights loaded from a .npz dataset."""

    R: jnp.ndarray
    dihedrals: jnp.ndarray
    weights: jnp.ndarray | None = None
    energies: jnp.ndarray | None = None


@dataclass(frozen=True)
class AlaninePlotCase:
    """Plot configuration for one alanine evaluation case."""

    target_variant: str
    sample_variant: str
    implicit_variant: str | None = None


ALANINE_SPECS: dict[str, AlanineSpec] = {
    "ala2_ha": AlanineSpec(
        variant="ala2_ha",
        indices=AlanineIndices(phi=(1, 3, 4, 6), psi=(3, 4, 6, 8)),
    ),
    "ala2_cb": AlanineSpec(
        variant="ala2_cb",
        indices=AlanineIndices(phi=(0, 1, 2, 4), psi=(1, 2, 4, 5)),
    ),
    "ala3_cb": AlanineSpec(
        variant="ala3_cb",
        indices=AlanineIndices(phi=(3, 4, 5, 7), psi=(4, 5, 7, 8)),
    ),
    "ala3_ha": AlanineSpec(
        variant="ala3_ha",
        indices=AlanineIndices(phi=(6, 8, 9, 11), psi=(8, 9, 11, 13)),
    ),
    "ala3_ha_1": AlanineSpec(
        variant="ala3_ha",
        indices=AlanineIndices(phi=(1, 3, 4, 6), psi=(3, 4, 6, 8)),
    ),
    "ala3_ha_2": AlanineSpec(
        variant="ala3_ha",
        indices=AlanineIndices(phi=(6, 8, 9, 11), psi=(8, 9, 11, 13)),
    ),
    "ala3_ha_3": AlanineSpec(
        variant="ala3_ha",
        indices=AlanineIndices(phi=(11, 13, 14, 16), psi=(13, 14, 16, 18)),
    ),
    "ala3_ha_capped": AlanineSpec(
        variant="ala3_ha_capped",
        indices=AlanineIndices(phi=(4, 5, 6, 8), psi=(5, 6, 8, 9)),
    ),
    "ala6_cb": AlanineSpec(
        variant="ala6_cb",
        indices=AlanineIndices(phi=(11, 12, 13, 15), psi=(12, 13, 15, 16)),
    ),
    "ala6_cb_1": AlanineSpec(
        variant="ala6_cb",
        indices=AlanineIndices(phi=(3, 4, 5, 7), psi=(4, 5, 7, 8)),
    ),
    "ala6_cb_2": AlanineSpec(
        variant="ala6_cb",
        indices=AlanineIndices(phi=(7, 8, 9, 11), psi=(8, 9, 11, 12)),
    ),
    "ala6_cb_3": AlanineSpec(
        variant="ala6_cb",
        indices=AlanineIndices(phi=(11, 12, 13, 15), psi=(12, 13, 15, 16)),
    ),
    "ala6_cb_4": AlanineSpec(
        variant="ala6_cb",
        indices=AlanineIndices(phi=(15, 16, 17, 19), psi=(16, 17, 19, 20)),
    ),
    "ala2_implicit": AlanineSpec(
        variant="ala2_implicit",
        indices=AlanineIndices(phi=(4, 6, 7, 8), psi=(6, 7, 8, 16)),
    ),
    "ala3_implicit": AlanineSpec(
        variant="ala3_implicit",
        indices=AlanineIndices(phi=(14, 16, 18, 24), psi=(16, 18, 24, 26)),
    ),
    "ala3_implicit_1": AlanineSpec(
        variant="ala3_implicit",
        indices=AlanineIndices(phi=(4, 6, 8, 14), psi=(6, 8, 14, 16)),
    ),
    "ala3_implicit_2": AlanineSpec(
        variant="ala3_implicit",
        indices=AlanineIndices(phi=(14, 16, 18, 24), psi=(16, 18, 24, 26)),
    ),
    "ala3_implicit_3": AlanineSpec(
        variant="ala3_implicit",
        indices=AlanineIndices(phi=(24, 26, 28, 34), psi=(26, 28, 34, 36)),
    ),
    "ala6_implicit": AlanineSpec(
        variant="ala6_implicit",
        indices=AlanineIndices(phi=(34, 36, 38, 44), psi=(36, 38, 44, 46)),
    ),
    "ala6_implicit_1": AlanineSpec(
        variant="ala6_implicit",
        indices=AlanineIndices(phi=(14, 16, 18, 24), psi=(16, 18, 24, 26)),
    ),
    "ala6_implicit_2": AlanineSpec(
        variant="ala6_implicit",
        indices=AlanineIndices(phi=(24, 26, 28, 34), psi=(26, 28, 34, 36)),
    ),
    "ala6_implicit_3": AlanineSpec(
        variant="ala6_implicit",
        indices=AlanineIndices(phi=(34, 36, 38, 44), psi=(36, 38, 44, 46)),
    ),
    "ala6_implicit_4": AlanineSpec(
        variant="ala6_implicit",
        indices=AlanineIndices(phi=(44, 46, 48, 54), psi=(46, 48, 54, 56)),
    ),
}

PALETTE = sns.color_palette("muted", n_colors=10)
PALETTE_PAPER = {
    "Exact": PALETTE[3],
    "Simulation": PALETTE[2],
    "Proposal": PALETTE[1],
    "Proposal (reweighted)": PALETTE[0],
    "Implicit": PALETTE[4],
}


ALANINE_PLOT_CASES: dict[str, AlaninePlotCase] = {
    "ala2_ha": AlaninePlotCase("ala2_ha", "ala2_ha", "ala2_implicit"),
    "ala2_cb": AlaninePlotCase("ala2_cb", "ala2_cb", "ala2_implicit"),
    "ala3_cb": AlaninePlotCase("ala3_cb", "ala3_cb", "ala3_implicit"),
    "ala3_ha": AlaninePlotCase("ala3_ha", "ala3_ha", "ala3_implicit"),
    "ala6_cb": AlaninePlotCase("ala6_cb", "ala6_cb", "ala6_implicit"),
    "ala3_ha_capped": AlaninePlotCase("ala3_ha_capped", "ala3_ha_capped", "ala3_implicit"),
    "ala3_ha_1": AlaninePlotCase("ala3_ha_1", "ala3_ha_1", "ala3_implicit_1"),
    "ala3_ha_2": AlaninePlotCase("ala3_ha_2", "ala3_ha_2", "ala3_implicit_2"),
    "ala3_ha_3": AlaninePlotCase("ala3_ha_3", "ala3_ha_3", "ala3_implicit_3"),
    "ala6_cb_1": AlaninePlotCase("ala6_cb_1", "ala6_cb_1", "ala6_implicit_1"),
    "ala6_cb_2": AlaninePlotCase("ala6_cb_2", "ala6_cb_2", "ala6_implicit_2"),
    "ala6_cb_3": AlaninePlotCase("ala6_cb_3", "ala6_cb_3", "ala6_implicit_3"),
    "ala6_cb_4": AlaninePlotCase("ala6_cb_4", "ala6_cb_4", "ala6_implicit_4"),
}


def get_alanine_plot_case(name: str) -> AlaninePlotCase:
    """Return the plotting configuration for one alanine case."""

    try:
        return ALANINE_PLOT_CASES[name]
    except KeyError:
        return AlaninePlotCase(name, name, None)


def configure_plot_style() -> None:
    """Apply the notebook-style plotting defaults."""

    sns.set_style("white")
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.labelsize": 20,
            "axes.titlesize": 30,
            "legend.fontsize": 10,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "lines.markersize": 3,
            "lines.linewidth": 3,
            "figure.dpi": 800,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def get_alanine_spec(variant: str) -> AlanineSpec:
    """Return the reusable configuration for an alanine peptide."""

    try:
        return ALANINE_SPECS[variant]
    except KeyError as exc:
        valid = ", ".join(sorted(ALANINE_SPECS))
        raise ValueError(f"Unknown alanine variant '{variant}'. Choose from: {valid}") from exc


def get_alanine_indices(variant: str) -> AlanineIndices:
    """Return phi/psi atom indices for an alanine peptide."""

    return get_alanine_spec(variant).indices


def load_npz_file(path: Path) -> dict[str, np.ndarray]:
    """Load a .npz file as a plain dictionary."""

    if path.suffix != ".npz":
        raise ValueError(f"Only .npz files are supported, got {path}")
    return dict(np.load(path, allow_pickle=True))


@jax.jit
@jax.vmap
def dihedral(p: jnp.ndarray) -> jnp.ndarray:
    """The code is taken and adapted from: http://stackoverflow.com/q/20305272/1128289

    Args:
        p: A set of points in the form of a jax numpy array with shape (batch, 4, 3).
    Returns:
        The dihedral angles in radians.
    """
    b = p[:-1] - p[1:]
    b = b.at[0].set(-b[0])
    v = jnp.array([v - (v.dot(b[1]) / b[1].dot(b[1])) * b[1] for v in [b[0], b[2]]])
    # Normalize vectors
    v /= jnp.sqrt(jnp.einsum("...i,...i", v, v)).reshape(-1, 1)
    b1 = b[1] / jnp.linalg.norm(b[1])
    x = jnp.dot(v[0], v[1])
    m = jnp.cross(v[0], b1)
    y = jnp.dot(m, v[1])
    return jnp.arctan2(y, x)


@jax.jit
def compute_pair_distance(R, i, j):
    pos_i = R[:, i, :]
    pos_j = R[:, j, :]
    diff = pos_i - pos_j
    distances = jnp.linalg.norm(diff, ord=2, axis=-1)
    return distances


def compute_alanine_dihedrals(coords: jnp.ndarray, variant: str) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute phi and psi dihedrals for one alanine dataset."""

    indices = get_alanine_indices(variant)
    phi = dihedral(coords[:, list(indices.phi), :])
    psi = dihedral(coords[:, list(indices.psi), :])
    return phi, psi


def clip_weights(log_weights: jnp.ndarray, clip: Optional[float] = None) -> jnp.ndarray:
    """Clip and normalize log-weights into a probability distribution."""

    if clip is None:
        return jax.nn.softmax(log_weights)

    cutoff = jnp.percentile(log_weights, clip)
    masked = jnp.where(log_weights < cutoff, log_weights, -jnp.inf)
    return jax.nn.softmax(masked)


def compute_alanine_dataset(
    path: Path,
    variant: str,
    clip: Optional[float] = None,
    energy_key: str = "U",
) -> AlanineDataset:
    """Load an alanine .npz file and return coordinates, dihedrals, and optional weights."""

    data = load_npz_file(path)
    if "R" not in data:
        raise ValueError(f"Expected an 'R' array in {path}")

    coords = jnp.asarray(data["R"])
    phi, psi = compute_alanine_dihedrals(coords, variant)
    dihedrals = jnp.stack([phi, psi], axis=1)

    weights = None
    if "logw" in data:
        weights = clip_weights(jnp.asarray(data["logw"]), clip=clip)
    elif "logp" in data and energy_key in data:
        logw = -jnp.asarray(data[energy_key]) / (kb * 300) - jnp.asarray(data["logp"])
        weights = clip_weights(logw, clip=clip)

    energies = jnp.asarray(data[energy_key]) if energy_key in data else None
    return AlanineDataset(R=coords, dihedrals=dihedrals, weights=weights, energies=energies)


def load_alanine_dihedrals(
    path: Path,
    variant: str,
    clip: float,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """Compatibility wrapper that returns dihedrals and weights only."""

    dataset = compute_alanine_dataset(path, variant=variant, clip=clip)
    return dataset.dihedrals, dataset.weights


def compute_ess_percent(weights: jnp.ndarray | None) -> jnp.ndarray:
    """Compute the effective sample size as a fraction of the sample count."""

    if weights is None:
        return jnp.asarray(1.0)

    positive = jnp.clip(weights, a_min=0.0)
    total = jnp.sum(positive)
    normalized = jnp.where(total > 0, positive / total, jnp.full_like(positive, 1.0 / positive.size))
    ess = 1.0 / jnp.sum(normalized**2)
    return ess / normalized.size


def compute_alanine_metric_summary(
    target_dihedrals: jnp.ndarray,
    sample_dihedrals: jnp.ndarray,
    bg_weights: jnp.ndarray | None,
    limits: Tuple[Tuple[float, float], Tuple[float, float]],
) -> dict[str, jnp.ndarray]:
    """Compute alanine metrics once without bootstrap resampling."""

    return {
        "JS_Divergence": compute_js_divergence(
            target_dihedrals,
            sample_dihedrals,
            limits=limits,
            bins=100,
            baseline=1e-10,
            sample_weights=bg_weights,
        ),
        "PMF_Error": compute_pmf_error(
            target_dihedrals,
            sample_dihedrals,
            limits=limits,
            bins=100,
            baseline=1e-10,
            sample_weights=bg_weights,
        ),
        "ESS_Percent": compute_ess_percent(bg_weights),
    }


def _metric_range(
    target: jnp.ndarray,
    sample: jnp.ndarray,
    limits: Optional[Tuple[Tuple[float, float], Tuple[float, float]]],
) -> list[list[jnp.ndarray]]:
    if limits is None:
        all_data = jnp.vstack([target, sample])
        x_min, x_max = all_data[:, 0].min(), all_data[:, 0].max()
        y_min, y_max = all_data[:, 1].min(), all_data[:, 1].max()
        return [[x_min, x_max], [y_min, y_max]]

    (x_min, x_max), (y_min, y_max) = limits
    return [[x_min, x_max], [y_min, y_max]]


def compute_js_divergence(
    target: jnp.ndarray,
    sample: jnp.ndarray,
    limits: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    bins: int = 64,
    baseline: float = 1e-6,
    sample_weights: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute the Jensen-Shannon divergence between two 2D histograms."""

    range_arr = _metric_range(target, sample, limits)
    hist_target, _, _ = jnp.histogram2d(target[:, 0], target[:, 1], bins=bins, range=range_arr)
    hist_sampled, _, _ = jnp.histogram2d(sample[:, 0], sample[:, 1], bins=bins, range=range_arr, weights=sample_weights)

    p_target = hist_target / (hist_target.sum() + baseline)
    p_sampled = hist_sampled / (hist_sampled.sum() + baseline)
    p = p_target.flatten()
    q = p_sampled.flatten()
    m = 0.5 * (p + q)
    return 0.5 * (jnp.sum(rel_entr(p, m)) + jnp.sum(rel_entr(q, m)))


def compute_pmf_error(
    target: jnp.ndarray,
    sample: jnp.ndarray,
    limits: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    bins: int = 64,
    baseline: float = 1e-6,
    sample_weights: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute the histogram PMF error used in the original notebook."""

    range_arr = _metric_range(target, sample, limits)
    hist_target, _, _ = jnp.histogram2d(target[:, 0], target[:, 1], bins=bins, range=range_arr)
    hist_sampled, _, _ = jnp.histogram2d(sample[:, 0], sample[:, 1], bins=bins, range=range_arr, weights=sample_weights)

    p_target = hist_target / (hist_target.sum() + baseline)
    p_sampled = hist_sampled / (hist_sampled.sum() + baseline)
    p = p_target.flatten()
    q = p_sampled.flatten()

    p_filled = jnp.where(p == 0, baseline, p)
    q_filled = jnp.where(q == 0, baseline, q)
    m_filled = (p_filled + q_filled) / 2
    weights_filled = m_filled / m_filled.sum()
    fe_loss = (-jnp.log(q_filled) - (-jnp.log(p_filled))) ** 2
    return jnp.sum(weights_filled * fe_loss)


@jax.jit
def energy_wasserstein(
    pred_energy: jax.Array,
    true_energy: jax.Array,
    n_quantiles: int = 1000,
) -> Tuple[jax.Array, jax.Array]:
    """Approximate 1- and 2-Wasserstein distances in one dimension."""

    pred_sorted = jnp.sort(pred_energy)
    true_sorted = jnp.sort(true_energy)
    p_idx = jnp.linspace(0.0, 1.0, len(pred_sorted))
    t_idx = jnp.linspace(0.0, 1.0, len(true_sorted))
    common_quantiles = jnp.linspace(0.0, 1.0, n_quantiles)

    pred_aligned = jnp.interp(common_quantiles, p_idx, pred_sorted)
    true_aligned = jnp.interp(common_quantiles, t_idx, true_sorted)
    w2 = jnp.sqrt(jnp.mean((pred_aligned - true_aligned) ** 2))
    w1 = jnp.mean(jnp.abs(pred_aligned - true_aligned))
    return w1, w2


@jax.tree_util.register_pytree_node_class
class TorusCost(costs.CostFn):
    """Squared geodesic distance on a torus."""

    def __call__(self, x: jax.Array, y: jax.Array) -> jax.Array:
        diff = jnp.abs(x - y)
        circ_diff = jnp.minimum(diff, 2 * jnp.pi - diff)
        return jnp.sum(circ_diff**2)

    def tree_flatten(self):
        return (), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls()


@jax.jit
def torus_w2(x0: jax.Array, x1: jax.Array, epsilon: float = 1e-2) -> jax.Array:
    """Sinkhorn distance on the torus for 2D angle pairs."""

    geom = pointcloud.PointCloud(x0, x1, cost_fn=TorusCost(), epsilon=epsilon)
    prob = linear_problem.LinearProblem(geom)
    solver = sinkhorn.Sinkhorn()
    out = solver(prob)
    return jnp.sqrt(out.reg_ot_cost)


def resample_alanine_coordinates(
    key: jax.Array,
    R: jnp.ndarray,
    weights: jnp.ndarray | None,
    replace: bool = True,
) -> jnp.ndarray:
    """Resample coordinates using optional normalized weights."""

    n_samples = R.shape[0]
    if weights is None:
        idx = jax.random.choice(key, n_samples, shape=(n_samples,), replace=replace)
    else:
        idx = jax.random.choice(key, n_samples, shape=(n_samples,), replace=replace, p=weights)
    return R[idx]


def resample_alanine_dataset(
    key: jax.Array,
    dataset: AlanineDataset,
    variant: str,
    replace: bool = True,
) -> AlanineDataset:
    """Resample R first and then recompute dihedrals from the resampled coordinates."""

    resampled_R = resample_alanine_coordinates(key, dataset.R, dataset.weights, replace=replace)
    phi, psi = compute_alanine_dihedrals(resampled_R, variant)
    return AlanineDataset(
        R=resampled_R,
        dihedrals=jnp.stack([phi, psi], axis=1),
        weights=dataset.weights,
        energies=None,
    )


def run_bootstrap_metrics(
    key: jax.Array,
    target_dihedrals: jnp.ndarray,
    sample_dihedrals: jnp.ndarray,
    bg_weights: jnp.ndarray | None,
    limits: Tuple[Tuple[float, float], Tuple[float, float]],
    n_bootstraps: int = 100,
    max_wass_samples: int = 10_000,
    compute_torus_w2: bool = False,
    use_tqdm: bool = True,
) -> dict[str, jnp.ndarray]:
    """Bootstrap the alanine metrics on angle clouds."""

    n_target = target_dihedrals.shape[0]
    n_sample = sample_dihedrals.shape[0]
    resample_probs = bg_weights

    def single_step(carry: None, scan_input: Tuple[jax.Array, jax.Array]) -> Tuple[None, jnp.ndarray]:
        _, carry_key = scan_input
        k1, k2, k3, k4 = jax.random.split(carry_key, 4)

        idx_target = jax.random.choice(k1, n_target, shape=(n_target,), replace=True)
        if resample_probs is not None:
            idx_sample = jax.random.choice(k2, n_sample, shape=(n_sample,), replace=True, p=resample_probs)
            idx_weights = jax.random.choice(k4, n_sample, shape=(n_sample,), replace=True)
            weights = bg_weights[idx_weights]
            ess_percent = compute_ess_percent(weights)
        else:
            idx_sample = jax.random.choice(k2, n_sample, shape=(n_sample,), replace=True)
            ess_percent = compute_ess_percent(None)

        t_dihedrals_boot = target_dihedrals[idx_target]
        s_dihedrals_boot = sample_dihedrals[idx_sample]
        js_div = compute_js_divergence(t_dihedrals_boot, s_dihedrals_boot, limits=limits, bins=100, baseline=1e-10)
        pmf_error = compute_pmf_error(t_dihedrals_boot, s_dihedrals_boot, limits=limits, bins=100, baseline=1e-10)

        subsample_size = min(max_wass_samples, n_target, n_sample)
        k3_target, k3_sample = jax.random.split(k3)
        target_sub_idx = (
            jax.random.choice(k3_target, n_target, shape=(subsample_size,), replace=False)
            if subsample_size < n_target
            else jnp.arange(n_target)
        )
        sample_sub_idx = (
            jax.random.choice(k3_sample, n_sample, shape=(subsample_size,), replace=False)
            if subsample_size < n_sample
            else jnp.arange(n_sample)
        )

        tw2 = (
            torus_w2(t_dihedrals_boot[target_sub_idx], s_dihedrals_boot[sample_sub_idx])
            if compute_torus_w2
            else jnp.asarray(jnp.nan)
        )
        return carry, jnp.array([js_div, pmf_error, ess_percent, tw2])

    keys = jax.random.split(key, n_bootstraps)
    step_ids = jnp.arange(n_bootstraps)
    scan_inputs = (step_ids, keys)
    scan_step = single_step
    if use_tqdm and scan_tqdm is not None:
        scan_step = scan_tqdm(n_bootstraps)(scan_step)

    _, results = jax.lax.scan(scan_step, None, scan_inputs)
    out: dict[str, jnp.ndarray] = {
        "JS_Divergence": results[:, 0],
        "PMF_Error": results[:, 1],
        "ESS_Percent": results[:, 2],
    }
    if compute_torus_w2:
        out["Torus_W2"] = results[:, 3]
    return out


def _metric_table_rows(
    results: dict[str, jnp.ndarray],
    direct_results: dict[str, jnp.ndarray] | None = None,
) -> tuple[list[tuple[str, float, float, float | None]], bool]:
    metrics_order = ["JS_Divergence", "PMF_Error", "ESS_Percent", "Torus_W2"]
    metrics_order = [name for name in metrics_order if name in results]

    rows: list[tuple[str, float, float, float | None]] = []
    has_direct = direct_results is not None
    for metric_name in metrics_order:
        values = jnp.asarray(results[metric_name])
        direct_value = None
        if direct_results is not None and metric_name in direct_results:
            direct_value = float(jnp.asarray(direct_results[metric_name]))
        rows.append((metric_name, float(jnp.mean(values)), float(jnp.std(values)), direct_value))
    return rows, has_direct


def _format_metric_table_text(
    title: str,
    results: dict[str, jnp.ndarray],
    *,
    direct_results: dict[str, jnp.ndarray] | None = None,
    precision: int = 6,
    target_count: int | None = None,
    sample_count: int | None = None,
) -> str:
    rows, has_direct = _metric_table_rows(results, direct_results)
    lines = [title]
    if target_count is not None or sample_count is not None:
        sample_parts: list[str] = []
        if target_count is not None:
            sample_parts.append(f"target={target_count}")
        if sample_count is not None:
            sample_parts.append(f"sample={sample_count}")
        lines.append(f"Samples: {', '.join(sample_parts)}")
    lines.extend(
        [
            "Metric | Mean | Std Dev" + (" | Direct" if has_direct else ""),
            "--- | --- | ---" + (" | ---" if has_direct else ""),
        ]
    )
    for metric_name, mean, std, direct_value in rows:
        row = f"{metric_name} | {mean:.{precision}f} | {std:.{precision}f}"
        if direct_value is not None:
            row += f" | {direct_value:.{precision}f}"
        lines.append(row)
    return "\n".join(lines)


def print_bootstrap_metrics(
    results: dict[str, jnp.ndarray],
    label: str,
    n_bootstraps: int,
    *,
    direct_results: dict[str, jnp.ndarray] | None = None,
    target_count: int | None = None,
    sample_count: int | None = None,
) -> None:
    """Print metric means, standard deviations, and optional direct values."""

    console = get_console()
    table = Table(title=f"[{label}] Bootstrap Metrics (n={n_bootstraps})")
    table.add_column("Metric", style="bold")
    table.add_column("Mean", justify="right")
    table.add_column("Std Dev", justify="right")
    if direct_results is not None:
        table.add_column("Direct", justify="right")
    if target_count is not None or sample_count is not None:
        sample_parts: list[str] = []
        if target_count is not None:
            sample_parts.append(f"target={target_count}")
        if sample_count is not None:
            sample_parts.append(f"sample={sample_count}")
        table.caption = f"Samples: {', '.join(sample_parts)}"

    rows, _ = _metric_table_rows(results, direct_results)
    for metric_name, mean, std, direct in rows:
        if direct is not None:
            table.add_row(metric_name, f"{mean:.6f}", f"{std:.6f}", f"{direct:.6f}")
        else:
            table.add_row(metric_name, f"{mean:.6f}", f"{std:.6f}")

    console.print(table)


def format_metric_cell(mean: float, std: float, precision: int = 4) -> str:
    """Format one LaTeX table cell."""

    return f"${mean:.{precision}f} \\pm {std:.{precision}f}$"


def format_metric_row(label: str, results: dict[str, jnp.ndarray], precision: int = 4) -> str:
    """Format a full LaTeX row from bootstrap results."""

    order = ["JS_Divergence", "PMF_Error", "ESS_Percent", "Torus_W2"]
    order = [name for name in order if name in results]
    cells = [
        format_metric_cell(float(jnp.mean(results[name])), float(jnp.std(results[name])), precision=precision)
        for name in order
    ]
    return f"{label} & " + " & ".join(cells) + r" \\"


def format_bootstrap_metrics_report(
    results: dict[str, jnp.ndarray],
    label: str,
    n_bootstraps: int,
    precision: int = 6,
    direct_results: dict[str, jnp.ndarray] | None = None,
    target_count: int | None = None,
    sample_count: int | None = None,
    without_reweight_results: dict[str, jnp.ndarray] | None = None,
    without_reweight_direct_results: dict[str, jnp.ndarray] | None = None,
    without_reweight_target_count: int | None = None,
    without_reweight_sample_count: int | None = None,
    without_reweight_n_bootstraps: int | None = None,
) -> str:
    """Format a plain-text bootstrap report for disk output."""

    sections = [
        _format_metric_table_text(
            f"[{label}] Reweight Metrics (n={n_bootstraps})",
            results,
            direct_results=direct_results,
            precision=precision,
            target_count=target_count,
            sample_count=sample_count,
        )
    ]

    if without_reweight_results is not None:
        sections.append(
            _format_metric_table_text(
                f"[{label}] Without Reweight Metrics (n={without_reweight_n_bootstraps or n_bootstraps})",
                without_reweight_results,
                direct_results=without_reweight_direct_results,
                precision=precision,
                target_count=without_reweight_target_count,
                sample_count=without_reweight_sample_count,
            )
        )

    return "\n\n".join(sections) + "\n"


def write_bootstrap_metrics_report(
    results: dict[str, jnp.ndarray],
    path: Path,
    label: str,
    n_bootstraps: int,
    precision: int = 6,
    direct_results: dict[str, jnp.ndarray] | None = None,
    target_count: int | None = None,
    sample_count: int | None = None,
    without_reweight_results: dict[str, jnp.ndarray] | None = None,
    without_reweight_direct_results: dict[str, jnp.ndarray] | None = None,
    without_reweight_target_count: int | None = None,
    without_reweight_sample_count: int | None = None,
    without_reweight_n_bootstraps: int | None = None,
) -> Path:
    """Write the bootstrap metrics report to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        format_bootstrap_metrics_report(
            results,
            label=label,
            n_bootstraps=n_bootstraps,
            precision=precision,
            direct_results=direct_results,
            target_count=target_count,
            sample_count=sample_count,
            without_reweight_results=without_reweight_results,
            without_reweight_direct_results=without_reweight_direct_results,
            without_reweight_target_count=without_reweight_target_count,
            without_reweight_sample_count=without_reweight_sample_count,
            without_reweight_n_bootstraps=without_reweight_n_bootstraps,
        )
    )
    return path


def compute_free_energy_1d(
    samples: jnp.ndarray,
    x_grid: jnp.ndarray,
    weights: jnp.ndarray | None = None,
    kT_value: float = 1.0,
    energy_cutoff: float | None = None,
) -> jnp.ndarray:
    """Compute a one-dimensional free-energy curve from samples."""

    kde = gaussian_kde(samples, weights=weights)
    prob = kde(x_grid)
    free_energy = -kT_value * jnp.log(prob)
    min_val = jnp.nanmin(jnp.where(jnp.isfinite(free_energy), free_energy, jnp.inf))
    free_energy = free_energy - min_val
    if energy_cutoff is not None:
        free_energy = jnp.where(free_energy > energy_cutoff, jnp.nan, free_energy)
    return free_energy


def _hist(
    ax: plt.Axes,
    data: jnp.ndarray,
    *,
    bins: int,
    color,
    label: str,
    range_: Optional[Tuple[float, float]] = None,
    weights: Optional[jnp.ndarray] = None,
    alpha: float = 0.7,
    edgecolor: str = "none",
    histtype: str = "bar",
    linestyle: str = "-",
    linewidth: Optional[float] = None,
) -> None:
    data = jnp.asarray(data)
    mask = jnp.isfinite(data)
    if weights is not None:
        weights = jnp.asarray(weights)
        mask = mask & jnp.isfinite(weights)
    if not bool(jnp.any(mask)):
        return
    data = data[mask]
    if weights is not None:
        weights = weights[mask]
    ax.hist(
        data,
        bins=bins,
        range=range_,
        density=True,
        color=color,
        edgecolor=edgecolor,
        alpha=alpha,
        label=label,
        weights=weights,
        histtype=histtype,
        linestyle=linestyle,
        linewidth=linewidth,
    )


def _plot_dihedral_fe_panel(
    ax: plt.Axes,
    kT_value: float,
    x_grid: jnp.ndarray,
    samples: tuple[jnp.ndarray, ...],
    labels: tuple[str, ...],
    colors: tuple[Any, ...],
    linestyles: tuple[str, ...],
    *,
    weights: Optional[tuple[Optional[jnp.ndarray], ...]] = None,
    energy_cutoff: Optional[float] = None,
    xlabel: str,
    ylabel: str,
    legend_kwargs: Optional[dict[str, Any]] = None,
) -> plt.Axes:
    weights = weights or (None,) * len(samples)
    for sample, weight, label, color, style in zip(samples, weights, labels, colors, linestyles):
        fe = compute_free_energy_1d(sample, x_grid, weights=weight, kT_value=kT_value, energy_cutoff=energy_cutoff)
        ax.plot(x_grid, fe, color=color, linestyle=style, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-jnp.pi, jnp.pi)
    if legend_kwargs is None:
        legend_kwargs = {}
    ax.legend(**legend_kwargs)
    ax.set_box_aspect(1)
    return ax


def _plot_alanine_dihedral_density(
    ax: plt.Axes,
    ref: jnp.ndarray,
    sample: jnp.ndarray,
    *,
    implicit: Optional[jnp.ndarray] = None,
    weights: Optional[jnp.ndarray] = None,
    xlabel: str,
) -> plt.Axes:
    plot_range = (-jnp.pi, jnp.pi)
    bins = 100

    _hist(ax, ref, bins=bins, range_=plot_range, color=PALETTE_PAPER["Simulation"], label="Explicit Solvent MD")
    if implicit is not None:
        _hist(
            ax,
            implicit,
            bins=bins,
            range_=plot_range,
            color=PALETTE_PAPER["Implicit"],
            edgecolor=PALETTE_PAPER["Implicit"],
            label="Implicit Solvent MD",
            histtype="step",
            alpha=1.0,
        )
    _hist(ax, sample, bins=bins, range_=plot_range, color=PALETTE_PAPER["Proposal"], label="CG-BG Proposal")
    _hist(
        ax,
        sample,
        bins=bins,
        range_=plot_range,
        color=PALETTE_PAPER["Proposal (reweighted)"],
        edgecolor=PALETTE_PAPER["Proposal (reweighted)"],
        label="CG-BG Reweighted",
        weights=weights,
        histtype="step",
        alpha=1.0,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.legend(loc="best", alignment="left")
    return ax


def plot_alanine_density(
    phi_ref: jnp.ndarray,
    phi_sample: jnp.ndarray,
    psi_ref: jnp.ndarray,
    psi_sample: jnp.ndarray,
    axes: tuple[plt.Axes, plt.Axes],
    *,
    phi_implicit: Optional[jnp.ndarray] = None,
    psi_implicit: Optional[jnp.ndarray] = None,
    weights: Optional[jnp.ndarray] = None,
) -> tuple[plt.Axes, plt.Axes]:
    """Plot phi and psi density panels for one alanine system."""

    ax_phi, ax_psi = axes
    _plot_alanine_dihedral_density(
        ax_phi,
        phi_ref,
        phi_sample,
        implicit=phi_implicit,
        weights=weights,
        xlabel=r"$\phi$",
    )
    _plot_alanine_dihedral_density(
        ax_psi,
        psi_ref,
        psi_sample,
        implicit=psi_implicit,
        weights=weights,
        xlabel=r"$\psi$",
    )
    return ax_phi, ax_psi


def plot_alanine_energy_density(
    energy_ref: jnp.ndarray,
    energy_sample: jnp.ndarray,
    energy_implicit: Optional[jnp.ndarray],
    ax: plt.Axes,
    weights: Optional[jnp.ndarray] = None,
) -> plt.Axes:
    """Plot the alanine energy histogram panel."""

    mask = (weights > 1e-10) if weights is not None else None
    energy_sample_masked = energy_sample if mask is None else energy_sample[mask]
    weights_masked = None if weights is None else weights[mask]

    _hist(ax, energy_ref, bins=100, color=PALETTE_PAPER["Simulation"], label="MD Reference")
    if energy_implicit is not None:
        _hist(ax, energy_implicit, bins=100, color=PALETTE_PAPER["Implicit"], label="Implicit MD")
    _hist(
        ax,
        energy_sample_masked,
        bins=100,
        color=PALETTE_PAPER["Proposal"],
        label="CG-BG (proposal)",
    )
    _hist(
        ax,
        energy_sample_masked,
        bins=100,
        color=PALETTE_PAPER["Proposal (reweighted)"],
        edgecolor=PALETTE_PAPER["Proposal (reweighted)"],
        label="CG-BG (reweighted)",
        weights=weights_masked,
        histtype="step",
        alpha=1.0,
        linewidth=plt.rcParams.get("lines.linewidth", 2),
    )
    ax.set_xlabel("Energy")
    ax.set_ylabel("Density")
    ax.legend(loc="best")
    ax.set_box_aspect(1)
    return ax


def plot_alanine_ramachandran_fes(
    kT_value: float,
    phi_sample: jnp.ndarray,
    psi_sample: jnp.ndarray,
    phi_ref: jnp.ndarray,
    psi_ref: jnp.ndarray,
    axes: tuple[plt.Axes, plt.Axes, plt.Axes],
    weights: Optional[jnp.ndarray] = None,
) -> tuple[plt.Axes, plt.Axes, plt.Axes]:
    """Plot reference, proposal, and reweighted Ramachandran free-energy maps."""

    ax_ref, ax_prop, ax_rew = axes
    kT_kcal = kT_value / 4.184
    ticks = [-jnp.pi, -jnp.pi / 2, 0, jnp.pi / 2, jnp.pi]
    tick_labels = [r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"]
    cmap = plt.get_cmap("viridis")

    h_ref, x_edges_ref, y_edges_ref = jnp.histogram2d(phi_ref, psi_ref, bins=100, density=True)
    h_ref = -kT_kcal * jnp.log(h_ref)
    x_ref, y_ref = jnp.meshgrid(x_edges_ref, y_edges_ref)
    ax_ref.pcolormesh(x_ref, y_ref, h_ref.T, cmap=cmap, vmax=5.25)
    ax_ref.set_xlabel(r"$\phi$")
    ax_ref.set_ylabel(r"$\psi$")
    ax_ref.set_box_aspect(1)
    ax_ref.set_xticks(ticks)
    ax_ref.set_xticklabels(tick_labels)
    ax_ref.set_yticks(ticks)
    ax_ref.set_yticklabels(tick_labels)
    ax_ref.set_xlim(-jnp.pi, jnp.pi)
    ax_ref.set_ylim(-jnp.pi, jnp.pi)

    h_prop, x_edges_prop, y_edges_prop = jnp.histogram2d(phi_sample, psi_sample, bins=100, density=True)
    h_prop = -kT_kcal * jnp.log(h_prop)
    x_prop, y_prop = jnp.meshgrid(x_edges_prop, y_edges_prop)
    ax_prop.pcolormesh(x_prop, y_prop, h_prop.T, cmap=cmap, vmax=5.25)
    ax_prop.set_xlabel(r"$\phi$")
    ax_prop.set_ylabel(r"$\psi$")
    ax_prop.set_box_aspect(1)
    ax_prop.set_xticks(ticks)
    ax_prop.set_xticklabels(tick_labels)
    ax_prop.set_yticks(ticks)
    ax_prop.set_yticklabels(tick_labels)
    ax_prop.set_xlim(-jnp.pi, jnp.pi)
    ax_prop.set_ylim(-jnp.pi, jnp.pi)

    h_rew, x_edges_rew, y_edges_rew = jnp.histogram2d(phi_sample, psi_sample, bins=100, density=True, weights=weights)
    h_rew = -kT_kcal * jnp.log(h_rew)
    x_rew, y_rew = jnp.meshgrid(x_edges_rew, y_edges_rew)
    ax_rew.pcolormesh(x_rew, y_rew, h_rew.T, cmap=cmap, vmax=5.25)
    ax_rew.set_xlabel(r"$\phi$")
    ax_rew.set_ylabel(r"$\psi$")
    ax_rew.set_box_aspect(1)
    ax_rew.set_xticks(ticks)
    ax_rew.set_xticklabels(tick_labels)
    ax_rew.set_yticks(ticks)
    ax_rew.set_yticklabels(tick_labels)
    ax_rew.set_xlim(-jnp.pi, jnp.pi)
    ax_rew.set_ylim(-jnp.pi, jnp.pi)
    return ax_ref, ax_prop, ax_rew


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    new_colors = cmap(jnp.linspace(minval, maxval, n))

    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval), new_colors
    )
    return new_cmap


def plot_alanine_ramachandran_paper(
    kT_value: float,
    phi: jnp.ndarray,
    psi: jnp.ndarray,
    ax: plt.Axes,
    weights: Optional[jnp.ndarray] = None,
) -> plt.Axes:
    """Plot reference, proposal, and reweighted Ramachandran free-energy maps."""

    kT_kcal = kT_value / 4.184
    ticks = [-jnp.pi, -jnp.pi / 2, 0, jnp.pi / 2, jnp.pi]
    tick_labels = [r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"]
    cmap = truncate_colormap("gist_earth", 0, 0.93)

    h_ref, x_edges_ref, y_edges_ref = jnp.histogram2d(phi, psi, bins=100, density=True, weights=weights)
    h_ref = -kT_kcal * jnp.log(h_ref)
    x_ref, y_ref = jnp.meshgrid(x_edges_ref, y_edges_ref)
    ax.pcolormesh(x_ref, y_ref, h_ref.T, cmap=cmap, vmax=5.25)
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$\psi$")
    ax.set_box_aspect(1)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels)
    ax.set_xlim(-jnp.pi, jnp.pi)
    ax.set_ylim(-jnp.pi, jnp.pi)
    return ax


def _plot_alanine_dihedral_free_energy(
    ax: plt.Axes,
    kT_value: float,
    ref: jnp.ndarray,
    sample: jnp.ndarray,
    *,
    implicit: Optional[jnp.ndarray] = None,
    weights: Optional[jnp.ndarray] = None,
    xlabel: str,
    legend_kwargs: dict[str, Any] | None = None,
) -> plt.Axes:
    x_grid = jnp.linspace(-jnp.pi, jnp.pi, 300)
    energy_cutoff = 22.5

    samples: list[jnp.ndarray] = [ref]
    labels: list[str] = ["MD Reference (explicit)"]
    colors: list[Any] = [PALETTE_PAPER["Simulation"]]
    styles: list[str] = ["-"]
    weights_list: list[Optional[jnp.ndarray]] = [None]

    if implicit is not None:
        samples.append(implicit)
        labels.append("MD Reference (implicit)")
        colors.append(PALETTE_PAPER["Implicit"])
        styles.append("-")
        weights_list.append(None)

    samples.extend([sample, sample])
    labels.extend(["CG-BG (proposal)", "CG-BG (reweighted)"])
    colors.extend([PALETTE_PAPER["Proposal"], PALETTE_PAPER["Proposal (reweighted)"]])
    styles.extend(["--", "--"])
    weights_list.extend([None, weights])

    _plot_dihedral_fe_panel(
        ax,
        kT_value,
        x_grid,
        tuple(samples),
        tuple(labels),
        tuple(colors),
        tuple(styles),
        weights=tuple(weights_list),
        energy_cutoff=energy_cutoff,
        xlabel=xlabel,
        ylabel=r"Free Energy / $k_B T$",
        legend_kwargs=legend_kwargs,
    )
    return ax


def plot_alanine_free_energy(
    kT_value: float,
    phi_ref: jnp.ndarray,
    phi_sample: jnp.ndarray,
    phi_implicit: Optional[jnp.ndarray],
    psi_ref: jnp.ndarray,
    psi_sample: jnp.ndarray,
    psi_implicit: Optional[jnp.ndarray],
    axes: tuple[plt.Axes, plt.Axes],
    *,
    weights: Optional[jnp.ndarray] = None,
) -> tuple[plt.Axes, plt.Axes]:
    """Plot phi and psi free-energy panels for one alanine system."""

    ax_phi, ax_psi = axes
    _plot_alanine_dihedral_free_energy(
        ax_phi,
        kT_value,
        phi_ref,
        phi_sample,
        implicit=phi_implicit,
        weights=weights,
        xlabel=r"$\phi$",
        legend_kwargs={"loc": "lower right", "alignment": "left", "fontsize": 7},
    )
    _plot_alanine_dihedral_free_energy(
        ax_psi,
        kT_value,
        psi_ref,
        psi_sample,
        implicit=psi_implicit,
        weights=weights,
        xlabel=r"$\psi$",
        legend_kwargs={"loc": "upper right", "alignment": "left", "fontsize": 7},
    )
    return ax_phi, ax_psi


def plot_alanine_phi_free_energy(
    kT_value: float,
    phi_ref: jnp.ndarray,
    phi_sample: jnp.ndarray,
    phi_implicit: Optional[jnp.ndarray],
    axes: tuple[plt.Axes],
    *,
    weights: Optional[jnp.ndarray] = None,
) -> tuple[plt.Axes]:
    """Plot the phi free-energy panel for one alanine system."""

    ax_phi = axes[0]
    _plot_alanine_dihedral_free_energy(
        ax_phi,
        kT_value,
        phi_ref,
        phi_sample,
        implicit=phi_implicit,
        weights=weights,
        xlabel=r"$\phi$",
        legend_kwargs={"loc": "upper left", "alignment": "left"},
    )
    return (ax_phi,)


def plot_alanine_plots(
    variant: str,
    target_path: Path,
    sample_path: Path,
    implicit_path: Path,
    *,
    clip: float,
    kT: float = float(kb * 300),
    run_bootstrap: bool = True,
    compute_torus_w2: bool = False,
    n_bootstraps: int = 500,
) -> dict[str, Any]:
    """Save the alanine figures for one spec selected by name.

    Returns a dict with ``images`` (name -> figure path) and ``metrics`` (scalar means).
    """

    configure_plot_style()
    plot_case = get_alanine_plot_case(variant)
    output_dir = Path("plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    target = compute_alanine_dataset(target_path, plot_case.target_variant, clip=None)
    sample = compute_alanine_dataset(sample_path, plot_case.sample_variant, clip=clip)
    implicit = (
        compute_alanine_dataset(implicit_path, plot_case.implicit_variant, clip=None)
        if plot_case.implicit_variant
        else None
    )

    phi_ref, psi_ref = target.dihedrals[:, 0], target.dihedrals[:, 1]
    phi_sample, psi_sample = sample.dihedrals[:, 0], sample.dihedrals[:, 1]
    phi_implicit = None if implicit is None else implicit.dihedrals[:, 0]
    psi_implicit = None if implicit is None else implicit.dihedrals[:, 1]

    fig_energy, ax_energy = plt.subplots(figsize=(4, 4))
    plot_alanine_energy_density(
        target.energies if target.energies is not None else jnp.zeros((target.R.shape[0],)),
        sample.energies if sample.energies is not None else jnp.zeros((sample.R.shape[0],)),
        None if implicit is None else implicit.energies,
        ax_energy,
        sample.weights,
    )
    energy_path = output_dir / f"{variant}_energy_distribution.png"
    fig_energy.tight_layout()
    fig_energy.savefig(energy_path)
    plt.close(fig_energy)

    fig_density, axes_density = plt.subplots(1, 2, figsize=(8, 4))
    plot_alanine_density(
        phi_ref,
        phi_sample,
        psi_ref,
        psi_sample,
        axes_density,
        phi_implicit=phi_implicit,
        psi_implicit=psi_implicit,
        weights=sample.weights,
    )
    density_path = output_dir / f"{variant}_density.png"
    fig_density.tight_layout()
    fig_density.savefig(density_path)
    plt.close(fig_density)

    fig_free, axes_free = plt.subplots(1, 2, figsize=(8, 4))
    plot_alanine_free_energy(
        kT,
        phi_ref,
        phi_sample,
        phi_implicit,
        psi_ref,
        psi_sample,
        psi_implicit,
        axes_free,
        weights=sample.weights,
    )
    free_path = output_dir / f"{variant}_free_energy.png"
    fig_free.tight_layout()
    fig_free.savefig(free_path)
    plt.close(fig_free)

    fig_ram, axes_ram = plt.subplots(1, 3, figsize=(12, 4))
    plot_alanine_ramachandran_fes(kT, phi_sample, psi_sample, phi_ref, psi_ref, axes_ram, sample.weights)
    ram_path = output_dir / f"{variant}_ramachandran_fes.png"
    fig_ram.tight_layout()
    fig_ram.savefig(ram_path)
    plt.close(fig_ram)

    console = get_console()
    console.print(f"[green]Saved alanine plots for variant[/green] '{variant}' in directory: {output_dir.resolve()}")

    images = {
        "energy_distribution": energy_path,
        "density": density_path,
        "free_energy": free_path,
        "ramachandran_fes": ram_path,
    }
    metrics: dict[str, float] = {}

    if run_bootstrap:
        key = jax.random.PRNGKey(0)
        weighted_bootstrap_results = run_bootstrap_metrics(
            key,
            target.dihedrals,
            sample.dihedrals,
            sample.weights,
            limits=((-jnp.pi, jnp.pi), (-jnp.pi, jnp.pi)),
            n_bootstraps=n_bootstraps,
            compute_torus_w2=compute_torus_w2,
        )
        unweighted_bootstrap_results = run_bootstrap_metrics(
            key,
            target.dihedrals,
            sample.dihedrals,
            None,
            limits=((-jnp.pi, jnp.pi), (-jnp.pi, jnp.pi)),
            n_bootstraps=n_bootstraps,
            compute_torus_w2=compute_torus_w2,
        )
        weighted_direct_results = compute_alanine_metric_summary(
            target.dihedrals,
            sample.dihedrals,
            sample.weights,
            limits=((-jnp.pi, jnp.pi), (-jnp.pi, jnp.pi)),
        )
        unweighted_direct_results = compute_alanine_metric_summary(
            target.dihedrals,
            sample.dihedrals,
            None,
            limits=((-jnp.pi, jnp.pi), (-jnp.pi, jnp.pi)),
        )
        metrics_file = output_dir / f"{variant}_bootstrap_metrics.txt"
        write_bootstrap_metrics_report(
            weighted_bootstrap_results,
            metrics_file,
            label=variant,
            n_bootstraps=n_bootstraps,
            direct_results=weighted_direct_results,
            target_count=int(target.dihedrals.shape[0]),
            sample_count=int(sample.dihedrals.shape[0]),
            without_reweight_results=unweighted_bootstrap_results,
            without_reweight_direct_results=unweighted_direct_results,
            without_reweight_target_count=int(target.dihedrals.shape[0]),
            without_reweight_sample_count=int(sample.dihedrals.shape[0]),
            without_reweight_n_bootstraps=n_bootstraps,
        )
        print_bootstrap_metrics(
            weighted_bootstrap_results,
            label=f"{variant} (reweight)",
            n_bootstraps=n_bootstraps,
            direct_results=weighted_direct_results,
            target_count=int(target.dihedrals.shape[0]),
            sample_count=int(sample.dihedrals.shape[0]),
        )
        print_bootstrap_metrics(
            unweighted_bootstrap_results,
            label=f"{variant} (without reweight)",
            n_bootstraps=n_bootstraps,
            direct_results=unweighted_direct_results,
            target_count=int(target.dihedrals.shape[0]),
            sample_count=int(sample.dihedrals.shape[0]),
        )

        for metric_name, values in weighted_bootstrap_results.items():
            metrics[f"eval/{variant}/{metric_name}"] = float(jnp.mean(values))

    return {"images": images, "metrics": metrics}


def plot_alanine_clip_sweep(
    variant: str | Sequence[str],
    target_path: Path | Sequence[Path],
    sample_path: Path | Sequence[Path],
    clip_percentiles: Sequence[float],
    *,
    output_path: Path | None = None,
    report_clip_percentiles: Sequence[float] | None = None,
    sample_labels: Sequence[str] | None = None,
    target_labels: Sequence[str] | None = None,
) -> Path:
    """Print reweighted metrics versus clip percentile and save the sweep plot.

    The console always shows the best available clip for each metric:
    JS min, PMF min, and ESS max.
    If report_clip_percentiles is provided, it additionally prints the metrics
    for those specific clip percentiles.

    Args:
        variant: Either a single alanine variant or a sequence aligned with
            target_path/sample_path. Each variant is plotted as its own curve.
        target_path: Either a single .npz path or a sequence of paths. When a
            sequence is provided, it must match the number of sample paths.
        sample_path: Either a single .npz path or a sequence of paths. When a
            sequence is provided, each dataset is plotted as a separate curve.
        sample_labels: Optional labels for each dataset. If omitted, labels are
            derived from the sample file stems.
        target_labels: Optional labels for each target dataset. If omitted,
            labels are derived from the target file stems.
    """

    clips = tuple(float(clip) for clip in clip_percentiles)
    if len(clips) == 0:
        raise ValueError("clip_percentiles must contain at least one percentile")

    def _normalize_inputs(
        path_input: Path | Sequence[Path],
        labels: Sequence[str] | None,
        *,
        label_name: str,
    ) -> tuple[list[Path], list[str]]:
        if isinstance(path_input, Path):
            paths = [path_input]
        else:
            paths = list(path_input)
        if not paths:
            raise ValueError(f"{label_name} must contain at least one path")
        if labels is None:
            resolved_labels = [path.stem for path in paths]
        else:
            resolved_labels = list(labels)
            if len(resolved_labels) != len(paths):
                raise ValueError(f"{label_name} labels must match the number of paths")
        return paths, resolved_labels

    def _normalize_variants(variant_input: str | Sequence[str]) -> list[str]:
        if isinstance(variant_input, str):
            return [variant_input]
        variants = list(variant_input)
        if not variants:
            raise ValueError("variant must contain at least one entry")
        return variants

    variants = _normalize_variants(variant)

    def _prepare_sample_data(path: Path, sample_variant: str) -> tuple[jnp.ndarray, jnp.ndarray, bool]:
        data = load_npz_file(path)
        if "R" not in data:
            raise ValueError(f"Expected an 'R' array in {path}")
        coords = jnp.asarray(data["R"])
        phi, psi = compute_alanine_dihedrals(coords, sample_variant)
        dihedrals = jnp.stack([phi, psi], axis=1)
        logw_base = None
        if "logw" in data:
            logw_base = jnp.asarray(data["logw"])
        elif "logp" in data and "U" in data:
            logw_base = -jnp.asarray(data["U"]) / (kb * 300) - jnp.asarray(data["logp"])

        if logw_base is None:
            weights_uniform = jnp.full((coords.shape[0],), 1.0 / coords.shape[0])
            return dihedrals, weights_uniform, False
        return dihedrals, logw_base, True

    def _scan_metrics(
        target_dihedrals: jnp.ndarray,
        dihedrals: jnp.ndarray,
        weight_source: jnp.ndarray,
        has_log_weights: bool,
    ) -> jnp.ndarray:
        limits = ((-jnp.pi, jnp.pi), (-jnp.pi, jnp.pi))

        def _step(carry: None, clip_value: jnp.ndarray) -> tuple[None, jnp.ndarray]:
            if has_log_weights:
                weights = clip_weights(weight_source, clip=clip_value)
            else:
                weights = weight_source
            weighted_summary = compute_alanine_metric_summary(
                target_dihedrals,
                dihedrals,
                weights,
                limits=limits,
            )
            metrics = jnp.array(
                [
                    weighted_summary["JS_Divergence"],
                    weighted_summary["PMF_Error"],
                    weighted_summary["ESS_Percent"],
                ]
            )
            return carry, metrics

        _, metrics_over_clips = jax.lax.scan(_step, None, jnp.asarray(clips))
        return metrics_over_clips

    target_paths, target_names = _normalize_inputs(
        target_path,
        target_labels,
        label_name="target_path",
    )
    sample_paths, sample_names = _normalize_inputs(
        sample_path,
        sample_labels,
        label_name="sample_path",
    )
    if len(sample_paths) != len(variants):
        raise ValueError("variant must match the number of sample paths")
    if len(target_paths) not in (1, len(sample_paths)):
        raise ValueError("target_path must be a single path or match the number of sample paths")

    metric_names = ["JS_Divergence", "PMF_Error", "ESS_Percent"]
    all_metrics: list[jnp.ndarray] = []
    for idx, path in enumerate(sample_paths):
        target_idx = 0 if len(target_paths) == 1 else idx
        plot_case = get_alanine_plot_case(variants[idx])
        if plot_case.target_variant not in ALANINE_PLOT_CASES:
            valid = ", ".join(sorted(ALANINE_PLOT_CASES))
            raise ValueError(f"plot_alanine_clip_sweep only supports alanine variants. Choose from: {valid}")
        target = compute_alanine_dataset(target_paths[target_idx], plot_case.target_variant, clip=None)
        dihedrals, weight_source, has_log_weights = _prepare_sample_data(path, plot_case.sample_variant)
        metrics_over_clips = _scan_metrics(target.dihedrals, dihedrals, weight_source, has_log_weights)
        all_metrics.append(metrics_over_clips)

    if not all_metrics:
        raise ValueError("No metrics were produced for plot_alanine_clip_sweep")

    def _value_for_clip(values: jnp.ndarray, clip: float) -> float | None:
        matches = np.where(np.isclose(np.asarray(clips), clip, rtol=0.0, atol=1e-6))[0]
        if matches.size == 0:
            return None
        return float(np.asarray(values)[int(matches[0])])

    def _format_clip_value(clip: float) -> str:
        return f"{clip:.4f}".rstrip("0").rstrip(".")

    def _summary_row(
        metric_label: str,
        values: jnp.ndarray,
        *,
        only_min: bool = False,
        only_max: bool = False,
        ignore_zero_clip: bool = False,
    ) -> tuple[str, str, str, str]:
        finite_mask = jnp.isfinite(values)
        if ignore_zero_clip:
            finite_mask = finite_mask & (jnp.asarray(clips) != 0)
        finite_values = values[finite_mask]
        finite_clips = jnp.asarray(clips)[finite_mask]
        if finite_values.size == 0:
            raise ValueError(f"No finite values were produced for {metric_label} in plot_alanine_clip_sweep")

        min_idx = int(jnp.argmin(finite_values))
        max_idx = int(jnp.argmax(finite_values))
        if only_max:
            return (
                metric_label,
                f"{float(finite_values[max_idx]):.6f}",
                _format_clip_value(float(finite_clips[max_idx])),
                "max",
            )
        return (
            metric_label,
            f"{float(finite_values[min_idx]):.6f}",
            _format_clip_value(float(finite_clips[min_idx])),
            "min",
        )

    def _render_summary_table(label: str, metrics: jnp.ndarray) -> Table:
        table = Table(title=f"Best Clip Percentile Summary: {label}")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_column("Clip percentile", justify="right")
        table.add_column("Extremum", justify="center")
        table.caption = "Default behavior: JS min, PMF min, ESS max"
        table.add_row(*_summary_row("JS Divergence", metrics[:, 0], only_min=True))
        table.add_row(*_summary_row("PMF Error", metrics[:, 1], only_min=True))
        table.add_row(*_summary_row("ESS Percent", metrics[:, 2], only_max=True, ignore_zero_clip=True))
        return table

    def _render_selected_clip_table(label: str, metrics: jnp.ndarray, selected_clips: Sequence[float]) -> Table:
        table = Table(title=f"Selected Clip Percentile Metrics: {label}")
        table.add_column("Clip percentile", justify="right")
        table.add_column("JS Divergence", justify="right")
        table.add_column("PMF Error", justify="right")
        table.add_column("ESS Percent", justify="right")

        for clip in selected_clips:
            clip_value = float(clip)
            js_value = _value_for_clip(metrics[:, 0], clip_value)
            pmf_value = _value_for_clip(metrics[:, 1], clip_value)
            ess_value = _value_for_clip(metrics[:, 2], clip_value)
            table.add_row(
                _format_clip_value(clip_value),
                "n/a" if js_value is None else f"{js_value:.6f}",
                "n/a" if pmf_value is None else f"{pmf_value:.6f}",
                "n/a" if ess_value is None else f"{ess_value:.6f}",
            )
        return table

    console = get_console()
    variants_label = ", ".join(variants)
    console.print(f"\n[bold]Clip Sweep Results for {variants_label}[/bold]")
    report_clips = None
    if report_clip_percentiles is not None:
        report_clips = list(report_clip_percentiles)
        if not report_clips:
            raise ValueError("report_clip_percentiles must contain at least one percentile when provided")

    for idx, (label, metrics) in enumerate(zip(sample_names, all_metrics)):
        target_idx = 0 if len(target_names) == 1 else idx
        label_with_target = f"{label} (variant={variants[idx]}, target={target_names[target_idx]})"
        console.print(_render_summary_table(label_with_target, metrics))
        if report_clips is not None:
            console.print(_render_selected_clip_table(label_with_target, metrics, report_clips))

    if output_path is None:
        if len(variants) == 1:
            output_path = Path("plots") / f"{variants[0]}_clip_sweep.png"
        else:
            output_path = Path("plots") / "clip_sweep_multi.png"

    get_console().print(f"[green]Printed alanine clip sweep for variant[/green] '{variant}'")
    return {
        "clips": jnp.asarray(clips),
        "metric_names": metric_names,
        "all_metrics": all_metrics,
        "sample_names": sample_names,
        "variants": variants,
        "target_names": target_names,
        "output_path": output_path,
    }


def V(x: jnp.ndarray, y: jnp.ndarray, biased: bool = False) -> jnp.ndarray:
    """Muller-Brown potential used by the MB plots."""

    term1 = -17.3 * jnp.exp(-0.0039 * (x - 48) ** 2 - 0.0391 * (y - 8) ** 2)
    term2 = -8.7 * jnp.exp(-0.0039 * (x - 32) ** 2 - 0.0391 * (y - 16) ** 2)
    term3 = -14.7 * jnp.exp(-0.0254 * (x - 24) ** 2 + 0.043 * (x - 24) * (y - 32) - 0.0254 * (y - 32) ** 2)
    term4 = 1.3 * jnp.exp(0.00273 * (x - 16) ** 2 + 0.0023 * (x - 16) * (y - 24) + 0.00273 * (y - 24) ** 2)
    if biased:
        x0 = 32.0
        width = 5.0
        height = -4.0
        bias_potential = height * jnp.exp(-((x - x0) ** 2) / (2 * width**2))
        return term1 + term2 + term3 + term4 + bias_potential
    return term1 + term2 + term3 + term4


def plot_mb(
    kT: float,
    x_ref: jnp.ndarray,
    x_gen: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
    axes: tuple[plt.Axes, plt.Axes] | None = None,
) -> tuple[plt.Axes, plt.Axes]:
    """Plot the two-panel Muller-Brown figure used in the notebook."""

    if axes is None:
        raise ValueError("axes must be provided for plot_mb")
    ax_pdf, ax_fe = axes

    x_vals = jnp.linspace(0, 50, 300)
    y_vals = jnp.linspace(0, 50, 300)

    def P(xi: jnp.ndarray) -> jnp.ndarray:
        V_y = jax.vmap(lambda y: V(xi, y))(y_vals)
        return jnp.trapezoid(jnp.exp(-V_y / kT), y_vals)

    P_x = jax.vmap(P)(x_vals)
    P_x /= jnp.trapezoid(P_x, x_vals)
    F_true = -kT * jnp.log(P_x)
    F_true -= jnp.min(F_true)

    F_ref = compute_free_energy_1d(x_ref, x_vals, kT_value=kT)
    F_gen = compute_free_energy_1d(x_gen, x_vals, kT_value=kT)
    F_rew = compute_free_energy_1d(x_gen, x_vals, weights=weights, kT_value=kT)

    ax_pdf.plot(x_vals, P_x, color=PALETTE_PAPER["Exact"], label="Exact")
    _hist(ax_pdf, x_ref, bins=100, color=PALETTE_PAPER["Simulation"], label="MD Reference")
    _hist(ax_pdf, x_gen, bins=100, color=PALETTE_PAPER["Proposal"], label="CG-BG (proposal)", alpha=0.4)
    _hist(
        ax_pdf,
        x_gen,
        bins=100,
        color=PALETTE_PAPER["Proposal (reweighted)"],
        label="CG-BG (reweighted)",
        edgecolor=PALETTE_PAPER["Proposal (reweighted)"],
        weights=weights,
        histtype="step",
        alpha=1.0,
    )
    ax_pdf.set_xlabel(r"$x$")
    ax_pdf.set_ylabel(r"$P(x)$")
    ax_pdf.set_xlim(10, 50)
    ax_pdf.set_ylim(-0.01, 0.2)
    ax_pdf.set_box_aspect(1)
    ax_pdf.legend()

    ax_fe.plot(x_vals, F_true, color=PALETTE_PAPER["Exact"], linestyle="-", label="Exact")
    ax_fe.plot(x_vals, F_ref, color=PALETTE_PAPER["Simulation"], linestyle="-", label="MD Reference")
    ax_fe.plot(x_vals, F_gen, color=PALETTE_PAPER["Proposal"], linestyle="--", label="CG-BG (proposal)")
    ax_fe.plot(x_vals, F_rew, color=PALETTE_PAPER["Proposal (reweighted)"], linestyle="--", label="CG-BG (reweighted)")
    ax_fe.set_xlim(10, 50)
    ax_fe.set_ylim(-0.5, 15)
    ax_fe.set_xlabel(r"$x$")
    ax_fe.set_ylabel(r"Free Energy / $k_B T$")
    ax_fe.legend(loc="best")
    ax_fe.set_yticks([0, 5, 10, 15])
    ax_fe.set_box_aspect(1)
    return ax_pdf, ax_fe


def plot_mb_plots(
    target_path: Path,
    sample_path: Path,
    *,
    kT: float = 1.0,
) -> dict[str, Any]:

    sample = jnp.load(sample_path, allow_pickle=True)
    weights = None
    if "logw" in sample:
        weights = clip_weights(jnp.asarray(sample["logw"]))
    x_gen = jnp.squeeze(jnp.asarray(sample["R"]))

    target = jnp.load(target_path, allow_pickle=True)
    x_ref = jnp.squeeze(target["R"])

    output_dir = Path("plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "mb_plots.png"
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    plot_mb(kT, x_ref, x_gen, weights, axes)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    get_console().print(f"[green]Saved Muller-Brown plots to[/green] {output_path}")

    return {"images": {"mb_plots": output_path}, "metrics": {}}
