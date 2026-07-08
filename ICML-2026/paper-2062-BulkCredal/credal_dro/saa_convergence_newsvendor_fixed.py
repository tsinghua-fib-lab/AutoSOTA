from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _Path

    _PKG_DIR = _Path(__file__).resolve().parent
    sys.path.insert(0, str(_PKG_DIR.parent))
    __package__ = _PKG_DIR.name

import argparse
import copy
import json
import math
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .bayes_conjugates import (
    default_prior_params,
    get_posterior_params,
    posterior_predictive_params,
    sample_posterior_predictive,
)
from .dataset import _student_t_base_params, sample_dgp
from .lv_bulk_set import build_score, dkw_select_threshold


def _make_pc_sampler(*, likelihood: str, posterior: str, pp_params: object, dim: int, rng: np.random.Generator):
    def _pc_sampler(n: int) -> np.ndarray:
        draws = sample_posterior_predictive(
            likelihood,
            posterior,
            pp_params,
            dim,
            int(n),
            generator=rng,
        )
        return np.asarray(draws, dtype=float).reshape(-1, dim)

    return _pc_sampler


def _ellipsoid_score_from_params(mu: np.ndarray, Sigma: np.ndarray):
    """Literal copy of the local helper used in the current main.py newsvendor branch."""
    mu = np.asarray(mu, dtype=float).reshape(-1)
    d = int(mu.size)
    Sigma = np.asarray(Sigma, dtype=float).reshape(d, d)
    Sigma = 0.5 * (Sigma + Sigma.T) + 1e-12 * np.eye(d)
    L = np.linalg.cholesky(Sigma)

    def score_fn(xi_arr: np.ndarray) -> np.ndarray:
        X = np.asarray(xi_arr, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        D = (X - mu).T
        Z = np.linalg.solve(L, D)
        return np.sqrt(np.sum(Z * Z, axis=0))

    meta = {"type": "ellipsoid", "mu": mu, "Sigma": Sigma}
    return score_fn, meta


def _rejection_sample_truncated_pc(
    *,
    pc_sampler,
    score_fn,
    t_hat: float,
    n_accept: int,
    rng_fill: np.random.Generator,
    warn_draw_threshold: int | None = None,
    draw_cap_factor: int = 5000,
) -> tuple[np.ndarray, int, int, float]:
    """Mirror the LV-BAS newsvendor rejection sampler from the current main.py."""
    if n_accept <= 0:
        raise ValueError(f"n_accept must be positive, got {n_accept}.")

    accepted: list[np.ndarray] = []
    n_acc = 0
    total_draws = 0
    max_draws = int(draw_cap_factor * n_accept)

    while n_acc < n_accept and total_draws < max_draws:
        remaining = n_accept - n_acc
        batch_size = max(2 * remaining, 256)
        xi_batch = pc_sampler(batch_size)
        mask = score_fn(xi_batch) <= float(t_hat) + 1e-10
        if np.any(mask):
            acc = np.asarray(xi_batch[mask], dtype=float)
            accepted.append(acc)
            n_acc += int(acc.shape[0])
        total_draws += int(batch_size)

    if warn_draw_threshold is not None and total_draws > int(warn_draw_threshold):
        warnings.warn(
            f"LV-BAS newsvendor: required {total_draws} posterior predictive draws to get {n_accept} accepted samples; "
            "Pc might have small mass on the bulk set.",
            UserWarning,
        )

    if not accepted:
        raise RuntimeError(
            "LV-BAS newsvendor: could not obtain any accepted posterior predictive samples in Xi_0."
        )

    xi_acc = np.concatenate(accepted, axis=0)
    accepted_before_trunc = int(xi_acc.shape[0])
    if accepted_before_trunc < n_accept:
        warnings.warn(
            f"LV-BAS newsvendor: accepted only {accepted_before_trunc}/{n_accept} points inside Xi_0; "
            "filling remainder by resampling with replacement from the accepted set.",
            UserWarning,
        )
        extra = rng_fill.choice(xi_acc.shape[0], size=n_accept - xi_acc.shape[0], replace=True)
        xi_acc = np.vstack([xi_acc, xi_acc[extra]])

    xi_trunc = np.asarray(xi_acc[:n_accept], dtype=float)
    accept_rate = float(accepted_before_trunc) / float(total_draws) if total_draws > 0 else 0.0
    return xi_trunc, int(total_draws), int(accepted_before_trunc), float(accept_rate)


def _default_gamma_bulks(n_select: int, delta: float = 0.05) -> list[float]:
    gamma_min = math.sqrt(math.log(2.0 / float(delta)) / (2.0 * float(n_select)))
    grid = [0.05, 0.075, 0.10, 0.15]
    return sorted({float(g) for g in grid})


def _build_bulk_and_pc_samplers(
    *,
    replication: int,
    data: np.ndarray,
    gamma_bulk: float,
    dkw_delta: float,
    posterior: str,
    likelihood: str,
    dim: int,
    theta_posterior_full: object,
    lv_use_pc_bulk_geometry: bool,
):
    """Replicate the current main.py bulk calibration / P_c construction for newsvendor."""
    n_in = int(data.shape[0])
    rng_lv = np.random.default_rng(seed=int(replication))

    perm = rng_lv.permutation(n_in)
    train_size = n_in // 2
    train_xi = np.asarray(data[perm[:train_size]], dtype=float)
    selection_xi = np.asarray(data[perm[train_size:]], dtype=float)

    if lv_use_pc_bulk_geometry:
        theta_prior_pc = default_prior_params(posterior, dim=dim)
        theta_posterior_pc_template = get_posterior_params(posterior, train_xi, theta_prior_pc)

        pp_pc_run = posterior_predictive_params(posterior, copy.deepcopy(theta_posterior_pc_template))
        pc_sampler_run = _make_pc_sampler(
            likelihood=likelihood,
            posterior=posterior,
            pp_params=pp_pc_run,
            dim=dim,
            rng=rng_lv,
        )

        mc_n = int(max(2000, min(20000, 20 * dim * dim)))
        mc = pc_sampler_run(mc_n)
        mu_pc = mc.mean(axis=0)
        Xc = mc - mu_pc
        Sigma_pc = (Xc.T @ Xc) / max(1, mc.shape[0] - 1)
        Sigma_pc += 1e-8 * np.eye(dim)
        score_fn, score_meta = _ellipsoid_score_from_params(mu_pc, Sigma_pc)

        ref_seed = int(1 + 104729 * int(replication) + round(1_000_000 * float(gamma_bulk)))
        rng_ref = np.random.default_rng(seed=ref_seed)
        pp_pc_ref = posterior_predictive_params(posterior, copy.deepcopy(theta_posterior_pc_template))
        pc_sampler_ref = _make_pc_sampler(
            likelihood=likelihood,
            posterior=posterior,
            pp_params=pp_pc_ref,
            dim=dim,
            rng=rng_ref,
        )
    else:
        score_fn, score_meta = build_score(train_xi, score_type="ellipsoid")

        pp_full_run = posterior_predictive_params(posterior, copy.deepcopy(theta_posterior_full))
        pc_sampler_run = _make_pc_sampler(
            likelihood=likelihood,
            posterior=posterior,
            pp_params=pp_full_run,
            dim=dim,
            rng=rng_lv,
        )

        ref_seed = int(1 + 104729 * int(replication) + round(1_000_000 * float(gamma_bulk)))
        rng_ref = np.random.default_rng(seed=ref_seed)
        pp_full_ref = posterior_predictive_params(posterior, copy.deepcopy(theta_posterior_full))
        pc_sampler_ref = _make_pc_sampler(
            likelihood=likelihood,
            posterior=posterior,
            pp_params=pp_full_ref,
            dim=dim,
            rng=rng_ref,
        )

    audit_scores = score_fn(selection_xi)
    dkw_info = dkw_select_threshold(audit_scores, gamma=float(gamma_bulk), delta=float(dkw_delta))

    if not dkw_info.get("exists", False) or not np.isfinite(dkw_info.get("t_hat", np.nan)):
        t_hat = float(np.max(audit_scores)) if audit_scores.size else 0.0
        warnings.warn(
            f"DKW certificate for gamma={gamma_bulk}, delta={dkw_delta} does not exist; "
            f"using t_hat=max(score) (r={dkw_info.get('r', np.nan)}).",
            UserWarning,
        )
    else:
        t_hat = float(dkw_info["t_hat"])

    return {
        "train_xi": train_xi,
        "selection_xi": selection_xi,
        "score_fn": score_fn,
        "score_meta": score_meta,
        "t_hat": t_hat,
        "dkw_info": dict(dkw_info),
        "pc_sampler_run": pc_sampler_run,
        "pc_sampler_ref": pc_sampler_ref,
    }


def _analyse_one_gamma(
    *,
    replication: int,
    data: np.ndarray,
    theta_posterior: object,
    gamma_bulk: float,
    num_likelihood_samples: int,
    reference_n_accept: int,
    dim: int,
    posterior: str,
    likelihood: str,
    dkw_delta: float,
    lv_use_pc_bulk_geometry: bool,
) -> tuple[pd.DataFrame, dict]:
    """One gamma_bulk analysis mirroring the current LV-BAS newsvendor path."""
    build = _build_bulk_and_pc_samplers(
        replication=replication,
        data=data,
        gamma_bulk=gamma_bulk,
        dkw_delta=dkw_delta,
        posterior=posterior,
        likelihood=likelihood,
        dim=dim,
        theta_posterior_full=theta_posterior,
        lv_use_pc_bulk_geometry=lv_use_pc_bulk_geometry,
    )

    score_fn = build["score_fn"]
    score_meta = build["score_meta"]
    t_hat = float(build["t_hat"])
    dkw_info = dict(build["dkw_info"])
    selection_xi = np.asarray(build["selection_xi"], dtype=float)
    train_xi = np.asarray(build["train_xi"], dtype=float)

    n_accept = int(num_likelihood_samples * 0.5)
    if n_accept <= 0:
        raise ValueError("num_likelihood_samples must be positive for LV-BAS newsvendor.")

    rng_fill_run = np.random.default_rng(seed=int(4_000_003 + replication))
    xi_trunc, total_draws, accepted_before_trunc, empirical_accept_rate = _rejection_sample_truncated_pc(
        pc_sampler=build["pc_sampler_run"],
        score_fn=score_fn,
        t_hat=t_hat,
        n_accept=n_accept,
        rng_fill=rng_fill_run,
        warn_draw_threshold=int(num_likelihood_samples),
    )

    ref_n_accept = max(int(reference_n_accept), int(n_accept))
    rng_fill_ref = np.random.default_rng(seed=int(8_000_011 + replication))
    xi_ref, ref_total_draws, ref_accepted_before_trunc, reference_accept_rate = _rejection_sample_truncated_pc(
        pc_sampler=build["pc_sampler_ref"],
        score_fn=score_fn,
        t_hat=t_hat,
        n_accept=ref_n_accept,
        rng_fill=rng_fill_ref,
        warn_draw_threshold=None,
    )
    reference_mean = np.asarray(xi_ref, dtype=float).mean(axis=0)
    reference_mean_avg = float(np.mean(reference_mean))
    reference_mean_norm = float(np.linalg.norm(reference_mean))

    cum = np.cumsum(xi_trunc, axis=0)
    sizes = np.arange(1, xi_trunc.shape[0] + 1, dtype=int)
    running_mean = cum / sizes[:, None]
    running_mean_avg = running_mean.mean(axis=1)
    running_mean_norm = np.linalg.norm(running_mean, axis=1)
    error_norm = np.linalg.norm(running_mean - reference_mean[None, :], axis=1)

    running_df = pd.DataFrame(
        {
            "gamma_bulk": float(gamma_bulk),
            "sample_size": sizes,
            "running_mean_avg": running_mean_avg,
            "reference_mean_avg": reference_mean_avg,
            "running_mean_norm": running_mean_norm,
            "reference_mean_norm": reference_mean_norm,
            "error_norm": error_norm,
            "reference_accept_rate": float(reference_accept_rate),
            "empirical_accept_rate": float(empirical_accept_rate),
            "accepted_before_trunc": int(accepted_before_trunc),
            "rejection_total_draws": int(total_draws),
            "reference_n_accept": int(ref_n_accept),
            "reference_accepted_before_trunc": int(ref_accepted_before_trunc),
            "reference_total_draws": int(ref_total_draws),
            "t_hat": float(t_hat),
            "dkw_exists": bool(dkw_info.get("exists", False)),
            "dkw_r": float(dkw_info.get("r", np.nan)),
        }
    )
    for j in range(dim):
        running_df[f"running_mean_{j+1}"] = running_mean[:, j]
        running_df[f"reference_mean_{j+1}"] = float(reference_mean[j])

    train_bulk_rate = float(np.mean(score_fn(train_xi) <= t_hat + 1e-10))
    select_bulk_rate = float(np.mean(score_fn(selection_xi) <= t_hat + 1e-10))

    summary = {
        "gamma_bulk": float(gamma_bulk),
        "n_accept": int(n_accept),
        "reference_n_accept": int(ref_n_accept),
        "reference_accept_rate": float(reference_accept_rate),
        "empirical_accept_rate": float(empirical_accept_rate),
        "accepted_before_trunc": int(accepted_before_trunc),
        "rejection_total_draws": int(total_draws),
        "reference_accepted_before_trunc": int(ref_accepted_before_trunc),
        "reference_total_draws": int(ref_total_draws),
        "t_hat": float(t_hat),
        "dkw_exists": bool(dkw_info.get("exists", False)),
        "dkw_r": float(dkw_info.get("r", np.nan)),
        "dkw_n": int(selection_xi.shape[0]),
        "dkw_delta": float(dkw_delta),
        "reference_mean_avg": reference_mean_avg,
        "reference_mean_norm": reference_mean_norm,
        "train_bulk_rate": train_bulk_rate,
        "select_bulk_rate": select_bulk_rate,
        "lv_use_pc_bulk_geometry": bool(lv_use_pc_bulk_geometry),
    }
    for j in range(dim):
        summary[f"reference_mean_{j+1}"] = float(reference_mean[j])
        summary[f"mu_bulk_{j+1}"] = float(np.asarray(score_meta["mu"], dtype=float)[j])
    Sigma_bulk = np.asarray(score_meta["Sigma"], dtype=float).reshape(dim, dim)
    iu = np.triu_indices(dim)
    for k, (i, j) in enumerate(zip(iu[0], iu[1]), start=1):
        summary[f"Sigma_bulk_triu_{k}"] = float(Sigma_bulk[i, j])

    return running_df, summary


def _pick_reference_column(running_df: pd.DataFrame) -> str:
    if "theoretical_mean_avg" in running_df.columns:
        return "theoretical_mean_avg"
    if "reference_mean_avg" in running_df.columns:
        return "reference_mean_avg"
    raise ValueError("Running CSV must contain either 'theoretical_mean_avg' or 'reference_mean_avg'.")



def _make_plot(running_df: pd.DataFrame, summary_df: pd.DataFrame, out_path: Path) -> None:
    plot_rc = {
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "font.size": 22,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{lmodern}\usepackage{amsmath}\usepackage{amssymb}",
        "font.family": "serif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }

    LABEL_FS = 22
    TICK_FS = 20
    LEGEND_FS = 20

    with plt.rc_context(plot_rc):
        fig, ax = plt.subplots(figsize=(10.5, 6.5))
        ref_col = _pick_reference_column(running_df)

        gamma_values = sorted(float(v) for v in summary_df["gamma_bulk"].dropna().unique().tolist())
        cmap = plt.get_cmap("viridis")
        xs = np.linspace(0.1, 0.9, len(gamma_values)) if len(gamma_values) > 1 else np.array([0.5])

        n_replications = int(running_df["replication"].nunique()) if "replication" in running_df.columns else 1

        for gamma_bulk, xcol in zip(gamma_values, xs):
            color = cmap(float(xcol))

            g_run = running_df[running_df["gamma_bulk"] == float(gamma_bulk)].copy()
            ref_val = np.abs(g_run[ref_col].to_numpy(dtype=float))
            denom = np.maximum(ref_val, 1e-18)
            g_run["rel_abs_error_avg"] = np.abs(
                g_run["running_mean_avg"].to_numpy(dtype=float)
                - g_run[ref_col].to_numpy(dtype=float)
            ) / denom

            g_plot = (
                g_run.groupby("sample_size", dropna=False)
                .agg(
                    mean_rel_abs_error=("rel_abs_error_avg", "mean"),
                    sd_rel_abs_error=("rel_abs_error_avg", "std"),
                    n=("rel_abs_error_avg", "count"),
                )
                .reset_index()
                .sort_values("sample_size")
            )

            se = g_plot["sd_rel_abs_error"].fillna(0.0).to_numpy(dtype=float) / np.sqrt(
                np.maximum(g_plot["n"].to_numpy(dtype=float), 1.0)
            )
            ci = 1.96 * se
            y = g_plot["mean_rel_abs_error"].to_numpy(dtype=float)
            y_low = np.maximum(0.0, y - ci)
            y_high = y + ci
            x = g_plot["sample_size"].to_numpy(dtype=float)

            g_sum = summary_df[summary_df["gamma_bulk"] == float(gamma_bulk)].copy()
            mean_ref_acc = float(g_sum["reference_accept_rate"].mean())

            label = rf"$\gamma={gamma_bulk:g}$, accept rate={100.0 * mean_ref_acc:.1f}\%"

            ax.plot(
                x,
                y,
                linewidth=3.0,
                color=color,
                label=label,
            )
            ax.fill_between(
                x,
                y_low,
                y_high,
                color=color,
                alpha=0.18,
                linewidth=0.0,
            )

        ax.axhline(0.05, color="black", linestyle="--", linewidth=1.5)
        ax.set_xlabel(r"Accepted sample size $N$", fontsize=22)
        ax.set_ylabel("Mean absolute error / theoretical value", fontsize=22)
        ax.set_title(
            rf"SAA convergence over {n_replications} independent replications (95\% CI)",
            fontsize=22,
        )
        ax.tick_params(axis="both", labelsize=20)
        ax.legend(fontsize=20, ncols=1)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, format="pdf", bbox_inches="tight")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAA convergence analysis for LV-BAS newsvendor truncation.")
    parser.add_argument("--replication", type=int, default=0)
    parser.add_argument("--num-replications", type=int, default=100)
    parser.add_argument("--num-observations", type=int, default=2000)
    parser.add_argument("--dim", type=int, default=5)
    parser.add_argument("--num-likelihood-samples", type=int, default=2500)
    parser.add_argument("--reference-n-accept", type=int, default=10000)
    parser.add_argument("--posterior", type=str, default="student_t_niw")
    parser.add_argument("--likelihood", type=str, default="multivariate_student_t")
    parser.add_argument("--dgp", type=str, default="student_t")
    parser.add_argument("--dkw-delta", type=float, default=0.05)
    parser.add_argument("--quad-r", type=int, default=20)
    parser.add_argument("--quad-phi", type=int, default=8)
    parser.add_argument("--quad-theta", type=int, default=16)
    parser.add_argument("--lv-use-pc-bulk-geometry", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("SAA-convergence"))
    parser.add_argument(
        "--gamma-bulk",
        dest="gamma_bulks",
        type=float,
        nargs="*",
        default=None,
        help="Optional explicit gamma_bulk grid. If omitted, a default grid including the minimal DKW-certifiable gamma is used.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dgp != "student_t":
        raise ValueError(f"This standalone analysis is only implemented for dgp='student_t', got {args.dgp!r}.")
    if args.posterior != "student_t_niw":
        raise ValueError(
            "This standalone analysis is only implemented for posterior='student_t_niw' to match the current newsvendor experiment. "
            f"Got {args.posterior!r}."
        )
    if args.likelihood != "multivariate_student_t":
        raise ValueError(
            "This standalone analysis is only implemented for likelihood='multivariate_student_t'. "
            f"Got {args.likelihood!r}."
        )
    if args.dim != 5:
        raise ValueError(f"This standalone analysis currently targets the 5D synthetic newsvendor only, got dim={args.dim}.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_select = int(args.num_observations) - int(args.num_observations) // 2
    gamma_bulks = (
        _default_gamma_bulks(n_select=n_select, delta=float(args.dkw_delta))
        if args.gamma_bulks is None or len(args.gamma_bulks) == 0
        else sorted({float(g) for g in args.gamma_bulks})
    )

    running_frames: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    replication_start = int(args.replication)
    replication_ids = list(range(replication_start, replication_start + int(args.num_replications)))

    for replication in replication_ids:
        generator = np.random.default_rng(seed=int(replication))
        data = sample_dgp(
            args.dgp,
            int(args.num_observations),
            dim=int(args.dim),
            contamination=0.0,
            contamination_type=None,
            generator=generator,
        )
        data = np.asarray(data, dtype=float).reshape(int(args.num_observations), int(args.dim))

        theta_prior = default_prior_params(args.posterior, dim=int(args.dim))
        theta_posterior = get_posterior_params(args.posterior, data, theta_prior)

        for gamma_bulk in gamma_bulks:
            run_df, summary = _analyse_one_gamma(
                replication=int(replication),
                data=data,
                theta_posterior=theta_posterior,
                gamma_bulk=float(gamma_bulk),
                num_likelihood_samples=int(args.num_likelihood_samples),
                reference_n_accept=int(args.reference_n_accept),
                dim=int(args.dim),
                posterior=args.posterior,
                likelihood=args.likelihood,
                dkw_delta=float(args.dkw_delta),
                lv_use_pc_bulk_geometry=bool(args.lv_use_pc_bulk_geometry),
            )
            run_df = run_df.copy()
            run_df["replication"] = int(replication)
            running_frames.append(run_df)

            summary = dict(summary)
            summary["replication"] = int(replication)
            summary_rows.append(summary)

    running_df = pd.concat(running_frames, axis=0, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows).sort_values(["gamma_bulk", "replication"]).reset_index(drop=True)

    plot_path = out_dir / "saa_convergence_plot.pdf"
    running_csv = out_dir / "saa_convergence_running.csv"
    summary_csv = out_dir / "saa_convergence_summary.csv"
    metadata_json = out_dir / "saa_convergence_metadata.json"

    _make_plot(running_df, summary_df, plot_path)
    running_df.to_csv(running_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    default_student_t_df, default_student_t_loc, default_student_t_shape, _ = _student_t_base_params(int(args.dim))
    metadata = {
        "replication_start": int(replication_start),
        "num_replications": int(args.num_replications),
        "replication_ids": [int(r) for r in replication_ids],
        "num_observations": int(args.num_observations),
        "dim": int(args.dim),
        "num_likelihood_samples": int(args.num_likelihood_samples),
        "reference_n_accept": int(args.reference_n_accept),
        "posterior": args.posterior,
        "likelihood": args.likelihood,
        "dgp": args.dgp,
        "dkw_delta": float(args.dkw_delta),
        "gamma_bulks": [float(g) for g in gamma_bulks],
        "lv_use_pc_bulk_geometry": bool(args.lv_use_pc_bulk_geometry),
        "default_student_t_df": float(default_student_t_df),
        "default_student_t_loc": np.asarray(default_student_t_loc, dtype=float).tolist(),
        "default_student_t_shape_triu": np.asarray(
            default_student_t_shape[np.triu_indices(int(args.dim))], dtype=float
        ).tolist(),
        "deprecated_quad_r": int(args.quad_r),
        "deprecated_quad_phi": int(args.quad_phi),
        "deprecated_quad_theta": int(args.quad_theta),
        "output_dir": str(out_dir),
    }
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved plot to {plot_path}")
    print(f"Saved running table to {running_csv}")
    print(f"Saved summary table to {summary_csv}")
    print(f"Saved metadata to {metadata_json}")


if __name__ == "__main__":
    main()
