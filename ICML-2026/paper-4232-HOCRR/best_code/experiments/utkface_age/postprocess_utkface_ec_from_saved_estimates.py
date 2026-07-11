#!/usr/bin/env python3
"""Compute (E,C)+M radii for UTKFace from saved (E,C,G)+M estimates.

This avoids rerunning model inference. By default, it reconstructs the standard
errors implied by the saved normal-approximation confidence intervals and
recomputes the C and E intervals with the 2-way (E,C) union-bound split before
running the variance+mean bounded certificate. A conservative mode that directly
reuses the saved 3-way (E,C,G) intervals is also available.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from regression_certifiers.certify import BoundedCertifierVarianceMean  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Postprocess UTKFace (E,C)+M radii from saved estimates.")
    p.add_argument(
        "--input_glob",
        default="outputs/utkface_grid/ours_100/utkface_sigma*_alpha*_merged.json",
    )
    p.add_argument(
        "--output_dir",
        default="outputs/utkface_grid/ours_100_with_ec",
    )
    p.add_argument("--quadrature-points", type=int, default=60)
    p.add_argument(
        "--ci-mode",
        choices=["recompute_ec_2way", "reuse_saved"],
        default="recompute_ec_2way",
        help=(
            "How to build E and C confidence bounds. recompute_ec_2way backs out "
            "standard errors from saved 3-way normal CIs and rebuilds 2-way (E,C) CIs; "
            "reuse_saved directly uses the saved 3-way bounds."
        ),
    )
    p.add_argument(
        "--ec-confidence",
        type=float,
        default=0.0,
        help="Target confidence for recomputed (E,C) bounds (0 = use each file's config confidence).",
    )
    p.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="Debug/smoke option: process only the first K samples per file (0 = all).",
    )
    p.add_argument("--workers", type=int, default=1, help="Parallel worker processes.")
    return p.parse_args()


def summarize(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "fraction_positive": float(np.mean(arr > 0.0)),
    }


def _infer_standard_error(point: float, lower: float, upper: float, z_value: float) -> float:
    """Recover the standard error from a symmetric normal-approximation CI."""
    if z_value <= 0.0:
        raise ValueError(f"z_value must be positive, got {z_value}")
    half_width = max(abs(float(upper) - float(point)), abs(float(point) - float(lower)))
    return float(half_width / z_value)


def _recompute_two_way_bounds(
    trial: dict,
    *,
    source_confidence: float,
    target_confidence: float,
) -> Dict[str, float]:
    """Rebuild E and C CIs using the 2-way (E,C) union-bound split."""
    source_delta = 1.0 - float(source_confidence)
    target_delta = 1.0 - float(target_confidence)

    # The saved UTKFace E/C/G estimates were produced with a 3-way split, and
    # each individual interval is two-sided.
    z_source = float(norm.ppf(1.0 - (source_delta / 3.0) / 2.0))
    z_target = float(norm.ppf(1.0 - (target_delta / 2.0) / 2.0))

    c_hat = float(trial["C_hat"])
    e_hat = float(trial["E_hat"])
    c_se = _infer_standard_error(c_hat, float(trial["C_lcb"]), float(trial["C_ucb"]), z_source)
    e_se = _infer_standard_error(e_hat, float(trial["E_lcb"]), float(trial["E_ucb"]), z_source)

    return {
        "C_lcb": float(c_hat - z_target * c_se),
        "C_ucb": float(c_hat + z_target * c_se),
        "E_lcb": float(e_hat - z_target * e_se),
        "E_ucb": float(e_hat + z_target * e_se),
        "C_se_reconstructed": c_se,
        "E_se_reconstructed": e_se,
        "source_z": z_source,
        "target_z": z_target,
    }


def compute_ec_radius(
    certifier: BoundedCertifierVarianceMean,
    trial: dict,
    *,
    source_confidence: float,
    target_confidence: float,
    ci_mode: str,
) -> Dict[str, object]:
    if ci_mode == "recompute_ec_2way":
        bounds = _recompute_two_way_bounds(
            trial,
            source_confidence=source_confidence,
            target_confidence=target_confidence,
        )
        bound_source = "recomputed_2way_from_saved_normal_ci"
    elif ci_mode == "reuse_saved":
        bounds = {
            "C_lcb": float(trial["C_lcb"]),
            "C_ucb": float(trial["C_ucb"]),
            "E_lcb": float(trial.get("E_lcb", trial["E_hat"])),
            "E_ucb": float(trial.get("E_ucb", trial["E_hat"])),
        }
        bound_source = "saved_3way_bounds"
    else:
        raise ValueError(f"Unknown ci_mode: {ci_mode}")

    c_ucb = float(bounds["C_ucb"])
    mean_candidates = [
        float(bounds.get("E_lcb", trial["E_hat"])),
        float(trial["E_hat"]),
        float(bounds.get("E_ucb", trial["E_hat"])),
    ]
    radii = [float(certifier.certify_point_from_estimates(c_ucb, e)) for e in mean_candidates]
    return {
        "ci_mode": ci_mode,
        "bound_source": bound_source,
        "source_confidence": float(source_confidence),
        "target_confidence": float(target_confidence),
        "C_ucb_used": c_ucb,
        "E_candidates_used": mean_candidates,
        **{k: float(v) for k, v in bounds.items()},
        "radius_bounded_ec_candidates": radii,
        "radius_bounded_ec": float(min(radii)),
    }


def compute_sample_ec(args_tuple):
    sample, sigma, m_bound, eps_y, certifier_confidence, quadrature_points, source_confidence, target_confidence, ci_mode = args_tuple
    certifier = BoundedCertifierVarianceMean(
        sigma=float(sigma),
        M=float(m_bound),
        eps_y=float(eps_y),
        confidence=float(certifier_confidence),
        quadrature_points=int(quadrature_points),
    )
    ec_trials = []
    for trial in sample.get("ecg_trials", []):
        ec_trials.append(
            compute_ec_radius(
                certifier,
                trial,
                source_confidence=float(source_confidence),
                target_confidence=float(target_confidence),
                ci_mode=str(ci_mode),
            )
        )
    if ec_trials:
        r_mean = float(np.mean([x["radius_bounded_ec"] for x in ec_trials]))
    else:
        r_mean = None
    sample["ec_from_saved_trials"] = ec_trials
    sample["ec_radius_from_saved_estimates"] = r_mean
    return sample, r_mean


def main() -> None:
    args = parse_args()
    paths = sorted(Path().glob(args.input_glob))
    if not paths:
        raise FileNotFoundError(f"No files matched: {args.input_glob}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: List[Dict[str, object]] = []

    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        cfg = data["config"]
        sigma = float(cfg["sigma"])
        m_bound = float(cfg.get("M", 116.0))
        eps_y = float(cfg["eps_y"])
        source_confidence = float(cfg["confidence"])
        target_confidence = float(args.ec_confidence) if args.ec_confidence > 0 else source_confidence

        ec_radii: List[float] = []
        samples = data["samples"]
        if args.max_points > 0:
            samples = samples[: args.max_points]
        task_args = [
            (
                sample,
                sigma,
                m_bound,
                eps_y,
                target_confidence,
                int(args.quadrature_points),
                source_confidence,
                target_confidence,
                str(args.ci_mode),
            )
            for sample in samples
        ]
        if int(args.workers) <= 1:
            for sample_idx, task_arg in enumerate(task_args, start=1):
                sample, r_mean = compute_sample_ec(task_arg)
                if r_mean is not None:
                    ec_radii.append(float(r_mean))
                if sample_idx % 10 == 0 or sample_idx == len(samples):
                    print(f"  {path.name}: processed {sample_idx}/{len(samples)}", flush=True)
        else:
            completed = 0
            updated_samples = []
            with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
                futures = [pool.submit(compute_sample_ec, task_arg) for task_arg in task_args]
                for fut in as_completed(futures):
                    sample, r_mean = fut.result()
                    updated_samples.append(sample)
                    if r_mean is not None:
                        ec_radii.append(float(r_mean))
                    completed += 1
                    if completed % 10 == 0 or completed == len(samples):
                        print(f"  {path.name}: processed {completed}/{len(samples)}", flush=True)
            data["samples"] = sorted(
                updated_samples,
                key=lambda s: int(s.get("sample_global_idx", s.get("sample_local_idx", -1))),
            )

        data.setdefault("postprocessing", {})
        data["postprocessing"]["ec_from_saved_estimates"] = {
            "description": (
                "(E,C)+M radii computed from saved (E,C,G)+M estimates without "
                "rerunning model inference."
            ),
            "source_file": str(path),
            "ci_mode": str(args.ci_mode),
            "source_confidence": source_confidence,
            "target_confidence": target_confidence,
            "source_split": "3-way over E, C, G",
            "target_split": "2-way over E, C" if args.ci_mode == "recompute_ec_2way" else "saved 3-way bounds reused",
            "quadrature_points": int(args.quadrature_points),
        }
        data["summary"]["bounded_ec_radius_from_saved_estimates"] = summarize(ec_radii)

        out_path = out_dir / path.name
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        s = data["summary"]["bounded_ec_radius_from_saved_estimates"]
        summary_rows.append(
            {
                "sigma": sigma,
                "n_points": len(ec_radii),
                "mean_radius_ec": s["mean"],
                "median_radius_ec": s["median"],
                "p10_radius_ec": s["p10"],
                "p90_radius_ec": s["p90"],
                "source_file": str(out_path),
            }
        )
        print(f"Wrote {out_path} mean EC radius={s['mean']:.6g}")

    csv_path = out_dir / "ec_from_saved_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sorted(summary_rows, key=lambda r: float(r["sigma"])))
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
