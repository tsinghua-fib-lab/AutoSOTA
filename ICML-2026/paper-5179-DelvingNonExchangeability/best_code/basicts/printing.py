"""Unified metric printing helper."""

from __future__ import annotations

from typing import Optional


def _format_alpha_line(stats: dict, alpha: Optional[float], target_cov: Optional[float]) -> str:
    observed = stats.get("observed_coverage", float("nan"))
    pi_width = stats.get("pi_width", float("nan"))
    pi_width_median = stats.get("pi_width_median", float("nan"))
    winkler = stats.get("winkler", float("nan"))
    node_cov_std = stats.get("node_cov_std", None)
    ssc_min = stats.get("ssc_min", None)
    width_err_corr = stats.get("width_error_pearson", None)
    if target_cov is None:
        target_cov = stats.get("target_coverage", None)

    if alpha is None:
        base = (
            f"Observed coverage: {observed:.3f} | PI-Width {pi_width:.3f} | "
            f"PI-MedWidth {pi_width_median:.3f} | Winkler {winkler:.3f}"
        )
        if node_cov_std is not None:
            base += f" | NodeCovStd {float(node_cov_std):.3f}"
        if ssc_min is not None:
            base += f" | SSC_min {float(ssc_min):.3f}"
        if width_err_corr is not None:
            base += f" | W-ErrCorr {float(width_err_corr):.3f}"
        return base

    delta = None
    if target_cov is not None:
        try:
            delta = observed - float(target_cov)
        except Exception:
            delta = None
    dstr = f"{delta:+.3f}" if delta is not None else "N/A"
    tgt = target_cov if target_cov is not None else float("nan")
    line = (
        f"alpha={float(alpha):.3f} | target {tgt:.3f} | observed {observed:.3f} | "
        f"delta {dstr} | PI-Width {pi_width:.3f} | PI-MedWidth {pi_width_median:.3f} | Winkler {winkler:.3f}"
    )
    if node_cov_std is not None:
        line += f" | NodeCovStd {float(node_cov_std):.3f}"
    if ssc_min is not None:
        line += f" | SSC_min {float(ssc_min):.3f}"
    if width_err_corr is not None:
        line += f" | W-ErrCorr {float(width_err_corr):.3f}"
    return line


def print_metrics(
    stats: dict,
    *,
    alpha: Optional[float] = None,
    target_cov: Optional[float] = None,
    header: Optional[str] = None,
) -> None:
    """Print a one-line formatted metric summary."""
    if header:
        print(header)
    line = _format_alpha_line(stats, alpha, target_cov)
    print(line)
