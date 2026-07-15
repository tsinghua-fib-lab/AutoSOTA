"""
Paper-facing display name mapping for algorithms and variants.

This module centralizes naming conventions used in plots/tables so that:
- figures are consistent across the repository,
- internal identifiers remain unchanged in code/evidence,
- paper-facing labels can be updated in one place.
"""

from __future__ import annotations

# ============================================================================
# Display name mapping
# ============================================================================

# Full mapping from internal identifiers to paper display names.
DISPLAY_NAMES: dict[str, str] = {
    # Residual bootstrapping (BERW family)
    "BERW": "Residual Bootstrapping",
    "BERW-Hetero": "Residual Bootstrapping",
    "BERW-HeteroRobust": "Residual Bootstrapping (robust)",
    "BERW-Homoskedastic": "Residual Bootstrapping (homo.)",
    "BERW-Hetero-LogW": "Residual Bootstrapping (log-w)",

    # Probe-and-switch variants
    "ProbeSwitch": "Probe-and-switch",
    "ProbeSwitch-MR(t=0.12)": "Probe-and-switch (τ=0.12)",
    "ProbeSwitch-MR(t=0.120)": "Probe-and-switch (τ=0.12)",
    "ProbeSwitch-MR(t=0.18)": "Probe-and-switch (τ=0.18)",
    "ProbeSwitch-MR(t=0.22)": "Probe-and-switch (τ=0.22)",
    # Legacy internal id used by some evidence packs
    "Switch-MisrankingProbe(t=0.12)": "Probe-and-switch (τ=0.12)",
    "ProbeSwitch-MR-LogW(t=0.12)": "Probe-and-switch (log-w, τ=0.12)",

    # CMA-ES baseline
    "CMA-ES": "CMA-ES",
    "CMA-ES-sep": "CMA-ES",
    "Sep-CMA-ES": "CMA-ES",

    # Resampling variants
    "CMA-ES-Resample(k=2)": "Resample (k=2)",
    "CMA-ES-Resample(k=3)": "Resample (k=3)",
    "CMA-ES-Resample(k=5)": "Resample (k=5)",
    "CMA-ES-Resample(k=10)": "Resample (k=10)",
    "CMA-ES-Resample(k=20)": "Resample (k=20)",

    # LRA-CMA-ES
    "LRA-CMA-ES": "LRA-CMA-ES",

    # UH-CMA-ES variants
    "UH-CMA-ES": "UH-CMA-ES",
    "UH-CMA-ES(maxevals=10)": "UH-CMA-ES (n=10)",
    "UH-CMA-ES(maxevals=30)": "UH-CMA-ES",
}

# Abbreviated labels for space-constrained contexts.
DISPLAY_NAMES_SHORT: dict[str, str] = {
    "BERW": "Res. boot.",
    "BERW-Hetero": "Res. boot.",
    "BERW-HeteroRobust": "Res. boot. (R)",
    "ProbeSwitch": "P&S",
    "ProbeSwitch-MR(t=0.12)": "P&S (0.12)",
    "ProbeSwitch-MR(t=0.120)": "P&S (0.12)",
    "ProbeSwitch-MR(t=0.18)": "P&S (0.18)",
    "ProbeSwitch-MR(t=0.22)": "P&S (0.22)",
    "Switch-MisrankingProbe(t=0.12)": "P&S (0.12)",
    "CMA-ES": "CMA-ES",
    "CMA-ES-sep": "CMA-ES",
    "CMA-ES-Resample(k=5)": "Resamp(5)",
    "CMA-ES-Resample(k=10)": "Resamp(10)",
    "LRA-CMA-ES": "LRA-CMA",
    "UH-CMA-ES": "UH-CMA",
    "UH-CMA-ES(maxevals=10)": "UH-CMA(10)",
    "UH-CMA-ES(maxevals=30)": "UH-CMA",
}


def get_display_name(name: str, *, short: bool = False) -> str:
    """
    Return the paper-facing display name for an internal identifier.

    Falls back to returning the original string if not found.
    """
    mapping = DISPLAY_NAMES_SHORT if short else DISPLAY_NAMES

    if name in mapping:
        return mapping[name]

    name_lower = str(name).lower()
    for k, v in mapping.items():
        if str(k).lower() == name_lower:
            return v

    # Light heuristic for variants: allow partial matching as a last resort.
    for k, v in mapping.items():
        k_lower = str(k).lower()
        if k_lower in name_lower or name_lower in k_lower:
            return v

    return str(name)


def get_display_names(names: list[str], *, short: bool = False) -> list[str]:
    return [get_display_name(n, short=short) for n in names]


# ============================================================================
# Depth registry (for fixed-budget figures)
# ============================================================================

# Reference configuration used across main COCO figures.
_REF_BUDGET = 4000  # B=100*d for d=40

# Curated depth/cost values used for depth annotations and the bubble plot.
#
# For UH-CMA-ES, these are meant to come from the evidence pack
# `evidence/uh_cmaes_cost_measurement/uh_cmaes_cost_summary.csv` produced by
# `tools/run_uh_cmaes_cost_measurement.py`. The constants here are a fallback
# so plotting scripts can still run if that evidence is absent.
ALGORITHM_DEPTHS: dict[str, dict[str, float | int | str]] = {
    "CMA-ES-sep": {
        "cost_per_candidate": 1.0,
        "depth": 266,  # 4000 / 15
        "notes": "Analytical: B / λ with λ=15 for d=40",
    },
    "CMA-ES-Resample(k=5)": {
        "cost_per_candidate": 5.0,
        "depth": 53,  # 4000 / (5*15)
        "notes": "Analytical: B / (k·λ)",
    },
    "CMA-ES-Resample(k=10)": {
        "cost_per_candidate": 10.0,
        "depth": 26,  # 4000 / (10*15)
        "notes": "Analytical: B / (k·λ)",
    },
    "BERW-Hetero": {
        "cost_per_candidate": 1.27,  # approx from diagnostic traces
        "depth": 211,  # median depth measured from traces (reference)
        "notes": "Measured from evidence/hansen_test_fixed_budget/diagnostics/traces",
    },
    "UH-CMA-ES(maxevals=30)": {
        # Measured under: d=40, B=4000, instances 1-15, high-misranking subset.
        "cost_per_candidate": 15.465,
        "depth": 17,
        "notes": "Measured; see evidence/uh_cmaes_cost_measurement/uh_cmaes_cost_summary.csv",
    },
    "UH-CMA-ES(maxevals=10)": {
        # Measured under: d=40, B=4000, instances 1-15, high-misranking subset.
        "cost_per_candidate": 7.1365,
        "depth": 37,
        "notes": "Measured; see evidence/uh_cmaes_cost_measurement/uh_cmaes_cost_summary.csv",
    },
}


def get_algorithm_depth(name: str, *, budget: int = _REF_BUDGET) -> int | None:
    """
    Get depth T (generations) for an algorithm under the given budget.

    Returns None if unknown.
    """
    if name not in ALGORITHM_DEPTHS:
        return None
    info = ALGORITHM_DEPTHS[name]
    depth = info.get("depth", None)
    if depth is None:
        return None
    try:
        depth_i = int(depth)
    except Exception:
        return None
    return int(depth_i * int(budget) / _REF_BUDGET)
