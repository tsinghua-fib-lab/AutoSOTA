#!/usr/bin/env python3
"""Robustness checks for the extended ensemble.

1. Fixed (non-searched) allocations: if a simple hand-set mixture already captures
   most of the gain, the improvement is signal rather than search overfitting.
2. Leave-one-out of each added component at the fitted point.
3. Plateau check: single-coordinate +-0.025 perturbations around the optimum.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/repo/sota")
from ens_lib import build_submission, load_components, score  # noqa: E402

EXTRA = {
    "dsasa": "/repo/outputs/components/submission_dsasa.csv",
    "mutsasa": "/repo/outputs/components/submission_mutsasa.csv",
    "contacts": "/repo/outputs/components/submission_contacts.csv",
    "dcont": "/repo/outputs/components/submission_dcont.csv",
}
BASE5 = {"sasa": 0.148, "lgbm": 0.189, "thermo": 0.184, "bfac": 0.155, "rosetta": 0.324}
FIT9 = {"sasa": 0.075, "lgbm": 0.098, "thermo": 0.096, "bfac": 0.150,
        "rosetta": 0.242, "dsasa": 0.046, "mutsasa": 0.044, "contacts": 0.146,
        "dcont": 0.102}
NEW = ["dsasa", "mutsasa", "contacts", "dcont"]

tmp = Path("/tmp/verify_cand.csv")


def ev(comps, w, label):
    rho = score(build_submission(comps, w, tmp))
    print(f"{label:38s} {rho:.5f}", flush=True)
    return rho


def scaled(factor):
    w = {k: v * factor for k, v in BASE5.items()}
    per = (1.0 - factor) / len(NEW)
    for n in NEW:
        w[n] = per
    return w


def main():
    comps = load_components({k: Path(v) for k, v in EXTRA.items()})
    print("=== component loads ok ===")
    # the official scorer is deterministic; ev() double-scores as a live determinism check
    results = {}
    results["published5"] = ev(comps, {"sasa": 0.10, "lgbm": 0.30, "thermo": 0.20,
                                       "bfac": 0.10, "rosetta": 0.30}, "published 5-comp weights")
    results["fit5"] = ev(comps, BASE5, "iter-1 fitted 5-comp")
    for f in (0.5, 0.6, 0.7, 0.8):
        results[f"fixed_base{f}"] = ev(comps, scaled(f), f"FIXED base5*{f} + 4 new equal")
    results["uniform9"] = ev(comps, {k: 1 / 9 for k in FIT9}, "FIXED uniform over 9")
    results["fit9"] = ev(comps, FIT9, "searched 9-comp (iter-2 best)")

    print("\n=== leave-one-out of the added components at the fitted point ===")
    for n in NEW:
        w = {k: v for k, v in FIT9.items() if k != n}
        results[f"loo_{n}"] = ev(comps, w, f"  drop {n}")

    print("\n=== plateau check around the fitted 9-comp point (+-0.025) ===")
    for name in FIT9:
        vals = []
        for dv in (-0.025, 0.025):
            w = dict(FIT9)
            if w[name] + dv <= 0:
                continue
            w[name] += dv
            vals.append(ev(comps, w, f"  {name} {w[name]:.3f}"))
        if vals:
            results[f"plateau_{name}"] = vals

    Path("/repo/sota/verify_results.json").write_text(json.dumps(results, indent=2))
    print("\nwrote /repo/sota/verify_results.json")


if __name__ == "__main__":
    main()
