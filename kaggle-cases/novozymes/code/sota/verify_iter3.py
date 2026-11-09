#!/usr/bin/env python3
"""Robustness battery for the iteration-3 ensemble.

Everything is scored by the official scorer as a subprocess.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/repo/sota")
from ens_lib import build_submission, load_components, score  # noqa: E402

EXTRA = {
    "mutsasa": "/repo/outputs/components/submission_mutsasa.csv",
    "dsasa": "/repo/outputs/components/submission_dsasa.csv",
    "contacts": "/repo/outputs/components/submission_contacts.csv",
    "dcont": "/repo/outputs/components/submission_dcont.csv",
    "nshell": "/repo/outputs/components/submission_nshell.csv",
    "wtcont": "/repo/outputs/components/submission_wtcont.csv",
}
ITER2 = {"sasa": 0.075, "lgbm": 0.098, "thermo": 0.096, "bfac": 0.150, "rosetta": 0.242,
         "mutsasa": 0.044, "dsasa": 0.046, "contacts": 0.146, "dcont": 0.102}
FIT11 = {"sasa": 0.000, "lgbm": 0.099, "thermo": 0.035, "bfac": 0.165, "rosetta": 0.258,
         "mutsasa": 0.031, "dsasa": 0.033, "contacts": 0.146, "dcont": 0.085,
         "nshell": 0.148, "wtcont": 0.000}

tmp = "/tmp/verify3_cand.csv"


def ev(comps, w, label):
    rho = score(build_submission(comps, w, tmp))
    print("%-46s %.5f" % (label, rho), flush=True)
    return rho


def main():
    comps = load_components({k: Path(v) for k, v in EXTRA.items()})
    res = {}
    res["iter2_fit9"] = ev(comps, ITER2, "iter-2 fitted 9-component point")
    res["fit11"] = ev(comps, FIT11, "iter-3 searched 11-component point")

    print("\n--- UNSARCHED mixtures (no weight search at all) ---")
    for wn in (0.05, 0.10, 0.15, 0.20, 0.30):
        w = {k: v * (1 - wn) for k, v in ITER2.items()}
        w["nshell"] = wn
        res[f"uns_9x{1-wn:.2f}+nshell{wn}"] = ev(
            comps, w, f"UNSEARCHED iter2*{1 - wn:.2f} + nshell {wn:.2f}")
    res["uniform11"] = ev(comps, {k: 1 / 11 for k in FIT11}, "UNSEARCHED uniform over 11")
    half = {k: 0.5 * FIT11[k] + 0.5 / 11 for k in FIT11}
    res["shrunk50"] = ev(comps, half, "fit11 shrunk 50% toward uniform")

    print("\n--- leave-one-out at the fitted point ---")
    for n in FIT11:
        if FIT11[n] == 0:
            continue
        w = {k: v for k, v in FIT11.items() if k != n}
        res[f"loo_{n}"] = ev(comps, w, f"  drop {n}")

    print("\n--- plateau: single-coordinate +-0.025 around the fitted point ---")
    for name in FIT11:
        for dv in (-0.025, 0.025):
            if FIT11[name] + dv <= 0:
                continue
            w = dict(FIT11)
            w[name] += dv
            res[f"plateau_{name}_{dv:+.3f}"] = ev(comps, w, f"  {name} {w[name]:.3f}")

    Path("/repo/sota/verify_iter3.json").write_text(json.dumps(res, indent=2))
    print("\nwrote /repo/sota/verify_iter3.json")


if __name__ == "__main__":
    main()
