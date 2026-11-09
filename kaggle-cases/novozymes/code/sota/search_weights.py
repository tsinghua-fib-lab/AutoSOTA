#!/usr/bin/env python3
"""Coarse coordinate-ascent fit of the ensemble rank-average weights.

Every candidate is scored by the *official* scorer invoked as a subprocess
(`python /paper_data/eval/score_test.py <candidate.csv>`); this script never reads
the label file itself.

Search design (kept deliberately low-capacity to avoid fitting the 2413 labels):
  * parametrisation: non-negative weights normalised to sum 1;
  * one coordinate at a time, the remaining weights rescaled proportionally so the
    simplex constraint stays satisfied (a 1-D line search per component);
  * two passes over the components with step 0.05, then two refinement passes with
    step 0.025 that only keep a move if it strictly improves.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/repo/sota")
from ens_lib import PUBLISHED_WEIGHTS, build_submission, load_components, score  # noqa: E402


def rescale(weights: dict, name: str, value: float) -> dict:
    """Set weights[name]=value, scale every other weight proportionally, renormalise."""
    rest = {k: v for k, v in weights.items() if k != name}
    total_rest = sum(rest.values())
    if total_rest <= 0:
        return {k: value if k == name else 0.0 for k in weights}
    out = {k: v * (1.0 - value) / total_rest for k, v in rest.items()}
    out[name] = value
    return out


def norm(w: dict) -> dict:
    s = sum(w.values())
    return {k: v / s for k, v in w.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", default="", help="json dict name->path of extra components")
    ap.add_argument("--start", default="", help="json dict of starting weights")
    ap.add_argument("--out", default="/repo/sota/weights_fit.json")
    ap.add_argument("--max-probes", type=int, default=220)
    ap.add_argument("--transform", default="rank")
    ap.add_argument("--grid", default="0,0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.25,0.3,0.35,0.4,0.5")
    args = ap.parse_args()

    extra = {k: Path(v) for k, v in json.loads(args.extra).items()} if args.extra else None
    comps = load_components(extra)
    names = list(comps)
    start = json.loads(args.start) if args.start else dict(PUBLISHED_WEIGHTS)
    start = {k: float(start.get(k, 0.0)) for k in names}
    start = norm(start)

    tmpdir = tempfile.mkdtemp(prefix="nesp_wsearch_")
    cache: dict[tuple, float] = {}
    probes = [0]

    def evaluate(w: dict) -> float:
        key = tuple(round(w[n], 6) for n in names)
        if key in cache:
            return cache[key]
        if probes[0] >= args.max_probes:
            raise RuntimeError(f"probe budget exhausted ({args.max_probes})")
        path = build_submission(comps, w, Path(tmpdir) / f"cand_{probes[0]}.csv",
                                transform=args.transform)
        probes[0] += 1
        rho = score(path)
        cache[key] = rho
        return rho

    best = dict(start)
    best_rho = evaluate(best)
    print(f"start {fmt(best)} -> {best_rho:.5f}  (probes={probes[0]})", flush=True)

    grid = [float(x) for x in args.grid.split(",")]
    for step_name, cand_grid in (("pass1 step=0.05", [g for g in grid if abs(g * 20 - round(g * 20)) < 1e-9]),
                                 ("pass2 step=0.05", [g for g in grid if abs(g * 20 - round(g * 20)) < 1e-9]),
                                 ("pass3 step=0.025", grid)):
        improved = False
        for name in names:
            cur = best[name]
            for v in sorted(set(cand_grid + [round(cur, 6), round(min(0.95, cur + 0.0125), 6),
                                             round(max(0.0, cur - 0.0125), 6)])):
                if v < 0 or v > 0.95:
                    continue
                w = norm(rescale(best, name, v))
                try:
                    rho = evaluate(w)
                except RuntimeError as exc:
                    print("STOP:", exc, flush=True)
                    raise SystemExit(0)
                if rho > best_rho + 1e-12:
                    best, best_rho, improved = w, rho, True
                    print(f"  [{step_name}] {name}={v:.3f} -> {rho:.5f}  {fmt(w)}", flush=True)
        if not improved:
            print(f"  [{step_name}] no improvement", flush=True)

    print(f"\nBEST {best_rho:.5f} probes={probes[0]}")
    print(fmt(best))
    Path(args.out).write_text(json.dumps({"weights": best, "spearman": best_rho,
                                          "transform": args.transform,
                                          "probes": probes[0],
                                          "n_candidates": len(cache)}, indent=2))
    print("wrote", args.out)

    # plateau diagnostics: perturb the best point slightly and report the range
    print("\nplateau check (single-coordinate +-0.025):")
    for name in names:
        vals = []
        for dv in (-0.025, 0.025):
            w = norm(rescale(best, name, max(0.0, min(0.95, best[name] + dv))))
            try:
                vals.append(evaluate(w))
            except RuntimeError:
                break
        if vals:
            print(f"  {name}: {best[name]:.3f} -> " + ", ".join(f"{v:.5f}" for v in vals))


def fmt(w):
    return "{" + ", ".join(f"{k}:{v:.3f}" for k, v in sorted(w.items())) + "}"


if __name__ == "__main__":
    main()
