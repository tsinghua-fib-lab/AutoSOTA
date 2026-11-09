#!/usr/bin/env python3
"""Probe candidate components with the official scorer.

For each candidate component CSV this reports
  * its standalone Spearman in both sign orientations, and
  * the Spearman of the current best ensemble with that component folded in at a
    fixed weight (0.08 of the total), using the sign-corrected orientation.
All scoring goes through `python /paper_data/eval/score_test.py` as a subprocess.

Usage: python sota/probe_components.py [name,name,...] '{"name": "/abs/path.csv", ...}'
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/repo/sota")
from ens_lib import build_submission, load_components, score  # noqa: E402

CFG = Path("/repo/sota/ensemble_config.json")
if CFG.exists():
    _cfg = json.loads(CFG.read_text())
    BEST = dict(_cfg["weights"])
    TRANSFORM = str(_cfg.get("rank_transform", "rank"))
    BASE_EXTRA = {Path(p).stem.replace("submission_", ""): p for p in _cfg["extra_components"]}
else:  # pragma: no cover
    BEST = {"sasa": 0.148, "lgbm": 0.189, "thermo": 0.184, "bfac": 0.155, "rosetta": 0.324}
    BASE_EXTRA = {}
    TRANSFORM = "rank"

NEW_WEIGHT = 0.08


def main():
    names = [n for n in sys.argv[1].split(",") if n] if len(sys.argv) > 1 else None
    extra_paths = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    all_extra = dict(BASE_EXTRA)
    all_extra.update(extra_paths)
    base_extra = load_components({k: Path(v) for k, v in BASE_EXTRA.items()}) if BASE_EXTRA else {}
    base = load_components()
    base.update(base_extra)
    rho0 = score(build_submission(base, BEST, Path("/tmp/probe_cand.csv"), transform=TRANSFORM))
    print(f"current best ensemble ({len(BEST)} comps, {TRANSFORM}): {rho0:.5f}", flush=True)

    out = {}
    for name, path in extra_paths.items():
        if names and name not in names:
            continue
        comps = dict(base)
        comps[name] = load_components({name: Path(path)})[name]
        s_pos = score(build_submission(comps, {name: 1.0}, "/tmp/probe_cand.csv", transform=TRANSFORM))
        s_neg = score(build_submission(comps, {name: -1.0}, "/tmp/probe_cand.csv", transform=TRANSFORM))
        best_sign = 1.0 if s_pos >= s_neg else -1.0
        s_standalone = max(s_pos, s_neg)
        comps[name] = comps[name] * best_sign
        scale = 1.0 - NEW_WEIGHT
        blend = {k: v * scale for k, v in BEST.items()}
        blend[name] = NEW_WEIGHT
        s_blend = score(build_submission(comps, blend, "/tmp/probe_cand.csv", transform=TRANSFORM))
        out[name] = dict(standalone_pos=s_pos, standalone_neg=s_neg,
                         standalone=s_standalone, sign=best_sign, blend=s_blend,
                         delta=s_blend - rho0)
        print(f"{name:10s} standalone(+{s_pos:.5f} / {-s_neg:+.5f}) best={s_standalone:.5f} "
              f"sign={best_sign:+.0f}  ensemble+w{NEW_WEIGHT} = {s_blend:.5f}  "
              f"delta={s_blend - rho0:+.5f}", flush=True)
    Path("/repo/sota/probe_results.json").write_text(json.dumps(out, indent=2))
    print("wrote /repo/sota/probe_results.json")


if __name__ == "__main__":
    main()
