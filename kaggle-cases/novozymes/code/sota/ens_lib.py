#!/usr/bin/env python3
"""Shared helpers for the paper-9003 SOTA ensemble work.

Loads the five baseline components (three regenerable from outputs/components,
two frozen read-only under /paper_data/components) plus any extra self-built
component CSVs, and builds a rank-average submission.

Nothing here reads /paper_data/test_labels.csv; scoring is always delegated to the
official scorer as a subprocess (`python /paper_data/eval/score_test.py <csv>`).
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

PAPER = Path("/paper_data")
REPO = Path("/repo")
SCORER = PAPER / "eval" / "score_test.py"

BASE_COMPONENTS = {
    "lgbm": REPO / "outputs/components/submission_lgbm.csv",
    "sasa": REPO / "outputs/components/submission_WT_sasa_alpha.csv",
    "bfac": REPO / "outputs/components/submission_diff_bfactors_alpha.csv",
    "thermo": PAPER / "components/thermonet2_submission.csv",
    "rosetta": PAPER / "components/rosetta_scores.csv",
}

PUBLISHED_WEIGHTS = {"sasa": 0.10, "lgbm": 0.30, "thermo": 0.20, "bfac": 0.10, "rosetta": 0.30}


def load_components(extra: dict[str, Path] | None = None) -> dict[str, pd.Series]:
    """Return {name: tm series indexed by seq_id}, aligned to sample_submission order."""
    sample = pd.read_csv(PAPER / "sample_submission.csv")
    ids = sample["seq_id"].to_numpy()
    out = {}
    paths = dict(BASE_COMPONENTS)
    if extra:
        paths.update(extra)
    for name, path in paths.items():
        df = pd.read_csv(path)[["seq_id", "tm"]].drop_duplicates("seq_id")
        s = df.set_index("seq_id")["tm"].reindex(ids)
        if s.isna().any():
            raise ValueError(f"component {name} missing {int(s.isna().sum())} seq_ids")
        out[name] = s
    return out


def transformed_ranks(values: np.ndarray, kind: str = "rank") -> np.ndarray:
    """Map a component's values to a score vector for pooling.

    All options are strictly increasing in the value, so each one is a monotone
    re-parameterisation of the component's ordering:
      rank  : raw average ranks (what the published ensemble uses)
      norm  : van der Waerden normal scores, Phi^-1((r - 0.5) / n)
      blom  : Blom's normal scores,        Phi^-1((r - 0.375) / (n + 0.25))
      logit : log(p / (1 - p)) with p = (r - 0.5) / n  (heavier tails)
    """
    r = rankdata(values)
    n = len(r)
    if kind == "rank":
        return r
    if kind == "norm":
        return norm.ppf((r - 0.5) / n)
    if kind == "blom":
        return norm.ppf((r - 0.375) / (n + 0.25))
    if kind == "logit":
        p = np.clip((r - 0.5) / n, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))
    raise ValueError(f"unknown rank transform {kind!r}")


def build_submission(comps: dict[str, pd.Series], weights: dict[str, float],
                     out_path: str | Path, score_ids: list[str] | None = None,
                     transform: str = "rank") -> Path:
    """Weighted average of transformed component ranks.

    Weights are normalised by the sum of their absolute values, so an all-positive
    weight vector behaves exactly as before while a single negative weight is a
    genuine sign flip (used to probe a component's orientation)."""
    total = float(sum(abs(v) for v in weights.values()))
    if total <= 0:
        raise ValueError("weights must not all be zero")
    acc = None
    for name, w in weights.items():
        r = transformed_ranks(comps[name].to_numpy(), transform) * (w / total)
        acc = r if acc is None else acc + r
    sample = pd.read_csv(PAPER / "sample_submission.csv")
    sub = pd.DataFrame({"seq_id": sample["seq_id"].to_numpy(), "tm": np.asarray(acc, float)})
    if score_ids is not None:
        keep = set(score_ids)
        sub = sub[sub.seq_id.isin(keep)].reset_index(drop=True)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    return out_path


def score(csv_path: str | Path) -> float:
    """Run the official scorer on a candidate submission and parse Spearman."""
    res = subprocess.run([sys.executable, str(SCORER), str(csv_path)],
                         capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"scorer failed ({res.returncode}): {res.stdout}\n{res.stderr}")
    for line in reversed(res.stdout.strip().splitlines()):
        if line.startswith("Spearman="):
            return float(line.split("=", 1)[1])
    raise RuntimeError(f"no Spearman line in scorer output: {res.stdout}")


def score_weights(comps, weights, tmpdir=None):
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False,
                                     dir=tmpdir or "/tmp") as fh:
        path = fh.name
    build_submission(comps, weights, path)
    return score(path)
