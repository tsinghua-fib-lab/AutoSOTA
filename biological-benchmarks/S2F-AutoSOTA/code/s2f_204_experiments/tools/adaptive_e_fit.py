#!/usr/bin/env python3
"""IDEA-603 (revised): per-assay adaptive PSSM weight e(neff), fit by strict
leave-one-assay-out CV on the sel-80 component dumps.

e conditions on the assay's OWN computed subMSA Henikoff effective-sequence
count (dataset/DMS_pssm_cache/*.npz 'neff' — the quantity that actually feeds
the weighted frequencies). ProteinGym's full-alignment MSA_Neff_L must NOT be
used (iter-2 evidence: it anticorrelates with the PSSM gain on both sets).

Fitted families (all monotone in neff):
  F1: e = e0 * (1 - exp(-neff / K))                     (2 free params)
  F2: e = e0 * sigmoid(a * (log10(neff) - b))           (3 free params)
  F3: e = clip(c0 + c1 * log10(neff), 0, emax)          (3 free params)

Baselines: global-e (the iter-2 plateau e=0.25) and the per-assay-optimal-e
ceiling (in-sample 1-D search; upper bound of ANY adaptive policy).

Acceptance (IDEA-603 guardrail): strict LOO-CV mean must clear the global-e
plateau (0.5113) by >= +0.002 on sel-80.

Usage: python tools/adaptive_e_fit.py <dump_dir>
"""
import argparse
import glob
import os
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata

DUMPS = "/home/dataset-assist-0/chenzhibin/s3f+autosota/S3F/results/component_dumps/sel80_iter2"
CACHE = "/home/dataset-assist-0/chenzhibin/s3f+autosota/S3F/dataset/DMS_pssm_cache"


def spearman(pred, target):
    """Average-rank Pearson (torchdrug convention)."""
    p = rankdata(pred)
    t = rankdata(target)
    cov = (p * t).mean() - p.mean() * t.mean()
    return float(cov / (p.std() * t.std() + 1e-10))


def load(dump_dir):
    assays = []
    for f in sorted(glob.glob(os.path.join(dump_dir, "*.npz"))):
        z = np.load(f)
        a = {k: z[k].astype(np.float64) for k in z.files}
        name = os.path.basename(f)[:-4]
        m = np.isfinite(a["target"])
        assays.append({
            "name": name,
            "fused": 0.70 * a["s2f"] + 0.30 * a["wt"],   # w=0, c=0.30 plateau
            "cov": (a["cov"] >= 1 - 1e-9).astype(np.float64),
            "pssm": a["pssm"],
            "target": a["target"][m],
            "neff": _neff(name),
        })
    return assays


def _neff(name):
    z = np.load(os.path.join(CACHE, name + ".npz"))
    return float(z["neff"])


def score_at(assay, e):
    """Per-assay spearman at one e (row blend with coverage gating)."""
    final = (1 - e * assay["cov"]) * assay["fused"] + e * assay["cov"] * assay["pssm"]
    m = np.isfinite(assay["target"])
    return spearman(final[m], assay["target"][m])


def mean_score(assays, e_fn):
    return float(np.mean([score_at(a, e_fn(a["neff"])) for a in assays]))


def fit_family(assays, family, theta0, bounds, x0):
    """Fit params on the given assays by maximizing mean spearman."""
    def neg(x):
        return -mean_score(assays, lambda n: family(n, *x))
    res = minimize(neg, x0, bounds=bounds, method="Nelder-Mead",
                   options={"xatol": 1e-3, "fatol": 1e-5, "maxiter": 200})
    return res.x, -res.fun


def loo_cv(assays, family, bounds, x0):
    """Strict LOO-CV: fit on 79, evaluate on the held-out assay."""
    scores = []
    for j in range(len(assays)):
        fit_set = assays[:j] + assays[j + 1:]
        theta, _ = fit_family(fit_set, family, None, bounds, x0)
        scores.append(score_at(assays[j], family(assays[j]["neff"], *theta)))
    return float(np.mean(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir", default=DUMPS, nargs="?")
    args = ap.parse_args()

    assays = load(args.dump_dir)
    neffs = np.array([a["neff"] for a in assays])
    print("[adaptive] %d assays; neff range %.1f..%.1f, median %.1f" %
          (len(assays), neffs.min(), neffs.max(), np.median(neffs)))

    # Reference: global-e plateau (iter-2 winner)
    flat = mean_score(assays, lambda n: 0.25)
    print("\n[adaptive] global e=0.25 (iter-2 plateau): %.4f" % flat)

    # Ceiling: per-assay optimal e (in-sample; upper bound for any policy)
    e_grid = np.arange(0.0, 0.61, 0.01)
    ceiling = np.mean([max(score_at(a, e) for e in e_grid) for a in assays])
    print("[adaptive] per-assay-optimal-e ceiling:    %.4f  (headroom %+.4f)"
          % (ceiling, ceiling - flat))

    families = {
        "F1 e0*(1-exp(-n/K))": (
            lambda n, e0, K: e0 * (1 - np.exp(-n / K)),
            [(0.05, 0.8), (1.0, 2000.0)], [0.4, 30.0]),
        "F2 e0*sigmoid(a*(log10n-b))": (
            lambda n, e0, a, b: e0 / (1 + np.exp(-a * (np.log10(n) - b))),
            [(0.05, 0.8), (0.05, 8.0), (0.0, 3.5)], [0.4, 1.5, 1.5]),
        "F3 clip(c0+c1*log10n,0,emax)": (
            lambda n, c0, c1, emax: np.clip(c0 + c1 * np.log10(n), 0.0, emax),
            [(-0.5, 0.8), (-0.5, 0.8), (0.05, 0.8)], [0.05, 0.12, 0.4]),
    }
    print("\n[adaptive] strict LOO-CV (fit 79 -> test 1, mean over 80):")
    for name, (fn, bounds, x0) in families.items():
        loo = loo_cv(assays, fn, bounds, x0)
        print("  %-28s LOO=%.4f (delta vs flat %+.4f)" % (name, loo, loo - flat))
        if loo - flat >= 0.002:
            theta, ins = fit_family(assays, fn, None, bounds, x0)
            print("    -> PASSES guardrail. In-sample fit: score=%.4f, theta=%s"
                  % (ins, np.round(theta, 3)))
    print("\n[adaptive] acceptance threshold: LOO >= %.4f" % (flat + 0.002))


if __name__ == "__main__":
    main()
