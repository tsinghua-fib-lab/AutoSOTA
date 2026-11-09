"""Weighted blend of runs' test predictions (runs/<name>/test_pred.json) and OOF; prints cv / public / private.

  python scripts/blend.py s2_pl:0.5 t4_tfm_pl:0.5 [--out runs/blend/test_pred.json]

CV is computed only when all runs share the same folds (same seed); private is for reporting only."""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ov.metrics import load_labels, score_test  # noqa: E402
from ov.paths import DATA, MODEL_COLS, RUNS  # noqa: E402


def run_oof(run):
    prep = np.load(os.path.join(RUNS, run, "prep.npz"))
    heads = {}
    for f in glob.glob(os.path.join(RUNS, run, "models", "*_preds.npz")):
        h = os.path.basename(f).rsplit("_f", 1)[0]
        d = np.load(f)
        heads.setdefault(h, np.zeros(prep["y"].shape, np.float32))[d["oof_idx"]] = d["oof"]
    return np.mean(list(heads.values()), 0), prep


def cv(p, prep):
    y, sn = prep["y"], prep["sn_filter"].astype(bool)
    e = []
    for c in range(3):
        a, b = y[sn, :, c].ravel(), p[sn, :, c].ravel()
        ok = ~np.isnan(a)
        e.append(np.sqrt(np.mean((a[ok] - b[ok]) ** 2)))
    return float(np.mean(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="name:weight")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    specs = [(r.split(":")[0], float(r.split(":")[1]) if ":" in r else 1.0) for r in args.runs]
    tot = sum(w for _, w in specs)

    blend, oof, preps = None, 0, []
    for name, w in specs:
        df = pd.read_json(os.path.join(RUNS, name, "test_pred.json"), lines=True).set_index("id").sort_index()
        arr = {c: [np.array(v) * w / tot for v in df[c]] for c in MODEL_COLS}
        if blend is None:
            blend = df.copy()
            for c in MODEL_COLS:
                blend[c] = arr[c]
        else:
            for c in MODEL_COLS:
                blend[c] = [a + b for a, b in zip(blend[c], arr[c])]
        o, prep = run_oof(name)
        oof = oof + o * w / tot
        preps.append(prep)
    same_folds = all((p["folds"] == preps[0]["folds"]).all() for p in preps)
    res = {"cv": cv(oof, preps[0]) if same_folds else None}
    sub = blend.reset_index()
    s = score_test(sub, load_labels(os.path.join(DATA, "test_labels.json")))
    res.update(public=round(s["public"][0], 5), private=round(s["private"][0], 5))
    print(" ".join(args.runs), json.dumps(res))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        sub.to_json(args.out, orient="records", lines=True)


if __name__ == "__main__":
    main()
