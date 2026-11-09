"""Rebuild the Kaggle OpenVaccine data from the DasLab/KaggleOpenVaccine release.

Outputs (data/):
  train.json        2400 constructs (Kaggle train.json schema). 1589 SN_filter=1 rows are
                    byte-identical to Kaggle; the 811 SN_filter=0 rows come from RYOS_FULL, whose
                    deg_Mg_pH10 / deg_Mg_50C columns are corrupted (copies of deg_pH10 / deg_50C),
                    so those two targets (and their errors) are set to NaN for those rows.
  test.json         3634 constructs (Kaggle test.json schema, no labels).
  test_labels.json  ground truth + `split` in {public, private, unscored}.
  bpps/<id>.npy     EternaFold base-pair probability matrices.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ov.folding import bprna_string, compute_bpps  # noqa: E402
from ov.paths import DATA, ERRORS, KOV, RAW, TARGETS  # noqa: E402

FULL_ERR = {
    "reactivity_error": "errors_reactivity",
    "deg_error_Mg_pH10": "errors_deg_Mg_pH10",
    "deg_error_pH10": "errors_deg_pH10",
    "deg_error_Mg_50C": "errors_deg_Mg_50C",
    "deg_error_50C": "errors_deg_50C",
}


def build_train(full):
    kag = pd.read_json(os.path.join(KOV, "data/Kaggle_RYOS_data/Kaggle_RYOS_trainset.json"))
    keep = ["id", "sequence", "structure", "predicted_loop_type", "signal_to_noise", "SN_filter",
            "seq_length", "seq_scored"] + ERRORS + TARGETS
    kag = kag[keep].copy()

    pt = full[full.split.isin(["public_train", "public_test"])]
    rest = pt[~pt.sequence.isin(set(kag.sequence))].copy()
    assert len(rest) == 2400 - len(kag), len(rest)

    rows = []
    for i, r in enumerate(rest.itertuples()):
        n = len(r.sequence)
        nan = [float("nan")] * 68
        d = {
            "id": "id_ryos1_%05d" % r.ID,
            "sequence": r.sequence,
            "structure": r.structure,
            "predicted_loop_type": bprna_string(r.structure),
            "signal_to_noise": float(r.signal_to_noise_reactivity),
            "SN_filter": int(r.SN_filter),
            "seq_length": n,
            "seq_scored": 68,
        }
        for t in TARGETS:
            d[t] = nan if t in ("deg_Mg_pH10", "deg_Mg_50C") else [float(x) for x in getattr(r, t)[:68]]
        for e, fe in FULL_ERR.items():
            d[e] = nan if e in ("deg_error_Mg_pH10", "deg_error_Mg_50C") else [float(x) for x in getattr(r, fe)[:68]]
        rows.append(d)
    train = pd.concat([kag, pd.DataFrame(rows)], ignore_index=True)
    train["mg_missing"] = train.id.str.startswith("id_ryos1_").astype(int)
    return train, kag


def build_test(full):
    t = pd.read_json(os.path.join(RAW, "reshaped_test_data.json"))
    t["id"] = "id_" + t["id"].astype(str)
    tf = full.dropna(subset=["test_filter"]).drop_duplicates("sequence").set_index("sequence")["test_filter"]
    split = np.where(t.seq_length == 107, "public",
                     np.where(t.sequence.map(tf) == 1, "private", "unscored"))
    t["split"] = split
    test = t[["id", "sequence", "structure", "predicted_loop_type", "seq_length", "seq_scored"]].copy()
    labels = t[["id", "split", "seq_length", "seq_scored"] + TARGETS].copy()
    return test, labels


def main():
    os.makedirs(DATA, exist_ok=True)
    full = pd.read_json(os.path.join(RAW, "RYOS_FULL_23Jul2021.json"))
    train, kag = build_train(full)
    test, labels = build_test(full)

    acc = np.mean([a == bprna_string(s) for a, s in zip(kag.predicted_loop_type, kag.structure)])
    print("bprna_string exact-match vs Kaggle loop types: %.4f" % acc)

    train.to_json(os.path.join(DATA, "train.json"), orient="records", lines=True)
    test.to_json(os.path.join(DATA, "test.json"), orient="records", lines=True)
    labels.to_json(os.path.join(DATA, "test_labels.json"), orient="records", lines=True)
    print("train", train.shape, "SN_filter", train.SN_filter.value_counts().to_dict())
    print("test", test.shape, "split", labels.split.value_counts().to_dict())

    n = compute_bpps(list(zip(train.id, train.sequence)) + list(zip(test.id, test.sequence)))
    print("computed bpps:", n)


if __name__ == "__main__":
    main()
