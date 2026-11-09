"""Score the released Nullrecurrent weights (4 heads x 5 folds) on the rebuilt test set."""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ov.folding import bprna_string, eternafold_mfe, load_bpps  # noqa: E402
from ov.metrics import load_labels, score_test  # noqa: E402
from ov.paths import DATA, KOV, MODEL_COLS, RUNS  # noqa: E402


def predict_official(test, batch_size=64):
    import tensorflow as tf
    from ov import nullrec

    preds = {}  # (head, fold) -> {id: [L, 5]}
    for length, grp in test.groupby("seq_length"):
        x, a = nullrec.build_inputs(list(grp.sequence), list(grp.structure), list(grp.predicted_loop_type),
                                    [load_bpps(i) for i in grp.id])
        for head, wdir in nullrec.OFFICIAL:
            tf.keras.backend.clear_session()
            base = nullrec.get_base(x.shape[2], a.shape[3])
            model = nullrec.get_model(base, head, x.shape[2], a.shape[3])
            for fold in range(5):
                model.load_weights(os.path.join(KOV, "model_files", wdir, "model_%d.h5" % fold))
                o1 = model.predict([x, a], batch_size=batch_size, verbose=0)
                o2 = model.predict([x[:, ::-1], a[:, ::-1, ::-1]], batch_size=batch_size, verbose=0)[:, ::-1]
                out = (o1 + o2) / 2
                d = preds.setdefault((head, fold), {})
                for i, uid in enumerate(grp.id):
                    d[uid] = out[i]
            print("length", length, "head", head, "done", flush=True)
    return preds


def blend(preds, ids):
    """Mean of the 20 models; clip to [-0.5, 6] for all but the first, as in the released script."""
    keys = list(preds)
    out = {}
    for uid in ids:
        acc = preds[keys[0]][uid].copy()
        for k in keys[1:]:
            acc += np.clip(preds[k][uid], -0.5, 6)
        out[uid] = acc / len(keys)
    return out


def to_frame(pred_by_id, ids):
    return pd.DataFrame({"id": ids, **{c: [pred_by_id[i][:, k].tolist() for i in ids] for k, c in enumerate(MODEL_COLS)}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=["kaggle", "eternafold"], default="kaggle",
                    help="kaggle: Vienna MFE + Kaggle loop types (competition setting); "
                         "eternafold: EternaFold MFE + recomputed loop types (DasLab post-hoc script)")
    ap.add_argument("--out", default=os.path.join(RUNS, "official"))
    args = ap.parse_args()

    test = pd.read_json(os.path.join(DATA, "test.json"), lines=True)
    if args.features == "eternafold":
        from multiprocessing import Pool
        with Pool(100) as p:
            test["structure"] = p.map(eternafold_mfe, list(test.sequence))
        test["predicted_loop_type"] = [bprna_string(s) for s in test.structure]

    os.makedirs(args.out, exist_ok=True)
    cache = os.path.join(args.out, "raw_preds_%s.pkl" % args.features)
    if os.path.exists(cache):
        preds = pd.read_pickle(cache)
    else:
        preds = predict_official(test)
        pd.to_pickle(preds, cache)
    ids = list(test.id)
    sub = to_frame(blend(preds, ids), ids)
    sub.to_json(os.path.join(args.out, "test_pred_%s.json" % args.features), orient="records", lines=True)

    labels = load_labels(os.path.join(DATA, "test_labels.json"))
    for k in preds:
        s = score_test(to_frame(preds[k], ids), labels)
        print("%-8s fold%d  public %.5f  private %.5f" % (k[0], k[1], s["public"][0], s["private"][0]))
    s = score_test(sub, labels)
    print("BLEND(%s)  public %.5f  private %.5f  private-cols %s" % (
        args.features, s["public"][0], s["private"][0], np.round(s["private"][1], 5)))


if __name__ == "__main__":
    main()
