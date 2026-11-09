"""python scripts/predict_official_sets.py X.npy A.npy out.npy -> [20, N, L, 5] (MODEL_COLS order)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ov import nullrec  # noqa: E402
from ov.paths import KOV  # noqa: E402
from ov.trainer import build, predict  # noqa: E402


def main():
    fx, fa, out = sys.argv[1:4]
    X, A = np.load(fx), np.load(fa, mmap_mode="r")
    preds = []
    for head, wdir in nullrec.OFFICIAL:
        _, model = build(None, X.shape[2], A.shape[3], head)
        for fold in range(5):
            model.load_weights(os.path.join(KOV, "model_files", wdir, "model_%d.h5" % fold))
            preds.append(predict(model, X, A).astype(np.float32))
        print(head, "done", flush=True)
    np.save(out, np.stack(preds))


if __name__ == "__main__":
    main()
