"""Score a test prediction file against the official ground truth.

  python scripts/evaluate.py runs/<name>/test_pred.json

Prediction file: JSON lines with `id` and list columns reactivity / deg_Mg_pH10 / deg_Mg_50C
(per-position predictions, at least seq_scored long). Prints public / private MCRMSE.
Private = the 1172 rescored 130-mers (Kaggle private LB, 1st place = 0.34198)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd  # noqa: E402

from ov.metrics import load_labels, score_test  # noqa: E402
from ov.paths import DATA  # noqa: E402

LEADERBOARD = {"1st (Nullrecurrent)": 0.34198, "2nd (Kazuki2)": 0.34266}


def main():
    pred = pd.read_json(sys.argv[1], lines=True)
    s = score_test(pred, load_labels(os.path.join(DATA, "test_labels.json")))
    print("public  MCRMSE %.5f" % s["public"][0])
    print("private MCRMSE %.5f   cols %s" % (s["private"][0], " ".join("%.5f" % v for v in s["private"][1])))
    for k, v in LEADERBOARD.items():
        print("  vs %-20s %.5f  (%+.5f)" % (k, v, s["private"][0] - v))


if __name__ == "__main__":
    main()
