"""Minimal DVM-AD quickstart.

Run with:
    python examples/quickstart.py
"""

import numpy as np
from sklearn.metrics import roc_auc_score

from dvmad import DVMAD


def main():
    rng = np.random.default_rng(0)
    X_train = rng.normal(0.0, 1.0, size=(500, 10))
    X_test = np.vstack(
        [
            rng.normal(0.0, 1.0, size=(200, 10)),  # inliers
            rng.normal(5.0, 1.0, size=(20, 10)),   # outliers
        ]
    )
    y_test = np.hstack([np.zeros(200), np.ones(20)])

    clf = DVMAD(contamination=0.1).fit(X_train)
    scores = clf.decision_function(X_test)
    labels = clf.predict(X_test)

    print(f"ROC AUC : {roc_auc_score(y_test, scores):.4f}")
    print(f"Predicted outliers: {int(labels.sum())} / {len(labels)}")
    print(f"Threshold: {clf.threshold_:.4f}")


if __name__ == "__main__":
    main()
