"""Drop DVM-AD into a PyOD-style benchmark next to other detectors.

Demonstrates that DVMAD works through the standard PyOD `BaseDetector`
contract: same `fit / decision_function / predict` calls as other PyOD
models such as IForest, OCSVM, LOF, KNN, etc.
"""

import numpy as np
from sklearn.metrics import roc_auc_score

from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM

from dvmad import DVMAD


def make_data(seed: int = 0):
    rng = np.random.default_rng(seed)
    X_train = rng.normal(0.0, 1.0, size=(800, 12))
    X_test = np.vstack(
        [
            rng.normal(0.0, 1.0, size=(300, 12)),
            rng.normal(4.0, 1.0, size=(30, 12)),
        ]
    )
    y_test = np.hstack([np.zeros(300), np.ones(30)])
    return X_train, X_test, y_test


def main():
    X_train, X_test, y_test = make_data()

    detectors = {
        "DVMAD": DVMAD(contamination=0.1),
        "IForest": IForest(contamination=0.1, random_state=0),
        "OCSVM": OCSVM(contamination=0.1),
        "LOF": LOF(contamination=0.1),
        "KNN": KNN(contamination=0.1),
    }

    print(f"{'Model':<10} | {'ROC AUC':>8}")
    print("-" * 23)
    for name, clf in detectors.items():
        clf.fit(X_train)
        scores = clf.decision_function(X_test)
        auc = roc_auc_score(y_test, scores)
        print(f"{name:<10} | {auc:>8.4f}")


if __name__ == "__main__":
    main()
