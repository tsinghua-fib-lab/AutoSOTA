"""
Evaluator for :class:`~aporia.label_propagation.WassersteinLabelPropagator`.

The evaluator wraps a fitted detector together with a held-out
``(X_test, y_test)`` and produces a flat metrics dictionary suitable for
aggregation into a long-form DataFrame.
"""

from __future__ import annotations

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class LabelPropagationEvaluator:
    """Compute classification metrics + margin statistics on a held-out set."""

    def __init__(
        self,
        detector,
        X_test: np.ndarray,
        y_test: np.ndarray,
        fisher_ref=None,
    ):
        self.detector = detector
        self.X_test = X_test
        self.y_test = y_test
        self.fisher_ref = fisher_ref

    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        ambiguity_eps: float | None = None,
        per_class: bool = True,
    ) -> dict:
        y_pred  = self.detector.predict(self.X_test)
        margins = self.detector.margins(self.X_test)
        abs_m   = np.abs(margins)

        if ambiguity_eps is None:
            ambiguity_eps = margins.mean() - 3 * margins.std()

        tn, fp, fn, tp = confusion_matrix(
            self.y_test, y_pred, labels=[0, 1]
        ).ravel()

        metrics = {
            "accuracy":  accuracy_score (self.y_test, y_pred),
            "f1":        f1_score       (self.y_test, y_pred, zero_division=0),
            "precision": precision_score(self.y_test, y_pred, zero_division=0),
            "recall":    recall_score   (self.y_test, y_pred, zero_division=0),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "mean_margin":     margins.mean(),
            "std_margin":      margins.std(),
            "mean_abs_margin": abs_m.mean(),
            "std_abs_margin":  abs_m.std(),
            "ambiguous_frac":  float(np.mean(abs_m <= ambiguity_eps)),
        }

        # ---- per-class margins ----
        if per_class:
            for c in np.unique(self.y_test):
                mask = self.y_test == c
                metrics[f"mean_margin_class_{c}"]     = margins[mask].mean()
                metrics[f"std_margin_class_{c}"]      = margins[mask].std()
                metrics[f"mean_abs_margin_class_{c}"] = abs_m  [mask].mean()
                metrics[f"std_abs_margin_class_{c}"]  = abs_m  [mask].std()

        # ---- optional: agreement with a Fisher reference ----
        if self.fisher_ref is not None:
            y_fisher = self.fisher_ref.predict(self.X_test)
            metrics["agreement_fisher"] = float(np.mean(y_pred == y_fisher))

            fisher_margins = self.fisher_ref.margins(self.X_test)
            confident = (
                np.abs(fisher_margins) > np.percentile(np.abs(fisher_margins), 50)
            )
            metrics["agreement_fisher_confident"] = float(
                np.mean(y_pred[confident] == y_fisher[confident])
            )

        return metrics
