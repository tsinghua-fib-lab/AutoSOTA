import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


class OODEvaluator:
    """
    Compute and display standard OOD detection metrics.

    Metrics
    -------
    AUROC  : Area under the ROC curve. Higher is better (1.0 = perfect).
    FPR95  : False-positive rate at the operating point where the true-positive
             rate on ID samples is at least 95%. Lower is better (0.0 = perfect).

    Convention: the score arrays must follow the "higher = more ID-like"
    convention used by the streaming OOD scorer. Positive label = 1 means
    in-distribution; negative label = 0 means out-of-distribution.
    """

    def compute(
        self,
        id_scores:  np.ndarray,
        ood_scores: np.ndarray,
    ) -> tuple:
        """
        Compute AUROC and FPR95 from two 1-D score arrays.

        Parameters
        ----------
        id_scores  : Scores for ID samples (label = 1).
        ood_scores : Scores for OOD samples (label = 0).

        Returns
        -------
        auroc : float in [0, 1].
        fpr95 : float in [0, 1].
        """
        y_true   = np.concatenate([np.ones(len(id_scores)),  np.zeros(len(ood_scores))])
        y_scores = np.concatenate([id_scores,                ood_scores])

        auroc       = roc_auc_score(y_true, y_scores)
        fpr, tpr, _ = roc_curve(y_true, y_scores)

        # Walk along the ROC curve and stop at the first point where TPR >= 95%.
        fpr95 = next(
            (fpr[i] for i in range(len(tpr)) if tpr[i] >= 0.95),
            1.0,   # Fallback: if TPR never reaches 95%, FPR95 = 1.0.
        )
        return auroc, fpr95

    def print_results(self, metrics: dict) -> None:
        """
        Print a formatted summary table with AUROC and FPR95 per OOD dataset
        plus an average row at the bottom.

        Parameters
        ----------
        metrics : dict mapping dataset name → {'AUROC': float, 'FPR95': float}.
        """
        print('\n' + '=' * 60)
        print(f"{'Dataset':<15} | {'AUROC':>8} | {'FPR95':>8}")
        print('-' * 60)

        aurocs, fprs = [], []
        for name, m in metrics.items():
            print(f"{name:<15} | {m['AUROC'] * 100:>7.2f}% | {m['FPR95'] * 100:>7.2f}%")
            aurocs.append(m['AUROC'])
            fprs.append(m['FPR95'])

        if aurocs:
            print('-' * 60)
            print(
                f"{'AVERAGE':<15} | {np.mean(aurocs) * 100:>7.2f}% "
                f"| {np.mean(fprs) * 100:>7.2f}%"
            )

        print('=' * 60)
