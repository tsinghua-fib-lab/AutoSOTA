"""
BRCA experiment: ensemble vs. sub-models LOCO importance on the BRCA gene
expression dataset with validated ground-truth genes.

Dataset: 572 patients, 50 genes, PAM50 subtype classification (4 classes).
Source: Catav et al. 2021 (MCI), Janssen et al. 2023 (UMFI).
Ground truth: 10 validated breast cancer driver genes.

The data can be found at https://github.com/TAU-MLwell/Marginal-Contribution-Feature-Importance/blob/main/BRCA_dataset/BRCA.csv

Usage:
    python run_brca.py --model_name logreg_l2 --seed 0
    python run_brca.py --model_name mlp --seed 0

Results are saved to: {results_dir}/{model_name}/seed_{seed}/
"""

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from hidimstat import LOCO
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample
from sklearn.utils.validation import check_is_fitted

GROUND_TRUTH_GENES = [
    "BCL11A",
    "EZH2",
    "IGF1R",
    "LFNG",
    "BRCA1",
    "SLC22A5",
    "CDK6",
    "BRCA2",
    "TEX14",
    "CCND1",
]


# ── Ensemble utilities ────────────────────────────────────────────────────


def _parallel_fit_with_indices_clf(estimator, X, y, random_state, max_samples):
    """Fit a classifier on a bootstrap sample, return fitted model and indices."""
    n_samples = X.shape[0]
    n_draw = int(n_samples * max_samples) if max_samples <= 1.0 else int(max_samples)
    indices = resample(
        np.arange(n_samples),
        replace=True,
        n_samples=n_draw,
        random_state=random_state,
    )
    fitted = clone(estimator).fit(X[indices], y[indices])
    return fitted, indices


class BaggingVotingClassifier(BaseEstimator, ClassifierMixin):
    """Ensemble classifier: bootstrap sampling + soft voting.

    Fits multiple estimators (with possibly different hyperparameters)
    on bootstrap samples. Predictions are averaged probabilities.
    """

    def __init__(self, estimators, max_samples=1.0, n_jobs=1, random_state=None):
        self.estimators = estimators
        self.max_samples = max_samples
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        rng = np.random.RandomState(self.random_state)
        seeds = rng.randint(np.iinfo(np.int32).max, size=len(self.estimators))
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_parallel_fit_with_indices_clf)(est, X, y, seed, self.max_samples)
            for (_, est), seed in zip(self.estimators, seeds)
        )
        self.estimators_, self.estimators_samples_ = zip(*results)
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        probas = np.array([est.predict_proba(X) for est in self.estimators_])
        return np.mean(probas, axis=0)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def build_ensemble(model_name, n_ensemble, max_samples, seed):
    """Build a BaggingVotingClassifier for the given model type.

    Parameters
    ----------
    model_name : str
        Model type: "logreg_l2" or "mlp".
    n_ensemble : int
        Number of ensemble members.
    max_samples : float
        Fraction of training samples per bootstrap.
    seed : int
        Random seed.

    Returns
    -------
    BaggingVotingClassifier
    """
    rng = np.random.default_rng(seed)

    if model_name == "logreg_l2":
        C_values = np.logspace(-2, 2, n_ensemble)
        estimators = [
            (
                f"lr_{k}",
                make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        C=float(C_values[k]),
                        max_iter=1000,
                        random_state=seed + k,
                    ),
                ),
            )
            for k in range(n_ensemble)
        ]
    elif model_name == "mlp":
        widths = rng.choice(np.arange(32, 256), n_ensemble, replace=False)
        n_layers = rng.integers(1, 4, size=n_ensemble)
        estimators = [
            (
                f"mlp_{k}",
                make_pipeline(
                    StandardScaler(),
                    MLPClassifier(
                        hidden_layer_sizes=tuple([int(widths[k])] * int(n_layers[k])),
                        max_iter=256,
                        early_stopping=True,
                        n_iter_no_change=20,
                        random_state=seed + k,
                    ),
                ),
            )
            for k in range(n_ensemble)
        ]
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return BaggingVotingClassifier(
        estimators=estimators,
        max_samples=max_samples,
        n_jobs=1,
        random_state=seed,
    )


def get_sub_models(model):
    """Extract sub-models from an ensemble."""
    if hasattr(model, "estimators_"):
        return model.estimators_
    raise ValueError("Model type not supported for sub-model extraction.")


# ── LOCO helpers ──────────────────────────────────────────────────────────


def loco_one(X, y, train_index, test_index, model, fold_id, n_jobs=1):
    """Compute LOCO importances for ensemble and sub-models on one fold."""
    n_features = X.shape[1]
    output_list = []
    X_train, X_test = X[train_index], X[test_index]

    model_c = model
    if hasattr(model_c, "n_jobs"):
        model_c.n_jobs = 1

    # Ensemble LOCO
    print(f"  Fold {fold_id}: LOCO ensemble ...")
    loco = LOCO(model_c, method="predict_proba", loss=log_loss, n_jobs=n_jobs)
    loco.fit(X_train, y[train_index])
    importances_full = loco.importance(X_test, y[test_index])
    output_list.append(
        pd.DataFrame(
            {
                "feature": np.arange(n_features),
                "importance": importances_full,
                "fold": fold_id,
                "model": "ensemble",
            }
        )
    )

    # Sub-models LOCO
    sub_model_list = get_sub_models(model_c)
    bootstrap_indices = (
        model_c.estimators_samples_
        if hasattr(model_c, "estimators_samples_")
        else [np.arange(len(train_index))] * len(sub_model_list)
    )

    print(f"  Fold {fold_id}: LOCO sub-models ...")
    for i, sub_model in enumerate(sub_model_list):
        boot_idx = bootstrap_indices[i]
        sub_model_c = deepcopy(sub_model)
        loco_sub = LOCO(
            sub_model_c, method="predict_proba", loss=log_loss, n_jobs=n_jobs
        )
        loco_sub.fit(X_train[boot_idx], y[train_index][boot_idx])
        importances_sub = loco_sub.importance(X_test, y[test_index])
        output_list.append(
            pd.DataFrame(
                {
                    "feature": np.arange(n_features),
                    "importance": importances_sub,
                    "fold": fold_id,
                    "model": f"sub_model_{i}",
                }
            )
        )

    return output_list


# ── Main ──────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="BRCA ensemble VIM experiment")
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/BRCA.csv",
        help="Path to BRCA.csv",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results_brca",
        help="Root results directory",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="logreg_l2",
        choices=["logreg_l2", "mlp"],
        help="Model type",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--n_folds", type=int, default=10, help="Number of CV folds")
    parser.add_argument(
        "--n_ensemble", type=int, default=10, help="Number of ensemble members"
    )
    parser.add_argument(
        "--max_samples",
        type=float,
        default=1.0,
        help="Fraction of training samples per bootstrap",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=10,
        help="Number of parallel jobs for LOCO computation",
    )
    return parser.parse_args()


def main(args):
    data_path = Path(args.data_path)
    results_dir = Path(args.results_dir)
    model_name = args.model_name
    seed = args.seed
    n_folds = args.n_folds
    n_ensemble = args.n_ensemble
    max_samples = args.max_samples
    n_jobs = args.n_jobs

    out_dir = results_dir / model_name / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    df = pd.read_csv(data_path).drop(columns=["Sample.ID"])
    le = LabelEncoder()
    y_arr = le.fit_transform(df["BRCA_Subtype_PAM50"])
    X_df = df.drop(columns=["BRCA_Subtype_PAM50"])

    # Permute non-driver gene columns (seeded)
    rng_perm = np.random.RandomState(seed)
    for col in [g for g in X_df.columns if g not in GROUND_TRUTH_GENES]:
        X_df[col] = rng_perm.permutation(X_df[col].values)

    gene_names = list(X_df.columns)
    support = np.array([i for i, g in enumerate(gene_names) if g in GROUND_TRUTH_GENES])
    X_arr = np.array(X_df, dtype=float)

    print(
        f"Samples: {X_arr.shape[0]}, Genes: {X_arr.shape[1]}, "
        f"Classes: {list(le.classes_)}"
    )
    print(f"Model: {model_name}, Seed: {seed}")

    # ── Fit ensemble per fold ─────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    scores_list = []
    models_list = []

    for fold_id, (train_idx, test_idx) in enumerate(cv.split(X_arr, y_arr)):
        print(f"\n── Fold {fold_id + 1}/{n_folds} ──")
        X_train, y_train = X_arr[train_idx], y_arr[train_idx]
        X_test, y_test = X_arr[test_idx], y_arr[test_idx]

        model = build_ensemble(model_name, n_ensemble, max_samples, seed + fold_id)
        model.fit(X_train, y_train)
        models_list.append(model)

        # Ensemble scores
        y_proba = model.predict_proba(X_test)
        scores_list.append(
            {
                "fold": fold_id,
                "model": "ensemble",
                "acc": accuracy_score(y_test, model.predict(X_test)),
                "roc_auc": roc_auc_score(y_test, y_proba, multi_class="ovr"),
            }
        )

        # Sub-model scores
        for i, est in enumerate(model.estimators_):
            y_proba_sub = est.predict_proba(X_test)
            scores_list.append(
                {
                    "fold": fold_id,
                    "model": f"sub_model_{i}",
                    "acc": accuracy_score(y_test, est.predict(X_test)),
                    "roc_auc": roc_auc_score(y_test, y_proba_sub, multi_class="ovr"),
                }
            )

    df_scores = pd.DataFrame(scores_list)
    df_scores["strategy"] = df_scores["model"].apply(
        lambda x: "ensemble" if x == "ensemble" else "sub_models"
    )
    df_scores.to_csv(out_dir / "scores.csv", index=False)

    df_agg = (
        df_scores.groupby(["strategy", "fold"])[["acc", "roc_auc"]].mean().reset_index()
    )
    for metric in ["acc", "roc_auc"]:
        for strat in ["ensemble", "sub_models"]:
            vals = df_agg.loc[df_agg["strategy"] == strat, metric]
            print(f"  {strat:12s} {metric}: {vals.mean():.3f} +/- {vals.std():.3f}")

    # ── LOCO importance ───────────────────────────────────────────────────
    print("\nComputing LOCO importances...")
    loco_output = Parallel(n_jobs=n_jobs)(
        delayed(loco_one)(
            X_arr,
            y_arr,
            train_idx,
            test_idx,
            models_list[fold_id],
            fold_id,
            n_jobs=1,
        )
        for fold_id, (train_idx, test_idx) in enumerate(cv.split(X_arr, y_arr))
    )
    loco_df = pd.concat([item for sublist in loco_output for item in sublist], axis=0)
    loco_df["gene"] = loco_df["feature"].apply(lambda x: gene_names[x])
    loco_df["is_driver"] = loco_df["gene"].isin(GROUND_TRUTH_GENES)
    loco_df.to_csv(out_dir / "loco_importances.csv", index=False)

    # ── Metrics ───────────────────────────────────────────────────────────
    n_features = len(gene_names)
    gt_labels = np.zeros(n_features)
    gt_labels[support] = 1

    def precision_at_k(importances, supp, k=10):
        top_k = set(np.argsort(importances)[-k:])
        return len(top_k & set(supp)) / k

    ens = loco_df[loco_df["model"] == "ensemble"]
    sub = loco_df[loco_df["model"].str.startswith("sub_model")]
    ens_imp = ens.groupby("feature")["importance"].mean().values
    sub_imp = (
        sub.groupby(["fold", "feature"])["importance"]
        .mean()
        .reset_index()
        .groupby("feature")["importance"]
        .mean()
        .values
    )
    metrics = {
        "model": model_name,
        "seed": seed,
        "p10_ens": precision_at_k(ens_imp, support),
        "p10_sub": precision_at_k(sub_imp, support),
        "auc_ens": roc_auc_score(gt_labels, ens_imp),
        "auc_sub": roc_auc_score(gt_labels, sub_imp),
    }
    print(f"\n  P@10 — ens: {metrics['p10_ens']:.2f}, sub: {metrics['p10_sub']:.2f}")
    print(f"  AUC  — ens: {metrics['auc_ens']:.3f}, sub: {metrics['auc_sub']:.3f}")

    pd.DataFrame([metrics]).to_csv(out_dir / "metrics.csv", index=False)
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
    print("Done.")
