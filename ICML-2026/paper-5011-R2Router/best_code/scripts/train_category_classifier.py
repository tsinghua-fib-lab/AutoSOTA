#!/usr/bin/env python3
"""
Train a 7-class query category classifier on Qwen3-0.6B embeddings.

Compares multiple classifiers with stratified K-fold cross-validation,
then trains the best one on the full training set.

Input: (8400, 1024) embeddings + category labels from global_index
Output: Best classifier saved to checkpoints/category_router/classifier.joblib

Usage:
    .venv/bin/python scripts/train_category_classifier.py [--k 5]
"""

import os
import sys
import json
import pickle
import numpy as np
from datetime import datetime
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from category_config import (
    EMBEDDINGS_PATH, ROUTER_DATA_PATH, CHECKPOINT_DIR,
    CATEGORY_NAMES, get_category,
)


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def build_classifiers():
    """Return dict of {name: sklearn classifier/pipeline} to compare."""
    return {
        "LogReg_C1": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                solver="lbfgs",
                C=1.0, max_iter=2000, random_state=42, n_jobs=-1,
            )),
        ]),
        "LogReg_C10": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                solver="lbfgs",
                C=10.0, max_iter=2000, random_state=42, n_jobs=-1,
            )),
        ]),
        "LogReg_C0.1": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                solver="lbfgs",
                C=0.1, max_iter=2000, random_state=42, n_jobs=-1,
            )),
        ]),
        "LinearSVC": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(
                C=1.0, max_iter=5000, random_state=42,
                dual="auto",
            )),
        ]),
        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf", C=10.0, gamma="scale",
                random_state=42, cache_size=1000,
            )),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=500, max_depth=None,
            random_state=42, n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=500, max_depth=None,
            random_state=42, n_jobs=-1,
        ),
        "KNN_5": KNeighborsClassifier(
            n_neighbors=5, n_jobs=-1,
        ),
        "KNN_10": KNeighborsClassifier(
            n_neighbors=10, n_jobs=-1,
        ),
        "LDA": LinearDiscriminantAnalysis(),
        "GaussianNB": GaussianNB(),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5, help="K for K-fold CV")
    args = parser.parse_args()

    K = args.k
    log(f"=== Category Classifier Training (K={K}-fold CV) ===\n")

    # 1. Load data
    log("Loading embeddings...")
    with open(EMBEDDINGS_PATH, "rb") as f:
        emb_data = pickle.load(f)
    embeddings = emb_data["embeddings"]  # (8400, 1024)
    log(f"  Embeddings: {embeddings.shape}")

    log("Loading router_data.json for global indices...")
    with open(ROUTER_DATA_PATH) as f:
        router_data = json.load(f)
    global_indices = [q["global index"] for q in router_data]
    assert len(global_indices) == embeddings.shape[0]

    # 2. Assign categories
    categories = np.array([get_category(gi) for gi in global_indices], dtype=np.int32)
    log(f"  Category distribution:")
    for i, name in enumerate(CATEGORY_NAMES):
        count = (categories == i).sum()
        log(f"    {i} {name}: {count} ({count/len(categories)*100:.1f}%)")

    # 3. Hold-out test set (80/20, stratified)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        embeddings, categories, np.arange(len(categories)),
        test_size=0.2, random_state=42, stratify=categories,
    )
    log(f"\n  Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # 4. K-fold CV on training set
    log(f"\n=== {K}-Fold Stratified Cross-Validation on Training Set ===\n")
    classifiers = build_classifiers()
    cv = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

    results = {}
    for name, clf in classifiers.items():
        log(f"  {name}...")
        try:
            scores = cross_val_score(clf, X_train, y_train, cv=cv,
                                     scoring="accuracy", n_jobs=1)
            mean_acc = scores.mean()
            std_acc = scores.std()
            results[name] = (mean_acc, std_acc, scores)
            log(f"    CV accuracy: {mean_acc:.4f} +/- {std_acc:.4f}  "
                f"(folds: {', '.join(f'{s:.4f}' for s in scores)})")

            # Save checkpoint for each classifier after CV
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            clf_checkpoint_path = os.path.join(CHECKPOINT_DIR, f"classifier_{name}.joblib")
            # Fit on full training set and save
            clf.fit(X_train, y_train)
            dump(clf, clf_checkpoint_path)
            log(f"    Saved checkpoint: {clf_checkpoint_path}")
        except Exception as e:
            log(f"    FAILED: {e}")
            results[name] = (0.0, 0.0, np.zeros(K))

    # 5. Rank classifiers
    log(f"\n{'='*70}")
    log(f"{'Classifier':<22} {'Mean CV Acc':>12} {'Std':>8}")
    log(f"{'='*70}")
    ranked = sorted(results.items(), key=lambda x: -x[1][0])
    for rank, (name, (mean_acc, std_acc, _)) in enumerate(ranked, 1):
        marker = " <-- BEST" if rank == 1 else ""
        log(f"  {rank:>2}. {name:<20} {mean_acc:>10.4f}   {std_acc:.4f}{marker}")

    best_name = ranked[0][0]
    best_cv_acc = ranked[0][1][0]
    log(f"\nBest classifier: {best_name} (CV={best_cv_acc:.4f})")

    # 6. Train best classifier on full training set, evaluate on held-out test
    log(f"\n=== Training {best_name} on full training set ===")
    best_clf = build_classifiers()[best_name]
    best_clf.fit(X_train, y_train)

    y_pred_train = best_clf.predict(X_train)
    y_pred_test = best_clf.predict(X_test)
    train_acc = (y_pred_train == y_train).mean()
    test_acc = (y_pred_test == y_test).mean()

    log(f"\n  Train accuracy: {train_acc:.4f}")
    log(f"  Test accuracy:  {test_acc:.4f}")

    log(f"\nClassification Report (test set):")
    report = classification_report(
        y_test, y_pred_test,
        target_names=CATEGORY_NAMES,
        digits=4,
    )
    print(report)

    log("Confusion Matrix (test set):")
    cm = confusion_matrix(y_test, y_pred_test)
    header = "".join(f"{name[:6]:>8}" for name in CATEGORY_NAMES)
    print(f"{'':>14}{header}")
    for i, row in enumerate(cm):
        row_str = "".join(f"{v:>8}" for v in row)
        print(f"{CATEGORY_NAMES[i]:>14}{row_str}")

    # 7. Also train & evaluate top-3 on held-out test for comparison
    log(f"\n=== Top-3 Classifiers on Held-out Test Set ===")
    log(f"{'Classifier':<22} {'CV Acc':>10} {'Test Acc':>10} {'Train Acc':>10}")
    log("-" * 55)
    for name, (cv_acc, _, _) in ranked[:3]:
        clf = build_classifiers()[name]
        clf.fit(X_train, y_train)
        ta = (clf.predict(X_train) == y_train).mean()
        te = (clf.predict(X_test) == y_test).mean()
        log(f"  {name:<20} {cv_acc:>10.4f} {te:>10.4f} {ta:>10.4f}")

    # 8. Save best classifier and metadata
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    clf_path = os.path.join(CHECKPOINT_DIR, "classifier.joblib")
    dump(best_clf, clf_path)
    log(f"\nSaved best classifier ({best_name}) to {clf_path}")

    # Save split indices
    split_path = os.path.join(CHECKPOINT_DIR, "train_test_split.pkl")
    with open(split_path, "wb") as f:
        pickle.dump({"train_idx": idx_train, "test_idx": idx_test}, f)
    log(f"Saved split indices to {split_path}")

    # Save CV results summary
    summary = {
        "best_classifier": best_name,
        "best_cv_accuracy": float(best_cv_acc),
        "test_accuracy": float(test_acc),
        "k_folds": K,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "all_results": {
            name: {"cv_mean": float(m), "cv_std": float(s)}
            for name, (m, s, _) in results.items()
        },
    }
    summary_path = os.path.join(CHECKPOINT_DIR, "classifier_cv_results.json")
    import json as json_mod
    with open(summary_path, "w") as f:
        json_mod.dump(summary, f, indent=2)
    log(f"Saved CV results to {summary_path}")

    log("\nDone!")


if __name__ == "__main__":
    main()
