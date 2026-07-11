#!/usr/bin/env python3
"""
Evaluate LCL model using KNN classification on CellTag dataset.
Matches the evaluation in LCL_cellTag_KNN_base.ipynb.
"""
import anndata as ad
import numpy as np
import argparse
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def evaluate_knn(train_h5ad, test_h5ad, train_embed, test_embed, n_neighbors=5, min_test_size=16):
    """
    Evaluate KNN classification accuracy.
    Returns KNN Test Error = 1 - accuracy.
    """
    print(f"Loading data...")
    adata_train = ad.read_h5ad(train_h5ad)
    adata_test = ad.read_h5ad(test_h5ad)

    train_labels = adata_train.obs["clone_id"].to_numpy()
    test_labels = adata_test.obs["clone_id"].to_numpy()

    train_embeddings = np.load(train_embed)
    test_embeddings = np.load(test_embed)

    print(f"Train labels: {train_labels.shape}, Train embeddings: {train_embeddings.shape}")
    print(f"Test labels: {test_labels.shape}, Test embeddings: {test_embeddings.shape}")

    # Filter out lineages with fewer than min_test_size cells in test set
    test_sizes = pd.Series(test_labels).value_counts()
    kept_lineages = test_sizes[test_sizes >= min_test_size].index

    test_keep = pd.Series(test_labels).isin(kept_lineages).to_numpy()
    X_test_f = test_embeddings[test_keep]
    y_test_f = test_labels[test_keep]

    print(f"Test cells after filtering (>= {min_test_size} cells/lineage): {X_test_f.shape[0]} / {test_embeddings.shape[0]}")

    # KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(train_embeddings, train_labels)

    # Predict
    y_pred = knn.predict(X_test_f)
    accuracy = accuracy_score(y_test_f, y_pred)
    test_error = 1.0 - accuracy

    print(f"\n=== RESULTS ===")
    print(f"KNN Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"KNN Test Error: {test_error:.4f}")

    return {
        "KNN_Test_Accuracy": round(float(accuracy), 4),
        "KNN_Test_Error": round(float(test_error), 4),
        "n_neighbors": n_neighbors,
        "n_test_cells": X_test_f.shape[0],
        "n_train_cells": train_embeddings.shape[0],
    }

if __name__ == "__main__":
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_h5ad", required=True)
    parser.add_argument("--test_h5ad", required=True)
    parser.add_argument("--train_embed", required=True)
    parser.add_argument("--test_embed", required=True)
    parser.add_argument("--n_neighbors", type=int, default=5)
    parser.add_argument("--min_test_size", type=int, default=16)

    args = parser.parse_args()

    results = evaluate_knn(
        args.train_h5ad,
        args.test_h5ad,
        args.train_embed,
        args.test_embed,
        args.n_neighbors,
        args.min_test_size,
    )

    # Also compute with min_test_size=1 (all test cells)
    print("\n=== Without lineage size filter ===")
    results_all = evaluate_knn(
        args.train_h5ad,
        args.test_h5ad,
        args.train_embed,
        args.test_embed,
        args.n_neighbors,
        min_test_size=1,
    )
