"""
Step 2: Compute Contrastive Prototypes (with Independent Option)

Supports two modes:
- antipodal (default): mu_H = -mu_T via difference vector
- independent (--independent): mu_T = normalize(mean_true), mu_H = normalize(mean_false)
"""

import argparse
import numpy as np
import os
from sklearn.model_selection import KFold


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def compute_antipodal_prototypes(X_train, y_train):
    """Antipodal prototypes via difference vector (baseline)."""
    X_true = X_train[y_train == 1]
    X_false = X_train[y_train == 0]
    mean_true = np.mean(X_true, axis=0)
    mean_false = np.mean(X_false, axis=0)
    diff_vec = mean_true - mean_false
    mu_T = normalize(diff_vec)
    mu_H = -mu_T
    cos_sim = np.dot(mu_T, mu_H)
    return mu_T, mu_H, cos_sim


def compute_independent_prototypes(X_train, y_train):
    """Independent prototypes: normalize each centroid independently."""
    X_true = X_train[y_train == 1]
    X_false = X_train[y_train == 0]
    mean_true = np.mean(X_true, axis=0)
    mean_false = np.mean(X_false, axis=0)
    mu_T = normalize(mean_true)
    mu_H = normalize(mean_false)
    cos_sim = np.dot(mu_T, mu_H)
    return mu_T, mu_H, cos_sim


def main():
    parser = argparse.ArgumentParser(description="Step 2: Compute prototypes with K-Fold CV")
    parser.add_argument('--feature_file', type=str, required=True)
    parser.add_argument('--num_folds', type=int, default=2)
    parser.add_argument('--save_dir', type=str, default='./prototypes')
    parser.add_argument('--independent', action='store_true',
                        help='Use independent (non-antipodal) prototypes')
    args = parser.parse_args()

    compute_fn = compute_independent_prototypes if args.independent else compute_antipodal_prototypes
    method = "independent" if args.independent else "antipodal"
    print("Method: %s" % method)

    print("Loading features from %s..." % args.feature_file)
    data = np.load(args.feature_file)
    X = data['activations']
    y = data['labels']
    q_indices = data['q_indices']
    print("Loaded %d samples with %d dimensions" % (len(X), X.shape[1]))

    unique_questions = np.unique(q_indices)
    print("Total unique questions: %d" % len(unique_questions))
    kf = KFold(n_splits=args.num_folds, shuffle=False)
    os.makedirs(args.save_dir, exist_ok=True)
    base_name = os.path.basename(args.feature_file).replace('.npz', '')

    for fold_idx, (train_q_idx, test_q_idx) in enumerate(kf.split(unique_questions)):
        print("\n" + "=" * 60)
        print("Processing Fold %d/%d" % (fold_idx + 1, args.num_folds))
        print("=" * 60)

        train_qs = unique_questions[train_q_idx]
        test_qs = unique_questions[test_q_idx]
        train_mask = np.isin(q_indices, train_qs)
        test_mask = np.isin(q_indices, test_qs)
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
        print("Train: %d samples from %d questions" % (len(X_train), len(train_qs)))
        print("Test:  %d samples from %d questions" % (len(X_test), len(test_qs)))

        mu_T, mu_H, cos_sim = compute_fn(X_train, y_train)
        print("Prototype cos_sim: %.6f" % cos_sim)

        # Classification accuracy
        scores_T = np.dot(X_test, mu_T)
        scores_H = np.dot(X_test, mu_H)
        test_preds = (scores_T > scores_H).astype(int)
        test_acc = np.mean(test_preds == y_test)
        print("Test classification accuracy: %.4f" % test_acc)

        # Also compute antipodal accuracy for comparison
        mu_T_a, _, _ = compute_antipodal_prototypes(X_train, y_train)
        anti_preds = (np.dot(X_test, mu_T_a) > 0).astype(int)
        anti_acc = np.mean(anti_preds == y_test)
        print("Antipodal baseline accuracy: %.4f" % anti_acc)

        save_path = os.path.join(args.save_dir, "%s_fold%d.npz" % (base_name, fold_idx))
        np.savez(save_path, mu_T=mu_T, mu_H=mu_H,
                 test_q_indices=test_qs, fold_idx=fold_idx,
                 prototype_method=method)
        print("Saved to %s" % save_path)

    print("\n" + "=" * 60)
    print("Step 2 Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
