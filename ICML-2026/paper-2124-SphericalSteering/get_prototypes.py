"""
Step 2: Compute Contrastive Prototypes

This script computes truthful (μ_T) and hallucination (μ_H) prototypes using
the difference vector method. This creates antipodal prototypes on the unit
sphere for maximum vMF discriminability.

The script uses K-Fold cross-validation at the question level to ensure
no data leakage between training and test sets.

Usage:
    python get_prototypes.py --feature_file ./features/llama3.1-8B_layer14.npz --save_dir ./prototypes

Output:
    Saves one .npz file per fold containing:
    - mu_T: Truthful prototype [hidden_dim]
    - mu_H: Hallucination prototype [hidden_dim]
    - test_q_indices: Question indices in the test fold
    - fold_idx: Fold index
    - train_accuracy: Classification accuracy on training set
    - test_accuracy: Classification accuracy on test set
"""

import argparse
import numpy as np
import os
from sklearn.model_selection import KFold


def normalize(v):
    """Normalize a vector to unit length."""
    norm = np.linalg.norm(v)
    if norm == 0: 
        return v
    return v / norm


def compute_contrastive_prototypes(X_train, y_train):
    """
    Compute contrastive prototypes using the difference vector method.
    
    This approach creates antipodal prototypes (μ_T = -μ_H) that maximize
    the separation between truthful and hallucination directions.
    
    Args:
        X_train: Training activations [N, D]
        y_train: Training labels [N] (1=truthful, 0=hallucination)
    
    Returns:
        mu_T: Truthful prototype (unit vector) [D]
        mu_H: Hallucination prototype (unit vector) [D]
        cos_sim: Cosine similarity between prototypes (should be -1.0)
    """
    X_true = X_train[y_train == 1]
    X_false = X_train[y_train == 0]
    
    # Compute centroids
    mean_true = np.mean(X_true, axis=0)
    mean_false = np.mean(X_false, axis=0)
    
    # Compute difference vector (truthful direction)
    diff_vec = mean_true - mean_false
    
    # Normalize to create unit prototypes
    mu_T = normalize(diff_vec)
    mu_H = -mu_T  # Antipodal prototype
    
    # Verify they are antipodal
    cos_sim = np.dot(mu_T, mu_H)
    
    return mu_T, mu_H, cos_sim


def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Compute contrastive prototypes with K-Fold CV"
    )
    parser.add_argument(
        '--feature_file', 
        type=str, 
        required=True,
        help="Path to feature .npz file from step 1"
    )
    parser.add_argument(
        '--num_folds', 
        type=int, 
        default=2,
        help="Number of folds for cross-validation"
    )
    parser.add_argument(
        '--save_dir', 
        type=str, 
        default='./prototypes',
        help="Directory to save prototypes"
    )
    
    args = parser.parse_args()
    
    # Load features
    print(f"Loading features from {args.feature_file}...")
    data = np.load(args.feature_file)
    X = data['activations']
    y = data['labels']
    q_indices = data['q_indices']
    
    print(f"Loaded {len(X)} samples with {X.shape[1]} dimensions")
    
    # Setup K-Fold at question level
    unique_questions = np.unique(q_indices)
    print(f"Total unique questions: {len(unique_questions)}")
    
    kf = KFold(n_splits=args.num_folds, shuffle=False)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Get base name for output files
    base_name = os.path.basename(args.feature_file).replace('.npz', '')
    
    for fold_idx, (train_q_idx, test_q_idx) in enumerate(kf.split(unique_questions)):
        print(f"\n{'='*60}")
        print(f"Processing Fold {fold_idx + 1}/{args.num_folds}")
        print(f"{'='*60}")
        
        train_qs = unique_questions[train_q_idx]
        test_qs = unique_questions[test_q_idx]
        
        # Create train/test masks
        train_mask = np.isin(q_indices, train_qs)
        test_mask = np.isin(q_indices, test_qs)
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
        
        print(f"Train: {len(X_train)} samples from {len(train_qs)} questions")
        print(f"Test:  {len(X_test)} samples from {len(test_qs)} questions")
        
        # Compute prototypes
        mu_T, mu_H, cos_sim = compute_contrastive_prototypes(X_train, y_train)
        
        # Save prototypes
        save_path = os.path.join(args.save_dir, f"{base_name}_fold{fold_idx}.npz")
        np.savez(
            save_path,
            mu_T=mu_T,
            mu_H=mu_H,
            test_q_indices=test_qs,
            fold_idx=fold_idx,
        )
        print(f"Saved to {save_path}")
    
    print(f"\n{'='*60}")
    print("Step 2 Complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
