"""Aggregation methods: Majority Vote, Dawid-Skene, Posterior Expected Credit."""

import numpy as np
from typing import Tuple, List, Optional
from scipy.special import logsumexp


def majority_vote(
    y: np.ndarray,
    credit_mapping: List[float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Majority vote aggregation.
    
    For each item: z_hat = mode(y)
    Score: S_a = (1/N) * sum_i v(z_hat_ai)
    
    Args:
        y: Observed labels of shape (n_agents, n_items, m)
        credit_mapping: Credit values for each class
    
    Returns:
        z_hat: Predicted labels of shape (n_agents, n_items)
        scores: Agent scores of shape (n_agents,)
    """
    n_agents, n_items, m = y.shape
    credit = np.array(credit_mapping)
    
    # Get label counts
    z_hat = np.zeros((n_agents, n_items), dtype=np.int32)
    
    for a in range(n_agents):
        for i in range(n_items):
            labels = y[a, i]
            counts = np.bincount(labels, minlength=3)
            # Tie-break: prefer higher class (more favorable)
            z_hat[a, i] = np.argmax(counts[::-1])
            z_hat[a, i] = 2 - z_hat[a, i]  # Reverse the index
    
    # Compute scores
    credits_per_item = credit[z_hat]
    scores = credits_per_item.mean(axis=1)
    
    return z_hat, scores


def dawid_skene_em(
    y: np.ndarray,
    annotators: np.ndarray,
    n_annotators: int,
    n_classes: int = 3,
    max_iter: int = 100,
    tol: float = 1e-6,
    init_confusion: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Dawid-Skene EM algorithm for label aggregation.
    
    Args:
        y: Observed labels of shape (n_agents, n_items, m)
        annotators: Annotator IDs of shape (n_agents, n_items, m)
        n_annotators: Total number of annotators
        n_classes: Number of classes
        max_iter: Maximum EM iterations
        tol: Convergence tolerance
        init_confusion: Optional initial confusion matrices (n_annotators, n_classes, n_classes)
    
    Returns:
        gamma: Posterior probabilities of shape (n_agents, n_items, n_classes)
        pi: Estimated confusion matrices of shape (n_annotators, n_classes, n_classes)
        class_prior: Estimated class priors of shape (n_classes,)
    """
    n_agents, n_items, m = y.shape
    K = n_classes
    R = n_annotators
    
    # Initialize confusion matrices
    if init_confusion is not None:
        pi = init_confusion.copy()
    else:
        # Initialize with slight diagonal dominance (0.8 on diagonal)
        pi = np.full((R, K, K), 0.2 / (K - 1))  # Off-diagonal: 0.1 each
        for r in range(R):
            np.fill_diagonal(pi[r], 0.8)        # Diagonal: 0.8
    
    # Initialize class prior as uniform
    class_prior = np.ones(K) / K
    
    # Flatten for easier iteration
    # Create index mapping
    total_items = n_agents * n_items
    
    # Precompute which annotators labeled which items
    # labels_by_item[idx] = list of (annotator, label) pairs
    labels_by_item = []
    for a in range(n_agents):
        for i in range(n_items):
            item_labels = []
            for j in range(m):
                r = annotators[a, i, j]
                label = y[a, i, j]
                item_labels.append((r, label))
            labels_by_item.append(item_labels)
    
    # Initialize gamma via majority vote
    gamma = np.zeros((total_items, K))
    for idx in range(total_items):
        labels = [label for _, label in labels_by_item[idx]]
        counts = np.bincount(labels, minlength=K)
        # Soft initialization: normalize counts to probabilities
        gamma[idx] = (counts + 1e-10) / (counts.sum() + K * 1e-10)
    
    # EM iterations
    for iteration in range(max_iter):
        old_pi = pi.copy()
        
        # E-step: compute posteriors
        for idx in range(total_items):
            # Log likelihood for each class
            log_lik = np.log(class_prior + 1e-10)
            
            for r, label in labels_by_item[idx]:
                log_lik += np.log(pi[r, :, label] + 1e-10)
            
            # Normalize to get posterior
            log_lik -= logsumexp(log_lik)
            gamma[idx] = np.exp(log_lik)
        
        # M-step: update confusion matrices and prior
        # Update class prior
        class_prior = gamma.mean(axis=0)
        class_prior = np.clip(class_prior, 1e-10, 1 - 1e-10)
        class_prior /= class_prior.sum()
        
        # Update confusion matrices
        # For each annotator, for each true class, count expected labels
        for r in range(R):
            for c in range(K):
                numerator = np.zeros(K)
                denominator = 0
                
                for idx in range(total_items):
                    for ann, label in labels_by_item[idx]:
                        if ann == r:
                            numerator[label] += gamma[idx, c]
                            denominator += gamma[idx, c]
                
                if denominator > 1e-10:
                    pi[r, c] = numerator / denominator
                else:
                    # Keep uniform if no observations
                    pi[r, c] = np.ones(K) / K
        
        # Check convergence
        if np.max(np.abs(pi - old_pi)) < tol:
            break
    
    # Reshape gamma back
    gamma = gamma.reshape(n_agents, n_items, K)
    
    return gamma, pi, class_prior


def posterior_expected_credit(
    y: np.ndarray,
    annotators: np.ndarray,
    n_annotators: int,
    credit_mapping: List[float],
    **em_kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Posterior Expected Credit (soft aggregation).
    
    S_a = (1/N) * sum_i sum_c gamma_ai,c * v(c)
    
    Args:
        y: Observed labels
        annotators: Annotator IDs
        n_annotators: Total annotators
        credit_mapping: Credit values
        **em_kwargs: Additional arguments for EM
    
    Returns:
        scores: Agent scores
        gamma: Posterior probabilities
    """
    gamma, pi, _ = dawid_skene_em(y, annotators, n_annotators, **em_kwargs)
    
    n_agents, n_items, n_classes = gamma.shape
    credit = np.array(credit_mapping)
    
    # Expected credit per item: sum_c gamma * v(c)
    expected_credit = np.einsum('aik,k->ai', gamma, credit)
    
    # Mean per agent
    scores = expected_credit.mean(axis=1)
    
    return scores, gamma


def aggregate_all_methods(
    y: np.ndarray,
    annotators: np.ndarray,
    n_annotators: int,
    credit_mapping: List[float],
    **em_kwargs
) -> dict:
    """
    Run all aggregation methods and return results.
    
    Returns dict with keys:
        - mv_labels, mv_scores
        - ds_labels, ds_scores, ds_gamma
        - pec_scores, pec_gamma
        - confusion_matrices
    """
    # Majority Vote
    mv_labels, mv_scores = majority_vote(y, credit_mapping)
    
    # Dawid-Skene EM (used for DS Hard and PEC)
    gamma, pi, class_prior = dawid_skene_em(
        y, annotators, n_annotators, **em_kwargs
    )
    
    credit = np.array(credit_mapping)
    
    # DS Hard
    ds_labels = np.argmax(gamma, axis=2)
    ds_scores = credit[ds_labels].mean(axis=1)
    
    # PEC
    expected_credit = np.einsum('aik,k->ai', gamma, credit)
    pec_scores = expected_credit.mean(axis=1)
    
    return {
        "mv_labels": mv_labels,
        "mv_scores": mv_scores,
        "ds_labels": ds_labels,
        "ds_scores": ds_scores,
        "ds_gamma": gamma,
        "pec_scores": pec_scores,
        "pec_gamma": gamma,
        "confusion_matrices": pi,
        "class_prior": class_prior,
    }
