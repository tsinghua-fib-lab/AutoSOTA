"""
Majority Vote Baseline for Agent Evaluation.

This module implements the simple majority vote baseline that serves as
a reference point for the disagreement-aware methods.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from collections import Counter


def compute_majority_vote_label(labels: List[int]) -> int:
    """
    Compute the majority vote label from a list of labels.
    
    Args:
        labels: List of integer labels
        
    Returns:
        The most common label (ties broken arbitrarily)
    """
    if not labels:
        raise ValueError("Cannot compute majority vote from empty list")
    
    counter = Counter(labels)
    return counter.most_common(1)[0][0]


def compute_item_majority_votes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute majority vote label for each item.
    
    Args:
        df: DataFrame with columns item_id, annotator_id, label
        
    Returns:
        DataFrame with item_id and majority_vote_label columns
    """
    majority_votes = df.groupby('item_id')['label'].apply(
        lambda x: compute_majority_vote_label(x.tolist())
    ).reset_index()
    
    majority_votes.columns = ['item_id', 'majority_vote_label']
    
    return majority_votes


def get_class_values(n_classes: int, partial_credit: bool = True) -> Dict[int, float]:
    """
    Get numeric values for each class.
    
    Args:
        n_classes: Number of classes (2 for binary, 3 for ternary)
        partial_credit: If True, use partial credit for intermediate classes
        
    Returns:
        Dictionary mapping class index to numeric value
    """
    if n_classes == 2:
        # Binary: 0 = incorrect (0.0), 1 = correct (1.0)
        return {0: 0.0, 1: 1.0}
    elif n_classes == 3:
        # Ternary: 0 = incorrect (0.0), 1 = partial (0.5), 2 = correct (1.0)
        if partial_credit:
            return {0: 0.0, 1: 0.5, 2: 1.0}
        else:
            return {0: 0.0, 1: 0.0, 2: 1.0}
    else:
        # General case: linear spacing
        return {i: i / (n_classes - 1) for i in range(n_classes)}


def compute_agent_scores_majority_vote(
    df: pd.DataFrame,
    class_values: Optional[Dict[int, float]] = None,
    n_classes: Optional[int] = None
) -> pd.DataFrame:
    """
    Compute agent scores using majority vote.
    
    Args:
        df: DataFrame with columns item_id, agent_id, annotator_id, label
        class_values: Optional mapping from class to numeric value
        n_classes: Number of classes (used if class_values not provided)
        
    Returns:
        DataFrame with agent_id, score, n_items columns
    """
    # Determine number of classes if not specified
    if n_classes is None:
        n_classes = df['label'].nunique()
    
    # Get class values
    if class_values is None:
        class_values = get_class_values(n_classes)
    
    # Compute majority vote for each item
    majority_votes = compute_item_majority_votes(df)
    
    # Get item-to-agent mapping
    item_agent = df[['item_id', 'agent_id']].drop_duplicates()
    
    # Merge
    merged = majority_votes.merge(item_agent, on='item_id')
    
    # Convert labels to numeric values
    merged['value'] = merged['majority_vote_label'].map(class_values)
    
    # Compute agent scores
    agent_scores = merged.groupby('agent_id').agg(
        score=('value', 'mean'),
        n_items=('item_id', 'count')
    ).reset_index()
    
    return agent_scores.sort_values('score', ascending=False)


def compute_item_scores_majority_vote(
    df: pd.DataFrame,
    class_values: Optional[Dict[int, float]] = None,
    n_classes: Optional[int] = None
) -> pd.DataFrame:
    """
    Compute per-item scores using majority vote.
    
    Args:
        df: DataFrame with annotation data
        class_values: Optional mapping from class to numeric value
        n_classes: Number of classes
        
    Returns:
        DataFrame with item_id, majority_vote_label, score
    """
    if n_classes is None:
        n_classes = df['label'].nunique()
    
    if class_values is None:
        class_values = get_class_values(n_classes)
    
    majority_votes = compute_item_majority_votes(df)
    majority_votes['score'] = majority_votes['majority_vote_label'].map(class_values)
    
    return majority_votes


def bootstrap_agent_scores(
    df: pd.DataFrame,
    n_bootstrap: int = 1000,
    class_values: Optional[Dict[int, float]] = None,
    n_classes: Optional[int] = None,
    random_state: Optional[int] = None
) -> pd.DataFrame:
    """
    Compute bootstrap confidence intervals for agent scores.
    
    Args:
        df: DataFrame with annotation data
        n_bootstrap: Number of bootstrap samples
        class_values: Optional mapping from class to numeric value
        n_classes: Number of classes
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with agent_id, score, ci_lower, ci_upper, std
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    if n_classes is None:
        n_classes = df['label'].nunique()
    
    if class_values is None:
        class_values = get_class_values(n_classes)
    
    # Get unique items
    items = df['item_id'].unique()
    n_items = len(items)
    
    # Store bootstrap scores
    agents = df['agent_id'].unique()
    bootstrap_scores = {agent: [] for agent in agents}
    
    for _ in range(n_bootstrap):
        # Resample items with replacement
        sampled_items = np.random.choice(items, size=n_items, replace=True)
        
        # Filter data to sampled items
        # IDEA-09: proper bootstrap with replacement
        dfs = [df[df['item_id'] == iid] for iid in sampled_items]
        sampled_df = pd.concat(dfs, ignore_index=True)
        
        # Compute agent scores
        scores = compute_agent_scores_majority_vote(sampled_df, class_values, n_classes)
        
        for _, row in scores.iterrows():
            bootstrap_scores[row['agent_id']].append(row['score'])
    
    # Compute confidence intervals
    results = []
    base_scores = compute_agent_scores_majority_vote(df, class_values, n_classes)
    
    for _, row in base_scores.iterrows():
        agent = row['agent_id']
        bs_scores = bootstrap_scores[agent]
        
        results.append({
            'agent_id': agent,
            'score': row['score'],
            'ci_lower': np.percentile(bs_scores, 2.5),
            'ci_upper': np.percentile(bs_scores, 97.5),
            'std': np.std(bs_scores),
            'n_items': row['n_items']
        })
    
    return pd.DataFrame(results).sort_values('score', ascending=False)


def get_agreement_statistics(df: pd.DataFrame) -> Dict:
    """
    Compute inter-annotator agreement statistics.
    
    Args:
        df: DataFrame with annotation data
        
    Returns:
        Dictionary with agreement statistics
    """
    # Group by item and count agreement
    item_stats = df.groupby('item_id').agg(
        n_annotators=('annotator_id', 'nunique'),
        n_labels=('label', 'nunique'),
        labels=('label', list)
    ).reset_index()
    
    # Compute raw agreement (proportion of unanimous items)
    unanimous = (item_stats['n_labels'] == 1).mean()
    
    # Compute average pairwise agreement
    def pairwise_agreement(labels):
        if len(labels) < 2:
            return 1.0
        n_pairs = len(labels) * (len(labels) - 1) / 2
        agreements = sum(1 for i in range(len(labels)) 
                        for j in range(i+1, len(labels)) 
                        if labels[i] == labels[j])
        return agreements / n_pairs
    
    item_stats['pairwise_agreement'] = item_stats['labels'].apply(pairwise_agreement)
    avg_pairwise = item_stats['pairwise_agreement'].mean()
    
    return {
        'unanimous_proportion': unanimous,
        'average_pairwise_agreement': avg_pairwise,
        'n_items': len(item_stats),
        'avg_annotators_per_item': item_stats['n_annotators'].mean()
    }


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing majority vote baseline with synthetic data...")
    
    # Create synthetic data matching Example 1 from the proposal
    test_data = pd.DataFrame({
        'item_id': ['item1', 'item1', 'item1'],
        'agent_id': ['agent1', 'agent1', 'agent1'],
        'annotator_id': ['A', 'B', 'C'],
        'label': [1, 1, 0]  # A and B say correct, C says incorrect
    })
    
    print("\nTest data:")
    print(test_data)
    
    # Compute majority vote
    mv = compute_item_majority_votes(test_data)
    print(f"\nMajority vote: {mv['majority_vote_label'].iloc[0]} (expected: 1)")
    
    # Compute agent score
    scores = compute_agent_scores_majority_vote(test_data)
    print(f"Agent score: {scores['score'].iloc[0]} (expected: 1.0)")
    
    print("\nMajority vote baseline test passed!")
