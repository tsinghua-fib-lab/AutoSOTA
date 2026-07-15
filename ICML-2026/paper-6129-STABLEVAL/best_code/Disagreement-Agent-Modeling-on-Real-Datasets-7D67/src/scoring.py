"""
Unified Scoring Module for Agent Evaluation.

This module provides a unified interface to compute agent scores using:
1. Majority Vote (baseline)
2. Dawid-Skene Hard (EM with majority vote initialization)
3. Posterior Expected Credit (EM with probabilistic scoring)

It also provides comparison and analysis tools.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .majority_vote import (
    compute_agent_scores_majority_vote,
    compute_item_scores_majority_vote,
    bootstrap_agent_scores as bootstrap_mv,
    get_class_values
)
from .disagreement_model import (
    DisagreementModel,
    compute_agent_scores_pec,
    compute_agent_scores_ds_hard,
    compute_posterior_expected_credit,
    compute_item_ambiguity,
    bootstrap_agent_scores_pec,
    bootstrap_agent_scores_ds_hard
)


@dataclass
class ScoringResults:
    """Container for all scoring results."""
    majority_vote: pd.DataFrame
    dawid_skene_hard: pd.DataFrame
    posterior_expected_credit: pd.DataFrame
    
    # Optional detailed results
    item_scores_mv: Optional[pd.DataFrame] = None
    item_scores_pec: Optional[pd.DataFrame] = None
    item_ambiguity: Optional[pd.DataFrame] = None
    annotator_quality: Optional[pd.DataFrame] = None
    
    # Bootstrap results (if computed)
    mv_bootstrap: Optional[pd.DataFrame] = None
    ds_bootstrap: Optional[pd.DataFrame] = None
    pec_bootstrap: Optional[pd.DataFrame] = None


def compute_all_scores(
    df: pd.DataFrame,
    n_classes: Optional[int] = None,
    class_values: Optional[Dict[int, float]] = None,
    compute_item_details: bool = True,
    verbose: bool = True,
    pec_temperature: float = 1.0,
    adaptive_c: float = 0.0,
    consistency_alpha: float = 0.0
) -> ScoringResults:
    """
    Compute agent scores using all three methods.
    
    Args:
        df: DataFrame with columns item_id, agent_id, annotator_id, label
        n_classes: Number of classes (auto-detected if None)
        class_values: Mapping from class to numeric value
        compute_item_details: Also compute per-item scores and ambiguity
        verbose: Print progress
        
    Returns:
        ScoringResults with all scores
    """
    if n_classes is None:
        n_classes = df['label'].nunique()
    
    if class_values is None:
        class_values = get_class_values(n_classes)
    
    if verbose:
        print("=" * 60)
        print("COMPUTING ALL AGENT SCORES")
        print("=" * 60)
    
    # 1. Majority Vote
    if verbose:
        print("\n1. Computing Majority Vote scores...")
    mv_scores = compute_agent_scores_majority_vote(df, class_values, n_classes)
    mv_scores = mv_scores.rename(columns={'score': 'score_mv'})
    
    # 2. Dawid-Skene Model (single model used for both DS Hard and PEC)
    if verbose:
        print("\n2. Fitting Dawid-Skene model...")
    model_ds = DisagreementModel(n_classes=n_classes, verbose=verbose)
    model_ds.adaptive_c = adaptive_c
    model_ds.consistency_alpha = consistency_alpha
    model_ds.fit(df, initialization='majority_vote')
    
    # 3. Dawid-Skene Hard Scores (using argmax of posteriors)
    if verbose:
        print("\n3. Computing Dawid-Skene (Hard) scores...")
    ds_scores = compute_agent_scores_ds_hard(model_ds, df, class_values)
    ds_scores = ds_scores.rename(columns={'score': 'score_ds'})
    
    # 4. Posterior Expected Credit (using DS posteriors with soft scoring)
    if verbose:
        print("\n4. Computing Posterior Expected Credit scores...")
    pec_scores = compute_agent_scores_pec(model_ds, df, class_values, temperature=pec_temperature)
    pec_scores = pec_scores.rename(columns={'score': 'score_pec'})
    
    # Compute item-level details if requested
    item_scores_mv = None
    item_scores_pec = None
    item_ambiguity = None
    annotator_quality = None
    
    if compute_item_details:
        if verbose:
            print("\n5. Computing item-level details...")
        item_scores_mv = compute_item_scores_majority_vote(df, class_values, n_classes)
        item_scores_pec = compute_posterior_expected_credit(model_ds, df, class_values, temperature=pec_temperature)
        item_ambiguity = compute_item_ambiguity(model_ds)
        annotator_quality = model_ds.get_annotator_quality_scores()
    
    if verbose:
        print("\n" + "=" * 60)
        print("SCORING COMPLETE")
        print("=" * 60)
    
    return ScoringResults(
        majority_vote=mv_scores,
        dawid_skene_hard=ds_scores,
        posterior_expected_credit=pec_scores,
        item_scores_mv=item_scores_mv,
        item_scores_pec=item_scores_pec,
        item_ambiguity=item_ambiguity,
        annotator_quality=annotator_quality
    )


def compute_all_scores_with_bootstrap(
    df: pd.DataFrame,
    n_classes: Optional[int] = None,
    class_values: Optional[Dict[int, float]] = None,
    n_bootstrap: int = 500,
    random_state: Optional[int] = 42,
    verbose: bool = True,
    pec_temperature: float = 1.0,
    adaptive_c: float = 0.0,
    consistency_alpha: float = 0.0
) -> ScoringResults:
    """
    Compute agent scores with bootstrap confidence intervals.
    
    Args:
        df: DataFrame with annotation data
        n_classes: Number of classes
        class_values: Mapping from class to numeric value
        n_bootstrap: Number of bootstrap iterations
        random_state: Random seed
        verbose: Print progress
        
    Returns:
        ScoringResults with bootstrap confidence intervals
    """
    if n_classes is None:
        n_classes = df['label'].nunique()
    
    if class_values is None:
        class_values = get_class_values(n_classes)
    
    # First compute base scores
    results = compute_all_scores(df, n_classes, class_values, verbose=verbose, pec_temperature=pec_temperature, adaptive_c=adaptive_c, consistency_alpha=consistency_alpha)
    
    if verbose:
        print(f"\nComputing bootstrap confidence intervals ({n_bootstrap} iterations)...")
    
    # Bootstrap for Majority Vote
    if verbose:
        print("  - Majority Vote bootstrap...")
    results.mv_bootstrap = bootstrap_mv(
        df, n_bootstrap, class_values, n_classes, random_state
    )
    
    # Bootstrap for Dawid-Skene Hard
    if verbose:
        print("  - Dawid-Skene (Hard) bootstrap...")
    results.ds_bootstrap = bootstrap_agent_scores_ds_hard(
        df, n_bootstrap, class_values, n_classes, random_state, verbose=False
    )
    
    # Bootstrap for PEC (uses same DS model but soft scoring)
    if verbose:
        print("  - Posterior Expected Credit bootstrap...")
    results.pec_bootstrap = bootstrap_agent_scores_pec(
        df, n_bootstrap, class_values, n_classes, 'majority_vote', random_state, verbose=False
    )
    
    if verbose:
        print("Bootstrap complete!")
    
    return results


def create_comparison_table(results: ScoringResults) -> pd.DataFrame:
    """
    Create a comparison table of agent scores across all methods.
    
    Returns:
        DataFrame with agent_id and scores from each method
    """
    # Merge all scores
    comparison = results.majority_vote[['agent_id', 'score_mv', 'n_items']].copy()
    comparison = comparison.merge(
        results.dawid_skene_hard[['agent_id', 'score_ds']],
        on='agent_id'
    )
    comparison = comparison.merge(
        results.posterior_expected_credit[['agent_id', 'score_pec']],
        on='agent_id'
    )
    
    # Add difference columns
    comparison['diff_ds_mv'] = comparison['score_ds'] - comparison['score_mv']
    comparison['diff_pec_mv'] = comparison['score_pec'] - comparison['score_mv']
    comparison['diff_pec_ds'] = comparison['score_pec'] - comparison['score_ds']
    
    # Add ranks
    comparison['rank_mv'] = comparison['score_mv'].rank(ascending=False).astype(int)
    comparison['rank_ds'] = comparison['score_ds'].rank(ascending=False).astype(int)
    comparison['rank_pec'] = comparison['score_pec'].rank(ascending=False).astype(int)
    
    return comparison.sort_values('score_pec', ascending=False)


def compute_ranking_stability(
    df: pd.DataFrame,
    n_subsets: int = 50,
    subset_fraction: float = 0.7,
    n_classes: Optional[int] = None,
    class_values: Optional[Dict[int, float]] = None,
    random_state: Optional[int] = 42,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compute ranking stability by comparing rankings across random annotator subsets.
    
    Args:
        df: DataFrame with annotation data
        n_subsets: Number of random subsets to evaluate
        subset_fraction: Fraction of annotators to include in each subset
        n_classes: Number of classes
        class_values: Mapping from class to numeric value
        random_state: Random seed
        verbose: Print progress
        
    Returns:
        DataFrame with stability metrics for each method
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    if n_classes is None:
        n_classes = df['label'].nunique()
    
    if class_values is None:
        class_values = get_class_values(n_classes)
    
    annotators = df['annotator_id'].unique()
    n_annotators = len(annotators)
    subset_size = int(n_annotators * subset_fraction)
    
    if verbose:
        print(f"Computing ranking stability with {n_subsets} subsets of {subset_size} annotators...")
    
    # Store rankings for each subset
    rankings_mv = []
    rankings_ds = []
    rankings_pec = []
    
    for i in range(n_subsets):
        # Sample annotators
        sampled_annotators = np.random.choice(annotators, size=subset_size, replace=False)
        subset_df = df[df['annotator_id'].isin(sampled_annotators)]
        
        try:
            # Compute scores
            results = compute_all_scores(subset_df, n_classes, class_values, 
                                        compute_item_details=False, verbose=False)
            
            # Store rankings
            rankings_mv.append(results.majority_vote.set_index('agent_id')['score_mv'].rank(ascending=False))
            rankings_ds.append(results.dawid_skene_hard.set_index('agent_id')['score_ds'].rank(ascending=False))
            rankings_pec.append(results.posterior_expected_credit.set_index('agent_id')['score_pec'].rank(ascending=False))
            
            if verbose and (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{n_subsets} subsets")
        except Exception as e:
            if verbose:
                print(f"  Warning: Subset {i+1} failed: {e}")
    
    # Compute stability metrics
    def compute_stability(rankings_list):
        if len(rankings_list) < 2:
            return {'mean_rank_std': np.nan, 'mean_rank_range': np.nan}
        
        rankings_df = pd.DataFrame(rankings_list)
        mean_std = rankings_df.std().mean()
        mean_range = (rankings_df.max() - rankings_df.min()).mean()
        
        return {
            'mean_rank_std': mean_std,
            'mean_rank_range': mean_range
        }
    
    stability = {
        'method': ['Majority Vote', 'Dawid-Skene (Hard)', 'Posterior Expected Credit'],
        'mean_rank_std': [
            compute_stability(rankings_mv)['mean_rank_std'],
            compute_stability(rankings_ds)['mean_rank_std'],
            compute_stability(rankings_pec)['mean_rank_std']
        ],
        'mean_rank_range': [
            compute_stability(rankings_mv)['mean_rank_range'],
            compute_stability(rankings_ds)['mean_rank_range'],
            compute_stability(rankings_pec)['mean_rank_range']
        ]
    }
    
    return pd.DataFrame(stability)


def identify_score_changes(
    comparison: pd.DataFrame,
    threshold: float = 0.05
) -> Dict[str, pd.DataFrame]:
    """
    Identify agents whose scores changed significantly between methods.
    
    Args:
        comparison: Comparison table from create_comparison_table
        threshold: Minimum absolute difference to be considered significant
        
    Returns:
        Dictionary with DataFrames of agents that moved up/down significantly
    """
    # Agents that improved from MV to PEC
    improved = comparison[comparison['diff_pec_mv'] > threshold].sort_values('diff_pec_mv', ascending=False)
    
    # Agents that declined from MV to PEC
    declined = comparison[comparison['diff_pec_mv'] < -threshold].sort_values('diff_pec_mv')
    
    # Agents with large rank changes
    comparison['rank_change'] = comparison['rank_mv'] - comparison['rank_pec']
    rank_movers = comparison[abs(comparison['rank_change']) > 2].sort_values('rank_change', ascending=False)
    
    return {
        'improved': improved,
        'declined': declined,
        'rank_movers': rank_movers
    }


def print_results_summary(results: ScoringResults) -> None:
    """Print a formatted summary of scoring results."""
    print("\n" + "=" * 70)
    print("AGENT SCORING RESULTS SUMMARY")
    print("=" * 70)
    
    comparison = create_comparison_table(results)
    
    print("\nAgent Scores Comparison:")
    print("-" * 70)
    print(comparison[['agent_id', 'score_mv', 'score_ds', 'score_pec', 
                      'rank_mv', 'rank_ds', 'rank_pec', 'n_items']].to_string(index=False))
    
    print("\n\nScore Differences (vs Majority Vote):")
    print("-" * 70)
    print(comparison[['agent_id', 'diff_ds_mv', 'diff_pec_mv']].to_string(index=False))
    
    if results.annotator_quality is not None:
        print("\n\nAnnotator Quality (Top 10):")
        print("-" * 70)
        print(results.annotator_quality.head(10).to_string(index=False))
    
    if results.item_ambiguity is not None:
        print("\n\nMost Ambiguous Items (Top 10):")
        print("-" * 70)
        print(results.item_ambiguity.head(10).to_string(index=False))


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing unified scoring module...")
    
    # Create synthetic data
    np.random.seed(42)
    
    items = [f"item_{i}" for i in range(20)]
    agents = ['agent_A', 'agent_B', 'agent_C']
    annotators = ['ann_1', 'ann_2', 'ann_3', 'ann_4']
    
    data = []
    for item in items:
        agent = np.random.choice(agents)
        true_label = np.random.choice([0, 1])
        
        for ann in annotators:
            # Add some noise based on annotator
            if ann == 'ann_1':  # High quality
                label = true_label if np.random.random() > 0.1 else 1 - true_label
            elif ann == 'ann_4':  # Low quality
                label = true_label if np.random.random() > 0.4 else 1 - true_label
            else:  # Medium quality
                label = true_label if np.random.random() > 0.2 else 1 - true_label
            
            data.append({
                'item_id': item,
                'agent_id': agent,
                'annotator_id': ann,
                'label': label
            })
    
    df = pd.DataFrame(data)
    
    # Compute all scores
    results = compute_all_scores(df, n_classes=2, verbose=True)
    
    # Print summary
    print_results_summary(results)
    
    # Compute stability
    print("\n\nRanking Stability Analysis:")
    print("-" * 70)
    stability = compute_ranking_stability(df, n_subsets=20, verbose=True)
    print(stability.to_string(index=False))
    
    print("\n\nUnified scoring module test completed!")
