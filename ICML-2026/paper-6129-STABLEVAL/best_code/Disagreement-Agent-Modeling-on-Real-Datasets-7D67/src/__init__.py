"""
Disagreement-Aware Evaluation Pipeline

A framework for evaluating AI agents using multiple annotator labels
with proper handling of disagreement and uncertainty.

Implements three scoring methods:
1. Majority Vote (baseline)
2. Dawid-Skene Hard (EM with majority vote initialization)
3. Posterior Expected Credit (probabilistic scoring)
"""

from .data_loader import (
    load_single_csv,
    load_all_data,
    get_data_summary,
    print_data_summary,
    create_item_agent_mapping,
    get_label_matrix
)

from .majority_vote import (
    compute_majority_vote_label,
    compute_item_majority_votes,
    compute_agent_scores_majority_vote,
    compute_item_scores_majority_vote,
    bootstrap_agent_scores,
    get_class_values,
    get_agreement_statistics
)

from .disagreement_model import (
    DisagreementModel,
    compute_posterior_expected_credit,
    compute_agent_scores_pec,
    compute_item_ambiguity,
    bootstrap_agent_scores_pec
)

from .scoring import (
    ScoringResults,
    compute_all_scores,
    compute_all_scores_with_bootstrap,
    create_comparison_table,
    compute_ranking_stability,
    identify_score_changes,
    print_results_summary
)

from .visualization import (
    plot_agent_scores_comparison,
    plot_score_scatter,
    plot_annotator_confusion_matrices,
    plot_annotator_quality,
    plot_ambiguity_distribution,
    plot_bootstrap_confidence_intervals,
    plot_ranking_stability,
    plot_score_differences,
    create_all_plots
)

__version__ = '1.0.0'
__all__ = [
    # Data loading
    'load_single_csv',
    'load_all_data',
    'get_data_summary',
    'print_data_summary',
    'create_item_agent_mapping',
    'get_label_matrix',
    
    # Majority vote
    'compute_majority_vote_label',
    'compute_item_majority_votes',
    'compute_agent_scores_majority_vote',
    'compute_item_scores_majority_vote',
    'bootstrap_agent_scores',
    'get_class_values',
    'get_agreement_statistics',
    
    # Disagreement model
    'DisagreementModel',
    'compute_posterior_expected_credit',
    'compute_agent_scores_pec',
    'compute_item_ambiguity',
    'bootstrap_agent_scores_pec',
    
    # Unified scoring
    'ScoringResults',
    'compute_all_scores',
    'compute_all_scores_with_bootstrap',
    'create_comparison_table',
    'compute_ranking_stability',
    'identify_score_changes',
    'print_results_summary',
    
    # Visualization
    'plot_agent_scores_comparison',
    'plot_score_scatter',
    'plot_annotator_confusion_matrices',
    'plot_annotator_quality',
    'plot_ambiguity_distribution',
    'plot_bootstrap_confidence_intervals',
    'plot_ranking_stability',
    'plot_score_differences',
    'create_all_plots'
]
