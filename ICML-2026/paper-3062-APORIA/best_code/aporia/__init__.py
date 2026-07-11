"""aporia: geometric analysis of small-sized LLM hallucinations.

The acronym stands for *Aggregate Prompt-wise Observation Retrieving
Instability via Asymmetry*; the word *aporia* (ἀπορία) is the Socratic
concept of puzzlement that surfaces when a fluent claim contradicts
itself — exactly what a hallucination is in the framing of this paper.
"""

__version__ = "0.1.0"

# ---- config ---------------------------------------------------------------
from .config import (
    Config,
    DatasetConfig,
    ExperimentConfig,
    CacheConfig,
    ModelSpec,
    load_config,
)

# ---- data -----------------------------------------------------------------
from .data import (
    extract_prompt_data,
    generate_fixed_test_sets,
    load_dataframe,
    prompt_ids_by_model,
    split_by_label,
    subsample_training_set,
)

# ---- projections ----------------------------------------------------------
from .projections import (
    CentroidFeaturesProjection,
    FisherProjection,
    IdentityProjection,
    ProjectionBase,
    RandomProjection,
    SupervisedUMAPProjection,
    WhitenedPCAProjection,
    fisher_direction,
)

# ---- structural -----------------------------------------------------------
from .structural import (
    analyse_prompt,
    collect_prompt_result,
    compute_distance_distributions,
    run_structural_analysis,
    wasserstein_GG_HH,
    wasserstein_null_model,
)

# ---- label propagation ----------------------------------------------------
from .label_propagation import (
    CentroidPropagator,
    SKLearnPropagator,
    WassersteinLabelPropagator,
    run_full_label_propagation_study,
    run_label_propagation_experiment,
)

# ---- evaluation -----------------------------------------------------------
from .evaluation import LabelPropagationEvaluator

# ---- sensitivity ----------------------------------------------------------
from .sensitivity import (
    run_full_lambda_sensitivity_study,
    run_lambda_sensitivity_experiment,
)

# ---- utils ----------------------------------------------------------------
from .utils import (
    aggregate_metric_over_prompts,
    build_model_size_order,
    matplotlib_latex_preamble,
    reorder_selected_keys_by_model_size,
    select_prompt_by_fraction,
    select_representative_prompts,
)

# ---- format / plotting ----------------------------------------------------
from .format   import fmt, fmt_pct, apply_deco, rank_decor
from .plotting import plot_metric_boxplots_two_panels

__all__ = [
    "__version__",
    # config
    "Config", "DatasetConfig", "ExperimentConfig", "CacheConfig", "ModelSpec",
    "load_config",
    # data
    "extract_prompt_data", "generate_fixed_test_sets", "load_dataframe",
    "prompt_ids_by_model", "split_by_label", "subsample_training_set",
    # projections
    "CentroidFeaturesProjection", "FisherProjection", "IdentityProjection",
    "ProjectionBase", "RandomProjection",
    "SupervisedUMAPProjection", "WhitenedPCAProjection", "fisher_direction",
    # structural
    "analyse_prompt", "collect_prompt_result", "compute_distance_distributions",
    "run_structural_analysis", "wasserstein_GG_HH", "wasserstein_null_model",
    # LP
    "CentroidPropagator", "SKLearnPropagator",
    "WassersteinLabelPropagator", "run_full_label_propagation_study",
    "run_label_propagation_experiment",
    # eval
    "LabelPropagationEvaluator",
    # sensitivity
    "run_full_lambda_sensitivity_study", "run_lambda_sensitivity_experiment",
    # utils
    "aggregate_metric_over_prompts", "build_model_size_order",
    "matplotlib_latex_preamble",
    "reorder_selected_keys_by_model_size", "select_prompt_by_fraction",
    "select_representative_prompts",
    # format / plotting
    "fmt", "fmt_pct", "apply_deco", "rank_decor",
    "plot_metric_boxplots_two_panels",
]


# ---- phase 3 additions ----------------------------------------------------
from .sensitivity import (
    aggregate_metric_over_lambda,
    aggregate_over_runs,
    compute_best_scores,
    compute_relative_loss,
    aggregate_relative_loss,
    select_lambda_min_regret,
    compute_average_best_lambda,
    global_stats,
)
from .structural import (
    run_structural_ablation_for_prompt,
    compute_separability,
    build_separability_df,
    prepare_separability_violin_df,
)
from .projections import (
    extract_fisher_directions,
)
from .utils import (
    cosine_similarity_matrix,
    collect_similarity_pairs,
)
