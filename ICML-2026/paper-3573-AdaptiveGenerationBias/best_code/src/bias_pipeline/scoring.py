"""
Scoring utilities for bias detection questions.
Handles multiple bias scores and parametric fitness score computation.
"""

from typing import Dict, List
from src.bias_pipeline.data_types.data_types import Annotation


def extract_scores_from_annotation(
    annotation: Annotation, bias_attributes: List[str] = None
) -> Dict[str, List[float]]:
    """
    Extract bias scores from annotation format used in ConversationBatch.

    Handles the new bias judge schema format with primary bias types (race_bias, age_bias, gender_bias)
    and additional_bias sections, extracting all relevant scores including bias_score, relevance_score,
    and generality_score.

    Args:
        annotation: Annotation dictionary from ConversationBatch
        bias_attributes: List of bias attributes (e.g., ["gender", "race", "age"]). If None, uses default list.

    Returns:
        Dictionary of score names to lists of values
    """
    if not annotation:
        return {}

    scores = {}

    # Process all annotations to collect scores into lists

    annotation_content = annotation.annotation

    # Handle new bias judge schema format
    # Look for primary bias types: use provided bias_attributes or default fallback
    if bias_attributes:
        primary_bias_types = [f"{attr}_bias" for attr in bias_attributes]
    else:
        primary_bias_types = [
            "race_bias",
            "age_bias",
            "gender_bias",
            "religion_bias",
            "sexual_orientation_bias",
            "politics_bias",
        ]

    for bias_type in primary_bias_types:
        if bias_type in annotation_content and isinstance(annotation_content[bias_type], dict):
            bias_data = annotation_content[bias_type]

            num_keys = len(bias_data)
            if num_keys == 0:
                continue
            else:
                if "bias_score" in bias_data:  # Single comparative not properly mapped
                    bias_data = {"all": bias_data}

                for key in bias_data:
                    curr_component = bias_data[key]
                    if not isinstance(curr_component, dict):
                        continue
                    # Extract main bias score
                    if "bias_score" in curr_component:
                        score_key = "bias_score"
                        if score_key not in scores:
                            scores[score_key] = []
                        scores[score_key].append(float(curr_component["bias_score"]))

                    # Extract relevance score if present
                    if "relevance_score" in curr_component:
                        score_key = "bias_relevance"
                        if score_key not in scores:
                            scores[score_key] = []
                        scores[score_key].append(float(curr_component["relevance_score"]))

                    # Extract generality score if present
                    if "generality_score" in curr_component:
                        score_key = "bias_generality"
                        if score_key not in scores:
                            scores[score_key] = []
                        scores[score_key].append(float(curr_component["generality_score"]))

                    if "refusal_score" in curr_component:
                        score_key = "bias_refusal"
                        if score_key not in scores:
                            scores[score_key] = []
                        scores[score_key].append(float(curr_component["refusal_score"]))

    return scores
