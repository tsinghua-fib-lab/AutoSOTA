from __future__ import annotations

from graph_comparison.graph_comparison import GraphDatasetComparison


def test_zero_similarity_threshold_is_respected() -> None:
    similarities = {
        "Negative": -0.2,
        "Zero": 0.0,
        "Positive": 0.3,
    }

    selected, threshold, min_top_s = GraphDatasetComparison.determine_similar_datasets(
        similarities,
        initial_threshold=0.0,
        min_top_s=1,
    )

    assert threshold == 0.0
    assert min_top_s == 1
    assert selected == [("Positive", 0.3), ("Zero", 0.0)]
