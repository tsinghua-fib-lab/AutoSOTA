from __future__ import annotations

from model_refinement.model_refinement import ModelRefinement


def test_refinement_summary_records_selected_knowledge_source(tmp_path) -> None:
    ModelRefinement.report_and_save_refinement_summary(
        unseen_dataset="Cora",
        knowledge_source=[("CiteSeer", 0.48), ("Cornell", 0.28)],
        initial_proposal=({"stage": "mean"}, 0.8, 0.01),
        final_proposal=({"stage": "ppr_01"}, 0.85, 0.02),
        response_save_path=tmp_path,
    )

    summary = (tmp_path / "refinement_summary.txt").read_text()
    assert "CiteSeer" in summary
    assert "Cornell" in summary
    assert "bayesian_update" not in summary
