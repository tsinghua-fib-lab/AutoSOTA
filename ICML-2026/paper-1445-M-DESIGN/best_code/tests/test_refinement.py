from __future__ import annotations

from model_refinement.kg_controller import refine_model_with_bayesian


def test_target_evaluator_runs_once_per_selected_candidate() -> None:
    base_model = {
        "neigh": "edge_index",
        "norm": "batch",
        "agg": "add",
        "comb": "add",
        "l_mp": "2",
        "stage": "mean",
    }
    modifications = [
        dict(base_model, neigh="edge_index_2hop"),
        dict(base_model, norm="degree_sys"),
        dict(base_model, agg="max"),
    ]

    class Retrieval:
        def __init__(self) -> None:
            self.target_calls = 0

        def retrieve_top_n_best(self, dataset: str, n: int = 1):
            return [(*base_model.values(), 0.9, 0.0)]

        def retrieve_model(self, dataset: str, model):
            return 0.9, 0.0

        def get_all_one_step_modifications(self, dataset: str, current_model):
            return [(model, 0.9 + idx * 0.01, 0.0) for idx, model in enumerate(modifications)]

        def evaluate_model(self, dataset: str, model):
            self.target_calls += 1
            return 0.1 * self.target_calls, 0.0

    retrieval = Retrieval()
    refine_model_with_bayesian(
        unseen_dataset="Target",
        task="node_classification",
        top_s_benchmarks=[("CiteSeer", 1.0)],
        knowledge_retrieval=retrieval,
        knowledge_estimator=None,
        initial_strategy="best",
        max_iter=2,
        window_size=40,
    )

    assert retrieval.target_calls == 3
