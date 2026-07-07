from __future__ import annotations

from pathlib import Path

from knowledge_retrieval.candidate_evaluator import CandidateEvaluationConfig, CandidateEvaluator
from knowledge_retrieval.knowledge_retrieval import KnowledgeRetrieval


MODEL = {
    "neigh": "edge_index",
    "norm": "degree_sys",
    "agg": "add",
    "comb": "add",
    "l_mp": "4",
    "stage": "skipsum",
}


def test_database_hit_returns_candidate_performance_without_training() -> None:
    retrieval = KnowledgeRetrieval(task="node_classification", evaluation_mode="database")
    best = retrieval.retrieve_top_n_best("Cora", n=1)[0]
    model = dict(zip(["neigh", "norm", "agg", "comb", "l_mp", "stage"], best[:6]))

    retrieved = retrieval.retrieve_model("Cora", model)
    evaluated = retrieval.evaluate_model("Cora", model)

    assert evaluated == retrieved
    assert evaluated[0] == best[6]


def test_released_artifact_scope_is_ecc_and_model_graph_only() -> None:
    root = retrieval_root = "knowledge_retrieval/knowledge_base"
    assert root == retrieval_root

    pt_files = sorted(Path(root).rglob("*.pt"))
    assert len(pt_files) == 66
    assert {path.name for path in pt_files} == {"ecc_predictor.pt", "model_graph.pt"}


def test_candidate_cache_id_includes_runtime_config() -> None:
    base_config = CandidateEvaluationConfig(
        graphgym_root=Path("GraphGym").resolve(),
        output_root=Path("outputs/candidate_runs").resolve(),
        repeat=3,
        max_epoch=None,
    )
    repeat_config = CandidateEvaluationConfig(
        graphgym_root=Path("GraphGym").resolve(),
        output_root=Path("outputs/candidate_runs").resolve(),
        repeat=5,
        max_epoch=None,
    )
    epoch_config = CandidateEvaluationConfig(
        graphgym_root=Path("GraphGym").resolve(),
        output_root=Path("outputs/candidate_runs").resolve(),
        repeat=3,
        max_epoch=10,
    )

    base_id = CandidateEvaluator("node_classification", base_config)._run_id("Cora", MODEL)

    assert base_id == CandidateEvaluator("node_classification", base_config)._run_id("Cora", MODEL)
    assert base_id != CandidateEvaluator("node_classification", repeat_config)._run_id("Cora", MODEL)
    assert base_id != CandidateEvaluator("node_classification", epoch_config)._run_id("Cora", MODEL)
