from __future__ import annotations

from knowledge_retrieval.knowledge_estimation import KnowledgeEstimation


def test_estimator_uses_custom_knowledge_base_root(tmp_path) -> None:
    knowledge_root = tmp_path / "knowledge_base"
    task_root = knowledge_root / "node"
    task_root.mkdir(parents=True)

    estimator = KnowledgeEstimation(
        task="node_classification",
        base_db_path=knowledge_root,
    )

    assert estimator.base_db_path == task_root
    assert estimator.get_dataset_path("Cora") == str(task_root / "cora")


def test_estimator_accepts_task_specific_knowledge_base_root(tmp_path) -> None:
    task_root = tmp_path / "node"
    task_root.mkdir()

    estimator = KnowledgeEstimation(
        task="node_classification",
        base_db_path=task_root,
    )

    assert estimator.base_db_path == task_root
    assert estimator.get_dataset_path("Cora") == str(task_root / "cora")
