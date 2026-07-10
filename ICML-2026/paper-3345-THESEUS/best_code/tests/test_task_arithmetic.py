import torch

from merge_and_rebase.merge import TaskVector, apply_task_vector, compose_task_vectors


def test_task_vector_roundtrip():
    base = {"w": torch.randn(4, 3), "b": torch.randn(3)}
    tuned = {"w": base["w"] + 0.1, "b": base["b"] - 0.2}

    tv = TaskVector.from_checkpoints(base, tuned, strict=True)
    out = apply_task_vector(base, tv, alpha=1.0, strict=True)

    assert torch.allclose(out["w"], tuned["w"])
    assert torch.allclose(out["b"], tuned["b"])


def test_compose_two_vectors():
    base = {"w": torch.zeros(2, 2), "b": torch.zeros(2)}
    t1 = {"w": torch.ones(2, 2), "b": torch.ones(2)}
    t2 = {"w": torch.ones(2, 2) * 2, "b": torch.ones(2) * 3}

    v1 = TaskVector.from_checkpoints(base, t1, strict=True)
    v2 = TaskVector.from_checkpoints(base, t2, strict=True)

    merged = compose_task_vectors(base, [v1, v2], weights=[0.5, 0.25], strict=True)
    assert torch.allclose(merged["w"], torch.ones(2, 2) * (0.5 * 1 + 0.25 * 2))
    assert torch.allclose(merged["b"], torch.ones(2) * (0.5 * 1 + 0.25 * 3))
