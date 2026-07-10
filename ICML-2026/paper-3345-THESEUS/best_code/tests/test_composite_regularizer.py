from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from merge_and_rebase.finetune.regularizers.base import BatchOverride, CheckpointArtifact, OptimizerBundle
from merge_and_rebase.finetune.regularizers.composite import CompositeRegularizer


@dataclass
class _Prepared:
    value: float
    override: BatchOverride | None = None
    closed: bool = False
    optimizer_bundles: tuple[OptimizerBundle, ...] = ()
    artifacts: tuple[CheckpointArtifact, ...] = ()

    def close(self) -> None:
        self.closed = True

    def prepare_batch(self, **kwargs) -> BatchOverride | None:
        return self.override

    def checkpoint_artifacts(self, **kwargs):
        return self.artifacts


class _Reg:
    def __init__(self, name: str, prepared: _Prepared) -> None:
        self.name = name
        self.prepared = prepared
        self.finalized = False

    def finalize_model(self, **kwargs):
        self.finalized = True
        return {"patched_blocks": 1}

    def prepare(self, **kwargs):
        return self.prepared, {"count": 1}

    def apply(self, prepared, **kwargs):
        outputs = kwargs["outputs"]
        return outputs.new_tensor(float(prepared.value))


def test_composite_regularizer_sums_children_and_closes(monkeypatch) -> None:
    reg_a = _Reg("a", _Prepared(value=1.5))
    reg_b = _Reg("b", _Prepared(value=2.5))
    monkeypatch.setattr(
        "merge_and_rebase.finetune.regularizers.composite.get_regularizer",
        lambda name: {"a": reg_a, "b": reg_b}[name],
    )

    composite = CompositeRegularizer()
    prepared, info = composite.prepare(
        model=nn.Linear(2, 2),
        device=torch.device("cpu"),
        regularization_cfg={"name": "composite", "regularizers": [{"name": "a"}, {"name": "b"}]},
    )

    loss = composite.apply(
        prepared,
        model=nn.Linear(2, 2),
        step=0,
        batch_index=0,
        outputs=torch.zeros(1),
    )
    assert float(loss.item()) == 4.0
    assert info["composite_children"] == 2
    prepared.close()
    assert reg_a.prepared.closed is True
    assert reg_b.prepared.closed is True


def test_composite_regularizer_rejects_multiple_batch_overrides(monkeypatch) -> None:
    override = BatchOverride(outputs=torch.zeros(1), primary_loss=torch.zeros(()))
    reg_a = _Reg("a", _Prepared(value=0.0, override=override))
    reg_b = _Reg("b", _Prepared(value=0.0, override=override))
    monkeypatch.setattr(
        "merge_and_rebase.finetune.regularizers.composite.get_regularizer",
        lambda name: {"a": reg_a, "b": reg_b}[name],
    )

    composite = CompositeRegularizer()
    prepared, _info = composite.prepare(
        model=nn.Linear(2, 2),
        device=torch.device("cpu"),
        regularization_cfg={"name": "composite", "regularizers": [{"name": "a"}, {"name": "b"}]},
    )

    try:
        prepared.prepare_batch(model=nn.Linear(2, 2))
    except ValueError as exc:
        assert "At most one child regularizer may override a batch" in str(exc)
    else:
        raise AssertionError("expected ValueError for multiple child batch overrides")


def test_composite_regularizer_collects_checkpoint_artifacts(monkeypatch) -> None:
    artifact = CheckpointArtifact(output_dir="tmp", filename="x.pt", payload={"a": 1})
    reg_a = _Reg("a", _Prepared(value=0.0, artifacts=(artifact,)))
    monkeypatch.setattr(
        "merge_and_rebase.finetune.regularizers.composite.get_regularizer",
        lambda name: {"a": reg_a}[name],
    )

    composite = CompositeRegularizer()
    prepared, _info = composite.prepare(
        model=nn.Linear(2, 2),
        device=torch.device("cpu"),
        regularization_cfg={"name": "composite", "regularizers": [{"name": "a"}]},
    )
    artifacts = prepared.checkpoint_artifacts(kind="best_ep")
    assert artifacts == (artifact,)


def test_composite_regularizer_finalizes_children(monkeypatch) -> None:
    reg_a = _Reg("a", _Prepared(value=0.0))
    reg_b = _Reg("b", _Prepared(value=0.0))
    monkeypatch.setattr(
        "merge_and_rebase.finetune.regularizers.composite.get_regularizer",
        lambda name: {"a": reg_a, "b": reg_b}[name],
    )

    composite = CompositeRegularizer()
    info = composite.finalize_model(
        model=nn.Linear(2, 2),
        device=torch.device("cpu"),
        regularization_cfg={"name": "composite", "regularizers": [{"name": "a"}, {"name": "b"}]},
    )

    assert reg_a.finalized is True
    assert reg_b.finalized is True
    assert info["composite_children"] == 2
    assert info["a.patched_blocks"] == 1
    assert info["b.patched_blocks"] == 1
