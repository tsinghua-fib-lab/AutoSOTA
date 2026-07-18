from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from diffusion_strings.adapters.from_bioemu import (
    _bioemu_pair_field,
    _to_flat_state,
    bioemu_cached_embedding_paths,
    bioemu_vector_fields_from_model,
    prepare_bioemu_embedding_cache,
)


@dataclass
class MockBatch:
    pos: torch.Tensor
    node_orientations: torch.Tensor
    batch: torch.Tensor
    num_graphs: int

    def to(self, device):
        return MockBatch(
            pos=self.pos.to(device),
            node_orientations=self.node_orientations.to(device),
            batch=self.batch.to(device),
            num_graphs=self.num_graphs,
        )

    def replace(self, **kwargs):
        values = {
            "pos": self.pos,
            "node_orientations": self.node_orientations,
            "batch": self.batch,
            "num_graphs": self.num_graphs,
        }
        values.update(kwargs)
        return MockBatch(**values)


@dataclass
class MockScore:
    pos: torch.Tensor
    node_orientations: torch.Tensor


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, batch, t):
        self.calls += 1
        return MockScore(
            pos=torch.full_like(batch.pos, 2.0),
            node_orientations=torch.full(
                (batch.node_orientations.shape[0], 3),
                3.0,
                device=batch.node_orientations.device,
                dtype=batch.node_orientations.dtype,
            ),
        )


class MockPosSDE:
    def marginal_prob(self, x, t, batch_idx):
        return x, torch.full_like(x, 2.0)

    def sde(self, x, t, batch_idx):
        return torch.full_like(x, 0.25), torch.full((x.shape[0],), 2.0)


class MockRotSDE:
    def get_score_scaling(self, t, batch_idx):
        return torch.full((batch_idx.shape[0],), 4.0, device=batch_idx.device)

    def sde(self, x, t, batch_idx):
        return torch.full((x.shape[0], 3), 0.5, device=x.device), torch.full(
            (x.shape[0],), 0.5, device=x.device
        )


def test_bioemu_pair_score_uses_one_model_call_and_restores_shapes():
    model = MockModel()
    pos = torch.zeros(2, 4, 3)
    node_orientations = torch.eye(3).expand(2, 4, 3, 3).clone()
    batch = MockBatch(
        pos=pos.reshape(-1, 3),
        node_orientations=node_orientations.reshape(-1, 3, 3),
        batch=torch.arange(8) // 4,
        num_graphs=2,
    )

    pos_score, rot_score = _bioemu_pair_field(
        pos,
        node_orientations,
        0.2,
        kind="score",
        score_model=model,
        sdes={"pos": MockPosSDE(), "node_orientations": MockRotSDE()},
        marginal_concentration_factor=1.0,
        batch=batch,
    )

    assert model.calls == 1
    assert pos_score.shape == pos.shape
    assert rot_score.shape == (2, 4, 3)
    assert torch.allclose(pos_score, torch.ones_like(pos_score))
    assert torch.allclose(rot_score, torch.full_like(rot_score, 12.0))


def test_bioemu_pair_b_uses_scaled_scores_for_both_components():
    model = MockModel()
    pos = torch.zeros(4, 3)
    node_orientations = torch.eye(3).expand(4, 3, 3).clone()
    batch = MockBatch(
        pos=pos,
        node_orientations=node_orientations,
        batch=torch.zeros(4, dtype=torch.long),
        num_graphs=1,
    )

    b_pos, b_rot = _bioemu_pair_field(
        pos,
        node_orientations,
        torch.tensor(0.2),
        kind="b",
        score_model=model,
        sdes={"pos": MockPosSDE(), "node_orientations": MockRotSDE()},
        marginal_concentration_factor=1.0,
        batch=batch,
    )

    assert model.calls == 1
    assert torch.allclose(b_pos, torch.full_like(pos, 1.75))
    assert torch.allclose(b_rot, torch.full((4, 3), 1.0))


def test_bioemu_state_shapes_must_be_compatible():
    pos = torch.zeros(2, 4, 3)
    node_orientations = torch.eye(3).expand(2, 5, 3, 3).clone()

    try:
        _to_flat_state(pos, node_orientations)
    except ValueError as exc:
        assert "matching leading dimensions" in str(exc)
    else:
        raise AssertionError("expected incompatible shapes to raise ValueError")


def test_bioemu_factory_refuses_uncached_embeddings(tmp_path):
    try:
        bioemu_vector_fields_from_model(
            score_model=MockModel(),
            sequence="GYDPETGTWG",
            sdes={"pos": MockPosSDE(), "node_orientations": MockRotSDE()},
            cache_embeds_dir=tmp_path,
        )
    except FileNotFoundError as exc:
        assert "BioEmu embeddings are not cached" in str(exc)
        assert "allow_embedding_generation=True" in str(exc)
    else:
        raise AssertionError("expected missing BioEmu embeddings to raise")


def test_prepare_bioemu_embedding_cache_imports_explicit_files(tmp_path):
    source_single = tmp_path / "single.npy"
    source_pair = tmp_path / "pair.npy"
    cache_dir = tmp_path / "cache"
    np.save(source_single, np.zeros((10, 4), dtype=np.float32))
    np.save(source_pair, np.zeros((10, 10, 8), dtype=np.float32))

    single_path, pair_path = prepare_bioemu_embedding_cache(
        "GYDPETGTWG",
        cache_embeds_dir=cache_dir,
        single_embeds_file=source_single,
        pair_embeds_file=source_pair,
    )

    expected_single, expected_pair = bioemu_cached_embedding_paths(
        "GYDPETGTWG",
        cache_dir,
    )
    assert single_path == expected_single
    assert pair_path == expected_pair
    assert single_path.exists()
    assert pair_path.exists()


def test_prepare_bioemu_embedding_cache_requires_both_explicit_files(tmp_path):
    source_single = tmp_path / "single.npy"
    np.save(source_single, np.zeros((10, 4), dtype=np.float32))

    try:
        prepare_bioemu_embedding_cache(
            "GYDPETGTWG",
            cache_embeds_dir=tmp_path / "cache",
            single_embeds_file=source_single,
        )
    except ValueError as exc:
        assert "both single_embeds_file and pair_embeds_file" in str(exc)
    else:
        raise AssertionError("expected incomplete explicit embedding paths to raise")
