import torch
from torch import Tensor

import pytest
from typing import Sequence

from model.embeddings import ComponentEmbedding, DistributionEmbedding, ObservationEmbedding


class TestComponentEmbedding:

    @pytest.mark.parametrize(
        ("state_size", "d_model", "hidden_layer_sizes", "x"),
        [
            (
                2,
                10,
                None,
                torch.tensor([1, 0, 0, 1, 0, 0, 1], dtype=torch.float32),
            ),
            (
                2,
                10,
                [10, 20],
                torch.tensor([1, 0, 0, 1, 0, 0, 1], dtype=torch.float32).broadcast_to(5, 7),
            )
        ]
    )
    def test_embed(self, state_size: int,
                   d_model: int,
                   hidden_layer_sizes: Sequence[int],
                   x: Tensor):
        embedding_function = ComponentEmbedding(state_size=state_size, d_model=d_model,
                                                hidden_layer_sizes=hidden_layer_sizes)
        embedding = embedding_function(x)
        assert embedding.shape == x.shape[:-1] + (d_model,)

    @pytest.mark.parametrize(
        ("state_size", "d_model", "hidden_layer_sizes", "x", "jitter"),
        [
            (
                    2,
                    10,
                    None,
                    torch.ones(10, dtype=torch.float32),
                    0.,
            ),
            (
                    2,
                    10,
                    [10, 10],
                    torch.ones(5, 10),
                    0.1,
            )
        ]
    )
    def test_de_embed(self, state_size: int,
                      d_model: int,
                      hidden_layer_sizes:
                      Sequence[int],
                      x: Tensor, jitter: float):
        batch_size = x.shape[:-1]
        embedding_function = ComponentEmbedding(state_size=state_size, d_model=d_model,
                                                hidden_layer_sizes=hidden_layer_sizes, jitter=jitter)
        decoded = embedding_function(x, reverse=True)
        scale = decoded[..., -state_size**2:].reshape(batch_size + (state_size, state_size))
        assert torch.greater_equal(torch.linalg.det(scale), jitter ** (2 * state_size)).all()
        assert torch.greater_equal(torch.diagonal(scale, dim1=-2, dim2=-1), 0).all()


class TestDistributionEmbedding:

    @pytest.mark.parametrize(
        ("n_components", "d_model", "hidden_layer_sizes", "x"),
        [
            (
                5,
                10,
                None,
                torch.tensor([1, 2, 3], dtype=torch.float32),
            ),
            (
                5,
                10,
                [10, 20],
                torch.tensor([1, 2, 3], dtype=torch.float32).broadcast_to(5, 3),
            )
        ]
    )
    def test_embed(self, n_components: int,
                   d_model: int,
                   hidden_layer_sizes: Sequence[int],
                   x: Tensor):
        embedding_function = DistributionEmbedding(n_params=x.shape[-1],
                                                   n_components=n_components,
                                                   d_model=d_model,
                                                   transform=torch.log,
                                                   embedding_hidden_layer_sizes=hidden_layer_sizes,
                                                   conversion_hidden_layer_sizes=hidden_layer_sizes)
        embedding = embedding_function(x)
        assert embedding.shape == x.shape[:-1] + (n_components, d_model)


class TestObservationEmbedding:

    @pytest.mark.parametrize(
        ("d_model", "hidden_layer_sizes", "z"),
        [
            (
                10,
                None,
                torch.tensor([1, 2, 3], dtype=torch.float32),
            ),
            (
                10,
                [10, 20],
                torch.tensor([1, 2, 3], dtype=torch.float32).broadcast_to(5, 3),
            )
        ]
    )
    def test_embed(self, d_model: int, hidden_layer_sizes: Sequence[int], z: Tensor):
        embedding_function = ObservationEmbedding(observation_size=z.shape[-1],
                                                  d_model=d_model,
                                                  hidden_layer_sizes=hidden_layer_sizes)
        embedding = embedding_function(z)
        assert embedding.shape == z.shape[:-1] + (d_model,)
