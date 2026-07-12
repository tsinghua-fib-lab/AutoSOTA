import torch
from torch import Tensor
from torch.types import _size

from typing import Optional
import pytest

from model.distribution_transformer import DistributionTransformer, TransformerKwargs
from model.embeddings import DistributionEmbedding, ComponentEmbedding, ObservationEmbedding, GammaEmbedding


@pytest.fixture
def conditional_transformer_kwargs():
    return TransformerKwargs(n_head=8)


class TestDistributionTransformer:

    @pytest.mark.parametrize(
        ("component_embedding", "prior_embedding", "n_components", "observation_embeddings", "phi", "z",
         "expected_shape"),
        [
            (
                ComponentEmbedding(1, 64),
                GammaEmbedding(64, 10),
                10,
                {"obs_1": ObservationEmbedding(3, 64)},
                torch.ones(2),
                {"obs_1": torch.ones(3)},
                (10, 3)
             ),
            (
                ComponentEmbedding(1, 64),
                GammaEmbedding(64, 10),
                10,
                {"obs_1": ObservationEmbedding(3, 64)},
                torch.ones(5, 2),
                {"obs_1": torch.ones(5, 3)},
                (5, 10, 3)
            ),
            (
                ComponentEmbedding(1, 64),
                None,
                10,
                {"obs_1": ObservationEmbedding(3, 64)},
                torch.ones(5, 10, 3),
                {"obs_1": torch.ones(5, 3)},
                (5, 10, 3)
            )
        ]

    )
    def test_forward(self,
                     component_embedding: ComponentEmbedding,
                     conditional_transformer_kwargs: TransformerKwargs,
                     n_components: int,
                     prior_embedding: Optional[DistributionEmbedding],
                     observation_embeddings: dict[str, ObservationEmbedding],
                     phi: Tensor,
                     z: dict[str, Tensor],
                     expected_shape: _size):
        distribution_transformer = DistributionTransformer(component_embedding=component_embedding,
                                                           n_components=n_components,
                                                           transformer_kwargs=conditional_transformer_kwargs,
                                                           prior_embedding=prior_embedding,
                                                           **observation_embeddings)
        phi_in, phi_out = distribution_transformer(phi, **z)
        assert phi_in.shape == expected_shape
        assert phi_out.shape == expected_shape
