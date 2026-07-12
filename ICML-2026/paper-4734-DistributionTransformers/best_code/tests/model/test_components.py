import torch
from torch import Tensor

import pytest
from typing import Sequence

from model.components import MLP


class TestMLP:

    @pytest.mark.parametrize(
        ("layer_sizes", "x"),
        [
            (
                [3, 5, 3],
                torch.ones(3)
            ),
            (
                [3, 5, 3],
                torch.ones(5, 3)
            )
        ]
    )
    def test_forward(self, layer_sizes: Sequence[int], x: Tensor):
        mlp = MLP(layer_sizes)
        assert mlp(x).shape == x.shape[:-1] + (layer_sizes[-1],)

