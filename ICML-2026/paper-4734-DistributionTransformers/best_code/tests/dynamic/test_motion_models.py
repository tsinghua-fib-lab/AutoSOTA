import torch
from torch import Tensor
from torch.types import _size
from torch.distributions import Distribution, MultivariateNormal

import pytest
from typing import Callable

from dynamic.motion_models import MotionModel, LTIMotionModel


class TestMotionModel:

    @pytest.mark.parametrize(
        ("state_size", "f", "x0_distribution", "noise_distribution", "sample_shape"),
        [
            (
                1,
                lambda x, n, k: x + torch.einsum("ij, ...j -> ...i", torch.ones(1, 1), n),
                MultivariateNormal(torch.zeros(1), torch.eye(1)),
                MultivariateNormal(torch.zeros(1), torch.eye(1)),
                (10,)
            ),
            (
                2,
                lambda x, n, k: x + torch.einsum("ij, ...j -> ...i", torch.ones(2, 3), n),
                MultivariateNormal(torch.zeros(2), torch.eye(2)),
                MultivariateNormal(torch.zeros(3), torch.eye(3)),
                (10, 5,)
            )
        ]
    )
    def test_sample(self, state_size: int,
                    f: Callable[[Tensor, Tensor, int], Tensor],
                    x0_distribution: Distribution,
                    noise_distribution: Distribution,
                    sample_shape: _size):
        motion_model = MotionModel(state_size, f, x0_distribution, noise_distribution)
        sample = motion_model.sample(sample_shape)
        assert sample.shape == sample_shape + (state_size,)


class TestLTIMotionModel:

    @pytest.mark.parametrize(
        ("state_transition_matrix", "process_noise_scale_cholesky", "x0_distribution", "sample_shape"),
        [
            (
                torch.ones(1, 1),
                torch.eye(1),
                MultivariateNormal(torch.zeros(1), torch.eye(1)),
                (10,)
            ),
            (
                torch.ones(2, 2),
                torch.ones(2, 3),
                MultivariateNormal(torch.zeros(2), torch.eye(2)),
                (10, 5,)
            )
        ]
    )
    def test_sample(self, state_transition_matrix: Tensor,
                    process_noise_scale_cholesky: Tensor,
                    x0_distribution: MultivariateNormal,
                    sample_shape: _size):
        motion_model = LTIMotionModel(state_transition_matrix, process_noise_scale_cholesky, x0_distribution)
        sample = motion_model.sample(sample_shape)
        assert sample.shape == sample_shape + state_transition_matrix.shape[-1:]
