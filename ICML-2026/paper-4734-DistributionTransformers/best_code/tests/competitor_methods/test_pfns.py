import torch
from torch import Tensor
from torch.types import _size
from torch.distributions import Distribution, Normal, MultivariateNormal, Beta, Exponential

import pytest
from typing import Union

from competitor_methods.pfns import RiemannDistribution, get_borders_from_prior


class TestRiemannDistribution:

    @pytest.mark.parametrize(
        ("weights", "borders", "infinite_support", "value", "expected_return"),
        [
            (
                torch.tensor([1., 1., 1.]),
                torch.tensor([-1., 0., 1., 2.]),
                True,
                torch.tensor(0.5),
                torch.tensor(-1.0986)
            ),
            (
                torch.tensor([1., 1., 1.]),
                torch.tensor([-1., 0., 1., 2.]),
                True,
                torch.tensor([-0.5, 0.5, 1.5]),
                torch.tensor([-1.7755, -1.0986, -1.7755])
            ),
            (
                torch.tensor([[1., 1., 1.]] * 3),
                torch.tensor([[-1., 0., 1., 2.]] * 3),
                True,
                torch.tensor([-0.5, 0.5, 1.5]),
                torch.tensor([-1.7755, -1.0986, -1.7755])
            ),
            (
                torch.tensor([1., 2., 3.]),
                torch.tensor([0., 1.4, 1.6, 3.]),
                False,
                torch.tensor([-0.5, 0.5, 1.5, 2.5, 3.5]),
                torch.tensor([-3.4028e38, -2.1282, 0.5108, -1.0296, -3.4028e38])
            ),
            (
                torch.tensor([[1., 1., 1.]] * 2),
                torch.tensor([[0., 1., 2., 3.]] * 2),
                False,
                torch.tensor([[[1.5, 2.5]] * 3] * 4),
                -1.0986 * torch.ones(4, 3, 2)
            ),
            (
                torch.tensor([[1., 1., 1.]] * 3),
                torch.tensor([[-1., 0., 1., 2.]] * 3),
                (True, False),
                torch.tensor([-0.5, 0.5, 1.5]),
                torch.tensor([-1.7755, -1.0986, -1.0986])
            )
        ]
    )
    def test_log_prob(self, weights: Tensor,
                      borders: Tensor,
                      infinite_support: bool,
                      value: Tensor,
                      expected_return: Tensor):
        p = RiemannDistribution(weights, borders, infinite_support)
        assert torch.allclose(p.log_prob(value), expected_return, rtol=0.01)

    @pytest.mark.parametrize(
        ("weights", "borders", "infinite_support", "sample_shape", "expected_shape"),
        [
            (
                torch.tensor([1., 1., 1.]),
                torch.tensor([-1., 0., 1., 2.]),
                True,
                torch.Size(),
                torch.Size()
            ),
            (
                torch.tensor([[1., 1., 1.]] * 2),
                torch.tensor([[-1., 0., 1., 2.]] * 2),
                (True, False),
                (10,),
                (10, 2)
            ),
            (
                torch.tensor([[1., 1., 1.]] * 2),
                torch.tensor([[-1., 0., 1., 2.]] * 2),
                (False, False),
                (10,),
                (10, 2)
            ),

        ]
    )
    def test_sample(self, weights: Tensor,
                    borders: Tensor,
                    infinite_support: Union[bool, tuple[bool, bool]],
                    sample_shape: _size,
                    expected_shape: _size):
        p = RiemannDistribution(weights, borders, infinite_support)
        sample = p.sample(sample_shape)
        assert sample.shape == expected_shape

    @pytest.mark.parametrize(
        ("weights", "borders", "infinite_support", "percentile", "expected_result"),
        [
            (
                    torch.tensor([1., 1., 1.]),
                    torch.tensor([-1., 0., 1., 2.]),
                    True,
                    0.99,
                    torch.tensor([-1.970, 2.970])
            ),
            (
                    torch.tensor([[1., 1., 1.]] * 2),
                    torch.tensor([[-1., 0., 1., 2.]] * 2),
                    (True, False),
                    0.99,
                    torch.tensor([[-1.970, 1.985]] * 2)
            ),
            (
                    torch.tensor([[1., 1., 1.]] * 2),
                    torch.tensor([[-1., 0., 1., 2.]] * 2),
                    (False, False),
                    0.99,
                    torch.tensor([[-0.985, 1.985]] * 2)
            ),

        ]
    )
    def test_conf(self, weights: Tensor,
                  borders: Tensor,
                  infinite_support: Union[bool, tuple[bool, bool]],
                  percentile: float,
                  expected_result: Tensor):
        p = RiemannDistribution(weights, borders, infinite_support)
        conf = p.conf(percentile)
        assert conf.shape == expected_result.shape
        assert torch.allclose(conf, expected_result)


@pytest.mark.parametrize(
    ("prior", "n_buckets", "infinite_support", "leftmost_border", "rightmost_border", "expected_result"),
    [
        (
            Normal(0, 1),
            10,
            True,
            -4,
            4.,
            torch.hstack([torch.tensor(-4.), Normal(0, 1).icdf(torch.linspace(0.1, 0.9, 9)),
                          torch.tensor(4.),])
        ),
        (
            MultivariateNormal(torch.tensor([0.]), torch.tensor([[1.]])),
            10,
            True,
            -4.,
            4.,
            torch.hstack([torch.tensor(-4.), Normal(0, 1).icdf(torch.linspace(0.1, 0.9, 9)),
                          torch.tensor(4.), ])
        ),
        (
            Beta(torch.ones(5, 3), torch.ones(5, 3)),
            10,
            False,
            0.,
            1.,
            torch.linspace(0, 1, 11).broadcast_to(5, 3, 11)
        ),
        (
            Exponential(1),
            10,
            (False, True),
            0.,
            10.,
            torch.hstack([torch.tensor(0.), Exponential(1).icdf(torch.linspace(0.1, 0.9, 9)),
                          torch.tensor(10.)])
        )

    ]
)
def test_get_borders_from_prior(prior: Distribution,
                                n_buckets: int,
                                infinite_support: Union[bool, tuple[bool, bool]],
                                leftmost_border: float,
                                rightmost_border: float,
                                expected_result: Tensor):
    assert torch.allclose(get_borders_from_prior(prior, n_buckets, infinite_support, leftmost_border, rightmost_border),
                          expected_result, atol=0.1)
