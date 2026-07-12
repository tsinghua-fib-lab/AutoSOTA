import torch
from torch import Tensor
from torch.distributions import Distribution, Normal, InverseGamma, MultivariateNormal, kl_divergence

import pytest
from typing import Optional, Callable

from distributions.utils import encode_gmm_sample, decode_gmm_sample
from distributions.utils import kl_divergence as utils_kl_divergence


@pytest.mark.parametrize(
    ("p", "q", "q_transform", "expected_result"),
    [
        (
            Normal(loc=0,
                   scale=1),
            Normal(loc=1,
                   scale=2),
            None,
            kl_divergence(Normal(loc=0,
                                 scale=1),
                          Normal(loc=1,
                                 scale=2))
        ),
        (
            MultivariateNormal(loc=torch.zeros(2),
                               covariance_matrix=torch.eye(2)),
            MultivariateNormal(loc=torch.ones(2),
                               covariance_matrix=2*torch.eye(2)),
            None,
            kl_divergence(MultivariateNormal(loc=torch.zeros(2),
                                             covariance_matrix=torch.eye(2)),
                          MultivariateNormal(loc=torch.ones(2),
                                             covariance_matrix=2*torch.eye(2)))
        ),
        (
            MultivariateNormal(loc=torch.zeros(2).broadcast_to(10, 2),
                               covariance_matrix=torch.eye(2).broadcast_to(10, 2, 2)),
            MultivariateNormal(loc=torch.ones(2).broadcast_to(10, 2),
                               covariance_matrix=2*torch.eye(2).broadcast_to(10, 2, 2)),
            None,
            kl_divergence(MultivariateNormal(loc=torch.zeros(2),
                                             covariance_matrix=torch.eye(2)),
                          MultivariateNormal(loc=torch.ones(2),
                                             covariance_matrix=2*torch.eye(2))).broadcast_to(10)
        ),
        (
            InverseGamma(1, 1),
            Normal(0, 1),
            torch.log,
            torch.tensor(0.33)
        )
    ]
)
def test_kl_divergence(p: Distribution, q: Distribution, q_transform: Optional[Callable[[Tensor], Tensor]],
                       expected_result: Tensor):
    div = utils_kl_divergence(p, q, q_transform)
    assert torch.allclose(div, expected_result, atol=0.01)


@pytest.mark.parametrize(
    ("sample", "expected_result"),
    [
        (
            torch.ones(5, 7),
            {
                "weights": torch.ones(5),
                "loc": torch.ones(5, 2),
                "covariance_matrix": torch.ones(5, 2, 2)
            }
        )
    ]
)
def test_decode_gmm_sample(sample: Tensor, expected_result: dict[str, Tensor]):
    assert all([torch.equal(param, expected)
                for param, expected in zip(decode_gmm_sample(sample).values(), expected_result.values())])


@pytest.mark.parametrize(
    ("sample", "expected_result"),
    [
        (
            {
                "weights": torch.ones(5),
                "loc": torch.ones(5, 2),
                "covariance_matrix": torch.ones(5, 2, 2)
            },
            torch.ones(5, 7),
        )
    ]
)
def test_encode_gmm_sample(sample: dict[str, Tensor], expected_result: Tensor):
    print(encode_gmm_sample(sample))
    assert torch.equal(encode_gmm_sample(sample), expected_result)
