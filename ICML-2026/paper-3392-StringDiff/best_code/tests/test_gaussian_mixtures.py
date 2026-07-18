import pytest

torch = pytest.importorskip("torch")

from diffusion_strings.gaussian_mixtures import (
    GMM_itp_rugged,
    MultivariateGaussianMixture_Analytical,
)


def test_single_gaussian_log_prob_and_score_at_data_time():
    weights = torch.tensor([1.0], dtype=torch.float64)
    locs = torch.tensor([[1.0, -2.0]], dtype=torch.float64)
    vars = torch.tensor([[[4.0, 0.0], [0.0, 9.0]]], dtype=torch.float64)
    model = MultivariateGaussianMixture_Analytical(weights, locs, vars, tol=0.0)
    x = torch.tensor([[1.0, -2.0], [3.0, 1.0]], dtype=torch.float64)

    dist = torch.distributions.MultivariateNormal(locs[0], covariance_matrix=vars[0])
    expected_score = -torch.einsum("ij,bj->bi", torch.linalg.inv(vars[0]), x - locs[0])

    assert torch.allclose(model.log_prob(x, torch.tensor(1.0)), dist.log_prob(x), atol=1e-10)
    assert torch.allclose(model.score(x, torch.tensor(1.0)), expected_score, atol=1e-10)


def test_rugged_gmm_prepares_vectorized_functions_and_drift_runs():
    weights = torch.tensor([1.0], dtype=torch.float64)
    locs = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    vars = torch.eye(2, dtype=torch.float64).unsqueeze(0)
    model = GMM_itp_rugged(weights, locs, vars, k=torch.tensor(2.0), r=0.1)
    x = torch.tensor([[0.3, -0.2]], dtype=torch.float64)

    assert hasattr(model, "prob")
    assert model.b(x, torch.tensor(0.5)).shape == x.shape
