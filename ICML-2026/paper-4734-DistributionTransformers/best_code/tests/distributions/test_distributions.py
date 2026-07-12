import pytest
import torch
from torch.types import _size

from distributions.distributions import (GaussianMixtureModel, GaussianMixtureModelConjugateMetaPrior,
                                         DirectGaussianObservationModel, MappedGaussianObservationModel,
                                         LinearGaussianObservationModel, CompleteDistribution, ObservationModel,
                                         MetaPrior)


class TestGaussianMixtureModel:

    @pytest.mark.parametrize(
        ("weights", "loc", "scale_parametrisation", "scale", "sample_shape", "expected_result"),
        [
            (
                    torch.ones(3) / 3,
                    torch.zeros(3, 2, dtype=torch.float32),
                    "covariance_matrix",
                    torch.eye(2, dtype=torch.float32).broadcast_to(3, 2, 2),
                    (1,),
                    (1, 2)
            ),
            (
                    torch.ones(5) / 5,
                    torch.zeros(5, 6, dtype=torch.float32),
                    "precision_matrix",
                    torch.eye(6, dtype=torch.float32).broadcast_to(5, 6, 6),
                    (3, 4),
                    (3, 4, 6)
            ),
            (
                    torch.ones(1),
                    torch.zeros(1, 1, dtype=torch.float32),
                    "scale_tril",
                    torch.eye(1, dtype=torch.float32).broadcast_to(1, 1, 1),
                    (1, 1),
                    (1, 1, 1)
            )
        ]
    )
    def test_sample(self, weights: torch.Tensor, loc: torch.Tensor, scale_parametrisation: str, scale: torch.Tensor,
                    sample_shape: _size, expected_result: _size):
        gmm = GaussianMixtureModel(weights, loc, **{scale_parametrisation: scale})
        sample = gmm.sample(sample_shape)
        assert sample.shape == expected_result


class TestGaussianMixtureModelConjugatePrior:

    @pytest.mark.parametrize(
        ("n_components", "state_size"),
        [
            (
                4,
                3
            ),
            (
                1,
                1
            )
        ]
    )
    def test_defaults(self, n_components: int, state_size: int):
        gmm_conjugate_meta_prior = GaussianMixtureModelConjugateMetaPrior(n_components=n_components,
                                                                          state_size=state_size)
        assert torch.equal(gmm_conjugate_meta_prior.weights_concentration, torch.ones(n_components) / n_components)
        assert torch.equal(gmm_conjugate_meta_prior.loc_loc, torch.zeros((n_components, state_size)))
        assert torch.equal(gmm_conjugate_meta_prior.loc_covariance_matrix,
                           torch.eye(state_size, dtype=torch.float32).broadcast_to(n_components,
                                                                                   state_size,
                                                                                   state_size))
        assert gmm_conjugate_meta_prior.loc_precision_matrix is None
        assert gmm_conjugate_meta_prior.loc_scale_tril is None
        assert gmm_conjugate_meta_prior.scale_parametrisation == "covariance_matrix"
        assert gmm_conjugate_meta_prior.scale_df == state_size+1
        assert torch.equal(gmm_conjugate_meta_prior.scale_covariance_matrix,
                           torch.eye(state_size, dtype=torch.float32).broadcast_to(n_components,
                                                                                   state_size, state_size))
        assert gmm_conjugate_meta_prior.scale_precision_matrix is None
        assert gmm_conjugate_meta_prior.scale_scale_tril is None
        assert gmm_conjugate_meta_prior.scale_eps == 1e-6
        assert gmm_conjugate_meta_prior.prior_size == n_components * (1 + state_size + state_size ** 2)

    @pytest.mark.parametrize(
        ("weights_concentration", "loc_loc", "loc_scale_parametrisation", "loc_scale", "scale_parametrisation",
         "scale_df", "scale_scale_parametrisation", "scale_scale", "scale_eps", "n_components", "state_size",
         "sample_shape", "expected_result"),
        [
            (
                None,
                None,
                "covariance_matrix",
                None,
                "covariance_matrix",
                None,
                "covariance_matrix",
                None,
                None,
                4,
                2,
                (3, 4),
                (3, 4, 4, 7)
            ),
            (
                torch.ones(3) / 3,
                torch.zeros((3, 3), dtype=torch.float32),
                "precision_matrix",
                torch.eye(3, 3, dtype=torch.float32).broadcast_to(3, 3, 3),
                "precision_matrix",
                4,
                "precision_matrix",
                torch.eye(3, 3, dtype=torch.float32).broadcast_to(3, 3, 3),
                10,
                10,
                10,
                (3, 4),
                (3, 4, 3, 13)
            ),
            (
                torch.ones(5) / 5,
                torch.zeros((5, 3), dtype=torch.float32),
                "scale_tril",
                torch.eye(3, 3, dtype=torch.float32).broadcast_to(5, 3, 3),
                "scale_tril",
                4,
                "scale_tril",
                torch.eye(3, 3, dtype=torch.float32).broadcast_to(5, 3, 3),
                1,
                10,
                10,
                (3, 4),
                (3, 4, 5, 13)
            ),
        ]
    )
    def test_sample(self, weights_concentration: torch.Tensor, loc_loc: torch.Tensor, loc_scale_parametrisation: str,
                    loc_scale: torch.Tensor, scale_parametrisation: str, scale_df: torch.Tensor,
                    scale_scale_parametrisation: str, scale_scale: torch.Tensor, scale_eps: float, n_components: int,
                    state_size: int, sample_shape: _size, expected_result: _size):
        gmm_conjugate_meta_prior = GaussianMixtureModelConjugateMetaPrior(weights_concentration=weights_concentration,
                                                                          loc_loc=loc_loc,
                                                                          **{"loc_"+loc_scale_parametrisation:
                                                                             loc_scale},
                                                                          scale_parametrisation=scale_parametrisation,
                                                                          scale_df=scale_df,
                                                                          **{"scale_"+scale_scale_parametrisation:
                                                                             scale_scale},
                                                                          scale_eps=scale_eps,
                                                                          n_components=n_components,
                                                                          state_size=state_size
                                                                          )
        assert gmm_conjugate_meta_prior.n_components == gmm_conjugate_meta_prior.weights_concentration.shape[-1]
        assert gmm_conjugate_meta_prior.state_size == gmm_conjugate_meta_prior.loc_loc.shape[-1]
        sample = gmm_conjugate_meta_prior.sample(sample_shape)
        assert sample.shape == expected_result
        decoded_sample = gmm_conjugate_meta_prior.decode_sample(sample)
        assert scale_parametrisation in decoded_sample
        assert torch.equal(gmm_conjugate_meta_prior.encode_sample(decoded_sample), sample)
        if scale_parametrisation == "covariance_matrix":
            assert torch.less_equal(torch.linalg.det(decoded_sample["covariance_matrix"]),
                                    gmm_conjugate_meta_prior.scale_eps ** -gmm_conjugate_meta_prior.state_size).all()
        elif scale_parametrisation == "precision_matrix":
            assert torch.greater_equal(torch.linalg.det(decoded_sample["precision_matrix"]),
                                       gmm_conjugate_meta_prior.scale_eps**gmm_conjugate_meta_prior.state_size).all()
        else:
            assert torch.less_equal(torch.linalg.det(decoded_sample["scale_tril"]),
                                    gmm_conjugate_meta_prior.scale_eps ** -0.5 * gmm_conjugate_meta_prior.state_size
                                    ).all()
        assert not torch.isnan(gmm_conjugate_meta_prior.log_prob(sample)).any()


class TestDirectGaussianObservationModel:

    @pytest.mark.parametrize(
        ("x", "scale", "scale_parametrisation"),
        [
            (
                torch.ones(2),
                torch.eye(2),
                "covariance_matrix"
            ),
            (
                torch.ones((5, 2)),
                torch.eye(2),
                "precision_matrix"
            ),
            (
                torch.ones((5, 4, 2)),
                torch.eye(2).broadcast_to(5, 4, 2, 2),
                "scale_tril"
            )
        ]
    )
    def test_condition(self, x: torch.Tensor, scale: torch.Tensor, scale_parametrisation: str):
        dgom = DirectGaussianObservationModel(**{scale_parametrisation: scale})
        dgom.condition_(x)
        assert torch.equal(dgom.distribution.loc, x)

    @pytest.mark.parametrize(
        ("x", "scale", "scale_parametrisation", "sample_shape", "expected_shape"),
        [
            (
                torch.ones(2),
                torch.eye(2),
                "covariance_matrix",
                (1,),
                (1., 2.)
            ),
            (
                torch.ones((5, 2)),
                torch.eye(2),
                "precision_matrix",
                (10, 20),
                (10, 20, 5, 2)
            ),
            (
                torch.ones((5, 4, 2)),
                torch.eye(2).broadcast_to(5, 4, 2, 2),
                "scale_tril",
                torch.Size(),
                (5, 4, 2)
            )
        ]
    )
    def test_sample(self, x: torch.Tensor, scale: torch.Tensor, scale_parametrisation: str, sample_shape: _size,
                    expected_shape: _size):
        dgom = DirectGaussianObservationModel(**{scale_parametrisation: scale})
        assert dgom.n_observations == x.shape[-1]
        dgom.condition_(x)
        sample = dgom.sample(sample_shape)
        assert sample.shape == expected_shape


class TestMappedGaussianObservationModel:

    @pytest.mark.parametrize(
        ("x", "scale", "scale_parametrisation", "mapping"),
        [
            (
                torch.ones(2),
                torch.eye(2),
                "covariance_matrix",
                torch.nn.Identity()
            ),
            (
                torch.ones((5, 2)),
                torch.eye(2),
                "precision_matrix",
                None
            ),
            (
                torch.ones((5, 4, 2)),
                torch.eye(2).broadcast_to(5, 4, 2, 2),
                "scale_tril",
                lambda x: x ** 2
            )
        ]
    )
    def test_condition(self, x: torch.Tensor, scale: torch.Tensor, scale_parametrisation: str, mapping: callable):
        dgom = MappedGaussianObservationModel(**{scale_parametrisation: scale}, mapping=mapping)
        dgom.condition_(x)
        assert torch.equal(dgom.distribution.loc, x)

    @pytest.mark.parametrize(
        ("x", "scale", "scale_parametrisation", "mapping", "sample_shape", "expected_shape"),
        [
            (
                torch.ones(2),
                torch.eye(2),
                "covariance_matrix",
                torch.nn.Identity(),
                (1,),
                (1., 2.)
            ),
            (
                torch.ones((5, 2)),
                torch.eye(2),
                "precision_matrix",
                None,
                (10, 20),
                (10, 20, 5, 2)
            ),
            (
                torch.ones((5, 4, 2)),
                torch.eye(2).broadcast_to(5, 4, 2, 2),
                "scale_tril",
                lambda x: x ** 2,
                torch.Size(),
                (5, 4, 2)
            )
        ]
    )
    def test_sample(self, x: torch.Tensor, scale: torch.Tensor, scale_parametrisation: str, mapping: callable,
                    sample_shape: _size, expected_shape: _size):
        mgom = MappedGaussianObservationModel(**{scale_parametrisation: scale}, mapping=mapping)
        assert mgom.n_observations == x.shape[-1]
        mgom.condition_(x)
        sample = mgom.sample(sample_shape)
        assert sample.shape == expected_shape


class TestLinearGaussianObservationModel:

    @pytest.mark.parametrize(
        ("x", "scale", "scale_parametrisation", "observation_matrix"),
        [
            (
                torch.ones(2),
                torch.eye(2),
                "covariance_matrix",
                torch.eye(2)
            ),
            (
                torch.ones((5, 2)),
                torch.eye(2),
                "precision_matrix",
                torch.eye(2)
            ),
            (
                torch.ones((5, 4, 2)),
                torch.eye(2).broadcast_to(5, 4, 2, 2),
                "scale_tril",
                torch.eye(2)
            )
        ]
    )
    def test_condition(self, x: torch.Tensor, scale: torch.Tensor, scale_parametrisation: str,
                       observation_matrix: torch.Tensor):
        dgom = LinearGaussianObservationModel(**{scale_parametrisation: scale}, observation_matrix=observation_matrix)
        dgom.condition_(x)
        assert torch.equal(dgom.distribution.loc, x)

    @pytest.mark.parametrize(
        ("x", "scale", "scale_parametrisation", "observation_matrix", "sample_shape", "expected_shape"),
        [
            (
                torch.ones(2),
                torch.eye(3),
                "covariance_matrix",
                torch.ones((3, 2)),
                (1,),
                (1., 3.)
            ),
            (
                torch.ones((5, 2)),
                torch.eye(2),
                "precision_matrix",
                torch.eye(2),
                (10, 20),
                (10, 20, 5, 2)
            ),
            (
                torch.ones((5, 4, 2)),
                torch.eye(2).broadcast_to(5, 4, 2, 2),
                "scale_tril",
                torch.eye(2),
                torch.Size(),
                (5, 4, 2)
            )
        ]
    )
    def test_sample(self, x: torch.Tensor, scale: torch.Tensor, scale_parametrisation: str,
                    observation_matrix: torch.Tensor, sample_shape: _size, expected_shape: _size):
        mgom = LinearGaussianObservationModel(**{scale_parametrisation: scale}, observation_matrix=observation_matrix)
        assert mgom.n_observations == observation_matrix.shape[-2]
        mgom.condition_(x)
        sample = mgom.sample(sample_shape)
        assert sample.shape == expected_shape


@pytest.fixture
def meta_prior():
    return GaussianMixtureModelConjugateMetaPrior(n_components=3, state_size=2)


@pytest.fixture
def observation_model():
    return {"obs": DirectGaussianObservationModel(covariance_matrix=torch.eye(2, dtype=torch.float32))}


class TestCompleteDistribution:

    @pytest.mark.parametrize(
        ("sample_shape", "expected_phi_shape", "expected_x_shape", "expected_z_shape"),
        [
            (
                torch.Size(),
                (3, 7),
                (2,),
                {"obs": (2,)}
            ),
            (
                (5, 10),
                (5, 10, 3, 7),
                (5, 10, 2),
                {"obs": (5, 10, 2)}
            )

        ]
    )
    def test_sample(self, meta_prior: MetaPrior, observation_model: dict[str, ObservationModel], sample_shape: _size,
                    expected_phi_shape: _size, expected_x_shape: _size, expected_z_shape: dict[str, _size]):
        complete_distribution = CompleteDistribution(meta_prior, **observation_model)
        phi, x, z = complete_distribution.sample(sample_shape)
        assert phi.shape == expected_phi_shape
        assert x.shape == expected_x_shape
        assert all([z[key].shape == expected_z_shape[key] for key in z])
