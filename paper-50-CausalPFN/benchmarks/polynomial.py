# A synthetic polynomial dataset for evaluating CATE estimation methods.

from typing import Callable, List, Tuple

import numpy as np

from .base import ATE_Dataset, CATE_Dataset, EvalDatasetCatalog

SamplerType = Callable[[tuple], np.ndarray]


class LaplaceSampler:
    """Samples noise from a laplace distribution upon calling"""

    def __init__(self, loc: float, scale: float):
        self.loc = loc
        self.scale = scale

    def __call__(self, shape: tuple | None = None) -> np.ndarray | float:
        if shape is None:
            return np.random.laplace(self.loc, self.scale)
        else:
            return np.random.laplace(self.loc, self.scale, shape)


class UniformSampler:
    """Samples noise from a uniform distribution upon calling"""

    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high

    def __call__(self, shape: tuple | None = None) -> np.ndarray | float:
        if shape is None:
            ret = np.random.rand()
        else:
            ret = np.random.rand(*shape)
        return ret * (self.high - self.low) + self.low


class GaussianSampler:
    """Samples noise from a gaussian distribution upon calling"""

    def __init__(self, loc: float, scale: float):
        self.loc = loc
        self.scale = scale

    def __call__(self, shape: tuple | None = None) -> np.ndarray | float:
        if shape is None:
            return np.random.normal(self.loc, self.scale)
        else:
            return np.random.normal(self.loc, self.scale, shape)


class UniformIntegerSampler:
    """Samples noise from a uniform integer distribution upon calling"""

    def __init__(self, low: int, high: int):
        self.low = low
        self.high = high

    def __call__(self, shape: tuple | None = None) -> np.ndarray | int:
        if shape is None:
            ret = np.random.randint(self.low, self.high + 1)
        else:
            ret = np.random.randint(self.low, self.high + 1, shape)
        return ret


class PolynomialDataset(EvalDatasetCatalog):

    def __init__(
        self,
        n_tables: int = 50,
        n_samples: int = 2048,
        test_ratio: float = 0.2,
        x_dim_dist: Callable[[], int] = UniformIntegerSampler(5, 10),
        noise_samplers: List[SamplerType] | SamplerType = [
            GaussianSampler(0.0, 1.0),
            UniformSampler(-1.0, 1.0),
            LaplaceSampler(0, 1.0),
        ],
        weight_sampler: List[SamplerType] | SamplerType = UniformSampler(-5.0, 5.0),
        covariate_sampler: List[SamplerType] | SamplerType = UniformSampler(-2.0, 2.0),
        degree_sampler: List[SamplerType] | SamplerType = UniformIntegerSampler(2, 4),
        standardize_treatment: bool = True,
        standardize_outcome: bool = True,
        standardize_covariates: bool = True,
        seed: int = 42,
    ) -> None:
        self.x_dim_dist = x_dim_dist

        if isinstance(noise_samplers, list):
            self.noise_samplers = noise_samplers
        else:
            self.noise_samplers = [noise_samplers]

        # for linear weights
        if isinstance(weight_sampler, list):
            self.weight_sampler = weight_sampler
        else:
            self.weight_sampler = [weight_sampler]

        # for the covariates
        if isinstance(covariate_sampler, list):
            self.covariate_sampler = covariate_sampler
        else:
            self.covariate_sampler = [covariate_sampler]

        # for the degree
        if isinstance(degree_sampler, list):
            self.degree_sampler = degree_sampler
        else:
            self.degree_sampler = [degree_sampler]

        self.n_samples = n_samples
        self.standardize_treatment = standardize_treatment
        self.standardize_outcome = standardize_outcome
        self.standardize_covariates = standardize_covariates

        self.n_tables = n_tables

        self.test_ratio = test_ratio

        self.seeds = [seed + i for i in range(self.n_tables)]

        super().__init__(n_tables, name="Polynomial")

    def sample_exogenous_noise(self, shape) -> np.ndarray:
        chosen_exogenous_sampler = self.noise_samplers[np.random.randint(len(self.noise_samplers))]
        return chosen_exogenous_sampler(shape)

    def sample_weights(self, shape) -> np.ndarray:
        chosen_weight_sampler = self.weight_sampler[np.random.randint(len(self.weight_sampler))]
        return chosen_weight_sampler(shape)

    def sample_covariates(self, shape) -> np.ndarray:
        chosen_covariate_sampler = self.covariate_sampler[np.random.randint(len(self.covariate_sampler))]
        return chosen_covariate_sampler(shape)

    def sample_degree(self) -> int:
        chosen_degree_sampler = self.degree_sampler[np.random.randint(len(self.degree_sampler))]
        return chosen_degree_sampler((1,))[0]

    def __len__(self):
        return self.n_tables

    def get_covariates_T_Y0_Y1_E_Y0_E_Y1_outcomes(self):

        n_dims = self.x_dim_dist()
        covariates = self.sample_covariates((self.n_samples, n_dims))
        degree = self.sample_degree()
        covariate_features = np.concatenate([covariates**i for i in range(1, degree + 1)], axis=1)
        features_dims = covariate_features.shape[1]

        if self.standardize_covariates:
            covariates = (covariates - covariates.mean(axis=0)) / (covariates.std(axis=0) + 1e-20)

        w_T = self.sample_weights((features_dims,))
        treatment_logits = np.einsum("np,p->n", covariate_features, w_T) + self.sample_exogenous_noise(
            (self.n_samples,)
        )
        if self.standardize_treatment:
            treatment_logits = (treatment_logits - treatment_logits.mean()) / (treatment_logits.std() + 1e-20)

        # Sigmoid function for treatment probabilities
        treatment_probs = 1 / (1 + np.exp(-treatment_logits))

        # Bernoulli sampling for treatments
        treatments = np.random.binomial(1, treatment_probs, size=self.n_samples)

        w_Y0 = self.sample_weights((features_dims,))
        w_Y1 = self.sample_weights((features_dims,))

        E_y0 = np.einsum("np,p->n", covariate_features, w_Y0)
        E_y1 = np.einsum("np,p->n", covariate_features, w_Y1)

        y0 = E_y0 + self.sample_exogenous_noise((self.n_samples,))
        y1 = E_y1 + self.sample_exogenous_noise((self.n_samples,))

        outcomes = np.where(treatments == 1, y1, y0)

        if self.standardize_outcome:
            outcomes_mean, outcomes_std = outcomes.mean(), outcomes.std() + 1e-20
        else:
            outcomes_mean, outcomes_std = 0, 1
        outcomes = (outcomes - outcomes_mean) / outcomes_std
        y0, y1 = (y0 - outcomes_mean) / outcomes_std, (y1 - outcomes_mean) / outcomes_std
        E_y0, E_y1 = (E_y0 - outcomes_mean) / outcomes_std, (E_y1 - outcomes_mean) / outcomes_std

        return covariates, treatments, y0, y1, E_y0, E_y1, outcomes

    def __getitem__(self, index) -> Tuple[CATE_Dataset, ATE_Dataset]:
        if index >= self.n_tables:
            raise IndexError("Index out of range for the dataset catalog")

        np.random.seed(self.seeds[index])

        covariates, treatments, _, _, E_y0, E_y1, outcomes = self.get_covariates_T_Y0_Y1_E_Y0_E_Y1_outcomes()

        cate = E_y1 - E_y0

        indices = np.random.permutation(covariates.shape[0])
        split_idx = int(len(indices) * (1 - self.test_ratio))
        X_train, t_train, y_train = (
            covariates[indices[:split_idx]],
            treatments[indices[:split_idx]],
            outcomes[indices[:split_idx]],
        )
        X_test, cate_test = covariates[indices[split_idx:]], cate[indices[split_idx:]]

        cate_dataset = CATE_Dataset(
            X_train=X_train,
            t_train=t_train,
            y_train=y_train,
            X_test=X_test,
            true_cate=cate_test,
        )

        ate_dataset = ATE_Dataset(
            X=covariates,
            t=treatments,
            y=outcomes,
            true_ate=float((E_y1 - E_y0).mean()),
        )

        return cate_dataset, ate_dataset
