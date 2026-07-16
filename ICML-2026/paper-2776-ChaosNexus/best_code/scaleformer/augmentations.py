"""
Training augmentations for multivariate time series
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scaleformer.utils import safe_standardize


@dataclass
class RandomDimSelectionTransform:
    """Randomly select a subset of dimensions from a timeseries"""

    num_dims: int
    random_seed: int = 0

    def __post_init__(self) -> None:
        self.rng: np.random.Generator = np.random.default_rng(self.random_seed)

    def __call__(self, timeseries: NDArray, axis: int = 0) -> NDArray:
        selected_dims = self.rng.choice(
            timeseries.shape[axis], self.num_dims, replace=False
        )
        return np.take(timeseries, selected_dims, axis=axis)


@dataclass
class StandardizeTransform:
    """Unecessary wrapper around safe_standardize"""

    def __call__(
        self,
        timeseries: NDArray,
        axis: int = -1,
        context: NDArray | None = None,
        **kwargs,
    ) -> NDArray:
        """
        :param timeseries: (num_channels, num_timepoints) timeseries to standardize
        """
        return safe_standardize(timeseries, axis=axis, context=context, **kwargs)


@dataclass
class FixedDimensionDelayEmbeddingTransform:
    """Delays embeds a timeseries to a fixed embedding dimension

    NOTE:
        - if the embedding dimension is non permissible (e.g. more delays than timepoints), an error is raised
        - if the dimension of the timeseries is greater than the embedding dimension, then embedding dimensions
          are randomly sampled from the channel dimension

    :param embedding_dim: embedding dimension for the delay embeddings
    :param channel_dim: dimension corresponding to channels
    :param time_dim: dimension corresponding to timepoints
    """

    embedding_dim: int
    random_seed: int = 0

    def __post_init__(self) -> None:
        self.rng: np.random.Generator = np.random.default_rng(self.random_seed)

    def __call__(self, timeseries: NDArray) -> NDArray:
        """
        :param timeseries: (num_channels, num_timepoints) timeseries to delay embed
        """
        assert timeseries.shape[1] > self.embedding_dim, (
            "Embedding dimension cannot be larger than the number of timepoints"
        )
        num_channels = timeseries.shape[0]
        per_dim_embed_dim = (self.embedding_dim - num_channels) // num_channels
        remaining_dims = (self.embedding_dim - num_channels) % num_channels

        if num_channels >= self.embedding_dim:
            selected_dims = self.rng.choice(
                num_channels, self.embedding_dim, replace=False
            )
            return timeseries[selected_dims]

        # Distribute remaining dimensions evenly
        extra_dims = [1 if i < remaining_dims else 0 for i in range(num_channels)]

        delay_embeddings = [
            np.stack(
                [
                    np.roll(timeseries[i], shift)
                    for shift in range(1, 1 + per_dim_embed_dim + extra)
                ]
            )[:, 1 + per_dim_embed_dim :]
            for i, extra in enumerate(extra_dims)
            if extra > 0 or per_dim_embed_dim > 0
        ]

        return np.vstack([timeseries[:, 1 + per_dim_embed_dim :], *delay_embeddings])


@dataclass
class RandomPhaseSurrogate:
    """Creates a phase surrogate of a timeseries by randomizing the phases of the Fourier coefficients

    :param cutoff: fraction of frequencies to keep
    :param random_seed: RNG seed
    """

    cutoff: float = 0.1
    random_seed: int = 0

    def __post_init__(self) -> None:
        self.rng: np.random.Generator = np.random.default_rng(self.random_seed)

    def __call__(self, timeseries: NDArray) -> NDArray:
        """
        :param timeseries: (num_channels, num_timepoints) timeseries to create a phase surrogate for
        """
        freqs = np.fft.rfft(timeseries, axis=1)
        surrogate_freqs = freqs * np.exp(2j * np.pi * np.random.rand(*freqs.shape))

        # cut off high frequencies
        n_freqs = surrogate_freqs.shape[1]
        surrogate_freqs[:, int(n_freqs * self.cutoff) :] = 0

        # preserve the DC component
        surrogate_freqs[:, 0] = freqs[:, 0]

        surrogates = np.fft.irfft(surrogate_freqs, axis=1)

        return surrogates


@dataclass
class RandomFourierSeries:
    """Creates a random Fourier series of the same shape as the input timeseries.

    Completely ignores the input content and generates a new signal by summing random sinusoids.
    The frequencies are sampled uniformly from [0, max_freq], amplitudes from [0, max_amp],
    and phases from [0, 2π].

    Args:
        max_freq: Maximum frequency component to include (Hz)
        max_amp: Maximum amplitude for each sinusoid
        num_components: Number of frequency components to sum
        random_seed: RNG seed
    """

    max_wavenumber: float = 10.0
    max_amp: float = 10.0
    random_seed: int = 0
    mode_range: tuple[int, int] = (5, 15)

    def __post_init__(self) -> None:
        self.rng: np.random.Generator = np.random.default_rng(self.random_seed)

    def __call__(self, timeseries: NDArray) -> NDArray:
        """Generate random Fourier series matching input shape using FFT"""
        num_channels, num_timepoints = timeseries.shape
        num_components = self.rng.integers(*self.mode_range, endpoint=True)

        freqs = self.rng.uniform(
            0, np.pi * self.max_wavenumber, (num_channels, num_components)
        )
        amps = self.rng.uniform(0, self.max_amp, (num_channels, num_components))
        phases = self.rng.uniform(0, 2 * np.pi, (num_channels, num_components))

        t = np.linspace(0, 1, num_timepoints)
        fourier_series = np.sum(
            amps[..., np.newaxis]
            * np.sin(2 * np.pi * freqs[..., np.newaxis] * t + phases[..., np.newaxis]),
            axis=1,
        )

        return fourier_series


@dataclass
class RandomTakensEmbedding:
    """Random Takens embedding of a single coordinate with delay 1

    Takes a (D,T) trajectory, randomly selects one dimension, and creates a delay embedding
    with delay 1, preserving the original dimensionality.

    :param random_seed: RNG seed
    """

    lag_range: tuple[int, int] = (1, 10)
    random_seed: int = 0

    def __post_init__(self) -> None:
        self.rng: np.random.Generator = np.random.default_rng(self.random_seed)

    def __call__(self, timeseries: NDArray) -> NDArray:
        """
        :param timeseries: (num_channels, num_timepoints) timeseries to delay embed
        """
        lag = self.rng.integers(*self.lag_range, endpoint=True)

        # select random dimension to embed
        embed_dim = self.rng.integers(0, timeseries.shape[0])

        # create delay embedding of selected dimension with specified time lag
        N = timeseries.shape[1] - lag * (timeseries.shape[0] - 1)
        delays = np.stack(
            [
                timeseries[embed_dim, i * lag : i * lag + N]
                for i in range(timeseries.shape[0])
            ]
        )

        return delays


@dataclass
class RandomConvexCombinationTransform:
    """Random convex combinations of coordinates with coefficients sampled from a dirichlet distribution

    :param num_combinations: number of random convex combinations to sample
    :param alpha: dirichlet distribution scale
    :param random_seed: RNG seed
    """

    alpha: float
    random_seed: int = 0
    dim_range: tuple[int, int] = (3, 10)

    def __post_init__(self) -> None:
        self.rng: np.random.Generator = np.random.default_rng(self.random_seed)

    def __call__(self, timeseries: NDArray) -> NDArray:
        dims = self.rng.integers(self.dim_range[0], self.dim_range[1], endpoint=True)
        coeffs = self.rng.dirichlet(
            self.alpha * np.ones(timeseries.shape[0]), size=dims
        )
        return coeffs @ timeseries


@dataclass
class RandomAffineTransform:
    """Random affine transformations of coordinates with coefficients sampled from a zero-mean Gaussian

    :param out_dim: output dimension of the linear map
    :param scale: gaussian distribution scale
    :param random_seed: RNG seed
    """

    scale: float
    random_seed: int = 0
    dim_range: tuple[int, int] = (3, 10)

    def __post_init__(self) -> None:
        self.rng: np.random.Generator = np.random.default_rng(self.random_seed)

    def __call__(self, timeseries: NDArray) -> NDArray:
        dims = self.rng.integers(self.dim_range[0], self.dim_range[1], endpoint=True)
        affine_transform = self.rng.normal(
            scale=self.scale, size=(dims, 1 + timeseries.shape[0])
        ) / np.sqrt(dims)
        return (
            affine_transform[:, :-1] @ timeseries + affine_transform[:, -1, np.newaxis]
        )


@dataclass
class RandomProjectedSkewTransform:
    """
    Randomly combines pairs of timeseries and linearly maps them into a common embedding space
    Linear maps are zero mean gaussian random matrices

    :param embedding_dim: embedding dimension for the skew projection
    :param scale: scale for the gaussian random projection matrices
    :param random_seed: RNG seed

    TODO:
        - figure out how to make this on-the-fly
        - maybe even deprecate this, is chaoticity preserved?
    """

    embedding_dim: int
    scale: float
    random_seed: int = 0

    def __post_init__(self) -> None:
        self.rng: np.random.Generator = np.random.default_rng(self.random_seed)

    def __call__(self, timeseries1: NDArray, timeseries2: NDArray) -> NDArray:
        proj1 = self.rng.normal(
            scale=self.scale, size=(self.embedding_dim, timeseries1.shape[0])
        )
        proj2 = self.rng.normal(
            scale=self.scale, size=(self.embedding_dim, timeseries2.shape[0])
        )
        return proj1 @ timeseries1 + proj2 @ timeseries2
