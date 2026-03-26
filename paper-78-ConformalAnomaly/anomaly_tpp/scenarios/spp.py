from scipy import stats
from anomaly_tpp import simulate
from anomaly_tpp.data import SequenceDataset


class StandardPoissonScenario:
    """Toy scenario, where ID sequences come from standard Poisson process.

    OOD sequences are drawn from a different synthetic TPP.
    The detectability parameter determines how different the two distributions are.

    """
    def __init__(self, t_max: float):
        self.t_max = t_max

    def sample_id(self, num_sequences: int) -> SequenceDataset:
        return SequenceDataset(
            [
                simulate.homogeneous_poisson(t_max=self.t_max, rate=1.0)
                for _ in range(num_sequences)
            ]
        )

    def sample_ood(self, num_sequences: int, detectability: float) -> SequenceDataset:
        """Return a dataset consisting of out-of-distribution sequences."""
        raise NotImplementedError()

    @property
    def name(self):
        return self.__class__.__name__


class Hawkes(StandardPoissonScenario):
    """OOD sequences come from a self-exciting process."""
    def __init__(self, t_max: float):
        super().__init__(t_max)

    def sample_ood(self, num_sequences: int, detectability: float) -> SequenceDataset:
        influence = detectability
        base_rates = 1 - influence  # ensure that expected num_events doesn't change
        return SequenceDataset(
            [
                simulate.hawkes_exp_kernels(
                    t_max=self.t_max, base_rates=base_rates, influence=influence
                )
                for _ in range(num_sequences)
            ]
        )


class RenewalUp(StandardPoissonScenario):
    """Interpolating between Expo(1) and Gamma(k, 1/k). k going up."""

    def __init__(self, t_max: float, max_k: float = 5.0):
        self.max_k = max_k
        super().__init__(t_max)

    def sample_ood(self, num_sequences: int, detectability: float) -> SequenceDataset:
        k = 1.0 + detectability * (self.max_k - 1.0)
        gamma_dist = stats.gamma(k, scale=(1.0 / k))
        return SequenceDataset(
            [
                simulate.renewal(t_max=self.t_max, renewal_distributions=gamma_dist)
                for _ in range(num_sequences)
            ]
        )


class RenewalDown(StandardPoissonScenario):
    """Interpolating between Expo(1) and Gamma(k, 1/k). k going down."""

    def __init__(self, t_max: float):
        super().__init__(t_max)

    def sample_ood(self, num_sequences: int, detectability: float) -> SequenceDataset:
        if detectability < 0 or detectability >= 1:
            raise ValueError(
                f"Detectability must be in [0, 1) range (got {detectability})"
            )
        k = 1 - detectability
        gamma_dist = stats.gamma(k, scale=(1.0 / k))
        return SequenceDataset(
            [
                simulate.renewal(t_max=self.t_max, renewal_distributions=gamma_dist)
                for _ in range(num_sequences)
            ]
        )


class SelfCorrecting(StandardPoissonScenario):
    """OOD sequences come from a self-correcting process."""
    def __init__(self, t_max: float, max_alpha: float = 1.0):
        self.max_alpha = max_alpha
        super().__init__(t_max)

    def sample_ood(self, num_sequences: int, detectability: float) -> SequenceDataset:
        alpha = detectability * self.max_alpha + 1e-5
        mu = alpha
        return SequenceDataset(
            [
                simulate.self_correcting(t_max=self.t_max, alpha=alpha, mu=mu)
                for _ in range(num_sequences)
            ]
        )
    

class Uniform(StandardPoissonScenario):
    """Generates sequences of uniform inter-event times."""
    def __init__(self, t_max: float):
        super().__init__(t_max)

    def sample_ood(self, num_sequences: int, detectability: float) -> SequenceDataset:
        return SequenceDataset(
            [
                simulate.uniform_sequence(t_max=self.t_max, det=detectability)
                for _ in range(num_sequences)
            ]
        )
    
class IncreasingRate(StandardPoissonScenario):
    """The rate of the homogeneous Poisson process increases."""
    def __init__(self, t_max: float, max_rate: float = 1.5):
        self.max_rate = max_rate
        super().__init__(t_max)

    def sample_ood(self, num_sequences: int, detectability: float) -> SequenceDataset:
        rate = 1.0 + detectability * (self.max_rate - 1.0)
        return SequenceDataset(
            [
                simulate.homogeneous_poisson(t_max=self.t_max, rate=rate)
                for _ in range(num_sequences)
            ]
        )


class DecreasingRate(StandardPoissonScenario):
    """The rate of the homogeneous Poisson process decreases."""
    def __init__(self, t_max: float, min_rate: float = 0.5):
        self.min_rate = min_rate
        super().__init__(t_max)

    def sample_ood(self, num_sequences: int, detectability: float) -> SequenceDataset:
        rate = 1.0 - detectability * (1.0 - self.min_rate)
        return SequenceDataset(
            [
                simulate.homogeneous_poisson(t_max=self.t_max, rate=rate)
                for _ in range(num_sequences)
            ]
        )


class InhomogeneousPoisson(StandardPoissonScenario):
    """Intensity becomes non-constant."""
    def __init__(self, t_max: float, max_amplitude: float = 2.0):
        self.max_amplitude = max_amplitude
        super().__init__(t_max)

    def sample_ood(self, num_sequences: int, detectability: float) -> SequenceDataset:
        amplitude = detectability * self.max_amplitude
        return SequenceDataset(
            [
                simulate.inhomogeneous_poisson(t_max=self.t_max, amplitude=amplitude)
                for _ in range(num_sequences)
            ]
        )


class Stopping(StandardPoissonScenario):
    """Events stop (rate changes to 0) after some time."""

    def __init__(self, t_max: float):
        super().__init__(t_max)

    def sample_ood(self, num_sequences: int, detectability: float) -> SequenceDataset:
        t_jump = self.t_max * (
            1.0 - 0.3 * detectability
        )  # time of the failure (absolute)
        sequences = [
            simulate.jump_poisson(
                t_max=self.t_max, t_jump=t_jump, rate_before=1.0, rate_after=0.0
            )
            for _ in range(num_sequences)
        ]
        return SequenceDataset(sequences)