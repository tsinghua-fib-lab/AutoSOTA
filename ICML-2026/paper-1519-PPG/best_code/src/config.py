import multiprocessing
from dataclasses import dataclass, field


@dataclass
class EnvironmentConfig:
    """Environment parameters."""

    gamma: float = 0.9
    eps: float = 0.5
    num_followers: int = 2
    beta: float = 5.0


@dataclass
class ExperimentConfig:
    """Experiment parameters."""

    max_iterations: int = 100
    lamda: float = 0.1
    reg: str = "L2"
    gradient: bool = True
    eta: float = 0.1
    sampling: bool = False
    n_sample: int = 100
    policy_gradient: bool = True
    nu: float = 0.1
    unregularized_obj: bool = False
    lagrangian: bool = False
    N: int = 10
    delta: float = 0.1
    B: float = 1.0
    feature_dim: int = 32
    nn_lr: float = 0.001
    warmup: int = 1
    num_trajectories: int = 30
    value_lr: float = 0.1


@dataclass
class AblationConfig:
    """Ablation study configuration."""

    num_seeds: int = 20
    n_jobs: int = field(default_factory=lambda: min(4, multiprocessing.cpu_count()))

    def get_seeds(self):
        return list(range(1, self.num_seeds + 1))


@dataclass
class EntropyAblationConfig(AblationConfig):
    """Entropy regularization ablation configuration."""

    entropy_lambdas: list = field(default_factory=lambda: [0.01, 0.5, 1, 2])
    max_iterations: int = 5
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    experiment: ExperimentConfig = field(
        default_factory=lambda: ExperimentConfig(
            max_iterations=5,
            lamda=0.1,
            reg="ER",
            gradient=True,
            eta=0.1,
            sampling=False,
            n_sample=100,
            policy_gradient=True,
            nu=0.1,
        )
    )


@dataclass
class LearningRateAblationConfig(AblationConfig):
    """Learning rate ablation configuration."""

    learning_rates: list = field(default_factory=lambda: [0.01, 0.05, 0.1, 0.2, 0.5])
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


@dataclass
class PlotConfig:
    """Plotting configuration."""

    figsize: tuple = (16, 6)
    dpi: int = 300
    font_size: int = 20
    legend_size: int = 14
    title_size: int = 22
    tick_size: int = 16
    linewidth: float = 2.0
    alpha: float = 0.2
    confidence_level: float = 0.95
    t_critical: float = 2.093

    def get_ci_multiplier(self, num_samples=20):
        import numpy as np

        return self.t_critical / np.sqrt(num_samples)


DEFAULT_ENV_CONFIG = EnvironmentConfig()
DEFAULT_EXPERIMENT_CONFIG = ExperimentConfig()
DEFAULT_ABLATION_CONFIG = AblationConfig()
DEFAULT_PLOT_CONFIG = PlotConfig()
