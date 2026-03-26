from typing import Optional, Tuple

from pfl.metrics import Metrics, StringMetricName, Weighted
from pfl.privacy import LaplaceMechanism, GaussianMechanism
from pfl.stats import TrainingStatistics

from .mechanism import SymmetricGaussianMechanism


class DataPrivacyGaussianMechanism(GaussianMechanism):

    def constrain_sensitivity(
            self,
            statistics: TrainingStatistics,
            name_formatting_fn=lambda n: StringMetricName(n),
            seed: Optional[int] = None) -> Tuple[TrainingStatistics, Metrics]:
        return statistics, Metrics()


class DataPrivacySymmetricGaussianMechanism(SymmetricGaussianMechanism):

    def constrain_sensitivity(
            self,
            statistics: TrainingStatistics,
            name_formatting_fn=lambda n: StringMetricName(n),
            seed: Optional[int] = None) -> Tuple[TrainingStatistics, Metrics]:
        return statistics, Metrics()


class DataPrivacyLaplaceMechanism(LaplaceMechanism):

    def constrain_sensitivity(
            self,
            statistics: TrainingStatistics,
            name_formatting_fn=lambda n: StringMetricName(n),
            seed: Optional[int] = None) -> Tuple[TrainingStatistics, Metrics]:
        return statistics, Metrics()


