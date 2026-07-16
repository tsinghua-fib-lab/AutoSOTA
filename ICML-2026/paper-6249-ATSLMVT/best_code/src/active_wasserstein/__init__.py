"""Active learning on Wasserstein trajectories."""

from .active import (
    AcquiredMeasurement,
    AcquisitionFunction,
    AcquisitionRecord,
    ActiveLearningLoop,
    LinearizedWassersteinGPSurrogate,
    MeasurementOracle,
    SurrogateModel,
)
from .acquisition import UncertaintySampler
from .geometry import (
    POTTransportSolver,
    TangentBasis,
    TransportResult,
    TangentVector,
    build_pca_tangent_basis,
    pca_vector_fields,
    wasserstein_barycenter,
)
from .inference import (
    GPyTorchHilbertPredictive,
    GPyTorchHilbertRegressor,
    InputScaler,
    KernelSpec,
    MaternKernelSpec,
    PredictiveProcess,
    RBFKernelSpec,
    TangentObservation,
    TangentObservationModel,
    reconstruct_distribution_at_time,
    reconstruct_distributions,
)
from .measures import (
    EmpiricalMeasure,
    ProbabilityMeasure,
    WeightedEmpiricalMeasure,
)
from .utils import (
    IdentityWarp,
    TimeGrid,
    WassersteinArcLengthWarp,
    compute_wasserstein_distance,
)
from .data import (
    CpsMonthlyTrajectory,
    OscillatorySequentialBranching,
    SchiebingerReprogrammingTrajectory,
    SequentialBranchingTrajectory,
)

__all__ = [
    "AcquiredMeasurement",
    "AcquisitionFunction",
    "AcquisitionRecord",
    "ActiveLearningLoop",
    "EmpiricalMeasure",
    "CpsMonthlyTrajectory",
    "SchiebingerReprogrammingTrajectory",
    "OscillatorySequentialBranching",
    "SequentialBranchingTrajectory",
    "POTTransportSolver",
    "build_pca_tangent_basis",
    "GPyTorchHilbertPredictive",
    "GPyTorchHilbertRegressor",
    "KernelSpec",
    "MaternKernelSpec",
    "ProbabilityMeasure",
    "WeightedEmpiricalMeasure",
    "UncertaintySampler",
    "RBFKernelSpec",
    "TangentBasis",
    "TangentObservation",
    "TangentObservationModel",
    "TangentVector",
    "LinearizedWassersteinGPSurrogate",
    "MeasurementOracle",
    "pca_vector_fields",
    "TimeGrid",
    "wasserstein_barycenter",
    "compute_wasserstein_distance",
    "InputScaler",
    "IdentityWarp",
    "PredictiveProcess",
    "reconstruct_distribution_at_time",
    "reconstruct_distributions",
    "WassersteinArcLengthWarp",
    "TransportResult",
    "SurrogateModel",
]


def main() -> None:
    print("Active Wasserstein GP module")
