from enum import Enum, StrEnum
from pathlib import Path


class Datasets(Enum):
    OU_PROCESS = Path("data/ou_processes/ou_process_data.npz")
    BLACK_SCHOLES = Path("data/rough_volatility/black-scholes_data.npz")
    BERGOMI = Path("data/rough_volatility/bergomi_data.npz")
    ROUGH_BERGOMI = Path("data/rough_volatility/rough_bergomi_data.npz")
    SIMPLE_RBERGOMI = Path("data/rough_volatility/simple_rbergomi_data.npz")
    SYNTHETIC_GBM = "synthetic_gbm"
    # Backwards-compatible alias (accepts dataset_name="simple_rough_bergomi" too).
    SIMPLE_ROUGH_BERGOMI = Path("data/rough_volatility/simple_rbergomi_data.npz")
    SG_SO3_SIMULATION = Path(
        "data/sg_so3_simulation/so3_simulation_rotmats_by_damping.npz"
    )
    OXFORD_MULTIMOTION_STATIC = Path("data/oxford_multimotion/swinging_4_static.npz")
    OXFORD_MULTIMOTION_TRANSLATIONAL = Path(
        "data/oxford_multimotion/swinging_4_translational.npz"
    )
    OXFORD_MULTIMOTION_UNCONSTRAINED = Path(
        "data/oxford_multimotion/swinging_4_unconstrained.npz"
    )
    SPD_WISHART_DIFFUSION = Path("data/synthetic_diffusions/wishart_diffusion_data.npz")
    PPG_DALIA = None


class ModelType(StrEnum):
    NCDE = "ncde"
    NRDE = "nrde"
    BNRDE = "bnrde"
    M_ODE = "m_ode"
    GRU = "gru"
    LSTM = "lstm"
    XLSTM = "xlstm"
    STACKED_XLSTM = "stacked_xlstm"


class Optimizer(StrEnum):
    ADAM = "adam"
    ADAMW = "adamw"
    MUON = "muon"


class HopfAlgebraType(StrEnum):
    SHUFFLE = "shuffle"
    GL = "gl"
    MKW = "mkw"


class RoughSolution(StrEnum):
    ITO = "ito"
    STRATONOVICH = "stratonovich"


class StepsizeControllerType(StrEnum):
    PID = "pid"
    CONSTANT = "constant"


class SolverType(StrEnum):
    EES252N = "ees252n"
    CFEES25 = "cfees25"
    TSIT5 = "tsit5"
    HEUN = "heun"


class AdjointType(StrEnum):
    RECURSIVE_CHECKPOINT = "recursive_checkpoint"
    REVERSIBLE = "reversble"


class ManifoldType(StrEnum):
    EUCLIDEAN = "euclidean"
    SO3 = "so3"
    SPD = "spd"


class HiddenStateMode(StrEnum):
    EUCLIDEAN = "euclidean"
    PROBLEM_MANIFOLD = "problem_manifold"


class ControlInterpolationType(StrEnum):
    HERMITE_CUBIC = "hermite_cubic"
    LINEAR = "linear"


class ExtrapolationSchemeType(StrEnum):
    LINEAR = "linear"
    HERMITE = "hermite"
    SG = "sg"
    SO3_SG = "so3_sg"
    PIECEWISE_MLP = "piecewiseMLP"


class LossType(StrEnum):
    MSE = "mse"
    RGE = "rge"
    SIGKER = "sigker"
    SIGKER_BRANCHED = "sigker_branched"
    FROBENIUS = "frobenius"


class TrainingMode(StrEnum):
    CONDITIONAL = "conditional"
    UNCONDITIONAL = "unconditional"


class FinalActivation(StrEnum):
    TANH = "tanh"
    IDENTITY = "identity"
