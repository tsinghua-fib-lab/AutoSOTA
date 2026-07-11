from .ncde import NeuralCDE
from .nrde import NeuralRDE
from .bnrde import BNRDE
from .m_ode import ManifoldNeuralODE
from .gru import GRU
from .lstm import LSTM
from .xlstm import XLSTM
from .stacked_xlstm import StackedXLSTM
from .extrapolation import (
    ExtrapolationScheme,
    LinearScheme,
    HermiteScheme,
    WeightedSGScheme,
    create_scheme,
)

Model = (
    NeuralCDE
    | NeuralRDE
    | BNRDE
    | ManifoldNeuralODE
    | GRU
    | LSTM
    | XLSTM
    | StackedXLSTM
)

__all__ = [
    "Model",
    "NeuralCDE",
    "NeuralRDE",
    "BNRDE",
    "ManifoldNeuralODE",
    "GRU",
    "LSTM",
    "XLSTM",
    "StackedXLSTM",
    "ExtrapolationScheme",
    "LinearScheme",
    "HermiteScheme",
    "WeightedSGScheme",
    "create_scheme",
]
