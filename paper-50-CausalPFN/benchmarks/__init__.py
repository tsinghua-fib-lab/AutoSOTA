from .acic2016 import ACIC2016Dataset
from .criteo import CriteoDataset
from .hillstrom import HillstromDataset
from .ihdp import IHDPDataset
from .lenta import LentaDataset
from .megafon import MegafonDataset
from .polynomial import PolynomialDataset
from .realcause import RealCauseLalondeCPSDataset, RealCauseLalondePSIDDataset
from .retail_hero import X5Dataset

__all__ = [
    "ACIC2016Dataset",
    "IHDPDataset",  # healthcare dataset
    "RealCauseLalondePSIDDataset",
    "RealCauseLalondeCPSDataset",
    "HillstromDataset",  # marketting dataset
    "CriteoDataset",  # marketting dataset
    "LentaDataset",  # marketting dataset
    "X5Dataset",  # marketting dataset
    "MegafonDataset",  # marketting dataset
    "PolynomialDataset",  # synthetic dataset
]
