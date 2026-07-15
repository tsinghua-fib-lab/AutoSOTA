"""
Configuration for each algorithm
"""

from .federated import Federated
from .fedasync import FedAsync
from .iceadmm import ICEADMM
from .iiadmm import IIADMM

__all__ = ["Federated", "FedAsync", "ICEADMM", "IIADMM"]
