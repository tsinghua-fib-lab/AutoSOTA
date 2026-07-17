"""
OTP-FM: Flow Matching with Optimal Transport Potentials.
"""

from otpfm.otpfm import OTPFM
from otpfm.training import Curriculum

try:
    from otpfm._version import __version__
except ImportError:
    __version__ = "0.0.0+unknown"

__all__ = ["OTPFM", "Curriculum"]
