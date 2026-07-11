from pkg_resources import get_distribution
__version__ = get_distribution('DeepKnockoffs').version

from .machine import KnockoffMachine
from .gaussian import GaussianKnockoffs
