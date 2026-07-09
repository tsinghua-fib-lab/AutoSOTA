"""Init file for adapters.""" 

from .projector import MultichannelProjector
from .var_selector import VarianceBasedSelector
from .diff_adapter import LinearChannelCombiner


__all__ = ['MultichannelProjector', 'VarianceBasedSelector', 'LinearChannelCombiner']
