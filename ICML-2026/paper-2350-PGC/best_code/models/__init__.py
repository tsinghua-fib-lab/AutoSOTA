"""PGC model package.

Re-exports the top-level :class:`PGCNetwork` together with its sub-components
so external code only needs to know about ``models.*``.
"""

from .encoder.residual_extractor import (
    QuantizationResidualExtractor,
    get_residual_extractor,
)
from .encoder.residual_stream import ResidualStream
from .encoder.rgb_stream import CHANNELS, RgbStream
from .lora.lora import (
    LoRALayer,
    LoRALinear,
    apply_lora_to_linear_layers,
    get_lora_params,
)
from .pgc import PGCNetwork
from .pgcm.peak_calibration import PeakGuidedCalibrationModule

__all__ = [
    "PGCNetwork",
    "RgbStream",
    "ResidualStream",
    "CHANNELS",
    "QuantizationResidualExtractor",
    "get_residual_extractor",
    "PeakGuidedCalibrationModule",
    "LoRALayer",
    "LoRALinear",
    "apply_lora_to_linear_layers",
    "get_lora_params",
]
