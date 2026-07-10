"""Feature encoder package.

Re-exports the dual-stream feature encoders described in paper Sec. 3.1
(Residual Stream + RGB Stream) and the quantization residual extractor
(paper Eq. 1).
"""

from .residual_extractor import (
    M_YUV,
    M_YUV_INV,
    QuantizationResidualExtractor,
    get_residual_extractor,
)
from .residual_stream import ResidualStream
from .rgb_stream import CHANNELS, RgbStream, resolve_local_dino_path

__all__ = [
    "M_YUV",
    "M_YUV_INV",
    "QuantizationResidualExtractor",
    "get_residual_extractor",
    "ResidualStream",
    "CHANNELS",
    "RgbStream",
    "resolve_local_dino_path",
]
