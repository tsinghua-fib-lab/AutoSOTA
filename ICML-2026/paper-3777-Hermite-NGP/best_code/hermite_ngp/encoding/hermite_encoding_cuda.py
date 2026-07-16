"""Back-compat alias for legacy import paths.

Original 2D and 3D example scripts both did
``from hermite_ngp.encoding.hermite_encoding_cuda import HermiteHashEncodingCUDA[_3D]``.
This shim re-exports the renamed 2D/3D classes so those scripts work unchanged.

For new code, prefer:
    from hermite_ngp.encoding import HermiteHashEncoding2D, HermiteHashEncoding3D
"""

from hermite_ngp.encoding.hermite_encoding_2d import (  # noqa: F401
    HermiteHashEncodingCUDA,
    HermiteHashEncoding2D,
    HermiteEncodingFunction,
    HermiteEncodingWithDerivativesFunction,
    get_hermite_encoding as get_hermite_encoding_2d,
    CUDA_AVAILABLE,
)
from hermite_ngp.encoding.hermite_encoding_3d import (  # noqa: F401
    HermiteHashEncodingCUDA_3D,
    HermiteHashEncoding3D,
    HermiteEncodingFunction_3D,
    HermiteEncodingWithDerivativesFunction_3D,
    get_hermite_encoding as get_hermite_encoding_3d,
)
