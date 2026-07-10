import os

import numpy as np
from numpy.typing import NDArray

# NDArray typing
NDArrayF32 = NDArray[np.float32]
NDArrayI8 = NDArray[np.int8]
NDArrayI16 = NDArray[np.int16]
NDArrayI32 = NDArray[np.int32]
NDArrayI64 = NDArray[np.int64]
NDArrayU8 = NDArray[np.uint8]
NDArrayU16 = NDArray[np.uint16]
NDArrayU32 = NDArray[np.uint32]
NDArrayU64 = NDArray[np.uint64]
# Path typing
PathLike = str | os.PathLike[str]
