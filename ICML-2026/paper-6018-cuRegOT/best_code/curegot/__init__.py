from ._cuda_loader import preload_cuda_runtime

preload_cuda_runtime()

# Load NumPy interface
from ._internal_numpy import *

# When the package is built using Pytorch, we typically
# need to import torch first to properly load some libraries
# such as libc10.so
try:
    import torch
    # Load PyTorch interface
    from ._internal_torch import *
except:
    pass

__version__ = "0.1.0"
