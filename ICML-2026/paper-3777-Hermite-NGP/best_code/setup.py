"""
Unified build script for Hermite-NGP CUDA extensions.

Builds 5 extensions in one pass:
  1. hermite_encoding_cuda           (2D encoding, pybind11)
  2. hermite_encoding_cuda_3d        (3D encoding, pybind11)
  3. hermite_mlp_cuda_v2             (2D SIREN MLP with analytic Laplacian)
  4. hermite_mlp_cuda_3d_v2          (3D SIREN MLP with analytic Laplacian)
  5. siren_hessian_cuda_3d           (standalone SIREN Hessian helper)

Install with:
    pip install -e .
or
    python setup.py install

CUDA arch defaults to RTX 4090 (sm_89) for encoding kernels and broader sm_70
for MLP kernels. Override with TORCH_CUDA_ARCH_LIST environment variable, e.g.:
    TORCH_CUDA_ARCH_LIST="8.0;8.9" pip install -e .
"""

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CSRC = os.path.join("hermite_ngp", "csrc")


def src(*names):
    return [os.path.join(CSRC, n) for n in names]


# Allow user to override compute capability via env var. By default, build for
# the common range from Volta (sm_70) up through Ada (sm_89).
DEFAULT_ARCH_LIST = os.environ.get(
    "TORCH_CUDA_ARCH_LIST", "7.0;7.5;8.0;8.6;8.9"
)

# Common compile flags
COMMON_CXX = ["-O3", "-std=c++17"] if os.name != "nt" else ["/O2", "/std:c++17"]
COMMON_NVCC = ["-O3", "--use_fast_math", "-std=c++17"]


def cuda_ext(name, sources):
    return CUDAExtension(
        name=name,
        sources=sources,
        extra_compile_args={"cxx": COMMON_CXX, "nvcc": COMMON_NVCC},
    )


ext_modules = [
    cuda_ext(
        "hermite_encoding_cuda",
        src("hermite_encoding_2d.cpp", "hermite_encoding_2d_kernel.cu"),
    ),
    cuda_ext(
        "hermite_encoding_cuda_3d",
        src("hermite_encoding_3d.cpp", "hermite_encoding_3d_kernel.cu"),
    ),
    cuda_ext(
        "hermite_mlp_cuda_v2",
        src("hermite_mlp_2d_kernel.cu"),
    ),
    cuda_ext(
        "hermite_mlp_cuda_3d_v2",
        src("hermite_mlp_3d_kernel.cu"),
    ),
    cuda_ext(
        "siren_hessian_cuda_3d",
        src("siren_hessian_3d_kernel.cu"),
    ),
]


# Respect TORCH_CUDA_ARCH_LIST for all extensions
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", DEFAULT_ARCH_LIST)


if __name__ == "__main__":
    setup(
        name="hermite-ngp",
        version="1.0.0",
        ext_modules=ext_modules,
        cmdclass={"build_ext": BuildExtension},
    )
