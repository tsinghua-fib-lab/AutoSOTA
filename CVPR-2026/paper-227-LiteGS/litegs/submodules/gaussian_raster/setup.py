from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="litegs_fused",
    packages=['litegs_fused'],
    package_dir={'litegs_fused':"."},
    ext_modules=[
        CUDAExtension(
            name="litegs_fused",
            sources=[
            "binning.cu",
            "compact.cu",
            "cuda_errchk.cpp",
            "ext_cuda.cpp",
            "raster.cu",
            "transform.cu"])
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
