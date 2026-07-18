import os
import pathlib
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


# Get the path to local glm
current_dir = pathlib.Path(__file__).parent.resolve()
glm_path = os.path.join(current_dir, "third_party", "glm")


setup(
    name="clm_kernels",
    packages=["clm_kernels"],
    ext_modules=[
        CUDAExtension(
            name="clm_kernels._C",
            sources=[
                "clm_kernels.cu",
                "ssim.cu",
                "adam.cu",
                "compute_sh_bwd.cu",
                "ext.cpp",
            ],
            include_dirs=[glm_path],
            extra_compile_args={"nvcc": ["-O3"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
