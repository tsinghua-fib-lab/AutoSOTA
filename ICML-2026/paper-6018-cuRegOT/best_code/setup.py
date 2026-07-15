import os
import sys
from pathlib import Path
from glob import glob
import requests
import tarfile
from pybind11.setup_helpers import Pybind11Extension
from setuptools.command.build_ext import build_ext
from setuptools import setup
import subprocess

__version__ = "0.1.0"

# Download CCCL source files and return include directory
def get_cccl_include():
    # The directory that contains setup.py
    SETUP_DIRECTORY = str(Path(__file__).resolve().parent)

    CCCL_URL = "https://github.com/NVIDIA/cccl/releases/download/v3.3.3/cccl-v3.3.3.tar.gz"
    CCCL_FILE = "cccl-v3.3.3.tar.gz"
    CCCL_DIRNAME = "cccl-v3.3.3"

    # Test whether the environment variable CCCL_INCLUDE_DIR is set
    # If yes, directly return this directory
    cccl_include_dir = os.environ.get("CCCL_INCLUDE_DIR", None)
    if cccl_include_dir is not None:
        return cccl_include_dir

    # If the directory already exists (e.g. from previous setup),
    # directly return it
    cccl_dir = os.path.join(SETUP_DIRECTORY, CCCL_DIRNAME)
    cccl_include_dir = os.path.join(cccl_dir, "include")
    if os.path.exists(cccl_include_dir):
        return cccl_include_dir

    # Filename for the downloaded CCCL source package
    download_target_file = os.path.join(SETUP_DIRECTORY, CCCL_FILE)
    response = requests.get(CCCL_URL, stream=True)
    with open(download_target_file, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            file.write(chunk)
    # Unzip package
    with tarfile.open(download_target_file, "r:gz") as tar:
        tar.extractall(filter="data")

    return cccl_include_dir

# We want to make the PyTorch interface of cuRegOT optional to reduce
# the number of dependent packages
#
# When user imports cuRegOT, the package should first load the NumPy interface,
# and then detect whether CUDA-based PyTorch is installed. If yes, then the
# PyTorch interface is also loaded
#
# To achieve this, it is better to build two .so files, one for NumPy and one for PyTorch
#
# Also note that the .so file built by the PyTorch toolchain also depends on PyTorch,
# so for the NumPy interface, we need to use a custom build_ext class
#
# Since setup() only supports one cmdclass["build_ext"] argument, we write a unified
# build_ext class that has different behaviors for different modules

# Default NVCC -gencode option
DEFAULT_GENCODE_CC = [75, 80, 86, 89, 90, 100, 103, 120]
DEFAULT_GENCODE_FLAGS = " ".join([f"-gencode=arch=compute_{cc},code=sm_{cc}" for cc in DEFAULT_GENCODE_CC])

# Return -gencode flags as a list of strings
def get_gencode_flags():
    gencode_flags = os.environ.get("GENCODE_FLAGS", None)
    if gencode_flags is None:
        gencode_flags = DEFAULT_GENCODE_FLAGS
    
    return gencode_flags.strip().split(" ")

# Try to import PyTorch's CUDA extension utilities
TORCH_BUILD = False
try:
    from torch.utils.cpp_extension import CUDA_HOME, CUDAExtension as TorchCUDAExtension, BuildExtension as TorchBuildExtension
    TORCH_BUILD = True

    # Use PyTorch to detect CUDA_HOME if it is not set
    if "CUDA_HOME" not in os.environ and CUDA_HOME is not None:
        os.environ["CUDA_HOME"] = CUDA_HOME

    # Get CUDA architectures that PyTorch was compiled for
    from torch.cuda import get_arch_list
    torch_cc = [str.removeprefix("sm_") for str in get_arch_list()]
    torch_cc = [int(str) for str in torch_cc if str.isdigit()]
    gencode_cc = list(set(DEFAULT_GENCODE_CC + torch_cc))
    gencode_cc = sorted(gencode_cc)
    DEFAULT_GENCODE_FLAGS = " ".join([f"-gencode=arch=compute_{cc},code=sm_{cc}" for cc in gencode_cc])

    print("PyTorch detected, will use PyTorch's BuildExtension and CUDAExtension to build PyTorch interface")
except ImportError:
    print("PyTorch not found, will only build NumPy interface")
    TORCH_BUILD = False

# TORCH_BUILD = False

# Check for CUDA
def find_cuda():
    """Find CUDA installation"""
    # Check common CUDA installation paths
    cuda_paths = [
        "/usr/local/cuda",
        "/opt/cuda",
        "/usr/cuda"
    ]

    # Also check CUDA_HOME environment variable
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        cuda_paths.insert(0, cuda_home)

    for path in cuda_paths:
        if os.path.exists(path):
            print(f"CUDA path found: {path}")
            return path

    return None

# Check for NVCC
def check_cuda_compiler():
    """Check if nvcc is available"""
    try:
        result = subprocess.run(["nvcc", "--version"],
                                capture_output=True, text=True, check=True)
        print(f"CUDA compiler found:\n{result.stdout}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False



# NumPy interface: use Pybind11Extension with custom build_ext class
# Custom build_ext class that supports CUDA
class CUDABuildExtension(build_ext):
    def build_extension(self, ext):
        # Check if this extension has CUDA files
        cuda_sources = [src for src in ext.sources if src.endswith(".cu")]
        cpp_sources = [src for src in ext.sources if not src.endswith(".cu")]

        if cuda_sources:
            print(f"Building extension {ext.name} with CUDA support")

            # Check CUDA path
            if not find_cuda():
                print("Warning: CUDA installation not found. Please install CUDA toolkit and set CUDA_HOME.")
                sys.exit(1)
            
            # Check NVCC
            if not check_cuda_compiler():
                print("Warning: nvcc compiler not found. Please ensure CUDA toolkit is installed and in PATH.")
                sys.exit(1)

            # Store the original objects and add to extra_objects
            if not hasattr(ext, "extra_objects"):
                ext.extra_objects = []

            # Compile CUDA files first
            for cuda_file in cuda_sources:
                obj_file = self.compile_cuda_file(cuda_file, ext)
                ext.extra_objects.append(obj_file)

            # Remove CUDA files from sources, keep only C++ files
            ext.sources = cpp_sources

            # Ensure CUDA libraries are linked
            if not hasattr(ext, "libraries"):
                ext.libraries = []
            for lib in ["cuda", "cudart"]:
                if lib not in ext.libraries:
                    ext.libraries.append(lib)

        # Let the parent class handle the rest
        super().build_extension(ext)

    def compile_cuda_file(self, cuda_file, ext):
        """Compile a single .cu file to .o object file and return the object file path"""
        output_file = os.path.splitext(cuda_file)[0] + ".o"

        # Build nvcc command
        nvcc_cmd = ["nvcc", "-c", "-O3", "--use_fast_math", "-Xcompiler", "-fPIC"]
        nvcc_cmd += get_gencode_flags()

        # Add virtual environment include directory
        venv_inc_dir = os.path.join(sys.exec_prefix, "include")
        if venv_inc_dir not in self.compiler.include_dirs:
            nvcc_cmd.append("-I" + venv_inc_dir)

        # Add include directories specified in ext_modules
        if hasattr(ext, "include_dirs"):
            for inc_dir in ext.include_dirs:
                if inc_dir not in self.compiler.include_dirs:
                    nvcc_cmd.append("-I" + inc_dir)

        # Add Python include directories
        for inc_dir in self.compiler.include_dirs:
            nvcc_cmd.append("-I" + inc_dir)

        # Add source and output
        nvcc_cmd.extend([cuda_file, "-o", output_file])

        print(f"Compiling CUDA file: {cuda_file}")
        print(" ".join(nvcc_cmd))
        print()
        subprocess.check_call(nvcc_cmd)

        return output_file

# Configuration for NumPy interface
cuda_path = find_cuda()
ext_modules = [
    Pybind11Extension(
        name="curegot._internal_numpy",
        sources=sorted(glob("src/*.cpp") + glob("src/*.cu")),
        include_dirs=[
            get_cccl_include(),
            os.path.join(cuda_path, "include") if cuda_path is not None else None
        ],
        library_dirs=[
            os.path.join(cuda_path, "lib64") if cuda_path is not None else None,
            os.path.join(cuda_path, "lib") if cuda_path is not None else None,
            os.path.join(cuda_path, "lib64", "stubs") if cuda_path is not None else None,
            os.path.join(cuda_path, "lib", "stubs") if cuda_path is not None else None
        ],
        libraries=["cuda", "cudart", "cudss"],
        define_macros=[
            ("VERSION_INFO", __version__),
            ("MODULE_NAME", "_internal_numpy")
        ],
        extra_compile_args=["-O3"],
        language="c++"
    )
]
cmdclass = {"build_ext": CUDABuildExtension}

# Modify ext_modules and cmdclass if we also build PyTorch interface
if TORCH_BUILD:
    # We may also need to build source distribution on machines that do not have
    # CUDA runtime. In this case, TorchCUDAExtension() may throw exceptions, and
    # we simply skip it
    try:
        # Build both NumPy and PyTorch interface
        ext_modules.append(
            TorchCUDAExtension(
                name="curegot._internal_torch",
                sources=sorted(glob("src/*.cpp") + glob("src/*.cu")),
                include_dirs=[
                    get_cccl_include(),
                    os.path.join(cuda_path, "include") if cuda_path is not None else None,
                    os.path.join(sys.exec_prefix, "include")
                ],
                library_dirs=[
                    os.path.join(cuda_path, "lib64") if cuda_path is not None else None,
                    os.path.join(cuda_path, "lib") if cuda_path is not None else None,
                    os.path.join(cuda_path, "lib64", "stubs") if cuda_path is not None else None,
                    os.path.join(cuda_path, "lib", "stubs") if cuda_path is not None else None,
                    os.path.join(sys.exec_prefix, "lib64"),
                    os.path.join(sys.exec_prefix, "lib")
                ],
                libraries=["cuda", "cudart", "cudss"],
                define_macros=[
                    ("VERSION_INFO", __version__),
                    ("MODULE_NAME", "_internal_torch"),
                    ("TORCH_BUILD", None)
                ],
                extra_compile_args={
                    "cxx": ["-O3"],
                    "nvcc": ["-O3", "--use_fast_math"] + get_gencode_flags()
                }
            )
        )

        # Define a unified build_ext class to handle both Pybind11Extension and TorchCUDAExtension
        class UnifiedBuildExtension(CUDABuildExtension, TorchBuildExtension):
            def build_extension(self, ext):
                if ext.name.endswith("_torch"):
                    TorchBuildExtension.build_extension(self, ext)
                else:
                    CUDABuildExtension.build_extension(self, ext)

        cmdclass = {"build_ext": UnifiedBuildExtension}
    except:
        pass

setup(
    name="curegot",
    version=__version__,
    author="Yixuan Qiu",
    author_email="yixuanq@gmail.com",
    url="https://github.com/yixuan/regot-cuda",
    description="CUDA-Accelerated Regularized Optimal Transport",
    long_description="A CUDA-accelerated library for regularized optimal transport computation.",
    packages=["curegot"],
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    zip_safe=False,
    python_requires=">=3.11"
)
