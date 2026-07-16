from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='hash_cpp', ext_modules=[cpp_extension.CUDAExtension('hash_cpp', ['hash.cpp', 'hash_kernel.cu'], libraries=["ssl", "crypto"], library_dirs = ["/usr/lib"])], cmdclass={'build_ext': cpp_extension.BuildExtension})
