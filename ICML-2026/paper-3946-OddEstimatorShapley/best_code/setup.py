"""Setup script for shapiq package with C extensions."""

from __future__ import annotations

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext

try:
    from Cython.Build import cythonize

    HAVE_CYTHON = True
except ImportError:
    HAVE_CYTHON = False


# Extend the default build_ext class to bootstrap numpy installation
# that are needed to build C extensions.
# see https://stackoverflow.com/questions/19919905/how-to-bootstrap-numpy-installation-in-setup-py
class BuildExt(_build_ext):
    """Custom build_ext command to include numpy headers."""

    def finalize_options(self) -> None:
        """Finalize options and set numpy setup flag."""
        _build_ext.finalize_options(self)
        if isinstance(__builtins__, dict):
            __builtins__["__NUMPY_SETUP__"] = False
        else:
            __builtins__.__NUMPY_SETUP__ = False
        import numpy as np

        self.include_dirs.append(np.get_include())


ext_modules = [
    Extension(
        "oddshap.linear._cext",
        # source path must include the src/ prefix because package uses src layout
        sources=["src/oddshap/linear/cext/_cext.cc"],
    ),
    Extension(
        "oddshap.interventional._cext",
        sources=["src/oddshap/interventional/cext/_cext.cc"],
        language="c",
        extra_compile_args=["-O3"],
    ),
    Extension(
        "oddshap.interventional.cpp_implementation",
        sources=[
            "src/oddshap/interventional/cpp_implementation/cext.cc",
            "src/oddshap/interventional/cpp_implementation/conversion.cpp",
            "src/oddshap/interventional/cpp_implementation/interactions.cpp",
        ],
        language="c++",
        extra_compile_args=["-O3"],
    ),
]

# Add Cython extension if available
if HAVE_CYTHON:
    ext_modules.append(
        Extension(
            "oddshap.interventional._interventional",
            sources=["src/oddshap/interventional/_interventional.pyx"],
            language="c++",
            extra_compile_args=["-O3"],
        )
    )

# Cythonize if available
if HAVE_CYTHON:
    ext_modules = cythonize(
        ext_modules,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "initializedcheck": False,
            "profile": True,
            "embedsignature": True,
        },
        **{
            "annotate": True,
            "gdb_debug": True,
        },
    )

setup(
    name="oddshap_paper",
    ext_modules=ext_modules,
    setup_requires=["numpy", "cython", "scipy"],
    cmdclass={"build_ext": BuildExt},
)
