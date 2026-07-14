"""
Helpers for loading the parallel transport utilities that back MARBLE geometry.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Callable

_PKG = __name__.rsplit(".", 1)[0]
_QUAL_NAME = f"{_PKG}.ptu_dijkstra"


def build_and_import_ptu_dijkstra(original_error: ModuleNotFoundError) -> ModuleType:
    """
    Attempt to import the ptu_dijkstra extension, compiling it on the fly if needed.
    """
    if _QUAL_NAME in sys.modules:
        return sys.modules[_QUAL_NAME]

    module_dir = Path(__file__).resolve().parent
    loaders: list[Callable[[Path], ModuleType | None]] = [
        _load_with_pyximport,
        _load_with_setuptools,
    ]
    for loader in loaders:
        module = loader(module_dir)
        if module is not None:
            return module

    raise ModuleNotFoundError(
        "Could not import or compile the ptu_dijkstra extension automatically. "
        "Please install Cython (pip install cython) or build the extension manually."
    ) from original_error


def _load_with_pyximport(module_dir: Path) -> ModuleType | None:
    try:
        import numpy as np
        import pyximport
    except ModuleNotFoundError:
        return None

    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

    pyximport.install(
        language_level=3,
        inplace=True,
        setup_args={"include_dirs": np.get_include()},
    )

    try:
        return importlib.import_module(_QUAL_NAME)
    except ModuleNotFoundError:
        return None


def _load_with_setuptools(module_dir: Path) -> ModuleType | None:
    source_c = module_dir / "ptu_dijkstra.c"
    if not source_c.exists():
        return None

    try:
        import numpy as np
        from setuptools import Extension
        from setuptools.command.build_ext import build_ext
        from setuptools.dist import Distribution
    except ModuleNotFoundError:
        return None

    ext = Extension(
        "ptu_dijkstra",
        sources=[str(source_c)],
        include_dirs=[np.get_include()],
    )
    dist = Distribution({"name": "ptu_dijkstra", "ext_modules": [ext]})
    cmd = build_ext(dist)
    build_dir = module_dir / "build"
    build_dir.mkdir(exist_ok=True)
    cmd.build_lib = str(module_dir)
    cmd.build_temp = str(build_dir)
    cmd.ensure_finalized()
    try:
        cmd.run()
    except Exception as exc:  # pragma: no cover - compilation failure
        raise ModuleNotFoundError(
            "Automatic compilation from ptu_dijkstra.c failed. "
            "Verify a compiler toolchain (e.g., gcc) is installed."
        ) from exc

    return importlib.import_module(_QUAL_NAME)

