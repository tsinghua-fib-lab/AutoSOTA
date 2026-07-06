from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

from pathlib import Path
import importlib.util
import sys
import sysconfig

import globals as g


_EXT_CACHE = {}


def _extension_suffix():
    """Return the Python extension suffix for the current interpreter."""
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not suffix:
        return ".so"
    return suffix


def _extension_path(name):
    """Return the expected on-disk path for a built extension module."""
    suffix = _extension_suffix()
    return g.FILE_STORAGE_PATH / "build" / name / f"{name}{suffix}"


def load_extension(name):
    """Load a prebuilt extension module or raise if missing."""
    if name in _EXT_CACHE:
        return _EXT_CACHE[name]

    path = _extension_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing built extension '{name}'. Expected {path}. Run `make kernels.build`."
        )

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load extension '{name}' from {path}.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    _EXT_CACHE[name] = module
    return module
