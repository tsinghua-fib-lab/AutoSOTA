"""Run a config file that defines CONFIG = {"entrypoint": "...", "args": {...}}."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


def _load_config(path: Path) -> dict:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load config from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "CONFIG"):
        raise ValueError(f"Config file {path} must define CONFIG dict.")
    config = getattr(module, "CONFIG")
    if not isinstance(config, dict):
        raise ValueError("CONFIG must be a dict.")
    return config


def _import_entrypoint(entrypoint: str):
    if ":" not in entrypoint:
        raise ValueError("entrypoint must be in format 'module:function'")
    mod_name, func_name = entrypoint.split(":", 1)
    module = __import__(mod_name, fromlist=[func_name])
    fn = getattr(module, func_name, None)
    if fn is None:
        raise AttributeError(f"Function {func_name} not found in {mod_name}")
    return module, fn


def _kwargs_to_argv(kwargs: dict) -> list[str]:
    argv: list[str] = []
    for key, value in kwargs.items():
        if value is None:
            continue
        opt = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                argv.append(opt)
            continue
        if isinstance(value, (list, tuple)):
            argv.extend([opt, ",".join(map(str, value))])
        elif isinstance(value, dict):
            argv.extend([opt, json.dumps(value)])
        else:
            argv.extend([opt, str(value)])
    return argv


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    parser = argparse.ArgumentParser(description="Run a config file")
    parser.add_argument("config", nargs="?", help="Path to config .py")
    parser.add_argument("--config", dest="config_flag", type=str, help="Path to config .py")
    args = parser.parse_args()

    config_value = args.config_flag or args.config
    if not config_value:
        parser.error("the following arguments are required: --config or config")
    config_path = Path(config_value)
    config = _load_config(config_path)
    entrypoint = config.get("entrypoint")
    if not entrypoint:
        raise ValueError("CONFIG must include 'entrypoint'")
    module, fn = _import_entrypoint(entrypoint)

    kwargs = config.get("args", {})
    if not isinstance(kwargs, dict):
        raise ValueError("CONFIG['args'] must be a dict")
    if config.get("pass_namespace", True):
        import argparse as _argparse
        use_cli_defaults = bool(config.get("use_cli_defaults", True))
        if use_cli_defaults and hasattr(module, "parse_args"):
            import sys as _sys
            old_argv = list(_sys.argv)
            try:
                _sys.argv = [old_argv[0], *_kwargs_to_argv(kwargs)]
                ns = module.parse_args()
            finally:
                _sys.argv = old_argv
            for key, val in kwargs.items():
                setattr(ns, key, val)
        else:
            ns = _argparse.Namespace(**kwargs)
        fn(ns)
    else:
        fn(**kwargs)


if __name__ == "__main__":
    main()
