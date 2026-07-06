from gitbud.gitbud import inject_repo_into_sys_path
inject_repo_into_sys_path()

import dataclasses

import globals as g


def apply_env_overrides(cfg):
    """Return a config with environment variable overrides applied (use sparingly)."""
    if not dataclasses.is_dataclass(cfg):
        return cfg

    updates = {}

    k_value = g.GET_ENV_VAR(g.ENV_OVERRIDE_K)
    if k_value is not None and hasattr(cfg, "k"):
        updates["k"] = int(k_value)

    s_value = g.GET_ENV_VAR(g.ENV_OVERRIDE_S)
    if s_value is not None and hasattr(cfg, "s"):
        updates["s"] = int(s_value)

    dtype_value = g.GET_ENV_VAR(g.ENV_OVERRIDE_DTYPE)
    if dtype_value is not None and hasattr(cfg, "dtype"):
        if dtype_value != g.DTYPE_FP32:
            raise ValueError(f"Only dtype {g.DTYPE_FP32} is supported.")
        updates["dtype"] = dtype_value

    if not updates:
        return cfg

    return dataclasses.replace(cfg, **updates)
