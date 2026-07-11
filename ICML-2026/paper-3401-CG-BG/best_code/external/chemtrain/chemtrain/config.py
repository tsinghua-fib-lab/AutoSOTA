"""Minimal global configuration.

- `update(name, value)` or `update(**kwargs)` mutates global config.
- `read(name)` returns the current value.

Currently supported keys:
- `async_dataloading` (bool): enable/disable threaded host-side batch prefetch.
"""

from __future__ import annotations

from typing import Any, Dict


_CONFIG: Dict[str, Any] = {
    "async_dataloading": False,
}

def update(name: str | None = None, value: Any | None = None, /, **kwargs: Any) -> None:
    """Update configuration.

    Supports either:
      - `update("flag_name", flag_value)`
      - `update(flag_name=flag_value, ...)`

    Returns:
      None (like `jax.config.update`).
    """
    if name is not None:
        if value is None:
            raise TypeError("config.update(name, value) missing required 'value'")
        _CONFIG[name] = value

    if kwargs:
        _CONFIG.update(kwargs)


def read(name: str, default: Any | None = None) -> Any:
    """Read a configuration value."""
    return _CONFIG.get(name, default)
