import sys
from importlib import import_module


def _alias(old_name: str, new_name: str) -> None:
    if old_name in sys.modules:
        return
    try:
        sys.modules[old_name] = import_module(new_name)
    except Exception:
        # If the target module can't be imported, silently ignore.
        # This keeps runtime behavior unchanged when the alias isn't needed.
        pass


_alias("models", "src.models")

