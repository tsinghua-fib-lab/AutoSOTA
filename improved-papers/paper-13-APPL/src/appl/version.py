from importlib.metadata import version

try:
    __version__ = version("applang")
except Exception:
    __version__ = "unknown"
