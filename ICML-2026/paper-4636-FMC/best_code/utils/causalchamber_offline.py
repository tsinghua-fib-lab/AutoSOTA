"""Offline cache for causalchamber's directory.yaml fetch.

`causalchamber.datasets.main` issues an unconditional `requests.get` for its
dataset directory at import time, which makes offline use impossible even when
all dataset files are already on disk. Calling `enable()` before importing
`causalchamber.datasets` patches `requests.get` so the directory is loaded from
a local cache when available, falling back to the network only on first use.
"""

from pathlib import Path
import os

import requests

DIRECTORY_URL = (
    "https://causalchamber.s3.eu-central-1.amazonaws.com/downloadables/directory.yaml"
)

_CACHE_PATH = Path(
    os.environ.get(
        "CAUSALCHAMBER_CACHE_DIR",
        str(Path.home() / ".cache" / "causalchamber"),
    )
) / "directory.yaml"


class _CachedResponse:
    def __init__(self, content: bytes):
        self.content = content
        self.status_code = 200


_original_get = requests.get
_enabled = False


def _patched_get(url, *args, **kwargs):
    if url == DIRECTORY_URL:
        if _CACHE_PATH.exists() and not os.environ.get("CAUSALCHAMBER_FORCE_REFRESH"):
            return _CachedResponse(_CACHE_PATH.read_bytes())
        try:
            resp = _original_get(url, *args, **kwargs)
        except requests.exceptions.RequestException:
            if _CACHE_PATH.exists():
                return _CachedResponse(_CACHE_PATH.read_bytes())
            raise
        if resp.status_code == 200:
            _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            _CACHE_PATH.write_bytes(resp.content)
        return resp
    return _original_get(url, *args, **kwargs)


def enable() -> None:
    """Patch `requests.get` to serve causalchamber's directory.yaml from cache.

    Must be called before `causalchamber.datasets` is imported anywhere in the
    process. Idempotent.
    """
    global _enabled
    if _enabled:
        return
    requests.get = _patched_get
    _enabled = True
