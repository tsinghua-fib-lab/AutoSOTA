"""Console encoding helpers shared by the entry-point scripts."""

from __future__ import annotations

import sys
from typing import Any


def force_utf8_stdout_stderr() -> None:
    """Best-effort switch of ``stdout``/``stderr`` to UTF-8.

    Progress output contains box-drawing and arrow characters. On a Windows
    console defaulting to cp1252 those raise ``UnicodeEncodeError`` mid-run,
    which kills a loop after real work has already been done; a cp1252 file
    redirection crashes identically, so this is deliberately not gated on
    ``isatty()``. It is a no-op for streams that are already UTF-8 or that
    expose an incompatible ``reconfigure()``.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure: Any = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        if (getattr(stream, "encoding", "") or "").lower().replace("-", "").replace("_", "") == "utf8":
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError, TypeError):
            pass
