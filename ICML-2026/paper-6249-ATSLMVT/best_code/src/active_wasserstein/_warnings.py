"""Warning filters shared by backend wrappers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import warnings


@contextmanager
def suppress_pot_warnings() -> Iterator[None]:
    """Ignore non-critical warnings emitted by POT backend calls."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
        )
        yield
