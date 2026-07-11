from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.traceback import install as _install_rich_traceback

_CONSOLE: Console | None = None


def get_console() -> Console:
    """Return a shared Rich Console instance."""

    global _CONSOLE
    if _CONSOLE is None:
        _CONSOLE = Console()
    return _CONSOLE


def install_rich_traceback(*, show_locals: bool = False, **kwargs: Any) -> None:
    """Install Rich traceback handler for prettier exceptions."""

    _install_rich_traceback(console=get_console(), show_locals=show_locals, **kwargs)


def get_progress(*, transient: bool = True) -> Progress:
    """Create a Rich Progress instance with spinner, bar, percent and ETA."""

    return Progress(
        SpinnerColumn(style="progress.spinner"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=get_console(),
        transient=transient,
    )
