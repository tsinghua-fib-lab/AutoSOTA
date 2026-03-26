import functools
import inspect
import os
import shutil
from typing import Any, Callable, Optional, Tuple, TypeVar

from rich.live import Live
from rich.panel import Panel
from rich.style import StyleType
from rich.syntax import Syntax

from .globals import global_vars
from .types import ParamSpec


def strip_for_continue(content: str, chars: str = " \t`{[('\"\n") -> str:
    """Strip the content so that the last part is more informative to be matched."""
    return content.rstrip(chars)


def split_last(content: str, split_marker: str = "\n") -> Tuple[Optional[str], str]:
    """Split the content at the last split marker, return content before and after the last marker.

    If the split marker is not found in the content, return None and the original content.

    Args:
        content: The content to split.
        split_marker: The marker to split the content at.

    Returns:
        A tuple of the content before and after the last marker.
        The first element could be None when the split marker is not found.
    """
    if split_marker not in content:
        return None, content
    result = content.rsplit(split_marker, 1)
    assert len(result) == 2
    return result[0], result[1]


def make_panel(
    content: str,
    *,
    title: Optional[str] = None,
    style: StyleType = "none",
    lexer: Optional[str] = None,
    truncate: bool = False,
) -> Panel:
    """Display a panel in the terminal with the given title and content."""
    if truncate:
        terminal_height = shutil.get_terminal_size().lines
        content = os.linesep.join(content.splitlines()[-terminal_height + 2 :])

    rich_configs = global_vars.configs.settings.logging.display.rich
    syntax = Syntax(
        content,
        lexer=lexer or rich_configs.lexer,
        theme=rich_configs.theme,
        line_numbers=rich_configs.line_numbers,
        word_wrap=rich_configs.word_wrap,
    )
    return Panel(syntax, title=title, style=style)


def get_live() -> Live:
    """Get the live object, create one if not exists."""
    with global_vars.live_lock:
        if global_vars.live is None:
            refresh_per_second = (
                global_vars.configs.settings.logging.display.rich.refresh_per_second
            )
            global_vars.live = Live(
                make_panel(
                    "Waiting for Response ...",
                    title="APPL Streaming",
                    style="magenta",
                ),
                refresh_per_second=refresh_per_second,
            )
            global_vars.live.start()
        return global_vars.live


def stop_live() -> None:
    """Stop the live object."""
    with global_vars.live_lock:
        if global_vars.live is not None:
            global_vars.live.stop()
            global_vars.live = None


P = ParamSpec("P")
T = TypeVar("T")
F = TypeVar("F", bound=Callable)  # function
R = TypeVar("R")  # return value
APPL_NEED_CTX_ATTR = "__need_ctx__"


def _copy_appl_attrs(func: Callable, new_func: Callable) -> None:
    for attr in [APPL_NEED_CTX_ATTR]:
        if hasattr(func, attr):
            setattr(new_func, attr, getattr(func, attr))


def need_ctx(func: Callable[P, T]) -> Callable[P, T]:
    """Decorate a function to mark it as needing a prompt context."""
    setattr(func, APPL_NEED_CTX_ATTR, True)
    return func


def partial(func: Callable[..., R], *args: Any, **kwargs: Any) -> Callable[..., R]:
    """Create a new function with partial application of the given arguments and keywords."""
    new_func = functools.partial(func, *args, **kwargs)
    _copy_appl_attrs(func, new_func)
    return new_func


def wraps(func: F) -> Callable[[Callable], F]:
    """Replace the functools.wraps to take care of the type hint."""

    def decorator(wrapper: Callable) -> F:
        new_wrapper = functools.wraps(func)(wrapper)
        _copy_appl_attrs(func, new_wrapper)
        return new_wrapper  # type: ignore

    return decorator


def get_source_code(func: Callable) -> Optional[str]:
    """Get the source code of a function."""
    try:
        return inspect.getsource(func)
    except OSError:
        return None
