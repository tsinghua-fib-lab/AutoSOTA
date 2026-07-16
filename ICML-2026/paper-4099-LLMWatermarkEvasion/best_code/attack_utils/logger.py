# From https://github.com/xy3/print-color/blob/main/print_color/print_color.py

import builtins
import sys
import typing
from typing import Any, Optional

import time

__all__ = ["print"]


class PrintColor:
    colors = {
        "purple": "\033[95m",
        "blue": "\033[94m",
        "green": "\033[92m",
        "yellow": "\033[33m",
        "red": "\033[31m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "black": "\033[30m",
        "white": "\033[00m",
    }

    # Abbreviations and aliases
    colors["v"] = colors["purple"]  # v for violet
    colors["p"] = colors["purple"]
    colors["b"] = colors["blue"]
    colors["g"] = colors["green"]
    colors["y"] = colors["yellow"]
    colors["r"] = colors["red"]
    colors["m"] = colors["magenta"]
    colors["c"] = colors["cyan"]
    colors["k"] = colors["black"]
    colors["w"] = colors["white"]

    backgrounds = {
        "grey": "\033[40m",
        "red": "\033[41m",
        "green": "\033[42m",
        "yellow": "\033[43m",
        "blue": "\033[44m",
        "magenta": "\033[45m",
        "cyan": "\033[46m",
        "white": "\033[47m",
    }

    # Abbreviations and aliases
    backgrounds["gray"] = backgrounds["grey"]
    backgrounds["gr"] = backgrounds["grey"]
    backgrounds["r"] = backgrounds["red"]
    backgrounds["g"] = backgrounds["green"]
    backgrounds["y"] = backgrounds["yellow"]
    backgrounds["b"] = backgrounds["blue"]
    backgrounds["m"] = backgrounds["magenta"]
    backgrounds["c"] = backgrounds["cyan"]
    backgrounds["w"] = backgrounds["white"]

    formats = {"bold": "\033[1m", "underline": "\033[4m", "blink": "\033[5m"}

    def __init__(self, *args: Any, **kwargs: Any):
        self.args = args
        self.kwargs = kwargs

    def print(self) -> None:
        color = self.kwargs.pop("color", None)
        if not color:
            color = self.kwargs.pop("colour", None)
        back = self.kwargs.pop("background", None)
        format = self.kwargs.pop("format", None)
        tag = self.kwargs.pop("tag", None)
        tag_color = self.kwargs.pop("tag_color", None)
        if not tag_color:
            tag_color = self.kwargs.pop("tag_colour", None)
        output = self.kwargs.get('file', sys.stdout)
        use_color = hasattr(output, 'isatty') and output.isatty()
        result = "¬".join(str(arg) for arg in self.args)

        if color and use_color:
            result = self.color(color) + result

        if tag:
            result = f"[{tag}] {result}"
            if tag_color and use_color:
                result = self.color(tag_color) + result
        if back and use_color:
            builtins.print(self.background(back), file=sys.stdout, end="")
        if format and use_color:
            builtins.print(self.format(format), file=sys.stdout, end="")
        if use_color:
            result += self.end
        builtins.print(*result.split("¬"), **self.kwargs)

    def color(self, color: str) -> str:
        return self.colors.get(color, self.default_color)

    def background(self, back: str) -> str:
        return self.backgrounds.get(back, self.default_color)

    def format(self, fmt: str) -> str:
        if isinstance(fmt, str):
            return self.formats.get(fmt, self.default_color)
        elif isinstance(fmt, list) or isinstance(fmt, tuple):
            return "".join([f for f in [self.formats.get(f, "") for f in fmt]])

    @property
    def end(self) -> str:
        return "\033[0m"

    @property
    def default_color(self) -> str:
        return "\033[0m"


Color = typing.Literal[
    "purple",
    "blue",
    "green",
    "yellow",
    "red",
    "magenta",
    "yan",
    "black",
    "white",
    "v",
    "p",
    "b",
    "g",
    "y",
    "r",
    "m",
    "c",
    "k",
    "w",
]

Background = typing.Literal[
    "grey",
    "red",
    "green",
    "yellow",
    "blue",
    "magenta",
    "cyan",
    "white",
    "gray",
    "gr",
    "r",
    "g",
    "y",
    "b",
    "m",
    "c",
    "w",
]

Format = typing.Literal["bold", "underline", "blink"]


_T_contra = typing.TypeVar("_T_contra", contravariant=True)


class SupportsWrite(typing.Protocol[_T_contra]):
    def write(self, __s: _T_contra) -> typing.Any: ...


@typing.no_type_check
def print(
    *values: object,
    sep: str = " ",
    end: str = "\n",
    file: Optional[SupportsWrite[str]] = None,
    flush: bool = False,
    color: Optional[Color] = None,
    background: Optional[Background] = None,
    format: Optional[Format] = None,
    tag: Optional[str] = None,
    tag_color: Optional[Color] = None,
    **kwargs: Any,
) -> None:
    printcolor = PrintColor(
        *values,
        sep=sep,
        end=end,
        file=file,
        flush=flush,
        color=color,
        background=background,
        format=format,
        tag=tag,
        tag_color=tag_color,
        **kwargs,
    )
    printcolor.print()


class ProgressLogger:
    last_label = ""
    last_start = -1.0

    @staticmethod
    def start(label: str) -> None:
        ProgressLogger.last_label = label
        ProgressLogger.last_start = time.time()
        print(f"{label}... ⏳", end="\r", flush=True)

    @staticmethod
    def stop() -> None:
        timediff = time.time() - ProgressLogger.last_start
        print(f"{ProgressLogger.last_label} DONE. ✅ Took {timediff:.2f}s.")
