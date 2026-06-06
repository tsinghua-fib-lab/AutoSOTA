from __future__ import annotations

import copy
import traceback
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from types import TracebackType
from typing import Optional, TypeVar, Union

from typing_extensions import override

from .context import PromptContext
from .printer import Indexing, PrinterPop, PrinterPush, PromptPrinter, PromptRecords
from .types import MessageRole


class PrinterModifier(AbstractContextManager):
    """The contextual compositor of the prompt printer.

    Controls the behavior of the prompt printer within the context manager.
    Should only be used within the APPL function.
    """

    __need_ctx__: bool = True

    def __init__(self, _ctx: Optional[PromptContext] = None):
        """Initialize the PrinterModifier object.

        Args:
            _ctx: The prompt context filled automatically by the APPL function.
        """
        self._ctx = _ctx

    @property
    def push_args(self) -> PrinterPush:
        """The arguments to push to the printer."""
        raise NotImplementedError

    def _enter(self) -> None:
        if self._ctx is None:
            raise ValueError(
                "Context is not provided, did you forget mark your function with @ppl?"
            )
        self._ctx.push_printer(self.push_args)

    def _exit(
        self,
        _exc_type: Optional[type[BaseException]],
        _exc_value: Optional[BaseException],
        _traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        if _exc_type:
            # logger.debug(f"Exception occurred: {__exc_value}")
            traceback.print_tb(_traceback)
            return False
        if self._ctx is not None:
            self._ctx.pop_printer()
        return None

    def __enter__(self) -> "PrinterModifier":
        self._enter()
        return self

    def __exit__(
        self,
        __exc_type: Optional[type[BaseException]],
        __exc_value: Optional[BaseException],
        __traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        return self._exit(__exc_type, __exc_value, __traceback)


class Compositor(PrinterModifier):
    """The contextual compositor of the prompts.

    This class represents a contextual compositor that modifies the behavior of the printer.
    It provides various options for customizing the output format of the prompts within this context manager.

    Attributes:
        _sep: The class default separator string is None, indicating inherit from the parent.
        _indexing: The class default indexing mode is empty indexing.
        _inc_indent: The class default indentation string is empty string.
        _new_indent: The class default new indentation string is None, indicating not overwrite the indent.
        _is_inline: The class default inline flag is False.
        _new_role: The class default role of the modifier is None, indicating not overwrite the role.
    """

    # The default value of the printer modifier
    _sep: Optional[str] = None  # Default: inherit from the parent
    _indexing: Indexing = Indexing()  # Default: empty indexing
    _inc_indent: str = ""  # Default: no delta indent
    _new_indent: Optional[str] = None  # Default: not overwrite the indent
    _is_inline: bool = False  # Default: not inline
    _new_role: Optional[MessageRole] = None  # Default: not overwrite the role

    def __init__(
        self,
        sep: Optional[str] = None,
        indexing: Union[Indexing, Optional[str]] = None,
        indent: Optional[Union[str, int]] = None,
        new_indent: Optional[Union[str, int]] = None,
        is_inline: Optional[bool] = None,
        role: Optional[MessageRole] = None,
        _ctx: Optional[PromptContext] = None,
    ):
        """Initialize the Compositor object.

        Args:
            sep:
                The separator string. Defaults to use the class default.
            indexing:
                The indexing mode. Defaults to use the class default.
            indent:
                The indentation string. Defaults to use the class default.
            new_indent:
                The new indentation string. Defaults to use the class default.
            is_inline:
                Flag indicating if the modifier is inline. Defaults to use the class default.
            role:
                The role of the modifier. Defaults to use the class default.
            _ctx: The prompt context filled automatically by the APPL function.
        """
        super().__init__(_ctx)
        if sep is not None:
            self._sep = sep
        if indexing is not None:
            if isinstance(indexing, str):
                indexing = Indexing(indexing)
            self._indexing = indexing
        else:
            if self._indexing is None:
                raise ValueError("Indexing must be provided.")
            self._indexing = copy.copy(self._indexing)
            # copy to avoid changing the class default
        if indent is not None:
            if isinstance(indent, int):
                indent = " " * indent
            self._inc_indent = indent
        if new_indent is not None:
            if isinstance(new_indent, int):
                new_indent = " " * new_indent
            self._new_indent = new_indent
        if is_inline is not None:
            self._is_inline = is_inline
        if role is not None:
            self._new_role = role

    @override
    @property
    def push_args(self) -> PrinterPush:
        return PrinterPush(
            self._new_role,
            self._sep,
            self._indexing,
            self._inc_indent,
            self._new_indent,
            self._is_inline,
        )

    def __enter__(self) -> "Compositor":
        super().__enter__()
        return self


# Essential for compile
class ApplStr(Compositor):
    """A compositor that represents a string in the prompt.

    Attributes:
        _sep: The class default separator string is an empty string.
        _new_indent: The class default new indentation string is an empty string.
        _is_inline: The class default inline flag is True.

    Examples:
        ```py
        >>> with ApplStr():
        ...     "Hello, "
        ...     "world!"
        <<< The prompt will be:
        Hello, world!
        ```
    """

    _sep = ""
    _new_indent = ""
    _is_inline = True
