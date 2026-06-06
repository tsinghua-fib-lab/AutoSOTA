"""Containg the compositor classes.

All examples shows the composed prompt in APPL functions.
"""

from __future__ import annotations

from types import TracebackType
from typing import Any, Dict, Iterable, Optional, Union

from .const import INDENT4 as INDENT
from .core import ApplStr, Compositor, Indexing, PromptContext
from .func import need_ctx


class LineSeparated(Compositor):
    r"""The line separated compositor.

    Attributes:
        _sep: The class default separator is "\n".

    Example:
        ```py
        >>> with LineSeparated():
        ...     "item1"
        ...     "item2"
        <<< The prompt will be:
        item1
        item2
        ```
    """

    _sep = "\n"


class DoubleLineSeparated(Compositor):
    r"""The double line separated compositor.

    Attributes:
        _sep: The class default separator is "\n\n".

    Example:
        ```py
        >>> with DoubleLineSeparated():
        ...     "item1"
        ...     "item2"
        <<< The prompt will be:
        item1

        item2
        ```
    """

    _sep = "\n\n"


class NoIndent(LineSeparated):
    """The list compositor with no indentation.

    Attributes:
        _inc_indent: The class default indentation is "".

    Example:
        ```py
        >>> with IndentedList():
        ...     with NoIndent():
        ...         "item1"
        ...     "item2"
        <<< The prompt will be:
        item1
            item2
        ```
    """

    _new_indent = ""


class IndentedList(LineSeparated):
    """The indented list compositor.

    Attributes:
        _inc_indent: The class default indentation is INDENT.

    Example:
        ```py
        >>> "BEGIN"
        ... with IndentedList():
        ...     "item1"
        ...     "item2"
        <<< The prompt will be:
        BEGIN
            item1
            item2
        ```
    """

    _inc_indent = INDENT


class NumberedList(LineSeparated):
    """The number list compositor.

    Attributes:
        _indexing: The class default indexing mode is "number".

    Example:
        ```py
        >>> with NumberedList():
        ...     "item1"
        ...     "item2"
        <<< The prompt will be:
        1. item1
        2. item2
        ```
    """

    _indexing = Indexing("number")


class LowerLetterList(LineSeparated):
    """The lower letter list compositor.

    Attributes:
        _indexing: The class default indexing mode is "lower".

    Example:
        ```py
        >>> with LowerLetterList():
        ...     "item1"
        ...     "item2"
        <<< The prompt will be:
        a. item1
        b. item2
        ```
    """

    _indexing = Indexing("lower")


class UpperLetterList(LineSeparated):
    """The upper letter list compositor.

    Attributes:
        _indexing: The class default indexing mode is "upper".

    Example:
        ```py
        >>> with UpperLetterList():
        ...     "item1"
        ...     "item2"
        <<< The prompt will be:
        A. item1
        B. item2
        ```
    """

    _indexing = Indexing("upper")


class LowerRomanList(LineSeparated):
    """The lower roman list compositor.

    Attributes:
        _indexing: The class default indexing mode is "roman".

    Example:
        ```py
        >>> with LowerRomanList():
        ...     "item1"
        ...     "item2"
        <<< The prompt will be:
        i. item1
        ii. item2
        ```
    """

    _indexing = Indexing("roman")


class UpperRomanList(LineSeparated):
    """The upper roman list compositor.

    Attributes:
        _indexing: The class default indexing mode is "Roman".

    Example:
        ```py
        >>> with UpperRomanList():
        ...     "item1"
        ...     "item2"
        <<< The prompt will be:
        I. item1
        II. item2
        ```
    """

    _indexing = Indexing("Roman")


class DashList(LineSeparated):
    """The dash list compositor.

    Attributes:
        _indexing: The class default indexing mode is "dash".

    Example:
        ```py
        >>> with DashList():
        ...     "item1"
        ...     "item2"
        <<< The prompt will be:
        - item1
        - item2
        ```
    """

    _indexing = Indexing("dash")


class StarList(LineSeparated):
    """The star list compositor.

    Attributes:
        _indexing: The class default indexing mode is "star".

    Example:
        ```py
        >>> with StarList():
        ...     "item1"
        ...     "item2"
        <<< The prompt will be:
        * item1
        * item2
        ```
    """

    _indexing = Indexing("star")


LetterList = UpperLetterList
"""The alias of UpperLetterList."""
RomanList = UpperRomanList
"""The alias of UpperRomanList."""


class Logged(LineSeparated):
    """The logged compositor, which is used to wrap the content with logs.

    Note the indent will also apply to the prolog and epilog.

    Attributes:
        _indent_inside:
            The class default indentation inside prolog and epilog is "".

    Example:
        ```py
        >>> with Logged(prolog="BEGIN", epilog="END"):
        ...     "item1"
        ...     "item2"
        <<< The prompt will be:
        BEGIN
        item1
        item2
        END
        ```
    """

    _indent_inside: Optional[str] = ""

    def __init__(
        self,
        *args: Any,
        prolog: str,
        epilog: str,
        indent_inside: Union[str, int, None] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the logged compositor.

        Args:
            *args: The arguments.
            prolog: The prolog string.
            epilog: The epilog string.
            indent_inside: The indentation inside the prolog and epilog.
            **kwargs: The keyword arguments.
        """
        self._prolog = prolog
        self._epilog = epilog
        if isinstance(indent_inside, int):
            indent_inside = " " * indent_inside
        if indent_inside is not None:
            if self._indent_inside is None:
                raise ValueError(
                    "Indentation inside is not allowed for this compositor."
                )
            self._indent_inside = indent_inside
        outer_indent = kwargs.pop("indent", None)
        super().__init__(indent=outer_indent, _ctx=kwargs.get("_ctx"))
        kwargs = self._get_kwargs_for_inner(kwargs)
        # The arguments are passed to the inner compositor
        self._indent_compositor = LineSeparated(*args, **kwargs)

    @property
    def prolog(self) -> str:
        """The prolog string."""
        return self._prolog

    @property
    def epilog(self) -> str:
        """The epilog string."""
        return self._epilog

    def _get_kwargs_for_inner(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        kwargs["indent"] = self._indent_inside
        return kwargs

    def _enter(self) -> None:
        super()._enter()
        if self._ctx is not None:
            self._ctx.add_string(self.prolog)
            self._indent_compositor.__enter__()

    def _exit(
        self,
        _exc_type: Optional[type[BaseException]],
        _exc_value: Optional[BaseException],
        _traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        if not _exc_type:
            if self._ctx is not None:
                self._indent_compositor.__exit__(None, None, None)
                self._ctx.add_string(self.epilog)
        else:
            if self._ctx is not None:
                self._indent_compositor.__exit__(_exc_type, _exc_value, _traceback)
        return super()._exit(_exc_type, _exc_value, _traceback)


class Tagged(Logged):
    """The tagged compositor, which is used to wrap the content with a tag.

    Note the indent will also applyt to the tag indicator.

    Attributes:
        _indent_inside:
            The class default indentation inside prolog and epilog is 4 spaces.

    Example:
        ```py
        >>> with Tagged("div", indent_inside=4):
        ...     "item1"
        ...     "item2"
        <<< The prompt will be:
        <div>
            item1
            item2
        </div>
        ```
    """

    _indent_inside: Optional[str] = ""

    def __init__(
        self,
        tag: str,
        *args: Any,
        attrs: Optional[Dict[str, str]] = None,
        tag_begin: str = "<{}{}>",
        tag_end: str = "</{}>",
        indent_inside: Union[str, int, None] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the tagged compositor.

        Args:
            tag: The tag name.
            *args: The arguments.
            attrs: The attributes of the tag.
            tag_begin: The format of tag begin string.
            tag_end: The format of tag end string.
            indent_inside: The indentation inside the tag.
            **kwargs: The keyword arguments.
        """
        self._tag = tag
        self._attrs = attrs
        self._tag_begin = tag_begin
        self._tag_end = tag_end
        prolog = tag_begin.format(tag, self.formated_attrs)
        epilog = tag_end.format(tag)
        super().__init__(
            *args, prolog=prolog, epilog=epilog, indent_inside=indent_inside, **kwargs
        )

    @property
    def formated_attrs(self) -> str:
        """The formatted attributes of the tag."""
        if self._attrs is None:
            return ""
        return " " + " ".join(f'{k}="{v}"' for k, v in self._attrs.items())


class InlineTagged(Tagged):
    """The inline tagged compositor, which is used to wrap the content with a tag.

    Attributes:
        _sep: The class default separator is "".
        _indexing: The class default indexing mode is no indexing.
        _new_indent: The class default indentation is "".
        _is_inline: The class default is True.
        _indent_inside: This class does not support indentation inside.

    Example:
        ```py
        >>> with InlineTagged("div", sep=","):
        ...     "item1"
        ...     "item2"
        <<< The prompt will be:
        <div>item1,item2</div>
        ```
    """

    def _get_kwargs_for_inner(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        # pass the arguments to the inner compositor
        kwargs["sep"] = kwargs.get("sep", self._sep)
        kwargs["indexing"] = kwargs.get("indexing", self._indexing)
        kwargs["new_indent"] = self._new_indent
        kwargs["is_inline"] = self._is_inline
        return kwargs

    _sep = ""
    _indexing = Indexing()
    _new_indent = ""
    _is_inline = True
    _indent_inside: Optional[str] = None


@need_ctx
def iter(
    lst: Iterable,
    compositor: Optional[Compositor] = None,
    _ctx: Optional[PromptContext] = None,
) -> Iterable:
    """Iterate over the iterable list with the compositor.

    Example:
        ```py
        >>> items = ["item1", "item2"]
        >>> for i in iter(items, NumberedList()):
        ...     i
        <<< The prompt will be:
        1. item1
        2. item2
        ```
    """
    # support tqdm-like context manager
    if compositor is None:
        compositor = NumberedList(_ctx=_ctx)

    entered = False
    try:
        for i in lst:
            if not entered:
                entered = True
                compositor.__enter__()
            yield i
    except Exception as e:
        # TODO: check the impl here
        if entered:
            if not compositor.__exit__(type(e), e, e.__traceback__):
                raise e
        else:
            raise e
    finally:
        if entered:
            compositor.__exit__(None, None, None)
