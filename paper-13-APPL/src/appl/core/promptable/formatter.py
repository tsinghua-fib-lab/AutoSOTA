from abc import ABCMeta
from typing import Any, Optional

from ..types.futures import String


# ABCMeta is required since Promptable is an abstract class
class FormatterMeta(ABCMeta):
    """Metaclass for classes that can be formatted."""

    def _get_format_str(cls):
        if fstr := getattr(cls, "fstr", None):
            return fstr
        return getattr(cls, "__format_str__", "{}")

    def _get_format_name(cls):
        if name := getattr(cls, "name", None):
            return name
        if docstr := getattr(cls, "__doc__", None):
            return docstr
        return getattr(cls, "__format_name__", cls.__name__)

    def __format__(cls, format_spec: str) -> str:
        fstr = cls._get_format_str()
        name = cls._get_format_name()
        return fstr.format(name, format_spec=format_spec)

    def __repr__(cls):
        return cls._get_format_name()


class Formattable(metaclass=FormatterMeta):
    """Base class for class objects that can be formatted.

    Example:
        ```py
        >>> class Example(Formattable):
        ...     fstr: str = "[{}]"
        ...     name: str = "example"
        >>> print(f"{Example}")
        [example]
        ```
    """

    fstr: str
    name: Optional[String]
