from abc import ABC, abstractmethod
from typing import Any

from ..types.futures import StringFuture


class Promptable(ABC):
    """Interface for objects that can be converted to a prompt string."""

    @abstractmethod
    def __prompt__(self) -> Any:
        """Convert the object to a prompt object."""
        raise NotImplementedError


def promptify(obj: Any) -> Any:
    """Convert an object to a prompt object if it is promptable."""
    if isinstance(obj, Promptable):
        s = obj.__prompt__()
        if isinstance(s, str):
            s = StringFuture(s)
        return s

    return StringFuture(str(obj))
