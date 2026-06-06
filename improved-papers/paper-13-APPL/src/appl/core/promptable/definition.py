from typing import Any, List, Optional

from ..types.futures import String
from .base import Promptable
from .formatter import Formattable


class Definition(Promptable, Formattable):
    """Represent a definition of a concept.

    Attributes:
        fstr: The format string for the definition.
        name: The name of the definition.
        desc: A description of the definition.
        _forks: A list of all instances of this class.
    """

    fstr: str = "{}"
    name: Optional[String] = None
    desc: String = ""
    _forks: List["Definition"] = []

    def __init__(
        self,
        name: Optional[String] = None,
        desc: Optional[String] = None,
        *,
        sep: String = ": ",
        details: Any = None,
        fstr: Optional[str] = None,
        var_name: Optional[str] = None,
    ):
        """Initialize the Definition with the given name and description.

        Args:
            name: The name of the definition.
            desc: A description of the definition.
            sep: The separator between the name and description.
            details: Additional details about the definition.
            fstr: The format string for the definition.
            var_name: The name of the variable that the definition is stored in.
        """
        self.name = name or self.name or self.__doc__
        if self.name is None:
            raise ValueError("Name must be provided for Definition.")

        if desc is not None:
            self.desc = desc
        self.sep = sep
        self.details = details
        if fstr is not None:
            self.fstr = fstr
        self.var_name = var_name or self.name

        self._forks.append(self)

    def __hash__(self) -> int:
        return hash(self.var_name)

    def __prompt__(self) -> List[Any]:
        # Used when add the Definition to the prompt
        desc = self.desc or ""
        res = [f"{self.name}{self.sep}{desc}"]
        if self.details is not None:
            res.append(self.details)
        return res

    def __repr__(self):
        # Used to display information
        desc = self.desc or ""
        s = f"{self.name}{self.sep}{desc}"
        if self.details is not None:
            s += f"\n{self.details}"
        return s

    def __str__(self):
        # Used for reference in the prompt
        return self.fstr.format(self.name)


class BracketedDefinition(Definition):
    """A Definition that is formatted with square brackets."""

    fstr = "[{}]"


def define(def_name: str, format_str: str = "{}") -> type:
    """Create a new Definition subclass with the given name and format string."""

    class CustomDef(Definition):
        name = def_name
        fstr = format_str

    return CustomDef


def define_bracketed(def_name: str) -> type:
    """Create a new BracketedDefinition subclass with the given name."""

    class CustomDef(BracketedDefinition):
        name = def_name

    return CustomDef
