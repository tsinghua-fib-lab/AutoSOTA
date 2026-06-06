from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Optional, Union

import roman
from loguru import logger

from .message import BaseMessage, Conversation, as_message
from .types import ContentPart, MessageRole, String, StringFuture


class Indexing:
    """The indexing method for the printer."""

    def __init__(
        self,
        method: Optional[str] = None,
        ind: int = 0,
        prefix: str = "",
        suffix: Optional[str] = None,
    ):
        """Initialize the indexing method."""
        self._method = method
        self._ind = ind
        self._prefix = prefix
        self._suffix = suffix

    def _get_index(self, ind: int) -> str:
        if self._method is None:
            return ""
        default_suffix = ". "
        if self._method == "number":
            base = str(ind + 1)
        elif self._method in ["lower", "upper", "letter", "Letter"]:
            if ind >= 26:
                raise ValueError("Letter-based indexing method only supports 26 items.")
            base = chr(ord("A") + ind)
            if self._method in ["lower", "letter"]:
                base = base.lower()
        elif self._method in ["roman", "Roman"]:
            base = roman.toRoman(ind + 1)
            if self._method == "roman":
                base = base.lower()
        else:
            default_suffix = " "
            if self._method == "star":
                base = "*"
            elif self._method == "dash":
                base = "-"
            elif self._method.startswith("sharp"):
                base = "#" * int(self._method[5:])
            else:
                base = self._method

        return self._prefix + base + (self._suffix or default_suffix)

    def get_index(self, ind: Optional[int] = None) -> str:
        """Get the index string for the current or given index."""
        if ind is None:
            ind = self._ind
            self._ind += 1
        if ind < 0:
            raise ValueError("Indexing method does not support negative indexing.")
        return self._get_index(ind)

    def __repr__(self) -> str:
        return f"Indexing(method={self._method!r}, ind={self._ind!r}, suffix={self._suffix!r})"


@dataclass
class PrinterState:
    """A state of the printer."""

    # settings
    role: Optional[MessageRole] = None
    """The role to be used for the message."""
    separator: str = "\n"
    """The separator to be used between texts."""
    indexing: Indexing = Indexing(None, 0)
    """The indexing method to be used."""
    indent: str = ""
    """The indent to be used in the beginning of each line."""
    # inline means the first indent and indexing is inherited from the previous state
    is_inline: bool = False
    """Whether the state is inline. Inline means the first indent and
    indexing is inherited from the previous non-inline state."""
    # states
    is_start: bool = True
    """Whether the state is at the start of the scope."""
    current_sep: str = ""
    """The current separator to be used between texts."""


@dataclass
class PrinterPush:
    """A record to push a new printer state to the stack."""

    new_role: Optional[MessageRole] = None
    """The new role to be used for the message."""
    separator: Optional[str] = None
    """The separator to be used between texts."""
    indexing: Optional[Indexing] = None
    """The indexing method to be used."""
    inc_indent: str = ""
    """The increment of the indent."""
    new_indent: Optional[str] = None
    """The new indent to be used."""
    is_inline: Optional[bool] = False
    """Whether the state is inline."""


@dataclass
class PrinterPop:
    """A record to pop the last printer state from the stack."""


RecordType = Union[BaseMessage, StringFuture, ContentPart, PrinterPush, PrinterPop]
"""Types allowed in the prompt records."""


class PromptRecords:
    """A class represents a list of prompt records."""

    def __init__(self) -> None:
        """Initialize the prompt records."""
        self._records: List[RecordType] = []

    @property
    def records(self) -> List[RecordType]:
        """The list of records."""
        return self._records

    def as_convo(self) -> Conversation:
        """Convert the prompt records to a conversation."""
        return PromptPrinter()(self)

    def record(self, record: Union[str, RecordType]) -> None:
        """Record a string, message, image, audio, printer push or printer pop."""
        if isinstance(record, str):  # compatible to str
            record = StringFuture(record)
        if (
            isinstance(record, StringFuture)
            or isinstance(record, ContentPart)
            or isinstance(record, BaseMessage)
            or isinstance(record, PrinterPush)
            or isinstance(record, PrinterPop)
        ):
            self._records.append(record)
        else:
            raise ValueError("Can only record Message, PrinterPush or PrinterPop")

    def extend(self, record: "PromptRecords") -> None:
        """Extend the prompt records with another prompt records."""
        self._records.extend(record._records)

    def copy(self) -> "PromptRecords":
        """Copy the prompt records."""
        return copy.deepcopy(self)

    def __str__(self) -> str:
        return str(self.as_convo())


class PromptPrinter:
    """A class to print prompt records as conversation.

    The printer maintains a stack of printer states about the
    current role, separator, indexing, and indentation.
    """

    def __init__(
        self, states: Optional[List[PrinterState]] = None, is_newline: bool = True
    ) -> None:
        """Initialize the prompt printer."""
        if states is None:
            states = [PrinterState()]
        self._states = states
        self._is_newline = is_newline

    @property
    def states(self) -> List[PrinterState]:
        """The stack of printer states."""
        return self._states

    def push(self, data: PrinterPush) -> None:
        """Push a new printer state to the stack."""
        self._push(**data.__dict__)

    def pop(self) -> None:
        """Pop the last printer state from the stack."""
        if len(self._states) == 1:
            raise ValueError("Cannot pop the first state.")
        self._states.pop()

    def _push(
        self,
        new_role: Optional[MessageRole] = None,
        separator: Optional[str] = None,
        indexing: Optional[Indexing] = None,
        inc_indent: str = "",
        new_indent: Optional[str] = None,
        is_inline: bool = False,
    ) -> None:
        state = self.states[-1]
        if new_role is None or new_role == state.role:
            new_role = state.role
            current_separator = state.current_sep
            default_separator = state.separator  # Use the same separator as parent
            default_indexing = state.indexing  # Use the same indexing as parent
        else:  # a new role started
            logger.debug(f"new role started {new_role}")
            if len(self.states) > 1:
                raise ValueError(
                    "Cannot start a new role when there are states in the stack."
                )
            state.is_start = True  # reset the outmost state
            state.current_sep = ""  # also reset the current separator
            current_separator = ""
            default_separator = "\n"
            default_indexing = Indexing(None, 0)  # create a empty indexing
            if new_indent is None:
                new_indent = ""  # reset the indent
                if inc_indent:
                    raise ValueError(
                        "Cannot specify inc_indent when new role started. "
                        "Use new_indent instead."
                    )

        if separator is None:
            separator = default_separator
        if indexing is None:
            indexing = default_indexing
        else:
            # Avoid changing the original indexing (could be a record)
            indexing = copy.copy(indexing)
        if new_indent is None:
            new_indent = state.indent + inc_indent  # increment the indent
        elif inc_indent:
            raise ValueError("Cannot specify both inc_indent and new_indent.")

        self._states.append(
            PrinterState(
                new_role,
                separator,
                indexing,
                new_indent,
                is_inline,
                True,
                # The current separator is inherited from the parent state
                # it will change to its own separator after the first print.
                current_sep=current_separator,
            )
        )

    def _print_str(self, content: String) -> StringFuture:
        state, previous = self._states[-1], self._states[:-1]
        role = state.role
        sep = state.current_sep
        indent = state.indent
        indexing = state.indexing
        if state.is_start:  # is the first content in this scope
            state.is_start = False
            state.current_sep = state.separator
            for st in previous[::-1]:
                if st.role != role:
                    break
                if st.is_start:
                    # after first print, change the separator to its own
                    st.is_start = False
                    st.current_sep = st.separator
                else:
                    break

            if state.is_inline:
                # inline means the first indent and indexing is
                # inherited from the previous non-inline state
                for st in previous[::-1]:
                    if st.role != role:
                        break
                    if not st.is_inline:
                        # Use the first non-inline's indent and indexing
                        indent = st.indent
                        indexing = st.indexing
                        break
        if sep.endswith("\n"):
            self._is_newline = True

        s = StringFuture(sep)
        if self._is_newline:
            if indent:
                s += indent
            self._is_newline = False
        if cur_idx := indexing.get_index():
            s += cur_idx
        s += content

        # TODO: maybe check whether `s` ends with newline
        return s

    def _print_message(self, content: String) -> BaseMessage:
        """Print a string as message with the current printer state."""
        role = self._states[-1].role  # the default role within the context
        content = self._print_str(content)
        return as_message(role, content)

    def _print(
        self, contents: Union[String, ContentPart, BaseMessage, PromptRecords]
    ) -> Conversation:
        convo = Conversation(system_messages=[], messages=[])

        def handle(rec: Union[RecordType, ContentPart, str]) -> None:
            if isinstance(rec, (str, StringFuture)):
                convo.append(self._print_message(rec))
            elif isinstance(rec, ContentPart):
                convo.append(as_message(self._states[-1].role, rec))
                state = self._states[-1]
                # reset current state after image or audio, TODO: double check
                state.is_start = True
                state.current_sep = ""
            elif isinstance(rec, BaseMessage):
                convo.append(rec)
                if rec.role is not None and len(self._states) == 1:
                    # change role in the outmost state
                    state = self._states[0]
                    state.is_start = True  # reset the outmost state
                    state.current_sep = ""  # also reset the current separator
                # TODO: what should be the behavior if the role is changed
                # in states other than the outmost one?
                # should such behavior being allowed?

            elif isinstance(rec, PrinterPush):
                self.push(rec)
            elif isinstance(rec, PrinterPop):
                self.pop()
            else:
                raise ValueError(f"Unknown record type {type(rec)}")

        if isinstance(contents, PromptRecords):
            for rec in contents.records:
                handle(rec)
        else:
            handle(contents)

        return convo

    __call__ = _print
