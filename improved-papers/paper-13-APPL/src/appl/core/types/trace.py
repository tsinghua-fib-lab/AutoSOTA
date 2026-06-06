import time
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, model_validator


class TraceEventBase(BaseModel):
    """A base class for trace events."""

    name: str
    """The name of the event."""
    time_stamp: float = None  # type: ignore
    """The time stamp of the event."""
    metadata: Optional[Dict] = None
    """The meta data of the event."""

    @model_validator(mode="after")
    def _check_time_stamp(self) -> "TraceEventBase":
        if self.time_stamp is None:
            # Set the time stamp to the current time if it is not set
            self.time_stamp = time.time()  # type: ignore
        return self


class FunctionCallEvent(TraceEventBase):
    """A class representing a function call event."""

    args: Dict
    """The arguments of the function call."""
    parent_func: Optional[str] = None
    """The name of the parent function."""


class FunctionReturnEvent(TraceEventBase):
    """A class representing a function return event."""

    ret: Any = None
    """The return value of the function."""


class GenerationInitEvent(TraceEventBase):
    """A class representing a generation init event."""

    parent_func: Optional[str] = None
    """The name of the parent function."""


class GenerationResponseEvent(TraceEventBase):
    """A class representing a generation response event."""

    args: Dict
    """The arguments of the generation call."""
    ret: Any
    """The return value of the generation call."""


class CompletionRequestEvent(TraceEventBase):
    """A class representing a completion request event."""

    parent_func: Optional[str] = None
    """The name of the parent function."""


class CompletionResponseEvent(TraceEventBase):
    """A class representing a completion response event."""

    args: Dict
    """The arguments of the completion call."""
    ret: Any
    """The return value of the completion call."""
    cost: Optional[float]
    """The api cost of the completion call."""


class TraceNode(BaseModel):
    """The node of a trace tree containing information about trace events."""

    type: str
    """The type of the trace node."""
    name: str
    """The name of the trace node."""
    parent: Optional["TraceNode"] = None
    """The parent of the trace node."""
    children: List["TraceNode"] = []
    """The children of the trace node."""
    args: Optional[Dict] = None
    """The arguments of the trace node."""
    ret: Any = None
    """The return value of the trace node."""
    start_time: float = 0.0
    """The start time of the trace node."""
    end_time: float = 0.0
    """The end time of the trace node."""
    info: Dict = {}
    """The extra information of the trace node."""
    metadata: Optional[Dict] = None
    """The meta data of the trace node."""

    @property
    def runtime(self) -> float:
        """The runtime of the trace node."""
        return self.end_time - self.start_time


class TraceEngineBase(ABC):
    """A base class for trace engines."""

    @property
    @abstractmethod
    def events(self) -> List[TraceEventBase]:
        """The list of events in the trace."""
        raise NotImplementedError

    @cached_property
    def min_timestamp(self) -> float:
        """The minimum time stamp of the events in the trace."""
        return min([event.time_stamp for event in self.events])

    @property
    @abstractmethod
    def trace_nodes(self) -> Dict[str, TraceNode]:
        """The dictionary of trace nodes in the trace."""
        raise NotImplementedError

    @abstractmethod
    def append(self, event: TraceEventBase) -> None:
        """Append an event to the trace."""
        raise NotImplementedError

    @abstractmethod
    def find_cache(self, name: str, args: Dict[str, Any]) -> Any:
        """Find a completion result in the cache.

        Args:
            name: The name of the completion.
            args: The arguments of the completion.

        Returns:
            The completion result if found, otherwise None.
        """
        raise NotImplementedError


class TracePrinterBase(ABC):
    """A base class for trace printers."""

    @abstractmethod
    def print(
        self, trace: TraceEngineBase, trace_metadata: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Print the trace."""
        raise NotImplementedError
