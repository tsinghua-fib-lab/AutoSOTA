import json
import os
import pickle
import re
from threading import Lock
from typing import Any, Dict, List, Optional, Type

from loguru import logger
from pydantic import BaseModel

from ..core.globals import global_vars
from ..core.trace import (
    CompletionRequestEvent,
    CompletionResponseEvent,
    FunctionCallEvent,
    FunctionReturnEvent,
    GenerationInitEvent,
    GenerationResponseEvent,
    TraceEngineBase,
    TraceEventBase,
    TraceNode,
)


class TraceEngine(TraceEngineBase):
    """The engine used to record the trace of a program execution."""

    def __init__(self, filename: str, mode: str = "write", strict: bool = True) -> None:
        """Initialize the TraceEngine.

        Args:
            filename: The filename storing the trace.
            mode: The mode of the trace, "write" or "read". Defaults to "write".
            strict:
                Whether to match strictly when used as a cache. Defaults to True.

                - True: matching according to the generation id, prompts, and
                    parameters. And cache stops to work whenever a match failed.
                - False: only matching prompts and parameters.
        """
        self._mode = mode
        self._strict = strict
        self._events: List[TraceEventBase] = []  # events read from the file
        self._trace_nodes: Dict[str, TraceNode] = {}
        self._gen_cache: Dict[str, List[Any]] = {}
        self._lock = Lock()

        if mode == "write":
            if os.path.exists(filename):
                logger.warning(f"Trace file {filename} already exists, overwriting")
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            self._file = open(filename, "wb+")
        elif mode == "read":
            if not os.path.exists(filename):
                raise FileNotFoundError(f"Trace file {filename} not found")
            self._file = open(filename, "rb+")
            self._read()
        else:
            raise ValueError(f"Invalid mode {mode}, only 'write' or 'read' allowed.")

    @property
    def events(self) -> List[TraceEventBase]:
        """The list of events in the trace."""
        return self._events

    @property
    def trace_nodes(self) -> Dict[str, TraceNode]:
        """The dictionary of trace nodes."""
        return self._trace_nodes

    @classmethod
    def convert_pydantic_class_to_schema(cls, class_: Type) -> Dict:
        """Convert a class to a schema.

        Args:
            class_: The class to convert
        """
        if issubclass(class_, BaseModel):
            return class_.model_json_schema()
        raise ValueError(f"Cannot convert class {class_} to schema")

    @classmethod
    def args_to_json(cls, args: Dict) -> Dict:
        """Serialize the values of the arguments to JSON format."""
        args_json = {}
        for k, v in args.items():
            if isinstance(v, type) and issubclass(v, BaseModel):
                v = cls.convert_pydantic_class_to_schema(v)
            # TODO: shall we serialize everything?
            # elif k != "message":
            #     try:
            #         v = json.dumps(v)
            #     except:
            #         v = str(v)
            args_json[k] = v
        return args_json

    def append(self, event: TraceEventBase) -> None:
        """Append an event to the trace."""
        # print(
        #     event.name,
        #     global_vars.current_func.get(),
        #     getattr(event, "parent_func", None),
        # )

        if hasattr(event, "args"):
            event.args = self.args_to_json(event.args)

        self._events.append(event)
        name, time_stamp = event.name, event.time_stamp
        if self._mode == "write":
            if isinstance(event, (FunctionCallEvent, GenerationInitEvent)):
                event.parent_func = self._last_func
            elif isinstance(event, CompletionRequestEvent):
                match = re.match(r"(.+)_raw_\d+", event.name)
                if match:
                    event.parent_func = match.group(1)
                else:
                    event.parent_func = self._last_func
                    logger.warning(
                        f"Unusual completion request name: {event.name}. "
                        "Using last function as parent event"
                    )

            with self._lock:
                logger.trace(f"add to trace {event}")
                pickle.dump(event, self._file)
                self._file.flush()

        assert name is not None

        def _merge_metadata(
            data: Optional[Dict], other: Optional[Dict]
        ) -> Optional[Dict]:
            if data is None:
                return other
            if other is None:
                return data
            return {**data, **other}

        if isinstance(event, FunctionCallEvent):
            newnode = self._add_node(name, event.parent_func, type="func")
            newnode.start_time = time_stamp
            newnode.args = event.args
            newnode.metadata = event.metadata
        elif isinstance(event, FunctionReturnEvent):
            node = self._get_node(name)
            if node:
                node.ret = event.ret
                node.end_time = time_stamp
                node.metadata = _merge_metadata(node.metadata, event.metadata)
        elif isinstance(event, GenerationInitEvent):
            newnode = self._add_node(name, event.parent_func, type="gen")
            newnode.start_time = time_stamp
            newnode.metadata = event.metadata
        elif isinstance(event, GenerationResponseEvent):
            node = self._get_node(name)
            if node:
                node.end_time = time_stamp
                node.args = event.args
                node.ret = event.ret
                node.metadata = _merge_metadata(node.metadata, event.metadata)
        elif isinstance(event, CompletionRequestEvent):
            newnode = self._add_node(name, event.parent_func, type="raw_llm")
            newnode.start_time = time_stamp
            newnode.metadata = event.metadata
        elif isinstance(event, CompletionResponseEvent):
            node = self._get_node(name)
            if node:
                node.end_time = time_stamp
                node.args = event.args
                node.ret = event.ret
                node.info["cost"] = event.cost
                node.metadata = _merge_metadata(node.metadata, event.metadata)

            # cached for raw completion response
            key = self._cache_key(name, event.args)
            # logger.debug(f"add to cache with key: {key}")
            if key not in self._gen_cache:
                self._gen_cache[key] = []
            self._gen_cache[key].append(event.ret)

    def find_cache(self, name: str, args: Dict) -> Any:
        """Find a cached response for a generation request.

        Args:
            name: The name of the generation request.
            args: The arguments of the generation request.
        """
        args = self.args_to_json(args)
        with self._lock:
            key = self._cache_key(name, args)
            # logger.debug(f"try to find cache with key: {key}")
            entry_list = self._gen_cache.get(key, None)
            if not entry_list or len(entry_list) == 0:
                return None
            entry = entry_list.pop(0)
            return entry

    def _add_node(
        self, name: str, parent_name: Optional[str] = None, type: str = "gen"
    ) -> TraceNode:
        parent = self._get_node(parent_name)
        newnode = TraceNode(type=type, name=name, parent=parent)
        if name in self._trace_nodes:
            raise ValueError(f"Node {name} already exists in trace")
        self._trace_nodes[name] = newnode
        if parent:
            parent.children.append(newnode)
        return newnode

    def _get_node(self, name: Optional[str]) -> Optional[TraceNode]:
        if name is None:
            return None
        if name not in self._trace_nodes:
            raise ValueError(f"Node {name} not found in trace")
        return self._trace_nodes[name]

    @property
    def _last_func(self) -> Optional[str]:
        return global_vars.current_func.get()

    def _cache_key(self, name: str, args: Dict) -> str:
        # pop the arguments that do not affect the result
        args.pop("stream", None)
        if self._strict:
            return f"{name} {args}"
        else:
            return f"{args}"

    def _read(self) -> None:
        while True:
            try:
                event: TraceEventBase = pickle.load(self._file)
                self.append(event)
            except EOFError:
                break
