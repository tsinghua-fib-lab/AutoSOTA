import contextvars
import datetime
import threading
from argparse import Namespace
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from enum import Enum
from typing import Any, Dict, Optional, Union

import pendulum
from pydantic import BaseModel
from rich.live import Live

from .config import DEFAULT_CONFIGS, APPLConfigs
from .types.caching import DBCacheBase
from .types.custom import MetaData
from .types.executor import global_executors
from .types.trace import TraceEngineBase


class GlobalVars:
    """Global variables."""

    now: pendulum.DateTime
    lock: threading.Lock
    # statistics
    gen_cnt: int
    num_requests: Dict[str, int]
    api_cost: Dict[str, float]
    func_run_cnt: Dict[str, int]
    # catching
    llm_cache: Optional[DBCacheBase]
    # logging and tracing
    current_func: contextvars.ContextVar[Optional[str]]
    trace_engine: Optional[TraceEngineBase]
    resume_trace: Optional[TraceEngineBase]
    # streaming
    live: Optional[Live]
    live_lock: threading.Lock

    def __init__(self):
        """Initialize the global variables."""
        self.now = pendulum.instance(datetime.datetime.now())
        self.lock = threading.Lock()
        self.gen_cnt = 0
        self.num_requests = defaultdict(int)
        self.api_cost = defaultdict(float)
        self.func_run_cnt = defaultdict(int)
        self.llm_cache = None
        self.current_func = contextvars.ContextVar("current_func", default=None)
        self.trace_engine = None
        self.resume_trace = None
        self.live = None
        self.live_lock = threading.Lock()
        self._configs = None
        self._metadata = None

    @property
    def metadata(self) -> MetaData:
        """Get the metadata."""
        if self._metadata is None:
            raise ValueError("metadata is not set")
        return self._metadata

    @metadata.setter
    def metadata(self, value: MetaData) -> None:
        """Set the metadata."""
        self._metadata = value

    @property
    def configs(self) -> APPLConfigs:
        """Get the configs."""
        if self._configs is None:
            raise ValueError("configs is not set")
        return self._configs

    @configs.setter
    def configs(self, value: APPLConfigs) -> None:
        """Set the configs."""
        self._configs = value

    def get(self, name: str, default: Any = None) -> Any:
        """Get the value of a global variable."""
        with self.lock:
            return getattr(self, name, default)

    def set(self, name: str, value: Any) -> None:
        """Set the value of a global variable."""
        with self.lock:
            setattr(self, name, value)

    def inc(
        self, name: str, delta: Union[int, float] = 1, key: Optional[str] = None
    ) -> Any:
        """Increment a global variable by a delta and return the new value."""
        with self.lock:
            value = getattr(self, name, 0)
            if isinstance(value, dict) and key is not None:
                value[key] = value.get(key, 0) + delta
                return value[key]
            else:
                if key is not None:
                    raise ValueError(
                        f"Cannot increment a non-dict variable {name} with a key"
                    )
                value += delta
                setattr(self, name, value)
                return value


# Singleton
global_vars = GlobalVars()
global_vars.configs = DEFAULT_CONFIGS

# thread-level vars
thread_local = threading.local()


def get_thread_local(name: str, default: Any = None) -> Any:
    """Get the value of a thread-local variable."""
    return getattr(thread_local, name, default)


def set_thread_local(name: str, value: Any) -> None:
    """Set the value of a thread-local variable."""
    setattr(thread_local, name, value)


def inc_thread_local(name: str, delta: Union[int, float] = 1) -> Any:
    """Increment a thread-local variable by a delta and return the new value."""
    value = get_thread_local(name, 0)
    value += delta
    setattr(thread_local, name, value)
    return value
