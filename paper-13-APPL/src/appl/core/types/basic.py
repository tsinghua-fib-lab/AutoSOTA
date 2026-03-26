from __future__ import annotations

import operator
import sys
from abc import ABC, abstractmethod
from argparse import Namespace
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Event, Thread
from types import CodeType, FunctionType, TracebackType
from typing import (
    IO,
    Any,
    Awaitable,
    Callable,
    ClassVar,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    SupportsIndex,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

from pydantic import BaseModel, Field

if sys.version_info < (3, 10):
    from typing_extensions import Concatenate, ParamSpec
else:
    from typing import Concatenate, ParamSpec

from loguru import logger

_T = TypeVar("_T")
# Type defs
OneOrMany = Union[_T, Sequence[_T]]
"""A type that can be either a single item or a sequence of items."""
MaybeOneOrMany = Union[_T, Sequence[_T], None]
"""A type that can be either a single item, a sequence of items, or None."""
