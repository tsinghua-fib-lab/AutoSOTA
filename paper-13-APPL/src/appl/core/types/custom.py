from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class ResponseType(str, Enum):
    """The type of generation response."""

    TEXT = "text"
    """A text completion."""
    TOOL_CALL = "tool_calls"
    """A list of tool calls."""
    OBJECT = "obj"
    """An instance of a response model."""
    IMAGE = "image"
    """An image."""
    UNFINISHED = "unfinished"
    """The response is not finished."""


@dataclass
class GitInfo:
    """Git information."""

    user: Optional[str] = None
    email: Optional[str] = None
    branch: Optional[str] = None
    commit_hash: Optional[str] = None


@dataclass
class MetaData:
    """Metadata for the run."""

    appl_version: str
    cwd: str
    run_cmd: str
    git_info: GitInfo
    exec_file_path: str
    exec_file_basename: str
    start_time: str
    dotenvs: List[str]
    appl_config_files: List[str]
    log_file: Optional[str] = None
    trace_file: Optional[str] = None
