import functools
import os
import subprocess
import sys
import time
from argparse import Namespace
from typing import Any, Callable, Dict, Optional, Type, TypeVar

import tiktoken
from dotenv.main import _walk_to_root
from loguru import logger

from .core.globals import global_vars
from .core.types import GitInfo

try:
    from langsmith import traceable as _langsmith_traceable  # type: ignore
except Exception:
    F = TypeVar("F", bound=Callable)

    # compatible to the case when langsmith is not installed
    def _langsmith_traceable(*trace_args: Any, **trace_kwargs: Any) -> Callable[[F], F]:
        def decorator(func: F) -> F:
            @functools.wraps(func)
            def inner(*args: Any, **kwargs: Any) -> Any:
                return func(*args, **kwargs)

            return inner  # type: ignore

        return decorator


def _is_interactive():
    """Decide whether this is running in a REPL or IPython notebook."""
    try:
        main = __import__("__main__", None, None, fromlist=["__file__"])
    except ModuleNotFoundError:
        return False
    return not hasattr(main, "__file__")


def get_folder(
    current_folder: Optional[str] = None,
    usecwd: bool = False,
) -> str:
    """Get the the current working directory."""
    if usecwd or _is_interactive() or getattr(sys, "frozen", False):
        # Should work without __file__, e.g. in REPL or IPython notebook.
        folder = os.getcwd()
    elif current_folder is not None:  # [ADD] option to specify the folder
        folder = current_folder
    else:
        # will work for .py files
        frame = sys._getframe()
        current_file = __file__

        while frame.f_code.co_filename == current_file:
            assert frame.f_back is not None
            frame = frame.f_back
        frame_filename = frame.f_code.co_filename
        folder = os.path.dirname(os.path.abspath(frame_filename))

    return folder


def find_files(folder: str, filenames: list[str]) -> list[str]:
    """Find files in the folder or its parent folders."""
    results = []
    for dirname in _walk_to_root(folder):
        for filename in filenames:
            check_path = os.path.join(dirname, filename)
            if os.path.isfile(check_path):
                results.append(check_path)
                # return the first found file among the filenames
                break
    return results


# rewrite find_dotenv, origin in https://github.com/theskumar/python-dotenv/blob/main/src/dotenv/main.py
def find_dotenv(
    filename: str = ".env",
    raise_error_if_not_found: bool = False,
    current_folder: Optional[str] = None,
    usecwd: bool = False,
) -> str:
    """Search in increasingly higher folders for the given file.

    Returns path to the file if found, or an empty string otherwise.
    """
    # Rewrited the original function to add the option to start with a custom folder
    folder = get_folder(current_folder, usecwd)
    results = find_files(folder, [filename])
    if results:
        return results[0]

    if raise_error_if_not_found:
        raise IOError("File not found")
    return ""


def get_git_info() -> GitInfo:
    """Get the git info of the current project."""

    def _run_git_cmd(cmd: list[str]) -> str:
        # run git command and capture stdout and stderr
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()

    git_user_name = _run_git_cmd(["git", "config", "user.name"])
    git_user_email = _run_git_cmd(["git", "config", "user.email"])
    git_branch = _run_git_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    git_commit_hash = _run_git_cmd(["git", "rev-parse", "HEAD"])
    return GitInfo(
        user=git_user_name,
        email=git_user_email,
        branch=git_branch,
        commit_hash=git_commit_hash,
    )


def namespace_to_dict(namespace: Namespace) -> Dict[str, Any]:
    """
    Recursively converts a nested argparse.Namespace object to a nested dictionary.

    Args:
        namespace: An argparse.Namespace object, potentially containing nested Namespace objects

    Returns:
        dict: A nested dictionary representing the namespace structure

    Example:
        args = parser.parse_args()
        args_dict = namespace_to_dict(args)
    """
    result: Dict[str, Any] = {}
    for key, value in vars(namespace).items():
        if isinstance(value, tuple):
            result[key] = tuple(
                namespace_to_dict(item) if hasattr(item, "__dict__") else item
                for item in value
            )
        elif isinstance(value, list):
            result[key] = [
                namespace_to_dict(item) if hasattr(item, "__dict__") else item
                for item in value
            ]
        elif hasattr(value, "__dict__"):
            result[key] = namespace_to_dict(value)
        else:
            result[key] = value
    return result


def get_num_tokens(
    prompt: str, model_name: Optional[str] = None, encoding_name: Optional[str] = None
) -> int:
    """Get the number of tokens in the prompt for the given model or encoding."""
    if encoding_name is None:
        if model_name is None:
            default = global_vars.configs.default_servers.default
            try:
                servers = global_vars.configs.servers or {}
                model_name = servers.get(default, {}).get("model", default)  # type: ignore
                encoding = tiktoken.encoding_for_model(model_name)
            except Exception:
                logger.warning(
                    f"Cannot find tokenizer for the default server {default}, "
                    "using gpt-4o's tokenizer to count tokens."
                )
                encoding = tiktoken.encoding_for_model("gpt-4o")
        else:
            encoding = tiktoken.encoding_for_model(model_name)
    else:
        encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(prompt))


def get_meta_file(trace_file: str) -> str:
    """Get the meta file storing metadata of the trace file."""
    # meta file derived from trace_file: *.pkl -> *_meta.json
    return os.path.splitext(trace_file)[0] + "_meta.json"


def timeit(func: Callable) -> Callable:
    """Time the execution of a function as a decorator."""

    @functools.wraps(func)
    def timer(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info(f"{func.__name__} executed in {end - start:.2f} seconds.")
        return result

    return timer


class LoguruFormatter:
    """Custom formatter for loguru logger."""

    def __init__(
        self,
        fmt: Optional[str] = None,
        max_length: Optional[int] = None,
        suffix_length: int = 0,
    ):
        """Initialize the formatter with the format string and max length of the message.

        Args:
            fmt: The format string for the log message.
            max_length: The maximum length of the message, truncate if longer.
            suffix_length: The length of the suffix to keep when truncating.
        """
        if fmt is None:
            fmt = global_vars.configs.settings.logging.format
        self.fmt = fmt.rstrip()
        self.max_length = max_length
        self.suffix_length = suffix_length

    def loguru_format(self, record: Dict) -> str:
        """Format the log message with the record."""
        msg = record["message"]
        fmt = self.fmt
        if self.max_length is not None and len(msg) > self.max_length:
            suffix_len = min(self.max_length, self.suffix_length)
            truncated = msg[: self.max_length - suffix_len]
            truncated += f"...(snipped {len(msg) - self.max_length} chars)"
            if suffix_len > 0:
                truncated += "..." + msg[-suffix_len:]
            record["trunc_message"] = truncated
            fmt = fmt.replace("{message}", "{trunc_message}")
        return fmt + "\n"
