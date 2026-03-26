"""appl - A Prompt Programming Language."""

from __future__ import annotations

import datetime
import inspect
import os
import sys
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import toml
import yaml
from dotenv import load_dotenv
from jsonargparse import ArgumentParser
from loguru import logger

logger.remove()

from .caching import DBCache
from .compositor import ApplStr as Str
from .compositor import iter
from .core import (
    Audio,
    BracketedDefinition,
    CallFuture,
    CompletionResponse,
    Definition,
    Generation,
    Image,
    Indexing,
    Promptable,
    PromptContext,
    PromptPrinter,
    PromptRecords,
    SchemaTool,
    StringFuture,
    Tool,
)
from .core import appl_compile as compile
from .core import appl_execute as execute
from .core import appl_format as format
from .core import appl_with_ctx as with_ctx
from .core.config import APPLConfigs, ConfigsDict, load_config
from .core.generation import get_gen_name_prefix, set_gen_name_prefix
from .core.globals import global_vars
from .core.io import dump_file, load_file
from .core.message import (
    AIMessage,
    Conversation,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from .core.patch import patch_threading
from .core.promptable import define, define_bracketed, promptify
from .core.trace import traceable
from .core.utils import need_ctx, partial, wraps
from .func import (
    as_func,
    as_tool,
    as_tool_choice,
    call,
    convo,
    empty_line,
    gen,
    grow,
    ppl,
    records,
    reset_context,
    str_future,
)
from .func import gen as completion  # create alias
from .role_changer import AIRole, SystemRole, ToolRole, UserRole
from .servers import server_manager
from .settings import merge_configs, update_appl_configs
from .tracing import TraceEngine, TraceLangfusePrinter, TraceLunaryPrinter, print_trace
from .utils import (
    LoguruFormatter,
    find_dotenv,
    find_files,
    get_folder,
    get_git_info,
    get_meta_file,
    timeit,
)
from .version import __version__


def get_parser(
    env_prefix: str = "", default_env: bool = True, **kwargs: Any
) -> ArgumentParser:
    """Get an argument parser with configurable APPL configs."""
    parser = ArgumentParser(env_prefix=env_prefix, default_env=default_env, **kwargs)
    parser.add_argument("--appl", type=APPLConfigs, default=global_vars.configs)
    return parser


def init(**kwargs: Any) -> None:
    """Overwrite APPL configs.

    see [default configs](../setup/#default-configs) for more details.

    Examples:
        ```python
        import appl

        appl.init(servers={"default": "gpt-4o"})
        ```
    """
    logger.warning(
        f"appl.init() is not mandatory to initialize APPL for appl>=0.2.0, "
        "please remove it if you are not changing the configs."
    )
    if kwargs:
        logger.info(f"Updating APPL configs with: \n{yaml.dump(kwargs)}")
        new_configs = merge_configs(global_vars.configs, **kwargs)
        update_appl_configs(new_configs)


@contextmanager
def init_within_thread(
    log_file_prefix: Optional[str] = None, gen_name_prefix: Optional[str] = None
) -> Any:
    """Initialize APPL to work with multi-threading, including logging and tracing.

    Args:
        log_file_prefix: The prefix for the log file. Defaults to use the path of the main log file.
        gen_name_prefix: The prefix for the generation name. Defaults to use the thread name.

    Examples:
        ```python
        def function_run_in_thread():
            with appl.init_within_thread():
                # do something within the thread
        ```
    """
    handler_id = None

    try:
        thread_name = threading.current_thread().name
        logging_settings = global_vars.configs.settings.logging

        def filter_thread_record(record: Dict) -> bool:
            assert hasattr(record["thread"], "name")
            # Use prefix match to filter the log records in different threads
            name = record["thread"].name
            return name == thread_name or name.startswith(thread_name + "_")

        if logging_settings.enable_file:
            if log_file_prefix is None:
                assert (
                    global_vars.metadata.log_file is not None
                ), "should have log file set"
                thread_log_path = os.path.join(
                    global_vars.metadata.log_file[: -len(".log")] + "_logs",
                    f"{thread_name}.log",
                )
            else:
                thread_log_path = f"{log_file_prefix}_{thread_name}.log"

            log_level = (
                logging_settings.log_file.log_level or logging_settings.log_level
            )
            # The logger append to the file by default, not overwrite.
            handler_id = logger.add(
                thread_log_path,
                level=log_level,
                format=logging_settings.format,
                filter=filter_thread_record,  # type: ignore
            )
        if gen_name_prefix:
            set_gen_name_prefix(gen_name_prefix)
            # ? shall we reset the prefix after exiting the context?
            logger.info(
                f"Thread {thread_name}, set generation name prefix as: {gen_name_prefix}"
            )

        if handler_id is None:
            logger.warning("logging is not enabled")
        yield thread_log_path
    except Exception as e:
        logger.error(f"Error in thread: {e}")
        raise e
    finally:
        if handler_id:
            logger.remove(handler_id)
