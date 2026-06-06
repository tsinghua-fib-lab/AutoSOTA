import threading
from inspect import signature
from typing import Any, Callable, Dict, Optional, TypeVar, Union, overload

from loguru import logger

from .globals import global_vars
from .types.trace import (
    CompletionRequestEvent,
    CompletionResponseEvent,
    FunctionCallEvent,
    FunctionReturnEvent,
    GenerationInitEvent,
    GenerationResponseEvent,
    TraceEngineBase,
    TraceEventBase,
    TraceNode,
    TracePrinterBase,
)
from .utils import get_source_code, wraps


def add_to_trace(event: TraceEventBase) -> None:
    """Add an event to the trace."""
    if global_vars.trace_engine:
        global_vars.trace_engine.append(event)


F = TypeVar("F", bound=Callable)


@overload
def traceable(func: F) -> F: ...


@overload
def traceable(
    func: Optional[str] = None,
    *,
    metadata: Optional[Dict] = None,
) -> Callable[[F], F]: ...


def traceable(
    func: Optional[Union[F, str]] = None,
    *,
    metadata: Optional[Dict] = None,
) -> Union[F, Callable[[F], F]]:
    """Make a function traceable.

    Args:
        func (str): The custom name of the function.
        metadata (Dict): The meta information of the function to be traced.
    """
    name: Optional[str] = None
    if metadata is None:
        metadata = {}

    def decorator(func: F) -> F:
        if "source_code" not in metadata:
            if source_code := get_source_code(func):
                metadata["source_code"] = source_code

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            func_id = name
            if func_id is None:
                func_id = func.__qualname__
            func_run_cnt = global_vars.inc("func_run_cnt", key=func_id) - 1
            func_id += f"_{func_run_cnt}"
            if global_vars.configs.settings.tracing.display_trace_info:
                logger.trace(
                    f"Tracking function {func_id} with parent {global_vars.current_func.get()} in thread {threading.current_thread()}"
                )

            def _get_bind_args():
                sig = signature(func)
                kwargs_copy = kwargs.copy()
                # remove special args that do not need to be passed
                for key in ["_ctx", "_locals", "_globals", "compositor"]:
                    if key not in sig.parameters:
                        kwargs_copy.pop(key, None)
                return sig.bind_partial(*args, **kwargs_copy)

            if global_vars.trace_engine:
                # NOTE: compute repr(args) might be time-consuming
                # TODO: jsonify the args
                add_to_trace(
                    FunctionCallEvent(
                        name=func_id,
                        args={
                            k: repr(v) for k, v in _get_bind_args().arguments.items()
                        },
                        metadata=metadata,
                    )
                )

            # set the current function, used for the function calls inside to get the parent function
            token = global_vars.current_func.set(func_id)

            # call the inner function
            ret = func(*args, **kwargs)

            # reset the current function name after the function call
            global_vars.current_func.reset(token)
            if global_vars.trace_engine:
                add_to_trace(FunctionReturnEvent(name=func_id, ret=repr(ret)))
                # TODO: replace the return value with the actual value when the computation of future is finished (in trace)
                # TODO: jsonify the ret
            return ret

        return wrapper  # type: ignore

    if callable(func):
        return decorator(func)  # type: ignore
    else:
        name = func
        return decorator


def find_in_trace(
    name: str, args: Dict[str, Any], trace: Optional[TraceEngineBase] = None
) -> Any:
    """Find a completion result in the trace.

    Args:
        name: The name of the completion.
        args: The arguments of the completion.
        trace: The trace to search in. Defaults to the global trace engine.

    Returns:
        The completion result if found, otherwise None.
    """
    trace = trace or global_vars.resume_trace
    if trace is None:
        return None
    return trace.find_cache(name, args)
