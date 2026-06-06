import copy
import inspect
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    get_origin,
    overload,
)

from loguru import logger
from pydantic import BaseModel

from .core import (
    BaseMessage,
    BaseTool,
    Compositor,
    Conversation,
    GenArgs,
    Generation,
    PrinterPop,
    PrinterPush,
    PromptContext,
    PromptFunc,
    PromptRecords,
    SchemaTool,
    Tool,
    need_ctx,
    partial,
    wraps,
)
from .core.globals import global_vars
from .core.printer import Indexing
from .core.response import CompletionResponse
from .core.runtime import appl_execute
from .core.trace import traceable
from .core.types import (
    CallFuture,
    ExecutorType,
    MaybeOneOrMany,
    OneOrMany,
    ParamSpec,
    StringFuture,
)
from .core.utils import get_source_code
from .servers import server_manager
from .utils import _langsmith_traceable

# https://docs.python.org/3/library/typing.html#typing.ParamSpec
# https://docs.python.org/3/library/typing.html#typing.Concatenate
# https://peps.python.org/pep-0612/
# Callable[P, T] is used for static type inference (Pylance)
P = ParamSpec("P")
T = TypeVar("T")
F = TypeVar("F", bound=Callable)  # function
M = TypeVar("M")  # model
R = TypeVar("R")  # return value


def auto_prime_gen(gen_func):
    """Decorate a generator to automatically prime the generator."""

    def wrapper(*args, **kwargs):
        gen = gen_func(*args, **kwargs)
        next(gen)  # prime the generator
        return gen

    return wrapper


@overload
def ppl(ctx: F) -> F: ...


@overload
def ppl(
    ctx: str = "new",
    compositor: Optional[Compositor] = None,
    *,
    default_return: Optional[Literal["prompt"]] = None,
    docstring_as: Optional[str] = None,
    auto_prime: bool = False,
    num_extra_wrappers: int = 0,
    new_ctx_func: Callable = PromptContext,
) -> Callable[[F], F]: ...


def ppl(
    ctx: Union[str, F] = "new",
    compositor: Optional[Compositor] = None,
    *,
    default_return: Optional[Literal["prompt"]] = None,
    docstring_as: Optional[str] = None,
    auto_prime: bool = False,
    num_extra_wrappers: int = 0,
    new_ctx_func: Callable = PromptContext,
) -> Union[Callable[[F], F], F]:
    """Decorate a function to mark it as an APPL function.

    The function contains a prompt context, which could be same as or
    copied from its caller function, or created from scratch, or resumed
    from the last run.

    Args:
        ctx (str):
            the method to deal with the child context, available methods includes:

            - (default) "new" or "new_ctx": create a brand new context.
            - "copy" or "copy_ctx":
                copy from the parent's context, the change will not
                affect the parent's context.
            - "same" or "same_ctx":
                use the same context as the parent's, the change will
                affect the parent's context.
            - "resume" or "resume_ctx":
                resume its own context from the last run.
                For the first run, it will use the parent's context.

        compositor (Compositor, optional):
            the default compositor to be used. Defaults to None.
        default_return (str, optional):
            The default return value, "prompt" means return the prompt within
            the function. Defaults to None.
        docstring_as (str, optional):
            Include the triple-quoted docstring as a message in the prompt.
            Options include "user" and "system". Defaults to None.
        auto_prime (bool, optional):
            set to True to automatically prime the generator. Defaults to False.
        num_extra_wrappers (int, optional):
            the number of extra wrappers to go back to the caller frame.
        new_ctx_func (Callable, optional):
            the function to create a new context. Defaults to PromptContext.
    """
    # The same doc string as PromptFunc (excluding the func argument)

    ctx_method: str = "new"

    def decorator(func: F) -> F:
        """Decorate a function as prompt function."""
        _is_class_method = False
        if "." in (qualname := func.__qualname__):
            # NOTE: this is a workaround for class methods, may not cover all cases
            qualnames = qualname.split(".")
            if qualnames[-2] != "<locals>":
                _is_class_method = True

        # ? should disable such usage?
        # if not _is_class_method and "<locals>" in qualname and ctx_method == "resume":
        #     raise ValueError("Cannot use 'resume' with local functions.")
        prompt_func = PromptFunc(
            func, ctx_method, compositor, default_return, docstring_as, new_ctx_func
        )

        metadata = {}
        if source_code := get_source_code(func):
            metadata["source_code"] = source_code

        @need_ctx
        @traceable(metadata=metadata)
        @_langsmith_traceable(name=func.__qualname__, metadata={"appl": "func"})  # type: ignore
        @wraps(func)
        def wrapper(
            *args: Any,
            _globals: Optional[Dict] = None,
            _locals: Optional[Dict] = None,
            **kwargs: Any,
        ) -> Any:
            # closure variables
            freevars = prompt_func.compiled_func.freevars
            if _locals is None:
                # * Workaround for closure variables
                # Default: use the locals from the caller
                frame = inspect.currentframe()
                num_wrappers = (4 if auto_prime else 3) + num_extra_wrappers
                for _ in range(num_wrappers):
                    if frame is None:
                        raise RuntimeError("No caller frame found")
                    # back to @_langsmith_traceable, @traceable, and the caller frame
                    frame = frame.f_back
                if frame is None:
                    raise RuntimeError("No caller frame found")
                _locals = frame.f_locals

                if len(freevars):
                    vars = {var: _locals.get(var, "NotFound") for var in freevars}
                    logger.debug(
                        f"For freevars of function {func.__name__}, "
                        f"automatically using locals from the caller: {vars}"
                    )
                    for var in freevars:
                        if var not in _locals:
                            logger.warning(
                                f"could not find variable {var} automatically from"
                                f"the caller frame for function {func.__name__}. "
                                "If you have wrapper around the function, you may need"
                                "to set the `num_extra_wrappers` in @ppl function."
                            )
            results = prompt_func(
                *args,
                _globals=_globals,
                _locals=_locals,
                _is_class_method=_is_class_method,
                **kwargs,
            )

            return results

        if auto_prime:
            wrapper = auto_prime_gen(wrapper)
        setattr(wrapper, "_prompt_func", prompt_func)
        return wrapper  # type: ignore

    if isinstance(ctx, str):
        ctx_method = ctx
        # used as a decorator with arguments (e.g., @ppl(ctx="copy"))
        # returns a decorator that takes a function as input
        return decorator
    else:
        # used as a single decorator (i.e., @ppl)
        return decorator(func=ctx)  # returns a wrapper


def reset_context(func: Callable) -> None:
    """Reset the context for APPL functions with the 'resume' context method."""
    if prompt_func := getattr(func, "_prompt_func", None):
        if reset_func := getattr(prompt_func, "_reset_context_func", None):
            reset_func()
            logger.info(f"Context reset for function {func.__name__}")
        else:
            logger.warning(f"Nothing to reset for function {func.__name__}")
    else:
        logger.warning(f"Not an APPL function: {func.__name__}, cannot reset context.")


def as_func(
    func: Callable[P, T],
    _globals: Optional[Dict] = None,
    _locals: Optional[Dict] = None,
) -> Callable[P, T]:
    """Fill the globals and locals for a ppl function.

    When locals not provided, it will use the locals from the caller.
    Commonly used for wrapper functions to pass the closure variables.
    """
    frame = inspect.currentframe()
    if _locals is None and frame is not None and frame.f_back is not None:
        _locals = frame.f_back.f_locals
    return partial(func, _globals=_globals, _locals=_locals)


def str_future(obj: Any) -> StringFuture:
    """Convert an object to a StringFuture object."""
    return StringFuture(obj)


def as_tool(func: Callable, **kwargs: Any) -> Tool:
    """Wrap a given function with additional predefined arguments into a Tool.

    This function allows converting a standard function into a 'Tool' by
    specifying the function and any additional arguments that should be
    pre-defined for it. These additional arguments are passed as keyword
    arguments and will be bound to the function within the Tool object,
    so that these arguments are not required when using this tool.

    Args:
        func (Callable):
            The function to be converted into a Tool.
        **kwargs:
            Keyword arguments that will be predefined for the function in
            the Tool object.

    Returns:
        Tool:
            An object encapsulating the given function and its predefined
            arguments, ready to be utilized as a Tool.

    Examples:
        Given a function `move_disk` that requires an environment and two
        pegs to move a disk from one peg to another in the Tower of Hanoi
        puzzle, one can create a tool with a predefined environment by:

        ```python
        def move_disk(env: HanoiEnv, from_peg: int, to_peg: int) -> str:
            pass

        env = HanoiEnv()
        tools = [as_tool(move_disk, env=env)]
        ```

        In this example, `move_disk` is encapsulated into a Tool with `env`
        predefined, so only `from_peg` and `to_peg` are required.
    """
    return Tool(func=func, **kwargs)


def as_tool_choice(obj: Union[str, Callable, BaseTool]) -> dict:
    """Build a tool choice argument for the OpenAI API from an object."""
    if isinstance(obj, BaseTool):
        name = obj.name
    else:
        name = getattr(obj, "__name__", str(obj))
    return dict(type="function", function=dict(name=name))


def call(
    func: Callable,
    *args: Any,
    executor_type: ExecutorType = ExecutorType.GENERAL_THREAD_POOL,
    **kwargs: Any,
) -> CallFuture:
    """Create a CallFuture object from a function and its arguments.

    The CallFuture object will call the function in a separate thread or process,
    therefore the function need to be thread-safe or process-safe.
    """
    return CallFuture(func, *args, executor_type=executor_type, **kwargs)


@need_ctx
def get_var(name: str, _ctx: PromptContext) -> Any:
    """Get a variable by name from the prompt context."""
    return getattr(_ctx, name)


@need_ctx
def records(_ctx: Optional[PromptContext] = None) -> PromptRecords:
    """Return the prompt defined in the current function.

    Similar to locals() in Python in some sense.
    """
    # add default value for _ctx to avoid the warning of type checker
    if _ctx is None:
        raise ValueError(
            "PromptContext is required for records, "
            "this function should be called within @ppl function."
        )
    return _ctx.records


@need_ctx
def convo(_ctx: Optional[PromptContext] = None) -> Conversation:
    """Return the full conversation in the context.

    Similar to globals() in Python in some sense.
    """
    # Added default value for _ctx to avoid the warning of type checker
    if _ctx is None:
        raise ValueError(
            "PromptContext is required for convo, "
            "this function should be called within @ppl function."
        )
    return _ctx.messages


def empty_line(num_lines: int = 1) -> PromptRecords:
    """Create empty lines regardless of other compositor."""
    records = PromptRecords()
    records.record(PrinterPush(separator="\n", indexing=Indexing(), new_indent=""))
    for _ in range(num_lines):
        records.record("")
    records.record(PrinterPop())
    return records


def build_tools(
    tools: OneOrMany[Union[BaseTool, Callable, Dict]],
) -> Sequence[BaseTool]:
    """Build a list of tools from the given tools or functions."""

    def convert_to_tool(tool: Union[BaseTool, Callable, Dict]) -> BaseTool:
        if isinstance(tool, BaseTool):
            return tool
        if callable(tool):
            return as_tool(tool)
        if isinstance(tool, dict):
            return SchemaTool(tool_schema=tool)
        raise ValueError(f"Invalid tool: {tool}")

    # process tools
    if isinstance(tools, BaseTool) or callable(tools) or isinstance(tools, dict):
        return [convert_to_tool(tools)]
    if isinstance(tools, Sequence):
        return [convert_to_tool(tool) for tool in tools]
    raise ValueError(f"Invalid tools: {tools}")


@need_ctx
def grow(content: Any, *, _ctx: Optional[PromptContext] = None) -> None:
    """Append the content to the prompt in the current context."""
    if _ctx is None:
        raise ValueError(
            "PromptContext is required for appending. "
            "Normally, it should be automatically filled."
        )
    appl_execute(content, _ctx=_ctx)


@need_ctx
def gen(
    server: Optional[str] = None,
    *,
    messages: Optional[Union[Conversation, List[BaseMessage], List[Dict]]] = None,
    max_tokens: Optional[int] = None,
    stop: MaybeOneOrMany[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    n: Optional[int] = None,
    tools: OneOrMany[Union[BaseTool, Callable, Dict]] = [],
    tool_format: str = "auto",
    stream: Optional[bool] = None,
    response_format: Optional[Union[dict, str, Type[M]]] = None,
    response_model: Optional[Type[M]] = None,
    max_relay_rounds: int = 0,
    mock_response: Optional[Union[CompletionResponse, str]] = None,
    messages_process_func: Optional[Callable[[Conversation], Conversation]] = None,
    _ctx: Optional[PromptContext] = None,
    **kwargs: Any,
) -> Generation[M]:
    """Send a generation request to the LLM backend.

    Args:
        server (str, optional):
            name of the backend server. Defaults to the default server set in the configs.
        messages (Union[Conversation, List[BaseMessage]], optional):
            the messages as the prompt for the LLM. Defaults to retrieve from the context.
        max_tokens (int, optional): maximum number of tokens to generate. Defaults to None.
        stop (str|Sequence[str], optional): stop sequence(s). Defaults to None.
        temperature (float, optional): temperature for sampling. Defaults to None.
        top_p (float, optional): nucleus sampling parameter. Defaults to None.
        n (int, optional): number of choices to generate. Defaults to 1.
        tools (BaseTool|Callable|Dict|Sequence[BaseTool|Dict|Callable], optional):
            tools can be used. Defaults to None.
        tool_format (str, optional): the format for the tools. Defaults to "auto".
        stream (bool, optional): whether to stream the results. Defaults to False.
        response_format (Union[dict, str, Type[M]], optional):
            OpenAI's argument specifies the response format. Defaults to None.
        response_model (Type[M], optional):
            instructor's argument specifies the response format as a Pydantic model.
            use `instructor_patch_mode` to specify the mode for patching the raw completion.
            Recommended to use `response_format` instead. Defaults to None.
        max_relay_rounds (int, optional):
            the maximum number of relay rounds to continue the unfinished text generation. Defaults to 0.
        mock_response (Union[CompletionResponse, str], optional):
            mock response for testing. Defaults to None.
        messages_process_func (Callable[[Conversation], Conversation], optional):
            a function to process the messages before sending to the LLM.
            Defaults to None.
        _ctx (PromptContext): prompt context, will be automatically filled.
        kwargs (Any): extra arguments for the generation.

    Returns:
        Generation: a future object representing the generation result
    """
    backend_server = server_manager.get_server(server)
    if isinstance(messages, list):
        messages = Conversation(messages=messages)
    if messages is None:
        if _ctx is None:
            raise ValueError(
                "PromptContext is required for generation when messages is not provided."
            )
        messages = _ctx.messages
    messages.materialize()  # materialize the messages
    # TODO: double check the correctness
    messages = copy.deepcopy(messages)  # freeze the prompt for the generation
    if messages_process_func:
        messages = messages_process_func(messages)

    if isinstance(response_format, str):
        if response_format != "json":
            raise ValueError(
                "Only 'json' is supported for response_format in string format."
            )
        response_format = {"type": "json_object"}

    response_format_is_pydantic_model = False
    if response_format is not None:
        if isinstance(response_format, dict):
            if "type" not in response_format:
                raise ValueError("response_format must specify the type.")
            if response_format["type"] not in ("json_object", "json_schema"):
                raise ValueError(
                    "Only 'json_object' and 'json_schema' are supported for response_format."
                )
        elif isinstance(response_format, type) and issubclass(
            response_format, BaseModel
        ):
            response_format_is_pydantic_model = True
        else:
            raise ValueError(
                "Invalid response_format, can be a dict or a Pydantic model."
            )

        if response_model is not None:
            raise ValueError(
                "response_format and response_model cannot be used together."
            )

    if max_relay_rounds > 0:
        if response_format_is_pydantic_model or response_model is not None:
            raise ValueError(
                "max_relay_rounds cannot be used when response_format is "
                "a Pydantic model or response_model is specified."
            )
        elif response_format is not None:
            logger.warning(
                "automatic continuation may not work well when response_format is specified. "
                "Recommend using plain text generation instead."
            )

    if (
        isinstance(get_origin(response_format), type)
        or get_origin(response_format) is Literal
        or isinstance(response_format, type)
        and not issubclass(response_format, BaseModel)
    ):

        class Response(BaseModel):
            response: response_format  # type: ignore

        response_format = Response  # type: ignore
        kwargs["_wrapped_attribute"] = "response"

    create_args = GenArgs(
        model=backend_server.model_name,
        messages=messages,
        max_tokens=max_tokens,
        stop=stop,
        temperature=temperature,
        top_p=top_p,
        n=n,
        tools=build_tools(tools),
        tool_format=tool_format,  # type: ignore
        stream=stream,
        response_format=response_format,  # type: ignore
        response_model=response_model,  # type: ignore
    )

    generation = Generation[M](
        backend_server,
        create_args,
        max_relay_rounds=max_relay_rounds,
        mock_response=mock_response,
        _ctx=_ctx,
        **kwargs,
    )

    @_langsmith_traceable(name=generation.id, metadata={"appl": "gen"})  # type: ignore
    def langsmith_trace(*args: Any, **kwargs: Any) -> None:
        pass

    langsmith_trace(backend_server, create_args, _ctx=_ctx, **kwargs)
    return generation


# def serialize(obj: Any) -> Any:
#     if hasattr(obj, "serialize"):
#         return obj.serialize()
#     else:
#         raise TypeError("Object of type '%s' is not serializable" % type(obj).__name__)
