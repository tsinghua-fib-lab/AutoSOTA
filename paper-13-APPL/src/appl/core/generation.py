from __future__ import annotations

import json
import threading
from typing import (
    Any,
    Callable,
    Generator,
    Generic,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

from loguru import logger
from pydantic import BaseModel
from rich.live import Live

from .context import PromptContext
from .globals import (
    get_thread_local,
    global_vars,
    inc_thread_local,
    set_thread_local,
)
from .message import AIMessage, BaseMessage, ToolMessage, UserMessage
from .promptable import Promptable
from .response import CompletionResponse, ReasoningContent
from .server import BaseServer, GenArgs
from .tool import BaseTool, ToolCall
from .trace import GenerationInitEvent, GenerationResponseEvent, add_to_trace
from .types import (
    CallFuture,
    ExecutorType,
    MessageRole,
    MessageRoleType,
    ResponseType,
    String,
    StringFuture,
)
from .utils import get_live, split_last, stop_live, strip_for_continue

M = TypeVar("M")
APPL_GEN_NAME_PREFIX_KEY = "_appl_gen_name_prefix"
LAST_LINE_MARKER = "<last_line>"
LAST_PART_MARKER = "<last_part>"


def set_gen_name_prefix(prefix: str) -> None:
    """Set the prefix for generation names in the current thread."""
    set_thread_local(APPL_GEN_NAME_PREFIX_KEY, prefix)


def get_gen_name_prefix() -> Optional[str]:
    """Get the prefix for generation names in the current thread."""
    gen_name_prefix = get_thread_local(APPL_GEN_NAME_PREFIX_KEY, None)
    if gen_name_prefix is None:
        thread_name = threading.current_thread().name
        if thread_name != "MainThread":
            gen_name_prefix = thread_name
    return gen_name_prefix


class Generation(Generic[M]):
    """Represents a generation call to the model."""

    def __init__(
        self,
        server: BaseServer,
        args: GenArgs,
        *,
        max_relay_rounds: int = 0,
        mock_response: Optional[Union[CompletionResponse, str]] = None,
        llm_executor_type: ExecutorType = ExecutorType.LLM_THREAD_POOL,
        lazy_eval: bool = False,
        _ctx: Optional[PromptContext] = None,
        **kwargs: Any,
        # kwargs used for extra args for the create method
    ) -> None:
        """Initialize the Generation object.

        Args:
            server: An LLM server where the generation request will be sent.
            args: The arguments of the generation call.
            max_relay_rounds: the maximum number of relay rounds to continue the unfinished text generation.
            mock_response: A mock response for the generation call.
            llm_executor_type: The type of the executor to run the LLM call.
            lazy_eval: If True, the generation call will be evaluated lazily.
            _ctx: The prompt context filled automatically by the APPL function.
            **kwargs: Extra arguments for the generation call.
        """
        # name needs to be unique and ordered, so it has to be generated in the main thread
        gen_name_prefix = get_gen_name_prefix()
        # take the value before increment
        self._cnt = inc_thread_local(f"{gen_name_prefix}_gen_cnt") - 1
        if gen_name_prefix is None:
            self._id = f"@gen_{self._cnt}"
        else:
            self._id = f"@{gen_name_prefix}_gen_{self._cnt}"

        self._server = server
        self._model_name = server.model_name
        self._args = args
        self._max_relay_rounds = max_relay_rounds
        self._mock_response = mock_response
        self._llm_executor_type = llm_executor_type
        self._lazy_eval = lazy_eval
        self._ctx = _ctx
        self._extra_args = kwargs
        self._num_raw_completions = 0
        self._cached_response: Optional[CompletionResponse] = None

        add_to_trace(GenerationInitEvent(name=self.id))
        log_llm_call_args = global_vars.configs.settings.logging.display.llm_call_args
        if log_llm_call_args:
            logger.info(
                f"Call generation [{self.id}] with args: {args} and kwargs: {kwargs}"
            )

        if isinstance(mock_response, CompletionResponse):

            def get_response() -> CompletionResponse:
                return mock_response

            self._call = self._wrap_response(get_response)
        else:
            if mock_response:
                # use litellm's mock response
                kwargs.update({"mock_response": mock_response})
            self._call = self._wrap_response(self._call_llm())

        # tools
        self._tools: Sequence[BaseTool] = args.tools
        self._name2tools = {tool.name: tool for tool in self._tools}

    def _call_llm(self) -> CallFuture[CompletionResponse]:
        """Call the LLM server asynchronously to get the completion response."""
        self._num_raw_completions += 1
        return CallFuture(
            self._server.create,
            executor_type=self._llm_executor_type,
            lazy_eval=self._lazy_eval,
            args=self._args,
            gen_id=f"{self.id}_raw_{self._num_raw_completions - 1}",
            **self._extra_args,
        )

    def _continue_llm(
        self, results: CompletionResponse, live: Optional[Live] = None
    ) -> CompletionResponse:
        assert results.message is not None, "Not support continue for empty message"

        cutoff_content = strip_for_continue(results.message)
        continue_prompt = global_vars.configs.prompts.continue_generation
        continue_prompt_alt = global_vars.configs.prompts.continue_generation_alt

        # Choose a proper split marker for the continuation
        for split_marker in ["\n", " ", ","]:
            content, last_part = split_last(cutoff_content, split_marker)
            if content is not None:  # found split_marker in the content
                prompt = (
                    continue_prompt if split_marker == "\n" else continue_prompt_alt
                )
                marker = LAST_LINE_MARKER if split_marker == "\n" else LAST_PART_MARKER
                break
        marked_cutoff_content = f"{content}{split_marker}{marker}{last_part}{marker}"
        prompt = prompt.format(last_marker=marker)

        messages = self._args.messages
        messages.append(AIMessage(content=marked_cutoff_content))
        messages.append(UserMessage(content=prompt))
        # print(messages, "\n") # DEBUG

        # call the LLM again and wait for the result
        response = self._call_llm().result()
        if response.type == ResponseType.UNFINISHED:
            response.streaming(
                title=f"Continue generation [{self.id}]",
                display_prefix_content=marked_cutoff_content + "\n",
                live=live,
            )

        # pop the last two messages
        for _ in range(2):
            messages.pop()

        results.update(response, split_marker)
        return response

    def _wrap_response(
        self, get_response: Callable[[], CompletionResponse]
    ) -> Callable[[], CompletionResponse]:
        """Wrap the LLM calls to address incomplete completion."""

        def inner() -> CompletionResponse:
            if self._cached_response is not None:
                return self._cached_response

            log_llm_usage = global_vars.configs.settings.logging.display.llm_usage
            log_llm_response = global_vars.configs.settings.logging.display.llm_response
            log_llm_cost = global_vars.configs.settings.logging.display.llm_cost

            results = response = get_response()

            if self._max_relay_rounds > 0:
                live = None
                streaming_mode = (
                    global_vars.configs.settings.logging.display.streaming_mode
                )
                need_live = self._args.stream and streaming_mode == "live"
                if response.type == ResponseType.UNFINISHED:
                    if need_live:
                        live = get_live()
                    response.streaming(title=f"Generation [{self.id}]", live=live)

                for i in range(self._max_relay_rounds):
                    if response.finish_reason in ["length"]:
                        generated_chars = len(results.message or "")
                        logger.info(
                            f"[Round {i + 1}/{self._max_relay_rounds}, "
                            f"generated {generated_chars} chars] "
                            f"Generation [{self.id}] was cut off due to max_tokens, "
                            "automatically continue the generation."
                        )
                        if need_live and live is None:
                            live = get_live()
                        response = self._continue_llm(results, live=live)
                    else:
                        break

                if live is not None:
                    stop_live()

            def handle_results(results: CompletionResponse) -> None:
                if log_llm_response:
                    logger.info(f"Generation [{self.id}] results: {results}")
                if results.usage and log_llm_usage:
                    logger.info(f"Generation [{self.id}] token usage: {results.usage}")

                num_requests = global_vars.inc(
                    "num_requests", key=self._model_name, delta=1
                )
                if log_llm_cost:
                    currency = getattr(self._server, "_cost_currency", "USD")
                    if self._mock_response is not None:
                        logger.info(
                            "Mock response, estimated cost for real request: "
                            f"{results.cost:.4f} {currency}"
                        )
                    elif results.cost is None:
                        logger.warning(
                            f"No cost information for generation [{self.id}]"
                        )
                    else:
                        total_cost = global_vars.inc(
                            "api_cost", key=self._model_name, delta=results.cost
                        )
                        logger.info(
                            f"API cost for request [{self.id}]: {results.cost:.4f}, "
                            f"Total cost on {self._model_name}: {total_cost:.4f} {currency}. "
                            f"Total requests on {self._model_name}: {num_requests}."
                        )
                create_args = self._server._get_create_args(
                    self._args, **self._extra_args
                )
                dump_args = create_args.copy()
                for k, v in dump_args.items():
                    if k in ["response_format", "response_model"]:
                        if isinstance(v, type) and issubclass(v, BaseModel):
                            dump_args[k] = json.dumps(v.model_json_schema(), indent=4)

                add_to_trace(
                    GenerationResponseEvent(
                        name=self.id, args=dump_args, ret=str(results)
                    )
                )

            results.register_post_finish_callback(handle_results)

            self._cached_response = results
            return results

        return inner

    @property
    def id(self) -> str:
        """The unique ID of the generation."""
        return self._id

    @property
    def response(self) -> CompletionResponse:
        """The response of the generation call."""
        # NOTE: the result of the call should be cached
        return self._call()

    @property
    def finished_response(self) -> CompletionResponse:
        """The finished completion response of the generation call."""
        return self.response.ensure_finished

    @property
    def response_type(self) -> ResponseType:
        """The type of the response."""
        return self.response.type

    @property
    def is_message(self) -> bool:
        """Whether the response is a text message."""
        return self.response_type == ResponseType.TEXT

    @property
    def is_tool_call(self) -> bool:
        """Whether the response is a tool call."""
        return self.response_type == ResponseType.TOOL_CALL

    @property
    def is_obj(self) -> bool:
        """Whether the response is an object."""
        return self.response_type == ResponseType.OBJECT

    @property
    def message(self) -> Optional[str]:
        """The message of the response."""
        return self.finished_response.message

    @property
    def tool_calls(self) -> List[ToolCall]:
        """The tool calls of the response."""
        return self.finished_response.tool_calls

    @property
    def response_obj(self) -> M:
        """The object of the response."""
        return self.finished_response.response_obj

    @property
    def results(self) -> Union[M, str, List[ToolCall]]:
        """The results of the response."""
        return self.finished_response.results

    @property
    def str_future(self) -> StringFuture:
        """The StringFuture representation of the response."""
        return StringFuture(self)

    @property
    def text_stream(
        self,
    ) -> Generator[Union[str, ReasoningContent, BaseModel], None, None]:
        """Get the response of the generation as a text stream."""
        return self.response.format_stream()

    def _call_tool(
        self,
        name: str,
        args: str,
        parallel: bool = False,
        executor_type: ExecutorType = ExecutorType.GENERAL_THREAD_POOL,
    ) -> Any:
        try:
            kwargs = json.loads(args)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing args: {args}") from e
        args_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
        if global_vars.configs.settings.logging.display.tool_calls:
            logger.info(f"Running tool call: {name}({args_str})")

        if name not in self._name2tools:
            raise ValueError(f"Error: Tool {name} not found")
        tool = self._name2tools[name]
        try:
            if parallel:
                res = CallFuture(tool, executor_type=executor_type, **kwargs)
            else:
                res = tool(**kwargs)
        except Exception as e:
            raise RuntimeError(f"Error running tool call: {name}({args_str})") from e

        return res

    def run_tool_calls(
        self,
        filter_fn: Optional[Callable[[List[ToolCall]], List[ToolCall]]] = None,
        parallel: bool = False,
        executor_type: ExecutorType = ExecutorType.GENERAL_THREAD_POOL,
        log_results: Optional[bool] = None,
    ) -> List[ToolMessage]:
        """Run all tool calls in the generation and return the results.

        Args:
            filter_fn:
                A function that takes a list of ToolCall objects and returns
                a filtered list of ToolCall objects. This function can be
                used to filter the tool calls that will be run.
            parallel: If True, run the tool calls in parallel. Default to False.
            executor_type:
                The type of the executor to run the tool calls, can be
                "general_thread_pool", "general_process_pool", "new_thread" or
                "new_process".
            log_results:
                If True, log the results of the tool calls. Note This will wait for
                the results to be ready. Default to use the setting in configs.

        Returns:
            A list of ToolMessage objects.
        """
        if not self.is_tool_call:
            raise ValueError("Error: The Generation is not a tool call")
        if log_results is None:
            log_results = global_vars.configs.settings.logging.display.tool_results
        tool_calls = self.tool_calls
        if filter_fn:
            tool_calls = filter_fn(tool_calls)
        messages = []
        for tc in tool_calls:
            role = MessageRole(MessageRoleType.TOOL, tc.name)
            try:
                tool_result = self._call_tool(
                    tc.name, tc.args, parallel=parallel, executor_type=executor_type
                )
                msg = ToolMessage(
                    tool_result, role=role, tool_call_id=tc.id, has_error=False
                )
            except Exception as e:
                logger.error(f"Error running tool call: {tc.name}({tc.args})")
                logger.error(e)
                msg = ToolMessage(str(e), role=role, tool_call_id=tc.id, has_error=True)
            messages.append(msg)
        if log_results:  # this will wait for the results to be ready
            for msg in messages:
                logger.info(f"Tool call result: {msg}")
        return messages

    def as_prompt(self) -> Union[AIMessage, StringFuture]:
        """Get the response of the generation as a promptable object."""
        if self._args.tools:
            if self.is_tool_call:
                return AIMessage(tool_calls=self.tool_calls)
        # return a future object of value: str(self._call()), without blocking
        return StringFuture(CallFuture(self._call))

    def __add__(self, other: Union[String, "Generation"]) -> StringFuture:
        # Assume generation is a string
        if isinstance(other, Generation):
            return self.str_future + other.str_future
        elif isinstance(other, (str, StringFuture)):
            return self.str_future + other
        raise TypeError(
            f"unsupported operand type(s) for +: 'Generation' and '{type(other)}'"
        )

    def __radd__(self, other: String) -> StringFuture:
        # Assume generation is a string
        if isinstance(other, (str, StringFuture)):
            return other + self.str_future
        raise TypeError(
            f"unsupported operand type(s) for +: '{type(other)}' and 'Generation'"
        )

    def __getattr__(self, name: str) -> Any:
        assert name != "response", "Internal Error within self.response"
        return getattr(self.response, name)

    def __str__(self) -> str:
        return str(self.response.results)

    def __repr__(self) -> str:
        return f"Generation(id={self.id})"

    def __call__(self):
        """Get the response of the generation call."""
        return self.response
