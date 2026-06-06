import json
import os
import shutil
import sys
import time
from typing import Any, Callable, Generator, List, Literal, Optional, Union

from litellm import CustomStreamWrapper, completion_cost, stream_chunk_builder
from litellm.exceptions import NotFoundError
from litellm.types.utils import Delta, Function, ModelResponse
from litellm.types.utils import Message as LiteLLMMessage
from loguru import logger
from openai import Stream
from openai.types import CompletionUsage
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import (
    ChoiceDelta,
    ChoiceDeltaToolCallFunction,
)
from pydantic import BaseModel, Field, model_validator
from rich.live import Live
from rich.panel import Panel
from rich.syntax import Syntax
from termcolor import colored
from termcolor._types import Color

from .globals import global_vars
from .tool import ToolCall
from .types import ResponseType
from .utils import get_live, make_panel, split_last, stop_live, strip_for_continue


class ReasoningContent(BaseModel):
    """The content of the reasoning."""

    content: str = Field(..., description="The content of the reasoning")

    def __str__(self) -> str:
        return self.content


class CompletionResponse(BaseModel):
    """A class wrapping the response from the LLM model.

    For a streaming response, it tracks the chunks of the response and
    builds the complete response when the streaming is finished.
    """

    raw_response: Any = Field(None, description="The raw response from the model")
    """The raw response from the model."""
    cost: Optional[float] = Field(None, description="The cost of the completion")
    """The cost of the completion."""
    usage: Optional[CompletionUsage] = Field(
        None, description="The usage of the completion"
    )
    """The usage of the completion."""
    finish_reason: Optional[str] = Field(
        None, description="The reason why the completion is finished for the top-choice"
    )
    """The reason why the completion is finished for the top-choice."""
    num_raw_completions: int = Field(1, description="The number of raw completions")
    """The number of raw completions."""
    chunks: List[Union[ModelResponse, ChatCompletionChunk]] = Field(
        [], description="The chunks of the response when streaming"
    )
    """The chunks of the response when streaming."""
    is_stream: bool = Field(False, description="Whether the response is a stream")
    """Whether the response is a stream."""
    is_finished: bool = Field(
        False, description="Whether the response stream is finished"
    )
    """Whether the response stream is finished."""
    post_finish_callbacks: List[Callable] = Field(
        [], description="The post finish callbacks"
    )
    """The post finish callbacks."""
    response_model: Any = Field(
        None, description="The BaseModel's subclass specifying the response format."
    )
    """The BaseModel's subclass specifying the response format."""
    response_obj: Any = Field(
        None, description="The response object of response model, could be a stream"
    )
    """The response object of response model, could be a stream."""
    message: Optional[str] = Field(
        None, description="The top-choice message from the completion"
    )
    """The top-choice message from the completion."""
    reasoning_content: Optional[str] = Field(
        None, description="The reasoning content from the completion if exists"
    )
    """The reasoning content from the completion if exists."""
    tool_calls: List[ToolCall] = Field([], description="The tool calls")
    """The tool calls."""

    @model_validator(mode="after")
    def _post_init(self) -> "CompletionResponse":
        self._finished_raw_response = None

        if isinstance(self.raw_response, (CustomStreamWrapper, Stream)):
            # ? supports for Async Steam?
            self.is_stream = True
        else:
            self._finish(self.raw_response)  # type: ignore
        return self

    def set_response_obj(self, response_obj: Any) -> None:
        """Set the response object."""
        self.response_obj = response_obj

    @property
    def ensure_finished(self) -> "CompletionResponse":
        """Ensure the response is finished."""
        if self.is_finished:
            return self
        self.streaming()
        return self

    @property
    def finished_raw_response(self) -> Union[ModelResponse, ChatCompletion]:
        """The completed raw response from the model. This will block until the response is finished."""
        if self.is_finished:
            return self._finished_raw_response  # type: ignore
        self.streaming()  # ? when we should set display to False?
        assert self.is_finished, "Response should be finished after streaming"
        return self._finished_raw_response  # type: ignore

    @property
    def results(self) -> Any:
        """The results of the response.

        Returns:
            message (str):
                The message if the response is a text completion.
            tool_calls (List[ToolCall]):
                The tool calls if the response is a list of tool calls.
            response_obj (Any):
                The object if the response is a response object.
        """
        if self.is_stream and not self.is_finished:
            self.streaming()  # display the stream and finish the response
        results: Any = self.message
        if self.response_obj is not None:
            results = self.response_obj
        elif len(self.tool_calls):
            results = self.tool_calls
        return results

    @property
    def type(self) -> ResponseType:
        """The type of the response."""
        if not self.is_finished:
            return ResponseType.UNFINISHED
        if self.response_model is not None and self.response_obj is not None:
            return ResponseType.OBJECT
        if len(self.tool_calls):
            return ResponseType.TOOL_CALL
        return ResponseType.TEXT

    def update(
        self, other: "CompletionResponse", split_marker: str = "\n"
    ) -> "CompletionResponse":
        """Update the response with the information contained in the other response."""
        if not self.is_finished:
            raise ValueError("Cannot update unfinished response")
        if self.type != other.type:
            raise ValueError(
                f"Cannot update response with type {self.type} "
                f"with another response of type {other.type}"
            )
        if self.type != ResponseType.TEXT:
            raise NotImplementedError("Not supported for non-text response")
        if self.message is None or other.message is None:
            raise ValueError("Not supported for empty message when updating")

        stripped_message = strip_for_continue(self.message)
        _, last_part = split_last(stripped_message, split_marker)
        message = other.message
        if last_part in message:
            # truncate the overlapping part, patch the messages together
            self.message = (
                stripped_message + message[message.index(last_part) + len(last_part) :]
            )
        else:
            self.message += message  # extend the message
            logger.warning(
                f"Last part {last_part} not found in the message. "
                "Appending the message directly."
            )

        def as_list(obj: Any) -> List[Any]:
            if isinstance(obj, list):
                return obj
            return [obj]

        for k in ["finish_reason", "response_model", "response_obj", "tool_calls"]:
            if getattr(self, k) is None:
                setattr(self, k, getattr(other, k))
        self.raw_response = as_list(self.raw_response) + as_list(other.raw_response)
        self.chunks += other.chunks
        self.num_raw_completions += other.num_raw_completions
        if other.cost is not None:
            self.cost = (self.cost or 0) + other.cost
        if other.usage is not None:

            def merge_usage(usage1: BaseModel, usage2: BaseModel) -> None:
                """Merge the usage from two responses recursively."""
                for k, v in usage2.model_dump().items():
                    if isinstance(v, int) or isinstance(v, float):
                        if hasattr(usage1, k):
                            setattr(usage1, k, getattr(usage1, k) + v)
                    elif isinstance(v, BaseModel):
                        merge_usage(getattr(usage1, k), v)

            merge_usage(self.usage, other.usage)  # type: ignore

        return self

    def streaming(
        self,
        display: Optional[str] = None,
        title: str = "APPL Streaming",
        display_prefix_content: str = "",
        live: Optional[Live] = None,
    ) -> "CompletionResponse":
        """Stream the response object and finish the response."""
        if not self.is_stream:
            raise ValueError("Cannot iterate over non-streaming response")
        if self.is_finished:
            return self

        if self.response_obj is not None:
            target = self.response_obj
        else:
            target = self.format_stream()
        # print(target)

        streaming_display_mode = (
            display or global_vars.configs.settings.logging.display.streaming_mode
        )
        if streaming_display_mode == "live":
            start_time = time.time()

            def panel(
                content: str, iter_index: Optional[int] = None, truncate: bool = False
            ) -> Panel:
                style = "magenta"
                display_title = title
                if iter_index is not None:
                    time_elapsed = time.time() - start_time
                    avg_iters_per_sec = (iter_index + 1) / time_elapsed
                    stream_info = (
                        f"[{time_elapsed:.3f}s ({avg_iters_per_sec:.2f} it/s)]"
                    )
                    display_title += f" - {stream_info}"
                return make_panel(
                    content, title=display_title, style=style, truncate=truncate
                )

            if live is None:
                live = get_live()
                need_stop = True
            else:
                need_stop = False
            content = display_prefix_content
            is_reasoning = False
            for i, chunk in enumerate(iter(target)):
                if isinstance(chunk, ReasoningContent):
                    if not is_reasoning:
                        content += "\n===== START REASONING =====\n"
                        is_reasoning = True
                    content += chunk.content
                elif isinstance(chunk, BaseModel):
                    content = json.dumps(chunk.model_dump(), indent=2)
                else:
                    if is_reasoning:
                        content += "\n===== END REASONING =====\n"
                        is_reasoning = False
                    content += str(chunk)
                live.update(panel(content, i, truncate=True))
                # live.refresh()  # might be too frequent
            # display untruncated content at the end
            live.update(panel(content, i))
            live.refresh()
            if need_stop:
                stop_live()
        elif streaming_display_mode == "print":
            last_content = ""
            is_reasoning = False

            def eprint(content: str, color: Optional[Color] = None) -> None:
                print(colored(content, color) if color else content, end="")
                sys.stdout.flush()

            eprint("\n===== START APPL STREAMING =====\n", color="magenta")
            self.register_post_finish_callback(
                lambda _: eprint("\n===== END APPL STREAMING =====\n", color="magenta"),
                order="first",
            )
            eprint(display_prefix_content, color="grey")
            for chunk in iter(target):
                if isinstance(chunk, ReasoningContent):
                    if not is_reasoning:
                        eprint("\n===== START REASONING =====\n", color="yellow")
                        is_reasoning = True
                    eprint(chunk.content, color="dark_grey")
                elif isinstance(chunk, BaseModel):
                    content = json.dumps(chunk.model_dump(), indent=2)
                    if last_content in content:
                        eprint(
                            content[content.index(last_content) :], color="dark_grey"
                        )
                    else:
                        eprint(content, color="dark_grey")
                    last_content = content
                else:
                    if is_reasoning:
                        eprint("\n===== END REASONING =====\n", color="yellow")
                        is_reasoning = False
                    eprint(str(chunk), color="dark_grey")

        elif streaming_display_mode == "none":
            for chunk in iter(target):
                pass
        else:
            raise ValueError(
                f"Unknown display mode for streaming: {streaming_display_mode}, only 'live', 'print' and 'none' are supported"
            )
        if self.response_obj is not None:
            self.set_response_obj(chunk)
        return self

    def register_post_finish_callback(
        self,
        callback: Callable,
        order: Literal["first", "last"] = "last",
    ) -> None:
        """Register a post finish callback.

        The callback will be called after the response is finished.
        """
        if self.is_finished:
            callback(self)
        else:
            if order not in ["first", "last"]:
                raise ValueError(
                    f"Unknown order argument: {order}, only 'first' and 'last' are supported"
                )
            if order == "last":
                self.post_finish_callbacks.append(callback)
            else:
                self.post_finish_callbacks.insert(0, callback)

    def format_stream(self) -> Generator[Union[str, ReasoningContent], None, None]:
        """Format the stream response as a text generator."""
        suffix = ""
        for chunk in iter(self):
            # chunk: Union[ModelResponse, ChatCompletionChunk]
            delta: Union[Delta, ChoiceDelta] = chunk.choices[0].delta  # type: ignore

            if delta is not None and isinstance(delta, Delta):
                if provider_specific_fields := delta.get(
                    "provider_specific_fields", None
                ):
                    if reasoning_content := provider_specific_fields.get(
                        "reasoning_content", None
                    ):
                        if delta.content:
                            raise ValueError(
                                "Reasoning content should not be provided when content is also provided"
                            )
                        yield ReasoningContent(content=reasoning_content)

                if delta.content is not None:
                    yield delta.content
                elif getattr(delta, "tool_calls", None):
                    f: Union[Function, ChoiceDeltaToolCallFunction] = delta.tool_calls[
                        0
                    ].function  # type: ignore
                    if f.name is not None:
                        if suffix:
                            yield f"{suffix}, "
                        yield f"{f.name}("
                        suffix = ")"
                    if f.arguments is not None:
                        yield f.arguments
        yield suffix

    def _finish(self, response: Any) -> None:
        if self.is_finished:
            logger.warning("Response already finished. Ignoring finish call.")
            return
        self.is_finished = True
        self._finished_raw_response = response
        self.usage = getattr(response, "usage", None)
        self.cost = 0.0
        try:
            self.cost = completion_cost(response)
        except Exception:
            pass
        # parse the message and tool calls
        if isinstance(response, (ModelResponse, ChatCompletion)):
            message = response.choices[0].message  # type: ignore
            self.finish_reason = response.choices[0].finish_reason
            if tool_calls := getattr(message, "tool_calls", None):
                for call in tool_calls:
                    self.tool_calls.append(ToolCall.from_openai_tool_call(call))
            elif message.content is not None:
                self.message = message.content
                if isinstance(message, LiteLLMMessage):
                    if provider_specific_fields := message.get(
                        "provider_specific_fields", None
                    ):
                        self.reasoning_content = provider_specific_fields.get(
                            "reasoning_content", None
                        )
            else:
                raise ValueError(f"Invalid response: {response}")
        elif response is None:
            logger.warning("Response is None, only used for testing")
        else:
            raise ValueError(f"Unknown response type: {type(response)}")

        # post finish hook
        for callback in self.post_finish_callbacks:
            try:
                callback(self)
            except Exception as e:
                logger.error(
                    f"Error when calling post finish callback {callback.__name__}: {e}"
                )
                raise e

    def _finish_stream(self) -> None:
        try:
            response = stream_chunk_builder(self.chunks)
        except Exception:
            logger.error("Error when building the response from the stream")
            raise
        self._finish(response)

    def __str__(self):
        if self.is_stream and not self.is_finished:
            return repr(self)
        return str(self.results)

    def __next__(self):
        if not self.is_stream:
            raise ValueError("Cannot iterate over non-streaming response")
        try:
            chunk = next(self.raw_response)
            self.chunks.append(chunk)
            return chunk
        except StopIteration:
            self._finish_stream()
            raise StopIteration

    def __iter__(self):
        if not self.is_stream:
            raise ValueError("Cannot iterate over non-streaming response")
        return self

    def __getattr__(self, name: str) -> Any:
        if not self.is_finished:
            logger.warning(
                f"Cannot get {name} attribute before the response is finished. "
                "Returning None."
            )
            return None
        return getattr(self.finished_raw_response, name)
