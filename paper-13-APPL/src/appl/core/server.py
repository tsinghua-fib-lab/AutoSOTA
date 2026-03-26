from abc import ABC, abstractmethod
from typing import Any, Callable, Literal, Optional, Sequence, Type, Union

from loguru import logger
from pydantic import BaseModel, Field
from typing_extensions import override

from .message import Conversation
from .response import CompletionResponse
from .tool import BaseTool, ToolCall
from .types import MaybeOneOrMany


class GenArgs(BaseModel):
    """Common arguments for generating a response from a model."""

    model: str = Field(..., description="The name of the backend model")
    messages: Conversation = Field(
        ..., description="The conversation to use as a prompt"
    )
    max_tokens: Optional[int] = Field(
        None, description="The maximum number of tokens to generate"
    )
    stop: MaybeOneOrMany[str] = Field(None, description="The stop sequence(s)")
    temperature: Optional[float] = Field(
        None, description="The temperature for sampling"
    )
    top_p: Optional[float] = Field(None, description="The nucleus sampling parameter")
    n: Optional[int] = Field(None, description="The number of choices to generate")
    tools: Sequence[BaseTool] = Field([], description="The tools can be used")
    tool_format: Literal["auto", "str"] = Field(
        "auto", description="The format for the tools"
    )
    stream: Optional[bool] = Field(None, description="Whether to stream the results")
    response_format: Optional[Union[dict, Type[BaseModel]]] = Field(
        None, description="OpenAI's argument specifies the response format."
    )
    response_model: Optional[Type[BaseModel]] = Field(
        None,
        description="instructor's argument specifies the response format as a Pydantic model.",
    )

    def preprocess(self, convert_func: Callable, is_openai: bool = False) -> dict:
        """Convert the GenArgs into a dictionary for creating the response."""
        # build dict, filter out the None values
        args = self.model_dump(exclude_none=True)

        # messages
        args["messages"] = convert_func(self.messages)

        # format the tool
        tools = self.tools
        tool_format = args.pop("tool_format")
        if len(tools):
            if tool_format == "auto":
                tool_format = "openai" if is_openai else "str"
            formatted_tools = []
            for tool in tools:
                tool_str: Any = None
                if tool_format == "openai":
                    tool_str = tool.openai_schema
                else:  # TODO: supports more formats
                    tool_str = str(tool)
                formatted_tools.append(tool_str)
            args["tools"] = formatted_tools
        else:
            args.pop("tools", None)
        return args


class BaseServer(ABC):
    """The base class for all servers.

    Servers are responsible for communicating with the underlying model.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The name of the model used by the server."""
        raise NotImplementedError

    @abstractmethod
    def _get_create_args(self, args: GenArgs, **kwargs: Any) -> dict:
        """Map the GenArgs to the arguments for creating the response."""
        raise NotImplementedError

    @abstractmethod
    def _convert(self, conversation: Conversation) -> Any:
        """Convert the conversation into prompt format for the model.

        Args:
            conversation (Conversation): The conversation to convert

        Returns:
            The prompt for the model in the format it expects
        """
        raise NotImplementedError

    @abstractmethod
    def _create(self, **kwargs: Any) -> CompletionResponse:
        """Create a CompletionResponse from the model with processed arguments.

        Args:
            kwargs: The arguments to pass to the model.

        Returns:
            CompletionResponse: The response from the model.
        """
        raise NotImplementedError

    def create(self, args: GenArgs, gen_id: str, **kwargs: Any) -> CompletionResponse:
        """Create a CompletionResponse from the model with given arguments.

        Args:
            args: The arguments for generating the response
            gen_id: The ID of the generation
            **kwargs: Additional keyword arguments
        Returns:
            The response from the model.
        """
        create_args = self._get_create_args(args, **kwargs)
        results = self._create(gen_id=gen_id, **create_args)
        return results

    @abstractmethod
    def close(self):
        """Close the server."""
        raise NotImplementedError


class DummyServer(BaseServer):
    """A dummy server for testing purposes."""

    @override
    @property
    def model_name(self) -> str:
        return "_dummy"

    def _get_create_args(self, args: GenArgs, **kwargs: Any) -> dict:  # type: ignore
        return kwargs

    def _convert(self, conversation: Conversation) -> Any:
        return conversation

    def _create(self, **kwargs) -> CompletionResponse:  # type: ignore
        message = kwargs.get("mock_response", "This is a dummy response")
        return CompletionResponse(message=message)  # type: ignore

    @override
    def close(self):
        pass
