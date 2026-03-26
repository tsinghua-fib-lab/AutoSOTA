import time
from typing import Any

from pydantic import BaseModel, Field, create_model

from appl import Generation, as_tool, gen, ppl
from appl.core import CompletionResponse, ToolCall


def removed_keyword(d: Any, key: str) -> Any:
    if isinstance(d, list):
        return [removed_keyword(x, key) for x in d]
    if isinstance(d, dict):
        return {k: removed_keyword(v, key) for k, v in d.items() if k != key}
    return d


def get_openai_schema(include_desc=True, add_default=False):
    args_schema = {
        "type": "object",
        "properties": {
            "x": {"type": "integer"},
            "y": {"type": "integer"},
        },
        "required": ["x", "y"],
    }
    if include_desc:
        args_schema["properties"]["x"]["description"] = "first number"
        args_schema["properties"]["y"]["description"] = "second number"
    if add_default:
        args_schema["properties"]["y"]["default"] = 1
        args_schema["required"].remove("y")
    schema = {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two numbers together",
            "parameters": args_schema,
        },
    }
    return schema


def test_as_tool():
    def add(x: int, y: int) -> int:
        """Add two numbers together

        Args:
            x (int): first number
            y (int): second number
        Returns:
            int: sum of x and y
        Raises:
            ValueError: if x or y is not an integer
        """
        if not isinstance(x, int) or not isinstance(y, int):
            raise ValueError("x and y must be integers")
        return x + y

    tool = as_tool(add)
    assert tool(1, 2) == 3
    assert tool.__doc__ == add.__doc__
    assert removed_keyword(tool.openai_schema, "title") == get_openai_schema()
    assert removed_keyword(tool.returns.model_json_schema(), "title") == {
        "properties": {
            "returns": {"type": "integer", "description": "sum of x and y"},
        },
        "type": "object",
        "required": ["returns"],
    }
    assert tool.raises == [
        {"type": "ValueError", "desc": "if x or y is not an integer"}
    ]

    def add(x, y=1):
        """Add two numbers together

        Args:
            x (int): first number
            y (int): second number
        """

    tool = as_tool(add)
    assert removed_keyword(tool.openai_schema, "title") == get_openai_schema(
        add_default=True
    )

    def add(x: int, y: int):
        """Add two numbers together"""

    tool = as_tool(add)
    assert removed_keyword(tool.openai_schema, "title") == get_openai_schema(
        include_desc=False
    )


class AddArgs(BaseModel):
    x: int = Field(..., description="first number")
    y: int = Field(1, description="second number")


def test_args_schema():
    def add(args: AddArgs):
        """Add two numbers together"""
        return args.x + args.y

    tool = as_tool(add)
    params = {}
    params["args"] = (AddArgs, ...)
    assert (
        tool.params.model_json_schema()
        == create_model("parameters", **params).model_json_schema()
    )

    def add(args: AddArgs = AddArgs(x=1, y=2)):
        """Add two numbers together

        Args:
            args (AddArgs): arguments
        """
        return args.x + args.y

    tool = as_tool(add)
    params = {}
    params["args"] = (AddArgs, Field(AddArgs(x=1, y=2), description="arguments"))
    assert (
        tool.params.model_json_schema()
        == create_model("parameters", **params).model_json_schema()
    )


def test_tool_call_sequential():
    response = CompletionResponse(
        tool_calls=[
            ToolCall(id="1", name="add", args='{"x": 1, "y": 2, "t": 0.1}'),
            ToolCall(id="2", name="mul", args='{"x": 2, "y": 3}'),
            ToolCall(id="3", name="add", args='{"x": 3, "y": 4}'),
            ToolCall(id="4", name="add", args='{"x": 5, "y": 6}'),
        ],
    )

    call_args = []

    def add(x: int, y: int, t: float = 0.0) -> int:
        time.sleep(t)
        call_args.append((x, y))
        return x + y

    tools = [as_tool(add)]

    def filter_fn(tool_calls: list[ToolCall]) -> list[ToolCall]:
        return [tc for tc in tool_calls if tc.name == "add"]

    @ppl
    def func():
        res = gen("_dummy", tools=tools, mock_response=response)
        return [x.get_content() for x in res.run_tool_calls(filter_fn=filter_fn)]

    assert func() == [3, 7, 11]
    assert call_args == [(1, 2), (3, 4), (5, 6)]


def test_tool_call_parallel():
    response = CompletionResponse(
        tool_calls=[
            ToolCall(id="1", name="add", args='{"x": 1, "y": 2}'),
            ToolCall(id="3", name="add", args='{"x": 3, "y": 4}'),
        ],
    )

    t = 0.2

    def add(x: int, y: int) -> int:
        time.sleep(t)
        return x + y

    tools = [as_tool(add)]

    @ppl
    def func():
        res = gen(tools=tools, mock_response=response)
        return [x.get_content() for x in res.run_tool_calls(parallel="thread")]

    start_time = time.time()
    assert func() == [3, 7]
    assert time.time() - start_time < t + 0.1
