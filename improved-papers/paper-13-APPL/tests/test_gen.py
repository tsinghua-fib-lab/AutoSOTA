from typing import List

import pytest

from appl import (
    AIRole,
    PromptRecords,
    SystemMessage,
    gen,
    ppl,
    records,
)

IMAGE_URL = "https://maproom.wpenginepowered.com/wp-content/uploads/OpenAI_Logo.svg_-500x123.png"


def test_gen():
    @ppl
    def func():
        "Hello"
        return gen("_dummy", mock_response="World")

    assert str(func()) == "World"

    @ppl
    def dummy():
        "Hello"
        return gen("_dummy")

    assert "dummy" in str(dummy())


def test_gen_outside_ppl():
    res = gen(
        "_dummy",
        messages=[{"role": "user", "content": "Hello"}],
        mock_response="World",
    )
    assert str(res) == "World"

    res = gen(
        "_dummy",
        messages=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "World"},
        ],
        mock_response="World",
    )
    assert str(res) == "World"

    res = gen(
        "_dummy",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": IMAGE_URL},
                    },
                ],
            }
        ],
        mock_response="OpenAI",
    )
    assert str(res) == "OpenAI"

    res = gen(
        "_dummy",
        messages=[
            {"role": "user", "content": "what's 1+1?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "t1",
                        "type": "function",
                        "function": {
                            "name": "add",
                            "arguments": '{"a": 1, "b": 1}',
                        },
                    }
                ],
            },
            {"role": "tool", "content": "2", "tool_call_id": "t1"},
        ],
        mock_response="2",
    )
    assert str(res) == "2"


def test_gen_tool_call_outside_ppl():
    tool_schema = {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["location"],
            },
        },
    }
    res = gen(
        "_dummy",
        messages=[{"role": "user", "content": "What's the weather in San Francisco?"}],
        tools=[tool_schema],
        mock_response="Sunny",
    )
    assert str(res) == "Sunny"
    res = gen(
        "_dummy",
        messages=[{"role": "user", "content": "What's the weather in San Francisco?"}],
        tools=tool_schema,
        mock_response="Sunny",
    )
    assert str(res) == "Sunny"


def test_chat_gen():
    name = "Test"
    p: List[PromptRecords] = []

    @ppl(auto_prime=True)
    def chat_gen():
        SystemMessage(f"Your name is {name}")
        reply = None
        while True:
            yield reply
            with AIRole():
                (reply := gen("_dummy", mock_response="World"))
            p.append(records())

    chat = chat_gen()
    assert str(chat.send("Hello")) == "World"
    print(p[0].as_convo().as_list())
    assert p[0].as_convo().as_list() == [
        {"role": "system", "content": "Your name is Test"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "World"},
    ]
    assert str(chat.send("Hello again")) == "World"
    assert p[1].as_convo().as_list() == [
        {"role": "system", "content": "Your name is Test"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "World"},
        {"role": "user", "content": "Hello again"},
        {"role": "assistant", "content": "World"},
    ]


def test_gen_error():
    with pytest.raises(ValueError) as excinfo:
        gen(mock_response="World")

    error_msg = str(excinfo.value)
    assert "PromptContext" in error_msg or "appl.init" in error_msg
