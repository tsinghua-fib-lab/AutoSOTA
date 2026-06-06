import pytest

from appl import AIRole, SystemRole, UserRole, ppl, records
from appl.core import (
    AIMessage,
    Conversation,
    PromptRecords,
    SystemMessage,
    ToolCall,
    UserMessage,
)


def test_message():
    sys = SystemMessage("You are a helpful assistant.")
    user = UserMessage("Hello, who are you")
    ai = AIMessage("I am a helpful assistant.")
    assert sys.is_system
    assert user.is_user
    assert ai.is_ai
    assert sys.get_dict() == {
        "role": "system",
        "content": "You are a helpful assistant.",
    }
    assert user.get_dict() == {"role": "user", "content": "Hello, who are you"}
    assert ai.get_dict() == {
        "role": "assistant",
        "content": "I am a helpful assistant.",
    }


def test_msg_role():
    @ppl
    def func() -> PromptRecords:
        "Hello"
        AIMessage("World")
        return records()

    assert func().as_convo().as_list() == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "World"},
    ]


def test_chat_example():
    @ppl
    def func() -> PromptRecords:
        SystemMessage("You are a helpful assistant.")
        for i in range(5):
            UserMessage("Hello")
            AIMessage("World")
        return records()

    expected = [{"role": "system", "content": "You are a helpful assistant."}]
    for i in range(5):
        expected.append({"role": "user", "content": "Hello"})
        expected.append({"role": "assistant", "content": "World"})
    assert func().as_convo().as_list() == expected


def test_system_msg():
    @ppl
    def func() -> PromptRecords:
        with SystemRole():
            "You are a helpful Assistant."
            "You should be helpful."
        with UserRole():
            "Hello"
        return records()

    assert func().as_convo().as_list() == [
        {
            "role": "system",
            "content": "You are a helpful Assistant.\nYou should be helpful.",
        },
        {"role": "user", "content": "Hello"},
    ]


def test_message_merge():
    @ppl
    def func() -> PromptRecords:
        SystemMessage("You are a helpful assistant.")
        UserMessage("Hello, who are you")
        AIMessage("I am ")
        AIMessage("a helpful ")
        AIMessage("assistant.")
        UserMessage("H")
        UserMessage("i")
        SystemMessage(" Add something to system prompt.")
        return records()

    assert func().as_convo().as_list() == [
        {
            "role": "system",
            "content": "You are a helpful assistant. Add something to system prompt.",
        },
        {"role": "user", "content": "Hello, who are you"},
        {"role": "assistant", "content": "I am a helpful assistant."},
        {"role": "user", "content": "Hi"},
    ]


def test_ai_tool_call():
    @ppl
    def func() -> PromptRecords:
        AIMessage(tool_calls=[ToolCall(id="1", name="add", args='{"x": 1, "y": 2}')])
        return records()

    assert func().as_convo().as_list() == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "add", "arguments": '{"x": 1, "y": 2}'},
                }
            ],
        },
    ]


def test_role_wrap():
    @ppl
    def func() -> PromptRecords:
        with AIRole():
            UserMessage("Hello")
            AIMessage("World")
        return records()

    assert func().as_convo().as_list() == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "World"},
    ]


def test_mix_role_usage():
    @ppl
    def func() -> PromptRecords:
        with AIRole():
            UserMessage("Hello")
            AIMessage("World")
            "Again"
        return records()

    assert func().as_convo().as_list() == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "WorldAgain"},
    ]


def test_multi_user():
    @ppl
    def func() -> PromptRecords:
        with UserRole("A"):
            "Hello"
        with UserRole("B"):
            "World"
        return records()

    assert func().as_convo().as_list() == [
        {"role": "user", "name": "A", "content": "Hello"},
        {"role": "user", "name": "B", "content": "World"},
    ]


def test_role_change():
    @ppl
    def func() -> PromptRecords:
        "Hello"
        with AIRole():
            "World"
        return records()

    assert func().as_convo().as_list() == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "World"},
    ]


def test_role_change_loop1():
    @ppl
    def func() -> PromptRecords:
        for i in range(5):
            "Hello"
            with AIRole():
                "World"
        return records()

    assert (
        func().as_convo().as_list()
        == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "World"},
        ]
        * 5
    )


def test_role_change_loop2():
    @ppl
    def func() -> PromptRecords:
        for i in range(3):
            "Hello"
            AIMessage("World")
        return records()

    assert (
        func().as_convo().as_list()
        == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "World"},
        ]
        * 3
    )


def test_role_change_loop3():
    @ppl
    def func() -> PromptRecords:
        "Hello"
        for i in range(5):
            with AIRole():
                "Foo"
            with UserRole():
                "Bar"
        return records()

    expected = [{"role": "user", "content": "Hello"}]
    for i in range(5):
        expected.append({"role": "assistant", "content": "Foo"})
        expected.append({"role": "user", "content": "Bar"})

    assert func().as_convo().as_list() == expected


def test_nested_role_error():
    with pytest.raises(ValueError) as excinfo:

        @ppl
        def func() -> PromptRecords:
            "Hello"
            for i in range(5):
                with UserRole():
                    "Foo"
                    with AIRole():
                        "Bar"
            return records()

        func()

    assert "Cannot start a new role" in str(excinfo.value)
