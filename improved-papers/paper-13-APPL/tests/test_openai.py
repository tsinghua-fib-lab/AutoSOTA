"""Testing openai API calls

NOTE: The tests could cost money if you have an openai key set.
NOTE: Some tests might fail due to randomness in the API.
You can skip these tests by setting the environment variable SKIP_MODEL_CALL_TESTS=1
"""

import os
import sys

import pytest
from loguru import logger
from pydantic import BaseModel

from appl import AIRole, Audio, Image, as_tool, gen, ppl
from appl.const import NEWLINE

# NOTE: init here could influence other tests in other files.

TEST_OPENAI_MODEL = "gpt-4o-mini"


@ppl
def add():
    "1+2="
    return str(gen(TEST_OPENAI_MODEL, max_tokens=10)).strip()


try:
    add()
    reason = None
    openai_api_callable = True
except Exception as e:
    reason = str(e)
    openai_api_callable = False

skip_tests = os.environ.get("SKIP_LLM_CALL_TESTS") == "1"
pytestmark = [
    pytest.mark.skipif(skip_tests, reason="Skipped due to setting"),
    pytest.mark.skipif(
        not openai_api_callable,
        reason=f"Error encountered when calling openai API: {reason}",
    ),
]


# @pytest.mark.skipif(skip_tests, reason="Skipped due to setting")
def test_openai():
    assert "3" in add()


def test_tool_call():
    def special_number(x: int) -> int:
        """A function to calculate the special number of an integer."""
        return x**2 + 1

    tools = [as_tool(special_number)]

    @ppl
    def func(x: int):
        f"What's the special number of {x}?"
        res = gen(tools=tools)
        return [x.get_content() for x in res.run_tool_calls()]

    assert func(5) == [special_number(5)]


IMAGE_URL = "https://maproom.wpenginepowered.com/wp-content/uploads/OpenAI_Logo.svg_-500x123.png"
AUDIO_URL = "https://cdn.openai.com/API/docs/audio/alloy.wav"


def test_image():
    @ppl
    def query():
        "Look, it's a commercial logo."
        Image(IMAGE_URL)
        "What's the text on the image? "
        "Your output should be in the format: The text on the image is: ..."
        return gen(TEST_OPENAI_MODEL, stop=NEWLINE)

    assert "OpenAI" in str(query())


def test_image_outside_ppl():
    assert "OpenAI" in str(
        gen(
            TEST_OPENAI_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Look, it's a commercial logo."},
                        {"type": "image_url", "image_url": {"url": IMAGE_URL}},
                        {
                            "type": "text",
                            "text": "What's the text on the image?\nYour output should be in the format: The text on the image is: ...",
                        },
                    ],
                }
            ],
            stop=NEWLINE,
        )
    )


def test_audio():
    @ppl
    def query():
        "what is the text in the audio?"
        Audio.from_url(AUDIO_URL, "wav")
        return gen("gpt-4o-audio-preview")

    assert (
        "The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years."
        in str(query())
    )


def test_response_format():
    class Result(BaseModel):
        result: int

    @ppl
    def func():
        "1+1="
        return gen(TEST_OPENAI_MODEL, response_format=Result).response_obj

    assert func().result == 2

    @ppl
    def func2():
        "1+1="
        return gen(TEST_OPENAI_MODEL, response_format=Result, stream=True).response_obj

    assert func2().result == 2


def test_auto_continue():
    @ppl
    def func():
        "Count from 1 to 50, in a format like 1, 2, ..."
        "BEGIN: "
        return gen(TEST_OPENAI_MODEL, max_tokens=100, max_relay_rounds=5)

    assert "45, 46, 47" in str(func())
