import os

import pytest

from appl import gen, ppl

TEST_REASONING_MODEL = "deepseek/deepseek-reasoner"


@ppl
def add():
    "1+2="
    return str(gen(TEST_REASONING_MODEL, max_tokens=10)).strip()


try:
    add()
    reason = None
    deepseek_api_callable = True
except Exception as e:
    reason = str(e)
    deepseek_api_callable = False

skip_tests = os.environ.get("SKIP_LLM_CALL_TESTS") == "1"
pytestmark = [
    pytest.mark.skipif(skip_tests, reason="Skipped due to setting"),
    pytest.mark.skipif(
        not deepseek_api_callable,
        reason=f"Error encountered when calling deepseek API: {reason}",
    ),
]


def test_reasoning():
    @ppl
    def func(x: int, y: int, stream: bool = False):
        f"think concisely, what's the result of {x}+{y}?"
        return gen("deepseek/deepseek-reasoner", stream=stream)

    res1 = func(1, 2, stream=False)
    assert "3" in res1.message
    assert res1.response.reasoning_content is not None

    res2 = func(1, 2, stream=True)
    assert "3" in res2.message
    # TODO: enable this assert after https://github.com/BerriAI/litellm/pull/8009 being merged
    # assert res2.response.reasoning_content is not None
