from typing import Optional

from appl import gen, ppl


@ppl
def func(
    stream: bool = True, max_tokens: Optional[int] = None, max_relay_rounds: int = 10
):
    # adopted from https://cookbook.openai.com/examples/how_to_stream_completions
    "Count to 50, with a comma between each number and no newlines. E.g., 1, 2, 3, ..."
    return gen(
        stream=stream,
        temperature=0.0,
        max_tokens=max_tokens,
        max_relay_rounds=max_relay_rounds,
    )


# content = func(stream=False)
# print(f"Content: {content}")

content = func(stream=True)
print(f"Content: {content}")

content = func(stream=True, max_tokens=25, max_relay_rounds=10)
print(f"Content: {content}")
