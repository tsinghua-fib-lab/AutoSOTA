from appl import gen, ppl
from appl.types import Generation, ReasoningContent


@ppl
def reasoning(x: int, y: int):
    f"think concisely, what's the result of {x}+{y}?"
    return gen("deepseek/deepseek-reasoner", stream=True)


def manual_stream(generation: Generation):
    is_reasoning = False
    for chunk in generation.text_stream:
        if isinstance(chunk, ReasoningContent):
            if not is_reasoning:
                print("<think>")
                is_reasoning = True
            print(chunk.content, end="", flush=True)
        else:
            if is_reasoning:
                print("</think>")
                is_reasoning = False
            print(chunk, end="", flush=True)

    print()


# APPL automatic streaming
print(reasoning(1, 2))
# manual streaming
# manual_stream(reasoning(1, 2))
