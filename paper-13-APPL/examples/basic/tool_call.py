import sympy

from appl import AIMessage, Generation, convo, gen, ppl, records


def is_lucky(x: int) -> bool:
    """Determine whether the input number is a lucky number.

    Args:
        x (int): The input number to be checked.

    Returns:
        bool: True if the number is a lucky number, False otherwise.
    """
    return sympy.isprime(x + 3)


@ppl
def func(x):
    f"Is {x} a lucky number?"

    # Initiate the generation with tool `is_lucky``,
    # which is built into a tool by automatically extracting
    # information from the function signature and docstring.
    # And then store the tool call messages into the prompt
    (actions := gen(tools=[is_lucky]))

    if actions.is_tool_call:  # LLM choose to call the tool
        # Run the tool calls and store the resulted ToolMessages into the prompt
        (results := actions.run_tool_calls())  # results is a list of ToolMessage
        # results[0].content contains the result of the first tool call

        # Let LLM generate the text answer while providing the tool information
        answer = gen(tools=[is_lucky], tool_choice="none")
    else:  # LLM choose to generate the answer directly
        answer = actions.message

    return answer


n = 2024
ans = func(n)
print(ans)
print(f"The correct answer is {is_lucky(n)}.")
