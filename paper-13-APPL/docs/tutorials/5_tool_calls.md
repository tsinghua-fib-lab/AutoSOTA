# Enabling Tool Calls for LMs

## Defining Tools with Python Functions
To streamline the process of LMs using tools, APPL provides a simple way to convert Python functions into tools that can be called by LMs. This is done by using the `as_tool` function, which automatically extract information from the function signature and docstring to create a tool specification from a Python function.

!!! warning "Docstring Format"
    The docstring needs to follow parsable formats like Google Style. We use [`docstring_parser`](https://github.com/rr-/docstring_parser) to parse the docstring.

For example, consider the following Python function that checks whether a number is a lucky number:

```python linenums="1"
import sympy

def is_lucky(x: int) -> bool:
    """Determine whether the input number is a lucky number.

    Args:
        x (int): The input number to be checked.

    Returns:
        bool: True if the number is a lucky number, False otherwise.
    """
    return sympy.isprime(x + 3)
```

It can be converted into a tool using: `tool = as_tool(is_lucky)`, and then the tool can be expressed in a JSON format following the OpenAI schema using `tool.openai_schema`:

```json
{
  "type": "function",
  "function": {
    "name": "is_lucky",
    "description": "Determine whether the input number is a lucky number.",
    "parameters": {
      "properties": {
        "x": {
          "description": "The input number to be checked.",
          "type": "integer"
        }
      },
      "required": ["x"],
      "type": "object"
    }
  }
}
```

!!! note "Description of the Tool"
    The description of the tool is extracted from the docstring. The description can be configured to different detail levels, for example, to include a long description, examples, etc.

## Generating and Running Tool Calls

APPL allows you to simply provide a list of Python functions as tools directly to the `gen` function, for example:
```python
actions = gen(tools=[is_lucky], tool_choice="required")
```

!!! info "Tool Calling Behavior"
    The tool calling behavior is useful to control how LMs call tools, which can be specified by the `tool_choice` parameter in the `gen` function. Options includes `auto` (default), `required`, `none`, or a specific choice. Please refer to [OpenAI's Documentation](https://platform.openai.com/docs/guides/function-calling/function-calling-behavior) for more details. APPL provides another helper function `as_tool_choice` to convert a function to a `tool_choice`, e.g. `gen(tools=[is_lucky], tool_choice=as_tool_choice(is_lucky))`.

The `actions` is a `Generation` object containing a list of tool calls, which can be easily executed via:

```python
results = actions.run_tool_calls()
```

where the `results` are a list of `ToolMessage` objects that can be directly captured to the prompt.

!!! info "Adding Tool Calls and Results to the Prompt"
    Both `Generation` and `ToolMessage` objects can be directly captured to the prompt. When the `Generation` contains tool calls, the tool calls will be captured to the prompt as a list of `AIMessage` objects containing the tool call information.

## Examples

### Complete Example
Now let's put everything together. The following code augments LM with a user-defined tool to answer a simple question:

```python linenums="1"
--8<-- "examples/basic/tool_call.py"
```

The conversation including the tool calls would look like this:

| Role             | Message                                                       |
| ---------------- | ------------------------------------------------------------- |
| *User*           | Is 2024 a lucky number?                                       |
| *Assistant*      | [ToolCall(id='call_...', name='is_lucky', args='{"x":2024}')] |
| *Tool(is_lucky)* | True                                                          |

The output will look like this:
```
Yes, 2024 is a lucky number!
The correct answer is True.
```

### Parallel Tool Calls
Similarly, let's try to reimplement the example used in [OpenAI's documentation](https://platform.openai.com/docs/guides/function-calling/parallel-function-calling), where the amount of code is significantly reduced.

```python linenums="1"
import json
from appl import gen, ppl

from typing import Literal


def get_current_weather(
    location: str, unit: Literal["celsius", "fahrenheit"] = "fahrenheit"
) -> str:
    """Get the current weather in a given location.

    Args:
        location (str): The city and state, e.g. San Francisco, CA
    """
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": unit})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})


@ppl
def get_weather_for_cities():
    "What's the weather like in San Francisco, Tokyo, and Paris?"

    (actions := gen(tools=[get_current_weather]))

    if actions.is_tool_call:
        (results := actions.run_tool_calls())
        answer = gen(tools=[get_current_weather], tool_choice="none")
    else:
        answer = actions.message

    return answer


print(get_weather_for_cities())
```


### More Examples in Cookbook
Now we are able to build more complex workflows involving tool calls, like the [ReAct method](https://arxiv.org/abs/2210.03629), as shown in this [cookbook](../cookbook/react.md).
