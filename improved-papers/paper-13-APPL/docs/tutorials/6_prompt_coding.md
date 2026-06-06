# Introducing Prompt Coding Helpers

To better utilize the power of LMs, a lot of effort has been put into designing prompts, and the complexity of prompts has been increasing as observed in recent researches like [ToolEmu](https://github.com/ryoungj/ToolEmu), [Swe-agent](https://github.com/princeton-nlp/SWE-agent), etc.

APPL provides two types of prompt coding helpers, [`Compositor`](#prompt-compositors) and [`Definition`](#prompt-definitions), to facilitate coding prompts in a structured and maintainable way. These helpers were originally designed in [PromptCoder](https://github.com/dhh1995/PromptCoder) and have been used to develop prompts in the [ToolEmu](https://github.com/ryoungj/ToolEmu) project with more than 20k tokens in total. By leveraging Python's idiomatic features, we have enhanced the usability and flexibility of these helpers.

## Prompt Compositors
It is common to write prompts in [Markdown or XML format](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/advanced-prompt-engineering?pivots=programming-language-chat-completions#add-clear-syntax) to interact with LMs, where the prompts are structured with different elements like headers, lists, tags, etc.

To facilitate the creation of structured prompts in a programmatic way, we provide a set of compositors that compose the text within their context into a structured prompt with the corresponding format. For example, the `NumberedList` is a context manager composes a list of text within its scope into a numbered list:
```python
with NumberedList():
    f"First item"
    f"Second item"
>>> composed into >>>
1. First item
2. Second item
```

You can also nest the compositors to create more complex structures, like:
```python linenums="1"
@ppl
def compose(items: list[str]):
    with Tagged("div", indent_inside=4):  # (1)
        "Display the items:"
        with NumberedList(indent=4): # (2)
            items # (3)
    return records()

print(compose(["item1", "item2"]))
```

1. The default is no indentation inside the tag, can be changed by setting the `indent_inside` parameter.
2. Explicitly set the indentation for the content inside the list.
3. Fits different number of items.

The composed prompt is:
```xml
<div>
    Display the items:
        1. item1
        2. item2
</div>
```

Check the list of available compositors in the [API Reference](../docs/compositor.md). 

## Prompt Definitions
To make the prompt more structured and clear, it is often helpful to define concepts in prompts and refer to them in different parts of the prompt.

APPL provides the `Definition` class for defining concepts by subclassing it. For example:

```python
class InputReq(Definition):
    name = "Input Requirement"
    fstr = "[{}]" # defines the format when referring to the concept
```

To include the concept in the prompt, you can instantiate the class with the description as an argument, for example:
```python
@ppl
def func():
    InputReq(desc="The input should be two numbers.")
```

!!! info "Referring to the concept"
    You can use the class name in the prompt (without instantiation), which will be replaced by a certain format containg the concept's name. For example,
    ```python
    >>> print(f"{InputReq}")
    [Input Requirement]
    ```
    In some cases you may need the raw name of the concept without formatting, you can use the `!r` flag:
    ```python
    >>> print(f"{InputReq!r}")
    Input Requirement
    ```

With the help of modern IDEs like VSCode, you can easily navigate to the definition and references of the concept. The IDEs can also provide highlights to class names, which makes it easier to distinguish between concepts defined in the prompt and other variables in the code. These features are especially useful when the prompt is large and complex. 

## Example
We use the example in [PromptCoder](https://github.com/dhh1995/PromptCoder?tab=readme-ov-file#usage) to illustrate the usage of these prompt coding helpers:

```python linenums="1"
from appl import BracketedDefinition as Def
from appl import define, ppl, records
from appl.compositor import *
from appl.const import EMPTY

# (1)
class InputReq(Def):
    name = "Input Requirement"

class OutputReq(Def):
    name = "Output Requirement"

@ppl
def get_prompt(opr: str, language: str):
    "## Requirements"
    with NumberedList():
        InputReq(desc="The input should be two numbers.") # (2)
        OutputReq(desc=f"The output should be the {opr} of the two numbers.") # (3)
    EMPTY
    "## Instruction"
    f"Write a function in {language} that satisfies the {InputReq} and {OutputReq}." # (4)
    return records()
```

1. Declare the input and output requirements classes for reference.
    Alternatively, but not recommended, you can define classes as follows:
    `InputReq = define("Input Requirement")` and `OutputReq = define("Output Requirement")`.
    But then VSCode cannot recognize it as a class.
2. Instantiate the input requirement with a description.
3. Instantiate the output requirement with a description.
4. The naming can be used to distinguish:
    - variable naming (e.g. language): the dynamic input.
    - class naming (e.g. InputReq): the reference to the concept.

The result of `get_prompt("sum", "Python")` will be:
```md
## Requirements
1. Input Requirement: The input should be two numbers.
2. Output Requirement: The output should be the sum of the two numbers.

## Instruction
Write a function in Python that satisfies the [Input Requirement] and [Output Requirement].
```

Overall, these helpers provide a more structured and maintainable way to write dynamic prompts.
