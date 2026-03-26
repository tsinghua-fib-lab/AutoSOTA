# Prompt Capture

Expression statements within *APPL functions* are captured as prompts based on the type of the value, when the value is not `None`.

The types can be categorized into three groups according to the behavior when converting to prompts: prompts, sequence of prompts, and custom promptable.

## 1. Prompts
The most basic prompt is a string (including `StringFuture`), which can be a string literal, formatted string or string variable. We convert the string to a message based on the role scope.

An already constructed `Message` object is also considered a prompt, as shown in the following example.

```python
from appl import ppl, records, AIMessage

@ppl
def greeting(name: str):
    "Hello world!"
    AIMessage("Nice to meet you.")
    return records()

print(greeting("APPL"))
# user: Hello world!
# assistant: Nice to meet you.
```

Besides, for LMs that support multimodal inputs, the `Image` are also captured as part of the prompt.

Note types like `int`, `float` and `bool` are not captured. To capture these values, you can explicitly convert them to strings or use them in formatted strings.

## 2. Sequence of Prompts
For types that subclass `Sequence`, such as `list` and `tuple`, the elements are recursively captured as prompts one by one.

```python
from appl import ppl, records

@ppl
def greeting(name: str):
    ["Hello world!", ("My name is APPL.", "Nice to meet you.")]
    return records()

print(greeting("APPL"))
# Hello world!
# My name is APPL.
# Nice to meet you.
```

## 3. Custom Promptable
You may also define custom types and the ways to convert them to prompts by subclassing `Promptable` and implementing the `__prompt__` method.

```python
from appl import Promptable, ppl, records

class MyPromptable(Promptable):
    def __init__(self, number: int):
        self.number = number

    def __prompt__(self):
        return f"The number is {self.number}."

@ppl
def custom_promptable():
    MyPromptable(42)
    return records()

print(custom_promptable())
# The number is 42.
```

!!! warning "Warning messages for unsupported types"
    APPL will display a warning message when encountering unsupported types that are not captured as prompts.
    ```bash
    WARNING | Cannot convert {VALUE} of type {TYPE} to prompt, ignore.
    ```