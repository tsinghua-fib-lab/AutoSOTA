# Frequently Asked Questions

## Language Related

### Prompt

??? question "What statements in *APPL function* are regarded as prompts?"
    In *APPL function*, the value of [expression statements](https://docs.python.org/3/reference/simple_stmts.html#expression-statements) are captured as prompts. For example, expressions `"str"` and `func()` are prompt statements while an assignment (e.g., `a = "str"`) is not.
    For expression statements, their value will be converted to prompts according to the types, where types like `int`, `float`, `None` will be ignored and `str` will be added to the prompt. Please read [this documentation](./tutorials/appendix/prompt_capture.md) for more details.

??? question "How to write multiline strings in *APPL function*?"
    It is not recommended to write multiline strings in *APPL function* directly. Instead, you can write them as multiple strings line by line. For example:
    ```python
    @ppl
    def my_function():
        "First line."
        "Second line."
    ```
    If you still want to write multiline strings, you may either define them outside the *APPL function*, or keep indented and use `textwrap.dedent` to remove the common leading whitespace. For example:
    ```python
    @ppl
    def my_function():
        textwrap.dedent(
            """
            First line.
            Second line.
            """
        )
    ```

### Generation

??? question "What parameters can be used in the `gen` function?"
    APPL supports various backends using [`litellm`](https://github.com/BerriAI/litellm), which unifies different LLM APIs into [the OpenAI format](https://platform.openai.com/docs/api-reference/chat/create). See [Generation Parameters](./tutorials/usage/servers.md#generation-parameters) for more details.

??? question "Is the `gen` function bind to the `AIRole`?"
    No, the `gen` function is not bind to the `AIRole` and can be used independently.
    The `AIRole` is a role scope indicate that the prompts within this scope are `AIMessage` (i.e., Assistant Messages).
