# Understanding APPL Function

*APPL functions* are the fundamental building blocks of APPL, marked by the `@ppl` decorator. As seen in the [QA examples](./2_qa_example.md), each APPL function is a self-contained module encapsulating LM prompts and Python workflows to realize the functionality.

## Difference to Python functions
*APPL functions* are extended from Python functions while designed to seamlessly blend LM prompts with Python codes. You can use Python syntax and libraries in *APPL functions* as you would in normal Python functions. Beyond normal Python functions, *APPL function* essentially provides a [Prompt Context](#prompt-context) that specially tailored for LM interactions to a Python function. New features of APPL functions include:

1. [Prompt Capturing](#prompt-capturing): You can easily define prompts with expression statements within *APPL functions*.
2. [Prompt Retrieval](#prompt-retrieval): prompts are automatically retrieved when making LM calls. You may also retrieve prompts in the context by predefined functions.
3. [Context Passing](#context-passing): The prompt context can be passed to other APPL functions with configurable options.

## Prompt Context
Each *APPL function* has a **prompt context**, which is an object that stores the prompts and other information. The context is automatically managed by the APPL framework, and you don't need to worry about it in most cases.

## Prompt Capturing
As you have seen in the [QA examples](./2_qa_example.md), you can define prompts with [expression statements](https://docs.python.org/3/reference/simple_stmts.html#expression-statements) within *APPL functions*, including string literals (e.g., `"Hello"`), formatted strings (e.g., `f"My name is {name}"`), or more complex expressions. For types that subclass `Sequence`, such as `list` and `tuple`, the elements are recursively captured as prompts one by one. You may also define custom types and the ways to convert them to prompts by subclassing `Promptable` and implementing the `__prompt__` method. See the [Appendix](./appendix/prompt_capture.md) for more details.

??? question "How about docstrings in *APPL functions*?"
    Docstring is a special expression statement in Python. There are two cases for docstrings in *APPL functions*:
    
    1. If the docstring is triple-quoted, it will **NOT** be captured as a prompt by default. To also include the docstring as a part of the prompt (as a system message or user message), you may specify `docstring_as` to be "system" or "user" in the `@ppl` decorator, like
        ```python
        @ppl(docstring_as="system")
        def my_function():
            """Docstring as system message.
            
            Details."""
            "Another prompt."
            return records()
        
        print(my_function())
        ```
        Outputs:
        ```
        Docstring as system message.

        Details.
        Another prompt.
        ```
    2. Otherwise, it will be captured as a prompt. But if the content is not meant to be the docstring of the function, it is recommended to use f-string instead.
        ```python
        @ppl
        def my_function():
            f"First prompt."
            "Second prompt."
        ```

??? question "How about multiline strings in *APPL functions*?"
    The multiline strings will be cleaned using `inspect.cleandoc` before being captured as prompts.
    It is recommend that you follows the indentation of the function.
    ```python
    @ppl
    def my_function():
        """Docstring""" # not in the prompt
        """First line.
        Second line.
            indented line."""
        return records()
    
    print(my_function())
    ```
    Outputs:
    ```
    First line.
    Second line.
        indented line.
    ```

??? info "Add image prompt"
    It's easy to add image prompts in *APPL functions* by using the `Image` class. The following example demonstrates how to add this [example image](https://maproom.wpenginepowered.com/wp-content/uploads/OpenAI_Logo.svg_-500x123.png) as part of prompt in an *APPL function*.
    ```python
    --8<-- "examples/basic/image_prompt.py"
    ```

??? info "Explicitly capture prompts"
    If you want to capture prompts explicitly, simply use the `grow` function to append the content to the prompt.
    ```python
    from appl import gen, grow, ppl

    @ppl
    def count(word: str, char: str):
        grow(f"How many {char}s are there in {word}?")
        return gen()
    ```

    In other words, there is an implicit `grow` function wrapped around the expression statements in *APPL functions*.

??? warning "Pay attention to return values of function calls"
    Function calls are also expression statements, which means their return values (when not `None`) may be captured as prompts based on the type. To avoid capturing the return value, you may write it as a assignment statement, for example when calling the `pop` function : `_ = {"example": "Hello World"}.pop("example")`.

## Prompt Retrieval
Similar to the local and global variables in Python (retrieved with `locals()` and `globals()` functions, respectively), you can retrieve the prompts captured in the current function (with `records()`) or the full conversation in the context (with `convo()`). This [example](#example) demonstrates how to retrieve the prompts captured in the current function and the full conversation in the context.

When making LM calls using `gen()`, the full conversation is automatically retrieved from the context as the prompt. Therefore, instead of passing the prompt explicitly, the position of the `gen()` function within the *APPL function* determines the prompt used for the generation.

## Context Passing
There are four different ways to pass the context when calling another *APPL function* (the callee) in an *APPL function* (the caller): **new**, **copy**, **same**, and **resume**.

![Context Management](https://raw.githubusercontent.com/appl-team/appl/main/docs/_assets/context.png)

1. **new**: The default behavior, create a new empty context.
2. **copy**: This is similar to *call by value* in programming languages. The callee's context is a copy of the caller's context, therefore the changes in the callee's context won't affect the caller's context.
3. **same**: This is similar to *call by reference* in programming languages. The callee's context is the same as the caller's context, therefore the changes in the callee's context will affect the caller's context.
4. **resume**: This resumes the context of the function each time it is called, i.e., the context is preserved across calls, making the function stateful. It copies the caller's context for the first call as the initial context. It is useful when you want to continue the conversation from the last call.

With these context management methods, you can now easily modularize your prompts as well as the workflow, so that they are more readable and maintainable.

## Example
In this example, we illustrate the usage of the first three context management methods and ways to decompose long prompts into smaller pieces using *APPL functions*. For the resume method, please refer to the [multi-agent chat example](../cookbook/multi_agent_chat.md).

```python linenums="1"
from appl import convo, gen, ppl, records

@ppl # (1)
def intro():
    f"Today is 2024/02/29."
    return records() # (2)

@ppl(ctx="same") # (3)
def addon():
    f"Dates should be in the format of YYYY/MM/DD." # (4)

@ppl(ctx="copy")  # (5)
def query(question: str):
    f"Q: {question}"
    f"A: "
    # print(convo())  # (6)
    return gen()

@ppl
def answer_questions(questions: list[str]):
    # long prompt can be decomposed into several smaller `appl functions`

    # method 1 (recommended): build the sub-prompts in an empty context
    intro()  # (7)

    # method 2: use the same context and modify it in the function
    addon()  # (8)

    return [query(q) for q in questions]

questions = [
    "What's the date tomorrow?",
    "What's the date yesterday?",
    "How many dates passed since 2024/02/02?",
]
for res in answer_questions(questions):
    print(res)
```

1. Use new context, the default method for passing context.
2. Local prompts are returned as a list of records, which can be add back to the caller's context.
3. Use the caller's context, which contains the prompt `Today is 2024/02/29.`.
4. The newly captured prompt influences the caller's context, now the caller's prompts contain both `Today is 2024/02/29.` and `Dates should be in the format of YYYY/MM/DD.`.
5. Copy the caller's context, the prompts captured here will not influence the caller's context.
6. Display the global prompts (`convo()`, means conversation) used for the generation (`gen()`).
7. The callee returns `PromptRecords` containing sub-prompts, which are added to the caller's context.
8. The callee returns `None`, the return value is not captured, but the context is already modified inside the callee.

Three queries are independent and run in parallel, where the prompts and possible responses of the generations are shown below:

=== "Overall output"
    The Overall output will looks like:
    ```plaintext
    2024/03/01
    2024/02/28
    27 dates have passed since 2024/02/02.
    ```
=== "First query"
    Prompt:
    ```plaintext
    Today is 2024/02/29.
    Dates should be in the format of YYYY/MM/DD.
    Q: What's the date tomorrow?
    A: 
    ```
    Output will looks like `2024/03/01`.
=== "Second query"
    Prompt:
    ```plaintext
    Today is 2024/02/29.
    Dates should be in the format of YYYY/MM/DD.
    Q: What's the date yesterday?
    A: 
    ```
    Output will looks like `2024/02/28`.
=== "Third query"
    Prompt:
    ```plaintext
    Today is 2024/02/29.
    Dates should be in the format of YYYY/MM/DD.
    Q: How many dates passed since 2024/02/02?
    A: 
    ```
    Output will looks like `27 dates have passed since 2024/02/02.`.

## Caveats
!!! warning "`@ppl` needs to be the last decorator."
    Since `@ppl` involves compiling the function, it should be the last decorator in the function definition, i.e. put it closest to the function definition. Otherwise, the function may not work as expected.

!!! warning "`@ppl` cannot be nested."
    Currently, `@ppl` cannot be nested within another `@ppl` function. You may define the inner function as a normal Python function and call it in the outer `@ppl` function.
