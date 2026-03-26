# Structured Output

For APIs that has supported specifying `response_format` as a Pydantic object, as in [OpenAI](https://platform.openai.com/docs/guides/structured-outputs/structured-outputs), this argument can be directly used within APPL's `gen` function. Otherwise, you can use [Instrcutor](https://github.com/jxnl/instructor/) that supports similar functionality with another argument `response_model`.

## Get Started

Let's use the example from the [instructor](https://github.com/jxnl/instructor/?tab=readme-ov-file#get-started-in-minutes) and implement in two ways:

```python linenums="1"
--8<-- "examples/usage/structured_output.py"
```

## Usage: Choices
One common use case of structured output is to make the response choose from a set of options. For example,
```python linenums="1"
@ppl
def answer(question: str):
    "Answer the question below."
    question
    return gen(response_format=Literal["Yes", "No"])
```

## Usage: Thoughts
Or extend with thoughts before the answer:
```python linenums="1"
--8<-- "examples/usage/structured_output_thoughts.py"
```

An example output will be:
```markdown
Thoughts:
1. **Identify the Odd Numbers:**
   - 15
   - 5
   - 13
   - 7
   - 1   
   
2. **Add the Odd Numbers Together:**
   \[
   15 + 5 + 13 + 7 + 1 = 41
   \]
   
3. **Determine Whether the Sum is Even or Odd:**
   - 41 is an odd number because it cannot be evenly divided by 2.

4. **Conclusion:** The odd numbers in the given group add up to an odd number, not an even number.
Answer: No
```

## With Streaming

For `response_format`, the streaming is captured and displayed, and the returned object is a complete object. For `response_model`, you need to make the `response_model` a `Partial` or `Iterable` object so that they can be streamed, and the response object is a generator that yields partial objects.

Let's slightly modify the [example](https://python.useinstructor.com/why/#partial-extraction) from instructor:

```python linenums="1"
--8<-- "examples/usage/structured_output_streaming.py"
```

It will gradually print the output:

```json
{
  'users': [
    {'name': 'Alice', 'age': 25},
    {'name': 'Bob', 'age': 30},
    {'name': 'Charlie', 'age': 20}
  ]
}
```