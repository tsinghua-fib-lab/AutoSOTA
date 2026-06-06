# Getting Started with APPL

In this section, we will set up your environment and run your first APPL program.

## Installation
You can simply install APPL from PyPI using pip:
```bash
pip install -U applang
```
More installation options can be found in the [installation guide](../install.md).

## Setup
You need to set up API keys or your own backends to interact with language models (LMs).

In this guide, we use OpenAI API as the default backend.
You can set your OpenAI API key in the `.env` file in the root directory of your project:
```
OPENAI_API_KEY=<your openai api key>
```
or export it as an environment variable:
```bash
export OPENAI_API_KEY=<your openai api key>
```

For setting up other backends, please refer to the [setup guide](../setup.md/#setup-llms).


## Hello World

Let's create a simple function that uses LM to respond to a greeting.

```py
@ppl  # the @ppl decorator marks the function as an `APPL function`
def greeting(name: str):
    f"Hello World! My name is {name}."  # Add text to the prompt
    return gen()  # call the default LM with the current prompt
```

In this example, the `@ppl` decorator (`@` stands for `a` here) marks the `hello_world` function as an *APPL function*. Within such a function, the standalone string `f"Hello World! My name is {name}."` is added to the prompt, and the `gen()` function calls LM to generate responses using the current prompt.

Below is a complete code snippet that demonstrates the usage of the `greeting` function:

```python
from appl import gen, ppl

@ppl
def greeting(name: str):
    f"Hello World! My name is {name}."
    return gen()

# call `greeting` as a normal Python function
print(greeting("APPL"))
```

The prompt for the generation is:
```
Hello World! My name is APPL.
```

The output will look like
```
Nice to meet you, APPL!
```

<!-- Continue with the [next tutorial](./2_qa_example.md) to learn concepts of APPL through question-answering examples. -->
