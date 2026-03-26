# 🍎APPL: A Prompt Programming Language

[![version](https://img.shields.io/pypi/v/applang.svg)](https://pypi.python.org/pypi/applang)
[![python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://mit-license.org/)
[![Discord](https://img.shields.io/badge/discord-orange)](https://discord.gg/q3x4Qwgj29)
[![arXiv](http://img.shields.io/badge/cs.AI-arXiv%3A2406.13161-B31B1B.svg?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2406.13161)

**APPL** is A Prompt Programming Language that extends Python to provide a Natural, Intuitive, Convenient, and Efficient (NICE) way to utilize Large Language Models (LLMs) such as GPT in your program. We believe Language Model will be an essential part of future software that help achieves more than what we can do today, and APPL is a step towards this future that seamlessly integrates programs and LLMs.

<video style="width: 100%" src="https://github.com/appl-team/appl/assets/12556773/5d75d3db-1b1c-48c9-97ec-e9d72a387e49" type="video/mp4" controls></video>

## Key Features
- **Readability and maintainability via seamless integration with Python.**  APPL seamlessly embeds natural language prompts into Python programs, maintaining prompts' readability while inheriting modularity, reusability, dynamism and the ecosystem from the host programming language.
- **Flexible prompt engineering.**  Except for allowing the utilization of Python control flows and the modularized decomposition of prompts, APPL offers prompt coding helpers to facilitate programming prompts in a modularized and maintainable way.
- **Automatic parallelization via asynchronous computation.**  APPL schedules LLM calls asynchronously, leveraging potential independence among them to facilitate efficient parallelization. This offloads the burden of users to manage synchronization manually, with almost no extra work.
- **Smooth tool calling integration.**  APPL provides intuitive ways to transform Python functions into tools that can be called by LLMs, making it easier for users to integrate existing Python libraries and functions with LLMs.
- **Tracing and Failure Recovery.** APPL traces the execution of LLM calls and supports recovery from failures, which is essential for debugging and error handling in the LLM programming paradigm.
- **More Features.** APPL has many more other features, such as an auto-continuation mechanism to continue the generation when the output token limit is exceeded.
- **Integrations.** APPL also provides a unified interface for multiple LLM backends using [`litellm`](https://docs.litellm.ai/docs/), [llm observability](https://appl-team.github.io/appl/tutorials/7_tracing/#visualizing-the-trace) using [`langfuse`](https://github.com/langfuse/langfuse) and [`lunary`](https://github.com/lunary-ai/lunary), and many other features.

## News
* **[2024-12-16]**: APPL 0.2.0 is released with many new features! Please check the [release note](https://github.com/appl-team/appl/releases/tag/v0.2.0) for more details.
* **[2024-07-12]**: We have improved our [tutorial](https://appl-team.github.io/appl/tutorials/). Please check them out for more detailed usage and examples.
<!-- and [cookbook](https://appl-team.github.io/appl/tutorials/) -->

## Quick Start

### Installation
You can simply install APPL from PyPI using pip:
```bash
pip install -U applang
```
More installation options can be found in the [installation guide](https://appl-team.github.io/appl/install).

### Setup
You need to set up API keys or your own LLM backends to interact with LLMs.

In this guide, we use OpenAI API as the default backend.
You can set your OpenAI API key in the `.env` file in the root directory of your project:
```
OPENAI_API_KEY=<your openai api key>
```

or export it as an environment variable:

```bash
export OPENAI_API_KEY=<your openai api key>
```

For setting up other backends, enabling tracing and recovering from traces, please refer to the [setup guide](https://appl-team.github.io/appl/setup).

### Hello World

To begin, let's create a simple function that uses LLM to respond to a greeting.

```python
from appl import gen, ppl

@ppl  # the @ppl decorator marks the function as an `APPL function`
def greeting(name: str):
    f"Hello World! My name is {name}."  # Add text to the prompt
    return gen()  # call the default LLM with the current prompt

print(greeting("APPL"))  # call `greeting` as a normal Python function
```

The prompt for the generation is:
```
Hello World! My name is APPL.
```

The output will look like
```
Nice to meet you, APPL!
```

In this example, the `@ppl` decorator (`@` stands for `a` here) marks the `hello_world` function as an *APPL function*. Within such a function, the standalone string `f"Hello World! My name is {name}."` is added to the prompt, and the `gen()` function calls LLM to generate responses using the current prompt. Moreover, explicitly appending the prompt is also supported using `grow`:

```python
from appl import gen, grow, ppl

@ppl  # the @ppl decorator marks the function as an `APPL function`
def greeting(name: str):
    grow(f"Hello World! My name is {name}.")  # grow the prompt
    return gen()  # call the default LLM with the current prompt

print(greeting("APPL"))  # call `greeting` as a normal Python function
```

### Question Answering

Let's then implement a question-answering system using APPL. In this example, the APPL program answers multiple questions about a quotation by first extracting the author's name (inspired by [this cookbook](https://cookbook.openai.com/articles/how_to_work_with_large_language_models)). [Here](https://colab.research.google.com/drive/1khZcleOrdLOWtUB4EMEQCjGA1vBaARI9) is a runnable Colab notebook of this example.

```python linenums="1" hl_lines="5 10 11 13"
from appl import AIRole, gen, ppl

@ppl(ctx="copy")  # copy the context from caller
def get_answer(question: str):
    question  # append to the prompt
    return gen()  # return as a future object

@ppl  # marks APPL function
def answer_questions(quotation: str, questions: list[str]):
    "Extract the name of the author from the quotation below and answer questions."
    quotation  # append to the prompt
    with AIRole():  # assistant message
        f"The name of the author is {gen(stop='.')}"  # specify the prefix
    return [get_answer(q) for q in questions]  # parallelize calls

quotation = '"Simplicity is the ultimate sophistication." -- Leonardo da Vinci'
questions = [
    "In what era did the author live?",
    # more questions can be added here
]
for ans in answer_questions(quotation, questions):
    print(ans)
```

The resulting conversation for the first question would look like (generated responses are in **bold**):

| Role        | Message                                                                                                                                            |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| *User*      | Extract the name of the author from the quotation below and answer questions.<br>"Simplicity is the ultimate sophistication." -- Leonardo da Vinci |
| *Assistant* | The name of the author is **Leonardo da Vinci.**                                                                                                   |
| *User*      | In what era did the author live?                                                                                                                   |
| *Assistant* | **Leonardo da Vinci lived during the Renaissance era.**                                                                                            |

In *APPL functions*, [expression statements](https://docs.python.org/3/reference/simple_stmts.html#expression-statements) are captured as prompts [based on the type of its value](https://appl-team.github.io/appl/tutorials/appendix/prompt_capture/). Notably, the f-string is processed part by part, so the `gen` function inside the f-string intuitively uses the contents before that. In this example, `The name of the author is ` serves as a prefix to guide the completion of the author's name.

After the author's name is extracted, the `get_answer` function is called multiple times in parallel to answer the questions, with the context being copied (detailed in [context-management](#context-management)), demonstrating the automatic parallelization feature of APPL.

On the other hand, this is a pretty long Langchain code that implements the same functionality, where you can feel the inflexibility of using prompt templates:
```python linenums="1" 
from concurrent.futures import ThreadPoolExecutor
from typing import List

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI()

messages = [
    (
        "user",
        "Extract the name of the author from the quotation below:\n{quotation}",
    ),
    ("assistant", "The name of the author is "),
]
author_prompt = ChatPromptTemplate.from_messages(messages)

messages = messages[:1] + [
    ("assistant", "The name of the author is {author}"),
    ("user", "{question}"),
]
question_prompt = ChatPromptTemplate.from_messages(messages)


def answer_questions(quotation: str, questions: List[str]):
    # First extract the author
    author_chain = author_prompt | llm | StrOutputParser()
    author = author_chain.invoke({"quotation": quotation})

    # Create question answering chain
    qa_chain = question_prompt | llm | StrOutputParser()

    def answer_single_question(question):
        return qa_chain.invoke(
            {"quotation": quotation, "author": author, "question": question}
        )

    # Answer each question in parallel using map
    with ThreadPoolExecutor() as executor:
        return list(executor.map(answer_single_question, questions))


quotation = '"Simplicity is the ultimate sophistication." -- Leonardo da Vinci'
questions = [
    "In what era did the author live?",
    "What is the most famous painting of the author?",
]
print(answer_questions(quotation, questions))
```

## RoadMap
- [x] Default to exclude """docstring""" from the prompt formation.
- [x] Add supports for LLM logging and tracing platforms to inspect the traces.
  - [x] Supported Lunary and Langfuse (open-source)
- [ ] Allow directly working with prompts without `ppl` decorator.
- [ ] Add more ... (contributions are welcome!)
  - [ ] Examples and tutorials to demonstrate the usage
  - [ ] Test cases to increase the coverage

## Tutorial and Cookbook
For a more comprehensive tutorial, please refer to the [tutorial](https://appl-team.github.io/appl/tutorials).

### Table of Contents
- [Introduction](https://appl-team.github.io/appl/tutorials/intro)
- [Getting Started](https://appl-team.github.io/appl/tutorials/1_get_started)
- [Example: QA with LMs](https://appl-team.github.io/appl/tutorials/2_qa_example)
- [APPL Function](https://appl-team.github.io/appl/tutorials/3_appl_function)
- [Concurrent LM Calls](https://appl-team.github.io/appl/tutorials/4_concurrent)
- [Tool Calls for LMs](https://appl-team.github.io/appl/tutorials/5_tool_calls)
- [Prompt Coding Helpers](https://appl-team.github.io/appl/tutorials/6_prompt_coding)
- [Using Tracing](https://appl-team.github.io/appl/tutorials/7_tracing)

### Cookbook and Applications
For more detailed usage and examples, please refer to the [cookbook](https://appl-team.github.io/appl/cookbook).

We use APPL to reimplement popular LLM and prompting algorithms in [Reppl](https://github.com/appl-team/reppl), such as:
* [Tree of Thoughts](https://github.com/princeton-nlp/tree-of-thought-llm) [[Re-implementation](https://github.com/appl-team/reppl/tree/main/tree-of-thoughts/)] [[APPL Example](examples/advanced/tree_of_thoughts/)]: deliberate problem solving with Large Language Models.

We use APPL to build popular LM-based applications, such as:
* [Wordware's TwitterPersonality](https://twitter.wordware.ai/)[[APPL implementation](https://github.com/appl-team/TwitterPersonality)]: analyzes your tweets to determine your Twitter personality.

We use APPL to build small LLM-powered libraries, such as:
* [AutoNaming](https://github.com/appl-team/AutoNaming): automatically generate names for experiments based on argparse arguments.
* [ExplErr](https://github.com/appl-team/ExplErr): a library for error explanation with LLMs.

## Working with Cursor (Experimental)
We provide [.cursorrules](https://github.com/appl-team/appl/blob/main/.cursorrules) to help you write APPL code with [Cursor](https://www.cursor.com/). You also setup the [Docs Symbol](https://docs.cursor.com/context/@-symbols/@-docs) with [APPL Docs](https://appl-team.github.io/appl/docs/). Thanks @xiumaoprompt for suggestion!

## Citation and Acknowledgment
If you find APPL helpful, please consider citing our paper:
```bibtex
@article{dong2024appl,
  title={APPL: A Prompt Programming Language for Harmonious Integration of Programs and Large Language Model Prompts},
  author={Dong, Honghua and Su, Qidong and Gao, Yubo and Li, Zhaoyu and Ruan, Yangjun and Pekhimenko, Gennady and Maddison, Chris J and Si, Xujie},
  journal={arXiv preprint arXiv:2406.13161},
  year={2024}
}
```

We would like to thank the open-source community for their contributions, where we learned from or used these libraries in our project, including
[instructor](https://github.com/jxnl/instructor),
[LiteLLM](https://github.com/BerriAI/litellm),
[LMQL](https://github.com/eth-sri/lmql),
[Guidance](https://github.com/guidance-ai/guidance),
[SGLang](https://github.com/sgl-project/sglang) and
[autogen](https://github.com/microsoft/autogen).

We also notice that there are more projects coming out to push the boundaries of prompt programming, such as [ell](https://github.com/MadcowD/ell) and [mirascope](https://github.com/Mirascope/mirascope/).

## License
This project is licensed under the terms of the MIT License.
