# Explaining Concepts with QA Examples

In this section, we will explain the core concepts of APPL using question-answering examples. These examples demonstrate how to use APPL to manage prompts, request and retrieve LM responses, and embed LM interactions in Python workflows.

## Extract Name from Quotation

Let's begin with a simple task used in [this cookbook](https://cookbook.openai.com/articles/how_to_work_with_large_language_models) that involves extracting the author's name from a quotation.

```python linenums="1" hl_lines="1 4 6"
@ppl # marks APPL function
def get_author_name(quotation: str):
    # string literal as a prompt statement
    "Extract the name of the author from the quotation below."
    # string variable as a prompt statement
    quotation
    # use the current prompts to call the LM to generate responses
    response = gen()
    # return the generated response as the answer to the question
    return response
```

The `@ppl` decorator marks the `get_author_name` function as an *APPL function*. Within the function, the string literal `"Extract the name of the author from the quotation below."` and variable `quotation` are added to the prompt. The `gen()` function is then called to generate responses using the current prompt. Finally, the generated response is returned as the answer to the question. This *APPL function* can be called like a normal Python function, as shown in the complete code snippet below.

```python linenums="1"
from appl import gen, ppl

@ppl
def get_author_name(quotation: str):
    "Extract the name of the author from the quotation below."
    quotation
    return gen()

quotation = '"Simplicity is the ultimate sophistication." -- Leonardo da Vinci'
print(get_author_name(quotation))
```

In this code, the call to the `get_author_name` function is equivalent to querying the LM with the prompt:

| Role   | Message                                                                                                                       |
| ------ | ----------------------------------------------------------------------------------------------------------------------------- |
| *User* | Extract the name of the author from the quotation below.<br>"Simplicity is the ultimate sophistication." -- Leonardo da Vinci |

The output may looks like:
```plaintext
The name of the author is Leonardo da Vinci.
```

### Specify Prefix for Response
Further, you may want to guide the response following a specific prefix so that the generated response is just the author's name like `Leonardo da Vinci` in the previous example.

To achieve this, you can specify the prefix in the prompt as shown below:

```python linenums="1" hl_lines="5 7"
@ppl
def get_author_name(quotation: str):
    "Extract the name of the author from the quotation."
    quotation
    with AIRole():  # specify the role of the prompt within this scope
        # specify the prefix for the response and then call the generation function
        f"The name of the author is {(answer := gen(stop='.'))}"
    return answer
```

Notably, the f-string is processed part by part, so the `gen` function inside the f-string intuitively uses the contents before that. The warlus operator `:=` is used to store the generated response in the `answer` variable and return it as the answer to the question.

The prompt to the LM includes the prefix `The name of the author is ` as an assistant message, and the generated response (in **bold**) is also added to the conversation as part of the f-string:

=== "Before `gen()`"
    | Role        | Message                                                                                                                       |
    | ----------- | ----------------------------------------------------------------------------------------------------------------------------- |
    | *User*      | Extract the name of the author from the quotation below.<br>"Simplicity is the ultimate sophistication." -- Leonardo da Vinci |
    | *Assistant* | The name of the author is                                                                                                     |
=== "After `gen()`"
    | Role        | Message                                                                                                                       |
    | ----------- | ----------------------------------------------------------------------------------------------------------------------------- |
    | *User*      | Extract the name of the author from the quotation below.<br>"Simplicity is the ultimate sophistication." -- Leonardo da Vinci |
    | *Assistant* | The name of the author is **Leonardo da Vinci**                                                                               |

??? note "Using `AIRole` context manager"
    The `AIRole` context manager is used to specify the message role of the prompts within its scope. That saying, the prompts within the `AIRole` block is treated as assistant messages. Similarly, you can use `SystemRole` to specify system messages.
    You can use `from appl import AIRole, SystemRole` to import these context managers.

## Answer Follow-up Questions

Let's extend the previous example to answer multiple independent follow-up questions about the author. The following code demonstrates how to answer questions about the author's era based on the extracted name.

<!-- [Here](https://colab.research.google.com/drive/1khZcleOrdLOWtUB4EMEQCjGA1vBaARI9) is a runnable Colab notebook of this example.  -->

```python linenums="1" hl_lines="4 16"
from appl import AIRole, gen, ppl
from appl.const import NEWLINE

@ppl(ctx="copy")  # copy the context from caller
def get_answer(question: str):
    question # only affect the prompt in the current context since the context is copied
    return gen()

@ppl
def answer_questions(quotation: str, questions: list[str]):
    "Extract the name of the author from the quotation below and answer questions."
    quotation
    with AIRole():
        f"The name of the author is {gen(stop=NEWLINE)}"
    # call sub-functions to answer questions independently
    return [get_answer(q) for q in questions]

quotation = '"Simplicity is the ultimate sophistication." -- Leonardo da Vinci'
questions = [
    "In what era did the author live?",
    "What is the most famous painting of the author?",
    # more questions can be added here
]
for ans in answer_questions(quotation, questions):
    print(ans)
```

Note the `@ppl(ctx="copy")` decorator in the `get_answer` function, which specify the method to obtain the context from its caller is by copying (more options explained in [context passing](./3_appl_function.md#context-passing)). This ensures that the context of the caller is not affected by the prompts in the `get_answer` function. The resulting conversation (convo) for the first and second question would look like (generated responses are in **bold**):

=== "Convo of `get_answer` (first)"
    | Role        | Message                                                                                                                                            |
    | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
    | *User*      | Extract the name of the author from the quotation below and answer questions.<br>"Simplicity is the ultimate sophistication." -- Leonardo da Vinci |
    | *Assistant* | The name of the author is **Leonardo da Vinci.**                                                                                                   |
    | *User*      | In what era did the author live?                                                                                                                   |
    | *Assistant* | **Leonardo da Vinci lived during the Renaissance era.**                                                                                            |
=== "Convo of `get_answer` (second)"
    | Role        | Message                                                                                                                                            |
    | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
    | *User*      | Extract the name of the author from the quotation below and answer questions.<br>"Simplicity is the ultimate sophistication." -- Leonardo da Vinci |
    | *Assistant* | The name of the author is **Leonardo da Vinci.**                                                                                                   |
    | *User*      | What is the most famous painting of the author?                                                                                                    |
    | *Assistant* | **The most famous painting of Leonardo da Vinci is the Mona Lisa.**                                                                                |
=== "Convo of `answer_questions`"
    | Role        | Message                                                                                                                                            |
    | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
    | *User*      | Extract the name of the author from the quotation below and answer questions.<br>"Simplicity is the ultimate sophistication." -- Leonardo da Vinci |
    | *Assistant* | The name of the author is **Leonardo da Vinci.**                                                                                                   |

The output of the code snippet would look like:
```plaintext
Leonardo da Vinci lived during the Renaissance era.
The most famous painting of Leonardo da Vinci is the Mona Lisa.
```

### Automatic Parallelization
Thanks to the asynchronous design of APPL, the indenpendent `gen` function calls in the `answer_questions` function can be executed in parallel, without introducing extra code for parallelization. See more details in [concurrent execution](./4_concurrent.md).

### Sequential Question Answering

In some cases, you may want to answer questions sequentially, where the answer to a question depends on the previous answers. The following code demonstrates how to answer questions sequentially by iterating over the questions and generating responses one by one.

```python linenums="1" hl_lines="9 11"
@ppl
def answer_questions(quotation: str, questions: list[str]):
    "Extract the name of the author from the quotation below and answer questions."
    quotation
    with AIRole():
        f"The name of the author is {gen(stop=NEWLINE)}"
    answers = []
    for q in questions:
        q
        with AIRole():
            (answer := gen())
            # obtain the gen response and add to the prompt, equivalent to
            # answer = gen()
            # answer
        answers.append(answer)
    return answers
```

The resulting conversation would look like (generated responses are in **bold**):

| Role        | Message                                                                                                                                            |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| *User*      | Extract the name of the author from the quotation below and answer questions.<br>"Simplicity is the ultimate sophistication." -- Leonardo da Vinci |
| *Assistant* | The name of the author is **Leonardo da Vinci.**                                                                                                   |
| *User*      | In what era did the author live?                                                                                                                   |
| *Assistant* | **Leonardo da Vinci lived during the Renaissance era.**                                                                                            |
| *User*      | What is the most famous painting of the author?                                                                                                    |
| *Assistant* | **The most famous painting of Leonardo da Vinci is the Mona Lisa.**                                                                                |