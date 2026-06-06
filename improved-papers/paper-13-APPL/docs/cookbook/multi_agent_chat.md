# Multi-Agent Chat
We demonstrate four different ways of implementing a multi-agent chat system using APPL. The four implementations are:

1. [**Resume**](#resume) **(Recommended)**: Uses the resume feature of the APPL function to store the state of the conversation. The context is stored in the instance of the class with a private variable.
2. [**History**](#history): Uses a variable to store the history of the conversation. The history is stored in the beginning and updated at the end of each turn.
3. [**Generator**](#generator): Utilizes Python's *yield* and generator capabilities, enabling functions to produce outputs in stages, temporarily hand over control, and later resume where they left off by calling the *send* method, which also allows for new inputs to be passed to the *yield* statement.
4. [**Same Context**](#same-context): Creates a context in the *init* function and uses the same context throughout the conversation.

## Implementations

### 1. Resume
```python linenums="1" 
--8<-- "examples/advanced/multi_agent_chat/option1_resume.py"
```


### 2. History
```python linenums="1"
--8<-- "examples/advanced/multi_agent_chat/option2_history.py"
```

### 3. Generator
```python linenums="1"
--8<-- "examples/advanced/multi_agent_chat/option3_generator.py"
```

### 4. Same Context
```python linenums="1"
--8<-- "examples/advanced/multi_agent_chat/option4_samectx.py"
```

## Example Conversation

The example conversation between agents:

| Role    | Message                                                                                                                                                                 |
| ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| *Alice* | Hello! How can I assist you today?                                                                                                                                      |
| *Bob*   | Hi there! I'm just here to chat and answer any questions you might have. How's your day going?                                                                          |
| *Alice* | Thank you for asking! My day is going well. How about you?                                                                                                              |
| *Bob*   | I'm just a virtual assistant, so I don't have feelings, but I'm here to help you with anything you need. Is there anything specific you'd like to talk about or ask me? |

The conversation seen by Alice:

| Role        | Message                                                                                        |
| ----------- | ---------------------------------------------------------------------------------------------- |
| *System*    | Your name is Alice.                                                                            |
| *User*      | Hello!                                                                                         |
| *Assistant* | **Hello! How can I assist you today?**                                                         |
| *User*      | Hi there! I'm just here to chat and answer any questions you might have. How's your day going? |
| *Assistant* | **Thank you for asking! My day is going well. How about you?**                                 |

The conversation seen by Bob:

| Role        | Message                                                                                                                                                                     |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| *System*    | Your name is Bob.                                                                                                                                                           |
| *User*      | Hello! How can I assist you today?                                                                                                                                          |
| *Assistant* | **Hi there! I'm just here to chat and answer any questions you might have. How's your day going?**                                                                          |
| *User*      | Thank you for asking! My day is going well. How about you?                                                                                                                  |
| *Assistant* | **I'm just a virtual assistant, so I don't have feelings, but I'm here to help you with anything you need. Is there anything specific you'd like to talk about or ask me?** |
