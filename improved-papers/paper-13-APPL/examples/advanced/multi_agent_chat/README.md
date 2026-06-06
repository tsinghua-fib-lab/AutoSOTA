# MultiAgent Chat
This folder contains four ways of implementing a multi-agent chat system using APPL. The four implementations are:
1. [**Resume**](option1_resume.py), Recommended: This implementation uses the resume feature of the APPL function to store the state of the conversation. The context is stored in the instance of the class with a private variable.
2. [**History**](option2_history.py): This implementation uses a variable to store the history of the conversation. The history is stored in the beginning and updated at the end of each turn.
3. [**Generator**](option3_generator.py): This implementation utilizes Python's *yield* and generator capabilities, enabling functions to produce outputs in stages, temporarily hand over control, and later resume where they left off by calling the *send* method, which also allows for new inputs to be passed to the *yield* statement.
4. [**Same Context**](option4_samectx.py): This implementation creates a context in the *init* function and uses the same context throughout the conversation.
