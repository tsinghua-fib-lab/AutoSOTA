from appl import convo, gen, ppl, records


@ppl  # use empty context
def intro():
    f"Today is 2024/02/29."
    return records()


@ppl(ctx="same")  # same as the caller's context
def addon():
    f"Dates should be in the format of YYYY/MM/DD."
    # The newly captured prompt will influence the caller's context


@ppl(ctx="copy")  # copy the caller's context
def query(question: str):
    # the prompts here will not influence the caller's context
    f"Q: {question}"
    f"A: "
    # print(convo())  # display the conversation used for the generation
    return gen()


@ppl
def answer_questions(questions: list[str]):
    # long prompt can be decomposed into several smaller `appl functions`
    # method 1 (recommended): build the sub-prompts in an empty context
    intro()  # returns prompt records, will be captured in this context
    # method 2: use the same context and modify it in the function
    addon()  # returns None, not captured, but the context is modified inside
    return [query(q) for q in questions]


questions = [
    "What's the date tomorrow?",
    "What's the date yesterday?",
    "How many dates passed since 2024/02/02?",
]
for res in answer_questions(questions):
    print(res)
