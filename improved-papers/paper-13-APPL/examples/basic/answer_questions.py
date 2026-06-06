from appl import AIRole, gen, ppl
from appl.const import NEWLINE


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
    "What is the most famous painting of the author?",
]
for ans in answer_questions(quotation, questions):
    print(ans)
