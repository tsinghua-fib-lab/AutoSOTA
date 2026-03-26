from typing import Literal

from pydantic import BaseModel, Field

from appl import gen, ppl


class Answer(BaseModel):
    # You can use the description annotaion to guide the generation of the structured output.
    thoughts: str = Field(..., description="The thoughts for thinking step by step.")
    answer: Literal["Yes", "No"]


@ppl
def answer(question: str):
    "Answer the question below."
    question
    # Use response_obj to retrieve the generation results will give correct type hint.
    return gen(response_format=Answer).response_obj


ans = answer(
    "The odd numbers in this group add up to an even number: 15, 32, 5, 13, 82, 7, 1."
)
print(f"Thoughts:\n{ans.thoughts}")
print(f"Answer: {ans.answer}")
