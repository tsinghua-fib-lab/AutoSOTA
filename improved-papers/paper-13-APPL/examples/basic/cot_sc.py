import time

from appl import gen, ppl


def parse_answer(answer: str):
    # parse the ANS from: The answer is [ANS].
    if (key := "The answer is ") in answer:
        return answer.split(key)[-1].split(".")[0].strip()
    return None


def get_mode(answers: list[str]):
    """Get the mode of the answers"""
    return max(set(answers), key=answers.count)


def marginalize(results: list):
    """Get the answer from the results and get the mode of the answers"""
    # explicitly syncronize the results using str()
    answers = [parse_answer(str(res)) for res in results]

    return get_mode(answers)


@ppl
def cot_consistency(cot_examples: list[str], question: str, num_trials: int):
    cot_examples  # the list of examples are captured into prompt one-by-one
    question
    results = [gen() for _ in range(num_trials)]  # concurrent generation
    return marginalize(results)  # marginalize the reasoning paths to get the answer


@ppl
def cot_consistency_sequential(cot_examples: list[str], question: str, num_trials: int):
    cot_examples  # the list of examples are captured into prompt one-by-one
    question
    results = [str(gen()) for _ in range(num_trials)]  # explicit syncronization
    return marginalize(results)  # marginalize the reasoning paths to get the answer


# example from https://www.promptingguide.ai/techniques/cot
cot_examples = [
    (
        "The odd numbers in this group add up to an even number: 4, 8, 9, 15, 12, 2, 1.\n"
        "A: Adding all the odd numbers (9, 15, 1) gives 25. The answer is False."
    ),
    (
        "The odd numbers in this group add up to an even number: 17, 10, 19, 4, 8, 12, 24.\n"
        "A: Adding all the odd numbers (17, 19) gives 36. The answer is True."
    ),
]
question = (
    "The odd numbers in this group add up to an even number: 15, 32, 5, 13, 82, 7, 1."
)


n = 5
start_time = time.time()
print(f"Parallel CoT-SC Answer: {cot_consistency(cot_examples, question, n)}")
print(f"Parallel CoT-SC takes {time.time() - start_time:.2f} seconds")

start_time = time.time()
print(
    f"Sequential CoT-SC Answer: {cot_consistency_sequential(cot_examples, question, n)}"
)
print(f"Sequential CoT-SC takes {time.time() - start_time:.2f} seconds")
