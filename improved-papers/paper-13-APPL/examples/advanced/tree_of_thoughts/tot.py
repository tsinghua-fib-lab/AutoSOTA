# Based on https://github.com/princeton-nlp/tree-of-thought-llm/tree/master/src/tot

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import pandas as pd
import sympy
from loguru import logger

from appl import gen, ppl, traceable

data_path = Path(__file__).parent / "24.csv"
parser = argparse.ArgumentParser()
parser.add_argument("--data-path", type=str, default=data_path)
parser.add_argument("--task-start-index", type=int, default=900)
parser.add_argument("--task-end-index", type=int, default=901)
parser.add_argument("--n-propose-sample", type=int, default=8)
parser.add_argument("--n-select-sample", type=int, default=5)
args = parser.parse_args()


def get_current_numbers(raw: str) -> str:
    # example: (left: 1 2 3)
    match = re.match(r".*\(left: (.*)\).*", raw, re.DOTALL)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"No match found for {raw}")


class Game24Task:
    def __init__(self, filepath="24.csv"):
        super().__init__()
        self.data = list(pd.read_csv(filepath)["Puzzles"])
        self.steps = 3

    def get_input(self, idx: int) -> str:
        return self.data[idx]

    def test_output(self, idx: int, expression: str) -> bool:
        numbers = re.findall(r"\d+", expression)
        problem_numbers = re.findall(r"\d+", self.data[idx])

        if sorted(numbers) != sorted(problem_numbers):
            return False

        try:
            return sympy.simplify(expression) == 24
        except Exception as e:
            return False


@dataclass
class Game24Candidate:
    start_numbers: str
    left_numbers: str
    steps: list[str]
    value: float = 0.0


# Examples and Prompts
def display(key: str, value: Union[str, List[str]]) -> str:
    if isinstance(value, list):
        return f"{key}:\n" + "\n".join(value)
    else:
        return f"{key}: {value}"


PROPOSE_EXAMPLES = [
    {
        "Input": "2 8 8 14",
        "Possible next steps": [
            "2 + 8 = 10 (left: 8 10 14)",
            "8 / 2 = 4 (left: 4 8 14)",
            "14 + 2 = 16 (left: 8 8 16)",
            "2 * 8 = 16 (left: 8 14 16)",
            "8 - 2 = 6 (left: 6 8 14)",
            "14 - 8 = 6 (left: 2 6 8)",
            "14 / 2 = 7 (left: 7 8 8)",
            "14 - 2 = 12 (left: 8 8 12)",
        ],
    }
]


@ppl
def propose(current_numbers: str):
    f"Propose at most {args.n_propose_sample} possible next steps without explanation"
    for example in PROPOSE_EXAMPLES:
        for k, v in example.items():
            display(k, v)
    f"Input: {current_numbers}"
    f"Possible next steps:"
    return gen()


@traceable
def get_all_proposals(candidates: List[Game24Candidate]) -> List[Game24Candidate]:
    all_proposals = []
    proposals = [propose(c.left_numbers) for c in candidates]
    for p, c in zip(proposals, candidates):
        for line in str(p).strip().split("\n"):
            try:
                all_proposals.append(
                    Game24Candidate(
                        start_numbers=c.left_numbers,
                        left_numbers=get_current_numbers(line),
                        steps=c.steps + [line.strip()],
                    )
                )
            except ValueError:
                pass
    return all_proposals


VALUE_EXAMPLES = [
    {"Input": "10 14", "Thoughts": ["10 + 14 = 24"], "Evaluation": "sure"},
    {
        "Input": "11 12",
        "Thoughts": ["11 + 12 = 23", "12 - 11 = 1", "11 * 12 = 132", "11 / 12 = 0.91"],
        "Evaluation": "impossible",
    },
    {
        "Input": "4 4 10",
        "Thoughts": ["4 + 4 + 10 = 18", "4 * 10 - 4 = 36", "(10 - 4) * 4 = 24"],
        "Evaluation": "sure",
    },
    {"Input": "4 9 11", "Thoughts": ["9 + 11 + 4 = 24"], "Evaluation": "sure"},
    {
        "Input": "5 7 8",
        "Thoughts": [
            "5 + 7 + 8 = 20",
            "(8 - 5) * 7 = 21",
            "I cannot obtain 24 now, but numbers are within a reasonable range",
        ],
        "Evaluation": "likely",
    },
    {
        "Input": "5 6 6",
        "Thoughts": [
            "5 + 6 + 6 = 17",
            "(6 - 5) * 6 = 6",
            "I cannot obtain 24 now, but numbers are within a reasonable range",
        ],
        "Evaluation": "likely",
    },
    {
        "Input": "10 10 11",
        "Thoughts": [
            "10 + 10 + 11 = 31",
            "(11 - 10) * 10 = 10",
            "10 10 10 are all too big",
        ],
        "Evaluation": "impossible",
    },
    {
        "Input": "1 3 3",
        "Thoughts": ["1 * 3 * 3 = 9", "(1 + 3) * 3 = 12", "1 3 3 are all too small"],
        "Evaluation": "impossible",
    },
]
VALUE_MAP = {
    "sure": 20.0,
    "likely": 1.0,
    "impossible": 0.001,
}


@ppl
def evaluate(current_numbers: str):
    "Evaluate if given numbers can reach 24 (sure/likely/impossible)"
    for example in VALUE_EXAMPLES:
        for k, v in example.items():
            display(k, v)
    f"Input: {current_numbers}"
    "Thoughts:"
    return gen()


@traceable
def get_all_values(proposals: List[Game24Candidate]) -> List[Game24Candidate]:
    values = [evaluate(p.left_numbers) for p in proposals]
    for p, v in zip(proposals, values):
        match = re.match(r".*Evaluation: (.*)", str(v), re.DOTALL)
        if match:
            if match.group(1) in VALUE_MAP:
                p.value = VALUE_MAP[match.group(1)]
            else:
                logger.warning(f"Invalid evaluation outcome: {match.group(1)}")
        else:
            logger.warning(f"No match found for evaluation: {v}")
    return proposals


# 5-shot
COT_EXAMPLES = [
    {
        "Input": "4 4 6 8",
        "Steps": [
            "4 + 8 = 12 (left: 4 6 12)",
            "6 - 4 = 2 (left: 2 12)",
            "2 * 12 = 24 (left: 24)",
        ],
        "Answer": "(6 - 4) * (4 + 8) = 24",
    },
    {
        "Input": "2 9 10 12",
        "Steps": [
            "12 * 2 = 24 (left: 9 10 24)",
            "10 - 9 = 1 (left: 1 24)",
            "24 * 1 = 24 (left: 24)",
        ],
        "Answer": "(12 * 2) * (10 - 9) = 24",
    },
    {
        "Input": "4 9 10 13",
        "Steps": [
            "13 - 10 = 3 (left: 3 4 9)",
            "9 - 3 = 6 (left: 4 6)",
            "4 * 6 = 24 (left: 24)",
        ],
        "Answer": "4 * (9 - (13 - 10)) = 24",
    },
    {
        "Input": "1 4 8 8",
        "Steps": [
            "8 / 4 = 2 (left: 1 2 8)",
            "1 + 2 = 3 (left: 3 8)",
            "3 * 8 = 24 (left: 24)",
        ],
        "Answer": "(1 + 8 / 4) * 8 = 24",
    },
    {
        "Input": "5 5 5 9",
        "Steps": [
            "5 + 5 = 10 (left: 5 9 10)",
            "10 + 5 = 15 (left: 9 15)",
            "15 + 9 = 24 (left: 24)",
        ],
        "Answer": "((5 + 5) + 5) + 9 = 24",
    },
]


@ppl
def get_answer(input_numbers: str, steps: List[str]):
    "Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Each step, you are only allowed to choose two of the remaining numbers to obtain a new number."
    for example in COT_EXAMPLES:
        for k, v in example.items():
            display(k, v)
    f"Input: {input_numbers}"
    f"Steps: {steps}"
    "Answer:"
    return gen()


JUDGE_EXAMPLES = [
    {
        "Input": "4 4 6 8",
        "Answer": "(4 + 8) * (6 - 4) = 24",
        "Judge": "sure",
    },
    {
        "Input": "2 9 10 12",
        "Answer": "2 * 12 * (10 - 9) = 24",
        "Judge": "sure",
    },
    {
        "Input": "4 9 10 13",
        "Answer": "(13 - 9) * (10 - 4) = 24",
        "Judge": "sure",
    },
    {
        "Input": "4 4 6 8",
        "Answer": "(4 + 8) * (6 - 4) + 1 = 25",
        "Judge": "impossible",
    },
    {
        "Input": "2 9 10 12",
        "Answer": "2 * (12 - 10) = 24",
        "Judge": "impossible",
    },
    {
        "Input": "4 9 10 13",
        "Answer": "(13 - 4) * (10 - 9) = 24",
        "Judge": "impossible",
    },
]


@ppl
def judge(input: str, answer: str):
    "Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Given an input and an answer, give a judgement (sure/impossible) if the answer is correct, i.e. it uses each input exactly once and no other numbers, and reach 24."
    for example in JUDGE_EXAMPLES:
        for k, v in example.items():
            display(k, v)
    f"Input: {input}"
    f"Answer: {answer}"
    "Judge:"
    return gen()


@traceable
def solve(input_numbers: str, steps: int = 3):
    current_candidates = [
        Game24Candidate(
            start_numbers=input_numbers, left_numbers=input_numbers, steps=[]
        )
    ]
    for _ in range(steps):
        # generation
        new_candidates = get_all_proposals(current_candidates)

        # evaluation
        new_candidates = get_all_values(new_candidates)

        # select
        current_candidates = sorted(
            new_candidates, key=lambda x: x.value, reverse=True
        )[: args.n_select_sample]

        for c in current_candidates:
            print(
                f"Start: {c.start_numbers}, Left: {c.left_numbers}, Value: {c.value}, Steps: {c.steps}"
            )

    filtered_candidates = [c for c in current_candidates if c.left_numbers == "24"]
    answers = [get_answer(input_numbers, c.steps) for c in filtered_candidates]
    judge_results = [judge(input_numbers, str(a)) for a in answers]
    candidates = []
    for candidate, answer, judge_result in zip(
        filtered_candidates, answers, judge_results
    ):
        answer = str(answer).rsplit("=", 1)[0].strip()
        print(f"Steps: {candidate.steps}, Answer: {answer}, Judge: {judge_result}")
        if str(judge_result).strip() == "sure":
            candidates.append(answer)
    return candidates


def main():
    task = Game24Task(args.data_path)

    num_solved = 0
    for idx in range(args.task_start_index, args.task_end_index):
        input_numbers = task.get_input(idx)
        candidates = solve(input_numbers, steps=task.steps)
        if len(candidates) > 0:
            test_result = task.test_output(idx, candidates[0])
            # print(input_numbers, candidates[0], test_result)
            result_msg = "correct" if test_result else "incorrect"
            logger.info(
                f"Proposed solution {candidates[0]} for {input_numbers} is {result_msg}"
            )
            num_solved += int(test_result)
        else:
            logger.info(f"No solution found for {input_numbers}")
        logger.info(
            f"solved {num_solved} out of {idx - args.task_start_index + 1} tasks, solved rate: {num_solved / (idx - args.task_start_index + 1)}"
        )

    total_tasks = args.task_end_index - args.task_start_index
    logger.info(
        f"[Summary] Solved {num_solved} out of {total_tasks} tasks, solved rate: {num_solved / total_tasks}"
    )


if __name__ == "__main__":
    main()
