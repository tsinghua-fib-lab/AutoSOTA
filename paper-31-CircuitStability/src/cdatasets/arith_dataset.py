"""arith_dataset.py
Generates a dataset of random arithmetic problems of various lengths with
different operations and outputs them as a json file.
"""


import random
from functools import partial

from .base import BaseDataset
from .prompts import PromptFormatter
from .utils import generic_collate

import numpy as np
from torch.utils.data import DataLoader


class ArithDataset(BaseDataset):
    """Dataset of arithmetic problems. Generates problems with the specified
    number of digits on the fly with options for few-shot prompting.
    """

    description = "Solve the following arithmetic problems."

    def __init__(self, op, dig1, dig2, n=1000, append_ans=True):
        super().__init__()
        self.op = op
        self.dig1 = dig1
        self.dig2 = dig2
        self.append_ans = append_ans
        self.n = n
        self._examples = []
        self._clean_examples = []
        self._corrupted_examples = []
        self._labels = []

    @property
    def examples(self):
        return self._examples

    def get_questions(self):
        self._examples = []
        for _ in range(self.n):
            op1, op2, target = self._one_prob(self.dig1, self.dig2)
            single_ex = {"input": f"{op1} {self.op} {op2} = ", "target": str(target)}
            self._examples.append(single_ex)
        return self._examples

    def _one_prob(self, dig1: int, dig2: int):
        if self.op == "/":
            assert (
                dig1 >= dig2
            ), "To ensure division results in integer \
            solutions operand 1 must have more digits than operand 2."
            op2 = random.randint(int(10 ** (dig2 - 1)), int(10**dig2 - 1))
            ans = random.randint(
                int(10 ** (dig1 - dig2)), int(10 ** (dig1 - dig2 + 1) - 1)
            )
            op1 = ans * op2

            if len(str(op1)) != dig1:
                ## shorten ans to account for op1 having more digits than dig1 ##
                ans = ans // 10 + 1
                op1 = ans * op2

            return (op1, op2, ans)
        op1 = random.randint(int(10 ** (dig1 - 1)), int(10**dig1 - 1))
        op2 = random.randint(int(10 ** (dig2 - 1)), int(10**dig2 - 1))
        if self.op == "+":
            return (op1, op2, op1 + op2)
        elif self.op == "-":
            return (op1, op2, op1 - op2)
        else:
            return (op1, op2, op1 * op2)

    def format_questions(self, formatter: PromptFormatter):
        if formatter.name == "chain-of-thought":
            raise NotImplementedError(
                "Chain-of-thought not supported for arithmetic problems."
            )
        Qs, As = [v["input"] for v in self._examples], [
            v["target"] for v in self._examples
        ]
        self._clean_examples = [
            formatter.format(
                self.description,
                ex["input"] + (ex["target"] if self.append_ans else ""),
                questions=Qs,
                answers=As,
            )
            for ex in self._examples
        ]
        self._corrupted_examples = self._clean_examples[:]
        random.shuffle(self._corrupted_examples)
        self._labels = self._clean_examples

    def to_dataloader(self, model, batch_size: int, collate_fn=None):
        collate_fn = partial(generic_collate, model)
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_fn)

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, idx):
        return (
            self._clean_examples[idx],
            self._corrupted_examples[idx],
            self._labels[idx],
        )
