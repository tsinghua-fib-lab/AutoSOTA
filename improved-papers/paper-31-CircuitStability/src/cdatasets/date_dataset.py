import os
import json
import random
import string
from pathlib import Path
from functools import partial

from .base import BaseDataset
from .prompts import PromptFormatter
from .utils import generic_collate

from torch.utils.data import DataLoader


CHOICES = [f"({v})" for v in string.ascii_uppercase]
COT = """\
Q: Today is Christmas Eve of 1937. What is the date 10 days ago in MM/DD/YYYY?
Options:
(A) 12/14/2026
(B) 12/14/1950
(C) 12/14/2007
(D) 12/14/1937
(E) 07/14/1938
(F) 12/14/1988
A: Let's think step by step.
If today is Christmas Eve of 1937, then today's date is December 24, 1937. 10 days before today is December 14, 1937, that is 12/14/1937. So the answer is (D).

Q: Tomorrow is 11/12/2019. What is the date one year ago from today in MM/DD/YYYY?
Options:
(A) 09/04/2018
(B) 11/11/2018
(C) 08/25/2018
(D) 11/02/2018
(E) 11/04/2018
A: Let's think step by step.
If tomorrow is 11/12/2019, then today is 11/11/2019. The date one year ago from today is 11/11/2018. So the answer is (B).

Q: Jane and John married on Jan 2, 1958. It is their 5-year anniversary today. What is the date tomorrow in MM/DD/YYYY?
Options:
(A) 01/11/1961
(B) 01/03/1963
(C) 01/18/1961
(D) 10/14/1960
(E) 01/03/1982
(F) 12/03/1960
A: Let's think step by step.
If Jane and John married on Jan 2, 1958, then and if it is their 5-year anniversary today, then today's date is Jan 2, 1963. The date tomorrow is Jan 3, 1963, that is 01/03/1963. So the answer is (B).
"""

PROB_HEADER = "Q: "


class DateDataset(BaseDataset):
    """Date understanding task."""

    description = "Infer the date from context."
    data_file = "date_understanding.json"

    def __init__(self, n=1000):
        super().__init__()
        self.n = n
        self._examples = []
        self._clean_examples = []
        self._corrupted_examples = []
        self._labels = []

    def get_questions(self):
        task = json.load(open(Path(__file__).parent / "data" / self.data_file))
        for ex in task["examples"]:
            single_input = ex["input"] + "\nOptions:"
            single_target = ""
            targets = list(ex["target_scores"].items())
            random.shuffle(targets)
            for i, (k, v) in enumerate(targets):
                single_input += "\n" + f"{CHOICES[i]} {k}"
                if v == 1:
                    single_target = CHOICES[i]
            self._examples.append({"input": single_input, "target": single_target})
        random.shuffle(self._examples)
        self._examples = self._examples[: self.n]

    def format_questions(self, formatter: PromptFormatter):
        global PROB_HEADER, COT
        Qs, As = [PROB_HEADER + f"{v['input']}" for v in self._examples], [
            f"\nA: {v['target']}" for v in self._examples
        ]
        answer_starter = "\nA: "
        if formatter.name == "chain-of-thought":
            answer_starter = "\nA: Let's think step by step."
        self._clean_examples = [
            formatter.format(
                task_description=self.description,
                prompt=v + answer_starter,
                questions=Qs,
                answers=As,
                cot=COT,
            )
            for v in Qs
        ]
        self._labels = [v["target"] for v in self._examples]
        for i in range(len(self._clean_examples)):
            # for each clean example, get example that has different target
            diff = list(
                filter(
                    lambda j: self._labels[j] != self._labels[i],
                    range(len(self._examples)),
                )
            )
            self._corrupted_examples.append(self._clean_examples[random.choice(diff)])

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
