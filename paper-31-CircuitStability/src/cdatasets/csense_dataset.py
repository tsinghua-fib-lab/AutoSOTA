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
Q: Where would someone most likely go to enjoy hiking?
Options:
A supermarket
B apartment
C museum
D mountain trail
E bank
A: Let's think step by step.
A supermarket is for shopping not hiking. An apartment is where people live not hiking. A museum is for viewing exhibits not hiking. A mountain trail is a natural location for hiking. A bank is for financial transactions not hiking. So the answer is (D).

Q: If someone wanted to attend a celebration, where would they most likely go?
Options:
A hospital
B library
C party
D graveyard
E office
A: Let's think step by step.
A hospital is for medical treatment not celebrations. A library is for reading and studying not celebrations. A party is a social gathering for celebrations. A graveyard is a burial ground not for celebrations. An office is a workplace not for celebrations. So the answer is C.

Q: If someone wanted to write something down, where would they most likely write it?
A basketball court
B notebook
C stadium
D kitchen
E garage
A: Let's think step by step.
A basketball course is for playing basketball not writing. A notebook is a place to write things down. A stadium is for sports events not writing. A kitchen is for cooking not writing. A garage is for parking cars not writing. So the answer is B.
"""

PROB_HEADER = "Q: "


class CommonSenseDataset(BaseDataset):
    """Common sense question answering task."""

    description = "Answer multiple-choice questions using common sense."
    data_file = "commonsense_qa.jsonl"

    def __init__(self, n=1000):
        super().__init__()
        self.n = n
        self._examples = []
        self._clean_examples = []
        self._corrupted_examples = []
        self._labels = []

    def get_questions(self):
        with open(Path(__file__).parent / "data" / self.data_file) as f:
            data = [json.loads(line) for line in f]
        for ex in data:
            single_input = ex["question"]["stem"] + "\nOptions:"
            single_target = f"({ex['answerKey']})"
            for choice in ex["question"]["choices"]:
                single_input += f"\n({choice['label']}) {choice['text']}"
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
