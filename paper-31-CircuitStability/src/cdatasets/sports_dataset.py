import json
import random
import string
from pathlib import Path
from functools import partial

from .base import BaseDataset
from .prompts import PromptFormatter
from .utils import generic_collate

from torch.utils.data import DataLoader


COT = """\
Determine whether an artificially constructed sentence relating to sports is plausible or not.

Q: Is the following sentence plausible? "Bam Adebayo scored a reverse layup in the Western Conference Finals."
A: Let's think step by step. Bam Adebayo is an American basketball player. Scoring a reverse layup in the Western Conference Finals is part of the NBA Finals. So the answer is yes.

Q: Is the following sentence plausible? "Santi Cazorla scored a touchdown."
A: Let's think step by step. Santi Cazorla is a soccer player. Touchdown is part of American football and rugby. So the answer is no.

Q: Is the following sentence plausible? "DeMar DeRozan was called for the goal tend."
A: Let's think step by step. DeMar DeRozan is an American basketball player. Goal tending is part of basketball. So the answer is yes.
"""
PROB_HEADER = "Q: "


class SportsDataset(BaseDataset):
    """Sports-related common sense reasoning task."""

    description = "Determine whether a sports-related sentence is plausible."
    data_file = "sports_understanding.json"

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
            self._examples.append({"input": ex["input"], "target": ex["target"]})
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
