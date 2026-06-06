import random
from functools import partial

from .base import BaseDataset
from .prompts import PromptFormatter
from .utils import generic_collate

from torch.utils.data import DataLoader


COT = """\
Q: Complete the rest of the sequence, making sure that the parentheses are closed properly. Input: [(){}
A: Let's think step by step.
We should process each input one by one and keep track of the stack configuration.
0: empty stack
1: [ ; stack: [
2: ( ; stack: [(
3: ) ; stack: [
4: { ; stack: [{
5: } ; stack: [
So the answer is ]

Q: Complete the rest of the sequence, making sure that the parentheses are closed properly. Input: {[]()
A: Let's think step by step.
We should process each input one by one and keep track of the stack configuration.
0: empty stack
1: { ; stack: {
2: [ ; stack: {[
3: ] ; stack: {
4: ( ; stack: {(
5: ) ; stack: {
So the answer is }

Q: Complete the rest of the sequence, making sure that the parentheses are closed properly. Input: {([])
A: Let's think step by step.
We should process each input one by one and keep track of the stack configuration.
0: empty stack
1: { ; stack: {
2: ( ; stack: {(
3: [ ; stack: {([
4: ] ; stack: {(
5: ) ; stack: {
So the answer is }\
"""

PROB_HEADER = "Q: Complete the rest of the sequence, making sure that the parentheses are closed properly. Input: "


class DyckDataset(BaseDataset):
    """A Dyck language task of variable difficulty."""

    description = "Correctly close a Dyck-n word."
    choices = [")", "}", "]"]

    def __init__(
        self, open_brackets="([{", closed_brackets=")]}", n=1000, max_length=15
    ):
        super().__init__()
        self.open_brackets = open_brackets
        self.closed_brackets = closed_brackets
        self.max_length = max_length
        self.n = n
        self._closed_dict = {
            o: c for c, o in zip(self.closed_brackets, self.open_brackets)
        }
        self._examples = []
        self._clean_examples = []
        self._corrupted_examples = []
        self._labels = []

    def get_questions(self):
        for _ in range(self.n):
            q, a = self._single_dyck()
            self._examples.append(
                {
                    "input": q,
                    "target": a,
                }
            )

    def _single_dyck(self):
        stack = []
        dyck = random.choice(self.open_brackets)
        stack.append(self._closed_dict[dyck])

        while len(dyck) + len(stack) < self.max_length:
            if len(stack) == 0 or random.random() < 0.5:
                o = random.choice(self.open_brackets)
                dyck += o
                stack.append(self._closed_dict[o])
            else:
                dyck += stack.pop()
        dyck += "".join(stack[::-1])
        return dyck[:-1], dyck[-1]

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
        self._corrupted_examples = self._clean_examples[:]
        random.shuffle(self._corrupted_examples)
        self._labels = [v["target"] for v in self._examples]

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
