import json
import random
import pickle
from pathlib import Path
from functools import partial

from .base import BaseDataset
from .prompts import PromptFormatter
from .utils import generic_collate

from torch.utils.data import DataLoader


class PromptDataset(BaseDataset):
    """Investigating prompting stability"""

    def __init__(self, fname, parts, part_size):
        super().__init__()
        self.parts = parts
        self.part_size = part_size
        self.fname = fname
        self._examples = []
        self._clean_examples = []
        self._corrupted_examples = []
        self._labels = []
        self._partition_index = 0
        self._pclean_examples = []
        self._pcorrupted_examples = []
        self._plabels = []

    def get_questions(self):
        task = pickle.load(open(self.fname, "rb"))
        for i in range(len(task["output"])):
            self._examples.append(
                {"input": task["input"][i], "output": task["output"][i]}
            )

        random.shuffle(self._examples)

    def format_questions(self, formatter: PromptFormatter = None):
        if formatter is not None:
            raise ValueError("PromptDataset does not support any formatting.")
        self._clean_examples = [v["output"] for v in self._examples]
        self._corrupted_examples = [v["input"] for v in self._examples]
        self._labels = [v["output"] for v in self._examples]

    def __len__(self):
        return len(self._pclean_examples)

    def __getitem__(self, idx):
        return (
            self._pclean_examples[idx],
            self._pcorrupted_examples[idx],
            self._plabels[idx],
        )

    @property
    def partition_index(self):
        return self._partition_index

    @partition_index.setter
    def partition_index(self, val):
        if val >= self.parts:
            print("tried setting pindex > parts; handling gracefully")
            val = self.parts - 1
        self._partition_index = val
        self._pclean_examples = self._clean_examples[
            self._partition_index
            * self.part_size : (self._partition_index + 1)
            * self.part_size
        ]
        self._pcorrupted_examples = self._corrupted_examples[
            self._partition_index
            * self.part_size : (self._partition_index + 1)
            * self.part_size
        ]
        self._plabels = self._labels[
            self._partition_index
            * self.part_size : (self._partition_index + 1)
            * self.part_size
        ]

    def to_dataloader(self, model, batch_size: int, collate_fn=None):
        collate_fn = partial(generic_collate, model)
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_fn)
