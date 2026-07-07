import random
from abc import abstractmethod
from pathlib import Path
from typing import List, Union

import torch
from loguru import logger
from torch.utils.data import Dataset


class Example(object):
    def __init__(
        self, file_path: Union[List, str], label: Union[List, int], label_name: str
    ):
        self.data_info = {}
        self.file_path = file_path

        if isinstance(file_path, list):
            # for video frames
            self.file_name = [Path(i).name for i in file_path]
        else:
            self.file_name = Path(file_path).name

        self.label_name = label_name
        self.label = torch.tensor(label, dtype=torch.long)

    def __repr__(self) -> str:
        message = f"Data point of type {self.__class__.__name__}\n"
        message += f"label(s): {self.label}\n"
        message += f"label name(s): {self.label_name}\n"
        message += f"file path(s): {self.file_path}\n\n"
        for key, value in self.data_info.items():
            message += f"{key}: {value}\n"
        return message

    @abstractmethod
    def load_example(self) -> torch.Tensor:
        raise NotImplementedError

    def fetch_example(self) -> tuple:
        data = self.load_example()

        datum = random.choice(data) if isinstance(data, list) else data
        assert datum.dtype == torch.float32, "inputs must be torch.float32"
        assert datum.size(
            -1) == self.data_info["width"], "width must be last dim"
        assert (
            datum.size(-2) == self.data_info["height"]
        ), "height must be second last dim"

        return data, self.label, self.label_name, self.file_name


class AbstractDataset(Dataset):
    def __init__(
        self,
        label_names: List,
        label2name: dict,
        examples: List,
    ):

        self.label_names = label_names
        self.label2name = label2name
        self.num_classes = len(label2name)
        self.examples = examples
        self._num_tries = 3

    @abstractmethod
    def create_examples(self) -> list:
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, data: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __len__(self):
        return len(self.examples)

    def _fetch_example(self, idx) -> tuple:
        num_tries = self._num_tries

        while num_tries > 0:
            try:
                return self.examples[idx].fetch_example()

            except Exception as excep:
                logger.warning(f"Error: {excep}")
                logger.warning(
                    f"Error in loading example {self.examples[idx].file_path}")
                idx = random.choice(range(len(self.examples)))
                num_tries = num_tries - 1

    def __getitem__(self, idx) -> dict:
        data, label, label_name, file_name = self._fetch_example(idx)
        data = self.preprocess(data)
        return data, label, label_name, file_name
