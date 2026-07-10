from typing import Iterable

import torch
def create_infinite_data_loader(dataset) -> Iterable:
    permutation = torch.randperm(len(dataset))
    while True:
        for idx in permutation:
            yield dataset[idx.item()]


