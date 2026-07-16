import itertools
from dataclasses import dataclass
from functools import partial
from typing import Callable, Iterator

import numpy as np
import torch
from gluonts.itertools import Cyclic, Map
from gluonts.transform import (
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    ValidationSplitSampler,
)
from scaleformer.chronos.dataset import (
    NumInstanceSampler,
    RegularWindowedSampler,
    SingleContextSampler,
)
from torch.utils.data import IterableDataset, get_worker_info

# used for prediction length in test mode when window style is single
# if you're predicting for more timepoints than this at a time...what are you doing??
MAX_PREDICTION_LENGTH = 1_000_000


class PseudoShuffledIterableDataset(IterableDataset):
    def __init__(self, base_dataset, shuffle_buffer_length: int = 100) -> None:
        super().__init__()
        self.base_dataset = base_dataset
        self.shuffle_buffer_length = shuffle_buffer_length
        self.generator = torch.Generator()

    def __iter__(self):
        shuffle_buffer = []

        for element in self.base_dataset:
            shuffle_buffer.append(element)
            if len(shuffle_buffer) >= self.shuffle_buffer_length:
                idx = torch.randint(
                    len(shuffle_buffer), size=(), generator=self.generator
                )
                yield shuffle_buffer.pop(idx)

        while shuffle_buffer:
            idx = torch.randint(len(shuffle_buffer), size=(), generator=self.generator)
            yield shuffle_buffer.pop(idx)


@dataclass
class TimeSeriesDataset(IterableDataset):
    datasets: list
    probabilities: list[float]
    context_length: int = 512
    prediction_length: int = 64
    model_type: str = "pretrain"
    mode: str = "train"
    np_dtype: np.dtype = np.dtype(np.float32)
    num_test_instances: int = 1
    window_style: str = "sampled"
    window_stride: int = 1
    transforms: list[Callable] | None = None
    augmentations: list[Callable] | None = None
    augmentation_rate: float = 0.0
    augmentation_probabilities: list[float] | None = None
    random_seed: int = 8097

    def __post_init__(self):
        assert len(self.probabilities) == len(self.datasets)
        assert self.mode in ("train", "validation", "test")
        self.eval_rng = np.random.default_rng(self.random_seed)

        if self.augmentations is None:
            return

        if self.augmentation_probabilities is None:
            self.augmentation_probabilities = [1.0 / len(self.augmentations)] * len(
                self.augmentations
            )

        assert len(self.augmentations) == len(self.augmentation_probabilities)
        # assert sum(self.augmentation_probabilities) == 1.0

    def shuffle(self, shuffle_buffer_length: int = 100):
        return PseudoShuffledIterableDataset(self, shuffle_buffer_length)

    def preprocess_entry(self, entry: dict, mode: str) -> dict:
        entry = {f: entry[f] for f in ["start", "target"]}
        entry["target"] = np.asarray(entry["target"], dtype=self.np_dtype)

        if mode == "train" and self.augmentations is not None:
            if np.random.rand() < self.augmentation_rate:
                augmentation_idx = np.random.choice(
                    len(self.augmentations), p=self.augmentation_probabilities
                )
                entry["target"] = self.augmentations[augmentation_idx](entry["target"])

        for transform in self.transforms or []:
            entry["target"] = transform(entry["target"])

        return entry

    def _create_instance_splitter(self, mode: str):
        assert mode in ["train", "test", "validation"]
        assert (
            self.window_style
            in [
                "sampled",  # randomly sample eval windows from each timeseries
                "rolling",  # take sliding windows of context_length with a stride of window_stride from each timeseries
                "single",  # get only the first context window from each timeseries, predict the rest
            ]
        ), "evaluation windows can only either be rolling or randomly sampled"

        test_sampler = {
            "sampled": partial(
                NumInstanceSampler, N=self.num_test_instances, rng=self.eval_rng
            ),
            "rolling": partial(RegularWindowedSampler, stride=self.window_stride),
            "single": SingleContextSampler,
        }[self.window_style]

        instance_sampler = {
            "train": ExpectedNumInstanceSampler(
                num_instances=1.0,
                min_instances=1,
                min_past=self.context_length,  # never sample behind the timeseries
                min_future=self.prediction_length,  # never sample too far ahead
            ),
            "test": test_sampler(
                min_past=self.context_length,
                min_future=self.prediction_length,
            ),
            "validation": ValidationSplitSampler(min_future=self.prediction_length),
        }[mode]

        prediction_length = (
            MAX_PREDICTION_LENGTH
            if mode == "test" and self.window_style == "single"
            else self.prediction_length
        )
        return InstanceSplitter(
            target_field="target",
            is_pad_field="is_pad",
            start_field="start",
            forecast_start_field="forecast_start",
            instance_sampler=instance_sampler,
            past_length=self.context_length,
            future_length=prediction_length,
            dummy_value=np.nan,
        )

    def create_training_data(self, data):
        data = Cyclic(data)
        split_transform = self._create_instance_splitter("train")
        data = split_transform.apply(data, is_train=True)
        return data

    def create_test_data(self, data):
        data = self._create_instance_splitter("test").apply(data, is_train=False)
        return data

    def create_validation_data(self, data):
        data = self._create_instance_splitter("validation").apply(data, is_train=False)
        return data

    def to_hf_format(self, entry: dict) -> dict:
        past_target = torch.tensor(entry["past_target"], dtype=torch.float32)

        # pretraining models do not need labels
        if self.model_type == "pretrain":
            return {"past_values": past_target}

        future_target = torch.tensor(entry["future_target"], dtype=torch.float32)
        return {"past_values": past_target, "future_values": future_target}

    def __iter__(self) -> Iterator:
        preprocessed_datasets = [
            Map(partial(self.preprocess_entry, mode=self.mode), dataset)
            for dataset in self.datasets
        ]

        if self.mode == "train":
            iterables = [
                self.create_training_data(dataset) for dataset in preprocessed_datasets
            ]
        elif self.mode == "test":
            iterables = [
                self.create_test_data(dataset) for dataset in preprocessed_datasets
            ]
        else:
            iterables = [
                self.create_validation_data(dataset)
                for dataset in preprocessed_datasets
            ]

        worker_info = get_worker_info()
        if worker_info is None:
            probs = list(self.probabilities)
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            iterables = list(itertools.islice(iterables, worker_id, None, num_workers))
            probs = list(
                itertools.islice(self.probabilities, worker_id, None, num_workers)
            )

        probs = [prob / sum(probs) for prob in probs]

        iterators = list(map(iter, iterables))
        if self.mode == "train":
            while True:
                idx = np.random.choice(range(len(iterators)), p=probs)
                try:
                    data = next(iterators[idx])
                    yield self.to_hf_format(data)
                except StopIteration:
                    probs[idx] = 0
                    if sum(probs) == 0:
                        return
                    probs = [prob / sum(probs) for prob in probs]
        else:
            for entry in itertools.chain(*iterators):
                yield self.to_hf_format(entry)
