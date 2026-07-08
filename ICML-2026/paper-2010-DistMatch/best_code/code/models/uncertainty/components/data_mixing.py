import random
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Dict

import torch
from torch.utils.data import Sampler

from loader.dataset import TsDataset
from models.uncertainty.score_service import get_score_param


class SubGroupMixSampler(Sampler[int]):
    """
    Mixes mini-batches with a limited amount of subgroup in an equal manner
    """
    def __init__(self, group_keys, mix_count, batch_size, shuffle=True):
        """
        :param group_keys: [dataset_len] -> group key of datasets
        :param mix_count: amount of subgroups mixed together in one minibatch
        """
        #  debug = torch.cat(group_keys).reshape(self.no_of_groups, len(group_keys) // self.no_of_groups) check if every group has same amount of entires!
        self._groups = torch.unique(torch.cat(group_keys)).long()
        self._batch_size = batch_size
        self._group_indices = torch.arange(len(group_keys)).reshape(self.no_of_groups, len(group_keys) // self.no_of_groups)
        self._shuffle = True
        self._mix_count = mix_count
        assert self._mix_count <= self._group_indices.shape[0]  # Mix Count cant be greater than available Time Series
        assert self._batch_size % self._mix_count == 0

    @property
    def no_of_groups(self):
        return len(self._groups)

    @property
    def batches_per_ep(self):
        return self._group_indices.numel() // self._batch_size + 1

    def __iter__(self) -> Iterator[int]:
        if self._shuffle:
            self._group_indices = self._group_indices[:, torch.randperm(self._group_indices.shape[1])]
        for _ in range(self.batches_per_ep):
            sampled_groups = self._groups[torch.randperm(self.no_of_groups)][:self._mix_count]
            sampled_cols = torch.randperm(self._group_indices.shape[1], dtype=torch.long)[:self._batch_size // self._mix_count]
            yield torch.index_select(torch.index_select(self._group_indices, 0, sampled_groups), 1, sampled_cols).flatten()

    def __len__(self) -> int:
        return self.batches_per_ep


@dataclass
class MixTsData:
    ts_id: str
    X_past: Optional[torch.Tensor]
    Y_past: Optional[torch.Tensor]
    X_step: Optional[torch.Tensor]
    Y_step: Optional[torch.Tensor]
    eps_past: Optional[torch.Tensor]
    score_param: Dict = field(default_factory=dict)


INFERENCE_MIX_MODE_ALL = "mix_all"
INFERENCE_MIX_TAKE_MIX_COUNT_RAND_FIXED = "mix_rand_fixed"
INFERENCE_MIX_TAKE_MIX_COUNT_RAND_CHOOSE = "mix_rand_choose"
INFERENCE_MIX_TAKE_ALL_SUBSET = "mix_all_subsets"


class MixDataService:

    def __init__(self, inference_mix_mode, mix_inference_count, mix_inference_draws, mix_inference_sample_mem_size,
                 ts_ids, pub_inference) -> None:
        super().__init__()
        assert inference_mix_mode in [None, INFERENCE_MIX_MODE_ALL, INFERENCE_MIX_TAKE_MIX_COUNT_RAND_FIXED,
                                      INFERENCE_MIX_TAKE_MIX_COUNT_RAND_CHOOSE]
        self._mode = inference_mix_mode
        self._ts_ids = list(set(ts_ids))
        self._mix_inference_count = mix_inference_count
        self._mix_inference_draws = mix_inference_draws
        self._mix_inference_sample_mem_size = mix_inference_sample_mem_size
        self._pub_inference = pub_inference
        if inference_mix_mode is not None:
            assert mix_inference_count is not None
            assert ts_ids is not None
            assert len(self._ts_ids) >= self._mix_inference_count
        if mix_inference_draws != 1:
            assert self._mode in [INFERENCE_MIX_TAKE_MIX_COUNT_RAND_CHOOSE, INFERENCE_MIX_MODE_ALL]

    @property
    def mode(self):
        return self._mode

    @property
    def _count_diff_const(self):
        return 0 if self._pub_inference else 1

    def pack_mix_inference_data(self, mix_datasets: List[TsDataset], max_past, step_after_start) -> Optional[List[MixTsData]]:
        relevant_data = self.select_mix_inference_data(mix_datasets)
        if len(relevant_data) > 0:
            def map_to_mix_data(data):
                step = data.test_step + step_after_start
                past_start = max(0, step - max_past)
                return MixTsData(
                    ts_id=data.ts_id,
                    X_past=data.X_full[past_start:step],
                    X_step=data.X_full[step].unsqueeze(0),
                    Y_past=data.Y_full[past_start:step],
                    Y_step=data.Y_full[step].unsqueeze(0),
                    eps_past=None,  # ToDo
                    score_param=get_score_param(data)
                )
            return [map_to_mix_data(d) for d in relevant_data]
        else:
            return None

    def get_mix_inference_count(self):
        if self._mode is None:
            return None
        if self._mode in [INFERENCE_MIX_MODE_ALL, INFERENCE_MIX_TAKE_MIX_COUNT_RAND_CHOOSE,
                          INFERENCE_MIX_TAKE_ALL_SUBSET]:
            return len(self._ts_ids) - self._count_diff_const
        else:  # INFERENCE_MIX_TAKE_MIX_COUNT_RAND_FIXED:
            return self._mix_inference_count - self._count_diff_const

    def select_mix_inference_data(self, mix_dataset: List[TsDataset]):
        if self._mode is None:
            return []
        if self._mode in [INFERENCE_MIX_MODE_ALL, INFERENCE_MIX_TAKE_MIX_COUNT_RAND_CHOOSE,
                          INFERENCE_MIX_TAKE_ALL_SUBSET]:
            return mix_dataset
        else:  # INFERENCE_MIX_TAKE_MIX_COUNT_RAND_FIXED:
            return random.sample(mix_dataset, self._mix_inference_count - self._count_diff_const)

    def select_mix_inference_step_ids(self, inference_ts_id):
        if self._mode is None:
            return None
        if self._mode in [INFERENCE_MIX_MODE_ALL, INFERENCE_MIX_TAKE_MIX_COUNT_RAND_FIXED, INFERENCE_MIX_TAKE_ALL_SUBSET]:
            return None  # Select None means to use all (if count fixed mode use all available)
        else:  # INFERENCE_MIX_TAKE_MIX_COUNT_RAND_CHOOSE:
            return random.sample([_id for _id in self._ts_ids if _id != inference_ts_id], self._mix_inference_count - self._count_diff_const)

    def select_mix_inference_subsets(self):
        if self._mode is None:
            return None
        if self._mode in [INFERENCE_MIX_MODE_ALL, INFERENCE_MIX_TAKE_MIX_COUNT_RAND_FIXED,
                          INFERENCE_MIX_TAKE_MIX_COUNT_RAND_CHOOSE]:
            return None  # Select None means to use all (if count fixed mode use all available)
        else:  # INFERENCE_MIX_TAKE_ALL_SUBSET:
            raise NotImplemented("Not Ready yet")

    def mix_inference_draws(self):
        return self._mix_inference_draws

    def merge_mix_draws(self, quantile_list):
        return torch.mean(torch.stack(quantile_list), dim=0)

    def select_mix_inference_mem_samples(self, mem_size, mix_mem_size):
        if self._mode is None or self._mix_inference_sample_mem_size is None:
            return None
        elif self._mode in [INFERENCE_MIX_TAKE_ALL_SUBSET]:
            raise NotImplemented("Not Ready Yet")
        elif self._mode in [INFERENCE_MIX_TAKE_MIX_COUNT_RAND_FIXED, INFERENCE_MIX_TAKE_MIX_COUNT_RAND_CHOOSE]:
            m_size = mem_size + ((self._mix_inference_count - self._count_diff_const) * mix_mem_size)
        else:  # INFERENCE_MIX_MODE_ALL
            m_size = mem_size + ((len(self._ts_ids) - self._count_diff_const) * mix_mem_size)
        return torch.randperm(m_size)[:self._mix_inference_sample_mem_size]
