import torch
from copy import copy
from enum import Enum, auto
from itertools import count

from dyllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    FULL = auto()
    SPARSE = auto()
    FINISHED = auto()


class Sequence:
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params=SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.FULL
        self.token_ids = copy(token_ids)
        self.mask_id = sampling_params.mask_id
        self.last_tokens = [token_ids[-1]]
        self.last_token_pos = [0]
        self.salient_ids = []
        self.output_length = sampling_params.max_new_tokens
        self.num_full_steps = sampling_params.num_full_steps
        self.num_steps = sampling_params.steps
        self.processed_steps = 0
        self.num_prompt_tokens = sum(1 for t in token_ids if t != self.mask_id)
        self.num_tokens = self.num_prompt_tokens
        self.temperature = sampling_params.temperature
        self.max_new_tokens = sampling_params.max_new_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.top_p = sampling_params.top_p
        self.top_k = sampling_params.top_k
        self.confidence_threshold = sampling_params.confidence_threshold

        self.block_size = sampling_params.block_size
        self.block_idx = 0

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[: self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens :]

    @property
    def idx_updated_rows(self):
        return self.idx_updated_rows

    @property
    def num_transfer_tokens(self):
        assert self.output_length % self.num_steps == 0
        return self.output_length // self.num_steps

    def update_token(self, pos: list[int], token_id: list[int]):
        for p, t in zip(pos, token_id):
            if self.token_ids[p] == self.mask_id and t != self.mask_id:
                self.num_tokens += 1
            self.token_ids[p] = t
        self.last_tokens = copy(token_id)
        self.last_token_pos = copy(pos)

    def update_block_idx(self):
        if self.num_completion_tokens > 0 and self.block_size > 0:
            self.block_idx = (self.num_completion_tokens) // self.block_size

    def __getstate__(self):
        return (
            self.num_tokens,
            self.num_prompt_tokens,
            self.token_ids if self.num_completion_tokens == 0 else self.last_tokens,
        )

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_tokens = state[-1]
