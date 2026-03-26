"""
A wrapper for a cache that generates inputs (caches, masks, etc) for parallel shared_cache
"""
from collections import defaultdict
from typing import Sequence, Optional, Dict, Union

import torch
import transformers
from torch import nn

from .cache_block import CacheBlock
from .combined_cache import CombinedCacheView


class SharedCacheManager(nn.Module):
    def __init__(self, cache_structure: Sequence[Sequence[CacheBlock]], write_to: Optional[Sequence[CacheBlock]] = None):
        """
        :param cache_structure: a List[num_workers] of sequences of CacheBlock that can be reused between workers,
          each worker's cache is formed by concatenating
          >>> promptA_cache, promptB_cache, workerA_cache, workerB_cache = [CacheBlock(...) for _ in range(4)]
          >>> SharedCacheManager(cache_structure=[
          ...     promptA_cache, workerB_cache, workerA_cache,  # A sees its prompt, the other guy (B), then himself
          ...     promptB_cache, workerA_cache, workerB_cache,  # B sees its prompt, the other guy (A), then himself
          ... ])
        :param write_to: a List[num_workers] of caches to which the new tokens will be written by default;
          if not specified, defaults to writing to the last cache in each worker's sequence
        """
        super().__init__()
        self.cache_structure, self.write_to = cache_structure, write_to

    def get_input_kwargs(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        write_to: Optional[Sequence[CacheBlock]] = None,
        active_worker_indices: Optional[Sequence[int]] = None,
    ) -> Dict[str, Union[torch.Tensor, CombinedCacheView]]:
        """
        Construct a dictionary of kwargs meant for AutoModelForCausalLM.forward that enables shared_cache with shared cache

        :param input_ids: a tensor of per-worker tokens [num_workers, length]
          Some of the input_ids can be None (or empty sequences) - such workers will be inactive during this shared_cache
        :param attention_mask: a boolean mask of the same shape as input_ids, where 0 denotes padding, [num_workers, length]
          Note that when using attention mask, shorter sequences must be padded from the left side.
        :param write_to: a List[num_workers] of caches to which the new tokens will be written. Default to self.write_to

        Return
         - yields a dictionary of kwargs that make the model run shared_cache with shared cache
         - guarantees that model(**yielded_input_dict).logits[:, -1] contains next token predictions for active workers
            in the same order as in input_ids; if some of the workers are missing,
         - after forward, the KVs for new tokens will be written to write_to (if not specified - use the write_to from __init__)
         - only the workers that are given inputs will run forward pass - the rest are unused
        """
        assert input_ids.ndim == 2  # [num_workers, num_tokens]
        assert attention_mask is None or attention_mask.shape == input_ids.shape, "mask must be for inputs only"
        input_attention_mask = attention_mask
        if active_worker_indices is None:
            cache_structure = self.cache_structure
            write_to = write_to if write_to is not None else self.write_to
        else:
            cache_structure = [self.cache_structure[worker_i] for worker_i in active_worker_indices]
            if write_to is None and self.write_to is not None:  # if both are None, keep None
                write_to = [self.write_to[worker_i] for worker_i in active_worker_indices]
        write_to = write_to if write_to is not None else [seq[-1] for seq in cache_structure]
        assert len(input_ids) == len(cache_structure) == len(write_to)

        # infer the size of each worker cache after new tokens are added; some caches affect multiple workers
        new_tokens_by_row = [input_ids.shape[1]] * len(input_ids) if attention_mask is None else attention_mask.sum(1)
        extra_length_by_block: Dict[CacheBlock, int] = defaultdict(lambda: 0)
        for write_target_cache, num_added_tokens in zip(write_to, new_tokens_by_row):
            extra_length_by_block[write_target_cache] += num_added_tokens

        worker_new_seq_lengths = [sum(
            cache_block.get_seq_length() + extra_length_by_block[cache_block] for cache_block in worker_sequence)
            for worker_sequence in cache_structure]

        # construct attention_mask, assuming left-padded cache
        max_length_per_worker = max(worker_new_seq_lengths)
        worker_new_seq_lengths = torch.tensor(worker_new_seq_lengths, dtype=torch.int64, device=input_ids.device)
        worker_padding_sizes = max_length_per_worker - worker_new_seq_lengths  # num zeros in left-padded cache

        position_range = torch.arange(max_length_per_worker, device=input_ids.device)
        cache_attention_mask = position_range[None, :] >= worker_padding_sizes[:, None]  # [num_workers, tokens_w_cache]

        cache_position = position_range[-input_ids.shape[1]:]  # [past_seen_tokens: past_seen_tokens + input length]
        position_ids = cache_position[None, :] - worker_padding_sizes[:, None]  # [num_workers, input_ids.shape[1]]
        print(input_attention_mask)
        return dict(
            input_ids=input_ids,
            attention_mask=cache_attention_mask,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=True,
            past_key_values=CombinedCacheView(
                cache_structure=cache_structure, write_to=write_to,
                input_mask=input_attention_mask, position_ids=position_ids, override_length=max_length_per_worker,
                rotary_cache=dict()  # <-- start a new cache for RoPE cos/sin values that will last for one forward pass
            )
        )


def test_basic_shared_cache_manager(model_names=("meta-llama/Llama-3.2-3B", "Qwen/Qwen2.5-3B")):
    for model_name in model_names:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
        config = transformers.AutoConfig.from_pretrained(model_name)
        cache_prompt, cache_w1, cache_w2 = (CacheBlock(config=config) for _ in range(3))
        cm = SharedCacheManager(cache_structure=[
            [cache_prompt, cache_w2, cache_w1],
            [cache_prompt, cache_w1, cache_w2],
        ], write_to=[cache_w1, cache_w2])

        # pre-fill prompt only - it will be used for both workers later
        with torch.no_grad():
            model(**tokenizer("A cat sat on a mat", return_tensors='pt'), use_cache=True, past_key_values=cache_prompt)
            assert cache_prompt.get_seq_length() > 0

        input_ids = tokenizer(["for worker 1", "for worker 2"], return_tensors='pt', add_special_tokens=False)['input_ids']
        input_kwargs = cm.get_input_kwargs(input_ids=input_ids)
        with torch.no_grad():
            logits = model(**input_kwargs).logits[..., -1, :]
            assert logits.shape[0] == len(cm.cache_structure)


if __name__ == '__main__':
    test_basic_shared_cache_manager()
