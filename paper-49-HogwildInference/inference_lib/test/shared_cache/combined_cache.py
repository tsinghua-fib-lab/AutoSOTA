import logging
import os

from typing import Dict, Sequence, Tuple, Optional, Any, List

import torch
import transformers

from .cache_block import CacheBlock, using_rotary_cache

logger = logging.getLogger(__name__)


class CombinedCacheView(transformers.cache_utils.Cache):
    """
    A collection of multiple partially shared individual CacheBlock instances that combines them just-in-time.
    This instance is created by a cache manager and is designed to last for a single forward pass.

    :param cache_structure: a List[num_workers] of sequences of CacheBlock that can be reused between workers,
        see SharedCacheManager (shared_cache_manager.py) for a more detailed documentation on cache_structure

    :param write_to: for each worker, the target cache that the new keys/values should be written to
        if write_to is not specified, the tokens will be written to the last cache in each worker's sequence

    :param input_mask: a boolean mask for cache update size, where 0 (False) denotes padding, [num_workers, length]
        The tokens where input_mask is False will *not* be added to KV cache, but may still be used as queries

    :param position_ids: overrides cache_position in case each individual worker's cache has different max positions

    :param rotary_cache: (optional) a dictionary in which to cache intermediate rotary cos/sin values. If not specified,
        these values will not be cached and will be recomputed. To start a new cache, feed it an empty dict.

    :param override_length: overrides get_usable_length computation to this value minus the provided input length
    """

    def __init__(
            self,
            cache_structure: Sequence[Sequence[CacheBlock]],
            write_to: Optional[Sequence[CacheBlock]] = None,
            input_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            override_length: Optional[int] = None,
            rotary_cache: Optional[dict] = None
    ):
        super().__init__()
        assert write_to is None or len(write_to) == len(cache_structure)
        assert input_mask is None or len(input_mask) == len(cache_structure)
        assert position_ids is None or len(position_ids) == len(cache_structure)
        self.cache_structure, self.write_to, self.rotary_cache = cache_structure, write_to, rotary_cache
        self.input_mask, self.position_ids, self.override_length = input_mask, position_ids, override_length
        if write_to is not None:
            assert all(write_target not in seq for write_target, seq in zip(self.cache_structure, self.write_to)), (
                "Some of the write targets for workers are not parts of their cache structure. This may indicate that "
                " the cache structure was changed but you forgot to update the write_to. You may bypass this error,"
                " if you know what you're doing, but it may cause some of the current key/value states to be invisible"
                " to the model during current step.")

    def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ combine multiple CacheBlocks to construct keys/values for each worker; write new entries to the chunk indicated in write_to (see __init__) """
        # if write_to is not specified, write to the last cache of each worker
        num_workers, num_new_tokens = len(self.cache_structure), key_states.shape[-2]
        write_to = [cache_seq[-1] for cache_seq in self.cache_structure] if self.write_to is None else self.write_to
        assert key_states.shape[0] == value_states.shape[0] == len(write_to) == num_workers
        assert key_states.shape[-2] == value_states.shape[-2] == num_new_tokens
        assert (len(set(write_to)) == len(write_to)), "duplicate write_to targets"  # should still work, but not tested
        if self.input_mask is None:
            position_selectors_by_worker = [slice(0, key_states.shape[-2])] * len(self.cache_structure)
        else:
            assert self.input_mask.shape == (num_workers, num_new_tokens)
            assert not self.input_mask.dtype.is_floating_point, "mask must be int/bool"
            position_selectors_by_worker = self.input_mask.to(dtype=torch.bool, device=key_states.device)
        if self.position_ids is None:
            cache_position_by_worker = [cache_kwargs.get("cache_position")] * len(self.cache_structure)
        else:
            assert len(self.position_ids) == num_workers and len(self.position_ids[0]) == num_new_tokens
            cache_position_by_worker = self.position_ids.to(device=key_states.device)

        if (cache_kwargs.keys() == {'cos', 'sin', 'cache_position'} or   # llama/qwen
                cache_kwargs.keys() == {'cos', 'sin', 'cache_position', 'sliding_window'}):  # gemma
            assert 'cos' in cache_kwargs and 'sin' in cache_kwargs
            assert cache_kwargs['cache_position'].ndim == 1
            assert cache_kwargs['cos'].ndim == cache_kwargs['sin'].ndim == 3  # batch, ninp, dim
            cache_kwargs_by_worker = [dict(
                cache_position=cache_position[position_selector],
                cos=cache_kwargs["cos"][worker_index: worker_index + 1, ..., position_selector, :],
                sin=cache_kwargs["sin"][worker_index: worker_index + 1, ..., position_selector, :]
            ) for worker_index, (cache_position, position_selector) in enumerate(
                zip(cache_position_by_worker, position_selectors_by_worker))
            ]
        elif cache_kwargs.keys() == {'cos', 'sin'}:  # deepseek style
            assert self.position_ids is not None
            assert cache_kwargs['cos'].ndim == cache_kwargs['sin'].ndim == 2  # all_positions, dim
            cache_kwargs_by_worker = [
                dict(cache_kwargs, cache_position=cache_position[position_selector])
                for cache_position, position_selector in zip(cache_position_by_worker, position_selectors_by_worker)]
        else:
            assert cache_kwargs is None, f"Unsupported cache_kwargs: {cache_kwargs}"
            cache_kwargs_by_worker = [None] * num_workers

        with using_rotary_cache(self.rotary_cache):
            # first, update all write_to targets since they can also be inputs to other workers and affect offsets
            for worker_index, (write_target, position_selector, worker_cache_kwargs) in enumerate(
                    zip(write_to, position_selectors_by_worker, cache_kwargs_by_worker)):
                write_target.append(
                    key_states=key_states[worker_index: worker_index + 1, ..., position_selector, :],
                    value_states=value_states[worker_index: worker_index + 1, ..., position_selector, :],
                    layer_idx=layer_idx,
                    cache_kwargs=worker_cache_kwargs
                )

            combined_keys, combined_values = combine_cache_from_structure(self.cache_structure, layer_idx=layer_idx)

        sliding_window = cache_kwargs.get('sliding_window')
        if sliding_window:  # a weird comparison (w/o 'is not None') copied from transformers
            effective_window_size = max(sliding_window, num_new_tokens)
            combined_keys = combined_keys[..., -effective_window_size:, :]
            combined_values = combined_values[..., -effective_window_size:, :]
        return combined_keys, combined_values

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Return current cache length, maximum among workers, for a given layer"""
        worker_cache_lengths = [sum(cache_block.get_seq_length(layer_idx) for cache_block in worker_sequence)
                                for worker_sequence in self.cache_structure]
        return max(worker_cache_lengths)

    # v-- fixes https://huggingface.co/deepseek-ai/DeepSeek-R1/blob/56d4cbbb4d29f4355bab4b9a39ccb717a14ad5ad/modeling_deepseek.py#L1654
    def get_max_length(self, *args, **kwargs):
        pass

    def get_max_cache_shape(self) -> Optional[int]:
        return None

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0):
        if self.override_length is not None:
            return self.override_length - new_seq_length
        return super().get_usable_length(new_seq_length, layer_idx=layer_idx)


def combine_cache_from_structure(cache_structure: Sequence[Sequence[CacheBlock]], layer_idx: int) -> Tuple[
    torch.Tensor, torch.Tensor]:
    """ concatenate KV tensors from all cache blocks with new RoPE rotation; reuse rotations when applicable """
    num_workers = len(cache_structure)
    kv_parts: List[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = [
        [None for _ in seq] for seq in cache_structure]
    cached_outputs: Dict[Tuple[CacheBlock, int], Tuple[torch.Tensor, torch.Tensor]] = {}
    for worker_i in range(num_workers):
        offset = 0
        for block_i in range(len(cache_structure[worker_i])):
            cache_block = cache_structure[worker_i][block_i]
            if (cache_block, offset) not in cached_outputs:
                cached_outputs[cache_block, offset] = cache_block.get_kv_with_offset(layer_idx=layer_idx, offset=offset)
            kv_parts[worker_i][block_i] = cached_outputs[cache_block, offset]
            offset += cache_block.get_seq_length(layer_idx)
    return collate_kv_with_left_padding(kv_parts)


@torch.compile(dynamic=True, disable=bool(int(os.environ.get("HOGWILD_NO_COMPILE", False))))
def collate_kv_with_left_padding(
        kv_parts: Sequence[Sequence[Tuple[torch.Tensor, torch.Tensor]]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    collate keys and values from multiple cache blocks, pad them on the left with zeros if needed
    :param kv_parts: a structure [worker_index, sequence_index, key_or_value] of KV tensors from CacheBlocks s.t.
      - the dimension -2 is reserved for sequence length
      - the number of chunks and the total sequence length can vary by worker, but all other dimensions cannot
      - the 0-th dimension is either 1 or batch size; it will be used for worker concatenation
    :returns: a tuple of (combined keys, combined values) of shape [workers * batch, ..., max_length, head_dim] s.t.
        - if worker sequences have varying length, they are left-padded with zeros
        - 0-th dimension = worker_size * batch_size, with worker_size being the outer one
        - {-2}-th dimension equals maximum sequence length (max over all workers)
        - the remaining dimensions are the same as in the individual caches
    """
    dtype, device = kv_parts[0][0][0].dtype, kv_parts[0][0][0].device
    key_part_shape, value_part_shape = tuple(kv_parts[0][0][0].shape), tuple(kv_parts[0][0][1].shape)
    sequence_lenghts = [sum(k.shape[-2] for (k, v) in worker_kv_parts) for worker_kv_parts in kv_parts]
    max_length = max(sequence_lenghts)
    num_workers = len(kv_parts)

    combined_key_shape = (num_workers,) + tuple(key_part_shape[:-2]) + (max_length,) + tuple(key_part_shape[-1:])
    combined_keys = torch.empty(combined_key_shape, dtype=dtype, device=device)
    combined_value_shape = (num_workers,) + tuple(value_part_shape[:-2]) + (max_length,) + tuple(value_part_shape[-1:])
    combined_values = torch.empty(combined_value_shape, dtype=dtype, device=device)

    for worker_index in range(num_workers):
        offset = max_length - sequence_lenghts[worker_index]  # starting from this index implements left padding
        if offset != 0:
            combined_keys[worker_index, ..., :offset, :].zero_()
            combined_values[worker_index, ..., :offset, :].zero_()
        for key_part, value_part in kv_parts[worker_index]:
            part_length = key_part.shape[-2]
            combined_keys[worker_index, ..., offset: offset + part_length, :].copy_(key_part, non_blocking=True)
            combined_values[worker_index, ..., offset: offset + part_length, :].copy_(value_part, non_blocking=True)
            offset += part_length
        assert offset == max_length
    return combined_keys.flatten(0, 1), combined_values.flatten(0, 1)


def test_collate_kv_with_left_padding():
    def make_kv(seq_length: int):
        return torch.randn(1, 8, seq_length, 128), torch.randn(1, 32, seq_length, 128)

    shared_kv = make_kv(2)
    kv_parts = [
        [make_kv(1), shared_kv, make_kv(3)],
        [make_kv(4), shared_kv, make_kv(8)],
        [],  # empty worker
        [shared_kv]
    ]

    keys, values = collate_kv_with_left_padding(kv_parts)
    max_length = max(sum(k.shape[-2] for k, v in seq) for seq in kv_parts)
    assert keys.shape[0] == values.shape[0] == len(kv_parts)
    assert keys.shape[-2] == values.shape[-2] == max_length
    for worker_index in range(len(kv_parts)):
        offset = max_length - sum(k.shape[-2] for k, v in kv_parts[worker_index])
        assert all(torch.all(torch.eq(k_or_v[worker_index, ..., :offset, :], 0)) for k_or_v in (keys, values))
        for k, v in kv_parts[worker_index]:
            assert torch.all(keys[worker_index, ..., offset: offset + k.shape[-2], :] == k)
            assert torch.all(values[worker_index, ..., offset: offset + v.shape[-2], :] == v)
            offset += k.shape[-2]


def test_combined_cache(model_names=("meta-llama/Llama-3.2-3B", "Qwen/Qwen2.5-3B"), atol: float = 1e-4):
    for model_name in model_names:
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
        config, rotary_emb = model.config, model.model.rotary_emb

        def _get_update_kwargs(num_tokens: int, batch_size: int = 1):
            key_states = torch.randn(batch_size, config.num_key_value_heads, num_tokens, 128)
            value_states = torch.randn(batch_size, config.num_key_value_heads, num_tokens, 128)
            offset = torch.randint(0, 100, size=()).item()
            cache_position = torch.arange(offset, offset + num_tokens)
            cos, sin = rotary_emb(key_states, cache_position.view(1, -1).tile(batch_size, 1))
            return dict(
                layer_idx=0,
                key_states=key_states,
                value_states=value_states,
                cache_kwargs=dict(cos=cos, sin=sin, cache_position=cache_position)
            )

        cache_prompt, cache_w1_private, cache_w1, cache_w2, cache_w3 = (CacheBlock(config=config) for _ in range(5))
        cache_structure = (
            (cache_prompt, cache_w2, cache_w3, cache_w1_private, cache_w1),  # worker 1
            (cache_prompt, cache_w1, cache_w3, cache_w2),  # worker 2
            (cache_prompt, cache_w1, cache_w2, cache_w3),  # worker 3
        )

        cache_prompt.append(**_get_update_kwargs(5))
        cache_w1_private.append(**_get_update_kwargs(2))  # <-- w1 gets 2 more tokens (11 instead of 9)
        cache_w1.append(**_get_update_kwargs(2))
        cache_w2.append(**_get_update_kwargs(1))
        cache_w3.append(**_get_update_kwargs(1))

        # case 1: call combine_cache_from_structure manually
        combined_keys, combined_values = combine_cache_from_structure(cache_structure, layer_idx=0)

        assert combined_keys.shape[0] == combined_values.shape[0] == 3  # num_workers
        assert combined_keys.shape[-2] == combined_values.shape[-2] == 11  # max length (based on w1)

        # worker 1 does not have zero padding
        assert not torch.all(combined_keys[0, ..., :2, :] == 0) and not torch.all(combined_values[0, ..., :2, :] == 0)
        # workers 2 and 3 are left-padded with zeros
        assert torch.all(combined_keys[1, ..., :2, :] == 0) and torch.all(combined_values[1, ..., :2, :] == 0)
        assert torch.all(combined_keys[2, ..., :2, :] == 0) and torch.all(combined_values[2, ..., :2, :] == 0)

        assert torch.all(combined_keys[0, ..., :5, :] == combined_keys[2, ..., 2:7, :])  # common prompt keys
        assert torch.all(combined_values[0, ..., :5, :] == combined_values[2, ..., 2:7, :])  # common prompt values

        w1_k, w1_v = cache_w1.get_kv_with_offset(layer_idx=0, offset=9)
        assert torch.allclose(combined_keys[0, ..., -2:, :], w1_k, atol=atol)
        assert torch.allclose(combined_values[0, ..., -2:, :], w1_v, atol=atol)

        w1_k_for_w23, w1_v_for_w23 = cache_w1.get_kv_with_offset(layer_idx=0, offset=5)
        assert torch.allclose(combined_keys[1:, ..., 7:9, :], w1_k_for_w23, atol=atol)  # note: 2 pad + 5 prompt = 7
        assert torch.allclose(combined_values[1:, ..., 7:9, :], w1_v_for_w23, atol=atol)

        w3_k, w3_v = cache_w3.get_kv_with_offset(layer_idx=0, offset=8)
        assert torch.allclose(combined_keys[2, ..., -1:, :], w3_k, atol=atol)
        assert torch.allclose(combined_values[2, ..., -1:, :], w3_v, atol=atol)

        w2_k_for_w1, w2_v_for_w1 = cache_w2.get_kv_with_offset(layer_idx=0, offset=5)
        assert torch.allclose(combined_keys[0, ..., 5:6, :], w2_k_for_w1, atol=atol)
        assert torch.allclose(combined_values[0, ..., 5:6, :], w2_v_for_w1, atol=atol)

        w1_private_k, w1_private_v = cache_w1_private.get_kv_with_offset(layer_idx=0, offset=7)
        assert torch.allclose(combined_keys[0, ..., 7:9, :], w1_private_k, atol=atol)
        assert torch.allclose(combined_values[0, ..., 7:9, :], w1_private_v, atol=atol)

        # case 2: use CombinedCacheView and update "by 0 tokens"
        combined_cache = CombinedCacheView(cache_structure)
        combined_keys_again, combined_values_again = combined_cache.update(
            **_get_update_kwargs(num_tokens=0, batch_size=3)  # 3 workers
        )
        assert torch.allclose(combined_keys, combined_keys_again, atol=atol)
        assert torch.allclose(combined_values, combined_values_again, atol=atol)
        assert cache_w1.get_seq_length() == 2
        assert cache_w2.get_seq_length() == cache_w3.get_seq_length() == 1

        # case 3: use CombinedCacheView and update by 1 token
        combined_cache = CombinedCacheView(
            cache_structure, input_mask=torch.tensor([[True, True], [False, False], [True, False]]))
        combined_keys_after, combined_values_after = combined_cache.update(
            **_get_update_kwargs(num_tokens=2, batch_size=3)  # 3 workers
        )
        assert combined_keys_after.shape[-2] == combined_keys.shape[-2] + 3  # +2 tokens for w1 and +1 token for w3
        assert cache_w1.get_seq_length() == 4  # was 2
        assert cache_w2.get_seq_length() == 1  # still 1 b/c of mask
        assert cache_w3.get_seq_length() == 2  # was 1

        # prompt: no changes
        assert torch.allclose(combined_keys[..., :5, :], combined_keys_after[..., :5, :], atol=atol)
        assert torch.allclose(combined_values[..., :5, :], combined_values_after[..., :5, :], atol=atol)

        # worker1 for 2 and 3 - no changes for first 2 tokens; index 9 below is because of left-padding 2
        assert torch.allclose(combined_keys[1:, ..., :9, :], combined_keys_after[1:, ..., :9, :], atol=atol)
        assert torch.allclose(combined_values[1:, ..., :9, :], combined_values_after[1:, ..., :9, :], atol=atol)

        # new 3rd and 4th token (8-th and 9th position): test that it is the correct tensor
        w1_k_for_w23, w1_v_for_w23 = cache_w1.get_kv_with_offset(layer_idx=0, offset=5)
        assert torch.allclose(combined_keys_after[1:, ..., 7:11, :], w1_k_for_w23,
                              atol=atol)  # note: 2 pad + 5 prompt = 7
        assert torch.allclose(combined_values_after[1:, ..., 7:11, :], w1_v_for_w23, atol=atol)

        # test 3rd worker cache from 2nd worker perspective with updated rotation (since w1 also added new token)
        w3_k_for_w2, w3_v_for_w2 = cache_w3.get_kv_with_offset(layer_idx=0, offset=9)
        assert torch.allclose(combined_keys_after[1, ..., 11:13, :], w3_k_for_w2,
                              atol=atol)  # note: 3 pad + 5 prompt + 4 w1 = 11
        assert torch.allclose(combined_values_after[1, ..., 11:13, :], w3_v_for_w2, atol=atol)


if __name__ == '__main__':
    test_collate_kv_with_left_padding()
    test_combined_cache()
