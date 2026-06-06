import warnings
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass

import torch
from torch import nn
from transformers import Cache

from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

#############################################################################
# Kernel
#########
try:
    from importlib.metadata import distribution
    torch.ops.load_library(str(distribution("hogwild").locate_file('')) + "/hogwild/hogatt.abi3.so")
except OSError:
    warnings.warn("Could not load hogatt.abi3.so file. Defaulting to slower eager realization.")


def hogwild_fused(queries: torch.Tensor, locations: torch.Tensor, keys: list[torch.Tensor], values: list[torch.Tensor],
                  scale: float, fragment_lengths, cosines: torch.Tensor, sines: torch.Tensor, *, rotated_queries, out):
    """Custom rope+attention kernel"""
    torch.ops.libhogatt.hogwild_fused(out, rotated_queries, scale, locations, queries.contiguous(), fragment_lengths, keys, values, cosines, sines)
    return out


def hogwild_sdpa(queries: torch.Tensor, locations: torch.Tensor, keys: list[torch.Tensor], values: list[torch.Tensor],
                 scale: float, fragment_lengths=None, out=None) -> torch.Tensor:
    if out is None:
        out = torch.empty((queries.size(1), queries.size(2), queries.size(3), values[0].size(3)), dtype=queries.dtype, device=queries.device)
    if fragment_lengths is None:
        fragment_lengths = torch.tensor([k.size(2) for k in keys], dtype=torch.int32, device=queries.device)
    keys = [k.contiguous() for k in keys]
    values = [v.contiguous() for v in values]
    torch.ops.libhogatt.hogwild_sdpa(out, scale, locations, queries.contiguous(), fragment_lengths, keys, values)
    return out


def hogwild_sdpa_pt(queries: torch.Tensor, locations: torch.Tensor, keys: list[torch.Tensor], values: list[torch.Tensor],
                    scale: float, return_intermediate_results: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pure pytorch implementation of Howild! attention. Useful for testing and debugging.
    """
    qk = []
    vals = []
    for f in range(len(keys)):
        # GQA replication
        key = keys[f].repeat_interleave(queries.size(-3) // keys[f].size(-3), -3)
        val = values[f].repeat_interleave(queries.size(-3) // values[f].size(-3), -3)
        # key and query position -> mask
        kp = torch.arange(0, key.size(-2), dtype=torch.int32, device=key.device)
        qp = locations[f]
        kp = kp[None, None, None, :]
        qp = qp[:, None, :, None]
        mask = kp > qp
        att = queries[f].to(torch.float32) @ key.transpose(-1, -2).to(torch.float32)
        att.masked_fill_(mask, float("-inf"))
        qk.append(att)
        vals.append(val)

    qk = torch.concat(qk, dim=-1)
    vals = torch.concat(vals, dim=-2).to(torch.float32)
    att = torch.softmax(scale * qk, dim=-1)
    result = att @ vals
    if return_intermediate_results:
        return result, qk, att
    else:
        return result.to(queries.dtype)


def hogwild_rope(queries: torch.Tensor, cosines: torch.Tensor, sines: torch.Tensor, out=None):
    if out is None:
        out = torch.empty((cosines.size(0), queries.size(0), queries.size(1), queries.size(2), queries.size(3)), dtype=queries.dtype, device=queries.device)
    torch.ops.libhogatt.hogwild_rope(out, queries, cosines, sines)
    return out


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return ((x * cos) + (rotate_half(x) * sin)).to(x.dtype)


#############################################################################
# Cache
#########

@dataclass
class InternalCacheMeta:
    cos: list[torch.Tensor] | list[None] | torch.Tensor
    sin: list[torch.Tensor] | list[None] | torch.Tensor
    loc: list[torch.Tensor] | list[None] | torch.Tensor
    cs: Cache = None


@dataclass
class CacheStructure:
    keys: list[torch.Tensor]  # keys of the fragment
    values: list[torch.Tensor]  # values of this fragment
    frags: torch.Tensor  # fragment lengths
    cos: torch.Tensor  # cosines to apply to query
    sin: torch.Tensor  # sines to apply to query
    loc: torch.Tensor  # relative location


class HogwildCache(Cache):
    def __init__(
            self,
            cache_structure: List[List[Cache]],
            model,
            write_to: Optional[List[Cache]] = None,
    ):
        self.model = model.model
        self.cache_structure = cache_structure
        self.write_to = write_to if write_to else [cl[-1] for cl in cache_structure]
        self.cosines = []
        self.sines = []
        self.locations = []
        self.segments = []
        self.frags = []
        self.queries_buffer = None
        self.att_buffer = None

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        # TODO THIS IS WRONG IF WE DO MERGING
        return self.cache_structure[0][-1].get_seq_length()

    def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> CacheStructure:
        # update the worker caches
        assert key_states.shape[0] == len(self.cache_structure)

        # assume each batch index corresponds to one worker
        mask: Optional[torch.Tensor] = cache_kwargs.get('mask', None)
        if mask is not None:
            new_tokens_per_worker = torch.sum(mask, dim=1)
            # assuming left-masking; only copy non-masked KVs to the cache
            for w in range(key_states.shape[0]):
                n = new_tokens_per_worker[w]
                # B H L D
                self.write_to[w].update(key_states[w:w + 1, :, -n:, :], value_states[w:w + 1, :, -n:, :], layer_idx, cache_kwargs)
        else:
            for w in range(key_states.shape[0]):
                self.write_to[w].update(key_states[w:w + 1, ...], value_states[w:w + 1, ...], layer_idx, cache_kwargs)

        if layer_idx == 0:
            mapping: Dict[int, InternalCacheMeta] = {}
            workers = len(self.cache_structure)

            for cs in self.cache_structure[0]:
                mapping[id(cs)] = InternalCacheMeta(
                    cos=[None] * workers, sin=[None] * workers, loc=[None] * workers, cs=cs)

            # and construct the info we need to actually run attention
            for w in range(key_states.shape[0]):
                pos = 0
                for cs in reversed(self.cache_structure[w]):
                    pos += cs.get_seq_length(layer_idx)
                    # at this point, pos already includes the newly-added tokens
                    # so, in order to match the right query position, we need to subtract the number of
                    # tokens currently being added
                    pos_t = torch.arange(pos - key_states.shape[2], pos, device=key_states.device, dtype=torch.int32)
                    mapping[id(cs)].loc[w] = pos_t

            # rearrange
            locations = []
            segments = []
            for entry in mapping.values():
                locations += entry.loc
                segments.append(entry.cs)

            locations = torch.stack(locations, dim=0)
            cosines, sines = self.model.rotary_emb(key_states, locations)
            self.cosines = cosines.reshape(len(segments), workers, locations.shape[1], cosines.shape[2]).to(torch.float)
            self.sines = sines.reshape(len(segments), workers, locations.shape[1], cosines.shape[2]).to(torch.float)
            self.locations = locations.reshape(len(segments), workers, locations.shape[1])
            self.segments = segments
            self.frags = torch.tensor([cs.get_seq_length(layer_idx) for cs in self.segments], dtype=torch.int32, device=self.cosines.device)
            # for some reason, having an explicit graph break is *essential* for good performance
            torch._dynamo.graph_break()
        keys = []
        vals = []
        for cs in self.segments:
            keys.append(cs.key_cache[layer_idx].contiguous())
            vals.append(cs.value_cache[layer_idx].contiguous())
        return CacheStructure(keys=keys, values=vals, cos=self.cosines, sin=self.sines, loc=self.locations, frags=self.frags)

    def get_queries_buffer(self, queries, layer_idx):
        if layer_idx == 0:
            self.queries_buffer = torch.empty((self.cosines.size(0), queries.size(0), queries.size(1),
                                               queries.size(2), queries.size(3)),
                                              dtype=queries.dtype, device=queries.device)
        return self.queries_buffer

    def get_att_buffer(self, r_queries, layer_idx):
        if layer_idx == 0:
            self.att_buffer = torch.empty((r_queries.size(1), r_queries.size(2), r_queries.size(3),
                                           self.cache_structure[0][0].value_cache[layer_idx].size(3)),
                                          dtype=r_queries.dtype, device=r_queries.device)
        return self.att_buffer

    def get_input_kwargs(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None):
        nw = len(self.cache_structure)
        assert input_ids.shape[0] == nw, (input_ids.shape, nw)

        kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'past_key_values': self,
            'use_cache': True,
        }

        if attention_mask is not None:
            new_tokens_per_worker = torch.sum(attention_mask, dim=1)
        else:
            new_tokens_per_worker = torch.full((nw,), input_ids.shape[1], dtype=torch.int64, device=input_ids.device)

        pos_ids = []
        for w in range(nw):
            e = self.cache_structure[w][-1].get_seq_length() + new_tokens_per_worker[w]
            s = e - input_ids.shape[1]
            pw = torch.arange(s, e, dtype=torch.int64, device=input_ids.device)
            pos_ids.append(pw)
        pos_ids = torch.stack(pos_ids, dim=0)
        kwargs['position_ids'] = pos_ids

        return kwargs


def merge_caches(target: Cache, source: Cache, model):
    locations = torch.full((1, source.get_seq_length()), target.get_seq_length(), dtype=torch.int32, device=source.key_cache[0].device)
    cos, sin = model.rotary_emb(source.key_cache[0], locations)
    for layer_id in range(len(target.key_cache)):
        # re-rotate
        key_states = apply_rotary_pos_emb(source.key_cache[layer_id], cos, sin)
        target.update(key_states, source.value_cache[layer_id], layer_id)


#############################################################################
# Model
#########


class AttentionModuleForQwen2(nn.Module):
    """Modified attention layer adapted to HogwildCache.
    """

    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=False
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: Tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor],
            past_key_value: Optional[HogwildCache] = None,
            cache_position: torch.LongTensor = None,
            position_ids: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        if input_shape[0] != 1:
            assert position_ids is not None, "When processing multiple workers, `position_ids` are mandatory"

        if position_ids is not None:
            assert position_ids.shape == hidden_states.shape[:-1], (position_ids.shape, hidden_states.shape)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        # queries will be rotated for individual segments, so nothing to do here
        key_states = apply_rotary_pos_emb(key_states, cos, sin)

        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position, 'mask': attention_mask}
        cache: CacheStructure = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if self.config._attn_implementation == "eager":
            queries = torch.tile(torch.unsqueeze(query_states, 0), (cache.cos.size(0), 1, 1, 1, 1))
            queries = apply_rotary_pos_emb(queries, cache.cos, cache.sin, unsqueeze_dim=2)
            attn_output = hogwild_sdpa_pt(queries, cache.loc, cache.keys, cache.values, self.scaling, False)
        else:
            rq = past_key_value.get_queries_buffer(query_states, layer_idx=self.layer_idx)
            attn_output = hogwild_fused(
                query_states,
                cache.loc,
                cache.keys,
                cache.values,
                self.scaling,
                cache.frags,
                cache.cos, cache.sin,
                rotated_queries=rq,
                out=past_key_value.get_att_buffer(rq, layer_idx=self.layer_idx),
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None


class AttentionModuleForQwen3(nn.Module):
    """Modified attention layer adapted to HogwildCache.
    """

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: Tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor],
            past_key_value: Optional[HogwildCache] = None,
            cache_position: torch.LongTensor = None,
            position_ids: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        if input_shape[0] != 1:
            assert position_ids is not None, "When processing multiple workers, `position_ids` are mandatory"

        if position_ids is not None:
            assert position_ids.shape == hidden_states.shape[:-1], (position_ids.shape, hidden_states.shape)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        # queries will be rotated for individual segments, so nothing to do here
        key_states = apply_rotary_pos_emb(key_states, cos, sin)

        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position, 'mask': attention_mask}
        cache: CacheStructure = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if self.config._attn_implementation == "eager":
            queries = torch.tile(torch.unsqueeze(query_states, 0), (cache.cos.size(0), 1, 1, 1, 1))
            queries = apply_rotary_pos_emb(queries, cache.cos, cache.sin, unsqueeze_dim=2)
            attn_output = hogwild_sdpa_pt(queries, cache.loc, cache.keys, cache.values, self.scaling, False)
        else:
            rq = past_key_value.get_queries_buffer(query_states, layer_idx=self.layer_idx)
            attn_output = hogwild_fused(
                query_states,
                cache.loc,
                cache.keys,
                cache.values,
                self.scaling,
                cache.frags,
                cache.cos, cache.sin,
                rotated_queries=rq,
                out=past_key_value.get_att_buffer(rq, layer_idx=self.layer_idx),
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None


def model_surgery(model):
    for l in model.model.layers:
        old = l.self_attn
        if isinstance(model.config, Qwen2Config):
            l.self_attn = AttentionModuleForQwen2(model.model.config, l.self_attn.layer_idx)
        elif isinstance(model.config, Qwen3Config):
            l.self_attn = AttentionModuleForQwen3(model.model.config, l.self_attn.layer_idx)
            l.self_attn.q_norm = old.q_norm
            l.self_attn.k_norm = old.k_norm
        else:
            raise NotImplementedError(f"Unsupported model type: {model.config.__class__}")

        l.self_attn.k_proj = old.k_proj
        l.self_attn.v_proj = old.v_proj
        l.self_attn.q_proj = old.q_proj
        l.self_attn.o_proj = old.o_proj

    # pass through the attention mask as-is
    model._update_causal_mask = lambda attention_mask, *args: attention_mask
    model.model._update_causal_mask = lambda attention_mask, *args: attention_mask
