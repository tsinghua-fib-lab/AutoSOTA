import sys

import torch
import torch.nn as nn
from torch import Tensor
from tensordict import TensorDict
from functools import partial
from typing import Tuple, Dict, Any, Callable, List, Union

from rl4co.models.zoo.am.decoder import AttentionModelDecoder as _AttentionModelDecoder
from rl4co.models.zoo.am.decoder import PrecomputedCache
from rl4co.utils.ops import batchify, unbatchify
from rl4co.envs import RL4COEnvBase
from rl4co.models.nn.attention import PointerAttention
from rl4co.models.nn.env_embeddings import env_context_embedding, env_dynamic_embedding
from rl4co.models.nn.env_embeddings.dynamic import StaticEmbedding

from models.env_embeddings.mtvrp import MTVRPContextEmbeddingRouteFinder
from models.nn.lora import LinearWithLoRA, LoRALayer, LoRANorm
from models.nn.lora import LinearWithGatedMultiLoRA, GatedMultiLoRALayer
from envs.mtvrp import MTVRPEnv


class AttentionModelDecoder(_AttentionModelDecoder):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        env_name: str = "tsp",
        context_embedding: nn.Module = None,
        dynamic_embedding: nn.Module = None,
        mask_inner: bool = True,
        out_bias_pointer_attn: bool = False,
        linear_bias: bool = False,
        use_graph_context: bool = True,
        check_nan: bool = True,
        sdpa_fn: callable = None,
    ):
        nn.Module.__init__(self)

        if isinstance(env_name, RL4COEnvBase):
            env_name = env_name.name
        self.env_name = env_name
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        assert embed_dim % num_heads == 0

        self.context_embedding = (
            env_context_embedding(self.env_name, {"embed_dim": embed_dim})
            if context_embedding is None
            else context_embedding
        )
        self.dynamic_embedding = (
            env_dynamic_embedding(self.env_name, {"embed_dim": embed_dim})
            if dynamic_embedding is None
            else dynamic_embedding
        )
        self.is_dynamic_embedding = (
            False if isinstance(self.dynamic_embedding, StaticEmbedding) else True
        )

        # MHA with Pointer mechanism (https://arxiv.org/abs/1506.03134)
        self.pointer = PointerAttention(
            embed_dim,
            num_heads,
            mask_inner=mask_inner,
            out_bias=out_bias_pointer_attn,
            check_nan=check_nan,
            sdpa_fn=sdpa_fn,
        )

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embed_dim
        #self.project_node_embeddings = nn.Linear(
        #    embed_dim, 3 * embed_dim, bias=linear_bias
        #)
        self.project_glimpse_key = nn.Linear(
            embed_dim, embed_dim, bias=linear_bias
        )
        self.project_glimpse_value = nn.Linear(
            embed_dim, embed_dim, bias=linear_bias
        )
        self.project_logit_key = nn.Linear(
            embed_dim, embed_dim, bias=linear_bias
        )

        self.project_fixed_context = nn.Linear(embed_dim, embed_dim, bias=linear_bias)
        self.use_graph_context = use_graph_context


    def _precompute_cache(
        self, embeddings: torch.Tensor, num_starts: int = 0
    ) -> PrecomputedCache:
        """Compute the cached embeddings for the pointer attention.

        Args:
            embeddings: Precomputed embeddings for the nodes
            num_starts: Number of starts for the multi-start decoding
        """
        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key_fixed = self.project_glimpse_key(embeddings)
        glimpse_val_fixed = self.project_glimpse_value(embeddings)
        logit_key_fixed = self.project_logit_key(embeddings)
        #(
        #    glimpse_key_fixed,
        #    glimpse_val_fixed,
        #    logit_key_fixed,
        #) = self.project_node_embeddings(embeddings).chunk(3, dim=-1)

        # Optionally disable the graph context from the initial embedding as done in POMO
        if self.use_graph_context:
            graph_context = self.project_fixed_context(embeddings.mean(1))
        else:
            graph_context = 0

        # Organize in a dataclass for easy access
        return PrecomputedCache(
            node_embeddings=embeddings,
            graph_context=graph_context,
            glimpse_key=glimpse_key_fixed,
            glimpse_val=glimpse_val_fixed,
            logit_key=logit_key_fixed,
        )




class LoRADecoder(nn.Module):
    def __init__(self,
                 decoder: AttentionModelDecoder = None,
                 lora_rank: int = 64,
                 lora_alpha: float = 0.5,
                 lora_use_gate: bool = True,
                 lora_act_func: str = 'sigmoid',
                 lora_use_linear: bool = False,
                 lora_context_embed: bool = True,
                 lora_glimpse_key: bool = True,
                 lora_glimpse_value: bool = True,
                 lora_logit_key: bool = True,
                 lora_fixed_context: bool = True,
                 lora_pointer: bool = True,
                 assign_lora: Callable = None,
                 ):
        super(LoRADecoder, self).__init__()

        if assign_lora == None:
            assign_lora = partial(
                LinearWithLoRA,
                rank=lora_rank,
                alpha=lora_alpha,
                use_gate=lora_use_gate,
                act_func=lora_act_func,
                use_linear=lora_use_linear,
            )

        self.lora_context_embed = lora_context_embed
        self.lora_glimpse_value = lora_glimpse_value
        self.lora_glimpse_key = lora_glimpse_key
        self.lora_logit_key = lora_logit_key
        self.lora_fixed_context = lora_fixed_context
        self.lora_pointer = lora_pointer

        self.decoder = self.assign_lora_layers(decoder, assign_lora)

    def assign_lora_layers(self, decoder, assign_lora):
        assert decoder.is_dynamic_embedding == False
        assert isinstance(decoder.context_embedding, MTVRPContextEmbeddingRouteFinder)

        if self.lora_context_embed:
            decoder.context_embedding.project_context = assign_lora(decoder.context_embedding.project_context)
        if self.lora_glimpse_key:
            decoder.project_glimpse_key = assign_lora(decoder.project_glimpse_key)
        if self.lora_glimpse_value:
            decoder.project_glimpse_value = assign_lora(decoder.project_glimpse_value)
        if self.lora_logit_key:
            decoder.project_logit_key = assign_lora(decoder.project_logit_key)
        if decoder.use_graph_context and self.lora_fixed_context:
            decoder.project_fixed_context = assign_lora(decoder.project_fixed_context)
        if self.lora_pointer:
            decoder.pointer.project_out = assign_lora(decoder.pointer.project_out)

        return decoder

    def forward(
            self,
            td: TensorDict,
            cached: PrecomputedCache,
            num_starts: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        return self.decoder(td, cached, num_starts)

    def _precompute_cache(
            self, embeddings: torch.Tensor, num_starts: int = 0
    ) -> PrecomputedCache:
        return self.decoder._precompute_cache(embeddings, num_starts)

    def pre_decoder_hook(
            self, td, env, embeddings, num_starts: int = 0
    ) -> Tuple[TensorDict, RL4COEnvBase, PrecomputedCache]:
        return self.decoder.pre_decoder_hook(td, env, embeddings, num_starts)


class MultiLoRADecoder(LoRADecoder):
    def __init__(self,
                 decoder: AttentionModelDecoder = None,
                 lora_rank: int = 64,
                 lora_alpha: float = 0.5,
                 lora_act_func: str = 'softmax',
                 lora_n_experts: int = 4,
                 lora_top_k: int = 4,
                 lora_temperature: float = 1.0,
                 lora_use_trainable_layer: bool = False,
                 lora_use_dynamic_topK: bool = False,
                 lora_use_basis_variants: bool = False,
                 lora_use_linear: bool = False,
                 lora_use_basis_variants_as_input: bool = False,
                 lora_context_embed: bool = True,
                 lora_glimpse_key: bool = True,
                 lora_glimpse_value: bool = True,
                 lora_logit_key: bool = True,
                 lora_fixed_context: bool = True,
                 lora_pointer: bool = True,
                 assign_lora: Callable = None,
                 ):
        if assign_lora == None:
            assign_lora = partial(
                LinearWithGatedMultiLoRA,
                rank=lora_rank,
                alpha=lora_alpha,
                act_func=lora_act_func,
                n_experts=lora_n_experts,
                top_k=lora_top_k,
                temperature=lora_temperature,
                use_trainable_layer=lora_use_trainable_layer,
                use_dynamic_topK=lora_use_dynamic_topK,
                use_basis_variants=lora_use_basis_variants,
                use_linear=lora_use_linear,
                use_basis_variants_as_input=lora_use_basis_variants_as_input,
            )

        super(MultiLoRADecoder, self).__init__(
            decoder=decoder,
            lora_context_embed=lora_context_embed,
            lora_glimpse_key=lora_glimpse_key,
            lora_glimpse_value=lora_glimpse_value,
            lora_logit_key=lora_logit_key,
            lora_fixed_context=lora_fixed_context,
            lora_pointer=lora_pointer,
            assign_lora=assign_lora
        )


