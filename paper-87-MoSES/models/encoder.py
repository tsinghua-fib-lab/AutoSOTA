import sys
from typing import Tuple, Union, Dict, Any, Callable, List

import torch
import torch.nn as nn

from rl4co.utils.pylogger import get_pylogger
from rl4co.models.nn.mlp import MLP
from torch import Tensor
from functools import partial

from models.env_embeddings.mtvrp import MTVRPInitEmbeddingRouteFinder, MTVRPPromptEmbedding
from models.nn.transformer import Normalization, TransformerBlock, ParallelGatedMLP, RMSNorm
from models.nn.lora import LinearWithLoRA, LoRALayer, LoRANorm
from models.nn.lora import LinearWithGatedMultiLoRA, GatedMultiLoRALayer
from envs.mtvrp import MTVRPEnv

log = get_pylogger(__name__)



class RouteFinderEncoder(nn.Module):
    """
    Encoder for RouteFinder model based on the Transformer Architecture.
    Here we include additional embedding from raw to embedding space, as
    well as more modern architecture options compared to the usual Attention Models
    based on POMO (including multi-task VRP ones).
    """

    def __init__(
        self,
        init_embedding: nn.Module = None,
        num_heads: int = 8,
        embed_dim: int = 128,
        num_layers: int = 6,
        feedforward_hidden: int = 512,
        normalization: str = "instance",
        use_prenorm: bool = False,
        use_post_layers_norm: bool = False,
        parallel_gated_kwargs: dict = None,
        **transformer_kwargs,
    ):
        super(RouteFinderEncoder, self).__init__()

        if init_embedding is None:
            init_embedding = MTVRPInitEmbeddingRouteFinder(embed_dim=embed_dim)
        else:
            log.warning("Using custom init_embedding")
        self.init_embedding = init_embedding

        self.layers = nn.Sequential(
            *(
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    normalization=normalization,
                    use_prenorm=use_prenorm,
                    feedforward_hidden=feedforward_hidden,
                    parallel_gated_kwargs=parallel_gated_kwargs,
                    **transformer_kwargs,
                )
                for _ in range(num_layers)
            )
        )

        self.post_layers_norm = (
            Normalization(embed_dim, normalization) if use_post_layers_norm else None
        )

    def forward(
        self, td: Tensor, mask: Union[Tensor, None] = None
    ) -> Tuple[Tensor, Tensor]:

        # Transfer to embedding space
        init_h = self.init_embedding(td)  # [B, N, H]

        # Process embedding
        h = init_h
        for layer in self.layers:
            h = layer(h, mask)

        # https://github.com/meta-llama/llama/blob/8fac8befd776bc03242fe7bc2236cdb41b6c609c/llama/model.py#L493
        if self.post_layers_norm is not None:
            h = self.post_layers_norm(h)

        # Return latent representation
        return h, init_h  # [B, N, H]




class LoRAEncoder(nn.Module):
    def __init__(self,
                 encoder: RouteFinderEncoder = None,
                 lora_rank: int = 64,
                 lora_alpha: float = 0.5,
                 lora_use_gate: bool = True,
                 lora_act_func: str = 'sigmoid',
                 lora_use_linear: bool = False,
                 lora_init_embed: bool = True,
                 lora_query: bool = True,
                 lora_key: bool = True,
                 lora_value: bool = True,
                 lora_projection: bool = True,
                 lora_ffn: bool = True,
                 lora_norm: bool = True,
                 assign_lora: Callable = None,
                 ):
        super(LoRAEncoder, self).__init__()
        if assign_lora == None:
            assign_lora = partial(
                LinearWithLoRA,
                rank=lora_rank,
                alpha=lora_alpha,
                use_gate=lora_use_gate,
                act_func=lora_act_func,
                use_linear=lora_use_linear,
            )

        self.lora_init_embed = lora_init_embed
        self.lora_query = lora_query
        self.lora_key = lora_key
        self.lora_value = lora_value
        self.lora_projection = lora_projection
        self.lora_ffn = lora_ffn
        self.lora_norm = lora_norm

        self.encoder = self.assign_lora_layers(encoder, assign_lora)

    def assign_lora_layers(self, encoder, assign_lora):
        assert isinstance(encoder.init_embedding, MTVRPInitEmbeddingRouteFinder)

        if self.lora_init_embed:
            encoder.init_embedding.project_global_feats = assign_lora(encoder.init_embedding.project_global_feats)
            encoder.init_embedding.project_customers_feats = assign_lora(encoder.init_embedding.project_customers_feats)

        for layer in encoder.layers:
            if self.lora_query:
                layer.attention.Wq = assign_lora(layer.attention.Wq)
            if self.lora_key:
                layer.attention.Wk = assign_lora(layer.attention.Wk)
            if self.lora_value:
                layer.attention.Wv = assign_lora(layer.attention.Wv)
            if self.lora_projection:
                layer.attention.out_proj = assign_lora(layer.attention.out_proj)
            if self.lora_ffn:
                if isinstance(layer.ffn, MLP):
                    for ffn_lin in layer.ffn.lins:
                        ffn_lin = assign_lora(ffn_lin)

                elif isinstance(layer.ffn, ParallelGatedMLP):
                    layer.ffn.l1 = assign_lora(layer.ffn.l1)
                    layer.ffn.l2 = assign_lora(layer.ffn.l2)
                    layer.ffn.l3 = assign_lora(layer.ffn.l3)
                else:
                    raise NotImplementedError
            if self.lora_norm:
                assert isinstance(layer.norm_attn.normalizer, RMSNorm)
                assert isinstance(layer.norm_ffn.normalizer, RMSNorm)
                layer.norm_attn.normalizer = assign_lora(layer.norm_attn.normalizer)
                layer.norm_ffn.normalizer = assign_lora(layer.norm_ffn.normalizer)

        if self.lora_norm and encoder.post_layers_norm is not None:
            encoder.post_layers_norm.normalizer = assign_lora(encoder.post_layers_norm.normalizer)

        return encoder
    

    def forward(self, td: Tensor, mask: Union[Tensor, None] = None) -> Tuple[Tensor, Tensor]:
        return self.encoder(td, mask)



class MultiLoRAEncoder(LoRAEncoder):
    def __init__(self,
                 encoder: RouteFinderEncoder = None,
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
                 lora_init_embed: bool = True,
                 lora_query: bool = True,
                 lora_key: bool = True,
                 lora_value: bool = True,
                 lora_projection: bool = True,
                 lora_ffn: bool = True,
                 lora_norm: bool = True,
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

        super(MultiLoRAEncoder, self).__init__(
            encoder = encoder,
            lora_init_embed = lora_init_embed,
            lora_query = lora_query,
            lora_key = lora_key,
            lora_value = lora_value,
            lora_projection = lora_projection,
            lora_ffn = lora_ffn,
            lora_norm = lora_norm,
            assign_lora = assign_lora,
        )




class CadaEncoder(nn.Module):
    def __init__(self,
                 init_embedding: nn.Module = None,
                 prompt_embedding: nn.Module = None,
                 num_heads: int = 8,
                 embed_dim: int = 128,
                 num_layers: int = 6,
                 feedforward_hidden: int = 512,
                 normalization: str = "rms",
                 use_prenorm: bool = False,
                 use_post_layers_norm: bool = False,
                 parallel_gated_kwargs: dict = None,
                 attn_sparse_ratio: float = 0.5,
                 sparse_applied_to_score: bool = None,
                 **transformer_kwargs,
                 ):
        super(CadaEncoder, self).__init__()
        self.num_layers = num_layers

        if init_embedding is None:
            init_embedding = MTVRPInitEmbeddingRouteFinder(embed_dim=embed_dim)
        else:
            log.warning("Using custom init_embedding")
        self.init_embedding = init_embedding

        if prompt_embedding is None:
            prompt_embedding = MTVRPPromptEmbedding(embed_dim=embed_dim, normalization=None)
        else:
            log.warning("Using custom prompt_embedding")
        self.prompt_embedding = prompt_embedding

        self.global_layers = nn.Sequential(
            *(
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    feedforward_hidden=feedforward_hidden,
                    normalization=normalization,
                    use_prenorm=use_prenorm,
                    parallel_gated_kwargs=parallel_gated_kwargs,
                    attn_sparse_ratio=1.0,
                    **transformer_kwargs,
                )
                for _ in range(num_layers)
            )
        )

        self.sparse_layers = nn.Sequential(
            *(
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    feedforward_hidden=feedforward_hidden,
                    normalization=normalization,
                    use_prenorm=use_prenorm,
                    parallel_gated_kwargs=parallel_gated_kwargs,
                    attn_sparse_ratio=attn_sparse_ratio,
                    sparse_applied_to_score=sparse_applied_to_score,
                    **transformer_kwargs,
                )
                for _ in range(num_layers)
            )
        )

        self.global_linears = nn.Sequential(
            *(nn.Linear(embed_dim, embed_dim, bias=True) for _ in range(num_layers))
        )
        self.sparse_linears = nn.Sequential(
            *(nn.Linear(embed_dim, embed_dim, bias=True) for _ in range(num_layers))
        )

        self.post_layers_norm = (
            Normalization(embed_dim, normalization) if use_post_layers_norm else None
        )


    def forward(
        self, td: Tensor, mask: Union[Tensor, None] = None
    ) -> Tuple[Tensor, Tensor]:

        has_open, has_tw, has_limit, has_backhaul, backhaul_class = MTVRPEnv.check_variants(td)
        prompt = torch.stack((has_open, has_tw, has_limit, has_backhaul), dim=-1).to(torch.float32)

        # Transfer to embedding space
        init_h = self.init_embedding(td)  # [B, N, H]
        init_h = self.prompt_embedding(prompt, init_h)

        hg = init_h
        hs = init_h
        for i in range(self.num_layers):
            hg_layer = self.global_layers[i]
            hs_layer = self.sparse_layers[i]
            hg_linear = self.global_linears[i]
            hs_linear = self.sparse_linears[i]

            _hg = hg_layer(hg, mask)
            _hs = hs_layer(hs, mask)

            hg = _hg + hg_linear(_hs)
            hs = _hs + hs_linear(_hg)


        # https://github.com/meta-llama/llama/blob/8fac8befd776bc03242fe7bc2236cdb41b6c609c/llama/model.py#L493
        if self.post_layers_norm is not None:
            hg = self.post_layers_norm(hg)

        # Return latent representation
        return hg, init_h  # [B, N, H]




class CadaLoRAEncoder(nn.Module):
    def __init__(self,
                 encoder: CadaEncoder = None,
                 lora_rank: int = 32,
                 lora_alpha: float = 0.5,
                 lora_use_gate: bool = True,
                 lora_act_func: str = 'sigmoid',
                 lora_use_linear: bool = False,
                 lora_init_embed: bool = True,
                 lora_prompt_embed: bool = True,
                 lora_query: bool = True,
                 lora_key: bool = True,
                 lora_value: bool = True,
                 lora_projection: bool = True,
                 lora_ffn: bool = True,
                 lora_fusion: bool = True,
                 lora_norm: bool = True,
                 assign_lora: Callable = None,
                 ):
        super(CadaLoRAEncoder, self).__init__()

        if assign_lora == None:
            assign_lora = partial(
                LinearWithLoRA,
                rank=lora_rank,
                alpha=lora_alpha,
                use_gate=lora_use_gate,
                act_func=lora_act_func,
                use_linear=lora_use_linear,
            )

        self.lora_init_embed = lora_init_embed
        self.lora_prompt_embed = lora_prompt_embed
        self.lora_query = lora_query
        self.lora_key = lora_key
        self.lora_value = lora_value
        self.lora_projection = lora_projection
        self.lora_ffn = lora_ffn
        self.lora_fusion = lora_fusion
        self.lora_norm = lora_norm

        self.encoder = self.assign_lora_layers(encoder, assign_lora)

    def assign_lora_layers(self, encoder: CadaEncoder, assign_lora):
        assert isinstance(encoder.init_embedding, MTVRPInitEmbeddingRouteFinder)
        assert isinstance(encoder.prompt_embedding, MTVRPPromptEmbedding)

        if self.lora_init_embed:
            encoder.init_embedding.project_global_feats = assign_lora(encoder.init_embedding.project_global_feats)
            encoder.init_embedding.project_customers_feats = assign_lora(encoder.init_embedding.project_customers_feats)

        if self.lora_prompt_embed:
            encoder.prompt_embedding.prompt_l1 = assign_lora(encoder.prompt_embedding.prompt_l1)
            encoder.prompt_embedding.prompt_l2 = assign_lora(encoder.prompt_embedding.prompt_l2)
            encoder.prompt_embedding.prompt_l3 = assign_lora(encoder.prompt_embedding.prompt_l3)

        def assign_transformer(layers):
            for layer in layers:
                if self.lora_query:
                    layer.attention.Wq = assign_lora(layer.attention.Wq)
                if self.lora_key:
                    layer.attention.Wk = assign_lora(layer.attention.Wk)
                if self.lora_value:
                    layer.attention.Wv = assign_lora(layer.attention.Wv)
                if self.lora_projection:
                    layer.attention.out_proj = assign_lora(layer.attention.out_proj)
                if self.lora_ffn:
                    if isinstance(layer.ffn, MLP):
                        for ffn_lin in layer.ffn.lins:
                            ffn_lin = assign_lora(ffn_lin)

                    elif isinstance(layer.ffn, ParallelGatedMLP):
                        layer.ffn.l1 = assign_lora(layer.ffn.l1)
                        layer.ffn.l2 = assign_lora(layer.ffn.l2)
                        layer.ffn.l3 = assign_lora(layer.ffn.l3)
                    else:
                        raise NotImplementedError
                if self.lora_norm:
                    assert isinstance(layer.norm_attn.normalizer, RMSNorm)
                    assert isinstance(layer.norm_ffn.normalizer, RMSNorm)
                    layer.norm_attn.normalizer = assign_lora(layer.norm_attn.normalizer)
                    layer.norm_ffn.normalizer = assign_lora(layer.norm_ffn.normalizer)

        assign_transformer(encoder.global_layers)
        assign_transformer(encoder.sparse_layers)

        for i, layer in enumerate(encoder.global_linears):
            if self.lora_fusion:
                encoder.global_linears[i] = assign_lora(layer)
        for i, layer in enumerate(encoder.sparse_linears):
            if self.lora_fusion:
                encoder.sparse_linears[i] = assign_lora(layer)

        if self.lora_norm and encoder.post_layers_norm is not None:
            encoder.post_layers_norm.normalizer = assign_lora(encoder.post_layers_norm.normalizer)

        return encoder



    def forward(self, td: Tensor, mask: Union[Tensor, None] = None) -> Tuple[Tensor, Tensor]:
        return self.encoder(td, mask)




class CadaMultiLoRAEncoder(CadaLoRAEncoder):
    def __init__(self,
                 encoder: CadaEncoder = None,
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
                 lora_init_embed: bool = True,
                 lora_prompt_embed: bool = True,
                 lora_query: bool = True,
                 lora_key: bool = True,
                 lora_value: bool = True,
                 lora_projection: bool = True,
                 lora_ffn: bool = True,
                 lora_fusion: bool = True,
                 lora_norm: bool = True,
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
                use_basis_variants_as_input=lora_use_basis_variants_as_input,
                use_linear=lora_use_linear,
            )

        super(CadaMultiLoRAEncoder, self).__init__(
            encoder = encoder,
            lora_init_embed = lora_init_embed,
            lora_prompt_embed = lora_prompt_embed,
            lora_query = lora_query,
            lora_key = lora_key,
            lora_value = lora_value,
            lora_projection = lora_projection,
            lora_ffn = lora_ffn,
            lora_fusion = lora_fusion,
            lora_norm = lora_norm,
            assign_lora = assign_lora,
        )



