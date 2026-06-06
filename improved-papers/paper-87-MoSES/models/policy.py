from typing import Callable, Dict, Any, List, Union, Optional, Tuple
import os, sys
import torch
from torch import Tensor
from tensordict import TensorDict


from rl4co.models.nn.env_embeddings.dynamic import StaticEmbedding
from rl4co.models.zoo.am import AttentionModelPolicy
from rl4co.utils.pylogger import get_pylogger

from models.encoder import RouteFinderEncoder, LoRAEncoder, MultiLoRAEncoder, CadaLoRAEncoder, CadaMultiLoRAEncoder
from models.decoder import AttentionModelDecoder, LoRADecoder, MultiLoRADecoder
from models.env_embeddings.mtvrp import (
    MTVRPContextEmbeddingRouteFinder,
    MTVRPInitEmbeddingRouteFinder,
    MTVRPPromptEmbedding,
)
from models.encoder import CadaEncoder
from models.nn.lora import GatedMultiLoRALayer
from envs.mtvrp import MTVRPEnv

from utils import collect_lora_state_dict, collect_multi_lora_state_dict, load_target_lora_module


from rl4co.envs import RL4COEnvBase, get_env
from rl4co.utils.decoding import (
    DecodingStrategy,
    get_decoding_strategy,
    get_log_likelihood,
)
from rl4co.utils.ops import calculate_entropy

log = get_pylogger(__name__)




class RouteFinderPolicy(AttentionModelPolicy):
    """
    Main RouteFinder policy based on the Transformer Architecture.
    We use the base AttentionModelPolicy for decoding (i.e. masked attention + pointer network)
    and our new RouteFinderEncoder for the encoder.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_encoder_layers: int = 6,
        num_heads: int = 8,
        normalization: str = "instance",
        feedforward_hidden: int = 512,
        parallel_gated_kwargs: dict = None,
        encoder_use_post_layers_norm: bool = False,
        encoder_use_prenorm: bool = False,
        env_name: str = "mtvrp",
        use_graph_context: bool = False,
        init_embedding: MTVRPInitEmbeddingRouteFinder = None,
        context_embedding: MTVRPContextEmbeddingRouteFinder = None,
        extra_encoder_kwargs: dict = {},
        linear_bias_decoder: bool = False,
        sdpa_fn: Callable = None,
        mask_inner: bool = True,
        out_bias_pointer_attn: bool = False,
        check_nan: bool = True,
        **kwargs,
    ):
        encoder = RouteFinderEncoder(
            init_embedding=init_embedding,
            num_heads=num_heads,
            embed_dim=embed_dim,
            num_layers=num_encoder_layers,
            feedforward_hidden=feedforward_hidden,
            normalization=normalization,
            use_prenorm=encoder_use_prenorm,
            use_post_layers_norm=encoder_use_post_layers_norm,
            parallel_gated_kwargs=parallel_gated_kwargs,
            **extra_encoder_kwargs,
        )

        if context_embedding is None:
            context_embedding = MTVRPContextEmbeddingRouteFinder(embed_dim=embed_dim)
        # mtvrp does not use dynamic embedding (i.e. only modifies the query, not key or value)
        dynamic_embedding = StaticEmbedding()

        decoder = AttentionModelDecoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            env_name=env_name,
            context_embedding=context_embedding,
            dynamic_embedding=dynamic_embedding,
            sdpa_fn=sdpa_fn,
            mask_inner=mask_inner,
            out_bias_pointer_attn=out_bias_pointer_attn,
            linear_bias=linear_bias_decoder,
            use_graph_context=use_graph_context,
            check_nan=check_nan,
        )

        super(RouteFinderPolicy, self).__init__(
            encoder=encoder,
            decoder=decoder,
            embed_dim=embed_dim,
            num_heads=num_heads,
            normalization=normalization,
            feedforward_hidden=feedforward_hidden,
            env_name=env_name,
            use_graph_context=use_graph_context,
            context_embedding=context_embedding,
            dynamic_embedding=dynamic_embedding,
            **kwargs,
        )





class LoRAPolicy(AttentionModelPolicy):
    def __init__(self,
                 embed_dim: int = 128,
                 num_encoder_layers: int = 6,
                 num_heads: int = 8,
                 normalization: str = "instance",
                 feedforward_hidden: int = 512,
                 parallel_gated_kwargs: dict = None,
                 encoder_use_post_layers_norm: bool = False,
                 encoder_use_prenorm: bool = False,
                 env_name: str = "mtvrp",
                 use_graph_context: bool = False,
                 init_embedding: MTVRPInitEmbeddingRouteFinder = None,
                 context_embedding: MTVRPContextEmbeddingRouteFinder = None,
                 extra_encoder_kwargs: dict = {},
                 linear_bias_decoder: bool = False,
                 sdpa_fn: Callable = None,
                 mask_inner: bool = True,
                 out_bias_pointer_attn: bool = False,
                 check_nan: bool = True,
                 lora_rank: int = None,
                 lora_alpha: float = None,
                 lora_use_gate: bool = True,
                 lora_act_func: str = 'sigmoid',
                 lora_use_linear: bool = False,
                 basis_policy_ckpt_path: str = None,
                 **kwargs,
                 ):

        basis_policy = RouteFinderPolicy(
            embed_dim=embed_dim,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
            normalization=normalization,
            feedforward_hidden=feedforward_hidden,
            parallel_gated_kwargs=parallel_gated_kwargs,
            encoder_use_post_layers_norm=encoder_use_post_layers_norm,
            encoder_use_prenorm=encoder_use_prenorm,
            env_name=env_name,
            use_graph_context=use_graph_context,
            init_embedding=init_embedding,
            context_embedding=context_embedding,
            extra_encoder_kwargs=extra_encoder_kwargs,
            linear_bias_decoder=linear_bias_decoder,
            sdpa_fn=sdpa_fn,
            mask_inner=mask_inner,
            out_bias_pointer_attn=out_bias_pointer_attn,
            check_nan=check_nan,
            **kwargs,
        )

        if basis_policy_ckpt_path != None:
            assert os.path.exists(basis_policy_ckpt_path), "Path is {}".format(basis_policy_ckpt_path)
            LoRAPolicy.load_basis_policy_weights(basis_policy, basis_policy_ckpt_path)
        LoRAPolicy.fix_basis_policy_weights(basis_policy)

        encoder = LoRAEncoder(
            encoder=basis_policy.encoder,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_use_gate=lora_use_gate,
            lora_act_func=lora_act_func,
            lora_use_linear=lora_use_linear,
        )
        decoder = LoRADecoder(
            decoder=basis_policy.decoder,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_use_gate=lora_use_gate,
            lora_act_func=lora_act_func,
            lora_use_linear=lora_use_linear,
        )

        super(LoRAPolicy, self).__init__(
            encoder=encoder,
            decoder=decoder,
            embed_dim=embed_dim,
            num_heads=num_heads,
            normalization=normalization,
            feedforward_hidden=feedforward_hidden,
            env_name=env_name,
            use_graph_context=use_graph_context,
            context_embedding=context_embedding,
            **kwargs,
        )

    @staticmethod
    def load_basis_policy_weights(basis_policy: RouteFinderPolicy, ckpt_path, strict=True):
        print('++++++++++Load Basis Policy Weights++++++++++')
        _policy_weights = torch.load(ckpt_path, map_location=torch.device("cpu"), weights_only=False)['state_dict']
        policy_weights = {}
        for name, weight in _policy_weights.items():
            assert name.split('.')[0] == 'policy'
            policy_weights[name.lstrip('policy.')] = weight
        basis_policy.load_state_dict(policy_weights, strict=strict)
        return

    @staticmethod
    def fix_basis_policy_weights(basis_policy: RouteFinderPolicy):
        for name, params in basis_policy.named_parameters():
            params.requires_grad = False
        return

    def lora_trainable_params(self):
        trainable_params = []
        for name, module in self.named_parameters():
            if "lora_layer" in name.split('.') or "gate_layer" in name.split('.'):
                assert module.requires_grad == True
                trainable_params.append(module)
        assert len(trainable_params) > 0
        return trainable_params



class MultiLoRAPolicy(AttentionModelPolicy):
    def __init__(self,
                 embed_dim: int = 128,
                 num_encoder_layers: int = 6,
                 num_heads: int = 8,
                 normalization: str = "instance",
                 feedforward_hidden: int = 512,
                 parallel_gated_kwargs: dict = None,
                 encoder_use_post_layers_norm: bool = False,
                 encoder_use_prenorm: bool = False,
                 env_name: str = "mtvrp",
                 use_graph_context: bool = False,
                 init_embedding: MTVRPInitEmbeddingRouteFinder = None,
                 context_embedding: MTVRPContextEmbeddingRouteFinder = None,
                 extra_encoder_kwargs: dict = {},
                 linear_bias_decoder: bool = False,
                 sdpa_fn: Callable = None,
                 mask_inner: bool = True,
                 out_bias_pointer_attn: bool = False,
                 check_nan: bool = True,
                 lora_rank: int = None,
                 lora_alpha: float = None,
                 lora_act_func: str = 'softmax',
                 lora_n_experts: int = 4,
                 lora_top_k: int = 4,
                 lora_temperature: float = 1.0,
                 lora_use_trainable_layer: bool = False,
                 lora_use_dynamic_topK: bool = False,
                 lora_use_basis_variants: bool = False,
                 lora_use_linear: bool = False,
                 lora_use_basis_variants_as_input: bool = False,
                 basis_policy_ckpt_path: str = None,
                 lora_modules_ckpt_path: List = None,
                 **kwargs,
                 ):
        basis_policy = RouteFinderPolicy(
            embed_dim=embed_dim,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
            normalization=normalization,
            feedforward_hidden=feedforward_hidden,
            parallel_gated_kwargs=parallel_gated_kwargs,
            encoder_use_post_layers_norm=encoder_use_post_layers_norm,
            encoder_use_prenorm=encoder_use_prenorm,
            env_name=env_name,
            use_graph_context=use_graph_context,
            init_embedding=init_embedding,
            context_embedding=context_embedding,
            extra_encoder_kwargs=extra_encoder_kwargs,
            linear_bias_decoder=linear_bias_decoder,
            sdpa_fn=sdpa_fn,
            mask_inner=mask_inner,
            out_bias_pointer_attn=out_bias_pointer_attn,
            check_nan=check_nan,
            **kwargs,
        )

        if basis_policy_ckpt_path != None:
            assert os.path.exists(basis_policy_ckpt_path)
            LoRAPolicy.load_basis_policy_weights(basis_policy, basis_policy_ckpt_path)
        LoRAPolicy.fix_basis_policy_weights(basis_policy)

        encoder = MultiLoRAEncoder(
            encoder = basis_policy.encoder,
            lora_rank = lora_rank,
            lora_alpha = lora_alpha,
            lora_act_func = lora_act_func,
            lora_n_experts = lora_n_experts,
            lora_top_k = lora_top_k,
            lora_temperature = lora_temperature,
            lora_use_trainable_layer = lora_use_trainable_layer,
            lora_use_dynamic_topK = lora_use_dynamic_topK,
            lora_use_basis_variants = lora_use_basis_variants,
            lora_use_linear = lora_use_linear,
            lora_use_basis_variants_as_input=lora_use_basis_variants_as_input,
        )
        decoder = MultiLoRADecoder(
            decoder = basis_policy.decoder,
            lora_rank = lora_rank,
            lora_alpha = lora_alpha,
            lora_act_func=lora_act_func,
            lora_n_experts=lora_n_experts,
            lora_top_k=lora_top_k,
            lora_temperature=lora_temperature,
            lora_use_trainable_layer=lora_use_trainable_layer,
            lora_use_dynamic_topK=lora_use_dynamic_topK,
            lora_use_basis_variants=lora_use_basis_variants,
            lora_use_linear = lora_use_linear,
            lora_use_basis_variants_as_input=lora_use_basis_variants_as_input,
        )

        super(MultiLoRAPolicy, self).__init__(
            encoder=encoder,
            decoder=decoder,
            embed_dim=embed_dim,
            num_heads=num_heads,
            normalization=normalization,
            feedforward_hidden=feedforward_hidden,
            env_name=env_name,
            use_graph_context=use_graph_context,
            context_embedding=context_embedding,
            **kwargs,
        )
        self.lora_n_experts = lora_n_experts

        lora_fixed_params = self.lora_fixed_params()
        if lora_modules_ckpt_path != None:
            self.load_lora_modules(lora_fixed_params, lora_modules_ckpt_path)


    def lora_fixed_params(self, ) -> Dict[str, Any]:
        state_dict = {}
        for name, module in self.named_parameters():
            if 'lora_layers' in name.split('.'):
                lora_num_idx = name.split('.').index('lora_layers') + 1
                if int(name.split('.')[lora_num_idx]) < self.lora_n_experts:
                    state_dict[name] = module
                    module.requires_grad = False
        return state_dict


    def load_lora_modules(self, lora_fixed_params, lora_modules_ckpt_path):
        print('++++++++++Load LoRA Module Weights++++++++++')
        multi_lora_state_dict = collect_multi_lora_state_dict(
            multi_lora_fixed_params=lora_fixed_params, lora_n_experts=self.lora_n_experts
        )
        assert len(lora_modules_ckpt_path) == len(multi_lora_state_dict)

        for source_lora_ckpt_path, target_state_dict in zip(lora_modules_ckpt_path, multi_lora_state_dict.values()):
            assert os.path.exists(source_lora_ckpt_path)
            print(source_lora_ckpt_path)
            source_state_dict = collect_lora_state_dict(ckpt_path=source_lora_ckpt_path)
            load_target_lora_module(source_state_dict, target_state_dict)


    def lora_trainable_params(self, return_dict=False) -> Union[List, Any]:
        trainable_params = {}
        for name, module in self.named_parameters():
            if 'lora_layers' in name.split('.'):
                lora_num_idx = name.split('.').index('lora_layers') + 1
                if int(name.split('.')[lora_num_idx]) == self.lora_n_experts:
                    assert name.split('.')[lora_num_idx - 2] == 'lora_layer'
                    assert module.requires_grad == True
                    trainable_params[name] = module
            elif 'gate_layers' in name.split('.'):
                assert name.split('.')[name.split('.').index('gate_layers') - 1] == 'lora_layer'
                assert module.requires_grad == True
                trainable_params[name] = module

        assert len(trainable_params) > 0
        if return_dict:
            return trainable_params
        else:
            return list(trainable_params.values())

    def collect_GatedMultiLoRALayer(self, ):
        self.GatedMultiLoRALayer = []
        for module in self.encoder.modules():
            if isinstance(module, GatedMultiLoRALayer):
                self.GatedMultiLoRALayer.append(module)
        for module in self.decoder.modules():
            if isinstance(module, GatedMultiLoRALayer):
                self.GatedMultiLoRALayer.append(module)

    def get_basis_variant_binary_vector(self, td):
        has_open, has_tw, has_limit, has_backhaul, backhaul_class = MTVRPEnv.check_variants(td)
        prompt = torch.stack((has_open, has_tw, has_limit, has_backhaul), dim=-1).to(torch.float32)
        return prompt

    def set_basis_variant_binary_vector(self, td):
        basis_variant_binary_vector = self.get_basis_variant_binary_vector(td)
        for module in self.GatedMultiLoRALayer:
            module.basis_variant_binary_vector = basis_variant_binary_vector
        return

    def forward(
        self,
        td: TensorDict,
        env = None,
        phase: str = "train",
        calc_reward: bool = True,
        return_actions: bool = True,
        return_entropy: bool = False,
        return_hidden: bool = False,
        return_init_embeds: bool = False,
        return_sum_log_likelihood: bool = True,
        actions=None,
        max_steps=1_000_000,
        **decoding_kwargs,
    ) -> dict:

        self.collect_GatedMultiLoRALayer()
        self.set_basis_variant_binary_vector(td)

        outdict = super(MultiLoRAPolicy, self).forward(
            td=td,
            env=env,
            phase=phase,
            calc_reward=calc_reward,
            return_actions=return_actions,
            return_entropy=return_entropy,
            return_hidden=return_hidden,
            return_init_embeds=return_init_embeds,
            return_sum_log_likelihood=return_sum_log_likelihood,
            actions=actions,
            max_steps=max_steps,
            **decoding_kwargs,
        )
        return outdict









class CadaPolicy(AttentionModelPolicy):
    def __init__(
        self,
        embed_dim: int = 128,
        num_encoder_layers: int = 6,
        num_heads: int = 8,
        normalization: str = "instance",
        feedforward_hidden: int = 512,
        parallel_gated_kwargs: dict = None,
        encoder_use_post_layers_norm: bool = False,
        encoder_use_prenorm: bool = False,
        env_name: str = "mtvrp",
        use_graph_context: bool = False,
        init_embedding: MTVRPInitEmbeddingRouteFinder = None,
        prompt_embedding: MTVRPPromptEmbedding = None,
        context_embedding: MTVRPContextEmbeddingRouteFinder = None,
        extra_encoder_kwargs: dict = {},
        linear_bias_decoder: bool = False,
        sdpa_fn: Callable = None,
        mask_inner: bool = True,
        out_bias_pointer_attn: bool = False,
        check_nan: bool = True,
        attn_sparse_ratio: float = 0.5,
        sparse_applied_to_score: float = True,
        **kwargs,
    ):
        encoder = CadaEncoder(
            init_embedding=init_embedding,
            prompt_embedding=prompt_embedding,
            num_heads=num_heads,
            embed_dim=embed_dim,
            num_layers=num_encoder_layers,
            feedforward_hidden=feedforward_hidden,
            normalization=normalization,
            use_prenorm=encoder_use_prenorm,
            use_post_layers_norm=encoder_use_post_layers_norm,
            parallel_gated_kwargs=parallel_gated_kwargs,
            attn_sparse_ratio=attn_sparse_ratio,
            sparse_applied_to_score=sparse_applied_to_score,
            **extra_encoder_kwargs,
        )

        if context_embedding is None:
            context_embedding = MTVRPContextEmbeddingRouteFinder(embed_dim=embed_dim)
        # mtvrp does not use dynamic embedding (i.e. only modifies the query, not key or value)
        dynamic_embedding = StaticEmbedding()

        decoder = AttentionModelDecoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            env_name=env_name,
            context_embedding=context_embedding,
            dynamic_embedding=dynamic_embedding,
            sdpa_fn=sdpa_fn,
            mask_inner=mask_inner,
            out_bias_pointer_attn=out_bias_pointer_attn,
            linear_bias=linear_bias_decoder,
            use_graph_context=use_graph_context,
            check_nan=check_nan,
        )

        super(CadaPolicy, self).__init__(
            encoder=encoder,
            decoder=decoder,
            embed_dim=embed_dim,
            num_heads=num_heads,
            normalization=normalization,
            feedforward_hidden=feedforward_hidden,
            env_name=env_name,
            use_graph_context=use_graph_context,
            context_embedding=context_embedding,
            dynamic_embedding=dynamic_embedding,
            **kwargs,
        )




class CadaLoRAPolicy(AttentionModelPolicy):
    def __init__(self,
                 embed_dim: int = 128,
                 num_encoder_layers: int = 6,
                 num_heads: int = 8,
                 normalization: str = "instance",
                 feedforward_hidden: int = 512,
                 parallel_gated_kwargs: dict = None,
                 encoder_use_post_layers_norm: bool = False,
                 encoder_use_prenorm: bool = False,
                 env_name: str = "mtvrp",
                 use_graph_context: bool = False,
                 init_embedding: MTVRPInitEmbeddingRouteFinder = None,
                 prompt_embedding: MTVRPPromptEmbedding = None,
                 context_embedding: MTVRPContextEmbeddingRouteFinder = None,
                 extra_encoder_kwargs: dict = {},
                 linear_bias_decoder: bool = False,
                 sdpa_fn: Callable = None,
                 mask_inner: bool = True,
                 out_bias_pointer_attn: bool = False,
                 check_nan: bool = True,
                 attn_sparse_ratio: float = 0.5,
                 sparse_applied_to_score: float = True,
                 lora_rank: int = None,
                 lora_alpha: float = None,
                 lora_use_gate: bool = True,
                 lora_act_func: str = 'sigmoid',
                 lora_use_linear: bool = False,
                 basis_policy_ckpt_path: str = None,
                 **kwargs,
                 ):

        basis_policy = CadaPolicy(
            embed_dim=embed_dim,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
            normalization=normalization,
            feedforward_hidden=feedforward_hidden,
            parallel_gated_kwargs=parallel_gated_kwargs,
            encoder_use_post_layers_norm=encoder_use_post_layers_norm,
            encoder_use_prenorm=encoder_use_prenorm,
            env_name=env_name,
            use_graph_context=use_graph_context,
            init_embedding=init_embedding,
            prompt_embedding=prompt_embedding,
            context_embedding=context_embedding,
            extra_encoder_kwargs=extra_encoder_kwargs,
            linear_bias_decoder=linear_bias_decoder,
            sdpa_fn=sdpa_fn,
            mask_inner=mask_inner,
            out_bias_pointer_attn=out_bias_pointer_attn,
            check_nan=check_nan,
            attn_sparse_ratio=attn_sparse_ratio,
            sparse_applied_to_score=sparse_applied_to_score,
            **kwargs,
        )

        if basis_policy_ckpt_path != None:
            assert os.path.exists(basis_policy_ckpt_path), "Path is {}".format(basis_policy_ckpt_path)
            LoRAPolicy.load_basis_policy_weights(basis_policy, basis_policy_ckpt_path)
        LoRAPolicy.fix_basis_policy_weights(basis_policy)

        encoder = CadaLoRAEncoder(
            encoder=basis_policy.encoder,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_use_gate=lora_use_gate,
            lora_act_func=lora_act_func,
            lora_use_linear=lora_use_linear,
        )
        decoder = LoRADecoder(
            decoder=basis_policy.decoder,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_use_gate=lora_use_gate,
            lora_act_func=lora_act_func,
            lora_use_linear=lora_use_linear,
        )

        super(CadaLoRAPolicy, self).__init__(
            encoder=encoder,
            decoder=decoder,
            embed_dim=embed_dim,
            num_heads=num_heads,
            normalization=normalization,
            feedforward_hidden=feedforward_hidden,
            env_name=env_name,
            use_graph_context=use_graph_context,
            context_embedding=context_embedding,
            **kwargs,
        )

    def lora_trainable_params(self):
        trainable_params = []
        for name, module in self.named_parameters():
            if "lora_layer" in name.split('.') or "gate_layer" in name.split('.'):
                assert module.requires_grad == True
                trainable_params.append(module)
        assert len(trainable_params) > 0
        return trainable_params









class CadaMultiLoRAPolicy(AttentionModelPolicy):
    def __init__(self,
                 embed_dim: int = 128,
                 num_encoder_layers: int = 6,
                 num_heads: int = 8,
                 normalization: str = "instance",
                 feedforward_hidden: int = 512,
                 parallel_gated_kwargs: dict = None,
                 encoder_use_post_layers_norm: bool = False,
                 encoder_use_prenorm: bool = False,
                 env_name: str = "mtvrp",
                 use_graph_context: bool = False,
                 init_embedding: MTVRPInitEmbeddingRouteFinder = None,
                 prompt_embedding: MTVRPPromptEmbedding = None,
                 context_embedding: MTVRPContextEmbeddingRouteFinder = None,
                 extra_encoder_kwargs: dict = {},
                 linear_bias_decoder: bool = False,
                 sdpa_fn: Callable = None,
                 mask_inner: bool = True,
                 out_bias_pointer_attn: bool = False,
                 check_nan: bool = True,
                 attn_sparse_ratio: float = 0.5,
                 sparse_applied_to_score: float = True,
                 lora_rank: int = None,
                 lora_alpha: float = None,
                 lora_act_func: str = 'softmax',
                 lora_n_experts: int = 4,
                 lora_top_k: int = 4,
                 lora_temperature: float = 1.0,
                 lora_use_trainable_layer: bool = False,
                 lora_use_dynamic_topK: bool = False,
                 lora_use_basis_variants: bool = False,
                 lora_use_linear: bool = False,
                 lora_use_basis_variants_as_input: bool = False,
                 basis_policy_ckpt_path: str = None,
                 lora_modules_ckpt_path: List = None,
                 **kwargs,
                 ):

        basis_policy = CadaPolicy(
            embed_dim=embed_dim,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
            normalization=normalization,
            feedforward_hidden=feedforward_hidden,
            parallel_gated_kwargs=parallel_gated_kwargs,
            encoder_use_post_layers_norm=encoder_use_post_layers_norm,
            encoder_use_prenorm=encoder_use_prenorm,
            env_name=env_name,
            use_graph_context=use_graph_context,
            init_embedding=init_embedding,
            prompt_embedding=prompt_embedding,
            context_embedding=context_embedding,
            extra_encoder_kwargs=extra_encoder_kwargs,
            linear_bias_decoder=linear_bias_decoder,
            sdpa_fn=sdpa_fn,
            mask_inner=mask_inner,
            out_bias_pointer_attn=out_bias_pointer_attn,
            check_nan=check_nan,
            attn_sparse_ratio=attn_sparse_ratio,
            sparse_applied_to_score=sparse_applied_to_score,
            **kwargs,
        )

        if basis_policy_ckpt_path != None:
            assert os.path.exists(basis_policy_ckpt_path)
            LoRAPolicy.load_basis_policy_weights(basis_policy, basis_policy_ckpt_path)
        LoRAPolicy.fix_basis_policy_weights(basis_policy)

        encoder = CadaMultiLoRAEncoder(
            encoder = basis_policy.encoder,
            lora_rank = lora_rank,
            lora_alpha = lora_alpha,
            lora_act_func = lora_act_func,
            lora_n_experts = lora_n_experts,
            lora_top_k = lora_top_k,
            lora_temperature = lora_temperature,
            lora_use_trainable_layer = lora_use_trainable_layer,
            lora_use_dynamic_topK = lora_use_dynamic_topK,
            lora_use_basis_variants = lora_use_basis_variants,
            lora_use_linear = lora_use_linear,
            lora_use_basis_variants_as_input=lora_use_basis_variants_as_input,
        )
        decoder = MultiLoRADecoder(
            decoder = basis_policy.decoder,
            lora_rank = lora_rank,
            lora_alpha = lora_alpha,
            lora_act_func=lora_act_func,
            lora_n_experts=lora_n_experts,
            lora_top_k=lora_top_k,
            lora_temperature=lora_temperature,
            lora_use_trainable_layer=lora_use_trainable_layer,
            lora_use_dynamic_topK=lora_use_dynamic_topK,
            lora_use_basis_variants=lora_use_basis_variants,
            lora_use_linear = lora_use_linear,
            lora_use_basis_variants_as_input=lora_use_basis_variants_as_input,
        )

        super(CadaMultiLoRAPolicy, self).__init__(
            encoder=encoder,
            decoder=decoder,
            embed_dim=embed_dim,
            num_heads=num_heads,
            normalization=normalization,
            feedforward_hidden=feedforward_hidden,
            env_name=env_name,
            use_graph_context=use_graph_context,
            context_embedding=context_embedding,
            **kwargs,
        )
        self.lora_n_experts = lora_n_experts

        lora_fixed_params = self.lora_fixed_params()
        if lora_modules_ckpt_path != None:
            self.load_lora_modules(lora_fixed_params, lora_modules_ckpt_path)
             
    def load_lora_modules(self, lora_fixed_params, lora_modules_ckpt_path):
        print('++++++++++Load LoRA Module Weights++++++++++')
        multi_lora_state_dict = collect_multi_lora_state_dict(
            multi_lora_fixed_params=lora_fixed_params, lora_n_experts=self.lora_n_experts
        )
        assert len(lora_modules_ckpt_path) == len(multi_lora_state_dict)

        for source_lora_ckpt_path, target_state_dict in zip(lora_modules_ckpt_path, multi_lora_state_dict.values()):
            assert os.path.exists(source_lora_ckpt_path)
            source_state_dict = collect_lora_state_dict(ckpt_path=source_lora_ckpt_path)
            load_target_lora_module(source_state_dict, target_state_dict)


    def lora_fixed_params(self, ) -> Dict[str, Any]:
        state_dict = {}
        for name, module in self.named_parameters():
            if 'lora_layers' in name.split('.'):
                lora_num_idx = name.split('.').index('lora_layers') + 1
                if int(name.split('.')[lora_num_idx]) < self.lora_n_experts:
                    state_dict[name] = module
                    module.requires_grad = False
        return state_dict


    def lora_trainable_params(self, return_dict=False) -> Union[List, Any]:
        trainable_params = {}
        for name, module in self.named_parameters():
            if 'lora_layers' in name.split('.'):
                lora_num_idx = name.split('.').index('lora_layers') + 1
                if int(name.split('.')[lora_num_idx]) == self.lora_n_experts:
                    assert name.split('.')[lora_num_idx - 2] == 'lora_layer'
                    assert module.requires_grad == True
                    trainable_params[name] = module
            elif 'gate_layers' in name.split('.'):
                assert name.split('.')[name.split('.').index('gate_layers') - 1] == 'lora_layer'
                assert module.requires_grad == True
                trainable_params[name] = module


        assert len(trainable_params) > 0
        if return_dict:
            return trainable_params
        else:
            return list(trainable_params.values())

    def collect_GatedMultiLoRALayer(self, ):
        self.GatedMultiLoRALayer = []
        for module in self.encoder.modules():
            if isinstance(module, GatedMultiLoRALayer):
                self.GatedMultiLoRALayer.append(module)
        for module in self.decoder.modules():
            if isinstance(module, GatedMultiLoRALayer):
                self.GatedMultiLoRALayer.append(module)

    def get_basis_variant_binary_vector(self, td):
        has_open, has_tw, has_limit, has_backhaul, backhaul_class = MTVRPEnv.check_variants(td)
        prompt = torch.stack((has_open, has_tw, has_limit, has_backhaul), dim=-1).to(torch.float32)
        return prompt

    def set_basis_variant_binary_vector(self, td):
        basis_variant_binary_vector = self.get_basis_variant_binary_vector(td)
        for module in self.GatedMultiLoRALayer:
            module.basis_variant_binary_vector = basis_variant_binary_vector
        return

    def forward(
        self,
        td: TensorDict,
        env = None,
        phase: str = "train",
        calc_reward: bool = True,
        return_actions: bool = True,
        return_entropy: bool = False,
        return_hidden: bool = False,
        return_init_embeds: bool = False,
        return_sum_log_likelihood: bool = True,
        actions=None,
        max_steps=1_000_000,
        **decoding_kwargs,
    ) -> dict:

        self.collect_GatedMultiLoRALayer()
        self.set_basis_variant_binary_vector(td)

        outdict = super(CadaMultiLoRAPolicy, self).forward(
            td=td,
            env=env,
            phase=phase,
            calc_reward=calc_reward,
            return_actions=return_actions,
            return_entropy=return_entropy,
            return_hidden=return_hidden,
            return_init_embeds=return_init_embeds,
            return_sum_log_likelihood=return_sum_log_likelihood,
            actions=actions,
            max_steps=max_steps,
            **decoding_kwargs,
        )
        return outdict








if __name__ == "__main__":

    pass




