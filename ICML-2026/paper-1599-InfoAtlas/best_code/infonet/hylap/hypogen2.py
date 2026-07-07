"""
HyLaP policy wrapper around HyPoGen2 HyperTr.

This module provides a thin adapter that maps language features (meta vector)
and observation features (base vector) to actions using an in-tree copy of
HyPoGen2 hypernetwork implementation (vendored minimal subset).

Design goals:
- Keep it minimal; do not duplicate HypoGen2 code
- Allow configuration from YAML via Hydra-style "_target_"
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

class HyLaPHypoGenPolicy(nn.Module):
    """Minimal wrapper over HyPoGen2 HyperTr (single-head).

    This wrapper performs a simple linear projection from language features
    (meta vector) to the latent task embedding z, then uses HyPoGen2's
    HyperTr to generate a task-specific policy MLP producing actions
    from the base vector (vision/state features).
    """

    def __init__(
        self,
        # Dimensions
        meta_dim: int,  # language feature dimension
        base_input_dim: int,  # concatenated vision (+ optional state) feature dim
        action_dim: int,  # output action dimension
        # Model sizes
        hidden_dim: int = 256,  # target MLP hidden size
        z_dim: int = 256,  # task embedding dimension (ftask_dim)
        horizon: int = 1,  # number of future actions to predict
        # HyPoGen2 hypernetwork internal sizes
        weight_dim: int = 128,
        enc_dec_dim: int = 256,
        opt_block_dim: int = 256,
        num_opt_mlp_layer: int = 2,
        num_enc_dec_layer: int = 2,
        num_layers: int = 3,  # number of iterative refinement layers in hypernet
        weight_split_dim: int = 512,
        nhead: int = 4,
        lr_scheme_method: str = "cosine",
        use_compile: bool = False,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__()

        # Use vendored minimal HyperTr implementation
        from .hypogen2_impl.hypernetworks import HyperTr

        self.meta_dim = meta_dim
        self.base_input_dim = base_input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.horizon = horizon

        # PerceiverIO-style attention for language feature processing
        self.language_processor = nn.Linear(meta_dim, z_dim)

        # Build the HyPoGen2 HyperTr (single MLP target network)
        self.hyper_tr = HyperTr(
            ftask_dim=z_dim,
            dim2=base_input_dim,  # input to target MLP
            dim3=hidden_dim,  # hidden size of target MLP
            dim4=action_dim * horizon,  # output flattened horizon actions
            num_layers=num_layers,
            weight_dim=weight_dim,
            enc_dec_dim=enc_dec_dim,
            opt_block_dim=opt_block_dim,
            num_opt_mlp_layer=num_opt_mlp_layer,
            num_enc_dec_layer=num_enc_dec_layer,
            weight_split_dim=weight_split_dim,
            nhead=nhead,
            lr_scheme_method=lr_scheme_method,
            use_compile=use_compile,
            **kwargs,
        )

        self.print_module_sizes()

    @torch.no_grad()
    def infer(
        self, meta_v: torch.Tensor, base_v: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Inference helper with no grad.

        Args:
            meta_v: Language feature tensor of shape (B, L, meta_dim)
            base_v: Base feature tensor (vision [+ state]) of shape (B, base_input_dim)
            attention_mask: Optional attention mask of shape (B, L) to mask padding tokens

        Returns:
            actions: Tensor of shape (B, horizon, action_dim)
        """
        self.eval()

        # Ensure meta_v is a sequence (B, L, meta_dim)
        if meta_v.dim() == 2:
            meta_v = meta_v.unsqueeze(1)  # (B, 1, meta_dim)

        z_sequence = self.language_processor(meta_v)  # (B, num_queries, z_dim)

        # Use the sequence of task embeddings directly with HyperTr
        actions = self.hyper_tr(z_sequence, base_v)
        actions = actions.view(actions.shape[0], self.horizon, self.action_dim)
        return actions

    def generate_weights(self, task_embedding: torch.Tensor) -> torch.Tensor:
        """Generate weights for the language processor.
        """
        z_sequence = self.language_processor(task_embedding)
        weight_dict = self.hyper_tr.generate_weights(z_sequence)
        return weight_dict

    def forward(
        self, meta_v: torch.Tensor, base_v: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
        early_sup: bool = False
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            meta_v: Language feature tensor of shape (B, L, meta_dim)
            base_v: Base feature tensor (vision [+ state]) of shape (B, base_input_dim)
            attention_mask: Optional attention mask of shape (B, L) to mask padding tokens

        Returns:
            actions: Tensor of shape (B, horizon, action_dim)
        """
        # Ensure meta_v is a sequence (B, L, meta_dim)
        if meta_v.dim() == 2:
            meta_v = meta_v.unsqueeze(1)  # (B, 1, meta_dim)

        z_sequence = self.language_processor(meta_v)

        # Use the sequence of task embeddings directly with HyperTr
        actions = self.hyper_tr(z_sequence, base_v, early_sup)
        if early_sup:
            actions = actions.view(-1, base_v.shape[0], self.horizon, self.action_dim)
        else:
            actions = actions.view(actions.shape[0], self.horizon, self.action_dim)
        return actions

    def analyze_module_sizes(self) -> list:
        """
        Analyze all modules in this HyperDiffAct model and return their sizes.

        Returns:
            List of tuples (module_name, parameter_count, size_mb)
        """
        from utils.module_size_printer import count_parameters, get_module_size_mb

        module_info = []

        # Main components
        components = {
            "language_processor": self.language_processor,
            "hyper_tr": self.hyper_tr,
        }

        # Analyze main components
        for name, module in components.items():
            param_count = count_parameters(module)
            size_mb = get_module_size_mb(module)
            module_info.append((name, param_count, size_mb))

        # Analyze HyperTr sub-components
        hyper_tr = self.hyper_tr
        hyper_tr_components = {
            "hyper_tr.target_net": hyper_tr.target_net,
            "hyper_tr.ftask_adapter": hyper_tr.ftask_adapter,
            "hyper_tr.encoders": hyper_tr.encoders,
            "hyper_tr.decoders": hyper_tr.decoders,
            "hyper_tr.opt_layer": hyper_tr.opt_layer if hyper_tr.replicate_blocks else hyper_tr.opt_layers,
            "hyper_tr.layer_norms": hyper_tr.layer_norms,
            "hyper_tr.init_weight_hypernet": hyper_tr.init_weight_hypernet,
        }

        for name, module in hyper_tr_components.items():
            param_count = count_parameters(module)
            size_mb = get_module_size_mb(module)
            module_info.append((name, param_count, size_mb))

        # Analyze individual encoders/decoders
        for i, (encoder, decoder) in enumerate(zip(hyper_tr.encoders, hyper_tr.decoders)):
            enc_params = count_parameters(encoder)
            enc_size = get_module_size_mb(encoder)
            module_info.append((f"hyper_tr.encoders[{i}]", enc_params, enc_size))

            dec_params = count_parameters(decoder)
            dec_size = get_module_size_mb(decoder)
            module_info.append((f"hyper_tr.decoders[{i}]", dec_params, dec_size))

        # Analyze lr_scheme buffer if it exists
        if hasattr(hyper_tr, "lr_scheme"):
            lr_scheme_params = hyper_tr.lr_scheme.numel()
            lr_scheme_size = lr_scheme_params * 4 / (1024 * 1024)  # float32 bytes to MB
            module_info.append(("hyper_tr.lr_scheme_buffer", lr_scheme_params, lr_scheme_size))

        # Analyze target network layers
        target_net = hyper_tr.target_net
        for i, submodule in enumerate(target_net.get_submodules()):
            params = count_parameters(submodule)
            size = get_module_size_mb(submodule)
            module_info.append((f"hyper_tr.target_net.fc{i}", params, size))

        # Analyze language processor components
        lang_proc = self.language_processor
        lang_components = {
            "language_processor": lang_proc,
        }

        for name, module in lang_components.items():
            if not isinstance(module, nn.Identity):  # Skip identity layers
                param_count = count_parameters(module)
                size_mb = get_module_size_mb(module)
                module_info.append((name, param_count, size_mb))

        # Analyze timestep embedding (freq_embed buffer)
        if hasattr(self, "freq_embed"):
            freq_embed_params = self.freq_embed.numel()
            freq_embed_size = freq_embed_params * 4 / (1024 * 1024)  # float32 bytes to MB
            module_info.append(("freq_embed_buffer", freq_embed_params, freq_embed_size))

        return module_info

    def print_module_sizes(self, sort_by: str = "size") -> None:
        """
        Print HyperDiffAct module sizes sorted from small to large.

        Args:
            sort_by: Either "size" (MB) or "params" (parameter count)
        """
        from utils.module_size_printer import print_model_summary

        module_info = self.analyze_module_sizes()
        print_model_summary(self, module_info, sort_by, "HyperDiffAct")
