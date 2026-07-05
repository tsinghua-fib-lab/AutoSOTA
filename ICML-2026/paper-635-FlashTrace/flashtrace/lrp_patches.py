"""LRP forward patches for AttnLRP implementation.

This module provides patched forward functions for transformer layers
that implement AttnLRP rules during the backward pass:

- RMSNorm: Identity rule via stopping gradient on variance
- Gated MLP: Identity rule on activation + Uniform rule on gate*up
- Attention: Uniform rule on Q@K^T and attention@V

Reference:
    AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers (ICML 2024)
    https://arxiv.org/abs/2402.05602
"""

import torch
from contextlib import contextmanager
from typing import Dict, Any, Optional, Callable
from functools import partial

from .lrp_rules import stop_gradient, divide_gradient, identity_rule_implicit


# Model configuration for different architectures
MODEL_CONFIGS = {
    "qwen3": {
        "modeling_module": "transformers.models.qwen3.modeling_qwen3",
        "rms_norm_class": "Qwen3RMSNorm",
        "mlp_class": "Qwen3MLP",
    },
    "qwen2": {
        "modeling_module": "transformers.models.qwen2.modeling_qwen2",
        "rms_norm_class": "Qwen2RMSNorm",
        "mlp_class": "Qwen2MLP",
    },
    "llama": {
        "modeling_module": "transformers.models.llama.modeling_llama",
        "rms_norm_class": "LlamaRMSNorm",
        "mlp_class": "LlamaMLP",
    },
}


def rms_norm_forward_lrp(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """RMSNorm forward with LRP identity rule.

    On normalization operations, we apply the identity rule by stopping
    the gradient flow through the variance calculation.

    This is equivalent to Equation 9 in the AttnLRP paper.
    """
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    # Stop gradient through the normalization factor
    hidden_states = hidden_states * stop_gradient(torch.rsqrt(variance + self.variance_epsilon))
    return self.weight * hidden_states.to(input_dtype)


def gated_mlp_forward_lrp(self, x: torch.Tensor) -> torch.Tensor:
    """Gated MLP forward with LRP rules.

    On the element-wise non-linear activation, we apply the identity rule.
    On the element-wise multiplication (gate * up), we apply the uniform rule.

    Both rules are implemented via the Gradient*Input framework.
    """
    gate_out = self.gate_proj(x)
    # Apply identity rule to the non-linear activation
    gate_out = identity_rule_implicit(self.act_fn, gate_out)

    # Element-wise multiplication of gate and up projections
    weighted = gate_out * self.up_proj(x)
    # Apply uniform rule (divide gradient by 2)
    weighted = divide_gradient(weighted, 2)

    return self.down_proj(weighted)


def wrap_attention_forward(forward_fn: Callable) -> Callable:
    """Wrap an attention forward function with LRP gradient rules.

    Applies the uniform rule to Q, K, V tensors:
    - Q and K: divide by 4 (accounts for Q@K^T matmul and softmax)
    - V: divide by 2 (accounts for attention@V matmul)

    Parameters
    ----------
    forward_fn : Callable
        The original attention forward function

    Returns
    -------
    Callable
        Wrapped attention forward function with LRP rules
    """
    def attention_forward_lrp(module, query, key, value, *args, **kwargs):
        # Apply uniform rule to Q, K, V
        # Factor of 4 for Q/K accounts for two matmuls (Q@K^T, then softmax/attention)
        # Factor of 2 for V accounts for single matmul with attention weights
        query = divide_gradient(query, 4)
        key = divide_gradient(key, 4)
        value = divide_gradient(value, 2)

        # Disable dropout during LRP computation
        if 'dropout' in kwargs:
            kwargs['dropout'] = 0.0

        return forward_fn(module, query, key, value, *args, **kwargs)

    return attention_forward_lrp


def dropout_forward_lrp(self, x: torch.Tensor) -> torch.Tensor:
    """Dropout forward that acts as identity.

    During LRP computation, we need gradient checkpointing which requires
    train mode, but we don't want actual dropout. This patches dropout
    to be identity.
    """
    return x


class LRPPatchState:
    """Stores original module states for restoration after LRP computation."""

    def __init__(self):
        self.original_forwards: Dict[str, Any] = {}
        self.original_attention_functions: Dict[str, Dict[str, Callable]] = {}
        self.original_eager_attention: Dict[str, Callable] = {}
        self.patched = False


def _get_modeling_module(model_type: str):
    """Dynamically import the modeling module for a given model type."""
    import importlib
    config = MODEL_CONFIGS.get(model_type)
    if config is None:
        raise ValueError(f"Unsupported model type: {model_type}. Supported: {list(MODEL_CONFIGS.keys())}")
    return importlib.import_module(config["modeling_module"])


def apply_lrp_patches(model, model_type: str = "qwen3") -> LRPPatchState:
    """Apply LRP patches to a model.

    This function patches the forward methods of various layers to implement
    AttnLRP rules during backward propagation.

    Parameters
    ----------
    model : transformers model
        The model to patch
    model_type : str
        Type of model architecture (qwen3, qwen2, llama)

    Returns
    -------
    LRPPatchState
        State object containing original methods for restoration
    """
    state = LRPPatchState()
    config = MODEL_CONFIGS.get(model_type)

    if config is None:
        raise ValueError(f"Unsupported model type: {model_type}")

    modeling_module = _get_modeling_module(model_type)

    # Patch RMSNorm layers
    rms_norm_cls = getattr(modeling_module, config["rms_norm_class"])
    state.original_forwards[config["rms_norm_class"]] = rms_norm_cls.forward
    rms_norm_cls.forward = rms_norm_forward_lrp

    # Patch MLP layers
    mlp_cls = getattr(modeling_module, config["mlp_class"])
    state.original_forwards[config["mlp_class"]] = mlp_cls.forward
    mlp_cls.forward = gated_mlp_forward_lrp

    # Patch Dropout layers
    from torch.nn import Dropout
    state.original_forwards["Dropout"] = Dropout.forward
    Dropout.forward = dropout_forward_lrp

    # Patch attention functions
    if hasattr(modeling_module, 'ALL_ATTENTION_FUNCTIONS'):
        state.original_attention_functions[model_type] = dict(modeling_module.ALL_ATTENTION_FUNCTIONS)
        new_attention_functions = {}
        for key, fn in modeling_module.ALL_ATTENTION_FUNCTIONS.items():
            new_attention_functions[key] = wrap_attention_forward(fn)
        modeling_module.ALL_ATTENTION_FUNCTIONS = new_attention_functions

    if hasattr(modeling_module, 'eager_attention_forward'):
        state.original_eager_attention[model_type] = modeling_module.eager_attention_forward
        modeling_module.eager_attention_forward = wrap_attention_forward(
            modeling_module.eager_attention_forward
        )

    state.patched = True
    return state


def remove_lrp_patches(state: LRPPatchState, model_type: str = "qwen3"):
    """Remove LRP patches and restore original forward methods.

    Parameters
    ----------
    state : LRPPatchState
        State object from apply_lrp_patches
    model_type : str
        Type of model architecture
    """
    if not state.patched:
        return

    config = MODEL_CONFIGS.get(model_type)
    if config is None:
        return

    modeling_module = _get_modeling_module(model_type)

    # Restore RMSNorm
    if config["rms_norm_class"] in state.original_forwards:
        rms_norm_cls = getattr(modeling_module, config["rms_norm_class"])
        rms_norm_cls.forward = state.original_forwards[config["rms_norm_class"]]

    # Restore MLP
    if config["mlp_class"] in state.original_forwards:
        mlp_cls = getattr(modeling_module, config["mlp_class"])
        mlp_cls.forward = state.original_forwards[config["mlp_class"]]

    # Restore Dropout
    if "Dropout" in state.original_forwards:
        from torch.nn import Dropout
        Dropout.forward = state.original_forwards["Dropout"]

    # Restore attention functions
    if model_type in state.original_attention_functions:
        modeling_module.ALL_ATTENTION_FUNCTIONS = state.original_attention_functions[model_type]

    if model_type in state.original_eager_attention:
        modeling_module.eager_attention_forward = state.original_eager_attention[model_type]

    state.patched = False


@contextmanager
def lrp_context(model, model_type: str = "qwen3"):
    """Context manager for applying LRP patches temporarily.

    This is the recommended way to use LRP patches as it ensures
    proper cleanup even if an exception occurs.

    Example
    -------
    >>> with lrp_context(model, model_type="qwen3"):
    ...     # Compute forward pass and backward for LRP
    ...     output = model(inputs_embeds=embeds)
    ...     output.logits[0, -1, :].max().backward()
    ...     relevance = (embeds * embeds.grad).sum(-1)

    Parameters
    ----------
    model : transformers model
        The model to patch
    model_type : str
        Type of model architecture (qwen3, qwen2, llama)

    Yields
    ------
    LRPPatchState
        The patch state (usually not needed by caller)
    """
    state = apply_lrp_patches(model, model_type)
    try:
        yield state
    finally:
        remove_lrp_patches(state, model_type)


def detect_model_type(model) -> str:
    """Attempt to detect the model type from a model instance.

    Parameters
    ----------
    model : transformers model
        The model to detect type for

    Returns
    -------
    str
        Detected model type (qwen3, qwen2, llama)

    Raises
    ------
    ValueError
        If model type cannot be detected
    """
    model_class_name = model.__class__.__name__.lower()

    if 'qwen3' in model_class_name:
        return 'qwen3'
    elif 'qwen2' in model_class_name:
        return 'qwen2'
    elif 'llama' in model_class_name:
        return 'llama'

    # Check config if available
    if hasattr(model, 'config'):
        config_name = getattr(model.config, 'model_type', '').lower()
        if 'qwen3' in config_name:
            return 'qwen3'
        elif 'qwen2' in config_name:
            return 'qwen2'
        elif 'llama' in config_name:
            return 'llama'

    raise ValueError(
        f"Could not detect model type from {model.__class__.__name__}. "
        f"Please specify model_type explicitly. Supported: {list(MODEL_CONFIGS.keys())}"
    )
