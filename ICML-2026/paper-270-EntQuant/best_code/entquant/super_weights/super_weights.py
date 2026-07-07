"""
Super weight detection algorithm.

Based on: "The Super Weight in Large Language Models" (https://arxiv.org/abs/2411.07191)

Super weights are identified by finding weights that cause activation spikes
in both the input and output of the down_proj (or equivalent) module.

Algorithm:
1. Run a forward pass and capture INPUT activations of down_proj -> find spike columns
2. Run a forward pass and capture OUTPUT activations of down_proj -> find spike rows
3. Super weight coordinate = (layer, output_spike_channel, input_spike_channel)
"""

import logging
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from ..utils import clear_cache, DeviceMap, get_matching_module_names

logger = logging.getLogger(__name__)


@dataclass
class SuperWeightsConfig:
    """Configuration for super weight detection.

    Super weights are individual weight parameters that, when removed, cause
    catastrophic degradation in model performance. They are identified by
    finding weights that produce abnormally large activations in both
    the input and output of the down_proj layer.

    Attributes:
        tokenizer: Tokenizer for the model.
        include: Glob pattern(s) for module names to include (e.g., "*mlp.down_proj").
                 If None, all modules are included by default.
        exclude: Glob pattern(s) for module names to exclude.
                 If None, no modules are excluded.
        test_text: Input text for the forward pass.
        spike_threshold: Activation magnitude threshold to consider as a spike.
        top_k: Number of top activations to consider per module.
        keep_activations: Whether to keep activation values.
        layer_types: Types of modules to consider (e.g., nn.Linear).
    """

    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | None = None
    include: str | list[str] | None = None
    exclude: str | list[str] | None = None
    layer_types: tuple[type[nn.Module]] | list[type[nn.Module]] = (nn.Linear,)
    spike_threshold: float = 50.0
    top_k: int = 1
    # Prompt taken from super weights paper (https://arxiv.org/abs/2411.07191)
    test_text: str = "Apple Inc. is a worldwide tech company."
    keep_activations: bool = False


@dataclass
class SuperWeight:
    """Coordinates and activation values for a single super weight."""

    row: int  # output dimension of layer (from output activation spike)
    col: int  # input dimension of layer (from input activation spike)
    input_activation: float
    output_activation: float
    input_activation_tensor: torch.Tensor | None = None  # full input activation (shared per layer)
    output_activation_tensor: torch.Tensor | None = None  # full output activation (shared per layer)


def _get_top_k_activations(
    model: PreTrainedModel,
    config: SuperWeightsConfig,
    input_or_output: bool,
) -> dict[str, tuple[list[tuple[float, tuple]], torch.Tensor | None]]:
    """
    Run forward pass and get top-k activation values and indices per matching module.

    Args:
        model: The model to analyze.
        config: Configuration for super weight detection.
        input_or_output: If True, capture input activations; if False,
            capture output activations.

    Returns:
        Dict mapping module_name -> (list of (value, index_tuple), full_activation_tensor_or_None)
    """
    activations: dict[str, torch.Tensor | tuple] = {}

    def create_hook(module_name: str):
        def hook(module, inputs, outputs):
            activations[module_name] = inputs if input_or_output else outputs

        return hook

    matching_names = get_matching_module_names(model, config.include, config.exclude, config.layer_types)
    hooks = [model.get_submodule(name).register_forward_hook(create_hook(name)) for name in matching_names]

    tokenizer = config.tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)
        logger.warning("Tokenizer not provided, using AutoTokenizer from model name_or_path.")

    device = next(model.parameters()).device
    inputs = tokenizer(config.test_text, return_tensors="pt").to(device)

    if model.training:
        model.eval()

    with torch.no_grad():
        model(**inputs)

    for hook in hooks:
        hook.remove()

    results = {}
    for module_name, hidden_states in activations.items():
        hidden_state = hidden_states[0] if isinstance(hidden_states, tuple) else hidden_states
        hidden_state = hidden_state.detach().cpu().float()
        l2_norm = torch.linalg.vector_norm(hidden_state)
        logger.debug(
            f"{module_name} {'input' if input_or_output else 'output'} activation L2 norm: {l2_norm.item():.6g}"
        )
        hidden_state_flat = hidden_state.view(-1).abs()

        # Get top-k values and their flat indices
        top_k = min(config.top_k, hidden_state_flat.numel())
        top_k_values, top_k_flat_indices = torch.topk(hidden_state_flat, top_k, dim=0)

        top_k_results = []
        for i in range(top_k):
            idx = torch.unravel_index(top_k_flat_indices[i], hidden_state.shape)
            idx_tuple = tuple(dim_idx.item() for dim_idx in idx)
            top_k_results.append((top_k_values[i].item(), idx_tuple))

        full_activation = hidden_state if config.keep_activations else None
        results[module_name] = (top_k_results, full_activation)

    return results


def find_super_weights(
    model: PreTrainedModel,
    config: SuperWeightsConfig,
) -> dict[str, list[SuperWeight]]:
    """
    Identify super weights in a model by detecting activation spikes.

    Args:
        model: The language model to analyze.
        config: Configuration for super weight detection.

    Returns:
        Dict mapping module_name -> list of SuperWeight objects.
        - row: output dimension of down_proj (from output activation spike)
        - col: input dimension of down_proj (from input activation spike)
        Empty list if no super weights found for that module.
        If config.keep_activations is True, each SuperWeight will also contain
        the full input/output activation tensors (shared across all coords in the same layer).
    """
    # Get input activation spikes (determines column = input channel)
    input_activations = _get_top_k_activations(model, config, input_or_output=True)

    # Get output activation spikes (determines row = output channel)
    output_activations = _get_top_k_activations(model, config, input_or_output=False)

    results: dict[str, list[SuperWeight]] = {module_name: [] for module_name in input_activations}

    for module_name in input_activations:
        input_top_k, input_tensor = input_activations[module_name]
        output_top_k, output_tensor = output_activations.get(module_name, ([], None))

        # Create super weight candidates from all combinations of top-k input and output spikes
        # that both exceed the threshold (deduplicated by (row, col))
        seen_coords: set[tuple[int, int]] = set()
        for input_val, input_idx in input_top_k:
            if abs(input_val) <= config.spike_threshold:
                continue
            for output_val, output_idx in output_top_k:
                if abs(output_val) <= config.spike_threshold:
                    continue
                col = input_idx[-1]  # input dimension
                row = output_idx[-1]  # output dimension
                if (row, col) in seen_coords:
                    continue
                seen_coords.add((row, col))
                results[module_name].append(
                    SuperWeight(
                        row=row,
                        col=col,
                        input_activation=input_val,
                        output_activation=output_val,
                        input_activation_tensor=input_tensor,
                        output_activation_tensor=output_tensor,
                    )
                )

    return results


def detect_super_weights(
    model_id: str,
    config: SuperWeightsConfig | None = None,
    device_map: DeviceMap = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    model_cls: type | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> dict[str, list[SuperWeight]]:
    """Detect super weights in a model.

    Loads the base model, runs a forward pass for activation spike
    detection, and frees the model.

    Args:
        model_id: HuggingFace model ID or local path.
        config: Super weight detection config.
        device_map: Device map for model loading.
        dtype: Model dtype for loading weights.
        model_cls: Model class override (default: AutoModelForCausalLM).
        model_kwargs: Extra kwargs for from_pretrained
            (e.g., trust_remote_code).

    Returns:
        Dict mapping module names to lists of SuperWeight.
    """
    if config is None:
        config = SuperWeightsConfig()

    cls = model_cls or AutoModelForCausalLM
    model_kwargs = model_kwargs or {}

    logger.info(f"Detecting super weights for {model_id} (device_map={device_map})")
    model = cls.from_pretrained(
        model_id,
        device_map=device_map,
        dtype=dtype,
        **model_kwargs,
    )
    model.eval()

    if config.tokenizer is None:
        config.tokenizer = AutoTokenizer.from_pretrained(model_id, **model_kwargs)

    result = find_super_weights(model, config)

    total = sum(len(v) for v in result.values())
    modules_with_sw = [name for name, v in result.items() if v]
    module_list = "\n  ".join(modules_with_sw)
    logger.info(f"Detected {total} super weights across {len(modules_with_sw)} modules:\n  {module_list}")

    del model
    clear_cache()

    return result


def detect_fallback_layers(
    model_id: str,
    config: SuperWeightsConfig | None = None,
    device_map: DeviceMap = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    model_cls: type | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> set[str]:
    """Detect super weight layers and return their names as a set.

    Convenience wrapper around :func:`detect_super_weights` that
    returns only the layer names (suitable for the ``fallback_layers``
    parameter of ``EntQuantModel``).

    Returns:
        Set of module names containing at least one super weight.
    """
    sw = detect_super_weights(
        model_id,
        config=config,
        device_map=device_map,
        dtype=dtype,
        model_cls=model_cls,
        model_kwargs=model_kwargs,
    )
    return {name for name, sws in sw.items() if sws}
