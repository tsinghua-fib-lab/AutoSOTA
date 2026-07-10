"""Model steering and probe loading utilities for MONICA."""
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression


def load_probe(probe_data: dict) -> LogisticRegression:
    """Load a logistic regression probe from saved coefficients."""
    clf = LogisticRegression()
    clf.coef_ = probe_data["coef"]
    clf.intercept_ = probe_data["intercept"]
    clf.classes_ = probe_data["classes"]
    clf.n_features_in_ = probe_data["coef"].shape[1]
    clf._check_feature_names = lambda *args, **kwargs: None
    return clf


class SteerWrapper:
    """Wrapper around a HuggingFace model that supports activation steering."""

    def __init__(self, model, steer_layers: list[int]):
        self._model = model
        self._steer_layers = sorted(int(l) for l in steer_layers)
        self._hooks = []
        self._control_vectors = {}
        self._control_scale = 0.0
        self._active = False

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def device(self):
        if hasattr(self._model, "device"):
            return self._model.device
        return next(self._model.parameters()).device

    def set_control(self, calibrator_vec: dict, scale: float, normalize: bool = True, layer_weights: dict = None):
        """Set steering vectors and scale."""
        self._control_vectors = {}
        self._layer_weights = {}
        for layer_idx, vec in calibrator_vec.items():
            layer_idx = int(layer_idx)
            if isinstance(vec, np.ndarray):
                vec = torch.from_numpy(vec).float()
            elif isinstance(vec, torch.Tensor):
                vec = vec.float()
            else:
                vec = torch.tensor(vec, dtype=torch.float32)

            if normalize:
                n = vec.norm()
                if n > 0:
                    vec = vec / n

            self._control_vectors[layer_idx] = vec
            if layer_weights is not None:
                self._layer_weights[layer_idx] = float(layer_weights.get(layer_idx, 1.0))

        self._control_scale = float(scale)
        self._active = True
        self._register_hooks()

    def _find_layer(self, layer_idx: int):
        """Find the transformer layer at the given index."""
        # Try standard LLaMA/Qwen architecture
        if hasattr(self._model, "model") and hasattr(self._model.model, "layers"):
            layers = self._model.model.layers
            if layer_idx < len(layers):
                return layers[layer_idx]
        # Try GPT-2 style
        if hasattr(self._model, "transformer") and hasattr(self._model.transformer, "h"):
            layers = self._model.transformer.h
            if layer_idx < len(layers):
                return layers[layer_idx]
        return None

    def _register_hooks(self):
        """Register forward hooks on specified layers to add steering vectors."""
        self.reset()

        for layer_idx in self._steer_layers:
            if layer_idx not in self._control_vectors:
                continue

            layer = self._find_layer(layer_idx)
            if layer is None:
                continue

            vec = self._control_vectors[layer_idx].to(self.device)

            def make_hook(v, idx):
                def hook(module, input, output):
                    if not self._active:
                        return output
                    scale = self._control_scale

                    layer_weight = self._layer_weights.get(idx, 1.0)
                    if isinstance(output, tuple):
                        hs = output[0]
                        if isinstance(hs, torch.Tensor) and hs.shape[-1] == v.shape[-1]:
                            hs = hs + scale * layer_weight * v.to(device=hs.device, dtype=hs.dtype)
                        return (hs,) + output[1:]
                    elif isinstance(output, torch.Tensor):
                        if output.shape[-1] == v.shape[-1]:
                            output = output + scale * layer_weight * v.to(device=output.device, dtype=output.dtype)
                        return output
                    return output
                return hook

            h = layer.register_forward_hook(make_hook(vec, layer_idx))
            self._hooks.append(h)

    def reset(self):
        """Remove all hooks."""
        self._active = False
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._control_vectors = {}
        self._control_scale = 0.0

    def generate(self, *args, **kwargs):
        """Generate text using the underlying model with active steering."""
        self._active = True
        try:
            return self._model.generate(*args, **kwargs)
        finally:
            pass

    def __getattr__(self, name):
        # Only called when normal attribute lookup fails
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._model, name)

def steerLRM(model, steer_layers: list[int]):
    """Create a steerable wrapper around a language model.
    
    Args:
        model: HuggingFace CausalLM model
        steer_layers: list of layer indices to apply steering at
    
    Returns:
        SteerWrapper instance
    """
    return SteerWrapper(model, steer_layers)


def get_punctuation_token_ids(tokenizer) -> list[int]:
    """Get token IDs for common punctuation marks.
    
    Args:
        tokenizer: HuggingFace tokenizer
    
    Returns:
        list of token IDs for punctuation
    """
    punctuation = [".", ",", "!", "?", ";", ":", "\n", ".\n", "?\n", "!\n"]
    ids = set()
    for p in punctuation:
        token_ids = tokenizer.encode(p, add_special_tokens=False)
        ids.update(token_ids)
    # Also add common punctuation tokens from vocab
    for token, token_id in tokenizer.get_vocab().items():
        if token in (".", ",", "!", "?", ";", ":", "\n"):
            ids.add(token_id)
    return sorted(ids)
