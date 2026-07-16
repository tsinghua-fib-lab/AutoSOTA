from typing import Any, List
import torch
import torch.nn as nn

from .types import CovarianceDict


class KFACRecorder:
    def __init__(
        self,
        model: nn.Module,
        layer_types: tuple = (nn.Linear, nn.Conv2d),
        only_trainable: bool = True,
    ) -> None:
        self.activations: CovarianceDict = {}
        self.backprops: CovarianceDict = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._enabled: bool = False

        target = model._module if hasattr(model, "_module") else model

        for name, module in target.named_modules():
            if not isinstance(module, layer_types):
                continue
            if only_trainable:
                has_trainable = any(p.requires_grad for p in module.parameters())
                if not has_trainable:
                    continue

            self._handles.append(
                module.register_forward_hook(self._make_activation_hook(name))
            )
            self._handles.append(
                module.register_full_backward_hook(self._make_backprop_hook(name))
            )

    def _make_activation_hook(self, name: str):
        def hook(_module: nn.Module, inputs: tuple, _outputs: Any) -> None:
            if self._enabled:
                self.activations[name] = inputs[0].detach()
        return hook

    def _make_backprop_hook(self, name: str):
        def hook(_module: nn.Module, _grad_input: Any, grad_output: Any) -> None:
            if self._enabled:
                self.backprops[name] = grad_output[0].detach()
        return hook

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def clear(self) -> None:
        self.activations.clear()
        self.backprops.clear()

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self):
        self.enable()
        return self

    def __exit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> bool:
        self.disable()
        self.clear()
        return False
