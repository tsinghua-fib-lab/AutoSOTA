"""Manifold-Adversarial Adapters for robust LLaVA models."""

__all__ = [
    "MAAAdapter",
    "inject_maa_adapters",
    "prepare_maa_model",
    "load_maa_adapter_state",
    "save_maa_adapter_state",
]


def __getattr__(name):
    if name == "MAAAdapter":
        from .adapters import MAAAdapter

        return MAAAdapter
    if name in {"load_maa_adapter_state", "save_maa_adapter_state"}:
        from .checkpoint import load_maa_adapter_state, save_maa_adapter_state

        return {
            "load_maa_adapter_state": load_maa_adapter_state,
            "save_maa_adapter_state": save_maa_adapter_state,
        }[name]
    if name in {"inject_maa_adapters", "prepare_maa_model"}:
        from .modeling import inject_maa_adapters, prepare_maa_model

        return {
            "inject_maa_adapters": inject_maa_adapters,
            "prepare_maa_model": prepare_maa_model,
        }[name]
    raise AttributeError(f"module 'maa' has no attribute {name!r}")
