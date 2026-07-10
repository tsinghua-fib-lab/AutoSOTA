"""Model registry for locality diffusion baselines."""

from __future__ import annotations

from typing import Any, Callable, Dict


MODEL_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_model(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(cls_or_factory: Callable[..., Any]) -> Callable[..., Any]:
        MODEL_REGISTRY[name.lower()] = cls_or_factory
        return cls_or_factory

    return decorator

#   cfg.model.name, dataset=dataset, device=cfg.experiment.device, num_steps=cfg.sampling.num_inference_steps, params=model_params,
def create_model(name: str, **kwargs: Any) -> Any:
    factory = MODEL_REGISTRY.get(name.lower())
    if factory is None:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")
    return factory(**kwargs)


# Import analytical methods (optional ones are skipped if absent).
for _name in (
    "nearest_dataset", "optimal", "wiener",
    "pca_locality", "pca_locality_channel_wise",
    "gaussian", "ours", "kamb", "faiss_static",
):
    try:
        __import__(f"{__name__}.{_name}", fromlist=[_name])
    except ImportError:
        pass

# ------------------------
# baseline registry
# ------------------------
BASELINE_REGISTRY: Dict[str, Callable[..., Any]] = {}

def register_baseline_model(name: str):
    def decorator(cls_or_factory: Callable[..., Any]):
        BASELINE_REGISTRY[name.lower()] = cls_or_factory
        return cls_or_factory
    return decorator

def create_baseline_model(name: str, **kwargs: Any) -> Any:
    factory = BASELINE_REGISTRY.get(name.lower())
    if factory is None:
        raise ValueError(f"Unknown baseline '{name}'. Available: {sorted(BASELINE_REGISTRY)}")
    return factory(**kwargs)

# import to register baseline models (triggers @register_baseline)
from . import baseline_unet  # noqa: E402,F401
from . import baseline_edm_unconditional  # noqa: E402,F401
## only for imagenet-1k
from . import baseline_edm_conditional  # noqa: E402,F401
