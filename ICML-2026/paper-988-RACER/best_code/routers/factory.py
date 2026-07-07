from __future__ import annotations
from .knn_router import KnnModule, RouterKNNDataset
from .mlp_router import MLPModule
_ROUTER_REGISTRY = {
    KnnModule.model_name: {
        "model": KnnModule,
        # "dataset": RouterKNNDataset
    },
    MLPModule.model_name: {
        "model": MLPModule,
    },
    # ... Other Routers ...
}

def build_router(name: str, args, device: str):
    name = name.lower()
    if name not in _ROUTER_REGISTRY:
        raise ValueError(f"Unknown router name: {name}. Available: {list(_ROUTER_REGISTRY.keys())}")
    return _ROUTER_REGISTRY[name]["model"](args, device)
