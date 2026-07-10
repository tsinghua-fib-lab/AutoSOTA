from typing import Dict, Optional, Tuple

import torch

from src.flow_matching.rate_matrix import RateMatrixDesigner, vf_denoiser, rvf_denoiser


class RateComputerStrategy:
    def compute(
        self,
        kappa_t: torch.Tensor,
        kappa_dot_t: torch.Tensor,
        node_mask: torch.Tensor,
        G_t: Tuple[torch.Tensor, torch.Tensor],
        G_1_pred: Tuple[torch.Tensor, torch.Tensor],
        z_0=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class DefaultRateComputer(RateComputerStrategy):
    def __init__(self, designer: RateMatrixDesigner):
        self.designer = designer

    def compute(self, kappa_t, kappa_dot_t, node_mask, G_t, G_1_pred, z_0=None):
        return self.designer.compute_graph_rate_matrix(kappa_t, node_mask, G_t, G_1_pred)


class VFDenoiserRateComputer(RateComputerStrategy):
    def __init__(self):
        self.impl = vf_denoiser()

    def compute(self, kappa_t, kappa_dot_t, node_mask, G_t, G_1_pred, z_0=None):
        return self.impl.compute_graph_rate_matrix(kappa_t, kappa_dot_t, node_mask, G_t, G_1_pred)


class RVFDenoiserRateComputer(RateComputerStrategy):
    def __init__(self):
        self.impl = rvf_denoiser()

    def compute(self, kappa_t, kappa_dot_t, node_mask, G_t, G_1_pred, z_0=None):
        return self.impl.compute_graph_rate_matrix(kappa_t, kappa_dot_t, node_mask, G_t, G_1_pred)


class RateComputerRegistry:
    def __init__(self):
        self._strategies: Dict[str, RateComputerStrategy] = {}

    def register(self, name: str, strategy: RateComputerStrategy) -> None:
        self._strategies[name] = strategy

    def get(self, name: str) -> RateComputerStrategy:
        if name not in self._strategies:
            raise KeyError(f"RateComputer '{name}' not found. Registered: {list(self._strategies.keys())}")
        return self._strategies[name]


def build_rate_registry(cfg, limit_dist) -> RateComputerRegistry:
    """Helper to pre-register strategies based on cfg and limit_dist."""
    registry = RateComputerRegistry()

    designer = RateMatrixDesigner(
        rdb=cfg.sample.get('rdb', 'general'),
        rdb_crit=cfg.sample.get('rdb_crit', 'dummy'),
        eta=cfg.sample.get('eta', 0.0),
        omega=cfg.sample.get('omega', 0.0),
        limit_dist=limit_dist,
    )
    registry.register("defog", DefaultRateComputer(designer))
    registry.register("vf_denoiser", VFDenoiserRateComputer())
    registry.register("rvf_denoiser", RVFDenoiserRateComputer())
    return registry


