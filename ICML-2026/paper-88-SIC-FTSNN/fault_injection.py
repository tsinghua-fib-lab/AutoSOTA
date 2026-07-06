from __future__ import annotations

import torch
import torch.nn as nn

from typing import List, Tuple, Optional, Dict

__all__ = ["FaultManager", "build_fault_manager", "get_fault_map"]

_SUPPORTED = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.LSTM, nn.GRU)

def _collect_param_paths(model: nn.Module, include_bias: bool = True
                         ) -> Tuple[List[str], List[torch.Size], List[str], List[torch.Size]]:
    """Collect dotted attribute paths and shapes for (weight[, bias]) tensors across supported modules."""
    w_names: List[str] = []
    w_shapes: List[torch.Size] = []
    b_names: List[str] = []
    b_shapes: List[torch.Size] = []
    for name, module in model.named_modules():
        if isinstance(module, _SUPPORTED):
            if getattr(module, "weight", None) is not None:
                w_names.append(f"{name}.weight" if name else "weight")
                w_shapes.append(module.weight.data.shape)
            if include_bias and getattr(module, "bias", None) is not None:
                b_names.append(f"{name}.bias" if name else "bias")
                b_shapes.append(module.bias.data.shape)
    return w_names, w_shapes, b_names, b_shapes


def _make_keep_mask_sporadic(shape: torch.Size, ratio: float, device: torch.device) -> torch.Tensor:
    """Element-wise Bernoulli selection: keep=1 (healthy), 0 (fault)."""
    if ratio <= 0:
        return torch.ones(shape, device=device)
    if ratio >= 1:
        return torch.zeros(shape, device=device)
    keep = (torch.rand(shape, device=device) > ratio).to(torch.float32)
    return keep


def _make_keep_mask_clustered(shape: torch.Size, ratio: float, device: torch.device) -> torch.Tensor:
    """Cluster by first dim (output row/channel). Choose ~ratio fraction of rows to fault."""
    if len(shape) == 0:
        return torch.ones((), device=device)
    rows = int(shape[0])
    rows = max(rows, 1)
    k = int(round(ratio * rows))
    k = max(0, min(rows, k))
    row_keep = torch.ones(rows, device=device, dtype=torch.float32)
    if k > 0:
        idx = torch.randperm(rows, device=device)[:k]
        row_keep[idx] = 0.0  # these rows are faulty
    view = [rows] + [1] * (len(shape) - 1)
    keep = row_keep.view(*view).expand(shape).contiguous()
    return keep


def _make_keep_mask(shape: torch.Size, ratio: float, distribution: str, device: torch.device) -> torch.Tensor:
    if distribution == "sporadic":
        return _make_keep_mask_sporadic(shape, ratio, device)
    if distribution == "clustered":
        return _make_keep_mask_clustered(shape, ratio, device)
    raise ValueError(f"Unknown distribution: {distribution}")


class FaultManager:
    """
    Fault applier for weights/biases.
    Parameters (via from_model/build_fault_manager):
        ratio         : fraction of connections affected (0..1)
        fault_type    : 'stuck' | 'random' | 'connectivity'
        distribution  : 'sporadic' | 'clustered'
        stuck_at      : constant value used for 'stuck'
        noise_std     : Gaussian std for 'random'
        limit         : Uniform bound for 'connectivity' (drawn in [-limit, +limit], once)
        include_bias  : whether to include bias tensors
    Behavior:
        - 'stuck'        : p <- p * keep + stuck_at * (1 - keep)
        - 'connectivity' : p <- p * keep + c * (1 - keep), where c ~ U(-limit, +limit) drawn once at setup
        - 'random'       : p <- p + N(0, noise_std) * (1 - keep), re-sampled EACH apply_ call
    """
    def __init__(
            self,
            weight_param_names: List[str],
            weight_keep_masks: List[torch.Tensor],
            weight_fixed_faults: List[Optional[torch.Tensor]],  # fixed for 'stuck'/'connectivity', None for 'random'
            param_types: List[str],
            bias_param_names: List[str],
            bias_keep_masks: List[torch.Tensor],
            bias_fixed_faults: List[Optional[torch.Tensor]],
            bias_param_types: List[str],
            fault_type: str,
            distribution: str,
            ratio: float,
            stuck_at: Optional[float],
            noise_std: Optional[float],
            limit: Optional[float],
            device: torch.device,
    ) -> None:
        self.weight_param_names = weight_param_names
        self.weight_keep_masks = weight_keep_masks
        self.weight_fixed_faults = weight_fixed_faults
        self.param_types = param_types

        self.bias_param_names = bias_param_names
        self.bias_keep_masks = bias_keep_masks
        self.bias_fixed_faults = bias_fixed_faults
        self.bias_param_types = bias_param_types

        self.fault_type = fault_type
        self.distribution = distribution
        self.ratio = ratio
        self.stuck_at = stuck_at
        self.noise_std = noise_std
        self.limit = limit
        self.device = device

    @staticmethod
    def _get_attr_by_path(root: nn.Module, dotted: str) -> torch.Tensor:
        obj = root
        parts = dotted.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        return getattr(obj, parts[-1])

    @classmethod
    def from_model(
            cls,
            model: nn.Module,
            *,
            ratio: float,
            fault_type: str = "stuck",
            distribution: str = "sporadic",
            stuck_at: Optional[float] = 0.0,
            noise_std: Optional[float] = 0.05,
            limit: Optional[float] = None,
            include_bias: bool = True,
            device: Optional[torch.device] = None,
    ) -> "FaultManager":
        device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        w_names, w_shapes, b_names, b_shapes = _collect_param_paths(model, include_bias=include_bias)

        weight_keep_masks: List[torch.Tensor] = []
        weight_fixed_faults: List[Optional[torch.Tensor]] = []
        param_types: List[str] = []
        prob = 175 / 1079  # Probability of injecting a fault 175 / 1079

        for shp in w_shapes:
            keep = _make_keep_mask(shp, ratio, distribution, device=device)
            weight_keep_masks.append(keep)

            if fault_type == "stuck":
                if stuck_at is None:
                    raise ValueError("stuck_at must be provided for 'stuck' faults")
                fixed = torch.zeros(shp, device=device)

                random_signs = torch.where(
                    torch.rand(shp, device=device) < prob, # < prob
                    -float(stuck_at),
                    float(stuck_at)
                )
                fixed[keep == 0] = random_signs[keep == 0]

                weight_fixed_faults.append(fixed)
                param_types.append("stuck")

            elif fault_type == "connectivity":
                if limit is None:
                    raise ValueError("limit must be provided for 'connectivity' faults")
                fixed = torch.zeros(shp, device=device)
                rnd = torch.where(
                    torch.rand(shp, device=device) < 0.5,                       # 끝점으로 갈지 말지 결정
                    torch.where(torch.rand(shp, device=device) < 1 - prob, 1.0, -1.0) * float(limit),  # ±limit
                    (torch.rand(shp, device=device) * 2.0 - 1.0) * float(limit)       # (-limit, limit) 균등
                )
                fixed[keep == 0] = rnd[keep == 0]
                weight_fixed_faults.append(fixed)
                param_types.append("connectivity")

            elif fault_type == "random":
                weight_fixed_faults.append(None)  # resample at apply_
                param_types.append("random")

            else:
                raise ValueError(f"Unsupported fault_type: {fault_type}")

        # Bias
        bias_keep_masks: List[torch.Tensor] = []
        bias_fixed_faults: List[Optional[torch.Tensor]] = []
        bias_param_types: List[str] = []
        if include_bias:
            for shp in b_shapes:
                keep = _make_keep_mask(shp, ratio, distribution, device=device)
                bias_keep_masks.append(keep)

                if fault_type == "stuck":
                    if stuck_at is None:
                        raise ValueError("stuck_at must be provided for 'stuck' faults")
                    fixed = torch.zeros(shp, device=device)
                    random_signs = torch.where(
                        torch.rand(shp, device=device) < prob,  # < prob
                        -float(stuck_at),
                        float(stuck_at)
                    )
                    fixed[keep == 0] = random_signs[keep == 0]
                    bias_fixed_faults.append(fixed)
                    bias_param_types.append("stuck")

                elif fault_type == "connectivity":
                    if limit is None:
                        raise ValueError("limit must be provided for 'connectivity' faults")
                    fixed = torch.zeros(shp, device=device)
                    rnd = torch.where(
                        torch.rand(shp, device=device) < 0.5,  # 끝점으로 갈지 말지 결정
                        torch.where(torch.rand(shp, device=device) < 1 - prob, 1.0, -1.0) * float(limit),  # ±limit
                        (torch.rand(shp, device=device) * 2.0 - 1.0) * float(limit)  # (-limit, limit) 균등
                    )
                    fixed[keep == 0] = rnd[keep == 0]
                    bias_fixed_faults.append(fixed)
                    bias_param_types.append("connectivity")

                elif fault_type == "random":
                    bias_fixed_faults.append(None)
                    bias_param_types.append("random")

                else:
                    raise ValueError(f"Unsupported fault_type: {fault_type}")

        else:
            b_names, bias_keep_masks, bias_fixed_faults, bias_param_types = [], [], [], []

        return cls(
            weight_param_names=w_names,
            weight_keep_masks=weight_keep_masks,
            weight_fixed_faults=weight_fixed_faults,
            param_types=param_types,
            bias_param_names=b_names,
            bias_keep_masks=bias_keep_masks,
            bias_fixed_faults=bias_fixed_faults,
            bias_param_types=bias_param_types,
            fault_type=fault_type,
            distribution=distribution,
            ratio=ratio,
            stuck_at=stuck_at,
            noise_std=noise_std,
            limit=limit,
            device=device,
        )

    @torch.no_grad()
    def apply_(self, model: nn.Module) -> None:
        # Weights
        for i, pname in enumerate(self.weight_param_names):
            p: torch.Tensor = self._get_attr_by_path(model, pname)
            keep = self.weight_keep_masks[i].to(p.device)
            ftype = self.param_types[i]
            if ftype in ("stuck", "connectivity"):
                fixed = self.weight_fixed_faults[i].to(p.device)
                p.mul_(keep)
                p.add_(fixed)
            elif ftype == "random":
                if self.noise_std is None:
                    raise ValueError("noise_std must be set for 'random' faults")
                sel = (keep == 0).to(p.dtype)
                noise = torch.randn_like(p) * float(self.noise_std)
                p.add_(noise * sel)

        # Bias
        for i, pname in enumerate(self.bias_param_names):
            p: torch.Tensor = self._get_attr_by_path(model, pname)
            keep = self.bias_keep_masks[i].to(p.device)
            ftype = self.bias_param_types[i]
            if ftype in ("stuck", "connectivity"):
                fixed = self.bias_fixed_faults[i].to(p.device)
                p.mul_(keep)
                p.add_(fixed)
            elif ftype == "random":
                if self.noise_std is None:
                    raise ValueError("noise_std must be set for 'random' faults")
                sel = (keep == 0).to(p.dtype)
                noise = torch.randn_like(p) * float(self.noise_std)
                p.add_(noise * sel)


def build_fault_manager(
        model: nn.Module,
        ratio: float,
        *,
        fault_type: str = "stuck",          # 'stuck' | 'random' | 'connectivity'
        distribution: str = "sporadic",     # 'sporadic' | 'clustered'
        stuck_at: Optional[float] = 0.0,    # for 'stuck'
        noise_std: Optional[float] = 0.05,  # for 'random'
        limit: Optional[float] = None,      # for 'connectivity'
        include_bias: bool = True,
) -> FaultManager:
    """Convenience factory. See module docstring for example usage."""
    return FaultManager.from_model(
        model,
        ratio=ratio,
        fault_type=fault_type,
        distribution=distribution,
        stuck_at=stuck_at,
        noise_std=noise_std,
        limit=limit,
        include_bias=include_bias,
    )

def get_fault_map(
        fault_mgr: "FaultManager",
        *,
        include_bias: bool = False,
) -> Dict[str, torch.Tensor]:
    fault_map: Dict[str, torch.Tensor] = {}

    # Weights: weight_param_names는 'layer.sub.weight' 같은 dotted path
    for pname, keep in zip(fault_mgr.weight_param_names, fault_mgr.weight_keep_masks):
        mask_fault = (keep == 0).to(torch.bool)
        if pname.endswith(".weight"):
            module_name = pname[:-7]
        elif pname == "weight":
            module_name = ""
        else:
            module_name = pname.rsplit(".", 1)[0]
        if module_name != "":
            fault_map[module_name] = mask_fault

    if include_bias:
        for pname, keep in zip(fault_mgr.bias_param_names, fault_mgr.bias_keep_masks):
            mask_fault = (keep == 0).to(torch.bool)
            if pname.endswith(".bias"):
                module_name = pname[:-5]
            elif pname == "bias":
                module_name = ""
            else:
                module_name = pname.rsplit(".", 1)[0]
            if module_name != "":
                fault_map[f"{module_name}.__bias__"] = mask_fault

    return fault_map