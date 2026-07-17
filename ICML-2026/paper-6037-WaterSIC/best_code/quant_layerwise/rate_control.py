"""Rate control for layerwise quantization.

Simple budget-based rate control that:
1. Tracks global bit budget across all layers.
2. Computes target rate for each layer based on remaining budget.
3. Supports weight-type multipliers (e.g., give wk/wq more bits).

Binary search in ZSIC is used to hit the target rate precisely.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class RateControlConfig:
    """Configuration for rate control."""

    enabled: bool = False

    # Global target average rate across all requested layers.
    global_target_rate_bits: float = 0.0

    # Clamp suggested target_rate_bits to [xmin, xmax].
    xmin: float = 0.05
    xmax: float = 16.0

    # Weight-type budget multipliers. Allocate more bits to harder weight types.
    # E.g., {"wk": 1.5, "wq": 1.25} gives wk 50% more bits, wq 25% more.
    # The total budget is preserved by normalizing across all weights.
    weight_budget_multipliers: Optional[Dict[str, float]] = None


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(float(x), float(lo)), float(hi)))


class RateController:
    """Budget-based rate controller with weight-type multipliers."""

    def __init__(
        self,
        *,
        cfg: RateControlConfig,
        layer_meta: Dict[str, Dict[str, Any]],
        existing: Optional[Dict[str, Dict[str, Any]]] = None,
        skip_layers: Optional[set] = None,
    ):
        """Initialize rate controller.

        Args:
            cfg: Rate control configuration.
            layer_meta: Dict mapping module_name -> {"numel": int, "weight": str}.
            existing: Already quantized layers with {"numel": int, "actual_rate": float}.
            skip_layers: Set of module names that will be stored at full precision (16 bits).
                         Budget is adjusted so quantized layers can still hit the target rate.
        """
        self.cfg = cfg
        self.layer_meta = layer_meta
        self.skip_layers = skip_layers or set()

        self.total_params = int(sum(int(m["numel"]) for m in layer_meta.values()))

        # Adjust budget to account for skipped layers (stored at 16 bits).
        # We want: (skip_params * 16 + quant_params * target_rate) / total_params = target_rate
        # Solving: quant_params * effective_rate = total_params * target_rate - skip_params * 16
        # So the budget for quantized layers must compensate for the 16-bit skipped layers.
        skip_params = sum(int(layer_meta[m]["numel"]) for m in self.skip_layers if m in layer_meta)
        quant_params = self.total_params - skip_params

        if quant_params > 0 and skip_params > 0:
            # Compute effective target rate for quantized layers to achieve global target
            # total_budget = total_params * target_rate
            # skip_cost = skip_params * 16
            # remaining_budget = total_budget - skip_cost (for quantized layers)
            total_budget = float(cfg.global_target_rate_bits) * float(self.total_params)
            skip_cost = float(skip_params) * 16.0
            remaining_for_quant = total_budget - skip_cost
            effective_rate_for_quant = remaining_for_quant / float(quant_params)

            print(f"[rate_control] Adjusting for {len(self.skip_layers)} skipped layers ({skip_params} params at 16 bits)")
            print(f"[rate_control] Global target: {cfg.global_target_rate_bits:.3f} bits/param")
            print(f"[rate_control] Effective target for quantized layers: {effective_rate_for_quant:.3f} bits/param")

            # Set total_budget_bits to only cover quantized layers at the effective rate
            # But we still track total_params for final average calculation
            self.total_budget_bits = remaining_for_quant
            self.skip_params = skip_params
        else:
            self.total_budget_bits = float(cfg.global_target_rate_bits) * float(self.total_params)
            self.skip_params = 0

        self.consumed_params = 0
        self.consumed_bits = 0.0

        # Track remaining params per weight type for dynamic budget allocation
        # Exclude skipped layers from the count (they're already accounted for in budget adjustment)
        self.remaining_params_by_wtype: Dict[str, int] = {}
        for mod, meta in layer_meta.items():
            if mod in self.skip_layers:
                continue  # Don't count skipped layers in remaining params
            wtype = str(meta.get("weight", "other"))
            self.remaining_params_by_wtype[wtype] = (
                self.remaining_params_by_wtype.get(wtype, 0) + int(meta["numel"])
            )

        # Process existing (already quantized) layers
        if existing:
            for mod, info in existing.items():
                numel = int(info.get("numel", layer_meta.get(mod, {}).get("numel", 0)))
                actual = float(info["actual_rate"])
                wtype = self._get_weight_type(mod)

                self.consumed_params += int(numel)
                self.consumed_bits += float(actual) * float(numel)

                if wtype in self.remaining_params_by_wtype:
                    self.remaining_params_by_wtype[wtype] = max(
                        0, self.remaining_params_by_wtype[wtype] - int(numel)
                    )

    def _get_weight_type(self, module_name: str) -> str:
        """Extract weight type from module name."""
        return str(self.layer_meta.get(module_name, {}).get("weight", "other"))

    def remaining_params(self) -> int:
        return int(max(0, self.total_params - self.consumed_params))

    def remaining_budget_bits(self) -> float:
        return float(self.total_budget_bits - self.consumed_bits)

    def desired_remaining_rate(self) -> float:
        """Compute the average rate needed for remaining layers to hit global target."""
        rem = self.remaining_params()
        if rem <= 0:
            return float("nan")
        return float(self.remaining_budget_bits() / float(rem))

    def _compute_target_rate_for_wtype(self, wtype: str, dead_params: int = 0) -> float:
        """Compute target rate for a weight type based on remaining budget.

        Allocates remaining budget proportionally across remaining weight types,
        respecting the multiplier ratios.

        Args:
            wtype: Weight type (e.g., "w2", "wo").
            dead_params: Number of dead params in the current weight. These are
                excluded from the denominator so the budget is distributed over
                alive params of the current weight + all remaining future params.

        Math:
          r_wtype = multiplier[wtype] * remaining_budget / weighted_remaining_params
          where weighted_remaining_params = sum(multiplier[w] * remaining_params[w] for all w)
        """
        mults = self.cfg.weight_budget_multipliers or {}
        remaining_budget = self.remaining_budget_bits()

        # Compute weighted sum of remaining params, excluding dead params
        weighted_remaining = 0.0
        for w, params in self.remaining_params_by_wtype.items():
            mult = float(mults.get(w, 1.0))
            adj_params = float(params)
            if w == wtype and dead_params > 0:
                adj_params = max(0.0, adj_params - float(dead_params))
            weighted_remaining += mult * adj_params

        if weighted_remaining <= 0:
            return float(self.cfg.global_target_rate_bits)

        # Rate for this weight type
        mult = float(mults.get(wtype, 1.0))
        rate = mult * remaining_budget / weighted_remaining

        return float(rate)

    def suggest_target_x(self, module_name: str, dead_params: int = 0) -> tuple[float, Dict[str, Any]]:
        """Suggest the target rate for the next layer.

        Args:
            module_name: Name of the module to quantize.
            dead_params: Number of dead params (elements) in this weight. When > 0,
                the budget is distributed over alive params of this weight + all
                remaining future params, giving more bits per alive element.

        Returns:
            (target_rate, info_dict) where target_rate is the suggested bits/param
            for alive elements.
        """
        wtype = self._get_weight_type(module_name)

        if self.cfg.weight_budget_multipliers:
            target_rate = self._compute_target_rate_for_wtype(wtype, dead_params=dead_params)
        else:
            # Simple mode: distribute remaining budget over remaining alive params
            rem = self.remaining_params() - dead_params
            if rem <= 0:
                target_rate = float(self.cfg.global_target_rate_bits)
            else:
                target_rate = self.remaining_budget_bits() / float(rem)

        # Clamp to valid range
        target_rate = _clip(target_rate, self.cfg.xmin, self.cfg.xmax)

        info = {
            "global_target_rate_bits": float(self.cfg.global_target_rate_bits),
            "desired_remaining_rate": float(self.desired_remaining_rate()),
            "target_for_wtype": float(target_rate),
            "suggested_target_x": float(target_rate),
            "weight_type": wtype,
            "remaining_params_for_wtype": self.remaining_params_by_wtype.get(wtype, 0),
            "dead_params": int(dead_params),
            "consumed_params": int(self.consumed_params),
            "consumed_bits": float(self.consumed_bits),
            "remaining_params": int(self.remaining_params()),
            "remaining_budget_bits": float(self.remaining_budget_bits()),
            "total_params": int(self.total_params),
            "total_budget_bits": float(self.total_budget_bits),
        }
        return float(target_rate), info

    def update(self, module_name: str, *, target_x: float, actual_rate: float) -> None:
        """Update controller after quantizing a layer.

        Args:
            module_name: Name of the quantized module.
            target_x: The target rate that was requested.
            actual_rate: The actual achieved rate (entropy + overhead).
        """
        meta = self.layer_meta.get(module_name)
        if meta is None:
            raise KeyError(f"Unknown module in layer_meta: {module_name}")

        numel = int(meta["numel"])
        wtype = self._get_weight_type(module_name)

        self.consumed_params += int(numel)
        self.consumed_bits += float(actual_rate) * float(numel)

        if wtype in self.remaining_params_by_wtype:
            self.remaining_params_by_wtype[wtype] = max(
                0, self.remaining_params_by_wtype[wtype] - int(numel)
            )

    def summary(self) -> Dict[str, Any]:
        """Get summary of current state."""
        current_weight_rates = {}
        if self.cfg.weight_budget_multipliers:
            for wtype in self.remaining_params_by_wtype.keys():
                current_weight_rates[wtype] = self._compute_target_rate_for_wtype(wtype)

        return {
            "cfg": asdict(self.cfg),
            "total_params": int(self.total_params),
            "total_budget_bits": float(self.total_budget_bits),
            "skip_layers": list(self.skip_layers),
            "skip_params": int(self.skip_params),
            "consumed_params": int(self.consumed_params),
            "consumed_bits": float(self.consumed_bits),
            "remaining_params": int(self.remaining_params()),
            "remaining_budget_bits": float(self.remaining_budget_bits()),
            "remaining_params_by_wtype": dict(self.remaining_params_by_wtype),
            "desired_remaining_rate": float(self.desired_remaining_rate()),
            "current_weight_rates": current_weight_rates,
        }

    def save_json(self, path: str) -> None:
        """Save state to JSON file."""
        with open(path, "w") as f:
            json.dump(self.summary(), f, indent=2)
