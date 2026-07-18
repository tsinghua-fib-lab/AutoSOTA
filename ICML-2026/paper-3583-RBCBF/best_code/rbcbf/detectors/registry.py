"""Factory utilities for detectors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from .alpha_cumulative import AlphaCumulativeDetector
from .base import Detector
from .peak_alpha import PeakAlphaDetector
from .route_a_union import RouteAUnionDetector
from .route_b_joint import RouteBJointDetector
from .single_scalar.base_scalar import ScalarSeriesDetector
from .single_scalar.factory import create_scalar_builder
from .uncertainty_gated import UncertaintyGatedDetector
from .windowed_contrib import WindowedContributionDetector

DETECTOR_TYPES = {
    "alpha_cumulative": AlphaCumulativeDetector,
    "windowed_contrib": WindowedContributionDetector,
    "uncertainty_gated": UncertaintyGatedDetector,
    "route_a_union": RouteAUnionDetector,
    "route_b_joint": RouteBJointDetector,
    "peak_alpha": PeakAlphaDetector,
}


def load_detectors_config(path: str | None) -> dict | None:
    if not path:
        return None
    cfg_path = Path(path)
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def build_detectors(
    config: dict | None,
    labels: Sequence[str],
    label_info: Dict[str, Dict[str, float]] | None = None,
) -> List[Detector]:
    if not config:
        return []
    detectors_cfg: Iterable[dict] = config.get("detectors", [])
    detectors: List[Detector] = []
    for det_cfg in detectors_cfg:
        det_type = det_cfg.get("type")
        if det_type not in DETECTOR_TYPES and det_type != "single_scalar":
            raise ValueError(f"Unknown detector type: {det_type}")
        name = det_cfg.get("name") or det_type
        beta_weights = resolve_beta_weights(labels, det_cfg, label_info)
        if det_type == "single_scalar":
            scalar_type = det_cfg.get("scalar_type")
            builder = create_scalar_builder(
                scalar_type,
                labels,
                det_cfg,
                beta_weights=beta_weights,
            )
            selector_cfg = {
                "selector": det_cfg.get("selector", "alpha"),
                "alpha": det_cfg.get("alpha", 0.6),
                "window": det_cfg.get("window", 8),
            }
            detectors.append(
                ScalarSeriesDetector(
                    name=name,
                    labels=labels,
                    builder=builder,
                    selector_cfg=selector_cfg,
                )
            )
            continue
        cls = DETECTOR_TYPES[det_type]
        if det_type in {"route_a_union", "route_b_joint", "peak_alpha"}:
            detectors.append(
                cls(name=name, labels=labels, config=det_cfg, beta_weights=beta_weights)
            )
        else:
            detectors.append(cls(name=name, labels=labels, config=det_cfg))
    return detectors


def resolve_beta_weights(
    labels: Sequence[str],
    config: dict,
    label_info: Dict[str, Dict[str, float]] | None,
) -> List[float]:
    mode = str(config.get("beta_mode", "uniform"))
    if mode != "auroc":
        return [1.0 for _ in labels]
    auroc_map = (label_info or {}).get("auroc", {})
    weights = [float(auroc_map.get(label, 1.0)) for label in labels]
    total = sum(weights)
    if total <= 0:
        return [1.0 for _ in labels]
    return [w / total for w in weights]


__all__ = ["build_detectors", "load_detectors_config", "DETECTOR_TYPES"]
