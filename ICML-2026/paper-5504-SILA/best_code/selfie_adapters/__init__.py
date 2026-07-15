"""SelfIE Adapters - Lightweight adapters for self-interpretation via activation patching."""

from selfie_adapters.projection import (
    ProjectionModule,
    ScaleOnlyProjection,
    ScalarAffineProjection,
    FullRankAffineProjection,
    LowRankOnlyProjection,
    ScalarAffinePlusLowRankProjection,
    create_projection_module,
)
from selfie_adapters.inference import SelfIEAdapter, load_adapter
from selfie_adapters.sae_utils import load_sae, ObservableLanguageModel

__version__ = "0.1.0"

__all__ = [
    # Projection modules
    "ProjectionModule",
    "ScaleOnlyProjection",
    "ScalarAffineProjection",
    "FullRankAffineProjection",
    "LowRankOnlyProjection",
    "ScalarAffinePlusLowRankProjection",
    "create_projection_module",
    # Inference utilities
    "SelfIEAdapter",
    "load_adapter",
    # SAE and model utilities
    "load_sae",
    "ObservableLanguageModel",
]
