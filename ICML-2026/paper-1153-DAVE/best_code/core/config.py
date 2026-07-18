from timm.models.vision_transformer import VisionTransformer

from pathlib import Path
from dataclasses import dataclass
from typing import Callable

from core.utils.augment import (
    DiffAugment, 
    AugCFG,
)
from core.utils.post_processing import (
    PostProcessing, 
    GaussianCFG, 
    BilateralCFG,
)
from core.utils.utils import (
    load_from_yaml, 
    load_timm_model,
)


@dataclass
class DAVEConfig:
    model_name: str
    model: VisionTransformer
    input_transform: Callable
    aug: DiffAugment
    post_proc: PostProcessing

    @classmethod
    def load_from_yaml(
        cls, 
        path: str | Path, 
        eps: float = 1e-8,
    ) -> "DAVEConfig":
        """
        Loads timm model config for DAVE Explainer.
        """
        cfg = load_from_yaml(path)
        
        model_name = cfg["model"]["model_name"]
        model_spec = cfg["model"]["model_spec"]

        model, input_transform = load_timm_model(model_spec)
        cls._check_vit_model(model)

        aug = DiffAugment(AugCFG(**cfg["augment"]))
        
        gauss_cfg = cfg["post_proc"]["gaussian"]
        bilat_cfg = cfg["post_proc"]["bilateral"]
        
        post_proc = PostProcessing(
            gaussian_cfg=GaussianCFG(**gauss_cfg),
            bilateral_cfg=BilateralCFG(**bilat_cfg),
            eps=eps,
        )

        return cls(
            model_name=model_name,
            model=model,
            input_transform=input_transform,
            aug=aug,
            post_proc=post_proc,
        )

    @classmethod
    def _check_vit_model(cls, model):
        """
        Checks if model is timm VisionTransformer.
        """
        if not isinstance(model, VisionTransformer):
            raise TypeError(
                f"Expected VisionTransformer, "
                f"got {type(model).__name__}"
            )
