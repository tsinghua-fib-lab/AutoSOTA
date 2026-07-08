"""
CSR Project - Default Configuration

All paths and hyperparameters are centralized here.
Override via YAML config files or command-line arguments.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ======================== Path Defaults ========================

@dataclass
class PathConfig:
    """Filesystem paths for datasets, models, and outputs.

    Override these defaults via environment variables, YAML config files,
    or command-line arguments.
    """
    # Dataset root: expects subdirectories like General/CIFAR10, FineGrained/OxfordPets, etc.
    data_root: str = os.environ.get("CSR_DATA_ROOT", "./data")

    # Pre-trained model weights (set via env vars or config file)
    openclip_pretrained: str = os.environ.get("CSR_OPENCLIP_PRETRAINED", "")
    fare_weights: str = os.environ.get("CSR_FARE_WEIGHTS", "")
    tecoa_weights: str = os.environ.get("CSR_TECOA_WEIGHTS", "")

    # HuggingFace CLIP cache (auto-detected)
    hf_clip_b16: str = os.path.expanduser("~/.cache/huggingface/hub/openai/clip-vit-base-patch16")
    hf_clip_b32: str = os.path.expanduser("~/.cache/huggingface/hub/openai/clip-vit-base-patch32")
    hf_clip_l14: str = os.path.expanduser("~/.cache/huggingface/hub/openai/clip-vit-large-patch14")

    # Output directories
    adv_save_root: str = "./outputs/adv_samples"
    results_root: str = "./results"


# ======================== CLIP Constants ========================

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

IMAGE_SIZE = 224


# ======================== CSR Defense Config ========================

@dataclass
class CSRConfig:
    """Hyperparameters for the CSR (CLIP Spectral Robustness) defense."""
    # Low-pass filter
    filter_type: str = "gaussian"       # "gaussian" | "butterworth" | "ideal"
    lpf_radius: int = 40
    butterworth_order: int = 2

    # Detection
    detect_thresh: float = 0.85

    # Purification (PGD-based)
    purify_steps: int = 3
    purify_eps: float = 4 / 255
    purify_alpha: float = 2 / 255


@dataclass
class TTCConfig:
    """Hyperparameters for the TTC (Test-Time Correction) defense."""
    eps: float = 4 / 255
    alpha: float = 2 / 255
    steps: int = 3
    tau: float = 0.3
    beta: float = 2.0


# ======================== Attack Config ========================

@dataclass
class AttackConfig:
    """Default hyperparameters for adversarial attacks."""
    epsilons: List[float] = field(default_factory=lambda: [1/255, 2/255, 4/255])
    pgd_steps: int = 10
    pgd_alpha_scale: float = 2.5
    adaptive_freq_radius: int = 40
    adaptive_hf_reg_weight: float = 1.0

    cw_steps: int = 30
    cw_kappa: float = 0.0
    cw_alpha_scale: float = 2.5

    apgd_steps: int = 30

    batch_size: int = 16
    sample_n: int = 1000


# ======================== Evaluation Config ========================

@dataclass
class EvalConfig:
    """Benchmark evaluation settings."""
    batch_size: int = 32
    num_workers: int = 4
    sample_n: Optional[int] = 1000

    datasets: List[str] = field(default_factory=lambda: [
        "General/ImageNet",
        "General/CIFAR10",
        "General/CIFAR100",
        "General/STL10",
        "General/Caltech101",
        "General/Caltech256",
        "FineGrained/OxfordPets",
        "FineGrained/Flowers102",
        "FineGrained/Food101",
        "FineGrained/StanfordCars",
        "Scene/SUN397",
        "Scene/Country211",
        "Domain/FGVCAircraft",
        "Domain/EuroSAT",
        "Domain/DTD",
        "Domain/PCAM",
    ])

    adv_subfolders: List[str] = field(default_factory=lambda: ["1_255", "2_255", "4_255"])


# ======================== Model Registry ========================

# OpenCLIP-based model registry (for open_clip.create_model_and_transforms)
OPENCLIP_REGISTRY: Dict[str, dict] = {
    "Openai-CLIP-L/14": {
        "path": None,
        "base_arch": "ViT-L-14-quickgelu",
    },
    "FARE-CLIP-L/14": {
        "path": os.environ.get("CSR_FARE_WEIGHTS", ""),
        "base_arch": "ViT-L-14-quickgelu",
    },
    "TeCoA-CLIP-L/14": {
        "path": os.environ.get("CSR_TECOA_WEIGHTS", ""),
        "base_arch": "ViT-L-14-quickgelu",
    },
}

# HuggingFace CLIP model registry (for transformers.CLIPModel)
HF_MODEL_CONFIGS: Dict[str, str] = {
    "CLIP-B-16": "/models/clip-vit-base-patch16",
    "CLIP-B-32": "/models/clip-vit-base-patch32",
    "CLIP-L-14": "/models/clip-vit-large-patch14",
}
