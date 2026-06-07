import os
import random
import re
from dataclasses import dataclass

import numpy as np
import torch


IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
MONET_COLUMNS = [
    "lesion_id",
    "image_type",
    "MONET_ulceration_crust",
    "MONET_hair",
    "MONET_vasculature_vessels",
    "MONET_erythema",
    "MONET_pigmented",
    "MONET_gel_water_drop_fluid_dermoscopy_liquid",
    "MONET_skin_markings_pen_ink_purple_pen",
]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class Runtime:
    device: torch.device
    pin_memory: bool
    use_amp: bool
    amp_device_type: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_and_flags() -> Runtime:
    if torch.cuda.is_available():
        return Runtime(torch.device("cuda"), True, True, "cuda")
    if torch.backends.mps.is_available():
        return Runtime(torch.device("mps"), False, False, "cpu")
    return Runtime(torch.device("cpu"), False, False, "cpu")


def extract_lesion_id(filename: str) -> str:
    base = os.path.basename(filename)
    match = re.match(r"(IL_\d+)", base)
    if match:
        return match.group(1)
    stem = os.path.splitext(base)[0]
    parts = stem.split("_")
    if len(parts) >= 2 and parts[0] == "IL":
        return "IL_" + parts[1]
    return stem.split("_")[0]


def get_label_from_path(path: str) -> int:
    return 1 if f"{os.sep}mel{os.sep}" in path else 0


def normalise_image_type(value: str) -> str:
    value = str(value).lower()
    return "clinical" if value.startswith("clinical") else "dermoscopic"


def find_melanoma_class(classes: list[str]) -> str | None:
    for class_name in classes:
        lowered = class_name.lower()
        if lowered == "mel" or lowered == "melanoma" or "mel" in lowered:
            return class_name
    return None
