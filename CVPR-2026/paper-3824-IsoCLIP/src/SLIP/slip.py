import os
from typing import Union

from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from .models import *

# Root directory where the script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Relative path to the models directory
MODEL_DIR = os.path.join(BASE_DIR, "data", "baselines", "SLIP")

_MODELS = {
    "SLIP-ViT-S-16": os.path.join(MODEL_DIR, "slip_small_100ep.pt"),
    "SLIP-ViT-B-16": os.path.join(MODEL_DIR, "slip_base_100ep.pt"),
    "SLIP-ViT-L-16": os.path.join(MODEL_DIR, "slip_large_100ep.pt"),
}


def _convert_image_to_rgb(image):
    return image.convert("RGB")


normalize = Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])

transform = Compose([
    Resize(224),
    CenterCrop(224),
    _convert_image_to_rgb,
    ToTensor(),
    normalize
])


def load_slip(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu"):
    if name not in _MODELS:
        raise RuntimeError(f"Model {name} not found in available models: {list(_MODELS.keys())}")

    model_path = _MODELS[name]

    try:
        state_dict = torch.load(model_path, map_location=device)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Model checkpoint not found at: {model_path}\n"
            f"Please download the pretrained SLIP model from the original repository:\n"
            f"https://github.com/facebookresearch/SLIP\n"
            f"and place it in the following directory:\n{model_path}"
        )

    # Load model architecture
    if name == "SLIP-ViT-S-16":
        model = SLIP_VITS16(**state_dict['args'].__dict__)
    elif name == "SLIP-ViT-B-16":
        model = SLIP_VITB16(**state_dict['args'].__dict__)
    elif name == "SLIP-ViT-L-16":
        model = SLIP_VITL16(**state_dict['args'].__dict__)
    else:
        raise RuntimeError(f"Model {name} not found")

    state_dict['state_dict'] = {k.replace('module.', ''): v for k, v in state_dict['state_dict'].items()}
    model.load_state_dict(state_dict['state_dict'])

    model.to(device)
    model.eval()
    return model, transform
