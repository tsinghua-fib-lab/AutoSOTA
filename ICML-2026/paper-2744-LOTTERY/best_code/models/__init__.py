from .resnet import RN18_10, RN50_10
from .wide_resnet import WRN28_10
import torch
import os

_MODELS_DIR = os.path.dirname(os.path.abspath(__file__))


def load_model(model_arch, semantic=True):
    """
    Load a pretrained image classifier for embedding extraction.

    Args:
        model_arch: 'Res18' for ResNet-18, 'WRN28' for WideResNet-28x10
        semantic: If True, return penultimate-layer features instead of logits

    Returns:
        model: Loaded and eval-mode PyTorch model
    """
    if "Res18" in model_arch:
        model_path = os.path.join(_MODELS_DIR, "checkpoint", "resnet-18.pth")
        model = RN18_10(semantic=semantic)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(model_path))
        model.eval()

    elif "WRN28" in model_arch:
        model_path = os.path.join(_MODELS_DIR, "checkpoint", "wide-resnet-28x10.pth")
        model = WRN28_10(semantic=semantic)
        model.load_state_dict(torch.load(model_path)['net'])
        model = torch.nn.DataParallel(model)
        model.eval()

    else:
        raise ValueError(f"Unknown model architecture: {model_arch}")

    return model
