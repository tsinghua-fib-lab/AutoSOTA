"""
Patch torchvision to be compatible with PyTorch 1.8.
Import this before importing torchdrug to avoid import errors.
"""
import sys
import importlib


def _patch_torchvision():
    """Monkey-patch torchvision __init__ to handle missing torch.ao module."""
    import torchvision

    # Patch the models import to be lazy
    original_init = torchvision.__dict__.copy()

    # Make models a lazy-loading module
    import types
    lazy_mod = types.ModuleType("torchvision.models")
    lazy_mod.__dict__["__file__"] = lazy_mod.__dict__.get("__file__", "")
    torchvision.models = lazy_mod

    # For datasets.utils.download_url (only thing ESM actually needs)
    # Make sure datasets is properly loaded
    from torchvision import datasets
    torchvision.datasets = datasets


_patch_torchvision()
