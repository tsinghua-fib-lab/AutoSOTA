import os

import torch
from torchvision import models


def load_init_model(arch_name, num_class, device, seed, pretrained=True, linear_probing=True, model_dir="./outputs/models"):
    """Load (or create and cache) the initial model checkpoint.

    The model is saved under ``<model_dir>/init/`` so the same initialisation
    is reused across runs with identical settings.
    """
    if pretrained:
        sub = "pretrained_linear_probing" if linear_probing else "pretrained_fine_tune_all"
    else:
        sub = "train_from_scratch"

    init_dir = os.path.join(model_dir, "init")
    os.makedirs(init_dir, exist_ok=True)

    cache_path = os.path.join(init_dir, f"{arch_name}_{num_class}classes_{sub}_seed{seed}.pt")

    if os.path.exists(cache_path):
        model = torch.load(cache_path, map_location=device, weights_only=False)
    else:
        weights = "IMAGENET1K_V1" if pretrained else None
        if arch_name == "vit-b":
            model = models.vit_b_16(weights=weights)
        elif arch_name == "vit-l":
            model = models.vit_l_16(weights=weights)
        else:
            raise ValueError(f"Unsupported architecture: {arch_name!r}. Choose from ['vit-b', 'vit-l'].")
        model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_class)
        torch.save(model, cache_path)
        print(f"Saved initial model to {cache_path}")

    if pretrained and linear_probing:
        for name, param in model.named_parameters():
            param.requires_grad = "heads.head" in name

    return model
