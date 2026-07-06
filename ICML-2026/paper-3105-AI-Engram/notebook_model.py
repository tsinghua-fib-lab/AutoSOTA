# Extracted from fig_resnet18_cifar10.ipynb cell 2
# ===== engram/models/cnn.py  (verbatim from the engram package) =====
import torch
import torch.nn as nn
from torchvision import models
from huggingface_hub import hf_hub_download

def load_resnet18_cifar(num_classes):
    m = models.resnet18(weights=None, num_classes=num_classes)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m

def load_pretrained_from_hf(dataset):
    HF_REPOS = {
        "cifar10":  {"repo_id": "edadaltocg/resnet18_cifar10",  "filename": "pytorch_model.bin", "num_classes": 10},
        "cifar100": {"repo_id": "edadaltocg/resnet18_cifar100", "filename": "pytorch_model.bin", "num_classes": 100},
    }
    spec = HF_REPOS[dataset]
    ckpt_path = hf_hub_download(repo_id=spec["repo_id"], filename=spec["filename"])
    obj = torch.load(ckpt_path, map_location="cpu")

    if isinstance(obj, dict) and "state_dict" in obj:
        state_dict = obj["state_dict"]
    else:
        state_dict = obj

    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        for pref in ("module.", "model."):
            if nk.startswith(pref):
                nk = nk[len(pref):]
        cleaned[nk] = v

    model = load_resnet18_cifar(spec["num_classes"])
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[warn] missing keys: {missing[:5]}{' ...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected[:5]}{' ...' if len(unexpected)>5 else ''}")
    return model

def resnet18_cifar10(pretrained=True):
    if pretrained:
        return load_pretrained_from_hf("cifar10")
    return load_resnet18_cifar(num_classses=10)

def resnet18_cifar100(pretrained=True):
    if pretrained:
        return load_pretrained_from_hf("cifar100")
    return load_resnet18_cifar(num_classses=100)