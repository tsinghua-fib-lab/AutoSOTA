import torch

from .model import TabDPTLongContextModel


class DictToObject:
    def __init__(self, d):
        for key, value in d.items():
            if isinstance(value, dict):
                value = DictToObject(value)
            setattr(self, key, value)


def object_to_dict(obj):
    if isinstance(obj, dict):
        return {k: object_to_dict(v) for k, v in obj.items()}
    elif hasattr(obj, "__dict__"):
        return {k: object_to_dict(v) for k, v in vars(obj).items()}
    elif isinstance(obj, list):
        return [object_to_dict(item) for item in obj]
    else:
        return obj


def load_pretrained_tabdpt_model(ckpt_path: str | None, ckpt: dict | None = None) -> TabDPTLongContextModel:
    assert ckpt_path is not None or ckpt is not None, "Either ckpt_path or ckpt must be provided."
    checkpoint = torch.load(ckpt_path, weights_only=False, map_location="cpu") if ckpt is None else ckpt
    config = DictToObject(checkpoint["cfg"])
    model = TabDPTLongContextModel.load(
        model_state=checkpoint["model"],
        config=config,
    )
    return model


def load_pretrained_tabdpt_config(ckpt_path: str | None, ckpt: dict | None = None) -> dict:
    assert ckpt_path is not None or ckpt is not None, "Either ckpt_path or ckpt must be provided."
    checkpoint = torch.load(ckpt_path, weights_only=False, map_location="cpu") if ckpt is None else ckpt
    return object_to_dict(checkpoint["cfg"])
