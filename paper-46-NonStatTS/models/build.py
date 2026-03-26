import os
from typing import Optional

import torch

from models import Autoformer, iTransformer, DLinear, PatchTST, FreTS, MICN, Pyraformer, TimesNet, OLS, Koopa, KNF


def build_model(cfg):
    assert cfg.MODEL.NAME in globals(), f"model {cfg.MODEL.NAME} is not defined"
    model_class = getattr(globals()[cfg.MODEL.NAME], "Model")
    if cfg.MODEL.NAME in ('Koopa', 'KNF'):
        model = model_class(cfg.MODEL, cfg)
    else:
        model = model_class(cfg.MODEL)

    if torch.cuda.is_available():
        model = model.cuda()

    return model


def build_norm_module(cfg):
    norm_module_name = cfg.NORM_MODULE.NAME
    if norm_module_name == "RevIN":
        from models.RevIN import RevIN
        norm_module = RevIN(cfg)
    elif norm_module_name == "SAN":
        from models.Statistics_prediction import Statistics_prediction
        norm_module = Statistics_prediction(cfg)
    elif norm_module_name == "DishTS":
        from models.DishTS import DishTS
        norm_module = DishTS(cfg)
    else:
        raise ValueError

    if torch.cuda.is_available():
        norm_module = norm_module.cuda()

    return norm_module


def save_best_model(cfg, model, optimizer, epoch=0, best_metric=0.0):
    model_path = os.path.join(cfg.TRAIN.CHECKPOINT_DIR, "checkpoint_best.pth")
    state = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'best_metric': best_metric
    }
    torch.save(state, model_path)
    print(f"Saved best model to {model_path}")


def load_best_model(cfg, model):
    model_path = os.path.join(cfg.TRAIN.CHECKPOINT_DIR, "checkpoint_best.pth")
    if os.path.isfile(model_path):
        print(f"Loading checkpoint from {model_path}")
        checkpoint = torch.load(model_path, map_location="cpu")

        state_dict = checkpoint['model_state']
        msg = model.load_state_dict(state_dict, strict=True)
        assert set(msg.missing_keys) == set()

        print(f"Loaded pre-trained model from {model_path}")
    else:
        print("=> no checkpoint found at '{}'".format(model_path))

    return model
