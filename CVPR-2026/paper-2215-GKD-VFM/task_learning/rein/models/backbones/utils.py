import torch.nn as nn
from typing import List
from mmengine.logging import MMLogger

first_set_requires_grad = True
first_set_train = True


def set_requires_grad(model: nn.Module, keywords: List[str]):
    """
    notice:key in name!
    """
    requires_grad_names = []
    num_params = 0
    num_trainable = 0
    for name, param in model.named_parameters():
        num_params += param.numel()
        if any(key in name for key in keywords):
            param.requires_grad = True
            requires_grad_names.append(name)
            num_trainable += param.numel()
        else:
            param.requires_grad = False
    global first_set_requires_grad
    if first_set_requires_grad:
        logger = MMLogger.get_current_instance()
        for name in requires_grad_names:
            logger.info(f"set_requires_grad----{name}")
        logger.info(
            f"Total trainable params--{num_trainable}, All params--{num_params}, Ratio--{num_trainable*100/num_params:.1f}%"
        )
        first_set_requires_grad = False


def _set_train(model: nn.Module, keywords: List[str], prefix: str = ""):
    train_names = []
    for name, child in model.named_children():
        fullname = ".".join([prefix, name])
        if any(name.startswith(key) for key in keywords):
            train_names.append(fullname)
            child.train()
        else:
            train_names += _set_train(child, keywords, prefix=fullname)
    return train_names


def set_train(model: nn.Module, keywords: List[str]):
    """
    notice:sub name startwith key!
    """
    model.train(False)
    train_names = _set_train(model, keywords)
    global first_set_train
    if first_set_train:
        logger = MMLogger.get_current_instance()
        for train_name in train_names:
            logger.info(f"set_train----{train_name}")
        first_set_train = False