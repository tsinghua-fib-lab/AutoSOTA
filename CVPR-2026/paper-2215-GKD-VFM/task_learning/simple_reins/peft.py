import torch.nn as nn
from typing import List
from torch import Tensor
from typing import Iterable
import torch

done_work = set()

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
        for name in requires_grad_names:
            print(f"set_requires_grad----{name}")
        print(
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
        for train_name in train_names:
            print(f"set_train----{train_name}")
        first_set_train = False


def get_pyramid_feature(
    features: List[Tensor], scales: List[float] = [4, 2, 1, 0.5]
) -> List[Tensor]:
    """
    features: List[Tensor]
     scales: List[float]=[4,2,1,0.5]
    """
    pyramid_feature = []
    for i in range(len(features)):
        pyramid_feature.append(
            nn.functional.interpolate(
                features[i],
                scale_factor=scales[i],
                mode="bilinear",
                align_corners=False,
            )
        )
    return pyramid_feature

def do_once(func: callable, workname=None):
    global done_work
    if workname is None:
        workname=func()
    if workname not in done_work:
        func()
        done_work.add(workname)


def show_key(d: dict, blank=0, name=""):
    print("-" * blank + name + ":" + " " * 5 + str(type(d)), end="")
    if isinstance(d, torch.Tensor):
        print(" " * 5, d.shape)
    else:
        print()
    if hasattr(d, "keys"):
        for key in d.keys():
            element = d[key] if isinstance(d, Iterable) else getattr(d, key)
            # print("-" * blank + key + ":" + " " * 5 + str(type(element)))
            show_key(element, blank + 2, name=key)
    elif isinstance(d, list) or isinstance(d, tuple):
        show_key(d[0], blank + 2, name="~")
