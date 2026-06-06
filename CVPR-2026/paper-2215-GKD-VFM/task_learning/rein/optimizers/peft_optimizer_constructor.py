# Copyright (c) OpenMMLab. All rights reserved.
import logging
from typing import List, Optional, Union
from mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper

import torch
import torch.nn as nn
from torch.nn import GroupNorm, LayerNorm

from mmengine.logging import print_log
from mmengine.registry import OPTIM_WRAPPER_CONSTRUCTORS, OPTIM_WRAPPERS, OPTIMIZERS
from mmengine.utils import is_list_of
from mmengine.utils.dl_utils import mmcv_full_available
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm, _InstanceNorm
from mmengine.optim.optimizer import DefaultOptimWrapperConstructor, OptimWrapper


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class PEFTOptimWrapperConstructor(DefaultOptimWrapperConstructor):
    def __init__(self, optim_wrapper_cfg: dict, paramwise_cfg: Optional[dict] = None):
        # assert "keywords" in optim_wrapper_cfg
        # self.keywords = optim_wrapper_cfg.pop("keywords")
        super().__init__(optim_wrapper_cfg, paramwise_cfg)

    def add_params(
        self,
        params: List[dict],
        module: nn.Module,
        prefix: str = "",
        is_dcn_module: Optional[Union[int, float]] = None,
    ) -> None:
        # get param-wise options
        custom_keys = self.paramwise_cfg.get("custom_keys", {})
        # first sort with alphabet order and then sort with reversed len of str
        sorted_keys = sorted(sorted(custom_keys.keys()), key=len, reverse=True)

        bias_lr_mult = self.paramwise_cfg.get("bias_lr_mult", None)
        bias_decay_mult = self.paramwise_cfg.get("bias_decay_mult", None)
        norm_decay_mult = self.paramwise_cfg.get("norm_decay_mult", None)
        dwconv_decay_mult = self.paramwise_cfg.get("dwconv_decay_mult", None)
        flat_decay_mult = self.paramwise_cfg.get("flat_decay_mult", None)
        bypass_duplicate = self.paramwise_cfg.get("bypass_duplicate", False)
        dcn_offset_lr_mult = self.paramwise_cfg.get("dcn_offset_lr_mult", None)

        # special rules for norm layers and depth-wise conv layers
        is_norm = isinstance(module, (_BatchNorm, _InstanceNorm, GroupNorm, LayerNorm))
        is_dwconv = (
            isinstance(module, torch.nn.Conv2d) and module.in_channels == module.groups
        )

        for name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            param_group = {"params": [param]}
            if bypass_duplicate and self._is_in(param_group, params):
                print_log(
                    f"{prefix} is duplicate. It is skipped since "
                    f"bypass_duplicate={bypass_duplicate}",
                    logger="current",
                    level=logging.WARNING,
                )
                continue
            if not param.requires_grad:
                params.append(param_group)
                continue

            # if the parameter match one of the custom keys, ignore other rules
            is_custom = False
            for key in sorted_keys:
                if key in f"{prefix}.{name}":
                    is_custom = True
                    lr_mult = custom_keys[key].get("lr_mult", 1.0)
                    param_group["lr"] = self.base_lr * lr_mult
                    if self.base_wd is not None:
                        decay_mult = custom_keys[key].get("decay_mult", 1.0)
                        param_group["weight_decay"] = self.base_wd * decay_mult
                    # add custom settings to param_group
                    for k, v in custom_keys[key].items():
                        param_group[k] = v
                    break

            if not is_custom:
                # bias_lr_mult affects all bias parameters
                # except for norm.bias dcn.conv_offset.bias
                if (
                    name == "bias"
                    and not (is_norm or is_dcn_module)
                    and bias_lr_mult is not None
                ):
                    param_group["lr"] = self.base_lr * bias_lr_mult

                if (
                    prefix.find("conv_offset") != -1
                    and is_dcn_module
                    and dcn_offset_lr_mult is not None
                    and isinstance(module, torch.nn.Conv2d)
                ):
                    # deal with both dcn_offset's bias & weight
                    param_group["lr"] = self.base_lr * dcn_offset_lr_mult

                # apply weight decay policies
                if self.base_wd is not None:
                    # norm decay
                    if is_norm and norm_decay_mult is not None:
                        param_group["weight_decay"] = self.base_wd * norm_decay_mult
                    # bias lr and decay
                    elif (
                        name == "bias"
                        and not is_dcn_module
                        and bias_decay_mult is not None
                    ):
                        param_group["weight_decay"] = self.base_wd * bias_decay_mult
                    # depth-wise conv
                    elif is_dwconv and dwconv_decay_mult is not None:
                        param_group["weight_decay"] = self.base_wd * dwconv_decay_mult
                    # flatten parameters except dcn offset
                    elif (
                        param.ndim == 1
                        and not is_dcn_module
                        and flat_decay_mult is not None
                    ):
                        param_group["weight_decay"] = self.base_wd * flat_decay_mult
            params.append(param_group)
            for key, value in param_group.items():
                full_name = f"{prefix}.{name}" if prefix else name
                if key == "params":
                    print_log(
                        f"paramwise_options -- {full_name}:num of {key}={sum(v.numel() for v in value)}",
                        logger="current",
                    )
                else:
                    print_log(
                        f"paramwise_options -- {full_name}:{key}={value}",
                        logger="current",
                    )

        if mmcv_full_available():
            from mmcv.ops import DeformConv2d, ModulatedDeformConv2d

            is_dcn_module = isinstance(module, (DeformConv2d, ModulatedDeformConv2d))
        else:
            is_dcn_module = False
        for child_name, child_mod in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            self.add_params(
                params, child_mod, prefix=child_prefix, is_dcn_module=is_dcn_module
            )

    def __call__(self, model: nn.Module) -> OptimWrapper:
        model.train()
        if hasattr(model, "module"):
            model = model.module

        optim_wrapper_cfg = self.optim_wrapper_cfg.copy()
        optim_wrapper_cfg.setdefault("type", "OptimWrapper")
        optimizer_cfg = self.optimizer_cfg.copy()
        # if no paramwise option is specified, just use the global setting
        if not self.paramwise_cfg:
            optimizer_cfg["params"] = model.parameters()
            optimizer = OPTIMIZERS.build(optimizer_cfg)
        else:
            # set param-wise lr and weight decay recursively
            params: List = []
            self.add_params(params, model)
            optimizer_cfg["params"] = params
            optimizer = OPTIMIZERS.build(optimizer_cfg)
        optim_wrapper = OPTIM_WRAPPERS.build(
            optim_wrapper_cfg, default_args=dict(optimizer=optimizer)
        )
        return optim_wrapper
