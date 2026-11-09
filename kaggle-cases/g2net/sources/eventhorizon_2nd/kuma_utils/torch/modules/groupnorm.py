import torch
import torch.nn.functional as F
from torch.nn.modules.normalization import GroupNorm as _GroupNorm


class GroupNorm2d(_GroupNorm):

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))


class GroupNorm3d(_GroupNorm):
    """
        Assume the data format is (B, C, D, H, W)
    """

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))


class GroupNorm1d(_GroupNorm):
    """
        Assume the data format is (N, C, W)
    """

    def _check_input_dim(self, input):
        if input.dim() != 3:
            raise ValueError('expected 3D input (got {}D input)'
                             .format(input.dim()))


def convert_groupnorm(module, num_groups=32):
    if isinstance(module, torch.nn.DataParallel):
        mod = module.module
        mod = convert_groupnorm(mod)

    mod = module
    for batchnorm, groupnorm in zip([torch.nn.modules.batchnorm.BatchNorm1d,
                                     torch.nn.modules.batchnorm.BatchNorm2d,
                                     torch.nn.modules.batchnorm.BatchNorm3d],
                                    [GroupNorm1d,
                                     GroupNorm2d,
                                     GroupNorm3d]):
        if isinstance(module, batchnorm):
            mod = groupnorm(
                num_groups, module.num_features,
                module.eps, module.affine)

            if module.affine:
                mod.weight.data = module.weight.data.clone().detach()
                mod.bias.data = module.bias.data.clone().detach()

    for name, child in module.named_children():
        mod.add_module(name, convert_groupnorm(child))

    return mod
