import torch.nn as nn

from .downsampler import Downsampler


class Swish(nn.Module):
    """
    https://arxiv.org/abs/1710.05941
    The hype was so huge that I could not help but try it
    """

    def __init__(self):
        super(Swish, self).__init__()
        self.s = nn.Sigmoid()

    def forward(self, x):
        return x * self.s(x)


def get_conv(in_f, out_f, kernel_size, stride=1, bias=True, pad="zero", downsample_mode="stride"):
    downsampler = None
    if stride != 1 and downsample_mode != "stride":

        if downsample_mode == "avg":
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == "max":
            downsampler = nn.MaxPool2d(stride, stride)
        elif downsample_mode in ["lanczos2", "lanczos3"]:
            downsampler = Downsampler(
                n_planes=out_f, factor=stride, kernel_type=downsample_mode, phase=0.5, preserve_size=True
            )
        else:
            assert False

        stride = 1

    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == "reflection":
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0

    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)

    layers = filter(lambda x: x is not None, [padder, convolver, downsampler])
    return nn.Sequential(*layers)


def get_activation(act_fun="LeakyReLU"):
    """
    Either string defining an activation function or module (e.g. nn.ReLU)
    """
    if isinstance(act_fun, str):
        if act_fun == "LeakyReLU":
            return nn.LeakyReLU(0.2, inplace=True)
        elif act_fun == "Swish":
            return Swish()
        elif act_fun == "ELU":
            return nn.ELU()
        elif act_fun == "none":
            return nn.Sequential()
        else:
            assert False
    else:
        return act_fun()
