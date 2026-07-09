from torch import nn


class Convolution(nn.Module):
    def __init__(self,
                 kernel_size,
                 out_channels,
                 dilation,
                 ):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(in_channels=1, out_channels=out_channels,
                              kernel_size=kernel_size, dilation=1, padding=padding)

    def forward(self, x):
        out = self.conv(x)
        out = out.transpose(1, 2)
        return out
