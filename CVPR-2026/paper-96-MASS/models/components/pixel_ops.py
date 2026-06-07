"""3D pixel shuffle and unshuffle operators.

Task encoding uses these layers to move information between spatial resolution
and channel dimensions for volumetric feature maps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'PixelShuffle3d',
    'PixelUnshuffle3d'
]

class PixelShuffle3d(nn.Module):
    '''
    This class is a 3d version of pixelshuffle.
    '''
    def __init__(self, scale):
        '''
        :param scale: upsample scale
        '''
        super().__init__()
        self.scale = scale

    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = channels // self.scale ** 3

        out_depth = in_depth * self.scale
        out_height = in_height * self.scale
        out_width = in_width * self.scale

        input_view = input.contiguous().view(batch_size, nOut, self.scale, self.scale, self.scale, in_depth, in_height, in_width)

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)


class PixelUnshuffle3d(nn.Module):
    '''
    This class is a 3d version of pixel unshuffle, which down-samples the input.
    '''
    def __init__(self, scale):
        '''
        :param scale: downsample scale
        '''
        super().__init__()
        self.scale = scale

    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = channels * (self.scale ** 3)

        out_depth = in_depth // self.scale
        out_height = in_height // self.scale
        out_width = in_width // self.scale

        input_view = input.view(batch_size, channels, out_depth, self.scale, out_height, self.scale, out_width, self.scale)
        # Permute to move the sub-voxel components to contiguous channel dimensions
        output = input_view.permute(0, 3, 5, 7, 1, 2, 4, 6).contiguous()
        return output.view(batch_size, nOut, out_depth, out_height, out_width)
