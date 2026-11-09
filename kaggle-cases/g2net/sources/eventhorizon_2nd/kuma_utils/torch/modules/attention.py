import torch
from torch import nn
import torch.nn.functional as F
from .pooling import ChannelPool


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=12):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        self.pool = ChannelPool()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=(
            kernel_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(self.pool(x))
        return self.sigmoid(x)


class CBAM2d(nn.Module):

    def __init__(self, in_planes, kernel_size=7, return_mask=False):
        super().__init__()

        self.ch_attn = ChannelAttention(in_planes)
        self.sp_attn = SpatialAttention(kernel_size)
        self.return_mask = return_mask

    def forward(self, x):
        # x: bs x ch x w x h
        x = self.ch_attn(x) * x
        sp_mask = self.sp_attn(x)
        x = sp_mask * x
        if self.return_mask:
            return sp_mask, x
        else:
            return x


class MultiInstanceAttention(nn.Module):
    '''
    Implementation of: 
    Attention-based Multiple Instance Learning
    https://arxiv.org/abs/1802.04712
    '''

    def __init__(self, feature_size, instance_size,
                 num_classes=1, hidden_size=512, gated_attention=False):
        super().__init__()

        self.gated = gated_attention

        self.attn_U = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.Tanh()
        )
        if self.gated:
            self.attn_V = nn.Sequential(
                nn.Linear(feature_size, hidden_size),
                nn.Sigmoid()
            )
        self.attn_W = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: bs x k x f
        # k: num of instance
        # f: feature dimension
        bs, k, f = x.shape
        x = x.view(bs*k, f)
        if self.gated:
            x = self.attn_W(self.attn_U(x) * self.attn_V(x))
        else:
            x = self.attn_W(self.attn_U(x))
        x = x.view(bs, k, self.attn_W.out_features)
        x = F.softmax(x.transpose(1, 2), dim=2)  # Softmax over k
        return x  # : bs x 1 x k
