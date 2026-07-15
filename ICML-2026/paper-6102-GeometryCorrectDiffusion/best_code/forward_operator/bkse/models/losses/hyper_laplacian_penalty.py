import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperLaplacianPenalty(nn.Module):
    def __init__(self, num_channels, alpha, eps=1e-6):
        super(HyperLaplacianPenalty, self).__init__()

        self.alpha = alpha
        self.eps = eps

        self.Kx = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).cuda()
        self.Kx = self.Kx.expand(1, num_channels, 3, 3)
        self.Kx.requires_grad = False
        self.Ky = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).cuda()
        self.Ky = self.Ky.expand(1, num_channels, 3, 3)
        self.Ky.requires_grad = False

    def forward(self, x):
        gradX = F.conv2d(x, self.Kx, stride=1, padding=1)
        gradY = F.conv2d(x, self.Ky, stride=1, padding=1)
        grad = torch.sqrt(gradX ** 2 + gradY ** 2 + self.eps)

        loss = (grad ** self.alpha).mean()

        return loss
