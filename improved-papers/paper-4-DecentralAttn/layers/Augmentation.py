import torch
import torch.nn as nn


class Jitter(nn.Module):
    # apply noise on each element
    def __init__(self, scale=0.):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        if self.training:
            x += torch.randn_like(x) * self.scale
        return x


class Scale(nn.Module):
    # scale each channel by a random scalar
    def __init__(self, scale=0.):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        if self.training:
            B, C, T = x.shape
            x *= 1 + torch.randn(B, C, 1, device=x.device) * self.scale
        return x


class Flip(nn.Module):
    # left-right flip
    def __init__(self, prob=0.):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        if self.training and torch.rand(1) < self.prob:
            return torch.flip(x, [-1])
        return x


class TemporalMask(nn.Module):
    # Randomly mask a portion of timestamps across all channels
    def __init__(self, ratio=0.):
        super().__init__()
        self.ratio = ratio

    def forward(self, x):
        if self.training:
            B, C, T = x.shape
            num_mask = int(T * self.ratio)
            mask_indices = torch.randperm(T)[:num_mask]
            x[:, :, mask_indices] = 0
        return x


class ChannelMask(nn.Module):
    # Randomly mask a portion of channels across all timestamps
    def __init__(self, ratio=0.):
        super().__init__()
        self.ratio = ratio

    def forward(self, x):
        if self.training:
            B, C, T = x.shape
            num_mask = int(C * self.ratio)
            mask_indices = torch.randperm(C)[:num_mask]
            x[:, mask_indices, :] = 0
        return x


class FrequencyMask(nn.Module):
    def __init__(self, ratio=0.):
        super().__init__()
        self.ratio = ratio

    def forward(self, x):
        if self.training:
            B, C, T = x.shape
            # Perform rfft
            x_fft = torch.fft.rfft(x, dim=-1)
            # Generate random indices for masking
            mask = torch.rand(x_fft.shape, device=x.device) > self.ratio
            # Apply mask
            x_fft = x_fft * mask
            # Perform inverse rfft
            x = torch.fft.irfft(x_fft, n=T, dim=-1)
        return x


def get_augmentation(augmentation):
    if augmentation.startswith("jitter"):
        if len(augmentation) == 6:
            return Jitter()
        return Jitter(float(augmentation[6:]))
    elif augmentation.startswith("scale"):
        if len(augmentation) == 5:
            return Scale()
        return Scale(float(augmentation[5:]))
    elif augmentation.startswith("drop"):
        if len(augmentation) == 4:
            return nn.Dropout(0.1)
        return nn.Dropout(float(augmentation[4:]))
    elif augmentation.startswith("flip"):
        if len(augmentation) == 4:
            return Flip()
        return Flip(float(augmentation[4:]))
    elif augmentation.startswith("frequency"):
        if len(augmentation) == 9:
            return FrequencyMask()
        return FrequencyMask(float(augmentation[9:]))
    elif augmentation.startswith("mask"):
        if len(augmentation) == 4:
            return TemporalMask()
        return TemporalMask(float(augmentation[4:]))
    elif augmentation.startswith("channel"):
        if len(augmentation) == 7:
            return ChannelMask()
        return ChannelMask(float(augmentation[7:]))
    elif augmentation == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown augmentation {augmentation}")