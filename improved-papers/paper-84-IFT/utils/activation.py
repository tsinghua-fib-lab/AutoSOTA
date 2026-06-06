from environment import *


class ALU(nn.Module):
    def __init__(self, w=1.):
        super().__init__()
        self.w = w

    def forward(self, x):
        return torch.abs(torch.where(x < 0, self.w * x, x))


class PLU(nn.Module):
    def __init__(self, w=1.):
        super().__init__()
        self.w = w

    def forward(self, x):
        return self.w * torch.atan2(torch.sin(x), torch.cos(x))


class SLU(nn.Module):
    def __init__(self, w=1.):
        super().__init__()
        self.w = w

    def forward(self, x):
        return torch.sin(self.w * x)


class CLU(nn.Module):
    def __init__(self, w=1.):
        super().__init__()
        self.w = w

    def forward(self, x):
        return torch.cos(self.w * x)
