from torch import nn


class PINN(nn.Module):
    def __init__(self, input_dim=2, output_dim=1):
        super().__init__()
        act = nn.Tanh()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), act,
            nn.Linear(64, 64), act,
            nn.Linear(64, 64), act,
            nn.Linear(64, 64), act,
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.net(x)
