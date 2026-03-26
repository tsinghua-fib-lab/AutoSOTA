import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, d_in, d_mid=512):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_mid * 2),
            nn.LeakyReLU(),
            nn.Linear(d_mid * 2, d_mid * 2),
            nn.LeakyReLU(),
            nn.Linear(d_mid * 2, d_mid),
            nn.LeakyReLU(),
            nn.Linear(d_mid, 2),
        )

    def forward(self, x):
        y_hat = self.mlp(x)
        return y_hat
    

class MLP_with_Head(nn.Module):
    def __init__(self, d_in, d_mid=512):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(d_in, d_mid * 2),
            nn.LeakyReLU(),
            nn.Linear(d_mid * 2, d_mid * 2),
            nn.LeakyReLU(),
            nn.Linear(d_mid * 2, d_mid),
            nn.LeakyReLU(),
        )

        self.last_fc = nn.Linear(d_mid, 2)

    def forward(self, x):
        feature = self.feature_extractor(x)
        y_hat = self.last_fc(feature)
        return y_hat