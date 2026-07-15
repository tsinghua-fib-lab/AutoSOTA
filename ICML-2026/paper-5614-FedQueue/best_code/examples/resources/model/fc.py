import torch.nn as nn


class FC(nn.Module):
    """
    A Fully connected layer.
    """

    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        out = self.fc(x)
        return out
