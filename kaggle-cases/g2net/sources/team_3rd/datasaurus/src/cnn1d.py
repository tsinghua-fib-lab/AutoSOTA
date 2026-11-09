# From https://www.kaggle.com/kit716/grav-wave-detection/data?select=g2net_models.py

import torch
from torch import nn
import torch.nn.functional as F


class ModelCNN_Dilations(nn.Module):
    """1D convolutional neural network with dilations. Classifier of the gravitaitonal waves
    Inspired by the https://arxiv.org/pdf/1904.08693.pdf
    """

    def __init__(self):
        super().__init__()
        self.init_conv = nn.Sequential(nn.Conv1d(3, 256, kernel_size=1), nn.ReLU())
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(256, 256, kernel_size=2, dilation=2 ** i),
                    nn.ReLU(),
                )
                for i in range(11)
            ]
        )
        self.out_conv = nn.Sequential(nn.Conv1d(256, 1, kernel_size=1), nn.ReLU())
        self.fc = nn.Linear(2049, 1)

    def forward(self, x):
        x = self.init_conv(x)
        for conv in self.convs:
            x = conv(x)
        x = self.out_conv(x)
        x = self.fc(x)
        x.squeeze_(1)
        return x


class Model1DCNN(nn.Module):
    """1D convolutional neural network. Classifier of the gravitational waves.
    Architecture from there https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.120.141103
    """

    def __init__(self, initial_channnels=8):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv1d(3, initial_channnels, kernel_size=64),
            nn.BatchNorm1d(initial_channnels),
            nn.ELU(),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv1d(initial_channnels, initial_channnels, kernel_size=32),
            nn.MaxPool1d(kernel_size=8),
            nn.BatchNorm1d(initial_channnels),
            nn.ELU(),
        )
        self.cnn3 = nn.Sequential(
            nn.Conv1d(initial_channnels, initial_channnels * 2, kernel_size=32),
            nn.BatchNorm1d(initial_channnels * 2),
            nn.ELU(),
        )
        self.cnn4 = nn.Sequential(
            nn.Conv1d(initial_channnels * 2, initial_channnels * 2, kernel_size=16),
            nn.MaxPool1d(kernel_size=6),
            nn.BatchNorm1d(initial_channnels * 2),
            nn.ELU(),
        )
        self.cnn5 = nn.Sequential(
            nn.Conv1d(initial_channnels * 2, initial_channnels * 4, kernel_size=16),
            nn.BatchNorm1d(initial_channnels * 4),
            nn.ELU(),
        )
        self.cnn6 = nn.Sequential(
            nn.Conv1d(initial_channnels * 4, initial_channnels * 4, kernel_size=16),
            nn.MaxPool1d(kernel_size=4),
            nn.BatchNorm1d(initial_channnels * 4),
            nn.ELU(),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(initial_channnels * 4 * 11, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.ELU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.ELU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.cnn5(x)
        x = self.cnn6(x)
        # print(x.shape)
        x = x.flatten(1)
        # x = x.mean(-1)
        # x = torch.cat([x.mean(-1), x.max(-1)[0]])
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    x = torch.randn(size=(16, 3, 4096))
    net = Model1DCNN(32)
    out = net(x)
    print(out.shape)
