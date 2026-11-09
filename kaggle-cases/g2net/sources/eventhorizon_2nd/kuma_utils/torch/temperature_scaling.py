'''
Temperature Scaling
https://github.com/gpleiss/temperature_scaling
Modified
'''
import torch
from torch import nn, optim
from torch.nn import functional as F
from sklearn.calibration import calibration_curve


class TemperatureScaler(nn.Module):

    def __init__(self, model, verbose=False):
        super().__init__()
        self.model = model
        self.verbose = verbose

    def forward(self, x):
        return self.temperature_scale(self.model(x))

    def temperature_scale(self, logits):
        temperature = self.temperature.expand(logits.size(0), -1)
        return logits / temperature

    def set_temperature(self, logits, labels):
        nll_criterion = nn.BCEWithLogitsLoss()
        probas = logits.sigmoid()
        self.temperature = nn.Parameter(torch.ones((1, probas.size(1))) * 1.5)

        before_temperature_nll = nll_criterion(logits, labels).item()
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        if self.verbose:
            print(f'Optimal temperature: {self.temperature}')
