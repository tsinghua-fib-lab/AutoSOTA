# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import override

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.rich import tqdm

from ..data import Dataset
from ..utils import get_device
from . import Predictor, PredictorConfig


@dataclass
class SimpleNetConfig(PredictorConfig):
    input_dim: int
    output_dim: int
    epochs: int
    lr: float


class SimpleNet(Predictor):
    def __init__(self, config: SimpleNetConfig):
        super().__init__(config)
        self.epochs = config.epochs
        self.device = get_device()
        self.model = nn.Sequential(nn.Linear(config.input_dim, config.output_dim), nn.Sigmoid()).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=config.lr)

    def train(self, X: np.ndarray, y: np.ndarray):
        model_dtype = next(self.model.parameters()).dtype
        X_tensor = torch.as_tensor(X, dtype=model_dtype, device=self.device)
        y_tensor = torch.as_tensor(y, dtype=model_dtype, device=self.device)
        tab = " " * 17
        for _ in tqdm(range(self.epochs), desc=tab + "Training SimpleNet"):
            self.optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            self.optimizer.step()

    @override
    def fit(self, dataset: "Dataset") -> None:
        """Fit SimpleNet to dataset."""
        X, y = dataset.load_predictor()
        self.train(X, y)

    @override
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            model_dtype = next(self.model.parameters()).dtype
            X_tensor = torch.as_tensor(X, dtype=model_dtype, device=self.device)
            predictions = self.model(X_tensor)
        return predictions.cpu().numpy()
