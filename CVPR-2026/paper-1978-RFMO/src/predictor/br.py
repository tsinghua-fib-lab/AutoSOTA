import torch
import torch.nn as nn

from .base import Predictor
from .losses import AsymmetricLoss

class BinaryRelevancePredictor(Predictor):

    def __init__(self, config: dict):
        super().__init__(config)
        self.threshold = config.get('threshold', 0.5)
        loss_config = config.get('loss', {})
        loss_type = loss_config.get('type', 'bce')
        loss_params = loss_config.get('params', {})
        if loss_type == 'bce':
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = AsymmetricLoss(**loss_params)

    def calculate_loss(self, targets, inputs, return_preds=False):
        outputs = self.net(inputs, targets)
        loss = self.loss_fn(outputs, targets)
        if return_preds:
            preds = (torch.sigmoid(outputs) > self.threshold).float()
            return loss, preds
        return loss
    
    def predict(self, inputs):
        outputs = self.net(inputs)
        preds = (torch.sigmoid(outputs) > self.threshold).float()
        return preds
    