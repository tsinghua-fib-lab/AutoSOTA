import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Predictor
from .utils import infer_f1_labels


logger = logging.getLogger(__name__)


class GibbsSamplingPredictor(Predictor):

    def __init__(self, config: dict):
        super().__init__(config)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.num_samples_to_infer = config['num_samples_to_infer']
        self.metric_to_infer = config['metric_to_infer']
        self.temperature = config['temperature']
        self.burn_in = config['burn_in']

    def calculate_loss(self, targets, inputs, return_preds=False):
        feat = self.net.forward_backbone(inputs)
        outputs = self.net.head(feat, targets)
        p0 = self.net.head.forward_p0(feat)
        all_negatives = (targets.sum(dim=1) == 0).float()
        loss = self.loss_fn(outputs, targets) + F.binary_cross_entropy(p0, all_negatives)
        if return_preds:
            predictions = self.predict(inputs)
            return loss, predictions
        return loss
    
    def evaluate_log_likelihood(self, targets, inputs, order, cut_off=-1):
        return self.net.evaluate_log_likelihood(inputs, targets, order, cut_off)
    
    def predict(self, inputs, order=None):
        x = self.net.forward_backbone(inputs)
        p0 = self.net.head.forward_p0(x)
        sampled_labels = self.net.sample(inputs, self.num_samples_to_infer, self.temperature, order=order, burn_in=self.burn_in)
        predictions = self._infer(sampled_labels, p0)
        return predictions

    def _calculate_P_matrix(self, sampled_labels):
        num_labels = sampled_labels.size(1)
        P = torch.zeros(num_labels, num_labels)

        for i in range(self.num_samples_to_infer):
            positive_labels = torch.where(sampled_labels[i] == 1)[0]
            num_positive_labels = len(positive_labels)
            if num_positive_labels == 0:
                continue
            for l in positive_labels:
                P[l, num_positive_labels-1] += 1

        P /= self.num_samples_to_infer

        return P

    def _infer(self, sampled_labels, p0):
        if self.metric_to_infer == 'sa':
            return self._infer_sa(sampled_labels)
        elif self.metric_to_infer == 'f1':
            return self._infer_f1(sampled_labels, p0)
        else:
            raise ValueError(f'Invalid metric_to_infer: {self.metric_to_infer}')

    def _infer_sa(self, sampled_labels):
        batch_size = sampled_labels.size(0)
        num_labels = sampled_labels.size(2)
        encoded_mask = 2 ** torch.arange(num_labels).to(self.device)
        sa_predictions = torch.zeros(batch_size, num_labels)
        for b in range(batch_size):
            encoded_labels_count = {}
            max_count, max_encoded_label = 0, 0
            for i in range(self.num_samples_to_infer):
                encoded_labels = torch.sum(sampled_labels[b, i] * encoded_mask).item()
                if encoded_labels not in encoded_labels_count:
                    encoded_labels_count[encoded_labels] = 0
                encoded_labels_count[encoded_labels] += 1
                if encoded_labels_count[encoded_labels] > max_count:
                    max_count = encoded_labels_count[encoded_labels]
                    max_encoded_label = int(encoded_labels)

            # decode the label
            for l in range(num_labels):
                if max_encoded_label & (1 << l):
                    sa_predictions[b, l] = 1

        return sa_predictions
    
    def _infer_f1(self, sampled_labels, p0):
        batch_size = sampled_labels.size(0)
        num_labels = sampled_labels.size(2)
        f1_predictions = torch.zeros(batch_size, num_labels)
        for b in range(batch_size):
            P = self._calculate_P_matrix(sampled_labels[b])

            f1_labels = infer_f1_labels(P, p0[b].item())
            for label in f1_labels:
                f1_predictions[b, label] = 1
        return f1_predictions

    def get_avg_P_matrix(self, test_loader):
        P = torch.zeros(self.net.n_classes, self.net.n_classes)
        with torch.no_grad():
            for inputs, _ in test_loader:
                sampled_labels = self.net.sample(inputs, self.num_samples_to_infer, self.temperature, burn_in=self.burn_in)
                for b in range(inputs.size(0)):
                    singe_P = self._calculate_P_matrix(sampled_labels[b])
                    P += singe_P

        return P / len(test_loader.dataset)
