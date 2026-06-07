import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Predictor
from .utils import infer_f1_labels


logger = logging.getLogger(__name__)


class AutoregressivePredictor(Predictor):

    def __init__(self, config: dict):
        super().__init__(config)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.num_samples_to_infer = config['num_samples_to_infer']
        self.metric_to_infer = config['metric_to_infer']
        self.temperature = config['temperature']
        self.num_ensemble_runs = config.get('num_ensemble_runs', 1)

    def calculate_loss(self, targets, inputs, return_preds=False):
        outputs = self.net(inputs, targets)
        loss = self.loss_fn(outputs, targets)
        if return_preds:
            predictions = self.predict(inputs)
            return loss, predictions
        return loss

    def evaluate_log_likelihood(self, targets, inputs, order, cut_off=-1):
        return self.net.evaluate_log_likelihood(inputs, targets, order, cut_off)

    @torch.no_grad()
    def predict(self, inputs, order=None, return_time=False):
        if self.metric_to_infer == 'greedy':
            predictions = self.net.greedy_predict(inputs)
            return predictions
        p0 = torch.zeros(inputs.size(0)).to(inputs.device)
        if self.num_ensemble_runs > 1:
            batch_size = inputs.size(0)
            num_labels = self.net.n_classes
            all_P = torch.zeros(batch_size, num_labels, num_labels).to(inputs.device)
            for run_idx in range(self.num_ensemble_runs):
                sampled_labels = self.net.sample(inputs, self.num_samples_to_infer // self.num_ensemble_runs, self.temperature, order=order)
                for b in range(batch_size):
                    all_P[b] += self._calculate_P_matrix(sampled_labels[b])
            all_P /= self.num_ensemble_runs
            f1_predictions = torch.zeros(batch_size, num_labels).to(inputs.device)
            for b in range(batch_size):
                f1_labels = infer_f1_labels(all_P[b], p0[b].item())
                for label in f1_labels:
                    f1_predictions[b, label] = 1
            if return_time:
                return f1_predictions, 0.0
            return f1_predictions
        else:
            sampled_labels = self.net.sample(inputs, self.num_samples_to_infer, self.temperature, order=order)
            if return_time:
                predictions, time = self._infer(sampled_labels, p0, return_time=return_time)
                return predictions, time
            else:
                predictions = self._infer(sampled_labels, p0, return_time=return_time)
                return predictions

    def order_selection(self, val_loader):
        # greedy search
        n_classes = self.net.n_classes
        greedy_best_order = []
        for idx in range(n_classes):
            max_ll, max_j = -np.inf, -1
            for j in range(n_classes):
                if j in greedy_best_order:
                    continue
                ll = 0.
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        ll += self.evaluate_log_likelihood(targets, inputs, np.array(greedy_best_order + [j]), cut_off=idx).sum().item()
                logger.info(f'Greedy Search, current order: {greedy_best_order + [j]}, log likelihood: {ll}')
                if ll > max_ll:
                    max_ll = ll
                    max_j = j
            greedy_best_order.append(max_j)
            logger.info(f'Greedy Search, select {max_j} at {idx}-th position')
        greedy_ll = max_ll
        logger.info(f'Greedy Search, best order: {greedy_best_order} with log likelihood: {greedy_ll}')

        # random search
        random_best_order = []
        max_ll = -np.inf
        for _ in range(1000):
            order = np.random.permutation(n_classes)
            ll = 0.
            with torch.no_grad():
                for inputs, targets in val_loader:
                    ll += self.evaluate_log_likelihood(targets, inputs, order).sum().item()
            if ll > max_ll:
                max_ll = ll
                random_best_order = order.tolist()
                logger.info(f'Random Search, find a better order: {order}, log likelihood: {ll}')
        random_ll = max_ll
        logger.info(f'Random Search, best order: {random_best_order} with log likelihood: {random_ll}')

        if greedy_ll > random_ll:
            return torch.tensor(greedy_best_order)
        else:
            return torch.tensor(random_best_order)

    def _calculate_P_matrix(self, sampled_labels):
        num_labels = sampled_labels.size(1)
        P = torch.zeros(num_labels, num_labels).to(sampled_labels.device)


        label_volume = torch.sum(sampled_labels, dim=1)
        L = sampled_labels * label_volume.unsqueeze(1)
        for l in range(num_labels):
            value_counts = torch.bincount(L[:, l].long(), minlength=num_labels+1)
            P[l] = value_counts[1:]

        P /= self.num_samples_to_infer

        return P

    def _infer(self, sampled_labels, p0, return_time=False):
        if self.metric_to_infer == 'sa':
            return self._infer_sa(sampled_labels)
        elif self.metric_to_infer == 'f1':
            return self._infer_f1(sampled_labels, p0, return_time)
        elif self.metric_to_infer == 'ha':
            return self._infer_ha(sampled_labels)
        else:
            raise ValueError(f'Invalid metric_to_infer: {self.metric_to_infer}')

    def _infer_ha(self, sampled_labels):
        batch_size = sampled_labels.size(0)
        num_labels = sampled_labels.size(2)
        ha_predictions = torch.zeros(batch_size, num_labels).to(sampled_labels.device)
        for b in range(batch_size):
            marginal_prob = torch.mean(sampled_labels[b], dim=0)
            ha_predictions[b] = (marginal_prob > 0.5).float()
        return ha_predictions

    def _infer_sa(self, sampled_labels):
        batch_size = sampled_labels.size(0)
        num_labels = sampled_labels.size(2)
        encoded_mask = 2 ** torch.arange(num_labels).to(self.device)
        sa_predictions = torch.zeros(batch_size, num_labels).to(sampled_labels.device)
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

    def _infer_f1(self, sampled_labels, p0, return_time=False):
        batch_size = sampled_labels.size(0)
        num_labels = sampled_labels.size(2)
        f1_predictions = torch.zeros(batch_size, num_labels).to(sampled_labels.device)
        all_time = 0.0
        for b in range(batch_size):
            P = self._calculate_P_matrix(sampled_labels[b])
            if return_time:
                f1_labels, time = infer_f1_labels(P, p0[b].item(), return_time=return_time)
                all_time += time
            else:
                f1_labels = infer_f1_labels(P, p0[b].item(), return_time=return_time)
            for label in f1_labels:
                f1_predictions[b, label] = 1

        if return_time:
            return f1_predictions, all_time
        return f1_predictions

    def get_avg_P_matrix(self, test_loader):
        P = torch.zeros(self.net.n_classes, self.net.n_classes)
        with torch.no_grad():
            for inputs, _ in test_loader:
                sampled_labels = self.net.sample(inputs, self.num_samples_to_infer, self.temperature)
                for b in range(inputs.size(0)):
                    singe_P = self._calculate_P_matrix(sampled_labels[b])
                    P += singe_P

        return P / len(test_loader.dataset)
