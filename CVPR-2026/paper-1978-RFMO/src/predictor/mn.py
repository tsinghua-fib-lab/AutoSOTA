import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .base import Predictor
from .utils import infer_f1_labels


def gaussian_cdf(mu, sigma, x):
    return 0.5 * (1 + torch.erf((x - mu) / (sigma * np.sqrt(2))))


def gamma_cdf(alpha, beta, x):
    gamma_dist = torch.distributions.gamma.Gamma(alpha, beta)
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    return gamma_dist.cdf(x)


class MultinomialPredictor(Predictor):
    def __init__(self, config: dict):
        super().__init__(config)
        self.loss_fn = nn.CrossEntropyLoss()
        
        self.temperature = config.get('temperature', 1.0)

    def calculate_loss(self, targets, inputs, return_preds=False):
        n_classes = self.net.n_classes
        outputs, p0 = self.net(inputs, targets)

        # binary cross entropy loss for all negatives
        all_negatives = (targets.sum(dim=1) == 0).float()
        loss = F.binary_cross_entropy(p0, all_negatives)
        for i in range(1, n_classes + 1):
            i_is_positive = targets[:, i-1]
            num_positives = targets.sum(dim=1)
            _target = (i_is_positive * num_positives).long()
            _outputs = outputs[:, i-1]
            fake_outputs = torch.cat([torch.zeros_like(_outputs[:, :1]), _outputs], dim=1)
            
            exp_outputs = torch.exp(_outputs)
            sum_exp_outputs = exp_outputs.sum(dim=1) + 1
            loss += (torch.log(sum_exp_outputs) - fake_outputs.gather(1, _target.unsqueeze(1)).squeeze()).mean()

        if return_preds:
            preds = self.predict(inputs)
            return loss, preds
        return loss

    def _calculate_P_matrix(self, outputs):
        n_classes = self.net.n_classes
        P = torch.zeros(n_classes, n_classes).to(outputs.device)
        for i in range(1, n_classes + 1):
            _outputs = outputs[i-1]
            exp_outputs = torch.exp(_outputs / self.temperature)
            sum_exp_outputs = exp_outputs.sum() + 1
            P[i-1] = exp_outputs / sum_exp_outputs

        return P

    @torch.no_grad()
    def predict(self, inputs, return_time=False):
        n_classes = self.net.n_classes
        outputs, p0 = self.net(inputs)

        predictions = torch.zeros(outputs.size(0), n_classes).to(outputs.device)
        all_time = 0
        for b in range(outputs.size(0)):
            P = self._calculate_P_matrix(outputs[b])
            if return_time:
                f1_labels, time = infer_f1_labels(P, p0[b].item(), return_time=True)
                all_time += time
            else:
                f1_labels = infer_f1_labels(P, p0[b].item())
            for label in f1_labels:
                predictions[b, label] = 1
        if return_time:
            return predictions, all_time
        return predictions


    def get_avg_P_matrix(self, test_loader):
        P = torch.zeros(self.net.n_classes, self.net.n_classes)
        with torch.no_grad():
            for inputs, _ in test_loader:
                outputs, p0 = self.net(inputs)
                for b in range(outputs.size(0)):
                    singe_P = self._calculate_P_matrix(outputs[b])
                    P += singe_P

        return P / len(test_loader.dataset)
