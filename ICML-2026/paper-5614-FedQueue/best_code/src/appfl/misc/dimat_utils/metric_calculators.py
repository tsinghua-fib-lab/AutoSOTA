"""
Metric computation for alignment between models.
Ported from DIMAT/utils/metric_calculators.py.
"""

import torch
from abc import ABC, abstractmethod


class MetricCalculator(ABC):
    @abstractmethod
    def update(self, batch_size, *feats, **aux_params):
        return NotImplemented

    @abstractmethod
    def finalize(self):
        return NotImplemented


def compute_correlation(covariance, eps=1e-7):
    std = torch.diagonal(covariance).sqrt()
    covariance = covariance / (torch.clamp(torch.outer(std, std), min=eps))
    return covariance


class CovarianceMetric(MetricCalculator):
    name = "covariance"

    def __init__(self):
        self.std = None
        self.mean = None
        self.outer = None

    def update(self, batch_size, *feats, **aux_params):
        feats = torch.cat(feats, dim=0)
        feats = torch.nan_to_num(feats, 0, 0, 0)

        std = feats.std(dim=1)
        mean = feats.mean(dim=1)
        outer = (feats @ feats.T) / feats.shape[1]

        if self.mean is None:
            self.mean = torch.zeros_like(mean)
        if self.outer is None:
            self.outer = torch.zeros_like(outer)
        if self.std is None:
            self.std = torch.zeros_like(std)

        self.mean += mean * batch_size
        self.outer += outer * batch_size
        self.std += std * batch_size

    def finalize(self, numel, covsave_path, node, eps=1e-4):
        self.outer /= numel
        self.mean /= numel
        self.std /= numel
        cov = self.outer - torch.outer(self.mean, self.mean)
        if torch.isnan(cov).any():
            raise ValueError(f"NaN detected in covariance matrix at node {node}")
        if (torch.diagonal(cov) < -0.0001).sum():
            import warnings

            warnings.warn(f"Negative diagonal in covariance matrix at node {node}")
        return cov


class MeanMetric(MetricCalculator):
    name = "mean"

    def __init__(self):
        self.mean = None

    def update(self, batch_size, *feats, **aux_params):
        feats = torch.cat(feats, dim=0)
        mean = feats.abs().mean(dim=1)
        if self.mean is None:
            self.mean = torch.zeros_like(mean)
        self.mean += mean * batch_size

    def finalize(self, numel, covsave_path, node, eps=1e-4):
        return self.mean / numel


def get_metric_fns(names):
    metrics = {}
    for name in names:
        if name == "mean":
            metrics[name] = MeanMetric
        elif name == "covariance":
            metrics[name] = CovarianceMetric
        else:
            raise NotImplementedError(name)
    return metrics
