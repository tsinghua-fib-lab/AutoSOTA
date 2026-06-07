import numpy as np
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = np.multiply(val, weight)
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum = np.add(self.sum, np.multiply(val, weight))
        self.count = self.count + weight
        self.avg = self.sum / self.count

    @property
    def value(self):
        return self.val

    @property
    def average(self):
        return np.round(self.avg, 5)


class MetricTracker(object):
    def __init__(self, window_size):
        self.window_size = window_size
        self.reset()

    def reset(self):
        self._is_ready = False
        self._metric_history = []
        self._sum_metric = 0

    def update(self, metric):
        if len(self._metric_history) == self.window_size:
            self._sum_metric -= self._metric_history.pop(0)
        self._metric_history.append(metric)
        self._sum_metric += metric
        self._is_ready = len(self._metric_history) == self.window_size

    def value(self):
        assert self._is_ready, 'Not enough values to compute the metric'
        return self._sum_metric / len(self._metric_history)
    
    def is_ready(self):
        return self._is_ready


def calc_mlc_metrics(preds, targets, num_classes) -> dict[str, float]:
    if isinstance(preds, list):
        preds = np.concatenate(preds)
    if isinstance(targets, list):
        targets = np.concatenate(targets)

    cls_tp = (preds * targets).sum(axis=0)
    cls_fp = (preds * (1 - targets)).sum(axis=0)
    cls_fn = ((1 - preds) * targets).sum(axis=0)
    ins_tp = (preds * targets).sum(axis=1)
    ins_fp = (preds * (1 - targets)).sum(axis=1)
    ins_fn = ((1 - preds) * targets).sum(axis=1)

    num_subset_correct = (preds == targets).all(axis=1).sum()
    num_instances = preds.shape[0]

    hamming_acc = 1 - (cls_fp.sum() + cls_fn.sum()) / (num_classes * num_instances)
    subset_acc = num_subset_correct / num_instances

    # print(f'cls_tp: {cls_tp.sum()}, cls_fp: {cls_fp.sum()}, cls_fn: {cls_fn.sum()}')

    def safe_divide(a, b):
        with np.errstate(divide='ignore', invalid='ignore'):
            c = np.true_divide(a, b)
            if np.isscalar(c):
                if np.isnan(c):  # 0 / 0
                    return 1.
            else:
                c[np.isnan(c)] = 1.
        return c
    
    micro_f1 = safe_divide(2 * cls_tp.sum(), (2 * cls_tp.sum() + cls_fp.sum() + cls_fn.sum()))
    macro_f1 = safe_divide(2 * cls_tp, 2 * cls_tp + cls_fp + cls_fn).mean()
    instance_f1 = safe_divide(2 * ins_tp, 2 * ins_tp + ins_fp + ins_fn).mean()

    metrics = {
        'hamming_accuracy': np.round(hamming_acc, 7),
        'subset_accuracy': np.round(subset_acc, 7),
        'instance_f1': np.round(instance_f1, 7),
        'micro_f1': np.round(micro_f1, 7),
        'macro_f1': np.round(macro_f1, 7),
    }
    return metrics

