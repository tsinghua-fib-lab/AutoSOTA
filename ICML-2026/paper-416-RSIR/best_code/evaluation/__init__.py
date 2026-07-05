import sys
from typing import *

import torch
import torch.nn.functional as F
import torchmetrics.functional as M


def recall(pred, target, k_or_thres, mean=True):
    r"""Calculating recall.

    Recall value is defined as below:

    .. math::
        Recall= \frac{TP}{TP+FN}

    Args:
        pred(torch.BoolTensor): [B, num_items] or [B]. The prediction result of the model with bool type values.
            If the value in the j-th column is `True`, the j-th highest item predicted by model is right.

        target(torch.FloatTensor): [B, num_target] or [B]. The ground truth.

    Returns:
        torch.FloatTensor: a 0-dimensional tensor.
    """
    if pred.dim() > 1:
        k = k_or_thres
        count = (target > 0).sum(-1)
        output = pred[:, :k].sum(dim=-1).float() / count
        if mean:
            return output.mean()
        else:
            return output
    else:
        thres = k_or_thres
        return M.recall(pred, target, task='binary', threshold=thres)


def precision(pred, target, k_or_thres, mean=True):
    r"""Calculate the precision.

    Precision are defined as:

    .. math::
        Precision = \frac{TP}{TP+FP}

    Args:
        pred(torch.BoolTensor): [B, num_items] or [B]. The prediction result of the model with bool type values.
            If the value in the j-th column is `True`, the j-th highest item predicted by model is right.

        target(torch.FloatTensor): [B, num_target] or [B]. The ground truth.

    Returns:
        torch.FloatTensor: a 0-dimensional tensor.
    """
    if pred.dim() > 1:
        k = k_or_thres
        output = pred[:, :k].sum(dim=-1).float() / k
        if mean:
            return output.mean()
        else:
            return output
    else:
        thres = k_or_thres
        if mean:
            return M.precision(pred, target, task='binary', threshold=thres)
        else:
            return M.precision(pred, target, task='binary', threshold=thres)


def f1(pred, target, k_or_thres, mean=True):
    r"""Calculate the F1.

    Args:
        pred(torch.BoolTensor): [B, num_items] or [B]. The prediction result of the model with bool type values.
            If the value in the j-th column is `True`, the j-th highest item predicted by model is right.

        target(torch.FloatTensor): [B, num_target] or [B]. The ground truth.

    Returns:
        torch.FloatTensor: a 0-dimensional tensor.
    """
    if pred.dim() > 1:
        k = k_or_thres
        count = (target > 0).sum(-1)
        output = 2 * pred[:, :k].sum(dim=-1).float() / (count + k)
        if mean:
            return output.mean()
        else:
            return output
    else:
        thres = k_or_thres
        if mean:
            return M.f1_score(pred, target, task='binary', threshold=thres)
        else:
            return M.f1_score(pred, target, task='binary', threshold=thres)


def map(pred, target, k):
    r"""Calculate the mean Average Precision(mAP).

    Args:
        pred(torch.BoolTensor): [B, num_items]. The prediction result of the model with bool type values.
            If the value in the j-th column is `True`, the j-th highest item predicted by model is right.

        target(torch.FloatTensor): [B, num_target]. The ground truth.

    Returns:
        torch.FloatTensor: a 0-dimensional tensor.
    """
    count = (target > 0).sum(-1)
    pred = pred[:, :k].float()
    output = pred.cumsum(dim=-1) / torch.arange(1, k+1).type_as(pred)
    output = (output * pred).sum(dim=-1) / \
        torch.minimum(count, k*torch.ones_like(count))
    return output.mean()


def _dcg(pred, k):
    k = min(k, pred.size(1))
    denom = torch.log2(torch.arange(k).type_as(pred) + 2.0).view(1, -1)
    return (pred[:, :k] / denom).sum(dim=-1)


def ndcg(pred, target, k, mean=True):
    r"""Calculate the Normalized Discounted Cumulative Gain(NDCG).

    Args:
        pred(torch.BoolTensor): [B, num_items]. The prediction result of the model with bool type values.
            If the value in the j-th column is `True`, the j-th highest item predicted by model is right.

        target(torch.FloatTensor): [B, num_target]. The ground truth.

    Returns:
        torch.FloatTensor: a 0-dimensional tensor.
    """
    pred_dcg = _dcg(pred.float(), k)
    #TODO replace target>0 with target
    ideal_dcg = _dcg(torch.sort((target > 0).float(), descending=True)[0], k)
    all_irrelevant = torch.all(target <= sys.float_info.epsilon, dim=-1)
    pred_dcg[all_irrelevant] = 0
    pred_dcg[~all_irrelevant] /= ideal_dcg[~all_irrelevant]
    if mean:
        return pred_dcg.mean()
    else:
        return pred_dcg


def mrr(pred, target, k, mean=True):
    r"""Calculate the Mean Reciprocal Rank(MRR).

    Args:
        pred(torch.BoolTensor): [B, num_items]. The prediction result of the model with bool type values.
            If the value in the j-th column is `True`, the j-th highest item predicted by model is right.

        target(torch.FloatTensor): [B, num_target]. The ground truth.

    Returns:
        torch.FloatTensor: a 0-dimensional tensor.
    """
    row, col = torch.nonzero(pred[:, :k], as_tuple=True)
    row_uniq, counts = torch.unique_consecutive(row, return_counts=True)
    idx = torch.zeros_like(counts)
    idx[1:] = counts.cumsum(dim=-1)[:-1]
    first = col.new_zeros(pred.size(0)).scatter_(0, row_uniq, col[idx]+1)
    output = 1.0 / first
    output[first == 0] = 0
    if mean:
        return output.mean()
    else:
        return output


def hits(pred, target, k):
    r"""Calculate the Hits.

    Args:
        pred(torch.BoolTensor): [B, num_items]. The prediction result of the model with bool type values.
            If the value in the j-th column is `True`, the j-th highest item predicted by model is right.

        target(torch.FloatTensor): [B, num_target]. The ground truth.

    Returns:
        torch.FloatTensor: a 0-dimensional tensor.
    """
    return torch.any(pred[:, :k] > 0, dim=-1).float().mean()


def logloss(pred, target):
    r"""Calculate the log loss (log cross entropy).

    Args:
        pred(torch.BoolTensor): [B, num_items]. The prediction result of the model with bool type values.
            If the value in the j-th column is `True`, the j-th highest item predicted by model is right.

        target(torch.FloatTensor): [B, num_target]. The ground truth.

    Returns:
        torch.FloatTensor: a 0-dimensional tensor.
    """
    if pred.dim() == target.dim():
        return F.binary_cross_entropy_with_logits(pred, target.float())
    else:
        return F.cross_entropy(pred, target)


def auc(pred, target):
    r"""Calculate the global AUC.

    Args:
        pred(torch.BoolTensor): [B, num_items]. The prediction result of the model with bool type values.
            If the value in the j-th column is `True`, the j-th highest item predicted by model is right.

        target(torch.FloatTensor): [B, num_target]. The ground truth.

    Returns:
        torch.FloatTensor: a 0-dimensional tensor.
    """
    target = target.type(torch.long)
    return M.auroc(pred, target, task='binary')


def accuracy(pred, target, thres=0.5):
    r"""Calculate the accuracy.

    Args:
        pred(torch.BoolTensor): [Batch_size]. The prediction result of the model with bool type values.
            If the value in the j-th column is `True`, the j-th highest item predicted by model is right.

        target(torch.FloatTensor): [Batch_size]. The ground truth.

        thres(float): Predictions below the thres will be marked as 0, otherwise 1.

    Returns:
        torch.FloatTensor: a 0-dimensional tensor.
    """
    return M.accuracy(pred, target, task='binary', threshold=thres)


def mse(pred, target):
    """Calculate Meas Square Error"""
    return M.mean_squared_error(pred, target)


def mae(pred, target):
    """Calculate Mean Absolute Error."""
    return M.mean_absolute_error(pred, target)


metric_dict = {
    'ndcg': ndcg,
    'precision': precision,
    'recall': recall,
    'map': map,
    'hit': hits,
    'mrr': mrr,
    'f1': f1,
    'mse': mse,
    'mae': mae,
    'auc': auc,
    'logloss': logloss,
    'accuracy': accuracy
}


def get_rank_metrics(metric):
    if not isinstance(metric, list):
        metric = [metric]
    topk_metrics = {'ndcg', 'precision', 'recall', 'map', 'mrr', 'hit', 'f1'}
    rank_m = [(m, metric_dict[m])
              for m in metric if m in topk_metrics and m in metric_dict]
    return rank_m


def get_pred_metrics(metric):
    if not isinstance(metric, list):
        metric = [metric]
    pred_metrics = {'mae', 'mse', 'auc', 'logloss', 'accuracy',
                    'precision', 'recall', 'f1'}
    pred_m = [(m, metric_dict[m])
              for m in metric if m in pred_metrics and m in metric_dict]
    return pred_m


def get_global_metrics(metric):
    if (not isinstance(metric, list)) and (not isinstance(metric, dict)):
        metric = [metric]
    global_metrics = {"auc"}
    global_m = [(m, metric_dict[m])
                for m in metric if m in global_metrics and m in metric_dict]
    return global_m


def get_eval_metrics(metric_names: Union[List[str], str], cutoffs: Union[List[int], int], validation: bool=False) -> List[str]:
    r""" Get metrics with cutoff for evaluation.

    Args:
        metrics_names(Union[List[str], str]): names of metrics which requires cutoff. Such as ["ndcg", "recall"].
        cutoffs(Union[List[int], int]): cutoffs for those metric. Such as [5,10].
        validation(bool): if validation, only the first element of cutoffs would be used.

    Returns:
        List[str]: metrics with cutoffs, such as ["ndcg@5", "ndcg@10", "recall@5", "recall@10"].
    """
    metric_names = metric_names if isinstance(metric_names, list) else [metric_names]
    rank_metrics = [m[0] for m in get_rank_metrics(metric_names)]
    pred_metrics = [m[0] for m in get_pred_metrics(metric_names)]
    if cutoffs is not None:
        cutoffs = cutoffs if isinstance(cutoffs, list) else [cutoffs]
        if validation:
            cutoffs = cutoffs[: 1]
        res = [f"{m}@{cut}" if m in rank_metrics else m for cut in cutoffs for m in metric_names]
    else:
        res = pred_metrics
    return res