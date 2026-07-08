import torch
import numpy as np
from crepes import ConformalClassifier

from substantive.faircp.calibration.calibration_methods import get_calib
from substantive.faircp.conformity.score_functions import get_score_fn
from substantive.faircp.structs.enums import (
    ConformalCategory,
    ScoreFunctionType,
    CalibrationType,
)
from torchmetrics.classification import MulticlassCalibrationError, MulticlassStatScores


def compute_nonconformity_score(
    calib_logits: torch.Tensor,
    calib_targets: torch.Tensor,
    test_logits: torch.Tensor,
    h_params: dict,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes non-conformity scores for conformal classifiers
    """
    score_fn_type = ScoreFunctionType.from_str(cfg["score_fn"])

    calib_method = get_calib(CalibrationType.TEMPERATURE)
    score_fn = get_score_fn(score_fn_type)

    # Temperature scaling before softmax
    calib_probs = calib_method.calibrate(calib_logits, h_params)
    test_probs = calib_method.calibrate(test_logits, h_params)

    return score_fn.get_scores(calib_probs, calib_targets, test_probs, h_params)


def get_conformal_set(
    non_conformity_scores_calib,
    non_conformity_scores_test,
    labels,
    conformal_category,
    confidence=0.95,
    bins_cal=None,
    bins_test=None,
):
    """
    Compute the conformal set based on conformal category - marginal, class-conditional, or group-balanced
    """

    if conformal_category == ConformalCategory.MARGINAL:
        if bins_cal is not None:
            raise ValueError("Bins must be None for marginal conformal category")

        cc_marginal = ConformalClassifier()
        cc_marginal.fit(non_conformity_scores_calib)

        # predict conformal set
        prediction_set = cc_marginal.predict_set(
            non_conformity_scores_test, confidence=confidence
        )

    elif conformal_category == ConformalCategory.CLASS_CONDITIONAL:
        if bins_cal is None:
            raise ValueError(
                "Bins must be provided for class-conditional conformal category"
            )

        cc_class_cond = ConformalClassifier()
        cc_class_cond.fit(non_conformity_scores_calib, bins_cal)

        # Class labels are remapped from 0 to n_classes in `get_loader` method
        class_labels = torch.tensor([x for x in range(len(labels))])

        # predict conformal set
        prediction_set = np.array(
            [
                cc_class_cond.predict_set(
                    non_conformity_scores_test,
                    np.full(len(non_conformity_scores_test), class_labels[c]),
                    confidence=confidence,
                )[:, c]
                for c in range(len(class_labels))
            ]
        ).T

    elif conformal_category == ConformalCategory.GROUP_BALANCED:
        if bins_cal is None or bins_test is None:
            raise ValueError(
                "Both calib and test bins must be provided for group-balanced conformal category"
            )

        cc_group_cond = ConformalClassifier()
        cc_group_cond.fit(non_conformity_scores_calib, bins_cal)

        prediction_set = cc_group_cond.predict_set(
            non_conformity_scores_test, bins_test, confidence=confidence
        )

    # convert the prediction set to labels
    prediction_labels = [
        np.array([idx for idx, val in enumerate(row) if val == 1])
        for row in prediction_set
    ]

    # avoid zero set sizes. Choose argmin from test_non_conformity_scores if no elements in the prediction_labels
    for i in range(len(prediction_labels)):
        if len(prediction_labels[i]) == 0:
            prediction_labels[i] = np.array([np.argmin(non_conformity_scores_test[i])])

    return prediction_labels


def calculate_metrics(
    logits,
    targets,
    prediction_sets,
    k=3,
    group=None,
    compute_detailed_accs=False,
    label_map=None,
    group_map=None,
):
    """
    Compute the metrics
    """
    prec_1, prec_k = accuracy(logits, targets, topk=(1, k))
    cvg, sz = coverage_size(prediction_sets, targets)
    ece = calibration_error(logits, targets)
    tp, fp, tn, fn, _ = classification_scores(logits, targets)

    metrics = {
        "top1": round(prec_1.item() / 100.0, 4),
        "topk": round(prec_k.item() / 100.0, 4),
        "tpr": round(tp.item() / (tp.item() + fn.item()), 4),
        "fpr": round(fp.item() / (fp.item() + tn.item()), 4),
        "ece": round(ece, 4),
        "coverage": cvg,
        "size": sz,
    }

    if label_map:
        unique_labels = torch.unique(targets)
        if compute_detailed_accs:
            label_accs = {}
            for label in unique_labels:
                label_mask = targets == label
                label_prec_1 = accuracy(
                    logits[label_mask], targets[label_mask], topk=(1,)
                )[0]
                label_accs[label_map[label.item()]] = round(
                    label_prec_1.item() / 100.0, 4
                )
            metrics["top1_acc_per_label"] = label_accs

        label_covs = {}
        label_sizes = {}
        for label in unique_labels:
            label_mask = targets == label
            filtered_prediction_sets = [
                pred_set
                for pred_set, mask_value in zip(prediction_sets, label_mask)
                if mask_value
            ]
            cvg, sz = coverage_size(filtered_prediction_sets, targets[label_mask])
            label_covs[label_map[label.item()]] = cvg
            label_sizes[label_map[label.item()]] = sz
        metrics["cov_per_label"] = label_covs
        metrics["size_per_label"] = label_sizes

    if (group is not None) and group_map:
        unique_groups = torch.unique(group)
        if compute_detailed_accs:
            group_accs = {}
            for grp in unique_groups:
                group_mask = group == grp
                group_prec_1 = accuracy(
                    logits[group_mask], targets[group_mask], topk=(1,)
                )[0]
                group_accs[group_map[grp.item()]] = round(
                    group_prec_1.item() / 100.0, 4
                )
            metrics["top1_acc_per_group"] = group_accs
            metrics["disparate_impact_acc"] = max(group_accs.values()) - min(
                group_accs.values()
            )

        group_covs = {}
        group_sizes = {}
        for grp in unique_groups:
            group_mask = group == grp
            filtered_prediction_sets = [
                pred_set
                for pred_set, mask_value in zip(prediction_sets, group_mask)
                if mask_value
            ]
            cvg, sz = coverage_size(filtered_prediction_sets, targets[group_mask])
            group_covs[group_map[grp.item()]] = cvg
            group_sizes[group_map[grp.item()]] = sz
        metrics["cov_per_group"] = group_covs
        metrics["size_per_group"] = group_sizes
        metrics["disparate_impact_cov"] = max(group_covs.values()) - min(
            group_covs.values()
        )
        metrics["disparate_impact_size"] = max(group_sizes.values()) - min(
            group_sizes.values()
        )

    return metrics


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum()
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def coverage_size(S, targets):
    covered = 0
    size = 0
    for i in range(targets.shape[0]):
        if targets[i].item() in S[i]:
            covered += 1
        size = size + S[i].shape[0]
    return float(covered) / targets.shape[0], size / targets.shape[0]


def calibration_error(logits_calib, targets_calib, n_bins=10, norm="l1"):
    """
    Computes the top-label multiclass expected calibration error for the specified number of bins `n_bins`
    logits_calib (Tensor): A float tensor of shape (N, C, ...) containing logits for each observation.
    targets_calib (Tensor): An int tensor of shape (N, ...) containing ground truth labels, and therefore only contain values
    in the [0, n_classes-1] range.
    """

    num_classes = logits_calib.size()[1]

    compute_calib_error = MulticlassCalibrationError(
        num_classes=num_classes, n_bins=n_bins, norm=norm
    )
    calib_probs = torch.softmax(logits_calib, dim=1)

    ece = compute_calib_error(calib_probs, targets_calib.reshape([-1]))

    return ece.item()


def classification_scores(logits, targets, top_k=1, average="micro"):
    """Computes a tensor of shape (..., 5), where the last dimension corresponds to [tp, fp, tn, fn, sup]
    (sup stands for support and equals tp + fn).
    N.B: specify average='micro'/'macro' for overall metrics and average=None for per class metrics
    """

    num_classes = logits.size()[1]

    metric = MulticlassStatScores(num_classes=num_classes, top_k=top_k, average=average)
    mcss = metric(logits, targets.reshape([-1]))

    return mcss
