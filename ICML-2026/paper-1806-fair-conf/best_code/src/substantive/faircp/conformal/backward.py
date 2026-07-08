import math
import numpy as np
import torch
from substantive.faircp.conformity.utils import calculate_metrics
from substantive.faircp.structs.conformal_input import AverageKConformalInput
from substantive.faircp.structs.conformal_result import ConformalResult


def backward_conformal_prediction(input: AverageKConformalInput) -> ConformalResult:
    cfg = input.cfg
    iteration = 0

    while iteration < 10:
        result = backward_conformal_prediction_internal(input)

        if result.metrics["coverage"] > 1 - cfg["alpha"]:
            return result

        cfg["back_cp_params"]["max_set_size_offset"] += 1
        iteration += 1

    return result


def backward_conformal_prediction_internal(
    input: AverageKConformalInput,
) -> ConformalResult:
    (
        cfg,
        logits_test,
        targets_test,
        logits_calib,
        targets_calib,
        # logits_val,
        # targets_val,
        groups_test,
        label_map,
        group_map,
        k,
        marginal_size,
    ) = (
        input.cfg,
        input.logits_test,
        input.targets_test,
        input.logits_calib,
        input.targets_calib,
        # input.logits_val,
        # input.targets_val,
        input.groups_test,
        input.label_map,
        input.group_map,
        input.k,
        input.marginal_size,
    )

    back_cp_params = cfg["back_cp_params"]
    epsilon = back_cp_params["epsilon"]
    max_set_size = math.ceil(marginal_size) + back_cp_params["max_set_size_offset"]
    tolerance = back_cp_params["tolerance"]

    # Convert logits to probabilities
    calib_probs = torch.softmax(logits_calib, dim=1).numpy()  # [n_calib, num_classes]
    test_probs = torch.softmax(logits_test, dim=1).numpy()  # [n_test, num_classes]
    calib_targets = targets_calib.numpy()  # [n_calib]
    num_classes = len(label_map)
    n_calib = len(calib_targets)
    n_test = test_probs.shape[0]

    calib_scores = np.array(
        [-np.log(calib_probs[i, calib_targets[i]] + epsilon) for i in range(n_calib)]
    )  # [n_calib]
    sum_calib = np.sum(calib_scores)  # Scalar sum for ratios

    conformal_sets = np.zeros((n_test, num_classes), dtype=np.int32)  # Binary mask
    # alpha_mins = np.zeros(n_test)  # Per-test alpha

    # list_alpha_i = []

    for j in range(n_test):
        test_scores = np.array(
            [-np.log(test_probs[j, k] + epsilon) for k in range(num_classes)]
        )  # [num_classes]

        rs = (n_calib + 1) * test_scores / (sum_calib + test_scores)  # [num_classes]
        moving_max_set_size = max_set_size
        non_zero_found = False

        while not non_zero_found:
            left, right = epsilon, 1.0 - epsilon
            while right - left > tolerance:
                alpha = (left + right) / 2
                threshold = 1.0 / alpha
                set_size = np.sum(rs < threshold)

                if set_size <= moving_max_set_size:
                    right = alpha  # Try smaller alpha
                else:
                    left = alpha  # Try larger alpha

            alpha_min = (left + right) / 2
            threshold = 1.0 / alpha_min
            set_size = np.sum(rs < threshold)

            if set_size > 0:
                non_zero_found = True
            else:
                moving_max_set_size += 1  # Expand allowable size and retry

        conformal_sets[j] = (rs < threshold).astype(np.int32)

    prediction_labels = [
        np.array([idx for idx, val in enumerate(row) if val == 1])
        for row in conformal_sets
    ]

    metrics_back = calculate_metrics(
        logits_test,
        targets_test,
        prediction_labels,
        k=k,
        group=groups_test,
        compute_detailed_accs=True,
        label_map=label_map,
        group_map=group_map,
    )
    # metrics_back["loo_coverage"] = 1 - alpha_loo
    # metrics_back["avg_test_cvg"] = avg_test_cvg
    # metrics_back["empirical_calib_coverage"] = empirical_calib_coverage

    return ConformalResult(metrics=metrics_back, predictions=prediction_labels)
