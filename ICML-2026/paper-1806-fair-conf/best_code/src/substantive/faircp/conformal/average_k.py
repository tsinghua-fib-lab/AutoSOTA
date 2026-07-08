from substantive.faircp.conformity.average_k import get_average_k_sets
from substantive.faircp.conformity.utils import calculate_metrics, coverage_size
from substantive.faircp.structs.conformal_input import AverageKConformalInput
from substantive.faircp.structs.conformal_result import AvgKConformalResult


def average_k_conformal_prediction(
    input: AverageKConformalInput,
) -> AvgKConformalResult:
    (
        cfg,
        logits_test,
        targets_test,
        logits_calib,
        targets_calib,
        logits_val,
        targets_val,
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
        input.logits_val,
        input.targets_val,
        input.groups_test,
        input.label_map,
        input.group_map,
        input.k,
        input.marginal_size,
    )

    target_coverage = round(1 - cfg["alpha"], 3)
    print(
        f"Performing binary search to find k that matches target coverage of {target_coverage} on validation set"
    )
    # binary search to find the k that matches the target coverage, start with k_low = 0 and k_high = conformal marginal set size
    k_low = 0
    k_high = marginal_size
    k_avgk = marginal_size
    while k_low < k_high:
        # No tunable parameters for avg-k
        preds_avgk_val = get_average_k_sets(logits_calib, logits_val, k_avgk)
        cvg_avgk_val, size_avgk_val = coverage_size(preds_avgk_val, targets_val)
        coverage = round(cvg_avgk_val, 3)
        print(
            f"target k = {k_avgk}, val actual size = {size_avgk_val}, target coverage = {target_coverage},  val actual coverage = {coverage}"
        )
        if coverage < target_coverage:
            if k_high == k_avgk:
                k_avgk += 0.5
                k_high += 0.5
            else:
                k_low = k_avgk
                k_avgk = (k_high + k_low) / 2
            print(
                f"Increasing k_avgk to {k_avgk}, searching beween {k_low} and {k_high}"
            )
        if coverage > target_coverage:
            k_high = k_avgk
            k_avgk = (k_high + k_low) / 2
            print(
                f"Decreasing k_avgk to {k_avgk}, searching beween {k_low} and {k_high}"
            )

        if coverage == target_coverage or round(k_low, 5) == round(k_high, 5):
            # termination conditions:
            # when coverage is within 0.0001 of target coverage, we use the current k_avgk as the final k
            # when k_low and k_high are within 0.00001 of each other, we use the current k_avgk as the final k
            print(
                f"Found k={k_avgk} that matches target coverage of {target_coverage} on validation set"
            )
            preds_avgk_calib = get_average_k_sets(logits_calib, logits_calib, k_avgk)
            cvg_avgk_calib, size_avgk_calib = coverage_size(
                preds_avgk_calib, targets_calib
            )
            preds_avgk_test = get_average_k_sets(logits_calib, logits_test, k_avgk)
            metrics_avgk = calculate_metrics(
                logits_test,
                targets_test,
                preds_avgk_test,
                k=k,
                group=groups_test,
                label_map=label_map,
                group_map=group_map,
            )
            print(
                f"Empirical coverage of average-k prediction sets on the validation set: {cvg_avgk_val: .4f}"
            )
            print(
                f"Size of average-k prediction sets on the validation set: {size_avgk_val: .4f}"
            )

            print(
                f"Empirical coverage of average-k prediction sets on the calibration set: {cvg_avgk_calib: .4f}"
            )
            print(
                f"Size of average-k prediction sets on the calibration set: {size_avgk_calib: .4f}"
            )
            break

    return AvgKConformalResult(
        predictions=preds_avgk_test, metrics=metrics_avgk, k_avgk=k_avgk
    )
