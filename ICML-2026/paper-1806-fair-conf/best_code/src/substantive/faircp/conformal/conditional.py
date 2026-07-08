from substantive.faircp.conformity.utils import (
    calculate_metrics,
    compute_nonconformity_score,
    get_conformal_set,
)
from substantive.faircp.structs.conformal_input import ConditionalConformalInput
from substantive.faircp.structs.conformal_result import ConformalResult
from substantive.faircp.conformity.hpo import run_hpo_conformal


def conditional_conformal_prediction(
    input: ConditionalConformalInput,
) -> ConformalResult:
    (
        cfg,
        logits_test,
        targets_test,
        logits_calib,
        targets_calib,
        used_labels,
        logits_val,
        targets_val,
        groups_test,
        groups_calib,
        groups_val,
        label_map,
        group_map,
        k,
        dataset_group_conformal_category,
    ) = (
        input.cfg,
        input.logits_test,
        input.targets_test,
        input.logits_calib,
        input.targets_calib,
        input.used_labels,
        input.logits_val,
        input.targets_val,
        input.groups_test,
        input.groups_calib,
        input.groups_val,
        input.label_map,
        input.group_map,
        input.k,
        input.dataset_group_conformal_category,
    )

    h_params_cond = cfg["h_params_conformal"]
    if cfg["hpo_iterations"] > 0:
        h_params_cond = run_hpo_conformal(
            logits_calib,
            targets_calib,
            logits_val,
            targets_val,
            used_labels,
            h_params_cond,
            cfg,
            conformal_category=dataset_group_conformal_category,
            bins_calib=groups_calib.numpy(),
            bins_test=groups_val.numpy(),
        )
    print(f"Best hyperparams for Conditional: {h_params_cond}")

    # Compute non conformity score for each class
    non_conf_scores_cond_calib, non_conf_scores_cond_test = compute_nonconformity_score(
        logits_calib, targets_calib, logits_test, h_params_cond, cfg
    )

    # Get conditional conformal sets for test set
    conformal_preds_cond_test = get_conformal_set(
        non_conf_scores_cond_calib,
        non_conf_scores_cond_test,
        labels=used_labels,
        confidence=1 - cfg["alpha"],
        conformal_category=dataset_group_conformal_category,
        bins_cal=groups_calib.numpy(),
        bins_test=groups_test.numpy(),
    )

    metrics_cond = calculate_metrics(
        logits_test,
        targets_test,
        conformal_preds_cond_test,
        k=k,
        group=groups_test,
        label_map=label_map,
        group_map=group_map,
    )

    return ConformalResult(predictions=conformal_preds_cond_test, metrics=metrics_cond)
