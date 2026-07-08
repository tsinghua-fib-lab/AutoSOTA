from substantive.faircp.conformity.utils import (
    calculate_metrics,
    compute_nonconformity_score,
    get_conformal_set,
)
from substantive.faircp.structs.conformal_input import ConformalInput
from substantive.faircp.structs.conformal_result import ConformalResult
from substantive.faircp.structs.enums import ConformalCategory
from substantive.faircp.conformity.hpo import run_hpo_conformal


def marginal_conformal_prediction(
    input: ConformalInput,
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
        label_map,
        group_map,
        k,
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
        input.label_map,
        input.group_map,
        input.k,
    )

    h_params_marg = cfg["h_params_conformal"]
    if cfg["hpo_iterations"] > 0:
        h_params_marg = run_hpo_conformal(
            logits_calib,
            targets_calib,
            logits_val,
            targets_val,
            used_labels,
            h_params_marg,
            cfg,
            conformal_category=ConformalCategory.MARGINAL,
        )
    print(f"Best hyperparams for Marginal: {h_params_marg}")
    # Compute non conformity score for each class
    non_conf_scores_marg_calib, non_conf_scores_marg_test = compute_nonconformity_score(
        logits_calib, targets_calib, logits_test, h_params_marg, cfg
    )
    print(f"non_conf_scores_marg_calib: {non_conf_scores_marg_calib.shape}")
    print(f"non_conf_scores_marg_test: {non_conf_scores_marg_test.shape}")
    # Get marginal conformal sets for test set
    conformal_preds_marg_test = get_conformal_set(
        non_conf_scores_marg_calib,
        non_conf_scores_marg_test,
        labels=used_labels,
        confidence=1 - cfg["alpha"],
        conformal_category=ConformalCategory.MARGINAL,
    )

    metrics_marg = calculate_metrics(
        logits_test,
        targets_test,
        conformal_preds_marg_test,
        k=k,
        group=groups_test,
        compute_detailed_accs=True,
        label_map=label_map,
        group_map=group_map,
    )

    return ConformalResult(
        metrics=metrics_marg,
        predictions=conformal_preds_marg_test,
    )
