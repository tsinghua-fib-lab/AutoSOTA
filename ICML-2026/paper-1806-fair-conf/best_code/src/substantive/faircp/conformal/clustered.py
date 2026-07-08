import numpy as np
from substantive.faircp.conformity.utils import calculate_metrics, compute_nonconformity_score
from substantive.faircp.structs.conformal_result import ConformalResult
from substantive.faircp.conformity.hpo import run_hpo_conformal
from substantive.faircp.structs.enums import ConformalCategory
from .clustered_cp import ClusterConfig, clustered_cp_class, clustered_cp_group
from substantive.faircp.structs.conformal_input import ClusteredLabelConformalInput, ClusteredGroupConformalInput
from internal.util.writer import Writer, get_writer

def _get_cluster_config_label(cfg: dict) -> ClusterConfig:
    cc = cfg.get("clustered_cp", {})
    return ClusterConfig(
        n_clusters=cc.get("M_label", 3),
        min_points_per_key=cc.get("min_points_per_key", 10),
        clustering_ratio=cc.get("gamma_label", 0.5),
        random_state=cc.get("random_state", 42),
        embedding_mode=cc.get("embedding_mode", "upper_percentiles"),
        summary_bins=cc.get("summary_bins", 50)
    )

def _get_cluster_config_group(cfg: dict) -> ClusterConfig:
    cc = cfg.get("clustered_cp", {})
    return ClusterConfig(
        n_clusters=cc.get("M_group", 3),
        min_points_per_key=cc.get("min_points_per_key", 10),
        clustering_ratio=cc.get("gamma_group", 0.5),
        random_state=cc.get("random_state", 42),
        embedding_mode=cc.get("embedding_mode", "upper_percentiles"),
        summary_bins=cc.get("summary_bins", 50)
    )


def clustered_label_conformal_prediction(input: ClusteredLabelConformalInput) -> ConformalResult:
    cfg = input.cfg
    alpha = cfg["alpha"]
    cluster_cfg = _get_cluster_config_label(cfg)
    dataset_name = cfg["dataset"]
    writer = get_writer(dataset_name, cfg=cfg)

    h_params = cfg["h_params_conformal"]
    if cfg["hpo_iterations"] > 0:
        h_params = run_hpo_conformal(
            input.logits_calib, input.targets_calib, input.logits_val, input.targets_val,
            input.used_labels, h_params, cfg, ConformalCategory.MARGINAL
        )

    scores_calib, scores_test = compute_nonconformity_score(
        input.logits_calib, input.targets_calib, input.logits_test, h_params, cfg
    )

    if scores_calib.ndim == 2:
        true_scores_calib = scores_calib[np.arange(len(input.targets_calib)), input.targets_calib.numpy()]
    else:
        true_scores_calib = scores_calib

    pred_sets = clustered_cp_class(
        true_scores_calib, scores_test, input.targets_calib.numpy(), cluster_cfg, alpha,
        writer, input.label_map
    )

    metrics = calculate_metrics(
        input.logits_test, input.targets_test, pred_sets, k=input.k,
        group=input.groups_test, label_map=input.label_map, group_map=input.group_map
    )
    return ConformalResult(predictions=pred_sets, metrics=metrics)

def clustered_group_conformal_prediction(input: ClusteredGroupConformalInput) -> ConformalResult:
    cfg = input.cfg
    alpha = cfg["alpha"]
    cluster_cfg = _get_cluster_config_group(cfg)
    dataset_name = cfg["dataset"]
    writer = get_writer(dataset_name, cfg=cfg)

    h_params = cfg["h_params_conformal"]
    if cfg["hpo_iterations"] > 0:
        h_params = run_hpo_conformal(
            input.logits_calib, input.targets_calib, input.logits_val, input.targets_val,
            input.used_labels, h_params, cfg, ConformalCategory.GROUP_BALANCED,
            bins_calib=input.groups_calib.numpy(), bins_test=input.groups_val.numpy()
        )

    scores_calib, scores_test = compute_nonconformity_score(
        input.logits_calib, input.targets_calib, input.logits_test, h_params, cfg
    )

    if scores_calib.ndim == 2:
        true_scores_calib = scores_calib[np.arange(len(input.targets_calib)), input.targets_calib.numpy()]
    else:
        true_scores_calib = scores_calib

    pred_sets = clustered_cp_group(
        true_scores_calib, scores_test, input.groups_calib.numpy(), input.groups_test.numpy(),
        cluster_cfg, alpha,
        writer, input.label_map, input.group_map
    )

    metrics = calculate_metrics(
        input.logits_test, input.targets_test, pred_sets, k=input.k,
        group=input.groups_test, label_map=input.label_map, group_map=input.group_map
    )
    return ConformalResult(predictions=pred_sets, metrics=metrics)
