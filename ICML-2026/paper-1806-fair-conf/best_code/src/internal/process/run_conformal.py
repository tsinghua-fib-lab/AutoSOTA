import time

import torch

from pathlib import Path

from typing import Any

from internal.dataset.dataloader_factory import get_loaders
from internal.dataset.dataset_utils import (
    check_dataset_balance,
    data_prep_to_generate_csv,
    format_and_write_to_csv,
)
from internal.model.model_factory import get_model
from internal.model.model_runner import get_logits_targets_groups
from internal.util.data_adapter import conformal_data_frame_to_fairness_input
from internal.util.metrics_processing import (
    process_avg_k_metric,
    process_backward_metric,
    process_conditional_metric,
    process_marginal_metric,
    process_clustered_metric,
)
from internal.util.writer import Writer, get_writer
from substantive.faircp.conformal.average_k import average_k_conformal_prediction
from substantive.faircp.conformal.backward import backward_conformal_prediction
from substantive.faircp.conformal.conditional import conditional_conformal_prediction
from substantive.faircp.conformal.marginal import marginal_conformal_prediction
from substantive.faircp.conformal.clustered import (
    clustered_label_conformal_prediction,
    clustered_group_conformal_prediction,
)

from substantive.faircp.conformal.setup import set_seed
from substantive.faircp.conformal.topk import top_k
from substantive.faircp.structs.conformal_input import (
    AverageKConformalInput,
    ConditionalConformalInput,
    ConformalInput,
    ClusteredLabelConformalInput,
    ClusteredGroupConformalInput,
)
from substantive.faircp.structs.fairness_input import FairnessInput

def _checkpoint_model(writer: Writer, model: Any, name: str = "model") -> None:
    
    # 1) PyTorch / torch.nn.Module
    if hasattr(model, "state_dict") and callable(getattr(model, "state_dict")):
        writer.write_checkpoint(name, model.state_dict())
        return

    # Infer a directory owned by the writer (best-effort; falls back to cwd)
    out_dir = (
        getattr(writer, "output_dir", None)
        or getattr(writer, "log_dir", None)
        or getattr(writer, "save_dir", None)
        or "."
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2) XGBoost sklearn API / Booster API
    if hasattr(model, "save_model") and callable(getattr(model, "save_model")):
        model_path = out_dir / f"{name}.xgb.json"
        model.save_model(str(model_path))
        return

    # 3) Fallback: joblib (or pickle)
    try:
        import joblib  
        joblib.dump(model, out_dir / f"{name}.joblib")
    except Exception:
        import pickle
        with open(out_dir / f"{name}.pkl", "wb") as f:
            pickle.dump(model, f)

def run_conformal(cfg: dict) -> tuple[Writer, FairnessInput]:
    dataset_name = cfg["dataset"]
    writer = get_writer(dataset_name, cfg=cfg)

    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    start_time = time.time()

    # Get specified dataset in the form of loaders
    dataset_class, loader_dict = get_loaders(cfg)

    # Get model specific for each dataset, trained from scratch or loaded from saved weights
    used_labels = (
        loader_dict["top_m_labels"]
        if dataset_class.uses_top_m_labels
        else [i for i in range(cfg["m"])]
    )

    model = get_model(
        cfg,
        device,
        dataset_class,
        loader_dict["train"],
        loader_dict["val"],
        used_labels,
    )

    #if cfg["save_model_ckpt"] is not None and cfg["model_checkpoint"] is None:
        #writer.write_checkpoint("model", model.state_dict())

    if cfg["save_model_ckpt"] is not None and cfg["model_checkpoint"] is None:
        _checkpoint_model(writer, model, name="model")

    label_map = dataset_class.get_id2label(return_dict=True)
    group_map = dataset_class.get_id2group(return_dict=True)

    run_model_inputs = RunModelInputs(
        dataset_class,
        model,
        device,
        writer,
        label_map,
        group_map,
    )

    logits_calib, targets_calib, groups_calib, _ = get_tensors_and_check_balance(
        run_model_inputs, loader_dict["calib"], "calib"
    )

    logits_test, targets_test, groups_test, input_identifiers_test = (
        get_tensors_and_check_balance(run_model_inputs, loader_dict["test"], "test")
    )

    if "calib_val" not in loader_dict:
        logits_val = logits_test
        targets_val = targets_test
        groups_val = groups_test
    else:
        logits_val, targets_val, groups_val, _ = get_tensors_and_check_balance(
            run_model_inputs, loader_dict["calib_val"], "val"
        )

    k = cfg["k"]
    print(f"Using alpha {cfg['alpha']:.4f}")

    conformal_input = ConformalInput(
        cfg=cfg,
        logits_test=logits_test,
        targets_test=targets_test,
        logits_calib=logits_calib,
        targets_calib=targets_calib,
        logits_val=logits_val,
        targets_val=targets_val,
        used_labels=used_labels,
        groups_test=groups_test,
        label_map=label_map,
        group_map=group_map,
        k=k,
    )
    marginal_result = marginal_conformal_prediction(conformal_input)
    marginal_size = process_marginal_metric(marginal_result.metrics, writer, cfg)

    conditional_input = ConditionalConformalInput(
        cfg=cfg,
        logits_test=logits_test,
        targets_test=targets_test,
        logits_calib=logits_calib,
        targets_calib=targets_calib,
        logits_val=logits_val,
        targets_val=targets_val,
        used_labels=used_labels,
        groups_test=groups_test,
        groups_calib=groups_calib,
        groups_val=groups_val,
        label_map=label_map,
        group_map=group_map,
        k=k,
        dataset_group_conformal_category=dataset_class.group_conformal_category,
    )
    conditional_result = conditional_conformal_prediction(conditional_input)
    process_conditional_metric(conditional_result.metrics, writer, cfg)

    avg_k_input = AverageKConformalInput(
        cfg=cfg,
        logits_test=logits_test,
        targets_test=targets_test,
        logits_calib=logits_calib,
        targets_calib=targets_calib,
        logits_val=logits_val,
        targets_val=targets_val,
        groups_test=groups_test,
        used_labels=used_labels,
        label_map=label_map,
        group_map=group_map,
        k=k,
        marginal_size=marginal_size,
    )
    backward_result = backward_conformal_prediction(avg_k_input)
    process_backward_metric(backward_result.metrics, writer, cfg)

    clustered_label_input = ClusteredLabelConformalInput(
        cfg=cfg,
        logits_test=logits_test,
        targets_test=targets_test,
        logits_calib=logits_calib,
        targets_calib=targets_calib,
        logits_val=logits_val,
        targets_val=targets_val,
        used_labels=used_labels,
        groups_test=groups_test,
        label_map=label_map,
        group_map=group_map,
        k=k,
    )
    clustered_label_result = clustered_label_conformal_prediction(clustered_label_input)
    process_clustered_metric(clustered_label_result.metrics, writer, cfg, "clustered_label")

    clustered_group_input = ClusteredGroupConformalInput(
        cfg=cfg,
        logits_test=logits_test,
        targets_test=targets_test,
        logits_calib=logits_calib,
        targets_calib=targets_calib,
        logits_val=logits_val,
        targets_val=targets_val,
        used_labels=used_labels,
        groups_test=groups_test,
        groups_calib=groups_calib,
        groups_val=groups_val,
        label_map=label_map,
        group_map=group_map,
        k=k,
    )
    clustered_group_result = clustered_group_conformal_prediction(clustered_group_input)
    process_clustered_metric(clustered_group_result.metrics, writer, cfg, "clustered_group")


    print(
        f"total time to run {cfg['dataset']} dataset : {time.time() - start_time:.2f} s"
    )

    ### Format data for csv
    print("Formatting data for CSV output")
    df = data_prep_to_generate_csv(
        marginal_result.predictions,
        conditional_result.predictions,
        backward_result.predictions,
        clustered_label_result.predictions,
        clustered_group_result.predictions,
        k=k,
        input_identifiers=input_identifiers_test,
        group_label=groups_test.numpy(),
        y=targets_test,
    )

    df = dataset_class.process_dataframe(df, loader_dict, k=k)

    fairness_input = conformal_data_frame_to_fairness_input(df, label_map, group_map)

    print("Preparing to save data to csv")
    format_and_write_to_csv(df, writer, dataset_name, cfg)

    return writer, fairness_input


class RunModelInputs:
    def __init__(
        self,
        dataset_class,
        model,
        device,
        writer,
        label_map,
        group_map,
    ):
        self.dataset_class = dataset_class
        self.model = model
        self.device = device
        self.writer = writer
        self.label_map = label_map
        self.group_map = group_map


def get_tensors_and_check_balance(input: RunModelInputs, data_loader, fold: str):
    logits, targets, groups, input_identifiers = get_logits_targets_groups(
        input.dataset_class, data_loader, input.model, input.device
    )
    check_dataset_balance(
        input.writer,
        fold,
        targets,
        groups,
        label_map=input.label_map,
        group_map=input.group_map,
    )

    return (logits, targets, groups, input_identifiers)
