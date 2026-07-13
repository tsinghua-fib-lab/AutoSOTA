import json
import os
import random
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "datasets"
DATA_ROOT.mkdir(parents=True, exist_ok=True)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
import yaml

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from tsl import logger
from tsl.data import SpatioTemporalDataset, SpatioTemporalDataModule, BatchMap, BatchMapItem
from tsl.data.datamodule.splitters import FixedIndicesSplitter
from tsl.data.preprocessing import StandardScaler
from tsl.datasets import MetrLA
import importlib
from tsl.experiment import Experiment

from conformal_model.scale.arch.config import QuantileModelConfig
from conformal_model.scale.arch.spec_decoupled_components import resolve_scale_model
from basicts.runners.scale_predictor import scalePredictor
from basicts.metrics.torch_metrics.coverage import (
    MaskedCoverage,
    MaskedDeltaCoverage,
    MaskedPIWidth,
    MaskedPIMedianWidth,
)
from basicts.metrics.torch_metrics.pinball_loss import MaskedMultiPinballLoss
from basicts.metrics.torch_metrics.winkler import MaskedWinklerScore
from basicts.utils.data_utils import parse_and_filter_indices
from basicts.utils.data_utils import find_close
from basicts.metrics.torch_metrics.wrappers import MaskedMetricWrapper
from basicts import printing
from basicts.runner import save_metrics, setup_file_logger, log_and_print_metrics

TARGET_QUANTILES = np.round(np.arange(0.025, 1, 0.025), 3).tolist()

from conformal_model.scale.arch.smooth_quantile_loss import SmoothQuantileLoss
def _collect_auto_tune_signal(dm: SpatioTemporalDataModule, *, max_batches: int = 8) -> np.ndarray | None:
    xs: list[torch.Tensor] = []
    loader = dm.train_dataloader()
    for i, batch in enumerate(loader):
        x = batch.input["x"]
        if x.dim() == 4:
            x = x[..., 0]
        xs.append(x.detach().cpu())
        if i + 1 >= int(max_batches):
            break
    if not xs:
        return None
    x_cat = torch.cat(xs, dim=0)  # (B,T,N)
    return x_cat.numpy()


def _get_dataset_class(name: str):
    ds = importlib.import_module("tsl.datasets")
    candidates = {
        "la": ["MetrLA"],
        "pems03": ["PeMS03", "Pems03", "PEMS03"],
        "pems04": ["PeMS04", "Pems04", "PEMS04"],
        "pems07": ["PeMS07", "Pems07", "PEMS07"],
        "pems08": ["PeMS08", "Pems08", "PEMS08"],
        "pems_bay": ["PemsBay", "PeMSBay", "PEMSBay"],
        "large_st": ["LargeST"],
    }
    for cls_name in candidates.get(name, []):
        if hasattr(ds, cls_name):
            return getattr(ds, cls_name)
    raise ImportError(f"Dataset class for '{name}' not found in tsl.datasets.")

def _make_dataset(ds_cls, *args, **kwargs):
    """Instantiate TSL datasets under DATA_ROOT."""
    try:
        from tsl import config as tsl_config
        tsl_config.data_dir = str(DATA_ROOT)
    except Exception:
        pass
    return ds_cls(*args, **kwargs)

def get_dataset(dataset_cfg):
    name = dataset_cfg["name"]
    if name == "la":
        dataset = _make_dataset(_get_dataset_class("la"))
    elif name in {"pems03", "pems04", "pems07", "pems08", "pems_bay"}:
        dataset = _make_dataset(_get_dataset_class(name))
    elif name == "large_st":
        hparams = dict(dataset_cfg.get("hparams", {}))
        root = hparams.pop("root", None) or str(DATA_ROOT / "large_st")
        dataset = _make_dataset(_get_dataset_class(name), root=root, **hparams)
    elif name == "air":
        dataset = AirQuality()
    elif name == "gpvar":
        dataset = GPVARDataset(**dataset_cfg["hparams"], p_max=0)
    else:
        raise ValueError(f"Dataset {name} not available.")
    return dataset


def _tod_dow_from_index(index: pd.Index) -> np.ndarray:
    if not isinstance(index, pd.DatetimeIndex):
        return np.zeros((len(index), 2), dtype=np.float32)
    minutes = index.hour * 60 + index.minute
    tod = minutes / float(24 * 60)
    dow = index.dayofweek / 7.0
    return np.stack([tod.astype(np.float32), dow.astype(np.float32)], axis=1)


def _collect_split(dataset: SpatioTemporalDataset, indices, batch_size: int, workers: int):
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=lambda batch: batch,
    )
    contexts, residuals, exogs, masks = [], [], [], []
    for batch in loader:
        xs = torch.stack([item.input["x"] for item in batch], dim=0)
        ys = torch.stack([item.target["y"] for item in batch], dim=0)
        contexts.append(xs.cpu().float())
        residuals.append(ys.cpu().float())
        if "u" in batch[0].input:
            us = torch.stack([item.input["u"] for item in batch], dim=0)
            exogs.append(us.cpu().float())
        if "mask_target" in batch[0].input:
            ms = torch.stack([item.input["mask_target"] for item in batch], dim=0)
            masks.append(ms.cpu().bool())

    context = torch.cat(contexts, dim=0)
    residuals = torch.cat(residuals, dim=0)
    exog = torch.cat(exogs, dim=0) if exogs else None
    mask = torch.cat(masks, dim=0) if masks else None
    return context, residuals, exog, mask


def _reshape_residuals(residuals: torch.Tensor) -> torch.Tensor:
    if residuals.dim() == 4 and residuals.size(1) == 1:
        residuals = residuals.squeeze(1)
    if residuals.dim() == 3:
        return residuals.permute(0, 2, 1).contiguous()
    return residuals


def _reshape_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
    if mask is None:
        return None
    if mask.dim() == 4 and mask.size(1) == 1:
        mask = mask.squeeze(1)
    if mask.dim() == 3:
        return mask.permute(0, 2, 1).contiguous()
    return mask


def _standardize(
    train: torch.Tensor, other: torch.Tensor | None, axis
) -> tuple[torch.Tensor, torch.Tensor | None, StandardScaler]:
    scaler = StandardScaler(axis=axis)
    train_np = train.numpy()
    scaler.fit(train_np)
    train_scaled = torch.tensor(scaler.transform(train_np), dtype=torch.float32)
    other_scaled = None
    if other is not None:
        other_scaled = torch.tensor(scaler.transform(other.numpy()), dtype=torch.float32)
    return train_scaled, other_scaled, scaler


def _inverse_standardize_tensor(tensor: torch.Tensor, scaler: StandardScaler) -> torch.Tensor:
    bias = torch.tensor(scaler.bias, dtype=tensor.dtype)
    scale = torch.tensor(scaler.scale, dtype=tensor.dtype)
    return tensor * (scale + 1e-8) + bias


def _transform_with_scaler(tensor: torch.Tensor, scaler: StandardScaler) -> torch.Tensor:
    transformed = scaler.transform(tensor.numpy())
    return torch.tensor(transformed, dtype=torch.float32)


def _jsonable_metrics(metrics: dict) -> dict:
    out = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            out[key] = float(value.detach().cpu().item())
        elif isinstance(value, (np.floating, np.integer)):
            out[key] = float(value)
        elif isinstance(value, dict):
            out[key] = _jsonable_metrics(value)
        else:
            out[key] = value
    return out


def _get_metric(metrics: dict, key: str):
    for prefix in ("test_", ""):
        k = f"{prefix}{key}"
        if k in metrics:
            return metrics[k]
    return None


def _build_per_alpha(metrics: dict, alphas):
    per_alpha = {}
    for alpha in alphas:
        tag = int((1 - alpha) * 100)
        observed = _get_metric(metrics, f"coverage_at_{tag}")
        pi_width = _get_metric(metrics, f"pi_width_at_{tag}")
        pi_width_median = _get_metric(metrics, f"pi_width_median_at_{tag}")
        winkler = _get_metric(metrics, f"winkler_at_{tag}")
        delta = _get_metric(metrics, f"delta_cov_at_{tag}")
        target_cov = 1.0 - float(alpha)
        if delta is None and observed is not None:
            delta = observed - target_cov
        per_alpha[float(alpha)] = {
            "observed_coverage": observed,
            "pi_width": pi_width,
            "winkler": winkler,
            "pi_width_median": pi_width_median,
            "delta_cov": delta,
            "target_coverage": target_cov,
        }
    return per_alpha




def run_experiment(cfg: DictConfig):
    run_cfg = cfg.get("run") or {}
    seed = run_cfg.get("seed") if isinstance(run_cfg, dict) else getattr(run_cfg, "seed", None)
    if seed is not None:
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    warnings.filterwarnings(
        "ignore",
        message=r".*'T' is deprecated and will be removed.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*DataFrame.replace.*method.*deprecated.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"The ``compute`` method of metric .* was called before the ``update`` method.*",
        category=UserWarning,
    )

    local_dir = Path(cfg.src_dir)

    residuals_input: pd.DataFrame = pd.read_hdf(local_dir / "residuals.h5", key="input")
    residuals_target: pd.DataFrame = pd.read_hdf(local_dir / "residuals.h5", key="target")
    with open(local_dir / "config.yaml", "r") as fp:
        src_config = yaml.load(fp, Loader=yaml.FullLoader)

    assert cfg.dataset.name == src_config["dataset"]["name"]
    horizon_val = int(cfg.get("horizon", src_config.get("horizon", 1)))
    src_horizon = int(src_config.get("horizon", horizon_val))
    if horizon_val != src_horizon:
        raise ValueError(
            f"Config horizon ({horizon_val}) does not match stage-1 horizon ({src_horizon})."
        )
    dataset = get_dataset(src_config["dataset"])

    if cfg.dataset.name in {"gpvar"}:
        ds_index = pd.Index(dataset.index)
    else:
        ds_index = dataset.index

    try:
        mask_target = pd.read_hdf(local_dir / "residuals.h5", key="target_mask")
        mask_target = mask_target.reindex(index=dataset.index)
    except KeyError:
        mask_target = None

    indices = np.load(local_dir / "indices.npz")

    covariates = dict(
        residuals_input=(residuals_input.reindex(index=ds_index), "t n f"),
        residuals_target=(residuals_target.reindex(index=ds_index), "t n f"),
    )

    tod_dow = _tod_dow_from_index(ds_index)
    if cfg.get("add_exogenous") and hasattr(dataset, "datetime_encoded") and isinstance(ds_index, pd.DatetimeIndex):
        day_sin_cos = dataset.datetime_encoded("day").values
        weekdays = dataset.datetime_onehot("weekday").values
        exog = np.concatenate([day_sin_cos, weekdays, tod_dow], axis=-1)
    else:
        exog = tod_dow
    covariates.update(u=exog)

    target_map = BatchMap()
    target_map["y"] = BatchMapItem("residuals_target", synch_mode="horizon", pattern="t n f", preprocess=False)

    input_map = BatchMap()
    if mask_target is not None:
        input_map["mask_target"] = BatchMapItem(["mask_target"], synch_mode="horizon", pattern="t n f")
        covariates.update(mask_target=mask_target.astype("bool"))

    if "u" in covariates:
        input_map["u"] = BatchMapItem("u", synch_mode="window", pattern="t f")

    inputs_ = ["residuals_input"]
    if cfg.get("target_as_input", True):
        inputs_.append("target")

    input_map["x"] = BatchMapItem(inputs_, synch_mode="window", pattern="t n f")

    torch_dataset = SpatioTemporalDataset(
        index=ds_index,
        target=dataset.dataframe(),
        mask=dataset.mask,
        covariates=covariates,
        window=cfg.window,
        stride=src_config["stride"],
        target_map=target_map,
        input_map=input_map,
        delay=src_config.get("delay", 0),
        horizon=horizon_val,
    )

    calib_all, test_indices = parse_and_filter_indices(torch_dataset, indices)
    val_len = int(cfg.val_len * len(calib_all))
    train_indices = calib_all[:-val_len - torch_dataset.samples_offset]
    val_indices = calib_all[-val_len:]

    calib_splitter = FixedIndicesSplitter(
        train_idxs=train_indices,
        val_idxs=val_indices,
        test_idxs=test_indices,
    )

    scale_axis = (0,) if cfg.get("scale_axis") == "node" else (0, 1)
    transform = {
        "target": StandardScaler(axis=scale_axis),
        "residuals_target": StandardScaler(axis=scale_axis),
        "residuals_input": StandardScaler(axis=scale_axis),
    }

    workers = int(cfg.workers) if "workers" in cfg else 0

    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=transform if cfg.get("apply_scaler", True) else {},
        splitter=calib_splitter,
        batch_size=cfg.batch_size,
        workers=workers,
    )
    dm.setup()

    adj = dataset.get_connectivity(**cfg.dataset.connectivity, train_slice=dm.train_slice)
    if isinstance(adj, torch.Tensor):
        adjacency = adj.detach().cpu().numpy()
    elif hasattr(adj, "toarray"):
        adjacency = adj.toarray()
    else:
        adjacency = np.asarray(adj)

    sample = next(iter(dm.train_dataloader()))
    context_channels = int(sample.input["x"].shape[-1])
    exog_dim = int(sample.input["u"].shape[-1]) if "u" in sample.input else 0

    cfg_model = QuantileModelConfig(
        input_length=int(sample.input["x"].shape[1]),
        num_nodes=int(sample.input["x"].shape[2]),
        quantiles=TARGET_QUANTILES,
        horizon=int(sample.target["y"].shape[1]),
        context_exog_dim=exog_dim,
        context_channels=context_channels,
    )

    model_hp = cfg.model.hparams.get("model", {})
    loss_hp = cfg.model.hparams.get("loss", {})
    backbone_params = model_hp.get("backbone", {})
    backbone_kwargs = dict(backbone_params)
    backbone_kwargs.update(model_hp.get("backbone_kwargs", {}))

    auto_tune_signal = _collect_auto_tune_signal(dm, max_batches=8)

    model_cls = resolve_scale_model(cfg.model.name if hasattr(cfg, "model") else None)
    model = model_cls(
        cfg=cfg_model,
        adjacency=adjacency,
        gso=torch.from_numpy(adjacency).float(),
        n_scales=int(model_hp.get("n_scales", 4)),
        kernel_type=str(model_hp.get("kernel_type", "mexican_hat")),
        n_high_scales=model_hp.get("n_high_scales"),
        auto_tune_signal=auto_tune_signal,
        backbone_out_dim=int(model_hp.get("backbone_out_dim", 128)),
        backbone_kwargs=backbone_kwargs,
        enable_gating=bool(model_hp.get("enable_gating", True)),
        high_stats_hidden_dim=model_hp.get("high_stats_hidden_dim"),
    )

    def get_metric_at_alpha(base_metric, target_alpha):
        idx_low = find_close(target_alpha / 2, TARGET_QUANTILES)
        idx_high = find_close(1 - target_alpha / 2, TARGET_QUANTILES)

        def preprocessing_fn(y_hat):
            return torch.stack((y_hat[idx_low], y_hat[idx_high]))

        return MaskedMetricWrapper(metric=base_metric, input_preprocessing=preprocessing_fn)

    log_metrics = {"pinball": MaskedMultiPinballLoss(qs=TARGET_QUANTILES)}
    for alpha in cfg.alphas:
        key = int((1 - alpha) * 100)
        log_metrics[f"coverage_at_{key}"] = get_metric_at_alpha(MaskedCoverage(), alpha)
        log_metrics[f"delta_cov_at_{key}"] = get_metric_at_alpha(MaskedDeltaCoverage(alpha=alpha), alpha)
        log_metrics[f"pi_width_at_{key}"] = get_metric_at_alpha(MaskedPIWidth(), alpha)
        log_metrics[f"pi_width_median_at_{key}"] = get_metric_at_alpha(MaskedPIMedianWidth(), alpha)
        log_metrics[f"winkler_at_{key}"] = get_metric_at_alpha(MaskedWinklerScore(alpha=alpha), alpha)
    horizon = int(cfg_model.horizon)
    for h in range(horizon):
        step = h + 1
        log_metrics[f"pinball_h{step}"] = MaskedMultiPinballLoss(qs=TARGET_QUANTILES, at=h)
        for alpha in cfg.alphas:
            key = int((1 - alpha) * 100)
            log_metrics[f"coverage_at_{key}_h{step}"] = get_metric_at_alpha(MaskedCoverage(at=h), alpha)
            log_metrics[f"delta_cov_at_{key}_h{step}"] = get_metric_at_alpha(MaskedDeltaCoverage(alpha=alpha, at=h), alpha)
            log_metrics[f"pi_width_at_{key}_h{step}"] = get_metric_at_alpha(MaskedPIWidth(at=h), alpha)
            log_metrics[f"pi_width_median_at_{key}_h{step}"] = get_metric_at_alpha(MaskedPIMedianWidth(at=h), alpha)
            log_metrics[f"winkler_at_{key}_h{step}"] = get_metric_at_alpha(MaskedWinklerScore(alpha=alpha, at=h), alpha)

    loss_fn = SmoothQuantileLoss(TARGET_QUANTILES, crossing_penalty_weight=loss_hp.get("crossing_penalty", 0.0), smoothness_weight=loss_hp.get("smoothness_weight", 0.0))
    scheduler_class = None
    scheduler_kwargs = None
    if cfg.get("lr_scheduler") is not None:
        scheduler_class = getattr(torch.optim.lr_scheduler, cfg.lr_scheduler.name)
        scheduler_kwargs = dict(cfg.lr_scheduler.hparams)
    predictor = scalePredictor(
        model=model,
        loss_fn=loss_fn,
        metrics=log_metrics,
        optim_class=torch.optim.AdamW,
        optim_kwargs=dict(lr=float(cfg.lr), weight_decay=float(cfg.weight_decay)),
        scheduler_class=scheduler_class,
        scheduler_kwargs=scheduler_kwargs,
        scale_target=cfg.scale_target,
        quantiles=TARGET_QUANTILES,
    )

    exp_logger = TensorBoardLogger(save_dir=cfg.run.dir, name=cfg.run.name)
    monitor_key = "val_winkler_at_90" if 0.1 in list(cfg.alphas) else f"val_winkler_at_{int((1 - cfg.alphas[0]) * 100)}"
    monitor_key = cfg.get("early_stop_metric", monitor_key)
    min_delta = float(cfg.get("early_stop_min_delta", 0.0))
    early_stop_callback = EarlyStopping(
        monitor=monitor_key,
        patience=cfg.patience,
        mode="min",
        min_delta=min_delta,
        check_on_train_epoch_end=False,
        verbose=True,
        strict=False,
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.run.dir, save_top_k=1, monitor=monitor_key, mode="min"
    )

    trainer = Trainer(
        max_epochs=cfg.epochs,
        limit_train_batches=cfg.get("train_batches", 1.0),
        default_root_dir=cfg.run.dir,
        logger=exp_logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=cfg.grad_clip_val,
        callbacks=[early_stop_callback, checkpoint_callback],
    )

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    trainer.fit(predictor, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.validate(predictor, dataloaders=dm.val_dataloader())
    test_results = trainer.test(predictor, dataloaders=dm.test_dataloader())

    raw_metrics = test_results[0] if test_results else {}
    per_alpha = _build_per_alpha(raw_metrics, cfg.alphas)

    run_dir = Path(cfg.run.dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    file_logger = setup_file_logger(run_dir, name="scale")

    for alpha in sorted(per_alpha.keys()):
        stats = per_alpha[alpha]
        log_and_print_metrics(
            stats,
            alpha=alpha,
            target_cov=stats.get("target_coverage"),
            logger=file_logger,
            print_fn=printing.print_metrics,
        )

    cfg_payload = OmegaConf.to_container(cfg, resolve=True)
    metadata = {"src_config": src_config}
    metrics_payload = {"per_alpha": per_alpha, "raw": _jsonable_metrics(raw_metrics)}
    save_metrics(run_dir, local_dir, metadata, cfg_payload, metrics_payload)

    logger.info(raw_metrics)
    return metrics_payload


if __name__ == "__main__":
    exp = Experiment(run_fn=run_experiment,
                     config_path=str(ROOT / "conformal_model" / "scale" / "config"),
                     config_name="default")
    res = exp.run()
    logger.info(res)
