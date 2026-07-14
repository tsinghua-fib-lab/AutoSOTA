#!/usr/bin/env python3
"""
Inductive LFADS training on the macaque reaching dataset with k-fold CV. Note that model-specific hardcoded constants are set here, below the imports.

This mirrors the inductive CEBRA runner:
- Uses the existing macaque preprocessing and k-fold splits.
- Trains LFADS on TRAIN trials only.
- Extracts LFADS factors (of dimension ) for TRAIN/VAL/TEST.
- Concatenates per-trial factor timepoints to feed an SVM probe.

Run via:
python3 run_macaque_multiday_cv.py \
--root_dir=/bsuhome/davejohnson408/scratch/vdw \
--model=lfads \
--days=0-43
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import Callback
try:
    from pytorch_lightning.plugins.environments import LightningEnvironment
except ImportError:  # pragma: no cover - older lightning
    try:
        from pytorch_lightning.plugins.environments import LightningEnvironment as _LE
    except Exception as exc:  # pragma: no cover - very old lightning
        raise ImportError("Unable to import LightningEnvironment from pytorch_lightning.") from exc
    else:
        LightningEnvironment = _LE

# Ensure project root on path for local imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_CODE_DIR = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_CODE_DIR))

from data_processing.lfads_macaque import (  # noqa: E402
    LfadsFoldData,
    prepare_lfads_fold_data,
    write_lfads_hdf5,
)
from models.comparisons.LFADS.lfads_torch.datamodules import BasicDataModule  # noqa: E402
from models.comparisons.LFADS.lfads_torch.model import LFADS  # noqa: E402
from models.comparisons.LFADS.lfads_torch.modules.augmentations import AugmentationStack  # noqa: E402
from models.comparisons.LFADS.lfads_torch.modules.priors import (  # noqa: E402
    AutoregressiveMultivariateNormal,
    MultivariateNormal,
)
from models.comparisons.LFADS.lfads_torch.modules.recons import Gaussian, Poisson  # noqa: E402
from models.comparisons.LFADS.lfads_torch.utils import send_batch_to_device  # noqa: E402
from models.nn_utilities import count_parameters  # noqa: E402
from os_utilities import create_experiment_dir, ensure_dir_exists, smart_pickle  # noqa: E402
from training.macaque_inductive_utils import (  # noqa: E402
    save_embeddings,
    train_and_evaluate_svm,
)
from training.kfold_results import compute_kfold_results  # noqa: E402

DEFAULT_CONFIG = (
    Path(__file__).resolve().parent.parent / "config" / "yaml_files" / "macaque" / "experiment.yaml"
)

DEFAULT_BATCH_SIZE = 128
DEFAULT_MAX_EPOCHS = 512
DEFAULT_LR_INIT = 0.005
# Uses scheduler settings in the experiment.yaml
DEFAULT_LR_STOP = None
DEFAULT_LR_DECAY = None
DEFAULT_LR_PATIENCE = None
DEFAULT_DROPOUT_RATE = 0.05
DEFAULT_CELL_CLIP = 5.0
DEFAULT_IC_POST_VAR_MIN = 1e-4

DEFAULT_IC_ENC_SEQ_LEN = 0
DEFAULT_IC_ENC_DIM = 256
DEFAULT_CI_ENC_DIM = 256
DEFAULT_CI_LAG = 1
DEFAULT_CON_DIM = 64
DEFAULT_CO_DIM = 32
DEFAULT_IC_DIM = 64
DEFAULT_GEN_DIM = 64
DEFAULT_FAC_DIM = 3  # 'embedding'/LFADS factors dimension

DEFAULT_IC_PRIOR_MEAN = 0.0
DEFAULT_IC_PRIOR_VAR = 0.1
DEFAULT_CO_PRIOR_TAU = 10.0
DEFAULT_CO_PRIOR_NVAR = 0.1

DEFAULT_L2_START_EPOCH = 0
DEFAULT_L2_INCREASE_EPOCH = 50
DEFAULT_L2_IC_ENC_SCALE = 0.0
DEFAULT_L2_CI_ENC_SCALE = 0.0
DEFAULT_L2_GEN_SCALE = 0.0
DEFAULT_L2_CON_SCALE = 0.0

DEFAULT_KL_START_EPOCH = 0
DEFAULT_KL_INCREASE_EPOCH = 50
DEFAULT_KL_IC_SCALE = 1.0
DEFAULT_KL_CO_SCALE = 1.0

DEFAULT_LOSS_SCALE = 1.0
DEFAULT_RECON_REDUCE_MEAN = True
DEFAULT_RECONSTRUCTION = "gaussian"  # gaussian | poisson

DEFAULT_SVM_KERNEL = "rbf"
DEFAULT_SVM_C = 1.0
DEFAULT_SVM_GAMMA = "scale"

LFADS_DATA_DIRNAME = "lfads_data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LFADS inductively with k-fold CV on the macaque reaching dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to experiment YAML.")
    parser.add_argument("--day_idx", type=int, help="Day index override (default from config).")
    parser.add_argument("--folds", type=int, nargs="+", help="Space-separated fold indices to run.")
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--seed", type=int, help="Random seed override.")
    parser.add_argument("--model_name", type=str, default="lfads", help="Experiment model name.")
    parser.add_argument("--experiment_dir", type=Path, help="Existing experiment directory to reuse.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max_epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--fac_dim", type=int, default=DEFAULT_FAC_DIM)
    parser.add_argument("--gen_dim", type=int, default=DEFAULT_GEN_DIM)
    parser.add_argument("--ic_dim", type=int, default=DEFAULT_IC_DIM)
    parser.add_argument("--ic_enc_dim", type=int, default=DEFAULT_IC_ENC_DIM)
    parser.add_argument("--ci_enc_dim", type=int, default=DEFAULT_CI_ENC_DIM)
    parser.add_argument("--con_dim", type=int, default=DEFAULT_CON_DIM)
    parser.add_argument("--co_dim", type=int, default=DEFAULT_CO_DIM)
    parser.add_argument("--ic_enc_seq_len", type=int, default=DEFAULT_IC_ENC_SEQ_LEN)
    parser.add_argument("--ci_lag", type=int, default=DEFAULT_CI_LAG)
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LR_INIT)
    parser.add_argument("--lr_patience", type=int, default=DEFAULT_LR_PATIENCE)
    parser.add_argument("--svm_kernel", type=str, default=DEFAULT_SVM_KERNEL)
    parser.add_argument("--svm_c", type=float, default=DEFAULT_SVM_C)
    parser.add_argument("--svm_gamma", type=str, default=DEFAULT_SVM_GAMMA)
    parser.add_argument("--lfads_data_dir", type=Path, help="Override base directory for LFADS HDF5 files.")
    parser.add_argument("--num_workers", type=int, help="DataLoader workers override.")
    return parser.parse_args()


def load_yaml_config(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_global_seed(seed: int, set_torch_seed: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if set_torch_seed:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def resolve_device_mode(device_arg: str) -> Tuple[str, int]:
    if device_arg == "cpu":
        return "cpu", 1
    if torch.cuda.is_available():
        return "gpu", 1
    return "cpu", 1


def resolve_int_setting(
    *,
    cli_value: Optional[int],
    config_value: Optional[int],
    fallback: int,
) -> int:
    if cli_value is not None:
        return int(cli_value)
    if config_value is not None:
        return int(config_value)
    return int(fallback)


def resolve_float_setting(
    *,
    cli_value: Optional[float],
    config_value: Optional[float],
    fallback: float,
) -> float:
    if cli_value is not None:
        return float(cli_value)
    if config_value is not None:
        return float(config_value)
    return float(fallback)


def build_lfads_model(
    *,
    encod_data_dim: int,
    encod_seq_len: int,
    recon_seq_len: int,
    fac_dim: int,
    gen_dim: int,
    ic_dim: int,
    ic_enc_dim: int,
    ci_enc_dim: int,
    con_dim: int,
    co_dim: int,
    ic_enc_seq_len: int,
    ci_lag: int,
    learning_rate: float,
    lr_patience: int,
    lr_stop: float,
    lr_decay: float,
    reconstruction: str,
) -> LFADS:
    reconstruction_key = str(reconstruction or DEFAULT_RECONSTRUCTION).lower()
    if reconstruction_key == "poisson":
        reconstruction_module = Poisson()
    elif reconstruction_key in ("gaussian", "normal"):
        reconstruction_module = Gaussian()
    else:
        raise ValueError(f"Unsupported reconstruction type: {reconstruction}")
    recon_list = torch.nn.ModuleList([reconstruction_module])
    readin = torch.nn.ModuleList([torch.nn.Identity()])
    readout = torch.nn.ModuleList(
        [torch.nn.Linear(fac_dim, encod_data_dim * reconstruction_module.n_params)]
    )
    train_aug_stack = AugmentationStack(transforms=[], batch_order=[], loss_order=[])
    infer_aug_stack = AugmentationStack(transforms=[], batch_order=[], loss_order=[])

    ic_prior = MultivariateNormal(
        mean=float(DEFAULT_IC_PRIOR_MEAN),
        variance=float(DEFAULT_IC_PRIOR_VAR),
        shape=int(ic_dim),
    )
    co_prior = AutoregressiveMultivariateNormal(
        tau=float(DEFAULT_CO_PRIOR_TAU),
        nvar=float(DEFAULT_CO_PRIOR_NVAR),
        shape=int(co_dim),
    )

    return LFADS(
        encod_data_dim=int(encod_data_dim),
        encod_seq_len=int(encod_seq_len),
        recon_seq_len=int(recon_seq_len),
        ext_input_dim=0,
        ic_enc_seq_len=int(ic_enc_seq_len),
        ic_enc_dim=int(ic_enc_dim),
        ci_enc_dim=int(ci_enc_dim),
        ci_lag=int(ci_lag),
        con_dim=int(con_dim),
        co_dim=int(co_dim),
        ic_dim=int(ic_dim),
        gen_dim=int(gen_dim),
        fac_dim=int(fac_dim),
        dropout_rate=float(DEFAULT_DROPOUT_RATE),
        reconstruction=recon_list,
        variational=True,
        co_prior=co_prior,
        ic_prior=ic_prior,
        ic_post_var_min=float(DEFAULT_IC_POST_VAR_MIN),
        cell_clip=float(DEFAULT_CELL_CLIP),
        train_aug_stack=train_aug_stack,
        infer_aug_stack=infer_aug_stack,
        readin=readin,
        readout=readout,
        loss_scale=float(DEFAULT_LOSS_SCALE),
        recon_reduce_mean=bool(DEFAULT_RECON_REDUCE_MEAN),
        lr_scheduler=True,
        lr_init=float(learning_rate),
        lr_stop=float(lr_stop),
        lr_decay=float(lr_decay),
        lr_patience=int(lr_patience),
        lr_adam_beta1=0.9,
        lr_adam_beta2=0.999,
        lr_adam_epsilon=1e-8,
        weight_decay=0.0,
        l2_start_epoch=int(DEFAULT_L2_START_EPOCH),
        l2_increase_epoch=int(DEFAULT_L2_INCREASE_EPOCH),
        l2_ic_enc_scale=float(DEFAULT_L2_IC_ENC_SCALE),
        l2_ci_enc_scale=float(DEFAULT_L2_CI_ENC_SCALE),
        l2_gen_scale=float(DEFAULT_L2_GEN_SCALE),
        l2_con_scale=float(DEFAULT_L2_CON_SCALE),
        kl_start_epoch=int(DEFAULT_KL_START_EPOCH),
        kl_increase_epoch=int(DEFAULT_KL_INCREASE_EPOCH),
        kl_ic_scale=float(DEFAULT_KL_IC_SCALE),
        kl_co_scale=float(DEFAULT_KL_CO_SCALE),
    )


def extract_factors(
    *,
    model: LFADS,
    datamodule: BasicDataModule,
    split_name: str,
    device: torch.device,
) -> np.ndarray:
    datamodule.setup()
    pred_loaders = datamodule.predict_dataloader()
    if 0 not in pred_loaders:
        raise ValueError("LFADS datamodule predict_dataloader missing session 0.")
    if split_name not in pred_loaders[0]:
        raise ValueError(f"Split '{split_name}' not available in LFADS datamodule.")

    factors_list: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in pred_loaders[0][split_name]:
            batch_dict = {0: batch}
            batch_dict = send_batch_to_device(batch_dict, device)
            output = model.predict_step(batch_dict, batch_idx=0, sample_posteriors=False)
            session_output = output[0]
            factors = session_output.factors.detach().cpu().numpy()
            factors_list.append(factors)

    if not factors_list:
        return np.empty((0, 0, 0), dtype=np.float32)
    return np.concatenate(factors_list, axis=0).astype(np.float32)


def flatten_factors(factors: np.ndarray) -> np.ndarray:
    if factors.ndim != 3:
        raise ValueError(f"Expected factors array with 3 dims, got shape {factors.shape}.")
    return factors.reshape(factors.shape[0], -1)


def safe_count_parameters(model: LFADS, log) -> Optional[int]:
    """
    Safely count parameters, logging a warning on failure.
    """

    try:
        return count_parameters(model)
    except Exception as exc:
        log(f"Warning: failed to count parameters: {exc}")
        return None


def _parse_epoch_from_checkpoint(path: str) -> Optional[int]:
    if not path:
        return None
    name = Path(path).stem
    if "epoch=" in name:
        try:
            epoch_token = name.split("epoch=")[-1].split("-")[0]
            return int(epoch_token)
        except Exception:
            return None
    if name.startswith("lfads-"):
        try:
            return int(name.split("lfads-")[-1])
        except Exception:
            return None
    return None


# Custom callback to save the best checkpoint based on SVM validation accuracy.
class SvmBestCheckpointCallback(Callback):
    def __init__(
        self,
        *,
        datamodule: BasicDataModule,
        lfads_fold: LfadsFoldData,
        svm_kernel: str,
        svm_c: float,
        svm_gamma: str,
        checkpoint_path: Path,
        device: torch.device,
        log_fn,
        patience: int,
        min_delta: float = 0.0,
    ) -> None:
        self.datamodule = datamodule
        self.lfads_fold = lfads_fold
        self.svm_kernel = svm_kernel
        self.svm_c = svm_c
        self.svm_gamma = svm_gamma
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.log_fn = log_fn
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best_val_acc = float("-inf")
        self.best_epoch: Optional[int] = None
        self.epochs_since_improve = 0

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return

        try:
            train_factors = extract_factors(
                model=pl_module,
                datamodule=self.datamodule,
                split_name="train",
                device=self.device,
            )
            valid_factors = extract_factors(
                model=pl_module,
                datamodule=self.datamodule,
                split_name="valid",
                device=self.device,
            )
        except Exception as exc:
            self.log_fn(f"Warning: SVM selection failed at epoch {trainer.current_epoch}: {exc}")
            return

        train_feats = flatten_factors(train_factors)
        valid_feats = flatten_factors(valid_factors)
        if train_feats.size == 0 or valid_feats.size == 0:
            self.log_fn(
                f"Warning: empty embeddings for SVM selection at epoch {trainer.current_epoch} "
                f"(train={train_feats.shape}, valid={valid_feats.shape})."
            )
            return

        empty_test_feats = np.empty((0, train_feats.shape[1]), dtype=train_feats.dtype)
        empty_test_labels = np.empty((0,), dtype=np.int64)
        svm_stats = train_and_evaluate_svm(
            train_feats=train_feats,
            train_labels=self.lfads_fold.train.condition_ids,
            val_feats=valid_feats,
            val_labels=self.lfads_fold.valid.condition_ids,
            test_feats=empty_test_feats,
            test_labels=empty_test_labels,
            test_trial_ids=None,
            kernel=self.svm_kernel,
            C=float(self.svm_c),
            gamma=self.svm_gamma,
        )
        val_acc = svm_stats.get("val_accuracy", float("nan"))
        if not math.isfinite(val_acc):
            self.log_fn(f"Warning: invalid SVM val accuracy at epoch {trainer.current_epoch}: {val_acc}")
            return

        if val_acc > (self.best_val_acc + self.min_delta):
            self.best_val_acc = float(val_acc)
            self.best_epoch = int(trainer.current_epoch) + 1
            self.epochs_since_improve = 0
            try:
                trainer.save_checkpoint(str(self.checkpoint_path))
                self.log_fn(
                    f"SVM-best checkpoint updated at epoch {self.best_epoch} "
                    f"(val_acc={self.best_val_acc:.4f})."
                )
            except Exception as exc:
                self.log_fn(f"Warning: failed to save SVM-best checkpoint: {exc}")
        else:
            self.epochs_since_improve += 1
            if self.patience > 0 and self.epochs_since_improve >= self.patience:
                self.log_fn(
                    f"Early stopping: SVM val accuracy did not improve for "
                    f"{self.epochs_since_improve} epochs (patience={self.patience})."
                )
                trainer.should_stop = True


class EpochTimingCallback(Callback):
    def __init__(self) -> None:
        self.train_times: Dict[int, float] = {}
        self.valid_times: Dict[int, float] = {}
        self._train_start: Optional[float] = None
        self._valid_start: Optional[float] = None

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        self._train_start = time.time()

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if trainer.sanity_checking:
            return
        if self._train_start is None:
            self._train_start = time.time()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        if self._train_start is None:
            return
        epoch = int(trainer.current_epoch) + 1
        self.train_times[epoch] = time.time() - self._train_start
        self._train_start = None

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        self._valid_start = time.time()

    def on_validation_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if trainer.sanity_checking:
            return
        if self._valid_start is None:
            self._valid_start = time.time()

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        if self._valid_start is None:
            return
        epoch = int(trainer.current_epoch) + 1
        self.valid_times[epoch] = time.time() - self._valid_start
        self._valid_start = None


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    training_cfg = cfg["training"]
    dataset_cfg = cfg["dataset"]
    optimizer_cfg = cfg.get("optimizer", {})
    scheduler_cfg = cfg.get("scheduler", {})

    seed = args.seed if args.seed is not None else int(dataset_cfg.get("split_seed", 123456))
    set_global_seed(seed)

    results_root = Path(training_cfg["root_dir"]).expanduser() / training_cfg["save_dir"]
    dataset_root = Path(training_cfg["root_dir"]).expanduser() / dataset_cfg["data_dir"]
    k_folds = int(training_cfg.get("k_folds", dataset_cfg.get("k_folds", 5)))

    if args.folds:
        fold_indices: List[int] = []
        seen = set()
        for idx in args.folds:
            if idx < 0 or idx >= k_folds:
                raise ValueError(f"Fold index {idx} is out of range for k_folds={k_folds}.")
            if idx not in seen:
                fold_indices.append(idx)
                seen.add(idx)
    else:
        fold_indices = list(range(k_folds))

    day_idx = args.day_idx if args.day_idx is not None else dataset_cfg.get("macaque_day_index")
    if day_idx is None:
        raise ValueError("day_idx must be provided via CLI or config.")

    if args.experiment_dir is not None:
        exp_dir_path = args.experiment_dir.expanduser()
        if not exp_dir_path.exists():
            raise FileNotFoundError(f"Experiment directory '{exp_dir_path}' does not exist.")
        ensure_dir_exists(exp_dir_path / "config", raise_exception=True)
        exp_dirs = {
            "exp_dir": str(exp_dir_path),
            "config_save_path": str(exp_dir_path / "config" / "config.yaml"),
        }
    else:
        exp_dirs = create_experiment_dir(
            root_dir=results_root,
            model_name=args.model_name,
            dataset_name=dataset_cfg.get("dataset", "macaque"),
            experiment_id=training_cfg.get("experiment_id"),
            config=cfg,
            verbosity=int(training_cfg.get("verbosity", 0)),
        )

    ensure_dir_exists(Path(exp_dirs["config_save_path"]).parent, raise_exception=True)
    with open(exp_dirs["config_save_path"], "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    resolved_batch_size = resolve_int_setting(
        cli_value=args.batch_size,
        config_value=None,
        fallback=DEFAULT_BATCH_SIZE,
    )
    resolved_max_epochs = resolve_int_setting(
        cli_value=args.max_epochs,
        config_value=None,
        fallback=DEFAULT_MAX_EPOCHS,
    )
    resolved_lr = resolve_float_setting(
        cli_value=args.learning_rate,
        config_value=None,
        fallback=DEFAULT_LR_INIT,
    )
    resolved_lr_patience = resolve_int_setting(
        cli_value=args.lr_patience,
        config_value=scheduler_cfg.get("patience"),
        fallback=10,
    )
    resolved_lr_stop = resolve_float_setting(
        cli_value=None,
        config_value=scheduler_cfg.get("min_lr"),
        fallback=1e-5,
    )
    resolved_lr_decay = resolve_float_setting(
        cli_value=None,
        config_value=scheduler_cfg.get("factor"),
        fallback=0.9,
    )
    resolved_num_workers = resolve_int_setting(
        cli_value=args.num_workers,
        config_value=training_cfg.get("dataloader_num_workers"),
        fallback=0,
    )
    early_stop_patience = resolve_int_setting(
        cli_value=None,
        config_value=training_cfg.get("no_valid_metric_improve_patience"),
        fallback=0,
    )

    accelerator, devices = resolve_device_mode(args.device)
    device = torch.device("cuda" if accelerator == "gpu" else "cpu")

    for fold_idx in fold_indices:
        fold_root = Path(exp_dirs["exp_dir"]) / f"fold_{fold_idx}"
        metrics_dir = fold_root / "metrics"
        models_dir = fold_root / "models"
        logs_dir = fold_root / "logs"
        config_dir = fold_root / "config"
        embeddings_dir = fold_root / "embeddings"
        lfads_data_dir = fold_root / LFADS_DATA_DIRNAME
        if args.lfads_data_dir is not None:
            lfads_data_dir = args.lfads_data_dir.expanduser().resolve() / f"fold_{fold_idx}"

        for dir_path in (metrics_dir, models_dir, logs_dir, config_dir, embeddings_dir, lfads_data_dir):
            ensure_dir_exists(dir_path, raise_exception=True)

        with open(config_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f)

        log_path = logs_dir / "training.log"

        def log(msg: str) -> None:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            line = f"[{timestamp}] {msg}"
            print(line)
            with open(log_path, "a", encoding="utf-8") as log_f:
                log_f.write(line + "\n")

        log(f"\nPreparing fold {fold_idx} data (seed={seed}, day_idx={day_idx})...")
        lfads_fold = prepare_lfads_fold_data(
            data_root=str(dataset_root),
            day_index=day_idx,
            k_folds=k_folds,
            fold_i=fold_idx,
            seed=seed,
        )

        h5_path = lfads_data_dir / "lfads_data.h5"
        write_lfads_hdf5(fold_data=lfads_fold, output_path=h5_path, include_test=True)

        datamodule = BasicDataModule(
            datafile_pattern=str(h5_path),
            batch_size=int(resolved_batch_size),
            num_workers=int(resolved_num_workers),
            sv_rate=0.0,
            sv_seed=0,
            dm_ic_enc_seq_len=int(args.ic_enc_seq_len),
        )

        model = build_lfads_model(
            encod_data_dim=int(lfads_fold.state_dim),
            encod_seq_len=int(lfads_fold.n_timepoints),
            recon_seq_len=int(lfads_fold.n_timepoints),
            fac_dim=int(args.fac_dim),
            gen_dim=int(args.gen_dim),
            ic_dim=int(args.ic_dim),
            ic_enc_dim=int(args.ic_enc_dim),
            ci_enc_dim=int(args.ci_enc_dim),
            con_dim=int(args.con_dim),
            co_dim=int(args.co_dim),
            ic_enc_seq_len=int(args.ic_enc_seq_len),
            ci_lag=int(args.ci_lag),
            learning_rate=float(resolved_lr),
            lr_patience=int(resolved_lr_patience),
            lr_stop=float(resolved_lr_stop),
            lr_decay=float(resolved_lr_decay),
            reconstruction=DEFAULT_RECONSTRUCTION,
        )

        log("Starting LFADS training...")
        svm_best_path = models_dir / "lfads_svm_best.ckpt"
        svm_best_cb = SvmBestCheckpointCallback(
            datamodule=datamodule,
            lfads_fold=lfads_fold,
            svm_kernel=args.svm_kernel,
            svm_c=float(args.svm_c),
            svm_gamma=args.svm_gamma,
            checkpoint_path=svm_best_path,
            device=device,
            log_fn=log,
            patience=early_stop_patience,
        )
        timing_cb = EpochTimingCallback()
        callbacks = [svm_best_cb, timing_cb]
        trainer_kwargs = {
            "max_epochs": int(resolved_max_epochs),
            "logger": False,
            "enable_checkpointing": False,
            "callbacks": callbacks,
            "plugins": [LightningEnvironment()],
        }

        # pytorch lightning trainer initialization
        try:
            trainer = pl.Trainer(
                gpus=(devices if accelerator == "gpu" else 0), 
                **trainer_kwargs
            )
        except TypeError:
            trainer = pl.Trainer(
                accelerator=accelerator,
                devices=devices,
                **trainer_kwargs,
            )

        # training
        train_fit_start = time.time()
        trainer.fit(model=model, datamodule=datamodule)
        train_fit_elapsed = time.time() - train_fit_start

        epochs_ran = int(trainer.current_epoch) + 1 if trainer.current_epoch is not None else 0

        # load best checkpoint
        best_checkpoint_path = None
        if svm_best_path.exists():
            best_checkpoint_path = svm_best_path
            log(f"Loading SVM-best checkpoint from {svm_best_path}")
            ckpt = torch.load(str(svm_best_path), map_location="cpu")
            model.load_state_dict(ckpt["state_dict"])
        else:
            log(
                "Warning: no SVM-best checkpoint was saved. "
                "Using final model weights instead."
            )
        model = model.to(device)

        log("Extracting LFADS factors...")
        train_factors = extract_factors(model=model, datamodule=datamodule, split_name="train", device=device)
        valid_factors = extract_factors(model=model, datamodule=datamodule, split_name="valid", device=device)
        test_factors = extract_factors(model=model, datamodule=datamodule, split_name="test", device=device)

        train_feats = flatten_factors(train_factors)
        valid_feats = flatten_factors(valid_factors)
        test_feats = flatten_factors(test_factors)

        svm_stats = train_and_evaluate_svm(
            train_feats=train_feats,
            train_labels=lfads_fold.train.condition_ids,
            val_feats=valid_feats,
            val_labels=lfads_fold.valid.condition_ids,
            test_feats=test_feats,
            test_labels=lfads_fold.test.condition_ids,
            test_trial_ids=lfads_fold.test.trial_ids,
            kernel=args.svm_kernel,
            C=float(args.svm_c),
            gamma=args.svm_gamma,
        )

        parameter_count = safe_count_parameters(model, log)
        best_epoch = svm_best_cb.best_epoch
        best_epoch_value = None
        if best_epoch is not None and best_epoch >= 1:
            best_epoch_value = int(best_epoch)
        else:
            best_epoch_value = max(epochs_ran, 1)

        mean_train_time = float("nan")
        mean_infer_time = float("nan")
        if timing_cb.train_times:
            train_subset = [t for e, t in timing_cb.train_times.items() if e <= best_epoch_value]
            if train_subset:
                mean_train_time = sum(float(t) for t in train_subset) / len(train_subset)
            else:
                earliest_epoch = min(timing_cb.train_times)
                mean_train_time = float(timing_cb.train_times[earliest_epoch])
        elif epochs_ran > 0:
            mean_train_time = float(train_fit_elapsed) / float(epochs_ran)
        if timing_cb.valid_times:
            valid_subset = [t for e, t in timing_cb.valid_times.items() if e <= best_epoch_value]
            if valid_subset:
                mean_infer_time = sum(float(t) for t in valid_subset) / len(valid_subset)
            else:
                earliest_epoch = min(timing_cb.valid_times)
                mean_infer_time = float(timing_cb.valid_times[earliest_epoch])
        log(
            f"[Fold {fold_idx}] Final metrics | "
            f"train_acc={svm_stats.get('train_accuracy', float('nan')):.4f} | "
            f"val_acc={svm_stats.get('val_accuracy', float('nan')):.4f} | "
            f"test_acc={svm_stats.get('test_accuracy', float('nan')):.4f}"
        )

        log("Saving artifacts...")
        svm_obj = svm_stats.get("svm")
        if svm_obj is not None:
            smart_pickle(str(models_dir / "svm.pkl"), svm_obj, overwrite=True)

        results = {
            "fold": fold_idx,
            "train_accuracy": float(svm_stats.get("train_accuracy", float("nan"))),
            "val_accuracy": float(svm_stats.get("val_accuracy", float("nan"))),
            "test_accuracy": float(svm_stats.get("test_accuracy", float("nan"))),
            "parameter_count": int(parameter_count) if parameter_count is not None else None,
            "best_checkpoint": str(best_checkpoint_path) if best_checkpoint_path is not None else "",
            "best_epoch": int(best_epoch_value),
            "mean_train_time": float(mean_train_time),
            "mean_infer_time": float(mean_infer_time),
        }
        with open(metrics_dir / "results.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(results, f)

        save_embeddings(embeddings_dir, "train", train_feats, lfads_fold.train.condition_ids, lfads_fold.train.trial_ids)
        save_embeddings(embeddings_dir, "valid", valid_feats, lfads_fold.valid.condition_ids, lfads_fold.valid.trial_ids)
        save_embeddings(embeddings_dir, "test", test_feats, lfads_fold.test.condition_ids, lfads_fold.test.trial_ids)

        log(f"Completed fold {fold_idx}. Results saved to {fold_root}.")

    if len(fold_indices) > 1:
        compute_kfold_results(
            parent_metrics_dir=Path(exp_dirs["exp_dir"]) / "metrics",
            timing_keys=("mean_train_time", "mean_infer_time"),
            weight_key="best_epoch",
        )


if __name__ == "__main__":
    main()

