# %%
import glob
import json
import logging
import math
import os
import random
import time

import hydra
import lightning as L
import numpy as np
import pandas as pd
import torch
import wandb
from data_setup import setup_dgp_args
from hydra.core.hydra_config import HydraConfig
from invert_poly_encoder import OraclePolyEncoder, PolyEncoder
from ivmodels import KClass
from ivmodels.tests import inverse_wald_test
from ivmodels.utils import oproj
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from mdcrl import LitAutoEncoder, SimDataset
from omegaconf import DictConfig, OmegaConf
from poly_decoder import PolyDecoder
from torch import nn
from torch.utils.data import DataLoader
from utils_train import (
    IVTrackingCallback,
    RuntimeInfoCallback,
    SaveValMetricsCallback,
    get_hardware_info,
)

# %%


def reformat_data(X):
    """
    Reformat data to matrices (np.array)
    """

    if X is None:
        return X
    if isinstance(X, pd.core.series.Series) or isinstance(
        X, pd.core.frame.DataFrame
    ):
        X = X.to_numpy().copy()
    if isinstance(X, torch.Tensor):
        X = X.numpy().copy()
    if X.ndim <= 1:
        X = X.reshape(-1, 1).copy()
    return X


def iv_wrapper(Y, D, Z, C=None, method="liml", fit_intercept=True):

    Y = reformat_data(Y)
    D = reformat_data(D)
    Z = reformat_data(Z)
    C = reformat_data(C)

    mod = KClass(kappa=method, fit_intercept=fit_intercept).fit(
        Z=Z, X=D, y=Y, C=C
    )

    ci = inverse_wald_test(Z=Z, X=D, y=Y, C=C, estimator=method)

    return mod.coef_, mod.intercept_, ci


def get_encoder(cfg, dim_z, dim_hu, poly_mix_weights, device):
    if cfg.encoder.type == "linear":
        return nn.Sequential(
            nn.Linear(dim_z, cfg.encoder.hidden_dim),
            nn.Linear(cfg.encoder.hidden_dim, cfg.encoder.hidden_dim),
            nn.Linear(cfg.encoder.hidden_dim, dim_hu),
        )
    elif cfg.encoder.type == "oracle":
        enc = OraclePolyEncoder(
            poly_mix_weights=poly_mix_weights, latent_dim=dim_hu, device=device
        )
        for p in enc.parameters():
            p.requires_grad = False
        return enc
    elif cfg.encoder.type == "poly":
        return PolyEncoder(
            data_dim=dim_z, latent_dim=dim_hu, poly_degree=2, device=device
        )
    elif cfg.encoder.type == "mlp":
        return nn.Sequential(
            nn.Linear(dim_z, cfg.encoder.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.encoder.hidden_dim, cfg.encoder.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.encoder.hidden_dim, dim_hu),
        )
    elif cfg.encoder.type == "mlpnorm":
        return nn.Sequential(
            nn.Linear(dim_z, cfg.encoder.hidden_dim),
            nn.LayerNorm(cfg.encoder.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.encoder.hidden_dim, cfg.encoder.hidden_dim),
            nn.LayerNorm(cfg.encoder.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.encoder.hidden_dim, dim_hu),
        )
    else:
        raise ValueError("Unsupported encoder type.")


def get_decoder(cfg, dim_z, dim_hu, device):
    if cfg.decoder.type == "poly":
        return PolyDecoder(
            data_dim=dim_z,
            latent_dim=dim_hu,
            poly_degree=cfg.data.polymix_degree,
            device=device,
        )
    elif cfg.decoder.type == "mlp":
        return nn.Sequential(
            nn.Linear(dim_hu, cfg.decoder.hidden_dim),
            nn.Linear(cfg.decoder.hidden_dim, dim_z),
        )
    elif cfg.decoder.type == "mlprelu":
        return nn.Sequential(
            nn.Linear(dim_hu, cfg.decoder.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.decoder.hidden_dim, cfg.decoder.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.decoder.hidden_dim, dim_z),
        )
    else:
        raise ValueError("Unsupported decoder type.")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    # Print identity immediately to the console/log file
    log = logging.getLogger(__name__)
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    array_idx = os.environ.get("SLURM_ARRAY_TASK_ID", "None")
    print(
        f"--- IDENTITY: Job={job_id} ArrayIdx={array_idx} Dir={os.getcwd()} ---"
    )
    log.info(f"Starting Sim ID: {cfg.sim_id}")

    out_dir = HydraConfig.get().runtime.output_dir
    with open(os.path.join(out_dir, "hardware_info.json"), "w") as f:
        json.dump(get_hardware_info(), f, indent=2)

    # 1. Reproducibility
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    seed_everything(cfg.main_seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Setup DGP for Simulations
    dataset_args, dim_z, poly_mix_weights = setup_dgp_args(cfg)
    torch.save(dataset_args, "dgp_args.pt")
    train_loader = DataLoader(
        SimDataset(
            num_draws=cfg.data.n_train,
            num_obs=[cfg.data.n_train for _ in range(cfg.data.n_pop)],
            **dataset_args,
            seed=cfg.data_seed,
        ),
        batch_size=cfg.data.batch_size,
        num_workers=0,
    )

    val_loader = DataLoader(
        SimDataset(
            num_draws=cfg.data.n_val,
            num_obs=[cfg.data.n_val for _ in range(cfg.data.n_pop)],
            **dataset_args,
            seed=cfg.data_seed + 10000,
        ),
        batch_size=cfg.data.batch_size,
        num_workers=0,
    )

    # 3. Model Setup
    dim_hu = cfg.model.dim_v + cfg.model.dim_w
    if poly_mix_weights is not None:
        poly_mix_weights = poly_mix_weights.to(device)
    encoder = get_encoder(cfg, dim_z, dim_hu, poly_mix_weights, device)
    decoder = get_decoder(cfg, dim_z, dim_hu, device)

    predictor = nn.Sequential(
        nn.Linear(cfg.model.dim_w, cfg.model.dim_w),
        nn.Tanh(),
        nn.Linear(cfg.model.dim_w, 1),
    )

    model = LitAutoEncoder(
        encoder=encoder,
        decoder=decoder,
        predictor=predictor,
        dim_z=dim_z,
        dim_v=cfg.model.dim_v,
        dim_w=cfg.model.dim_w,
        lam1=cfg.loss.lam1,
        lam2=cfg.loss.lam2,
        lam3=cfg.loss.lam3,
        inv_loss_type=cfg.loss.inv_loss_type,
        inv_ker_poly_degree=cfg.loss.inv_ker_poly_degree,
        inv_ker_rbf_sigma=cfg.loss.inv_ker_rbf_sigma,
        inv_ker_rbf_scales=list(cfg.loss.inv_ker_rbf_scales),
        rbf_global_sigma=cfg.loss.rbf_global_sigma,
        ind_loss_type=cfg.loss.ind_loss_type,
        ind_ker_poly_degree=cfg.loss.ind_ker_poly_degree,
        ind_ker_rbf_sigma=cfg.loss.ind_ker_rbf_sigma,
        ind_ker_rbf_scales=list(cfg.loss.ind_ker_rbf_scales),
        dim_v_true=cfg.data.dim_v_true,
        dim_w_true=cfg.data.dim_w_true,
    )

    # 4. Logger & Trainer
    run_id = f"{cfg.exp_id}_ds{cfg.data_seed}_sim{cfg.sim_id}"
    try:
        custom_settings = wandb.Settings(init_timeout=300, _service_wait=120)
    except TypeError:
        custom_settings = wandb.Settings(init_timeout=300)
    logger = WandbLogger(
        project=f"rl4mr-{cfg.exp_id}",
        name=f"ds{cfg.data_seed}-sim{cfg.sim_id}",
        id=run_id,
        resume="allow",
        settings=custom_settings,
    )
    run_info = {
        "sim_id": cfg.sim_id,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID", "local"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", "0"),
        "node": os.environ.get("SLURMD_NODENAME", "unknown"),
    }
    with open("run_identity.json", "w") as f:
        json.dump(run_info, f)
    if logger and hasattr(logger, "experiment"):
        try:
            exp = logger.experiment
            if callable(exp):
                exp = exp()
            if hasattr(exp, "config") and hasattr(exp.config, "update"):
                exp.config.update(run_info, allow_val_change=True)
        except Exception as e:
            print(f"Warning: Could not update wandb config: {e}")

    monitor_loader = train_loader

    iv_callback = IVTrackingCallback(
        dataloader=monitor_loader,
        iv_wrapper_func=iv_wrapper,  # Pass your function reference
        z_regex="^hW",  # We want to track the LEARNED hW, not original Z
        every_n_epochs=5,  # Run every 5 epochs to save time
    )

    # 1. Best Total Loss (Primary) - Also saves 'last.ckpt'
    checkpoint_total = ModelCheckpoint(
        dirpath="checkpoints",
        save_top_k=1,
        filename="best-total-{epoch:02d}",
        monitor="val/tot_loss",
        mode="min",
        save_last=True,  # Keep the very last epoch here
    )

    # 2. Best Reconstruction (MSE)
    checkpoint_best_rec = ModelCheckpoint(
        dirpath="checkpoints",
        save_top_k=1,
        filename="best-mse-{epoch:02d}",  # Unique Name
        monitor="val/rec_loss",
        mode="min",
    )

    # 3. Best Invariance
    checkpoint_best_inv = ModelCheckpoint(
        dirpath="checkpoints",
        save_top_k=1,
        filename="best-inv-{epoch:02d}",  # Unique Name
        monitor="val/inv_loss",
        mode="min",
    )

    # 4. Best Independence
    checkpoint_best_ind = ModelCheckpoint(
        dirpath="checkpoints",
        save_top_k=1,
        filename="best-ind-{epoch:02d}",  # Unique Name
        monitor="val/ind_loss",
        mode="min",
    )

    # 5. Periodic Snapshot (Every 250 epochs)
    checkpoint_periodic = ModelCheckpoint(
        dirpath="checkpoints",
        save_top_k=-1,  # Keep all snapshots
        every_n_epochs=250,
        filename="periodic-{epoch:02d}",
        monitor=None,
        save_last=False,  # Redundant, handled by checkpoint_total
    )

    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        logger=logger,
        log_every_n_steps=20,  # currently 20 batches per epoch during training
        enable_progress_bar=False,
        callbacks=[
            checkpoint_total,
            checkpoint_best_rec,
            checkpoint_best_inv,
            checkpoint_best_ind,
            checkpoint_periodic,
            SaveValMetricsCallback(out_dir="metrics"),
            RuntimeInfoCallback(),
            iv_callback,
        ],
        accelerator="auto",
        # fast_dev_run=True,
        # devices=1 if torch.cuda.is_available() else None,
    )

    # 5. Training
    resume_ckpt = cfg.get("resume_from_checkpoint", None)

    if resume_ckpt and os.path.exists(resume_ckpt):
        # --- 1. Rename the LAST checkpoint ---
        checkpoint = torch.load(
            resume_ckpt, map_location="cpu", weights_only=False
        )
        current_epoch = checkpoint.get("epoch", 0)

        dir_name = os.path.dirname(resume_ckpt)
        new_last_name = os.path.join(
            dir_name, f"last-epoch-{current_epoch}.ckpt"
        )
        os.rename(resume_ckpt, new_last_name)
        print(f"Renamed {resume_ckpt} to {new_last_name}")
        resume_ckpt = new_last_name

        # --- 2. Rename the BEST checkpoint ---
        # Find any file starting with 'best-' in the checkpoints folder
        best_files = glob.glob(os.path.join(dir_name, "best-*.ckpt"))
        for bf in best_files:
            # Avoid renaming a file that already has 'segment' in the name
            if "segment" not in bf:
                # Example: best-epoch=150.ckpt -> best-segment-before-epoch-245.ckpt
                new_best_name = os.path.join(
                    dir_name, f"best-segment-before-epoch-{current_epoch}.ckpt"
                )
                os.rename(bf, new_best_name)
                print(f"Preserved old best: {bf} -> {new_best_name}")

    start_time = time.time()
    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_ckpt)
    end_time = time.time()
    duration = end_time - start_time
    final_epoch = trainer.current_epoch
    stats_file = "training_stats.json"
    new_entry = {
        "epoch_reached": final_epoch,
        "train_time_sec": duration,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    # Append to the file instead of overwriting
    # We use 'a' for append mode and write one JSON object per line
    with open(stats_file, "a") as f:
        f.write(json.dumps(new_entry) + "\n")

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
