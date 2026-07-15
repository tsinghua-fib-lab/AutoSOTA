import torch
import torch.nn as nn
from omegaconf import DictConfig
from lightning.pytorch import seed_everything
from hydra.utils import instantiate
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Any

from data import create_dataloaders
from data.loader import create_last_layer_dataloaders
from train.oe import OperatorEncoder, train_oe
from train.utils import (
    optional_ckpt_path,
    finish_wandb,
    get_dataset_data_name_lower,
    make_checkpoint_callback,
    make_run_name,
    make_trainer,
    make_wandb_logger,
    should_share_xy_normalizer,
    split_dataset_result,
    strip_name_keys,
)
from train.diffusion.dm import DiffusionModel


class LastLayerDataset(Dataset):
    """Pre-computes last-layer weights (w) using OperatorEncoder (no ridge)."""

    def __init__(self, operatorencoder, dataloader, device="cuda"):
        super().__init__()
        self.data = []
        operatorencoder.to(device)
        operatorencoder.eval()

        print(f"Pre-computing optimal weights for {len(dataloader)} batches...")
        with torch.no_grad():
            for x, y in tqdm(dataloader):
                x, y = x.to(device), y.to(device)
                if not hasattr(operatorencoder, "encode_weights"):
                    raise ValueError(
                        "LastLayerDataset expects an OperatorEncoder with encode_weights(x,y)."
                    )
                w = operatorencoder.encode_weights(x, y)
                x_cpu, y_cpu, w_cpu = x.cpu(), y.cpu(), w.cpu()
                for i in range(x.shape[0]):
                    self.data.append((x_cpu[i], y_cpu[i], w_cpu[i]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DLL(DiffusionModel):
    """
    Diffusion Last Layer.

    Diffusion modeling happens on the weights (w) of the last layer.
    Inference: x -> features; x -> diffusion model -> w; y = features * w.
    """

    def __init__(
        self,
        operatorencoder: nn.Module,
        model: nn.Module,
        optimizer_config: DictConfig,
        scheduler_config: DictConfig | None = None,
        T: int = 100,
        samples_per_example: int = 1,
        ema_decay: float = 0.9999,
        x_normalizer: Any = None,
        y_normalizer: Any = None,
        test_samples_per_example: int | None = None,
        rollout_k_chunk: int = 8,
        rollout_traj_chunk: int | None = None,
        stochastic_n_chunk: int = 32,
        stochastic_k_chunk: int = 4,
        viz_num_trajectories_default: int = 3,
        viz_horizon_percentiles_default: tuple = (1, 20, 40, 60, 80, 100),
        eval_mode: str | None = None,
        eval_solver: str = "euler",
    ):
        if x_normalizer is None:
            x_normalizer = y_normalizer
        super().__init__(
            model=model,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
            T=T,
            samples_per_example=samples_per_example,
            ema_decay=ema_decay,
            x_normalizer=x_normalizer,
            y_normalizer=y_normalizer,
            test_samples_per_example=test_samples_per_example,
            rollout_k_chunk=rollout_k_chunk,
            rollout_traj_chunk=rollout_traj_chunk,
            stochastic_n_chunk=stochastic_n_chunk,
            stochastic_k_chunk=stochastic_k_chunk,
            viz_num_trajectories_default=viz_num_trajectories_default,
            viz_horizon_percentiles_default=viz_horizon_percentiles_default,
            eval_mode=eval_mode,
            eval_solver=eval_solver,
        )

        self.operatorencoder = operatorencoder
        self.operatorencoder.eval()
        self.operatorencoder.requires_grad_(False)
        self.save_hyperparameters(
            ignore=[
                "operatorencoder",
                "model",
                "optimizer_config",
                "scheduler_config",
                "x_normalizer",
                "y_normalizer",
            ]
        )

    def training_step(self, batch, batch_idx):
        x, y, w = batch
        loss = self.compute_diffusion_loss(x, w)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, w = batch
        loss = self.compute_diffusion_loss(x, w)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        """
        Override DiffusionModel.test_step:
        - DLL dataloader yields (x, y, w) where w is the flow target (last-layer weights).
        - Base DiffusionModel.test_step expects (x_cond, y_true) and will crash unpacking.
        """
        x, y, w = batch

        loss = self.compute_diffusion_loss(x, w, use_ema=False)
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)

        return loss

    def _reconstruct_signal(self, features, w):
        """Projects weights w back to signal y using features: y = Features * w."""
        B, C = features.shape[0], features.shape[1]
        Cy = int(w.shape[-1])
        if features.ndim == 3:  # 1D
            X = features.permute(0, 2, 1)
        elif features.ndim == 4:  # 2D
            X = features.permute(0, 2, 3, 1).reshape(B, -1, C)
        else:
            raise ValueError(f"Unsupported feature shape: {tuple(features.shape)}")
        Y_hat_flat = torch.matmul(X, w)  # [B, N, Cy]
        if features.ndim == 3:
            y_hat = Y_hat_flat.permute(0, 2, 1)
        elif features.ndim == 4:
            H, W = features.shape[2], features.shape[3]
            y_hat = Y_hat_flat.view(B, H, W, Cy).permute(0, 3, 1, 2)

        return y_hat

    @torch.inference_mode()
    def generate_y_samples(
        self,
        x: torch.Tensor,
        num_samples: int = 1,
        use_ema: bool | None = None,
        *,
        num_steps: int | None = None,
        solver: str = "euler",
    ) -> torch.Tensor:
        """Generate samples for y by generating weights w, then reconstructing."""
        K = int(num_samples)
        B = x.shape[0]
        if K > 1:
            x_in = x.repeat_interleave(K, dim=0)  # [B*K, ...]
        else:
            x_in = x
        features = self.operatorencoder(x_in)
        w_gen = self.sample(
            cond=x_in, num_steps=num_steps, solver=solver, use_ema=use_ema
        )
        y_gen = self._reconstruct_signal(features, w_gen)
        if K > 1:
            y_gen = y_gen.view(B, K, *y_gen.shape[1:]).transpose(0, 1)
        else:
            y_gen = y_gen.unsqueeze(0)

        return y_gen


def train_dll(cfg: DictConfig) -> None:
    if "seed" in cfg:
        seed_everything(int(cfg.seed), workers=True)
    if getattr(cfg, "training_oe", None) is None:
        raise ValueError("Missing training config: expected `training_oe`.")
    oe_training_cfg = cfg.training_oe
    oe_module = train_oe(cfg, return_model=True)
    dataset_result = instantiate(cfg.dataset)
    (
        train_set,
        val_set,
        test_set,
        val_trjs,
        test_trjs,
        val_stochastic,
        test_stochastic,
    ) = split_dataset_result(dataset_result)
    data_name_lower = get_dataset_data_name_lower(cfg)
    share_xy_stage1 = should_share_xy_normalizer(
        data_name_lower=data_name_lower, training_cfg=oe_training_cfg
    )

    train_loader, val_loader, test_loader, xn, yn = create_dataloaders(
        train_set,
        val_set,
        test_set,
        batch_size=int(oe_training_cfg.batch_size),
        num_workers=int(oe_training_cfg.num_workers),
        normalization_mode=str(oe_training_cfg.normalization_mode),
        share_xy_normalizer=share_xy_stage1,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ll_train = LastLayerDataset(oe_module, train_loader, device=device)
    ll_val = LastLayerDataset(oe_module, val_loader, device=device)
    ll_test = LastLayerDataset(oe_module, test_loader, device=device)
    ll_train_loader, ll_val_loader, ll_test_loader, _, _, _ = (
        create_last_layer_dataloaders(
            ll_train,
            ll_val,
            ll_test,
            batch_size=int(cfg.training_dll.batch_size),
            num_workers=int(cfg.training_dll.num_workers),
            normalization_mode=str(cfg.training_dll.normalization_mode),
        )
    )
    viz_pcts = getattr(cfg.dll, "viz_horizon_percentiles", (1, 20, 40, 60, 80, 100))

    dll_module = DLL(
        operatorencoder=oe_module,
        model=instantiate(strip_name_keys(cfg.dll.model)),
        optimizer_config=cfg.dll.optimizer,
        scheduler_config=getattr(cfg.dll, "scheduler", None),
        T=int(cfg.dll.T),
        samples_per_example=int(cfg.dll.samples_per_example),
        ema_decay=getattr(cfg.dll, "ema_decay", 0.9999),
        x_normalizer=xn,
        y_normalizer=yn,
        test_samples_per_example=getattr(cfg.dll, "test_samples_per_example", None),
        rollout_k_chunk=int(getattr(cfg.dll, "rollout_k_chunk", 8)),
        rollout_traj_chunk=getattr(cfg.dll, "rollout_traj_chunk", None),
        stochastic_n_chunk=int(getattr(cfg.dll, "stochastic_n_chunk", 32)),
        stochastic_k_chunk=int(getattr(cfg.dll, "stochastic_k_chunk", 4)),
        viz_num_trajectories_default=int(getattr(cfg.dll, "viz_num_trajectories", 3)),
        viz_horizon_percentiles_default=viz_pcts,
        eval_mode=getattr(getattr(cfg, "evaluation", None), "mode", None),
        eval_solver=getattr(cfg.dll, "eval_solver", "euler"),
    )
    if val_trjs is not None:
        dll_module.val_trajectories = val_trjs
    if test_trjs is not None:
        dll_module.test_trajectories = test_trjs
    if val_stochastic is not None:
        dll_module.val_stochastic = val_stochastic
    if test_stochastic is not None:
        dll_module.test_stochastic = test_stochastic

    model_cfg = getattr(getattr(cfg, "dll", None), "model", None)
    model_name = (
        getattr(model_cfg, "model_name", None)
        or getattr(model_cfg, "mode_name", None)
        or getattr(cfg, "model_name", "DLL")
    )
    run_name = make_run_name(model_name, "DLL")
    data_name = (
        getattr(getattr(cfg, "dataset", None), "data_name", "Exp")
        if getattr(cfg, "dataset", None)
        else "Exp"
    )
    project_name = f"{data_name}_GEN"
    wandb_logger = make_wandb_logger(cfg, project_name=project_name, run_name=run_name)
    eval_mode = DiffusionModel._normalize_eval_mode(
        getattr(getattr(cfg, "evaluation", None), "mode", None)
    )
    if eval_mode == "stochastic":
        if val_stochastic is None or test_stochastic is None:
            raise ValueError(
                "evaluation.mode='stochastic' requires dataset extras: val_stochastic and test_stochastic."
            )
        monitor_metric = "val_stochastic_ed"
    elif eval_mode == "rollout":
        if val_trjs is None or test_trjs is None:
            raise ValueError(
                "evaluation.mode='rollout' requires dataset extras: val_trjs and test_trjs."
            )
        monitor_metric = "val_rollout_nrmse"
    else:
        monitor_metric = "val_loss"
    dll_cb = make_checkpoint_callback(
        project_name=project_name,
        run_name=run_name,
        filename_base="DLL",
        filename="DLL-best-{epoch:02d}",
        monitor=monitor_metric,
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    trainer = make_trainer(
        training_cfg=cfg.training_dll,
        max_epochs=int(cfg.training_dll.epochs),
        train_loader_len=len(ll_train_loader),
        logger=wandb_logger,
        callbacks=[dll_cb],
    )

    # Check for pretrained DLL checkpoint
    dll_pretrained = getattr(cfg.dll, "pretrained_ckpt", None)
    dll_pretrained_path = optional_ckpt_path(dll_pretrained)

    if dll_pretrained_path is not None:
        if not dll_pretrained_path.is_file():
            raise FileNotFoundError(
                f"dll.pretrained_ckpt was provided but file not found: {dll_pretrained_path}"
            )
        print(f"Loading pretrained DLL from: {dll_pretrained_path}")
        ckpt = torch.load(str(dll_pretrained_path), map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"]
        # _data_shape buffer size depends on the specific data shape seen during
        # training; skip it to avoid size mismatch with the uninitialized buffer.
        state_dict.pop("_data_shape", None)
        dll_module.load_state_dict(state_dict, strict=False)
        print("Skipping training, running evaluation with pretrained DLL checkpoint...")
        # State dict already loaded; pass ckpt_path=None to avoid Lightning
        # re-loading the checkpoint (which would fail on _data_shape size mismatch)
        ckpt_path = None
    else:
        trainer.fit(dll_module, ll_train_loader, ll_val_loader)
        ckpt_path = getattr(dll_cb, "best_model_path", None) or "best"

    try:
        trainer.test(
            dll_module,
            dataloaders=ll_test_loader,
            ckpt_path=ckpt_path,
            weights_only=False,
        )
    except TypeError:
        trainer.test(dll_module, dataloaders=ll_test_loader, ckpt_path=ckpt_path)
    finish_wandb(wandb_logger)
