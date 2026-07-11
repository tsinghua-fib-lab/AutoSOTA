from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # For logging
import os
import time
import logging
from typing import Dict, Any, Union, List, Tuple  # Added Tuple
from abc import ABC, abstractmethod  # For abstract base class
from copy import deepcopy
from collections import OrderedDict

from tqdm import tqdm

from .losses import NoisePredictionLoss, MechanicalPropertyLoss, GeneratedCurveMatchingLoss, GeneratedPeakMatchingLoss
from .optimizers import get_optimizer, get_lr_scheduler
from ..evaluation import compute_fid, compute_kid
from ..evaluation.metrics import calculate_r2, calculate_relative_l2_error, evaluate_mechanical_properties
from ..models.diffusion_model import SpacedDiffusion

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

################################################################################
#                              EMA Helper Functions                            #
################################################################################

def update_ema(ema_model: nn.Module, model: nn.Module, decay: float = 0.9999):
    """Move the parameters of ``ema_model`` towards those of ``model`` with the
    given ``decay`` rate (0 → copy weights exactly, 1 → do nothing)."""
    with torch.no_grad():
        ema_params = OrderedDict(ema_model.named_parameters())
        model_params = OrderedDict(model.named_parameters())
        for name, param in model_params.items():
            ema_params[name].mul_(decay).add_(param.data, alpha=1.0 - decay)


def requires_grad(model: nn.Module, flag: bool = True):
    """Globally enable/disable gradient computation for the parameters of ``model``."""
    for p in model.parameters():
        p.requires_grad = flag

################################################################################
#                                  BaseTrainer                                 #
################################################################################


class BaseTrainer(ABC):
    """
    Abstract Base Class for training generative models.
    """

    def __init__(
        self,
        model: nn.Module,  # Can be a single model or a dict of models (e.g., for GANs)
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: DataLoader,
        optimizer_config: Union[Dict[str, Any], List[Dict[str, Any]]],  # Single or list for GANs
        scheduler_config: Union[Dict[str, Any], List[Dict[str, Any]], None],  # Single, list, or None
        loss_config: Dict[str, Any],
        epochs: int,
        device: Union[str, torch.device],
        model_keys: List[str] = None,
        log_dir: str = 'runs/',
        checkpoint_dir: str = 'checkpoints/',
        save_interval: int = 10,
        eval_interval: int = 1,
        use_ema: bool = False,
        ema_decay: float = 0.9999,
        accumulation_steps: int = 1,
        patience: int = float('inf'),
    ):
        self.device = torch.device(device)
        self.model = model  # This might be a single nn.Module or a dict of them
        if isinstance(self.model, nn.Module):
            self.model = self.model.to(self.device)
        elif isinstance(self.model, dict):  # For GANs
            for m_name, m_instance in self.model.items():
                self.model[m_name] = m_instance.to(self.device)

        # -------------------- EMA -------------------- #
        self.use_ema = use_ema and isinstance(self.model, nn.Module)
        self.ema_decay = ema_decay
        if self.use_ema:
            self.ema_model = deepcopy(self.model)
            self.ema_model.to(self.device)
            requires_grad(self.ema_model, False)
            update_ema(self.ema_model, self.model, decay=0)  # start in sync
            self.ema_model.eval()
            logging.info(
                f"EMA model created (decay={self.ema_decay}). Gradient tracking disabled."
            )
        elif use_ema and not isinstance(self.model, nn.Module):
            logging.warning(
                "EMA requested but not supported for dictionary-style (e.g. GAN) models."
            )

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.epochs = epochs
        self.start_epoch = 1  # For resuming training
        self.log_dir = log_dir
        self.label = datetime.now().strftime("%Y%m%d%H%M")
        self.checkpoint_dir = checkpoint_dir
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        self.loss_config = loss_config
        self.best_val_loss = float('inf')
        self.best_model = model
        self.model_keys = model_keys
        self.accumulation_steps = accumulation_steps
        self.patience = patience

        # --- Setup Optimizer(s) and LR Scheduler(s) ---
        self._setup_optimizers_and_schedulers(optimizer_config, scheduler_config)

        logging.info(f"{self.__class__.__name__} initialized.")

    # ---------------------------------------------------------------------
    # EMA helpers
    # ---------------------------------------------------------------------
    def _update_ema(self):
        if self.use_ema:
            update_ema(self.ema_model, self.model, self.ema_decay)

    # ---------------------------------------------------------------------
    # Optimizer / Scheduler setup (unchanged)
    # ---------------------------------------------------------------------
    def _setup_optimizers_and_schedulers(self, optimizer_config, scheduler_config):
        """Helper to set up single or multiple optimizers/schedulers."""
        if isinstance(optimizer_config, list):  # For GANs with multiple optimizers
            self.optimizer = {}
            self.scheduler = {} if scheduler_config else None
            model_names = list(self.model.keys())  # Assuming model is a dict for GANs

            if not isinstance(scheduler_config, list) and scheduler_config is not None:
                logging.warning(
                    "Optimizer config is a list, but scheduler config is not. Schedulers might not match optimizers."
                )

            for i, opt_conf in enumerate(optimizer_config):
                model_key = opt_conf.get('model_name', model_names[i])  # Assign optimizer to model
                if model_key not in self.model:
                    raise ValueError(
                        f"Model name '{model_key}' in optimizer config not found in self.model."
                    )

                self.optimizer[model_key] = get_optimizer(
                    self.model[model_key],
                    optimizer_name=opt_conf['name'],
                    learning_rate=opt_conf['lr'],
                    weight_decay=opt_conf.get('weight_decay', 0.0),
                )
                if self.scheduler and scheduler_config and i < len(scheduler_config):
                    sch_conf = scheduler_config[i]
                    self.scheduler[model_key] = get_lr_scheduler(
                        self.optimizer[model_key],
                        scheduler_name=sch_conf['name'],
                        **sch_conf.get('params', {}),
                    )
                elif self.scheduler is not None:
                    self.scheduler[model_key] = None  # No scheduler for this optimizer
            logging.info("Multiple optimizers and schedulers configured.")

        else:  # Single optimizer
            self.optimizer = get_optimizer(
                self.model,  # Assumes self.model is a single nn.Module
                optimizer_name=optimizer_config['name'],
                learning_rate=optimizer_config['lr'],
                weight_decay=optimizer_config.get('weight_decay', 0.0),
            )
            if scheduler_config and scheduler_config['name'] != 'none':
                self.scheduler = get_lr_scheduler(
                    self.optimizer,
                    scheduler_name=scheduler_config['name'],
                    **scheduler_config.get('params', {}),
                )
            else:
                self.scheduler = None
            logging.info("Single optimizer and scheduler configured.")

    # ---------------------------------------------------------------------
    # Abstract methods to implement per‑trainer logic
    # ---------------------------------------------------------------------
    @abstractmethod
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Runs a single training epoch. Must be implemented by subclasses."""

    @abstractmethod
    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Evaluates the model on the validation set. Must be implemented by subclasses."""

    @abstractmethod
    def test(self, *args, **kwargs) -> Dict[str, float]:
        """Evaluates the model on the test set. Must be implemented by subclasses."""

    @abstractmethod
    def sample(self, *args, **kwargs):
        pass

    # ---------------------------------------------------------------------
    # Public training loop
    # ---------------------------------------------------------------------
    def train(self):
        # --- Setup Logging and Checkpointing ---
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, self.__class__.__name__,
                                                         self.label))  # Subdirectory for each trainer type
        logging.info(f"Starting training with {self.__class__.__name__}...")

        patience_idx = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            train_losses = self._train_epoch(epoch)
            self._log_metrics(train_losses, epoch, 'Train')

            # Step learning rate scheduler(s) if not ReduceLROnPlateau
            self._step_schedulers(epoch, val_metric=None, is_plateau_type=False)
            patience_idx += 1

            if epoch % self.eval_interval == 0 or epoch == self.epochs:
                val_metrics = self._validate_epoch(epoch)
                self._log_metrics(val_metrics, epoch, 'Validation')

                # For ReduceLROnPlateau, step based on a validation metric
                main_val_metric_name = list(val_metrics.keys())[0] if val_metrics else None
                main_val_metric_value = val_metrics.get(main_val_metric_name) if main_val_metric_name else None

                self._step_schedulers(
                    epoch, val_metric=main_val_metric_value, is_plateau_type=True
                )

                # Save best model based on a primary validation loss/metric
                if (
                    main_val_metric_value is not None
                    and main_val_metric_value < self.best_val_loss
                ):
                    self.best_val_loss = main_val_metric_value
                    if self.use_ema:
                        self.best_model = deepcopy(self.ema_model)
                    else:
                        self.best_model = deepcopy(self.model)

                    # Reset patience_idx
                    patience_idx = 0
                    logging.info(
                        f"Epoch {epoch}: New best validation metric ({main_val_metric_name}: {self.best_val_loss:.4f}). Saving model."
                    )
                    self.save_checkpoint(epoch, is_best=True)

            if epoch % self.save_interval == 0 or epoch == self.epochs:
                self.save_checkpoint(epoch, is_best=False)  # Regular save

            if patience_idx >= self.patience:
                logging.info(f"Early stopping at epoch {epoch}.")
                break

        logging.info(f"Training finished for {self.__class__.__name__}.")
        self.writer.close()

    # ---------------------------------------------------------------------
    # Metric logging, scheduler stepping, checkpoints (updated for EMA)
    # ---------------------------------------------------------------------
    def _log_metrics(self, metrics: Dict[str, float], epoch: int, stage: str):
        """Logs metrics to console and TensorBoard."""
        log_message = f"{stage} Epoch [{epoch}/{self.epochs}]"
        for name, value in metrics.items():
            log_message += f", {name}: {value:.4f}"
            self.writer.add_scalar(f'{stage}_{self.__class__.__name__}/{name}', value, epoch)
        logging.info(log_message)
        if stage == 'Train' and self.scheduler:
            if isinstance(self.optimizer, dict):
                for i, (opt_name, opt_instance) in enumerate(self.optimizer.items()):
                    self.writer.add_scalar(
                        f'LearningRate/{opt_name}', opt_instance.param_groups[0]['lr'], epoch
                    )
            else:
                self.writer.add_scalar('LearningRate/main', self.optimizer.param_groups[0]['lr'], epoch)

    def _step_schedulers(self, epoch: int, val_metric: float = None, is_plateau_type: bool = False):
        if not self.scheduler:
            return
        if isinstance(self.scheduler, dict):
            for sch_name, sch_instance in self.scheduler.items():
                if sch_instance:
                    if isinstance(sch_instance, ReduceLROnPlateau):
                        if is_plateau_type and val_metric is not None:
                            sch_instance.step(val_metric)
                    elif not is_plateau_type:
                        sch_instance.step()
        else:
            if isinstance(self.scheduler, ReduceLROnPlateau):
                if is_plateau_type and val_metric is not None:
                    self.scheduler.step(val_metric)
            elif not is_plateau_type:
                self.scheduler.step()

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint_name = f'checkpoint_epoch_{epoch}.pt'
        if is_best:
            checkpoint_name = 'best_model.pt'
        checkpoint_path = os.path.join(self.checkpoint_dir, self.__class__.__name__, self.label, checkpoint_name)
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        state = {'epoch': epoch, 'best_val_loss': self.best_val_loss, 'label': self.label}

        if isinstance(self.model, nn.Module):
            state['model_state_dict'] = self.model.state_dict()
            state['optimizer_state_dict'] = self.optimizer.state_dict()
            if self.scheduler:
                state['scheduler_state_dict'] = self.scheduler.state_dict()
            if self.use_ema:
                state['ema_state_dict'] = self.ema_model.state_dict()
        elif isinstance(self.model, dict):
            state['model_state_dict'] = {name: m.state_dict() for name, m in self.model.items()}
            state['optimizer_state_dict'] = {name: o.state_dict() for name, o in self.optimizer.items()}
            if self.scheduler:
                state['scheduler_state_dict'] = {name: s.state_dict() for name, s in self.scheduler.items() if s}

        state['config'] = {
            'loss_config': self.loss_config,
            'use_ema': self.use_ema,
            'ema_decay': self.ema_decay,
        }

        try:
            torch.save(state, checkpoint_path)
            logging.info(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            logging.error(f"Error saving checkpoint {checkpoint_path}: {e}")

    def load_checkpoint(self, checkpoint_path: str):
        if not os.path.exists(checkpoint_path):
            logging.error(f"Checkpoint file not found: {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            if isinstance(self.model, nn.Module):
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if self.scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                if self.use_ema and 'ema_state_dict' in checkpoint:
                    self.ema_model.load_state_dict(checkpoint['ema_state_dict'])
            elif isinstance(self.model, dict):
                for name, model_state in checkpoint['model_state_dict'].items():
                    self.model[name].load_state_dict(model_state)
                for name, opt_state in checkpoint['optimizer_state_dict'].items():
                    self.optimizer[name].load_state_dict(opt_state)
                if self.scheduler and 'scheduler_state_dict' in checkpoint:
                    for name, sch_state in checkpoint['scheduler_state_dict'].items():
                        if self.scheduler.get(name) and sch_state:
                            self.scheduler[name].load_state_dict(sch_state)

            self.start_epoch = checkpoint['epoch'] + 1
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.label = checkpoint['label']

            logging.info(
                f"Checkpoint loaded from {checkpoint_path}. Resuming from epoch {self.start_epoch}."
            )
            return self.start_epoch
        except Exception as e:
            logging.error(f"Error loading checkpoint {checkpoint_path}: {e}")
            raise

    @staticmethod
    def evaluate_metrics(true_arr: Union[torch.Tensor, np.ndarray],
                         gen_arr: Union[torch.Tensor, np.ndarray],
                         peak_params = {'height': 0, 'distance': 10, 'prominence': 0.5}) -> Dict[str, float]:
        """
        Evalute the performance of generative models.
        :param true_arr: True force-extension curves. (batch_size, seq_len, in_channels)
        :param gen_arr: Generated force-extension curves. (batch_size, seq_len, in_channels)
        :param peak_params: Parameters for finding peaks of force-extension curves.
        :return: {fid, kid_mean, kid_std, r2_score_curves, rel_l2_error, {property_evaluation_metrics}}
        """
        # Convert input tensor to ndarray
        if isinstance(true_arr, torch.Tensor):
            true_arr = true_arr.detach().cpu().numpy()
        if isinstance(gen_arr, torch.Tensor):
            gen_arr = gen_arr.detach().cpu().numpy()

        # Calculate metrics
        fid = compute_fid(true_arr, gen_arr)
        kid_mean, kid_std = compute_kid(true_arr, gen_arr)
        r2_score_curves = calculate_r2(true_arr, gen_arr)
        rel_l2_error = calculate_relative_l2_error(true_arr, gen_arr)

        property_evaluation_metrics = evaluate_mechanical_properties(
            true_arr,
            gen_arr,
            property_extraction_params={'find_peaks': peak_params}
        )

        return {"fid": fid, "kid_mean": kid_mean, "kid_std": kid_std,
                "r2_score_curves": r2_score_curves, "rel_l2_error": rel_l2_error,
                "property_evaluation_metrics": property_evaluation_metrics}



################################################################################
#                                Sub‑class Trainers                            #
################################################################################


class DiffusionModelTrainer(BaseTrainer):
    """Trainer for ConditionalDiffusionModel."""

    def __init__(
        self,
        model: nn.Module,  # Specific model type
        diffusion: SpacedDiffusion,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: DataLoader,
        optimizer_config: Dict[str, Any],
        scheduler_config: Dict[str, Any],
        loss_config: Dict[str, Any],
        epochs: int,
        device: Union[str, torch.device],
        model_keys: List[str] = None,
        log_dir: str = 'runs/',
        checkpoint_dir: str = 'checkpoints/',
        save_interval: int = 10,
        eval_interval: int = 1,
        use_ema: bool = False,
        ema_decay: float = 0.9999,
        accumulation_steps: int = 1,
        patience: int = float('inf'),
    ):
        super().__init__(
            model,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            optimizer_config,
            scheduler_config,
            loss_config,
            epochs,
            device,
            model_keys,
            log_dir,
            checkpoint_dir,
            save_interval,
            eval_interval,
            use_ema,
            ema_decay,
            accumulation_steps,
            patience
        )
        # Diffusion‑specific
        self.diffusion = diffusion
        self.mech_prop_loss_fn = None
        self.curve_match_loss_fn = None

        self.noise_loss_weight = self.loss_config.get('noise_weight', 1.0)
        self.mech_prop_loss_weight = self.loss_config.get('mech_prop_weight', 0.0)
        self.curve_match_loss_weight = self.loss_config.get('curve_match_weight', 0.0)

        if self.mech_prop_loss_weight > 0:
            self.mech_prop_loss_fn = MechanicalPropertyLoss(
                self.loss_config.get('mech_prop_weights')
            )
            logging.info("Mechanical property loss enabled for DiffusionModelTrainer.")

        if self.curve_match_loss_weight > 0:
            self.curve_match_loss_fn = GeneratedCurveMatchingLoss(
                self.loss_config.get('curve_match_type', 'mse')
            )
            logging.info("Generated curve matching loss enabled for DiffusionModelTrainer.")

    def _construct_condition(self, sequence, condition):
        y = None
        if sequence.shape[-1] != 1 and condition.shape[-1] != 1:
            y = torch.cat([sequence, condition], dim=1) # Need to modify
        elif sequence.shape[-1] != 1:
            y = sequence
        elif condition.shape[-1] != 1:
            y = condition
        return y

    # ------------------------------------------------------------------
    # Training / validation epochs
    # ------------------------------------------------------------------
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        running_losses = {
            'total_loss': 0.0,
            'noise_loss': 0.0,
            'mech_prop_loss': 0.0,
        }
        num_batches = len(self.train_dataloader)

        start_time = time.time()

        for batch_idx, batch in tqdm(enumerate(self.train_dataloader), total=num_batches):
            x_0 = batch['fe_curve'].to(self.device)
            model_kwargs = dict()
            for key, value in batch.items():
                if key in self.model_keys:
                    model_kwargs[key] = value.to(self.device)
            t = torch.randint(0, self.diffusion.num_timesteps, (x_0.shape[0],), device=self.device)

            self.optimizer.zero_grad()

            # Noise prediction loss
            loss_dict = self.diffusion.training_losses(self.model, x_0, t, model_kwargs)
            noise_loss = loss_dict['loss'].mean()
            current_total_loss = self.noise_loss_weight * noise_loss
            running_losses['noise_loss'] += noise_loss.item()

            # (Optional) Additional losses can go here ...
            if 'pred_x' in loss_dict:
                pred_x = loss_dict['pred_x']
                mech_prop_loss = MechanicalPropertyLoss(self.loss_config.get('mech_prop_weights'))(x_0, pred_x)
                current_total_loss += self.mech_prop_loss_weight * mech_prop_loss
                running_losses['mech_prop_loss'] += mech_prop_loss.item()

            current_total_loss.backward()
            self.optimizer.step()
            self._update_ema()  # <-- EMA update
            running_losses['total_loss'] += current_total_loss.item()

        epoch_losses = {
            name: (loss_sum / num_batches if num_batches > 0 else 0)
            for name, loss_sum in running_losses.items()
        }
        epoch_losses['time_seconds'] = time.time() - start_time
        return epoch_losses

    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        running_losses = {
            'total_loss': 0.0,
            'noise_loss': 0.0,
            'mech_prop_loss': 0.0,
        }
        num_batches = len(self.val_dataloader)
        start_time = time.time()

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader):
                x_0 = batch['fe_curve'].to(self.device)
                model_kwargs = dict()
                for key, value in batch.items():
                    if key in self.model_keys:
                        model_kwargs[key] = value.to(self.device)
                t = torch.randint(0, self.diffusion.num_timesteps, (x_0.shape[0],), device=self.device)

                loss_dict = self.diffusion.training_losses(self.model, x_0, t, model_kwargs)
                noise_loss = loss_dict['loss'].mean()
                current_total_loss = self.noise_loss_weight * noise_loss
                # (Optional) Additional losses can go here ...
                if 'x_start_' in loss_dict:
                    pred_x = loss_dict['x_start']
                    mech_prop_loss = GeneratedPeakMatchingLoss(self.loss_config.get('mech_prop_weights'))(x_0, pred_x)
                    current_total_loss += self.mech_prop_loss_weight * mech_prop_loss
                    running_losses['mech_prop_loss'] += mech_prop_loss.item()
                running_losses['noise_loss'] += noise_loss.item()
                running_losses['total_loss'] += current_total_loss.item()

        epoch_losses = {
            name: (loss_sum / num_batches if num_batches > 0 else 0)
            for name, loss_sum in running_losses.items()
        }
        epoch_losses['time_seconds'] = time.time() - start_time
        return epoch_losses

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        conditions,
        num_samples_per_input: int = 1,
        *,
        inter_step_save_path: str = None,
        max_batch: int | None = 128,
        use_ema_model: bool | None = None,
        use_ddim = True,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Generate curves with optional chunking & EMA support."""
        if num_samples_per_input < 1:
            raise ValueError("num_samples_per_input must be ≥ 1")

        # Which network to use -------------------------------------------------
        net = self.model
        net.eval()

        # Duplicate inputs when >1 samples / input
        for k, v in conditions.items():
            if num_samples_per_input > 1:
                conditions[k] = v.repeat_interleave(num_samples_per_input, dim=0)
            else:
                conditions[k] = v
            total = conditions[k].size(0)

        max_batch = max_batch or total
        outputs: list[torch.Tensor] = []

        for s in range(0, total, max_batch):
            e = min(s + max_batch, total)
            out_shape = (e - s, net.out_channels, net.seq_len)

            input_conditions = dict()
            for k, v in conditions.items():
                input_conditions[k] = v[s:e]

            if use_ddim:
                samples = self.diffusion.ddim_sample_loop(
                    net,
                    out_shape,
                    clip_denoised=False,
                    model_kwargs=input_conditions,
                    progress=True,
                    device=self.device,
                    eta=eta,
                    save_path=inter_step_save_path,
                )
            else:
                samples = self.diffusion.p_sample_loop(
                    net,
                    out_shape,
                    clip_denoised=False,
                    model_kwargs=input_conditions,
                    progress=True,
                    device=self.device,
                )


            outputs.append(samples.cpu())

        return torch.cat(outputs, dim=0)

    # ------------------------------------------------------------------
    # Test‑set evaluation
    # ------------------------------------------------------------------
    def test(self, num_samples_per_input=1, use_ddim=True, **kwargs) -> (np.ndarray, np.ndarray, Dict[str, float]):
        """Generate on **test_dataloader** and compute FID / KID metrics."""
        self.model.eval()
        true_buf, gen_buf, pdb_buf = [], [], []

        logging.info("Generated {} batches of curves".format(len(self.test_dataloader)))

        with torch.no_grad():
            for batch in self.test_dataloader:
                true_curves = batch["fe_curve"].to(self.device)

                model_kwargs = dict()
                for key, value in batch.items():
                    if key in self.model_keys:
                        model_kwargs[key] = value.to(self.device)

                gen_curves = self.sample(model_kwargs, num_samples_per_input=num_samples_per_input,
                                         use_ddim=use_ddim, **kwargs)

                if num_samples_per_input > 1:
                    true_curves = true_curves.repeat_interleave(num_samples_per_input, dim=0)

                pdb_buf.extend(batch["pdb_id"])
                true_buf.append(true_curves.cpu())
                gen_buf.append(gen_curves.cpu())

        true_arr = torch.cat(true_buf, dim=0).transpose(1, 2).numpy()
        gen_arr = torch.cat(gen_buf, dim=0).transpose(1, 2).numpy()
        pdb_arr = np.array(pdb_buf)

        # Save generated curve
        checkpoint_path = os.path.join(self.checkpoint_dir, self.__class__.__name__, self.label)
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        np.save(os.path.join(checkpoint_path, 'true_curves'), true_arr)
        np.save(os.path.join(checkpoint_path, 'generated_curves'), gen_arr)
        np.save(os.path.join(checkpoint_path, 'test_pdb_ids.npy'), pdb_arr)

        logging.info("Generated curves saved in {}".format(checkpoint_path))

        return true_arr, gen_arr

    def predict(self,
                features: dict,
                chain: str,
                res_start_idx: int = None,
                res_end_idx: int = None,
                num_samples_per_input: int = 50,
                use_ddim: bool = True,
                save_path: str = None,
                **kwargs):
        dataset = self.test_dataloader.dataset.dataset
        condition = dataset.build_condition(features, chain, res_start_idx, res_end_idx) # From features

        self.model.eval()

        model_kwargs = dict()
        for key, value in condition.items():
            if key in self.model_keys:
                model_kwargs[key] = value.to(self.device)

        gen_curves = self.sample(model_kwargs, num_samples_per_input=num_samples_per_input,
                                         use_ddim=use_ddim, **kwargs)

        gen_curves = gen_curves.transpose(1, 2).cpu().numpy()

        if save_path:
            np.save(save_path, gen_curves)

        return gen_curves


