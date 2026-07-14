import warnings

import pytorch_lightning as pl
import torch
from nlb_tools.evaluation import (
    bits_per_spike,
    eval_psth,
    speed_tp_correlation,
    velocity_decoding,
)
from scipy.linalg import LinAlgWarning

from ..metrics import ExpSmoothedMetric
from ..utils import send_batch_to_device


class NLBEvaluation(pl.Callback):
    """Computes and logs all evaluation metrics for the Neural Latents
    Benchmark to tensorboard. These include `co_bps`, `fp_bps`,
    `behavior_r2`, `psth_r2`, and `tp_corr`.

    To enable this functionality, install nlb_tools
    (https://github.com/neurallatents/nlb_tools).
    """

    def __init__(self, log_every_n_epochs=20, decoding_cv_sweep=False):
        """Initializes the callback.

        Parameters
        ----------
        log_every_n_epochs : int, optional
            The frequency with which to plot and log, by default 100
        decoding_cv_sweep : bool, optional
            Whether to run a cross-validated hyperparameter sweep to
            find optimal regularization values, by default False
        """
        self.log_every_n_epochs = log_every_n_epochs
        self.decoding_cv_sweep = decoding_cv_sweep
        self.smth_metrics = {}

    def on_validation_epoch_end(self, trainer, pl_module):
        """Logs plots at the end of the validation epoch.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The trainer currently handling the model.
        pl_module : pytorch_lightning.LightningModule
            The model currently being trained.
        """
        # Skip evaluation for most epochs to save time
        if (trainer.current_epoch % self.log_every_n_epochs) != 0:
            return
        # Get the dataloaders
        pred_dls = trainer.datamodule.predict_dataloader()
        s = 0
        val_dataloader = pred_dls[s]["valid"]
        train_dataloader = pred_dls[s]["train"]
        # Create object to store evaluation metrics
        metrics = {}
        # Get entire validation dataset from datamodule
        (input_data, recon_data, *_), (behavior,) = trainer.datamodule.valid_data[s]
        recon_data = recon_data.detach().cpu().numpy()
        behavior = behavior.detach().cpu().numpy()
        # Pass the data through the model
        # TODO: Replace this with Trainer.predict? Hesitation is that switching to
        # Trainer.predict for posterior sampling is inefficient because we can't
        # tell it how many forward passes to use.
        rates = []
        for batch in val_dataloader:
            batch = send_batch_to_device({s: batch}, pl_module.device)
            output = pl_module.predict_step(batch, None, sample_posteriors=False)[s]
            rates.append(output.output_params)
        rates = torch.cat(rates).detach().cpu().numpy()
        # Compute co-smoothing bits per spike
        _, n_obs, n_heldin = input_data.shape
        heldout = recon_data[:, :n_obs, n_heldin:]
        rates_heldout = rates[:, :n_obs, n_heldin:]
        co_bps = bits_per_spike(rates_heldout, heldout)
        metrics["nlb/co_bps"] = max(co_bps, -1.0)
        # Compute forward prediction bits per spike
        forward = recon_data[:, n_obs:]
        rates_forward = rates[:, n_obs:]
        fp_bps = bits_per_spike(rates_forward, forward)
        metrics["nlb/fp_bps"] = max(fp_bps, -1.0)
        # Get relevant training dataset from datamodule
        _, (train_behavior,) = trainer.datamodule.train_data[s]
        train_behavior = train_behavior.detach().cpu().numpy()
        # Get model predictions for the training dataset
        train_rates = []
        for batch in train_dataloader:
            batch = send_batch_to_device({s: batch}, pl_module.device)
            output = pl_module.predict_step(batch, None, sample_posteriors=False)[s]
            train_rates.append(output.output_params)
        train_rates = torch.cat(train_rates).detach().cpu().numpy()
        # Get firing rates for observed time points
        rates_obs = rates[:, :n_obs]
        train_rates_obs = train_rates[:, :n_obs]
        # Compute behavioral decoding performance
        if behavior.ndim < 3:
            tp_corr = speed_tp_correlation(heldout, rates_obs, behavior)
            metrics["nlb/tp_corr"] = tp_corr
        else:
            with warnings.catch_warnings():
                # Ignore LinAlgWarning from early in training
                warnings.filterwarnings("ignore", category=LinAlgWarning)
                behavior_r2 = velocity_decoding(
                    train_rates_obs,
                    train_behavior,
                    trainer.datamodule.train_decode_mask,
                    rates_obs,
                    behavior,
                    trainer.datamodule.valid_decode_mask,
                    self.decoding_cv_sweep,
                )
            metrics["nlb/behavior_r2"] = max(behavior_r2, -1.0)
        # Compute PSTH reconstruction performance
        if hasattr(trainer.datamodule, "psth"):
            psth = trainer.datamodule.psth
            cond_idxs = trainer.datamodule.valid_cond_idx
            jitter = getattr(trainer.datamodule, "valid_jitter", None)
            psth_r2 = eval_psth(psth, rates_obs, cond_idxs, jitter)
            metrics["nlb/psth_r2"] = max(psth_r2, -1.0)
        # Compute smoothed metrics
        for k, v in metrics.items():
            if k not in self.smth_metrics:
                self.smth_metrics[k] = ExpSmoothedMetric(coef=0.7)
            self.smth_metrics[k].update(v, 1)
        # Log actual and smoothed metrics
        pl_module.log_dict(
            {
                **metrics,
                **{k + "_smth": m.compute() for k, m in self.smth_metrics.items()},
            }
        )
        # Reset the smoothed metrics (per-step aggregation not necessary)
        [m.reset() for m in self.smth_metrics.values()]
