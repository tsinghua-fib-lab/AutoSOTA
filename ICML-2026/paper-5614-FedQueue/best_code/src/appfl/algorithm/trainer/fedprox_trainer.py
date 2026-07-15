import torch
from torch.nn import Module
from omegaconf import DictConfig
from typing import Optional, Any, Union, Dict, OrderedDict
from torch.utils.data import Dataset

from appfl.algorithm.trainer.vanilla_trainer import VanillaTrainer


class FedProxTrainer(VanillaTrainer):
    """
    FedProx trainer: adds a proximal penalty to the local objective to limit
    client drift under heterogeneous data.

        min_w  F_k(w) + (mu/2) * ||w - w^t||^2

    where w^t is the global model received at the start of each round.

    Extra train_configs field:
      mu (float): proximal coefficient (default 0.01)

    All other fields (mode, num_local_steps/num_local_epochs, optim, etc.)
    are inherited from VanillaTrainer.
    """

    def __init__(
        self,
        model: Optional[Module] = None,
        loss_fn: Optional[Module] = None,
        metric: Optional[Any] = None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        train_configs: DictConfig = DictConfig({}),
        logger: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            metric=metric,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_configs=train_configs,
            logger=logger,
            **kwargs,
        )
        self.mu = float(self.train_configs.get("mu", 0.01))
        self._snapshot_global_params()
        self._global_params_device = None

    def _snapshot_global_params(self):
        """Save a CPU copy of all trainable parameter tensors as the proximal anchor."""
        self.global_params = {
            name: param.detach().cpu().clone()
            for name, param in self.model.named_parameters()
        }

    def load_parameters(self, params: Union[Dict, OrderedDict, Any]):
        """Load the new global model and refresh the proximal anchor."""
        super().load_parameters(params)
        self._snapshot_global_params()

    def train(self, **kwargs):
        """Move global params to the training device, run local training, then release."""
        if self.global_params is not None and self.mu > 0:
            self._global_params_device = {
                name: p.to(self.device) for name, p in self.global_params.items()
            }
        else:
            self._global_params_device = None
        super().train(**kwargs)
        self._global_params_device = None

    def _train_batch(self, optimizer, data, target):
        """One gradient step with the FedProx proximal regularisation."""
        device = self.device
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = self.model(data)
        loss = self.loss_fn(output, target)

        if self._global_params_device is not None and self.mu > 0:
            prox = torch.tensor(0.0, device=device)
            for name, param in self.model.named_parameters():
                if name in self._global_params_device:
                    prox = prox + torch.sum(
                        (param - self._global_params_device[name]) ** 2
                    )
            loss = loss + (self.mu / 2.0) * prox

        loss.backward()
        if getattr(self.train_configs, "clip_grad", False):
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.train_configs.clip_value,
                norm_type=self.train_configs.clip_norm,
            )
        optimizer.step()
        return (
            loss.item(),
            output.detach().cpu().numpy(),
            target.detach().cpu().numpy(),
        )
