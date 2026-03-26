from collections import namedtuple

import torch
import torch.autograd as autograd
from torch import nn
from typing import Optional

from ..dictionary import Dictionary, JumpReluAutoEncoder
from ..trainers.trainer import (
    SAETrainer,
    get_lr_schedule,
    get_sparsity_warmup_fn,
    set_decoder_norm_to_unit_norm,
    remove_gradient_parallel_to_decoder_directions,
)


class RectangleFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return ((x > -0.5) & (x < 0.5)).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(x <= -0.5) | (x >= 0.5)] = 0
        return grad_input


class JumpReLUFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        ctx.save_for_backward(x, threshold, torch.tensor(bandwidth))
        return x * (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        x_grad = (x > threshold).float() * grad_output
        threshold_grad = (
            -(threshold / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None  # None for bandwidth


class StepFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        ctx.save_for_backward(x, threshold, torch.tensor(bandwidth))
        return (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        x_grad = torch.zeros_like(x)
        threshold_grad = (
            -(1.0 / bandwidth) * RectangleFunction.apply((x - threshold) / bandwidth) * grad_output
        )
        return x_grad, threshold_grad, None  # None for bandwidth


class JumpReluTrainer(nn.Module, SAETrainer):
    """
    Trains a JumpReLU autoencoder.

    Note does not use learning rate or sparsity scheduling as in the paper.
    """

    def __init__(
        self,
        steps: int,  # total number of steps to train for
        activation_dim: int,
        dict_size: int,
        layer: int,
        lm_name: str,
        dict_class=JumpReluAutoEncoder,
        seed: Optional[int] = None,
        # TODO: What's the default lr use in the paper?
        lr: float = 7e-5,
        bandwidth: float = 0.001,
        sparsity_penalty: float = 1.0,
        warmup_steps: int = 1000,  # lr warmup period at start of training and after each resample
        sparsity_warmup_steps: Optional[int] = 2000,  # sparsity warmup period at start of training
        decay_start: Optional[int] = None,  # decay learning rate after this many steps
        target_l0: float = 20.0,
        device: str = "cpu",
        wandb_name: str = "JumpRelu",
        submodule_name: Optional[str] = None,
    ):
        super().__init__()

        # TODO: Should just be args, and this should be commonised
        assert layer is not None, "Layer must be specified"
        assert lm_name is not None, "Language model name must be specified"
        self.lm_name = lm_name
        self.layer = layer
        self.submodule_name = submodule_name
        self.device = device
        self.steps = steps
        self.lr = lr
        self.seed = seed

        self.bandwidth = bandwidth
        self.sparsity_coefficient = sparsity_penalty
        self.warmup_steps = warmup_steps
        self.sparsity_warmup_steps = sparsity_warmup_steps
        self.decay_start = decay_start
        self.target_l0 = target_l0

        # TODO: Better auto-naming (e.g. in BatchTopK package)
        self.wandb_name = wandb_name

        # TODO: Why not just pass in the initialised autoencoder instead?
        self.ae = dict_class(
            activation_dim=activation_dim,
            dict_size=dict_size,
            device=device,
        ).to(self.device)

        # Parameters from the paper
        self.optimizer = torch.optim.Adam(self.ae.parameters(), lr=lr, betas=(0.0, 0.999), eps=1e-8)

        lr_fn = get_lr_schedule(
            steps,
            warmup_steps,
            decay_start,
            resample_steps=None,
            sparsity_warmup_steps=sparsity_warmup_steps,
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

        self.sparsity_warmup_fn = get_sparsity_warmup_fn(steps, sparsity_warmup_steps)

        # Purely for logging purposes
        self.dead_feature_threshold = 10_000_000
        self.num_tokens_since_fired = torch.zeros(dict_size, dtype=torch.long, device=device)
        self.dead_features = -1
        self.logging_parameters = ["dead_features"]

    def loss(self, x: torch.Tensor, step: int, logging=False, **_):
        # Note: We are using threshold, not log_threshold as in this notebook:
        # https://colab.research.google.com/drive/1PlFzI_PWGTN9yCQLuBcSuPJUjgHL7GiD#scrollTo=yP828a6uIlSO
        # I had poor results when using log_threshold and it would complicate the scale_biases() function

        sparsity_scale = self.sparsity_warmup_fn(step)
        x = x.to(self.ae.W_enc.dtype)

        pre_jump = x @ self.ae.W_enc + self.ae.b_enc
        f = JumpReLUFunction.apply(pre_jump, self.ae.threshold, self.bandwidth)

        active_indices = f.sum(0) > 0
        did_fire = torch.zeros_like(self.num_tokens_since_fired, dtype=torch.bool)
        did_fire[active_indices] = True
        self.num_tokens_since_fired += x.size(0)
        self.num_tokens_since_fired[active_indices] = 0
        self.dead_features = (
            (self.num_tokens_since_fired > self.dead_feature_threshold).sum().item()
        )

        recon = self.ae.decode(f)

        recon_loss = (x - recon).pow(2).sum(dim=-1).mean()
        l0 = StepFunction.apply(f, self.ae.threshold, self.bandwidth).sum(dim=-1).mean()

        sparsity_loss = (
            self.sparsity_coefficient * ((l0 / self.target_l0) - 1).pow(2) * sparsity_scale
        )
        loss = recon_loss + sparsity_loss

        if not logging:
            return loss
        else:
            return namedtuple("LossLog", ["x", "recon", "f", "losses"])(
                x,
                recon,
                f,
                {
                    "l2_loss": recon_loss.item(),
                    "loss": loss.item(),
                },
            )

    def update(self, step, x):
        x = x.to(self.device)
        loss = self.loss(x, step=step)
        loss.backward()

        # We must transpose because we are using nn.Parameter, not nn.Linear
        self.ae.W_dec.grad = remove_gradient_parallel_to_decoder_directions(
            self.ae.W_dec.T, self.ae.W_dec.grad.T, self.ae.activation_dim, self.ae.dict_size
        ).T
        torch.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)

        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        # We must transpose because we are using nn.Parameter, not nn.Linear
        self.ae.W_dec.data = set_decoder_norm_to_unit_norm(
            self.ae.W_dec.T, self.ae.activation_dim, self.ae.dict_size
        ).T

        return loss.item()

    @property
    def config(self):
        return {
            "trainer_class": "JumpReluTrainer",
            "dict_class": "JumpReluAutoEncoder",
            "lr": self.lr,
            "steps": self.steps,
            "seed": self.seed,
            "activation_dim": self.ae.activation_dim,
            "dict_size": self.ae.dict_size,
            "device": self.device,
            "layer": self.layer,
            "lm_name": self.lm_name,
            "wandb_name": self.wandb_name,
            "submodule_name": self.submodule_name,
            "bandwidth": self.bandwidth,
            "sparsity_penalty": self.sparsity_coefficient,
            "sparsity_warmup_steps": self.sparsity_warmup_steps,
            "target_l0": self.target_l0,
        }
