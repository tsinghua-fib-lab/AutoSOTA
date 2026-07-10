from typing import Optional

import torch
import torch.nn.functional as F
from lightning import LightningModule


class PCE(LightningModule):
    def __init__(
        self,
        architecture: list[torch.nn.Sequential],
        iters: int,
        e_lr: float,
        w_lr: float,
    ):
        super().__init__()

        self.save_hyperparameters()

        # Store all layers and register them properly as parameters
        self.layers = torch.nn.ModuleList(architecture)

        self.errors = None  # Needs to be initialized with an input x

        self.iters = iters
        self.e_lr = e_lr
        self.w_lr = w_lr

    def y_pred(self, x: torch.Tensor):
        s_i = x
        for e_i, layer_i in zip(self.errors + [0.0], self.layers):
            s_i = e_i + layer_i(s_i)
        return s_i

    def class_loss(self, y_pred: torch.Tensor, y: torch.Tensor):
        # For error optimization: reduction = "sum"
        # For weight optimization: reduction = "mean"
        # (but we just manually divide by batch_size in training_step)
        return 0.5 * F.mse_loss(y_pred, y, reduction="sum")

    def configure_optimizers(self):
        return torch.optim.Adam(self.layers.parameters(), lr=self.w_lr)

    def E(self, x: torch.Tensor, y: torch.Tensor):
        """
        Calculates the energy using only the errors

        DANGER: don't use this E to train the params, or you'll be backpropping!
        """
        E_errors = 0.5 * sum(torch.linalg.vector_norm(e, ord=2, dim=None) ** 2 for e in self.errors)

        return E_errors + self.class_loss(self.y_pred(x), y)

    def E_local(self, x: torch.Tensor, y: torch.Tensor):
        """
        Calculates the energy using only local interactions (no backprop!)
        Specifically, it infers the states from the errors and returns the states-based energy.

        By construction, the value is exactly equal to the energy using only errors,
        but its computational graph is different and enforces local weight updates.
        """
        E = 0.0
        s_i = x
        for e_i, layer_i in zip(self.errors, self.layers[:-1]):
            s_i_pred = layer_i(s_i)  # tracking the computational graph...
            s_i = (e_i + s_i_pred).detach()  # detach => no backprop!

            E += 0.5 * F.mse_loss(s_i_pred, s_i, reduction="sum")

        y_pred = self.layers[-1](s_i)
        return E + self.class_loss(y_pred, y)

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        if y is None:
            # Inference is easy: all errors are zero
            self.errors = [0.0] * (len(self.layers) - 1)

        else:  # Training is more difficult
            self.minimize_error_energy(x, y)

        # We don't need to return anything during training.
        # At inference, we can easily access the error values through self.errors

    def minimize_error_energy(self, x: torch.Tensor, y: torch.Tensor):
        """Novel PC energy minimization, using errors instead of states"""

        # Deactivate autograd on params
        for p in self.layers.parameters():
            p.requires_grad_(False)

        # Initialize self.errors to the right shape using a forward pass
        self.init_zero_errors(x)

        # Minimize energy via the errors
        error_optim = torch.optim.SGD(self.errors, lr=self.e_lr)
        for _ in range(self.iters):
            error_optim.zero_grad()
            E = self.E(x, y)
            E.backward()
            error_optim.step()

        # Log final energy
        self.log("E_errors", E, prog_bar=True)

        # Re-activate autograd on params
        for p in self.layers.parameters():
            p.requires_grad_(True)

    @torch.no_grad()
    def init_zero_errors(self, x: torch.Tensor):
        """Creates trainable errors via a feedforward pass"""
        self.errors = [
            torch.zeros_like(x := layer_i(x), requires_grad=True) for layer_i in self.layers[:-1]
        ]

    def on_fit_start(self):
        # Store batch_size for easy access
        self.batch_size = self.trainer.datamodule.batch_size

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx):
        self.forward(x=batch["img"], y=batch["y"])

        # IMPORTANT: calculate the energy using the states!
        # (needed for local weight updates + good sanity check)
        E_final = self.E_local(x=batch["img"], y=batch["y"])

        self.log("E_local", E_final, prog_bar=True)

        # For weight optimization, we must average E over the batch.
        return E_final / self.batch_size  # = loss function for Lightning to minimize wrt params

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx):
        self.forward(x=batch["img"])

        # Log the dataset-specific metrics
        node_dict = {"y": self.y_pred(x=batch["img"])}
        self.log_dict(
            self.trainer.datamodule.metrics(node_dict, batch, prefix="val_"), prog_bar=True
        )

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx):
        self.forward(x=batch["img"])

        # Log the dataset-specific metrics
        node_dict = {"y": self.y_pred(x=batch["img"])}
        self.log_dict(
            self.trainer.datamodule.metrics(node_dict, batch, prefix="test_"), prog_bar=True
        )

    def predict_step(self, batch: dict[str, torch.Tensor], batch_idx):
        self.forward(x=batch["img"])

        print("Loss =", self.class_loss(self.y_pred(x=batch["img"]), y=batch["y"]).item())
        return {"y": self.y_pred(x=batch["img"])}

    ### STATE OPTIMIZATION ###
    # Below, we define PC in its regular, state-based formulation.
    # This is only for comparison to the error-based formulation above.

    def get_states_from_errors(self, x: torch.Tensor):
        """Returns the states corresponding to the errors, including y_pred"""
        return [(x := e_i + layer_i(x)) for e_i, layer_i in zip(self.errors + [0.0], self.layers)]

    def E_states_only(self, x: torch.Tensor, y: torch.Tensor, states: list[torch.Tensor]):
        """
        Calculates the energy using only the states, which need to be given as inputs.
        No errors are used here.
        """

        def half_mse_loss(y_pred, y):
            return 0.5 * F.mse_loss(y_pred, y, reduction="sum")

        losses = [half_mse_loss] * len(states) + [self.class_loss]
        states = [x] + states + [y]

        return sum(
            loss(layer(s_i), s_ip1)
            for s_i, s_ip1, layer, loss in zip(states[:-1], states[1:], self.layers, losses)
        )

    def minimize_state_energy(self, x: torch.Tensor, y: torch.Tensor, iters: int, s_lr: float):
        """Classical PC energy minimization using states"""

        # Deactivate autograd on params
        for p in self.layers.parameters():
            p.requires_grad_(False)

        # Initialize states using a feedforward pass
        def ff_init(s):
            return [(s := layer(s).detach().requires_grad_(True)) for layer in self.layers[:-1]]

        states = ff_init(x)

        # Minimize energy via the states
        state_optim = torch.optim.SGD(states, lr=s_lr)
        for _ in range(iters):
            state_optim.zero_grad()
            E = self.E_states_only(x, y, states)
            E.backward()
            state_optim.step()

        # Re-activate autograd on params
        for p in self.layers.parameters():
            p.requires_grad_(True)

        # No need to store in self.states, just return for later use in callbacks
        return states
