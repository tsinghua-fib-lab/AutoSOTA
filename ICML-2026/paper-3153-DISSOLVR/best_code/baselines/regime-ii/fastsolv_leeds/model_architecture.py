# ============================================================================
# FILE 4: model_architecture.py - FASTSOLV Model Definition
# ============================================================================

"""
Model architecture for FASTSOLV - adapted from the paper's code
"""

import torch
from torch.utils.data import Dataset as TorchDataset
from fastprop.data import inverse_standard_scale, standard_scale
from fastprop.model import fastprop as _fastprop
from typing import Literal
from types import SimpleNamespace


class SolubilityDataset(TorchDataset):
    """Dataset for solubility prediction with temperature gradients"""
    
    def __init__(
        self,
        solute_features: torch.Tensor,
        solvent_features: torch.Tensor,
        temperature: torch.Tensor,
        solubility: torch.Tensor,
        solubility_gradient: torch.Tensor,
    ):
        self.solute_features = solute_features
        self.solvent_features = solvent_features
        self.temperature = temperature
        self.solubility = solubility
        self.solubility_gradient = solubility_gradient
        self.length = len(solubility)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return (
            (
                self.solute_features[index],
                self.solvent_features[index],
                self.temperature[index],
            ),
            self.solubility[index],
            self.solubility_gradient[index],
        )


class Concatenation(torch.nn.Module):
    """Concatenate inputs along dimension 1"""
    def forward(self, batch):
        return torch.cat(batch, dim=1)


class ClampN(torch.nn.Module):
    """Clamp inputs to ±n standard deviations"""
    def __init__(self, n: float = 3.0) -> None:
        super().__init__()
        self.n = n

    def forward(self, batch: torch.Tensor):
        return torch.clamp(batch, min=-self.n, max=self.n)

    def extra_repr(self) -> str:
        return f"n={self.n}"


def _build_mlp(input_size, hidden_size, act_fun, num_layers):
    """Build multi-layer perceptron"""
    modules = []
    for i in range(num_layers):
        in_size = input_size if i == 0 else hidden_size
        modules.append(torch.nn.Linear(in_size, hidden_size))
        
        if act_fun == "sigmoid":
            modules.append(torch.nn.Sigmoid())
        elif act_fun == "tanh":
            modules.append(torch.nn.Tanh())
        elif act_fun == "relu":
            modules.append(torch.nn.ReLU())
        elif act_fun == "relu6":
            modules.append(torch.nn.ReLU6())
        elif act_fun == "leakyrelu":
            modules.append(torch.nn.LeakyReLU())
        else:
            raise TypeError(f"Unknown activation: {act_fun}")
    
    return modules


class fastpropSolubility(_fastprop):
    """
    FASTSOLV model with Sobolev training for temperature-dependent solubility prediction
    
    Architecture:
    - Concatenate solute features + solvent features + temperature
    - Input activation (sigmoid or clamp)
    - Hidden layers with activation
    - Single output for log solubility
    """
    
    def __init__(
        self,
        num_layers: int = 2,
        hidden_size: int = 1800,
        activation_fxn: Literal["relu", "leakyrelu"] = "relu",
        input_activation: Literal["sigmoid", "clamp3"] = "sigmoid",
        num_features: int = 1613,
        learning_rate: float = 0.0001,
        target_means: torch.Tensor = None,
        target_vars: torch.Tensor = None,
        solute_means: torch.Tensor = None,
        solute_vars: torch.Tensor = None,
        solvent_means: torch.Tensor = None,
        solvent_vars: torch.Tensor = None,
        temperature_means: torch.Tensor = None,
        temperature_vars: torch.Tensor = None,
    ):
        super().__init__(
            input_size=num_features,
            hidden_size=1,  # will be overwritten
            fnn_layers=0,  # will be overwritten
            readout_size=1,
            num_tasks=1,
            learning_rate=learning_rate,
            problem_type="regression",
            target_names=[],
            target_means=target_means,
            target_vars=target_vars,
        )
        
        # Remove parent's FNN and readout
        del self.fnn
        del self.readout
        self.readout = SimpleNamespace(out_features=1)

        # Register scaling parameters
        self.register_buffer("solute_means", solute_means)
        self.register_buffer("solute_vars", solute_vars)
        self.register_buffer("solvent_means", solvent_means)
        self.register_buffer("solvent_vars", solvent_vars)
        self.register_buffer("temperature_means", temperature_means)
        self.register_buffer("temperature_vars", temperature_vars)

        # Build custom FNN
        fnn_modules = [
            Concatenation(),
            ClampN(n=3.0) if input_activation == "clamp3" else torch.nn.Sigmoid()
        ]
        
        _input_size = num_features * 2 + 1  # solute + solvent + temperature
        fnn_modules += _build_mlp(_input_size, hidden_size, activation_fxn, num_layers)
        fnn_modules.append(torch.nn.Linear(
            hidden_size if num_layers else _input_size, 1
        ))
        
        self.fnn = torch.nn.Sequential(*fnn_modules)
        self.save_hyperparameters()

    def forward(self, batch):
        """Forward pass through the network"""
        return self.fnn(batch)

    def predict_step(self, batch):
        """Prediction with proper scaling"""
        # Validate scalers are present
        for stat_obj, name in [
            (self.solute_means, "solute_means"),
            (self.solute_vars, "solute_vars"),
            (self.solvent_means, "solvent_means"),
            (self.solvent_vars, "solvent_vars"),
            (self.temperature_means, "temperature_means"),
            (self.temperature_vars, "temperature_vars"),
            (self.target_means, "target_means"),
            (self.target_vars, "target_vars"),
        ]:
            if stat_obj is None:
                raise RuntimeError(f"Missing scaler: {name}")

        solute_feats, solvent_feats, temperature = batch[0]
        
        # Scale inputs
        solute_feats = standard_scale(solute_feats, self.solute_means, self.solute_vars)
        solvent_feats = standard_scale(solvent_feats, self.solvent_means, self.solvent_vars)
        temperature = standard_scale(temperature, self.temperature_means, self.temperature_vars)
        
        # Forward pass
        with torch.inference_mode():
            logits = self.forward((solute_feats, solvent_feats, temperature))
        
        # Unscale output
        return inverse_standard_scale(logits, self.target_means, self.target_vars)

    @torch.enable_grad()
    def _custom_loss(self, batch, name):
        """
        Sobolev loss: MSE on both solubility and its temperature gradient
        """
        (_solute, _solvent, temperature), y, y_grad = batch
        temperature.requires_grad_()
        
        y_hat = self.forward((_solute, _solvent, temperature))
        y_loss = torch.nn.functional.mse_loss(y_hat, y, reduction="mean")
        
        # Calculate gradient of prediction w.r.t. temperature
        (y_grad_hat,) = torch.autograd.grad(
            y_hat,
            temperature,
            grad_outputs=torch.ones_like(y_hat),
            retain_graph=True,
        )
        
        # Gradient loss (ignore NaN gradients)
        _scale_factor = 10.0
        y_grad_loss = _scale_factor * (y_grad_hat - y_grad).pow(2).nanmean()
        
        loss = y_loss + y_grad_loss
        
        # Logging
        self.log(f"{name}_{self.training_metric}_scaled_loss", loss)
        self.log(f"{name}_logS_scaled_loss", y_loss)
        self.log(f"{name}_dlogSdT_scaled_loss", y_grad_loss)
        
        return loss, y_hat

    def _plain_loss(self, batch, name):
        """Standard MSE loss (no gradient penalty)"""
        (_solute, _solvent, temperature), y, _ = batch
        y_hat = self.forward((_solute, _solvent, temperature))
        loss = torch.nn.functional.mse_loss(y_hat, y, reduction="mean")
        self.log(f"{name}_{self.training_metric}_scaled_loss", loss)
        return loss, y_hat

    def _loss(self, batch, name):
        """Choose between Sobolev and plain loss"""
        return self._custom_loss(batch, name)

    def training_step(self, batch, batch_idx):
        return self._loss(batch, "train")[0]

    def validation_step(self, batch, batch_idx):
        loss, y_hat = self._loss(batch, "validation")
        self._human_loss(y_hat, batch, "validation")
        return loss

    def test_step(self, batch, batch_idx):
        loss, y_hat = self._loss(batch, "test")
        self._human_loss(y_hat, batch, "test")
        return loss


if __name__ == "__main__":
    # Test model instantiation
    model = fastpropSolubility()
    print(model)
    
    # Test forward pass
    batch_size = 4
    num_features = 1613
    
    solute = torch.rand(batch_size, num_features)
    solvent = torch.rand(batch_size, num_features)
    temperature = torch.rand(batch_size, 1)
    
    output = model((solute, solvent, temperature))
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, 1)
    print("Model test passed!")
