#!/usr/bin/env python3

from typing import Optional
import torch
from torch import Tensor
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.constraints.constraints import Interval


def fit_gp_model(
        X: Tensor, 
        objective_X: Tensor, 
        kernel: Optional[torch.nn.Module] = None,
        cost_X: Optional[Tensor] = None, 
        unknown_cost: bool = False,
        gaussian_likelihood: bool = False,
        noise_level: float = 1e-6,
        output_standardize: bool = False,
    ):
    # Ensure X is a 2D tensor [num_data, num_features]
    if X.ndim == 1:
        X = X.unsqueeze(dim=-1)
    
    # Ensure objective_X is a 2D tensor [num_data, 1]
    if objective_X.ndim == 1:
        objective_X = objective_X.unsqueeze(dim=-1)

    # Ensure cost_X is a 2D tensor [num_data, 1]
    if unknown_cost == True:
        if cost_X.ndim == 1:
            log_cost_X = torch.log(cost_X).unsqueeze(dim=-1)
        else:
            log_cost_X = torch.log(cost_X)

    Y = torch.cat((objective_X, log_cost_X), dim=-1) if unknown_cost else objective_X
        
    if gaussian_likelihood:
        _, aug_batch_shape = SingleTaskGP.get_batch_dimensions(
                train_X=X,
                train_Y=Y,
            )
        likelihood = GaussianLikelihood(
                batch_shape=aug_batch_shape,
                noise_constraint=Interval(lower_bound=noise_level, upper_bound=10*noise_level),
            )
    else:
        Yvar = torch.ones(len(Y)) * noise_level
        likelihood = FixedNoiseGaussianLikelihood(noise=Yvar)

    # Outcome transform
    if output_standardize == True:
        outcome_transform = Standardize(m=Y.shape[-1])
    else:
        outcome_transform = None
   
    model = SingleTaskGP(train_X=X, train_Y=Y, likelihood = likelihood, covar_module=kernel, outcome_transform=outcome_transform)

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    return model


# LCBench utils
def normalize_config(config):
    # Convert each value to a torch tensor (ensuring float type for calculations)
    batch = torch.tensor(config["batch_size"])
    lr = torch.tensor(config["learning_rate"])
    units = torch.tensor(config["max_units"])
    momentum = torch.tensor(config["momentum"])
    weight_decay = torch.tensor(config["weight_decay"])
    layers = torch.tensor(float(config["num_layers"]))
    dropout = torch.tensor(config["max_dropout"])
    
    # For log-scaled parameters: batch size, learning rate, and max units.
    batch_norm = (torch.log(batch) - torch.log(torch.tensor(16.0))) / (torch.log(torch.tensor(512.0)) - torch.log(torch.tensor(16.0)))
    lr_norm = (torch.log(lr) - torch.log(torch.tensor(1e-4))) / (torch.log(torch.tensor(1e-1)) - torch.log(torch.tensor(1e-4)))
    units_norm = (torch.log(units) - torch.log(torch.tensor(64.0))) / (torch.log(torch.tensor(1024.0)) - torch.log(torch.tensor(64.0)))
    
    # For linearly scaled parameters.
    momentum_norm = (momentum - 0.1) / (0.99 - 0.1)
    weight_decay_norm = (weight_decay - 1e-5) / (1e-1 - 1e-5)
    layers_norm = (layers - 1) / (4 - 1)
    
    # Dropout is already between 0 and 1.
    dropout_norm = dropout

    # Combine into a 7-dimensional tensor.
    normalized_vector = torch.stack([
        batch_norm, 
        lr_norm, 
        momentum_norm, 
        weight_decay_norm, 
        layers_norm, 
        units_norm, 
        dropout_norm
    ])
    
    return normalized_vector
