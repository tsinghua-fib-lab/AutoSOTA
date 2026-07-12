"""
Experiment evaluating method on problem of finding posterior for GP hyperparameters
"""

from copy import copy, deepcopy
from functools import partial
from random import sample
from typing import Union, Sequence, Callable, Optional
import gpytorch
from matplotlib import pyplot as plt
from sympy import hyper
import torch
from torch import Tensor
from torch import nn
from torch.distributions import Normal, MultivariateNormal, constraints, Distribution, Uniform, Normal, Independent
from torch.distributions.utils import lazy_property
from torch.types import _size
from competitor_methods.pfns import RiemannDistribution, PFN, get_borders_from_prior
from workflows.train import train_pfn
from distributions.utils import decode_gmm_sample, plot_distributions

from gpytorch import add_jitter
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from torch.func import vmap, jacrev

from distributions.distributions import (InverseGammaMetaPrior, ObservationModel, CompleteDistribution,
                                         GaussianMixtureModel, MetaPrior)
from distributions.utils import plot_distributions
from model.embeddings import DistributionEmbedding
from model.distribution_transformer import DistributionTransformer
from distributions.utils import gmm_bounds_func
from model.embeddings import ComponentEmbedding, GammaEmbedding, ObservationEmbedding
from experiments.sources.static.gp_predictive_hyperprior import CompleteDistributionGPPredictive, ExactGPModel, GPPredictiveObservationModel, HyperpriorEmbedding, MeanScaleMetaPrior


def plot_gp(predicted_gp_posteriors: dict[str, Distribution],
            color_pallette: dict[str, str],
            gp_posterior_draws: list[Distribution],
            train_X: torch.Tensor,
            train_Y: torch.Tensor,
            linspace_points: int,
            x_domain_size: float,
            phi,
            hyperpior=False) -> plt.Figure:
    """
    Function to plot a (1 dimensional) distribution, or a pair of (1 dimensional) distributions.

    Args:
        predicted_gp_posterior: Predicted distributions over y values at consequtive x's.
        true_gp_posterior: True distributions over y values at consequtive x's.

    Returns:
        Figure object.

    """

    fig, ax = plt.subplots()

    x = torch.linspace(0, x_domain_size, linspace_points)

    for model_name, predicted_gp_posterior in predicted_gp_posteriors.items():
        if isinstance(predicted_gp_posterior, GaussianMixtureModel):
            mean_per_component = predicted_gp_posterior.loc[...,0]
            weights_per_component = predicted_gp_posterior.weights
            var_per_component = predicted_gp_posterior.covariance_matrix[...,0, 0]
            
            mean_mean = torch.sum(mean_per_component * weights_per_component, dim=-1)
            mean_var = torch.sum(var_per_component * weights_per_component, dim=-1)
            var_mean = torch.sum(weights_per_component * (mean_per_component - mean_mean.unsqueeze(-1))**2, dim=-1)
            
            mean_predicted = mean_mean
            std_predicted = (mean_var + var_mean) ** 0.5
            
            lower_confidence_predicted = mean_predicted + 1.96 * std_predicted
            upper_confidence_predicted = mean_predicted - 1.96 * std_predicted
            
        else:
            mean_predicted = predicted_gp_posterior.mean
            confidence_predicted = predicted_gp_posterior.conf(1 - 0.025)
            lower_confidence_predicted = confidence_predicted[..., 0].flatten()
            upper_confidence_predicted = confidence_predicted[..., 1].flatten()
            
        ax.plot(x, mean_predicted.cpu().detach(), color=color_pallette[model_name], label=model_name)
        ax.fill_between(x, lower_confidence_predicted.cpu().detach(), upper_confidence_predicted.cpu().detach(), color=color_pallette[model_name], alpha=0.2)

    for gp_posterior_draw in gp_posterior_draws:
        mean_true = gp_posterior_draw.loc.flatten()
        std_true = torch.diag(gp_posterior_draw.covariance_matrix) ** 0.5
        upper_confidence_true = mean_true + 1.96 * std_true
        lower_confidence_true = mean_true - 1.96 * std_true

        ax.plot(x, mean_true.cpu().detach(), color='orange', label="True")
        ax.fill_between(x, lower_confidence_true.cpu().detach(), upper_confidence_true.cpu().detach(), color='orange', alpha=0.2)

    if train_X != None and train_Y != None:
        ax.scatter(train_X.cpu().detach(), train_Y.cpu().detach(), label="Training Data")
    
    ax.set_title(f"concentration:{round(phi[0,2].item(),2)} rate:{round(phi[0,3].item(),2)}")
    ax.set_ylabel("Y")
    ax.legend(loc="best")
    plt.show()
    return fig

def get_marginal_posterior(
    phi_out,
    scale_parametrisation,
    marginalise_y=False,
    marginalise_lengthscale=False,
):
    
    assert marginalise_y or marginalise_lengthscale
    assert not(marginalise_y and marginalise_lengthscale)
    
    var_ix = 0 if marginalise_lengthscale else 1
    
    weigths = phi_out[...,0]
    loc = phi_out[..., 1: 3][..., [var_ix]]
    covariance_matrix = phi_out[..., -2 ** 2:].reshape(*phi_out.shape[:-1], 2, 2)[..., var_ix, var_ix].unsqueeze(-1).unsqueeze(-1)

    return GaussianMixtureModel(loc=loc, weights=weigths, covariance_matrix=covariance_matrix)
    


def run(n_components: int,
        state_size: int,
        meta_prior_kwargs: dict,
        component_embedding_kwargs: dict,
        observation_embedding_kwargs: dict[str, dict],
        distribution_embedding_kwargs: dict,
        transformer_kwargs: dict,
        training_kwargs: dict,
        testing_kwargs: dict,
        _run=None,
        *args, **kwargs):
    
    with torch.no_grad():
        device = "cuda:0" if torch.cuda.is_available() else 'cpu:0'
        
        meta_prior = MeanScaleMetaPrior(**meta_prior_kwargs)

        observation_model = {observation_type: GPPredictiveObservationModel(observation_type=observation_type)
                            for observation_type in ["dataset", "query"]}

        # Complete distribution
        complete_distribution = CompleteDistributionGPPredictive(meta_prior, **observation_model)

        # Distribution transformer
        d_model = transformer_kwargs["d_model"]
        component_embedding = ComponentEmbedding(state_size=state_size, d_model=d_model, **component_embedding_kwargs)
        observation_embedding = {key: ObservationEmbedding(d_model=d_model, observation_size= (meta_prior_kwargs["x_dimensions"] + (1 if key=="dataset" else 0)), **kwargs)
                                for key, kwargs in observation_embedding_kwargs.items()}
        prior_embedding = HyperpriorEmbedding(d_model=d_model,n_components=n_components,state_size=state_size, **distribution_embedding_kwargs, **component_embedding_kwargs)
        model = DistributionTransformer(component_embedding=component_embedding,
                                        transformer_kwargs=transformer_kwargs,
                                        n_components=n_components,
                                        prior_embedding=prior_embedding,
                                        sample_space_transform=lambda x: torch.stack([x[...,0], torch.log(x[...,1])], dim=-1),
                                        **observation_embedding)

        model.load_state_dict(torch.load(kwargs.get("model_path"), weights_only=True))
        scale_parametrisation = model.component_embedding.scale_parametrisation
        
        n_samples = 1
        phi, x, z = complete_distribution.sample((n_samples,))
        _, phi_out = model(phi, **z)
        phi_prior_dict = complete_distribution.meta_prior.decode_sample(phi)
        posterior_losses = -GaussianMixtureModel(**decode_gmm_sample(phi_out, scale_parametrisation)
                                            ).log_prob(model.sample_space_transform(x))

        posterior_losses -= torch.logdet(vmap(jacrev(model.sample_space_transform))
                                            (x.reshape(torch.Size([n_samples])
                                                       + torch.Size([2]))
                                             )[...,1, 1].reshape(torch.Size([n_samples]) + torch.Size([1])
                                                       + torch.Size([1])))
        posterior_losses = torch.nanmean(posterior_losses)
        print("Ours Test NLL FULL", posterior_losses.item())

        competitor_kwargs = testing_kwargs["competitor_kwargs"]
        pfn_kwargs = copy(competitor_kwargs["pfns"])
        del pfn_kwargs["training_kwargs"]

        pfn = PFN(**pfn_kwargs, **deepcopy(model.observation_embeddings))
        pfn.load_state_dict(torch.load(kwargs.get("pfn_path"), weights_only=True))
        
        marginal_posterior = get_marginal_posterior(phi_out[0,...], scale_parametrisation, marginalise_y=True)
        
        posterior_losses = -marginal_posterior.log_prob(model.sample_space_transform(x)[...,[1]])
        posterior_losses -= torch.logdet(vmap(jacrev(model.sample_space_transform))
                                            (x.reshape(torch.Size([n_samples])
                                                       + torch.Size([2]))
                                             )[...,1, 1].reshape(torch.Size([n_samples]) + torch.Size([1])
                                                       + torch.Size([1])))
        posterior_losses = torch.nanmean(posterior_losses)
        print("Ours Test NLL Marginal", posterior_losses.item())
        
        meta_prior_pfn = MeanScaleMetaPrior(marginalise_y=True, **meta_prior_kwargs)
        complete_distribution_pfn = CompleteDistributionGPPredictive(meta_prior_pfn, marginalise_y=True, **observation_model)
        phi_buckets, x_buckets = complete_distribution_pfn.sample((training_kwargs["steps_per_epoch"], 1))[:2]
        pfn = pfn.cpu()
        borders = get_borders_from_prior(complete_distribution_pfn.meta_prior.prior(
                            **complete_distribution_pfn.meta_prior.decode_sample(phi_buckets)), pfn.n_buckets, pfn.infinite_support,
                            pfn.leftmost_border, pfn.rightmost_border).mean(dim=0)
        pfn.borders = borders
        
        phi_out = pfn(**z)[0,...]
        pfn_posterior = RiemannDistribution(phi_out, pfn.borders, pfn.infinite_support)
        print("PFN test marginal", -torch.nanmean(pfn_posterior.log_prob(x[...,1])).item())
        
        fig = plot_distributions(marginal_posterior, pfn_posterior, torch.log, None, bounds=(0,5))
        
        fig.savefig("PFN_vs_DT_ls.png")

    #test_gp(model, complete_distribution,
    #     bounds_func=partial(gmm_bounds_func,
    #                         scale_parametrisation=component_embedding_kwargs["scale_parametrisation"]),
    #     linspace_size=1000,
    #     hyperpior=True,
    #     _run=_run, **testing_kwargs)