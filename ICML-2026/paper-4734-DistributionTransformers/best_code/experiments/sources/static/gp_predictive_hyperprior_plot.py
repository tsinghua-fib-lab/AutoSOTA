"""
Experiment evaluating method on problem of finding posterior for GP hyperparameters
"""

from copy import copy, deepcopy
from functools import partial
from random import sample
from typing import Union, Sequence, Callable, Optional
import gpytorch
from gpytorch.likelihoods import GaussianLikelihood
from matplotlib import pyplot as plt
from sympy import hyper
import torch
from torch import Tensor
from torch import nn
from torch.distributions import Normal, MultivariateNormal, constraints, Distribution, Uniform, Normal, Independent, InverseGamma
from torch.distributions.utils import lazy_property
from torch.types import _size
from competitor_methods.pfns import RiemannDistribution, PFN, get_borders_from_prior
from distributions.utils import decode_gmm_sample
import pyro
from pyro.infer.mcmc import NUTS, MCMC
import pyro.distributions as dist
from gpytorch import add_jitter
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from torch.nn import Module as TModule

from gpytorch.priors import UniformPrior, GammaPrior
from gpytorch.priors.prior import Prior
from gpytorch.priors.utils import _bufferize_attributes, _del_attributes

from distributions.distributions import (InverseGammaMetaPrior, ObservationModel, CompleteDistribution,
                                         GaussianMixtureModel, MetaPrior)
from model.embeddings import DistributionEmbedding
from model.distribution_transformer import DistributionTransformer
from distributions.utils import gmm_bounds_func
from model.embeddings import ComponentEmbedding, GammaEmbedding, ObservationEmbedding
from experiments.sources.static.gp_predictive_hyperprior import NOISE_VAR, CompleteDistributionGPPredictive, ExactGPModel, GPPredictiveObservationModel, HyperpriorEmbedding, MeanScaleMetaPrior

MCMC_SAMPLES = 2000

class InverseGammaPrior(Prior, InverseGamma):

    def __init__(self, concentration, rate, validate_args=False, transform=None):
        TModule.__init__(self)
        InverseGamma.__init__(self, concentration=concentration, rate=rate, validate_args=validate_args)
        _bufferize_attributes(self, ("concentration", "rate"))
        self._transform = transform

    def expand(self, batch_shape):
        batch_shape = torch.Size(batch_shape)
        return InverseGammaPrior(self.concentration.expand(batch_shape), self.rate.expand(batch_shape))

    def __call__(self, *args, **kwargs):
        return super(InverseGamma, self).__call__(*args, **kwargs)
    
def get_marginal_posterior(
    phi_out,
    scale_parametrisation,
    marginalise_y=False,
    marginalise_lengthscale=False,
):
    
    assert marginalise_y or marginalise_lengthscale
    assert not(marginalise_y and marginalise_lengthscale)
    
    var_ix = 0 if marginalise_lengthscale else 1
    
    weigth = phi_out[...,0]
    loc = phi_out[..., 1: 3]
    scale = phi_out[..., -2 ** 2:].reshape(*phi_out.shape[:-1], 2, 2)
    var = torch.diagonal(scale, dim1=-2, dim2=-1)
    marginal_phi_out = torch.stack([weigth, loc[:,:,var_ix], var[:,:,var_ix]], dim=-1)

    return GaussianMixtureModel(**decode_gmm_sample(marginal_phi_out, scale_parametrisation))

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

    plt.style.use("seaborn-v0_8-paper")

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
        elif isinstance(predicted_gp_posterior, MultivariateNormal):
            if predicted_gp_posterior.mean.dim() == 2:
                mean_predicted = predicted_gp_posterior.mean.mean(dim=0)
                std_predicted = (predicted_gp_posterior.variance.mean(dim=0) + predicted_gp_posterior.mean.var(dim=0)) ** 0.5
            else:
                mean_predicted = predicted_gp_posterior.mean
                std_predicted = predicted_gp_posterior.variance ** 0.5
            
            lower_confidence_predicted = mean_predicted + 1.96 * std_predicted
            upper_confidence_predicted = mean_predicted - 1.96 * std_predicted
            
        else:
            mean_predicted = predicted_gp_posterior.mean
            confidence_predicted = predicted_gp_posterior.conf(1 - 0.025)
            lower_confidence_predicted = confidence_predicted[..., 0].flatten()
            upper_confidence_predicted = confidence_predicted[..., 1].flatten()
            
        ax.plot(x, mean_predicted.cpu().detach(), color=color_pallette[model_name], label=model_name)
        ax.fill_between(x, lower_confidence_predicted.cpu().detach(), upper_confidence_predicted.cpu().detach(), color=color_pallette[model_name], alpha=0.4)

    for ix, gp_posterior_draw in enumerate(gp_posterior_draws):
        ax.plot(x, gp_posterior_draw.cpu().detach(), color='orange', label="NUTS draws" if ix ==0 else None)

    if train_X != None and train_Y != None:
        ax.scatter(train_X.cpu().detach(), train_Y.cpu().detach(), label="Training Data", zorder=100)
    
    ax.set_ylabel("Y")
    ax.set_xlabel("X")
    ax.legend(loc="best")
    plt.show()
    return fig

class BatchGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, mean_module, covar_module):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def fit_NUTS(train_x, train_y, model, likelihood):
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    def pyro_model(x, y):
        with gpytorch.settings.fast_computations(False, False, False):
            sampled_model = model.pyro_sample_from_prior()
            output = sampled_model.likelihood(sampled_model(x))
            pyro.sample("obs", output, obs=y)
        return y

    nuts_kernel = NUTS(pyro_model)
    mcmc_run = MCMC(nuts_kernel, num_samples=MCMC_SAMPLES, warmup_steps=100, disable_progbar=False)
    mcmc_run.run(train_x, train_y)
    
    return mcmc_run.get_samples()


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
    
    competitor_kwargs = testing_kwargs["competitor_kwargs"]
    pfn_kwargs = copy(competitor_kwargs["pfns"])
    del pfn_kwargs["training_kwargs"]

    pfn = PFN(**pfn_kwargs, **deepcopy(model.observation_embeddings))
    pfn.load_state_dict(torch.load(kwargs.get("pfn_path"), weights_only=True))
    
    phi, x, z = complete_distribution.sample()

    linspace_size = 1000

    x_domain_size = complete_distribution.meta_prior.decode_sample(phi)["x_domain_size"]
    z["query"] = torch.linspace(0, x_domain_size, linspace_size).reshape(
        linspace_size, 1
    )

    train_x = z["dataset"][:, 0]
    train_y = z["dataset"][:, 1]
    hyperparams = complete_distribution.meta_prior.decode_sample(phi)

    z["dataset"] = z["dataset"].unsqueeze(0).expand((linspace_size,) + z["dataset"].shape)
    phi = phi.unsqueeze(0).expand((linspace_size,) + phi.shape)
    lengthscale = x[..., 1]

    kernel = ScaleKernel(RBFKernel())
    kernel.base_kernel.lengthscale = lengthscale
    kernel.outputscale = hyperparams['covariance_matrix']**0.5

    mean_function = ConstantMean()
    mean_function.constant = hyperparams['loc']
    
    likelihood = complete_distribution.observation_model["dataset"].likelihood
 
    exact_gp = ExactGPModel(train_x, train_y, likelihood, mean_function, kernel)
    exact_gp.eval()
    true_posterior = exact_gp(z["query"].flatten())
    
    
    if MCMC_SAMPLES > 0:

        kernel = ScaleKernel(RBFKernel())
        kernel.base_kernel.lengthscale = lengthscale
        kernel.outputscale = hyperparams['covariance_matrix']**0.5

        mean_function = ConstantMean()
        mean_function.constant = hyperparams['loc']
        
        likelihood = GaussianLikelihood()
        likelihood.noise = NOISE_VAR
        
        exact_gp = ExactGPModel(train_x, train_y, likelihood, mean_function, kernel)

        exact_gp.covar_module.base_kernel.register_prior(
            "lengthscale_prior", 
            InverseGammaPrior(1, 2),
            "lengthscale"
        )

        expanded_test_x = z["query"].unsqueeze(0)
        samples = fit_NUTS(train_x, train_y, exact_gp, likelihood)
        
        sampled_kernel = ScaleKernel(RBFKernel(batch_shape=[1000]))
        sampled_kernel.base_kernel.lengthscale = samples["covar_module.base_kernel.lengthscale_prior"][-1000:].reshape(1000, 1, 1)
        sampled_kernel.outputscale = hyperparams['covariance_matrix']**0.5
        
        gpmodel = BatchGPModel(train_x, train_y, likelihood, mean_function, sampled_kernel)
        gpmodel.eval()
        mcmc_posteriors = gpmodel(expanded_test_x)

    
    phi_in, phi_out = model(phi, **z)
    model_prior = GaussianMixtureModel(**decode_gmm_sample(phi_in, scale_parametrisation))
    model_posterior = GaussianMixtureModel(**decode_gmm_sample(phi_out, scale_parametrisation))
    
    weigth = phi_out[:,:,0]
    loc = phi_out[..., [0]]
    scale = phi_out[..., -2 ** 2:].reshape(*phi_out.shape[:-1], 2, 2)
    var = torch.diagonal(scale, dim1=-2, dim2=-1)
    marginal_posterior= get_marginal_posterior(phi_out, scale_parametrisation, marginalise_y=True)
    untransformed_ls_samples = marginal_posterior.sample([1])
    transformed_ls_samples = torch.exp(untransformed_ls_samples)
    
    #sampled_kernel = ScaleKernel(RBFKernel(batch_shape=[1000]))
    #sampled_kernel.base_kernel.lengthscale = transformed_ls_samples.reshape(1000, 1, 1)
    #sampled_kernel.outputscale = hyperparams['covariance_matrix']**0.5
    
    #gpmodel = BatchGPModel(train_x, train_y, likelihood, mean_function, sampled_kernel)
    #gpmodel.eval()
    #dt_posterior = gpmodel(z["query"])
    
    dt_posterior = get_marginal_posterior(phi_out, scale_parametrisation, marginalise_lengthscale=True)

    meta_prior_pfn = MeanScaleMetaPrior(marginalise_lengthscale=True, **meta_prior_kwargs)
    complete_distribution_pfn = CompleteDistributionGPPredictive(meta_prior_pfn, marginalise_lengthscale=True, **observation_model)
    phi_buckets, x_buckets = complete_distribution_pfn.sample((training_kwargs["steps_per_epoch"], 1))[:2]
    
    borders = get_borders_from_prior(complete_distribution_pfn.meta_prior.prior(
                        **complete_distribution_pfn.meta_prior.decode_sample(phi_buckets)), pfn.n_buckets, pfn.infinite_support,
                        pfn.leftmost_border, pfn.rightmost_border).mean(dim=0)
    pfn.borders = borders
    
    pfn = pfn.cpu()
    phi_out = pfn(**z)
    pfn_posterior = RiemannDistribution(phi_out, pfn.borders, pfn.infinite_support)

    print("DT NLL:",-torch.mean(dt_posterior.log_prob(true_posterior.loc.view(-1,1))).item())
    print("PFN NLL:",-torch.mean(pfn_posterior.log_prob(true_posterior.loc)).item())
    #print("MCMC NLL:",-torch.mean(mcmc_posteriors.log_prob(true_posterior.loc.view(1,-1))).item())
    print("Oracle NLL:",-torch.mean(true_posterior.log_prob(true_posterior.loc)).item())
    
    plot = plot_gp(
        {
            "ours": model_prior,
        },
        {
            "ours": "blue",
        },
        [],
        train_x,
        train_y,
        linspace_points=linspace_size,
        x_domain_size=x_domain_size,
        phi=phi
    )
    plot.savefig(_run.observers[0].dir + "\\prior_plot.png")
    
    plot = plot_gp(
        {
            "Oracle": true_posterior,
            "DT": dt_posterior,
            "PFN": pfn_posterior,
            "MCMC": mcmc_posteriors,
        },
        {
            "DT": "blue",
            "PFN": "green",
            "MCMC": "violet",
            "Oracle": "orange"
        },
        [],
        train_x,
        train_y,
        linspace_points=linspace_size,
        x_domain_size=x_domain_size,
        phi=phi
    )

    print(f"LS:{round(lengthscale.item(), 4)}, Prior InverseGamma({round(phi[0, 2].item(),2)}, {round(phi[0, 3].item(),2)})")
    plot.savefig(_run.observers[0].dir + "plot.png")
    plot.savefig(_run.observers[0].dir + "plot.pdf")

#test_gp(model, complete_distribution,
#     bounds_func=partial(gmm_bounds_func,
#                         scale_parametrisation=component_embedding_kwargs["scale_parametrisation"]),
#     linspace_size=1000,
#     hyperpior=True,
#     _run=_run, **testing_kwargs)