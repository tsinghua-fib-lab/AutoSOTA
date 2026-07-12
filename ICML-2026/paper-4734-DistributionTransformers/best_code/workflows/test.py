"""
Testing workflow
"""

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.nn import Identity
from torch.func import vmap, jacrev
from torch.distributions import MultivariateNormal, Distribution

from time import time
from typing import Callable, Optional
from copy import copy
from math import sqrt, ceil

from model.distribution_transformer import DistributionTransformer
from distributions.distributions import CompleteDistribution, GaussianMixtureModel, ObservationModel
from distributions.utils import decode_gmm_sample, encode_gmm_sample, kl_divergence, plot_distributions, gmm_bounds_func
from competitor_methods.variational_inference import VI
from competitor_methods.pfns import RiemannDistribution, PFN
from competitor_methods.ekf import EKF
from competitor_methods.particle_filter import ParticleFilter
try:
    from competitor_methods.tabpfn import test_tabpfn
except (ImportError, ModuleNotFoundError):
    test_tabpfn = None
from competitor_methods.ace import get_ace_model, predict_w_ace
from workflows.train import train_pfn, train_ace
from workflows.utils import get_model_size
from workflows.metrics import *
from dynamic.motion_models import LTIMotionModel
from dynamic.filters import LTIFilter
from dynamic.utils import plot_filtered_series

from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean

try:
    from tabpfn import TabPFNRegressor
except (ImportError, ModuleNotFoundError):
    TabPFNRegressor = None


def test_conjugate_prior(model: DistributionTransformer,
                         complete_distribution: CompleteDistribution,
                         conjugacy_update: Callable[[dict[str, Tensor], dict[str, Tensor], str], dict[str, Tensor]],
                         competitor_kwargs: Optional[dict[str, dict]] = None,
                         inverse_transform: Optional[Callable[[Tensor], Tensor]] = None,
                         n_test_priors: int = 1000,
                         n_kl_samples: int = 10000,
                         plot: bool = False,
                         bounds_func: Optional[Callable[[dict[str, Tensor]], tuple[float, float]]] = None,
                         gpu_device: str = "cuda:0",
                         _run=None,
                         *args,
                         **kwargs
                         ) -> None:
    """
    Standard testing routine for experiments involving conjugate priors

    Args:
        model: Model to Test.
        complete_distribution: Complete distribution over priors, state and observation.
        conjugacy_update: Function taking a dict of prior parameters, a dict of observations, and returning a dict of
            posterior parameters.
        competitor_kwargs: Dictionary of dictionaries of parameters for competitor methods.
            Defaults to None.
        inverse_transform: Transform from sample space of GMM approximation to prior.
            Defaults to None.
        n_test_priors: Number of priors to test model with.
            Defaults to 1000.
        n_kl_samples: Number of samples to take when computing KL divergences.
            Defaults to 10000.
        plot: Whether to plot.
            Defaults to False.
        gpu_device: GPU device.
            Defaults to "cuda:0".
        bounds_func: Function to calculate plotting bounds from exact distribution parameters.
            Defaults to an estimate of the 5-95%ile from 10000 samples
        _run: Sacred run object.

    """

    with torch.no_grad():
        device = gpu_device if torch.cuda.is_available() else 'cpu:0'
        scale_parametrisation = model.component_embedding.scale_parametrisation

        # Test inputs
        phi, x, z = complete_distribution.sample((n_test_priors,))

        # Device
        phi = phi.to(device)
        z = {key: val.to(device) for key, val in z.items()}
        model = model.to(device)

        # Exact solution
        phi_prior_dict = complete_distribution.meta_prior.decode_sample(phi)
        prior = complete_distribution.meta_prior.prior(**phi_prior_dict)
        phi_posterior_dict = conjugacy_update(phi_prior_dict, z, device)
        exact_posterior = complete_distribution.meta_prior.prior(**phi_posterior_dict)

        # Model solution
        start_time = time()
        phi_in, phi_out = model(phi.to(device), **z)
        model_inference_time = time() - start_time
        print(f"Model inference time (1000 priors): {model_inference_time}")
        model_prior = GaussianMixtureModel(**decode_gmm_sample(phi_in, scale_parametrisation))
        model_posterior = GaussianMixtureModel(**decode_gmm_sample(phi_out, scale_parametrisation))

        model_prior_kl_divergence = kl_divergence(prior, model_prior, model.sample_space_transform,
                                                  n_kl_samples)
        model_expected_prior_kl_divergence = model_prior_kl_divergence.mean().item()
        model_std_prior_kl_divergence = model_prior_kl_divergence.std().item()
        model_posterior_kl_divergence = kl_divergence(exact_posterior, model_posterior, model.sample_space_transform,
                                                      n_kl_samples)
        model_expected_posterior_kl_divergence = model_posterior_kl_divergence.mean().item()
        model_std_posterior_kl_divergence = model_posterior_kl_divergence.std().item()

        model_posterior_nll = -model_posterior.log_prob(model.sample_space_transform(x)
                                                        .reshape(model_posterior.batch_shape
                                                                 + model_posterior.event_shape).to(device))
        model_posterior_nll -= torch.logdet(vmap(jacrev(model.sample_space_transform))
                                            (x.reshape(model_posterior.batch_shape).to(device)
                                             ).reshape(*prior.batch_shape, 1, 1))
        model_posterior_expected_nll = model_posterior_nll.mean().item()
        model_posterior_std_nll = model_posterior_nll.std().item()

        model_size = get_model_size(model)

        print(f"GMM approximation prior mean KL divergence: {model_expected_prior_kl_divergence}\n"
              f"Posterior mean KL divergence: {model_expected_posterior_kl_divergence}\n")

        if "vi" in competitor_kwargs:
            # VI solution
            before_vi = time()
            vi = VI(model.state_size, prior, complete_distribution.observation_model,
                    inverse_transform, **competitor_kwargs["vi"]).to(device)
            torch.set_grad_enabled(True)
            vi.fit(z, **competitor_kwargs["vi"])
            torch.set_grad_enabled(False)
            vi_time = time() - before_vi
            vi_posterior_kl_divergence = kl_divergence(exact_posterior, vi.distribution(),
                                                       model.sample_space_transform,
                                                       n_kl_samples)
            vi_expected_kl_divergence = vi_posterior_kl_divergence.mean().item()
            vi_std_kl_divergence = vi_posterior_kl_divergence.std().item()
            vi_nll = -vi.distribution().log_prob(model.sample_space_transform(x)
                                                 .reshape(model_posterior.batch_shape
                                                          + model_posterior.event_shape).to(device))
            vi_nll -= torch.logdet(vmap(jacrev(model.sample_space_transform))
                                            (x.reshape(model_posterior.batch_shape).to(device)
                                             ).reshape(*prior.batch_shape, 1, 1))
            vi_expected_nll = vi_nll.mean().item()
            vi_std_nll = vi_nll.std().item()
            vi_elbo = -vi.posterior_loss(z, n_samples=n_test_priors)
            vi_expected_elbo = vi_elbo.mean().item()
            vi_std_elbo = vi_elbo.std().item()
            model_elbo = -vi.posterior_loss(z, n_samples=n_test_priors, distribution=model_posterior)
            model_expected_elbo = model_elbo.mean().item()
            model_std_elbo = model_elbo.std().item()

        if "pfns" in competitor_kwargs:
            assert torch.prod(torch.tensor(prior.event_shape)).item() == 1, \
                "pfns only supported for univariate output distributions"
            # PFN solution
            pfn_kwargs = copy(competitor_kwargs["pfns"])
            del pfn_kwargs["training_kwargs"]
            pfn = PFN(**pfn_kwargs, **model.observation_embeddings)
            torch.set_grad_enabled(True)
            pfn, _ = train_pfn(pfn, complete_distribution, _run=_run, **competitor_kwargs["pfns"]["training_kwargs"])
            torch.set_grad_enabled(False)
            pfn = pfn.to(device)
            start_time = time()
            phi_out = pfn(**z)
            pfn_inference_time = time() - start_time
            pfn_posterior = RiemannDistribution(phi_out, pfn.borders, pfn.infinite_support)

            pfn_posterior_kl_divergence = kl_divergence(exact_posterior, pfn_posterior,
                                                        None,
                                                        n_kl_samples)
            pfn_expected_kl_divergence = pfn_posterior_kl_divergence.mean().item()
            pfn_std_kl_divergence = pfn_posterior_kl_divergence.std().item()
            pfn_nll = -pfn_posterior.log_prob(x.reshape(pfn_posterior.batch_shape).to(device))
            pfn_expected_nll = pfn_nll.mean().item()
            pfn_std_nll = pfn_nll.std().item()

            pfn_size = get_model_size(pfn)

            if "vi" in competitor_kwargs:
                pfn_elbo = -vi.posterior_loss(z, n_samples=n_test_priors, distribution=pfn_posterior,
                                              inverse_transform=Identity())
                pfn_expected_elbo = pfn_elbo.mean().item()
                pfn_std_elbo = pfn_elbo.std().item()
        
        if "tabpfn" in competitor_kwargs:
            assert torch.prod(torch.tensor(prior.event_shape)).item() == 1, \
                "pfns only supported for univariate output distributions"
            
            tabpfn_kwargs = copy(competitor_kwargs["tabpfn"])

            start_time = time()
            
            tabpfn_posterior, tabpfn_nll, tabpfn_regressor = test_tabpfn(
                {key: val for key, val in phi_prior_dict.items()}, 
                x, 
                {key: val for key, val in z.items()},
                complete_distribution,
                tabpfn_trainsize=tabpfn_kwargs["n_training_samples"]
            )
            
            tabpfn_inference_time = time() - start_time
            tabpfn_size = get_model_size(tabpfn_regressor.model_)

            tabpfn_expected_nll = tabpfn_nll.mean().item()
            tabpfn_std_nll = tabpfn_nll.std().item()
            
            tabpfn_posterior_kl_divergence = kl_divergence(exact_posterior, tabpfn_posterior,
                                                        None,
                                                        n_kl_samples)
            tabpfn_expected_kl_divergence = tabpfn_posterior_kl_divergence.mean().item()
            tabpfn_std_kl_divergence = tabpfn_posterior_kl_divergence.std().item()
            
        if "ace" in competitor_kwargs:

            # ACE solution
            ace_kwargs = copy(competitor_kwargs["ace"])
            del ace_kwargs["training_kwargs"]

            size_of_x = torch.tensor(prior.event_shape).item() if torch.tensor(prior.event_shape).numel() > 0 else 1
            num_latent = size_of_x + complete_distribution.meta_prior.prior_size + sum(obs.n_observations for obs in complete_distribution.observation_model.values())
            ace = get_ace_model(num_latent=num_latent, **ace_kwargs['transformer_kwargs'])
            torch.set_grad_enabled(True)
            ace, _ = train_ace(ace, complete_distribution, sample_space_transform=model.sample_space_transform, _run=_run, **competitor_kwargs["ace"]["training_kwargs"])
            torch.set_grad_enabled(False)
            
            start_time = time()
            ace_posterior = predict_w_ace(phi.to(device), x.to(device), {k:v.to(device) for k,v in z.items()}, ace.to(device))
            ace_inference_time = time() - start_time
            
            ace_posterior_kl_divergence = kl_divergence(exact_posterior, ace_posterior, model.sample_space_transform, n_kl_samples)
            ace_expected_kl_divergence = ace_posterior_kl_divergence.mean().item()
            ace_std_kl_divergence = ace_posterior_kl_divergence.std().item()

            ace_nll = -ace_posterior.log_prob(x.reshape(ace_posterior.batch_shape + ace_posterior.event_shape).to(device))
            ace_expected_nll = ace_nll.mean().item()
            ace_std_nll = ace_nll.std().item()
            
            ace_size = get_model_size(ace)

        # Test inputs
        phi, x, z = complete_distribution.sample()

        # Device
        model.cpu()

        # Exact solution
        phi_prior_dict = complete_distribution.meta_prior.decode_sample(phi)
        prior = complete_distribution.meta_prior.prior(**phi_prior_dict)
        phi_posterior_dict = conjugacy_update(phi_prior_dict, z, "cpu")
        exact_posterior = complete_distribution.meta_prior.prior(**phi_posterior_dict)

        # Model solution
        start_time = time()
        phi_in, phi_out = model(phi, **z)
        model_single_inference_time = time() - start_time
        model_prior = GaussianMixtureModel(**decode_gmm_sample(phi_in, scale_parametrisation))
        model_posterior = GaussianMixtureModel(**decode_gmm_sample(phi_out, scale_parametrisation))

        if _run is not None:
            _run.info.update({
                "model_expected_prior_kl_divergence": model_expected_prior_kl_divergence,
                "model_std_prior_kl_divergence": model_std_prior_kl_divergence,
                "model_expected_posterior_kl_divergence": model_expected_posterior_kl_divergence,
                "model_std_posterior_kl_divergence": model_std_posterior_kl_divergence,
                "model_posterior_expected_nll": model_posterior_expected_nll,
                "model_posterior_std_nll": model_posterior_std_nll,
                "model_inference_time": model_inference_time,
                "model_single_inference_time": model_single_inference_time,
                "model_size": model_size
            })
        if "vi" in competitor_kwargs:
            _run.info.update({
                "vi_time": vi_time,
                "model_expected_elbo": model_expected_elbo,
                "model_std_elbo": model_std_elbo,
                "vi_expected_kl_divergence": vi_expected_kl_divergence,
                "vi_std_kl_divergence": vi_std_kl_divergence,
                "vi_expected_elbo": vi_expected_elbo,
                "vi_std_elbo": vi_std_elbo,
                "vi_expected_nll": vi_expected_nll,
                "vi_std_nll": vi_std_nll
            })
        if "pfns" in competitor_kwargs:
            _run.info.update({
                "pfn_inference_time": pfn_inference_time,
                "pfn_expected_kl_divergence": pfn_expected_kl_divergence,
                "pfn_std_kl_divergence": pfn_std_kl_divergence,
                "pfn_expected_nll": pfn_expected_nll,
                "pfn_std_nll": pfn_std_nll,
                "pfn_size": pfn_size
            })
            if "vi" in competitor_kwargs:
                _run.info.update({
                    "pfn_expected_elbo": pfn_expected_elbo,
                    "pfn_std_elbo": pfn_std_elbo,
                })
        if "tabpfn" in competitor_kwargs:
            _run.info.update({
                "tabpfn_inference_time": tabpfn_inference_time,
                "tabpfn_expected_nll": tabpfn_expected_nll,
                "tabpfn_std_nll": tabpfn_std_nll,
                "tabpfn_size": tabpfn_size,
                "tabpfn_expected_kl_divergence": tabpfn_expected_kl_divergence,
                "tabpfn_std_kl_divergence": tabpfn_std_kl_divergence
            })
        if "ace" in competitor_kwargs:
            _run.info.update({
                "ace_inference_time": ace_inference_time,
                "ace_expected_nll": ace_expected_nll,
                "ace_std_nll": ace_std_nll,
                "ace_expected_kl_divergence": ace_expected_kl_divergence,
                "ace_std_kl_divergence": ace_std_kl_divergence,
                "ace_size": ace_size
            })

        # Plotting
        if plot:
            assert torch.prod(torch.tensor(prior.event_shape)).item() == 1, \
                "plotting only supported for single output distributions"

            if bounds_func is None:
                def bounds_func(params_dict: dict[str, Tensor]) -> tuple[float, float]:
                    dist = complete_distribution.meta_prior.prior(**params_dict)
                    samples = dist.sample((10000,))
                    samples = samples.sort().values
                    return samples[499].item(), samples[9499].item()


            prior_plot = plot_distributions(prior, model_prior, None, model.sample_space_transform,
                                            bounds_func(phi_prior_dict))

            posterior_plot = plot_distributions(exact_posterior, model_posterior, None, model.sample_space_transform,
                                                bounds_func(phi_posterior_dict))

            if "vi" in competitor_kwargs:
                vi = VI(model.state_size, prior, complete_distribution.observation_model, inverse_transform)
                torch.set_grad_enabled(True)
                start_time = time()
                vi.fit(z, **competitor_kwargs["vi"])
                vi_single_time = time() - start_time
                torch.set_grad_enabled(False)
                vi_posterior = vi.distribution()
                vi_posterior_plot = plot_distributions(exact_posterior, vi_posterior, None,
                                                       model.sample_space_transform, bounds_func(phi_posterior_dict))

            if "pfns" in competitor_kwargs:
                pfn = pfn.cpu()
                start_time = time()
                phi_out = pfn(**z)
                pfn_single_inference_time = time() - start_time
                pfn_posterior = RiemannDistribution(phi_out, pfn.borders, pfn.infinite_support)
                pfn_posterior_plot = plot_distributions(exact_posterior, pfn_posterior, None,
                                                        None, bounds_func(phi_posterior_dict))
            
            if "tabpfn" in competitor_kwargs:
                start_time = time()
                tabpfn_posterior, _, _ = test_tabpfn(
                    {key: val for key, val in phi_prior_dict.items()}, 
                    x, 
                    {key: val for key, val in z.items()},
                    complete_distribution,
                    tabpfn_trainsize=tabpfn_kwargs["n_training_samples"]
                )
                tabpfn_single_inference_time = time() - start_time
                tabpfn_posterior_plot = plot_distributions(tabpfn_posterior, exact_posterior, None,
                                                        None, bounds_func(phi_posterior_dict))
            
            if "ace" in competitor_kwargs:
                ace = ace.cpu()
                start_time = time()
                ace_posterior = predict_w_ace(phi, x, z, ace)
                ace_single_inference_time = time() - start_time
                ace_posterior_plot = plot_distributions(exact_posterior, ace_posterior, None,
                                                        model.sample_space_transform, bounds_func(phi_posterior_dict))

            if _run is not None:
                prior_plot.savefig(_run.observers[0].dir + "\\prior_plot.pdf", format="pdf")
                posterior_plot.savefig(_run.observers[0].dir + "\\posterior_plot.pdf", format="pdf")
                if "vi" in competitor_kwargs:
                    _run.info.update({
                        "vi_single_time": vi_single_time
                    })
                    vi_posterior_plot.savefig(_run.observers[0].dir + "\\vi_posterior_plot.pdf", format="pdf")

                if "pfns" in competitor_kwargs:
                    pfn_posterior_plot.savefig(_run.observers[0].dir + "\\pfn_posterior_plot.pdf", format="pdf")
                    _run.info.update({
                        "pfn_single_inference_time": pfn_single_inference_time
                    })
                if "tabpfn" in competitor_kwargs:
                    tabpfn_posterior_plot.savefig(_run.observers[0].dir + "\\tabpfn_posterior_plot.pdf", format="pdf")
                    _run.info.update({
                        "tabpfn_single_inference_time": tabpfn_single_inference_time
                    })
                if "ace" in competitor_kwargs:
                    ace_posterior_plot.savefig(_run.observers[0].dir + "\\ace_posterior_plot.pdf", format="pdf")
                    _run.info.update({
                        "ace_single_inference_time": ace_single_inference_time
                    })


def test(model: DistributionTransformer,
         complete_distribution: CompleteDistribution,
         competitor_kwargs: Optional[dict[str, dict]] = None,
         inverse_transform: Optional[Callable[[Tensor], Tensor]] = None,
         n_test_priors: int = 1000,
         n_kl_samples: int = 10000,
         plot: bool = False,
         bounds_func: Optional[Callable[[dict[str, Tensor]], tuple[float, float]]] = None,
         gpu_device: str = "cuda:0",
         _run=None
         ) -> None:
    """
    Standard testing routine for experiments not involving conjugate priors

    Args:
        model: Model to Test.
        complete_distribution: Complete distribution over priors, state and observation.
        competitor_kwargs: Dictionary of dictionaries of parameters for competitor methods.
            Defaults to None.
        inverse_transform: Transform from sample space of GMM approximation to prior.
            Defaults to None.
        n_test_priors: Number of priors to test model with.
            Defaults to 1000.
        n_kl_samples: Number of samples to take when computing KL divergences.
            Defaults to 10000.
        plot: Whether to plot.
            Defaults to False.
        gpu_device: GPU device.
            Defaults to "cuda:0".
        bounds_func: Function to calculate plotting bounds from exact distribution parameters.
            Defaults to an estimate of the 5-95%ile from 10000 samples
        _run: Sacred run object.

    """

    with torch.no_grad():
        competitor_kwargs = dict() if competitor_kwargs is None else competitor_kwargs

        device = gpu_device if torch.cuda.is_available() else 'cpu:0'

        scale_parametrisation = model.component_embedding.scale_parametrisation

        # Test inputs
        phi, x, z = complete_distribution.sample((n_test_priors,))

        # Device
        phi = phi.to(device)
        z = {key: val.to(device) for key, val in z.items()}
        model = model.to(device)

        # Exact prior
        phi_prior_dict = complete_distribution.meta_prior.decode_sample(phi)
        prior = complete_distribution.meta_prior.prior(**phi_prior_dict)

        # Model solution
        start_time = time()
        phi_in, phi_out = model(phi.to(device), **z)
        model_inference_time = time() - start_time
        model_prior = GaussianMixtureModel(**decode_gmm_sample(phi_in, scale_parametrisation))
        model_posterior = GaussianMixtureModel(**decode_gmm_sample(phi_out, scale_parametrisation))

        prior_kl_divergence = kl_divergence(prior, model_prior, model.sample_space_transform,
                                            n_kl_samples)
        model_expected_prior_kl_divergence = prior_kl_divergence.mean().item()
        model_std_prior_kl_divergence = prior_kl_divergence.std().item()

        model_posterior_nll = -model_posterior.log_prob(model.sample_space_transform(x)
                                                        .reshape(model_posterior.batch_shape
                                                                 + model_posterior.event_shape).to(device))
        model_posterior_nll -= torch.logdet(vmap(jacrev(model.sample_space_transform))
                                            (x.reshape(model_posterior.batch_shape
                                                       + model_posterior.event_shape).to(device)
                                             ).reshape(prior.batch_shape + model_posterior.event_shape
                                                       + model_posterior.event_shape))
        model_posterior_expected_nll = model_posterior_nll.mean().item()
        model_posterior_std_nll = model_posterior_nll.std().item()

        model_size = get_model_size(model)

        print(f"GMM approximation prior mean KL divergence: {prior_kl_divergence.mean().item()}")

        if "vi" in competitor_kwargs:
            # VI solution
            before_vi = time()
            vi = VI(model.state_size, prior, complete_distribution.observation_model, inverse_transform,
                    **competitor_kwargs["vi"]).to(device)
            torch.set_grad_enabled(True)
            vi.fit(z, **competitor_kwargs["vi"])
            torch.set_grad_enabled(False)
            vi_time = time() - before_vi
            vi_nll = -vi.distribution().log_prob(model.sample_space_transform(x)
                                                 .reshape(model_posterior.batch_shape
                                                          + model_posterior.event_shape).to(device))
            vi_nll -= torch.logdet(vmap(jacrev(model.sample_space_transform))
                                   (x.reshape(model_posterior.batch_shape
                                              + model_posterior.event_shape).to(device)
                                    ).reshape(prior.batch_shape + model_posterior.event_shape
                                              + model_posterior.event_shape))
            vi_expected_nll = vi_nll.mean().item()
            vi_std_nll = vi_nll.std().item()
            vi_elbo = -vi.posterior_loss(z, n_samples=n_test_priors)
            vi_expected_elbo = vi_elbo.mean().item()
            vi_std_elbo = vi_elbo.std().item()
            model_elbo = -vi.posterior_loss(z, n_samples=n_test_priors, distribution=model_posterior)
            model_expected_elbo = model_elbo.mean().item()
            model_std_elbo = model_elbo.std().item()

        if "pfns" in competitor_kwargs:
            assert torch.prod(torch.tensor(prior.event_shape)).item() == 1, \
                "pfns only supported for univariate output distributions"
            # PFN solution
            pfn_kwargs = copy(competitor_kwargs["pfns"])
            del pfn_kwargs["training_kwargs"]
            pfn = PFN(**pfn_kwargs, **model.observation_embeddings)
            torch.set_grad_enabled(True)
            pfn, _ = train_pfn(pfn, complete_distribution, _run=_run, **competitor_kwargs["pfns"]["training_kwargs"])
            torch.set_grad_enabled(False)
            pfn = pfn.to(device)
            start_time = time()
            phi_out = pfn(**z)
            pfn_inference_time = time() - start_time
            pfn_posterior = RiemannDistribution(phi_out, pfn.borders, pfn.infinite_support)

            pfn_nll = -pfn_posterior.log_prob(x.reshape(pfn_posterior.batch_shape).to(device))
            pfn_expected_nll = pfn_nll.mean().item()
            pfn_std_nll = pfn_nll.std().item()

            pfn_size = get_model_size(pfn)

            if "vi" in competitor_kwargs:
                pfn_elbo = -vi.posterior_loss(z, n_samples=n_test_priors, distribution=pfn_posterior,
                                              inverse_transform=Identity())
                pfn_expected_elbo = pfn_elbo.mean().item()
                pfn_std_elbo = pfn_elbo.std().item()
        
        if "tabpfn" in competitor_kwargs:
            assert torch.prod(torch.tensor(prior.event_shape)).item() == 1, \
                "pfns only supported for univariate output distributions"
            
            tabpfn_kwargs = copy(competitor_kwargs["tabpfn"])

            start_time = time()
            
            _, tabpfn_nll, tabpfn_regressor = test_tabpfn(
                {key: val for key, val in phi_prior_dict.items()}, 
                x, 
                {key: val for key, val in z.items()},
                complete_distribution,
                tabpfn_trainsize=tabpfn_kwargs["n_training_samples"]
            )
            
            tabpfn_inference_time = time() - start_time
            tabpfn_size = get_model_size(tabpfn_regressor.model_)

            tabpfn_expected_nll = tabpfn_nll.mean().item()
            tabpfn_std_nll = tabpfn_nll.std().item()
        
        if "ace" in competitor_kwargs:

            # ACE solution
            ace_kwargs = copy(competitor_kwargs["ace"])
            del ace_kwargs["training_kwargs"]

            size_of_x = torch.tensor(prior.event_shape).item() if torch.tensor(prior.event_shape).numel() > 0 else 1
            num_latent = size_of_x + complete_distribution.meta_prior.prior_size + sum(obs.n_observations for obs in complete_distribution.observation_model.values())
            ace = get_ace_model(num_latent=num_latent, **ace_kwargs['transformer_kwargs'])
            torch.set_grad_enabled(True)
            ace, _ = train_ace(ace, complete_distribution, sample_space_transform=model.sample_space_transform, _run=_run, **competitor_kwargs["ace"]["training_kwargs"])
            torch.set_grad_enabled(False)
            
            start_time = time()
            ace_posterior = predict_w_ace(phi.to(device), x.to(device), {k:v.to(device) for k,v in z.items()}, ace.to(device))
            ace_inference_time = time() - start_time

            ace_nll = -ace_posterior.log_prob(x.reshape(ace_posterior.batch_shape + ace_posterior.event_shape).to(device))
            ace_expected_nll = ace_nll.mean().item()
            ace_std_nll = ace_nll.std().item()
            
            ace_size = get_model_size(ace)

        # Single problem run

        # Test inputs
        phi, x, z = complete_distribution.sample()

        # Device
        model.cpu()

        phi_prior_dict = complete_distribution.meta_prior.decode_sample(phi)
        prior = complete_distribution.meta_prior.prior(**phi_prior_dict)

        # Model solution
        start_time = time()
        phi_in, phi_out = model(phi, **z)
        model_single_inference_time = time() - start_time
        model_prior = GaussianMixtureModel(**decode_gmm_sample(phi_in, scale_parametrisation))
        model_posterior = GaussianMixtureModel(**decode_gmm_sample(phi_out, scale_parametrisation))

        if _run is not None:
            _run.info.update({
                "model_expected_prior_kl_divergence": model_expected_prior_kl_divergence,
                "model_std_prior_kl_divergence": model_std_prior_kl_divergence,
                "model_inference_time": model_inference_time,
                "model_single_inference_time": model_single_inference_time,
                "model_posterior_expected_nll": model_posterior_expected_nll,
                "model_posterior_std_nll": model_posterior_std_nll,
                "model_size": model_size,
            })
            if "vi" in competitor_kwargs:
                _run.info.update({
                    "vi_time": vi_time,
                    "model_expected_elbo": model_expected_elbo,
                    "model_std_elbo": model_std_elbo,
                    "vi_expected_elbo": vi_expected_elbo,
                    "vi_std_elbo": vi_std_elbo,
                    "vi_expected_nll": vi_expected_nll,
                    "vi_std_nll": vi_std_nll
                })
            if "pfns" in competitor_kwargs:
                _run.info.update({
                    "pfn_inference_time": pfn_inference_time,
                    "pfn_expected_nll": pfn_expected_nll,
                    "pfn_std_nll": pfn_std_nll,
                    "pfn_size": pfn_size
                })
                if "vi" in competitor_kwargs:
                    _run.info.update({
                        "pfn_expected_elbo": pfn_expected_elbo,
                        "pfn_std_elbo": pfn_std_elbo,
                    })
            if "tabpfn" in competitor_kwargs:
                _run.info.update({
                    "tabpfn_inference_time": tabpfn_inference_time,
                    "tabpfn_expected_nll": tabpfn_expected_nll,
                    "tabpfn_std_nll": tabpfn_std_nll,
                    "tabpfn_size": tabpfn_size
                })

            if "ace" in competitor_kwargs:
                _run.info.update({
                    "ace_inference_time": ace_inference_time,
                    "ace_expected_nll": ace_expected_nll,
                    "ace_std_nll": ace_std_nll,
                    "ace_size": ace_size
                })

        # Plotting
        if plot:
            assert torch.prod(torch.tensor(prior.event_shape)).item() == 1, \
                "plotting only supported for single output distributions"

            if bounds_func is None:
                def bounds_func(params_dict: dict[str, Tensor]) -> tuple[float, float]:
                    dist = complete_distribution.meta_prior.prior(**params_dict)
                    samples = dist.sample((10000,))
                    samples = samples.sort().values
                    return samples[499].item(), samples[9499].item()

            prior_plot = plot_distributions(prior, model_prior, None, model.sample_space_transform,
                                            bounds_func(phi_prior_dict), n_kl_samples=n_kl_samples)

            if len(competitor_kwargs) == 0:
                model_posterior_plot = plot_distributions(model_posterior, None, model.sample_space_transform,
                                                          n_kl_samples=n_kl_samples)

            if "vi" in competitor_kwargs:
                vi = VI(model.state_size, prior, complete_distribution.observation_model, inverse_transform)
                torch.set_grad_enabled(True)
                start_time = time()
                vi.fit(z, **competitor_kwargs["vi"])
                vi_single_time = time() - start_time
                torch.set_grad_enabled(False)
                vi_posterior = vi.distribution()
                vi_posterior_plot = plot_distributions(vi_posterior, model_posterior, model.sample_space_transform,
                                                       model.sample_space_transform, bounds_func(phi_prior_dict),
                                                       n_kl_samples=None)

            if "pfns" in competitor_kwargs:
                pfn = pfn.cpu()
                start_time = time()
                phi_out = pfn(**z)
                pfn_single_inference_time = time() - start_time
                pfn_posterior = RiemannDistribution(phi_out, pfn.borders, pfn.infinite_support)
                pfn_posterior_plot = plot_distributions(pfn_posterior, model_posterior, None,
                                                        model.sample_space_transform, bounds_func(phi_prior_dict),
                                                        n_kl_samples=None)
            
            if "tabpfn" in competitor_kwargs:
                start_time = time()
                tabpfn_posterior, _, _ = test_tabpfn(
                    {key: val for key, val in phi_prior_dict.items()}, 
                    x, 
                    {key: val for key, val in z.items()},
                    complete_distribution,
                    tabpfn_trainsize=tabpfn_kwargs["n_training_samples"]
                )
                tabpfn_single_inference_time = time() - start_time
                tabpfn_posterior_plot = plot_distributions(tabpfn_posterior, model_posterior, None,
                                                        model.sample_space_transform, bounds_func(phi_prior_dict),
                                                        n_kl_samples=None)

            if "ace" in competitor_kwargs:
                ace = ace.cpu()
                start_time = time()
                ace_posterior = predict_w_ace(phi, x, z, ace)
                ace_single_inference_time = time() - start_time
                ace_posterior_plot = plot_distributions(ace_posterior, model_posterior, None,
                                                        model.sample_space_transform, bounds_func(phi_prior_dict),
                                                        n_kl_samples=None)

            if _run is not None:
                prior_plot.savefig(_run.observers[0].dir + "\\prior_plot.pdf", format="pdf")
                if len(competitor_kwargs) == 0:
                    model_posterior_plot.savefig(_run.observers[0].dir + "\\model_posterior_plot.pdf", format="pdf")
                if "vi" in competitor_kwargs:
                    _run.info.update({
                        "vi_single_time": vi_single_time
                    })
                    vi_posterior_plot.savefig(_run.observers[0].dir + "\\vi_posterior_plot.pdf", format="pdf")

                if "pfns" in competitor_kwargs:
                    pfn_posterior_plot.savefig(_run.observers[0].dir + "\\pfn_posterior_plot.pdf", format="pdf")
                    _run.info.update({
                        "pfn_single_inference_time": pfn_single_inference_time
                    })
                if "tabpfn" in competitor_kwargs:
                    tabpfn_posterior_plot.savefig(_run.observers[0].dir + "\\tabpfn_posterior_plot.pdf", format="pdf")
                    _run.info.update({
                        "tabpfn_single_inference_time": tabpfn_single_inference_time
                    })
                if "ace" in competitor_kwargs:
                    ace_posterior_plot.savefig(_run.observers[0].dir + "\\ace_posterior_plot.pdf", format="pdf")
                    _run.info.update({
                        "ace_single_inference_time": ace_single_inference_time
                    })


def test_quantum(model: DistributionTransformer,
         complete_distribution: CompleteDistribution,
         competitor_kwargs: Optional[dict[str, dict]] = None,
         inverse_transform: Optional[Callable[[Tensor], Tensor]] = None,
         n_test_priors: int = 1000,
         n_kl_samples: int = 10000,
         plot: bool = False,
         bounds_func: Optional[Callable[[dict[str, Tensor]], tuple[float, float]]] = None,
         gpu_device: str = "cuda:0",
         _run=None
         ) -> None:
    """
    Testing routine for quantum experiment

    Args:
        model: Model to Test.
        complete_distribution: Complete distribution over priors, state and observation.
        competitor_kwargs: Dictionary of dictionaries of parameters for competitor methods.
            Defaults to None.
        inverse_transform: Transform from sample space of GMM approximation to prior.
            Defaults to None.
        n_test_priors: Number of priors to test model with.
            Defaults to 1000.
        n_kl_samples: Number of samples to take when computing KL divergences.
            Defaults to 10000.
        plot: Whether to plot.
            Defaults to False.
        gpu_device: GPU device.
            Defaults to "cuda:0".
        bounds_func: Function to calculate plotting bounds from exact distribution parameters.
            Defaults to an estimate of the 5-95%ile from 10000 samples
        _run: Sacred run object.

    """

    with torch.no_grad():
        competitor_kwargs = dict() if competitor_kwargs is None else competitor_kwargs

        device = gpu_device if torch.cuda.is_available() else 'cpu:0'

        scale_parametrisation = model.component_embedding.scale_parametrisation

        # Test inputs
        phi, x, z = complete_distribution.sample((n_test_priors,))

        # Device
        phi = phi.to(device)
        z = {key: val.to(device) for key, val in z.items()}
        z_tensor = torch.cat([zi for _, zi in z.items()], dim=-1)
        model = model.to(device)

        # Exact prior
        phi_prior_dict = complete_distribution.meta_prior.decode_sample(phi)
        prior = complete_distribution.meta_prior.prior(**phi_prior_dict)

        # Model solution
        start_time = time()
        phi_in, phi_out = model(phi.to(device), **z)
        model_inference_time = time() - start_time
        model_prior = GaussianMixtureModel(**decode_gmm_sample(phi_in, scale_parametrisation))
        model_posterior = GaussianMixtureModel(**decode_gmm_sample(phi_out, scale_parametrisation))

        prior_kl_divergence = kl_divergence(prior, model_prior, model.sample_space_transform,
                                            n_kl_samples)
        model_prior_kl_divergence = prior_kl_divergence.mean().item()
        model_prior_kl_divergence_conf = prior_kl_divergence.std().item() * 1.96 / sqrt(prior_kl_divergence.numel())

        model_posterior_nll, model_posterior_nll_conf = nll(model_posterior, x, model.sample_space_transform)
        model_posterior_rmse, model_posterior_rmse_conf = rmse(model_posterior, x, inverse_transform)

        def rbf_no_normalisation(x: Tensor) -> Tensor:
            d = torch.cdist(x, x)
            mask = ~torch.eye(d.shape[-1], dtype=torch.bool, device=x.device)
            scale = d[0, mask].median() if d.dim() > 2 else d[mask].median()
            return torch.exp(-0.5 * d ** 2 / scale ** 2)

        model_posterior_mmd_prior, model_posterior_mmd_prior_conf = mmd(model_posterior, x, None, phi, inverse_transform, z_kernel=rbf_no_normalisation)
        model_posterior_mmd_joint, model_posterior_mmd_joint_conf = mmd(model_posterior, x, z_tensor, phi, z_kernel=rbf_no_normalisation)

        model_size = get_model_size(model)

        print(f"GMM approximation prior mean KL divergence: {prior_kl_divergence.mean().item()}")

        if "vi" in competitor_kwargs:
            # VI solution
            vi = VI(model.state_size, prior, complete_distribution.observation_model, inverse_transform,
                    **competitor_kwargs["vi"]).to(device)
            vi_expected_nll_series = []
            vi_conf_nll_series = []
            vi_time_series = []
            if "repeats" in competitor_kwargs["vi"]:
                repeats = competitor_kwargs["vi"]["repeats"]
            else:
                repeats = 1

            for i in range(repeats):
                before_vi = time()
                torch.set_grad_enabled(True)
                vi.fit(z, epoch=i, num_epochs=repeats, **competitor_kwargs["vi"])
                torch.set_grad_enabled(False)
                vi_time = time() - before_vi
                vi_nll = -vi.distribution().log_prob(model.sample_space_transform(x)
                                                     .reshape(model_posterior.batch_shape
                                                              + model_posterior.event_shape).to(device))
                vi_nll -= torch.logdet(vmap(jacrev(model.sample_space_transform))
                                       (x.reshape(model_posterior.batch_shape
                                                  + model_posterior.event_shape).to(device)
                                        ).reshape(prior.batch_shape + model_posterior.event_shape
                                                  + model_posterior.event_shape))
                vi_expected_nll = vi_nll.mean().item()
                vi_std_nll = vi_nll.std().item()
                vi_conf_nll = vi_nll.std().item() * 1.96 / sqrt(n_test_priors)
                vi_expected_nll_series.append(vi_expected_nll)
                
                vi_conf_nll_series.append(vi_conf_nll)
                vi_time_series.append(vi_time)

            vi_posterior_nll, vi_posterior_nll_conf = nll(vi.distribution(), x, model.sample_space_transform)
            vi_posterior_rmse, vi_posterior_rmse_conf = rmse(vi.distribution(), x, inverse_transform)
            vi_posterior_mmd_prior, vi_posterior_mmd_prior_conf = mmd(vi.distribution(), x, None, phi, inverse_transform, z_kernel=rbf_no_normalisation)
            vi_posterior_mmd_joint, vi_posterior_mmd_joint_conf = mmd(vi.distribution(), x, z_tensor, phi, inverse_transform, z_kernel=rbf_no_normalisation)

            vi_time = sum(vi_time_series)
            vi_time_series = np.array(vi_time_series).cumsum()



        if "pfns" in competitor_kwargs:
            assert torch.prod(torch.tensor(prior.event_shape)).item() == 1, \
                "pfns only supported for univariate output distributions"
            # PFN solution
            pfn_kwargs = copy(competitor_kwargs["pfns"])
            del pfn_kwargs["training_kwargs"]
            pfn = PFN(**pfn_kwargs, **model.observation_embeddings)
            torch.set_grad_enabled(True)
            pfn, _ = train_pfn(pfn, complete_distribution, _run=_run, **competitor_kwargs["pfns"]["training_kwargs"])
            torch.set_grad_enabled(False)
            pfn = pfn.to(device)
            start_time = time()
            phi_out = pfn(**z)
            pfn_inference_time = time() - start_time
            pfn_posterior = RiemannDistribution(phi_out, pfn.borders, pfn.infinite_support)

            pfn_posterior_nll, pfn_posterior_nll_conf = nll(pfn_posterior, x)
            pfn_posterior_rmse, pfn_posterior_rmse_conf = rmse(pfn_posterior, x)
            pfn_posterior_mmd_prior, pfn_posterior_mmd_prior_conf = mmd(pfn_posterior, x, None, phi, z_kernel=rbf_no_normalisation)
            pfn_posterior_mmd_joint, pfn_posterior_mmd_joint_conf = mmd(pfn_posterior, x, z_tensor, phi, z_kernel=rbf_no_normalisation)

            pfn_size = get_model_size(pfn)
        
        if "tabpfn" in competitor_kwargs:
            assert torch.prod(torch.tensor(prior.event_shape)).item() == 1, \
                "pfns only supported for univariate output distributions"
            
            tabpfn_kwargs = copy(competitor_kwargs["tabpfn"])

            start_time = time()
            
            tabpfn_posterior, tabpfn_nll, tabpfn_regressor = test_tabpfn(
                {key: val for key, val in phi_prior_dict.items()}, 
                x, 
                {key: val for key, val in z.items()},
                complete_distribution,
                tabpfn_trainsize=tabpfn_kwargs["n_training_samples"]
            )
            
            tabpfn_inference_time = time() - start_time
            tabpfn_size = get_model_size(tabpfn_regressor.model_)

            tabpfn_posterior_nll, tabpfn_posterior_nll_conf = nll(tabpfn_posterior, x)
            tabpfn_posterior_rmse, tabpfn_posterior_rmse_conf = rmse(tabpfn_posterior, x)
            tabpfn_posterior_mmd_prior, tabpfn_posterior_mmd_prior_conf = mmd(tabpfn_posterior, x, None, phi, z_kernel=rbf_no_normalisation)
            tabpfn_posterior_mmd_joint, tabpfn_posterior_mmd_joint_conf = mmd(tabpfn_posterior, x, z_tensor, phi, z_kernel=rbf_no_normalisation)

        """
        if "vi" in competitor_kwargs:
            # Plot loss timeseries
            plt.style.use(['seaborn-v0_8-paper'])
            fig, ax = plt.subplots()
            legend_order = []
            ax.plot(vi_time_series, vi_expected_nll_series,
                    "-", color="tab:orange")
            legend_order.append("SVI")
            if "pfns" in competitor_kwargs:
                ax.plot([pfn_inference_time, vi_time_series[-1]], [pfn_expected_nll] * 2,
                        "--", color="tab:green")
                legend_order.append("PFN")
            if "tabpfn" in competitor_kwargs:
                ax.plot([tabpfn_inference_time, vi_time_series[-1]], [tabpfn_expected_nll] * 2,
                        "--", color="tab:red")
                legend_order.append("TabPFNv2")
            ax.plot([model_inference_time, vi_time_series[-1]], [model_posterior_expected_nll] * 2,
                    "--", color="tab:blue")
            ax.legend(legend_order + ["Distribution Transformer"])
            ax.fill_between(vi_time_series, np.array(vi_expected_nll_series) + np.array(vi_conf_nll_series),
                            np.array(vi_expected_nll_series) - np.array(vi_conf_nll_series),
                            color="tab:orange", alpha=0.5)
            if "pfns" in competitor_kwargs:
                ax.fill_between([pfn_inference_time, vi_time_series[-1]],
                                np.array([pfn_expected_nll] * 2) + np.array([pfn_conf_nll] * 2),
                                np.array([pfn_expected_nll] * 2) - np.array([pfn_conf_nll] * 2),
                                color="tab:green", alpha=0.5)
            if "tabpfn" in competitor_kwargs:
                ax.fill_between([tabpfn_inference_time, vi_time_series[-1]],
                                np.array([tabpfn_expected_nll] * 2) + np.array([tabpfn_conf_nll] * 2),
                                np.array([tabpfn_expected_nll] * 2) - np.array([tabpfn_conf_nll] * 2),
                                color="tab:red", alpha=0.5)
            ax.fill_between([model_inference_time, vi_time_series[-1]],
                            np.array([model_posterior_expected_nll] * 2) + np.array([model_posterior_conf_nll] * 2),
                            np.array([model_posterior_expected_nll] * 2) - np.array([model_posterior_conf_nll] * 2),
                            color="tab:blue", alpha=0.5)
            ax.set_xlabel(f"Inference Time per {n_test_priors} Problem Batch (s)")
            ax.set_ylabel("Negative Log-Likelihood")
            ax.set_xscale("log")
            fig.savefig(_run.observers[0].dir + "\\loss_series.pdf", format="pdf")
            plt.show()
        """
        
        if "ace" in competitor_kwargs:
            # ACE solution
            ace_kwargs = copy(competitor_kwargs["ace"])
            del ace_kwargs["training_kwargs"]

            size_of_x = torch.tensor(prior.event_shape).item() if torch.tensor(prior.event_shape).numel() > 0 else 1
            size_of_phi = complete_distribution.meta_prior.prior_size 
            size_of_z = sum(
                torch.tensor(obs.event_shape).item() if torch.tensor(obs.event_shape).numel() > 0 else 1
                for obs in complete_distribution.observation_model.values()
            )
            num_latent = size_of_x + size_of_phi + size_of_z
            ace = get_ace_model(num_latent=num_latent, **ace_kwargs['transformer_kwargs'])
            torch.set_grad_enabled(True)
            ace, _ = train_ace(ace, complete_distribution, sample_space_transform=model.sample_space_transform, _run=_run, **competitor_kwargs["ace"]["training_kwargs"])
            torch.set_grad_enabled(False)
            
            start_time = time()
            ace_posterior = predict_w_ace(phi.to(device), x.to(device), {k:v.to(device) for k,v in z.items()}, ace.to(device))
            ace_inference_time = time() - start_time

            ace_posterior_nll, ace_posterior_nll_conf = nll(ace_posterior, x, model.sample_space_transform)
            ace_posterior_rmse, ace_posterior_rmse_conf = rmse(ace_posterior, x, inverse_transform)
            ace_posterior_mmd_prior, ace_posterior_mmd_prior_conf = mmd(ace_posterior, x, None, phi, inverse_transform, z_kernel=rbf_no_normalisation, bootstrap_samples=100, bootstrap_downsampling=10)
            ace_posterior_mmd_joint, ace_posterior_mmd_joint_conf = mmd(ace_posterior, x, z_tensor, phi,
                                                                        inverse_transform, z_kernel=rbf_no_normalisation, bootstrap_samples=100, bootstrap_downsampling=10)
            
            ace_size = get_model_size(ace)

        # Single problem run

        # Test inputs
        phi, x, z = complete_distribution.sample()

        # Device
        model.cpu()

        phi_prior_dict = complete_distribution.meta_prior.decode_sample(phi)
        prior = complete_distribution.meta_prior.prior(**phi_prior_dict)

        # Model solution
        start_time = time()
        phi_in, phi_out = model(phi, **z)
        model_single_inference_time = time() - start_time
        model_prior = GaussianMixtureModel(**decode_gmm_sample(phi_in, scale_parametrisation))
        model_posterior = GaussianMixtureModel(**decode_gmm_sample(phi_out, scale_parametrisation))

        if _run is not None:
            _run.info.update({
                "model_prior_kl_divergence": model_prior_kl_divergence,
                "model_prior_kl_divergence_conf": model_prior_kl_divergence_conf,
                "model_inference_time": model_inference_time,
                "model_single_inference_time": model_single_inference_time,
                "model_posterior_nll": model_posterior_nll,
                "model_posterior_nll_conf": model_posterior_nll_conf,
                "model_posterior_rmse": model_posterior_rmse,
                "model_posterior_rmse_conf": model_posterior_rmse_conf,
                "model_posterior_mmd_prior": model_posterior_mmd_prior,
                "model_posterior_mmd_prior_conf": model_posterior_mmd_prior_conf,
                "model_posterior_mmd_joint": model_posterior_mmd_joint,
                "model_posterior_mmd_joint_conf": model_posterior_mmd_joint_conf,
                "model_size": model_size,
            })
            if "vi" in competitor_kwargs:
                _run.info.update({
                    "vi_time": vi_time,
                    "vi_posterior_nll": vi_posterior_nll,
                    "vi_posterior_nll_conf": vi_posterior_nll_conf,
                    "vi_posterior_rmse": vi_posterior_rmse,
                    "vi_posterior_rmse_conf": vi_posterior_rmse_conf,
                    "vi_posterior_mmd_prior": vi_posterior_mmd_prior,
                    "vi_posterior_mmd_prior_conf": vi_posterior_mmd_prior_conf,
                    "vi_posterior_mmd_joint": vi_posterior_mmd_joint,
                    "vi_posterior_mmd_joint_conf": vi_posterior_mmd_joint_conf,
                })
            if "pfns" in competitor_kwargs:
                _run.info.update({
                    "pfn_inference_time": pfn_inference_time,
                    "pfn_posterior_nll": pfn_posterior_nll,
                    "pfn_posterior_nll_conf": pfn_posterior_nll_conf,
                    "pfn_posterior_rmse": pfn_posterior_rmse,
                    "pfn_posterior_rmse_conf": pfn_posterior_rmse_conf,
                    "pfn_posterior_mmd_prior": pfn_posterior_mmd_prior,
                    "pfn_posterior_mmd_prior_conf": pfn_posterior_mmd_prior_conf,
                    "pfn_posterior_mmd_joint": pfn_posterior_mmd_joint,
                    "pfn_posterior_mmd_joint_conf": pfn_posterior_mmd_joint_conf,
                    "pfn_size": pfn_size
                })
            if "tabpfn" in competitor_kwargs:
                _run.info.update({
                    "tabpfn_inference_time": tabpfn_inference_time,
                    "tabpfn_posterior_nll": tabpfn_posterior_nll,
                    "tabpfn_posterior_nll_conf": tabpfn_posterior_nll_conf,
                    "tabpfn_posterior_rmse": tabpfn_posterior_rmse,
                    "tabpfn_posterior_rmse_conf": tabpfn_posterior_rmse_conf,
                    "tabpfn_posterior_mmd_prior": tabpfn_posterior_mmd_prior,
                    "tabpfn_posterior_mmd_prior_conf": tabpfn_posterior_mmd_prior_conf,
                    "tabpfn_posterior_mmd_joint": tabpfn_posterior_mmd_joint,
                    "tabpfn_posterior_mmd_joint_conf": tabpfn_posterior_mmd_joint_conf,
                    "tabpfn_size": tabpfn_size
                })
            if "ace" in competitor_kwargs:
                _run.info.update({
                    "ace_inference_time": ace_inference_time,
                    "ace_posterior_nll": ace_posterior_nll,
                    "ace_posterior_nll_conf": ace_posterior_nll_conf,
                    "ace_posterior_rmse": ace_posterior_rmse,
                    "ace_posterior_rmse_conf": ace_posterior_rmse_conf,
                    "ace_posterior_mmd_prior": ace_posterior_mmd_prior,
                    "ace_posterior_mmd_prior_conf": ace_posterior_mmd_prior_conf,
                    "ace_posterior_mmd_joint": ace_posterior_mmd_joint,
                    "ace_posterior_mmd_joint_conf": ace_posterior_mmd_joint_conf,
                    "ace_size": ace_size
                })

        # Plotting
        if plot:
            assert torch.prod(torch.tensor(prior.event_shape)).item() == 1, \
                "plotting only supported for single output distributions"

            if bounds_func is None:
                def bounds_func(params_dict: dict[str, Tensor]) -> tuple[float, float]:
                    dist = complete_distribution.meta_prior.prior(**params_dict)
                    samples = dist.sample((10000,))
                    samples = samples.sort().values
                    return samples[499].item(), samples[9499].item()

            prior_plot = plot_distributions(prior, model_prior, None, model.sample_space_transform,
                                            bounds_func(phi_prior_dict), n_kl_samples=n_kl_samples)

            if len(competitor_kwargs) == 0:
                model_posterior_plot = plot_distributions(model_posterior, None, model.sample_space_transform,
                                                          n_kl_samples=n_kl_samples)

            if "vi" in competitor_kwargs:
                vi = VI(model.state_size, prior, complete_distribution.observation_model, inverse_transform)
                torch.set_grad_enabled(True)
                start_time = time()
                vi.fit(z, **competitor_kwargs["vi"])
                vi_single_time = time() - start_time
                torch.set_grad_enabled(False)
                vi_posterior = vi.distribution()
                vi_posterior_plot = plot_distributions(vi_posterior, model_posterior, model.sample_space_transform,
                                                       model.sample_space_transform, bounds_func(phi_prior_dict),
                                                       n_kl_samples=None)

            if "pfns" in competitor_kwargs:
                pfn = pfn.cpu()
                start_time = time()
                phi_out = pfn(**z)
                pfn_single_inference_time = time() - start_time
                pfn_posterior = RiemannDistribution(phi_out, pfn.borders, pfn.infinite_support)
                pfn_posterior_plot = plot_distributions(pfn_posterior, model_posterior, None,
                                                        model.sample_space_transform, bounds_func(phi_prior_dict),
                                                        n_kl_samples=None)
            
            if "tabpfn" in competitor_kwargs:
                start_time = time()
                tabpfn_posterior, _, _ = test_tabpfn(
                    {key: val for key, val in phi_prior_dict.items()}, 
                    x, 
                    {key: val for key, val in z.items()},
                    complete_distribution,
                    tabpfn_trainsize=tabpfn_kwargs["n_training_samples"]
                )
                tabpfn_single_inference_time = time() - start_time
                tabpfn_posterior_plot = plot_distributions(tabpfn_posterior, model_posterior, None,
                                                        model.sample_space_transform, bounds_func(phi_prior_dict),
                                                        n_kl_samples=None)
            
            if "ace" in competitor_kwargs:
                ace = ace.cpu()
                start_time = time()
                ace_posterior = predict_w_ace(phi, x, z, ace)
                ace_single_inference_time = time() - start_time
                ace_posterior_plot = plot_distributions(ace_posterior, model_posterior, None,
                                                        model.sample_space_transform, bounds_func(phi_prior_dict),
                                                        n_kl_samples=None)

            if _run is not None:
                prior_plot.savefig(_run.observers[0].dir + "\\prior_plot.pdf", format="pdf")
                if len(competitor_kwargs) == 0:
                    model_posterior_plot.savefig(_run.observers[0].dir + "\\model_posterior_plot.pdf", format="pdf")
                if "vi" in competitor_kwargs:
                    _run.info.update({
                        "vi_single_time": vi_single_time
                    })
                    vi_posterior_plot.savefig(_run.observers[0].dir + "\\vi_posterior_plot.pdf", format="pdf")

                if "pfns" in competitor_kwargs:
                    pfn_posterior_plot.savefig(_run.observers[0].dir + "\\pfn_posterior_plot.pdf", format="pdf")
                    _run.info.update({
                        "pfn_single_inference_time": pfn_single_inference_time
                    })
                if "tabpfn" in competitor_kwargs:
                    tabpfn_posterior_plot.savefig(_run.observers[0].dir + "\\tabpfn_posterior_plot.pdf", format="pdf")
                    _run.info.update({
                        "tabpfn_single_inference_time": tabpfn_single_inference_time
                    })
                if "ace" in competitor_kwargs:
                    ace_posterior_plot.savefig(_run.observers[0].dir + "\\ace_posterior_plot.pdf", format="pdf")
                    _run.info.update({
                        "ace_single_inference_time": ace_single_inference_time
                    })

def test_lti_filter(model: DistributionTransformer,
                    motion_model: LTIMotionModel,
                    observation_model: dict[str, ObservationModel],
                    competitor_kwargs: Optional[dict[str, dict]] = None,
                    series_length: int = 1000,
                    n_test_series: int = 1000,
                    plotting_kwargs: Optional[dict] = None,
                    gpu_device: str = "cuda:0",
                    _run=None
                    ) -> None:
    """
    Test model on Bayesian filtering task. Currently restricted to GMM initial priors.

    Args:
        model: Model to test.
        motion_model: Motion model for dynamical system.
        observation_model: Observation model for dynamical system.
        competitor_kwargs: Kwargs for competitor methods.
            Defaults to None.
        series_length: Length of series to test on.
            Defaults to 1000.
        n_test_series: Number of series to test on.
            Defaults to 1000.
        plotting_kwargs: Plotting kwargs. Set to None to disable plotting.
            Defaults to NOne.
        gpu_device: GPU device to test on.
            Defaults to cuda:0.
        _run: Sacred run object.

    Returns:

    """
    with torch.no_grad():
        device = gpu_device if torch.cuda.is_available() else 'cpu:0'
        model.to(device)
        scale_parametrisation = model.component_embedding.scale_parametrisation

        model_size = get_model_size(model)

        successful_sample_flag = False
        sample_attempts = 0
        while not successful_sample_flag:
            try:
                series = motion_model.sample((series_length, n_test_series))

                for obs_model in observation_model.values():
                    obs_model.condition_(series)

                successful_sample_flag = True
            except Exception as e:
                sample_attempts += 1
                if sample_attempts >= 10:
                    raise e


        observation_series = {key: obs_model.sample().to(device) for key, obs_model in observation_model.items()}
        observation_series_tensor = torch.cat([z for _, z in observation_series.items()], dim=-1).cpu()

        # Model solution
        filter = LTIFilter(model, motion_model)

        start_time = time()
        filtered_series_dict, _ = filter.filter(observation_series, motion_model.x0_distribution)
        model_density = GaussianMixtureModel(**filtered_series_dict)
        model_inference_time = (time() - start_time) / series_length

        model_nll, model_nll_conf = nll(model_density, series)
        model_rmse, model_rmse_conf = rmse(model_density, series)
        model_mmd_prior, model_mmd_prior_conf = mmd(model_density, series, bootstrap_samples=100, bootstrap_downsampling=10)
        model_mmd_joint, model_mmd_joint_conf = mmd(model_density, series, observation_series_tensor, bootstrap_samples=100, bootstrap_downsampling=10)

        if "ekf" in competitor_kwargs:
            ekf = EKF(model.state_size, motion_model, **observation_model)

            start_time = time()
            ekf_filtered_series_dict, _ = ekf.filter(observation_series, motion_model.x0_distribution)
            ekf_density = MultivariateNormal(**ekf_filtered_series_dict)
            ekf_inference_time = (time() - start_time) / series_length

            ekf_nll, ekf_nll_conf = nll(ekf_density, series)
            ekf_rmse, ekf_rmse_conf = rmse(ekf_density, series)
            ekf_mmd_prior, ekf_mmd_prior_conf = mmd(ekf_density, series, bootstrap_samples=100,
                                                        bootstrap_downsampling=10)
            ekf_mmd_joint, ekf_mmd_joint_conf = mmd(ekf_density, series, observation_series_tensor, bootstrap_samples=100, bootstrap_downsampling=10)

        if "particle_filter" in competitor_kwargs:
            particle_filter_kwargs = copy(competitor_kwargs["particle_filter"])
            if "chunk_size" in particle_filter_kwargs:
                chunk_size = particle_filter_kwargs.pop("chunk_size")
            else:
                chunk_size = None
            n_particles = particle_filter_kwargs.pop("n_particles")

            if isinstance(n_particles, int):
                n_particles = [n_particles]

            pf_nlls = []
            pf_nll_confs = []
            pf_inference_times = []

            for n in n_particles:
                particle_filter_kwargs["n_particles"] = n
                particle_filter = ParticleFilter(model.state_size, motion_model, **observation_model,
                                                 **particle_filter_kwargs)

                start_time = time()
                if chunk_size is None:
                    particles, _ = particle_filter.filter(observation_series, motion_model.x0_distribution)
                    particle_filter_density = particle_filter.fit_density(particles)
                else:
                    particle_filter_densities = []
                    n_chunks = int(ceil(observation_series[list(observation_series.keys())[0]].shape[1] / chunk_size))
                    for i in range(n_chunks):
                        observation_series_chunk = {key: val[:, i::n_chunks] for key, val in observation_series.items()}
                        particles, _ = particle_filter.filter(observation_series_chunk, motion_model.x0_distribution)
                        particle_filter_densities.append(particle_filter.fit_density(particles))
                    particle_filter_density = MultivariateNormal(
                        loc=torch.concat([filter.loc for filter in particle_filter_densities], dim=0),
                        scale_tril=torch.concat([filter.scale_tril for filter in particle_filter_densities], dim=0)
                    )


                particle_filter_inference_time = (time() - start_time) / series_length

                particle_filter_nll, particle_filter_nll_conf = nll(particle_filter_density, series)
                particle_filter_rmse, particle_filter_rmse_conf = rmse(particle_filter_density, series)
                particle_filter_mmd_prior, particle_filter_mmd_prior_conf = mmd(particle_filter_density, series, bootstrap_samples=100,
                                                            bootstrap_downsampling=10)
                particle_filter_mmd_joint, particle_filter_mmd_joint_conf = mmd(particle_filter_density, series, observation_series_tensor, bootstrap_samples=100, bootstrap_downsampling=10)

                pf_nlls.append(particle_filter_nll)
                pf_nll_confs.append(particle_filter_nll_conf)
                pf_inference_times.append(particle_filter_inference_time)


        # Single problem run

        # Device
        model.cpu()

        series = motion_model.sample((series_length,))

        for obs_model in observation_model.values():
            obs_model.condition_(series)

        observation_series = {key: obs_model.sample().cpu() for key, obs_model in observation_model.items()}

        # Model solution
        filter = LTIFilter(model, motion_model)

        start_time = time()
        filtered_series_dict, _ = filter.filter(observation_series, motion_model.x0_distribution)
        _ = GaussianMixtureModel(**filtered_series_dict)
        model_single_inference_time = (time() - start_time) / series_length

        if "ekf" in competitor_kwargs:
            start_time = time()
            ekf_filtered_series_dict, _ = ekf.filter(observation_series, motion_model.x0_distribution)
            _ = MultivariateNormal(**ekf_filtered_series_dict)
            ekf_single_inference_time = (time() - start_time) / series_length

        if "particle_filter" in competitor_kwargs:

            pf_single_inference_times = []

            for n in n_particles:
                particle_filter_kwargs["n_particles"] = n
                particle_filter = ParticleFilter(model.state_size, motion_model, **observation_model,
                                                 **particle_filter_kwargs)
                start_time = time()
                particles, _ = particle_filter.filter(observation_series, motion_model.x0_distribution)
                _ = particle_filter.fit_density(particles)
                particle_filter_single_inference_time = (time() - start_time) / series_length
                pf_single_inference_times.append(particle_filter_single_inference_time)

        if _run is not None:
            _run.info.update({
                "model_inference_time": model_inference_time,
                "model_single_inference_time": model_single_inference_time,
                "model_nll": model_nll,
                "model_nll_conf": model_nll_conf,
                "model_rmse": model_rmse,
                "model_rmse_conf": model_rmse_conf,
                "model_mmd_prior": model_mmd_prior,
                "model_mmd_prior_conf": model_mmd_prior_conf,
                "model_mmd_joint": model_mmd_joint,
                "model_mmd_joint_conf": model_mmd_joint_conf,
                "model_size": model_size,
            })
            if "ekf" in competitor_kwargs:
                _run.info.update({
                    "ekf_inference_time": ekf_inference_time,
                    "ekf_single_inference_time": ekf_single_inference_time,
                    "ekf_nll": ekf_nll,
                    "ekf_nll_conf": ekf_nll_conf,
                    "ekf_rmse": ekf_rmse,
                    "ekf_rmse_conf": ekf_rmse_conf,
                    "ekf_mmd_prior": ekf_mmd_prior,
                    "ekf_mmd_prior_conf": ekf_mmd_prior_conf,
                    "ekf_mmd_joint": ekf_mmd_joint,
                    "ekf_mmd_joint_conf": ekf_mmd_joint_conf,
                })
            if "particle_filter" in competitor_kwargs:
                _run.info.update({
                    "particle_filter_inference_time": particle_filter_inference_time,
                    "particle_filter_single_inference_time": particle_filter_single_inference_time,
                    "particle_filter_nll": particle_filter_nll,
                    "particle_filter_nll_conf": particle_filter_nll_conf,
                    "particle_filter_rmse": particle_filter_rmse,
                    "particle_filter_rmse_conf": particle_filter_rmse_conf,
                    "particle_filter_mmd_prior": particle_filter_mmd_prior,
                    "particle_filter_mmd_prior_conf": particle_filter_mmd_prior_conf,
                    "particle_filter_mmd_joint": particle_filter_mmd_joint,
                    "particle_filter_mmd_joint_conf": particle_filter_mmd_joint_conf,
                    "particle_filter_nll_series": pf_nlls,
                    "particle_filter_nll_conf_series": pf_nll_confs,
                    "particle_filter_inference_time_series": pf_inference_times,
                    "particle_filter_single_inference_time_series": pf_single_inference_times,
                })

        # Plotting first dimension of state space
        if plotting_kwargs is not None:
            # Select dimension
            dim = plotting_kwargs["dim"]

            filtered_series_dict["loc"] = filtered_series_dict["loc"][..., dim].unsqueeze(-1)
            filtered_series_dict[scale_parametrisation] = \
                filtered_series_dict[scale_parametrisation].diagonal(dim1=-2, dim2=-1)[..., dim].unsqueeze(
                    -1).unsqueeze(-1)
            filtered_series = encode_gmm_sample(filtered_series_dict, scale_parametrisation)
            model_filter_distribution = GaussianMixtureModel(**filtered_series_dict)

            """
            bounds = list(zip(*[gmm_bounds_func(decode_gmm_sample(dist, scale_parametrisation))
                                for dist in filtered_series]))
            bounds = (max(bounds[0]), min(bounds[1]))
            bounds = (max(bounds[0], series.max().item() + 1), min(bounds[1], series.min().item() - 1))
            """

            bounds = None

            model_series_plot = plot_filtered_series(model_filter_distribution, series[..., dim].unsqueeze(-1), bounds,
                                                     **plotting_kwargs)

            if "ekf" in competitor_kwargs and "particle_filter" in competitor_kwargs:
                ekf_filtered_series_dict["loc"] = ekf_filtered_series_dict["loc"][..., dim].unsqueeze(-1)
                ekf_filtered_series_dict["covariance_matrix"] = \
                    ekf_filtered_series_dict["covariance_matrix"].diagonal(dim1=-2, dim2=-1)[..., dim].unsqueeze(
                        -1).unsqueeze(-1)
                ekf_filter_distribution = MultivariateNormal(**ekf_filtered_series_dict)
                ekf_series_plot = plot_filtered_series(ekf_filter_distribution, series[..., dim].unsqueeze(-1), bounds,
                                                       **plotting_kwargs)
                combined_series_plot = plot_filtered_series([ekf_filter_distribution,
                                                             model_filter_distribution],
                                                            series[..., dim].unsqueeze(-1), bounds,
                                                            cmaps=["OrRd", "BuPu"],
                                                            legend_labels=["EKF Filter Density",
                                                                           "Distribution Transformer Filter Density"],
                                                            **plotting_kwargs)

            if _run is not None:
                model_series_plot.savefig(_run.observers[0].dir + "\\model_series_plot.pdf", format="pdf")

                if "ekf" in competitor_kwargs:
                    ekf_series_plot.savefig(_run.observers[0].dir + "\\ekf_series_plot.pdf", format="pdf")
                    combined_series_plot.savefig(_run.observers[0].dir + "\\combined_series_plot.pdf", format="pdf")
