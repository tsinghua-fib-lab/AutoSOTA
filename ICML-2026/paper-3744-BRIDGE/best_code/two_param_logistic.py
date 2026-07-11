# MIT License

# Copyright (c) 2019 John Lalor <john.lalor@nd.edu> and Pedro Rodriguez <me@pedro.ai>
# Modifications (c) 2025 Allen Institute for AI

# Adapted from: https://github.com/nd-ball/py-irt/blob/master/py_irt/models/two_param_logistic.py

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import pyro
import pyro.distributions as dist
import torch

import torch.distributions.constraints as constraints

from pyro.infer import SVI, Trace_ELBO, EmpiricalMarginal, TraceEnum_ELBO
from pyro.infer.mcmc import MCMC, NUTS
from pyro.optim import Adam, SGD

import pyro.contrib.autoguide as autoguide

import pandas as pd

from functools import partial
from py_irt.models import abstract_model

import numpy as np


@abstract_model.IrtModel.register("2pl")
class TwoParamLogistic(abstract_model.IrtModel):
    """2PL IRT model"""

    def __init__(
        self, 
        *, 
        priors: str, 
        num_items: int, 
        num_subjects: int, 
        verbose=False, 
        device: str = "cpu",
        **kwargs
    ):
        super().__init__(
            num_items=num_items, num_subjects=num_subjects, device=device, verbose=verbose
        )
        if priors not in ["vague", "hierarchical"]:
            raise ValueError("Options for priors are vague and hierarchical")
        self.priors = priors

    def model_vague(self, subjects, items, obs):
        """Initialize a 2PL model with vague priors"""
        with pyro.plate("thetas", self.num_subjects, device=self.device):
            ability = pyro.sample(
                "theta",
                dist.Normal(
                    torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device)
                ),
            )

        with pyro.plate("bs", self.num_items, device=self.device):
            diff = pyro.sample(
                "b",
                dist.Normal(
                    torch.tensor(0.0, device=self.device), torch.tensor(0.1, device=self.device)
                ),
            )
            slope = pyro.sample(
                "a",
                dist.LogNormal(
                    torch.tensor(0.0, device=self.device), torch.tensor(0.1, device=self.device)
                ),
            )

        with pyro.plate("observe_data", obs.size(0), device=self.device):
            pyro.sample(
                "obs",
                dist.Bernoulli(logits=(slope[items] * (ability[subjects] - diff[items]))),
                obs=obs,
            )

    def guide_vague(self, subjects, items, obs):
        """Initialize a 2PL guide with vague priors"""
        # register learnable params in the param store
        m_theta_param = pyro.param(
            "loc_ability", torch.zeros(self.num_subjects, device=self.device)
        )
        s_theta_param = pyro.param(
            "scale_ability",
            torch.ones(self.num_subjects, device=self.device),
            constraint=constraints.positive,
        )
        m_b_param = pyro.param("loc_diff", torch.zeros(self.num_items, device=self.device))
        s_b_param = pyro.param(
            "scale_diff",
            torch.empty(self.num_items, device=self.device).fill_(1.0e1),
            constraint=constraints.positive,
        )
        m_a_param = pyro.param(
            "loc_slope",
            torch.ones(self.num_items, device=self.device),
            constraint=constraints.positive,
        )
        s_a_param = pyro.param(
            "scale_slope",
            torch.empty(self.num_items, device=self.device).fill_(1.0e-6),
            constraint=constraints.positive,
        )

        # guide distributions
        with pyro.plate("thetas", self.num_subjects, device=self.device):
            dist_theta = dist.Normal(m_theta_param, s_theta_param)
            pyro.sample("theta", dist_theta)
        with pyro.plate("bs", self.num_items, device=self.device):
            dist_b = dist.Normal(m_b_param, s_b_param)
            pyro.sample("b", dist_b)

            dist_a = dist.LogNormal(m_a_param, s_a_param)
            pyro.sample("a", dist_a)

    def model_hierarchical(self, subjects, items, obs):
        """Initialize a 2PL model with hierarchical priors"""
        mu_b = pyro.sample(
            "mu_b",
            dist.Normal(
                torch.tensor(0.0, device=self.device), torch.tensor(1.0e6, device=self.device)
            ),
        )
        u_b = pyro.sample(
            "u_b",
            dist.Gamma(
                torch.tensor(1.0, device=self.device), torch.tensor(1.0, device=self.device)
            ),
        )
        mu_theta = pyro.sample(
            "mu_theta",
            dist.Normal(
                torch.tensor(0.0, device=self.device), torch.tensor(1.0e6, device=self.device)
            ),
        )
        u_theta = pyro.sample(
            "u_theta",
            dist.Gamma(
                torch.tensor(1.0, device=self.device), torch.tensor(1.0, device=self.device)
            ),
        )
        mu_a = pyro.sample(
            "mu_a",
            dist.Normal(
                torch.tensor(0.0, device=self.device), torch.tensor(1.0e6, device=self.device)
            ),
        )
        u_a = pyro.sample(
            "u_a",
            dist.Gamma(
                torch.tensor(1.0, device=self.device), torch.tensor(1.0, device=self.device)
            ),
        )
        with pyro.plate("thetas", self.num_subjects, device=self.device):
            ability = pyro.sample("theta", dist.Normal(mu_theta, 1.0 / u_theta))
        with pyro.plate("bs", self.num_items, device=self.device):
            diff = pyro.sample("b", dist.Normal(mu_b, 1.0 / u_b))
            slope = pyro.sample("a", dist.LogNormal(mu_a, 1.0 / u_a))
        with pyro.plate("observe_data", obs.size(0)):
            pyro.sample(
                "obs",
                dist.Bernoulli(logits=slope[items] * (ability[subjects] - diff[items])),
                obs=obs,
            )

    def guide_hierarchical(self, subjects, items, obs):
        """Initialize a 2PL guide with hierarchical priors"""
        loc_mu_b_param = pyro.param("loc_mu_b", torch.tensor(0.0, device=self.device))
        scale_mu_b_param = pyro.param(
            "scale_mu_b", torch.tensor(1.0e1, device=self.device), constraint=constraints.positive
        )
        loc_mu_theta_param = pyro.param("loc_mu_theta", torch.tensor(0.0, device=self.device))
        scale_mu_theta_param = pyro.param(
            "scale_mu_theta",
            torch.tensor(1.0e1, device=self.device),
            constraint=constraints.positive,
        )
        loc_mu_a_param = pyro.param("loc_mu_a", torch.tensor(0.0, device=self.device))
        scale_mu_a_param = pyro.param(
            "scale_mu_a", torch.tensor(1.0e1, device=self.device), constraint=constraints.positive
        )
        alpha_b_param = pyro.param(
            "alpha_b", torch.tensor(1.0, device=self.device), constraint=constraints.positive
        )
        beta_b_param = pyro.param(
            "beta_b", torch.tensor(1.0, device=self.device), constraint=constraints.positive
        )
        alpha_theta_param = pyro.param(
            "alpha_theta", torch.tensor(1.0, device=self.device), constraint=constraints.positive
        )
        beta_theta_param = pyro.param(
            "beta_theta", torch.tensor(1.0, device=self.device), constraint=constraints.positive
        )
        alpha_a_param = pyro.param(
            "alpha_a", torch.tensor(1.0, device=self.device), constraint=constraints.positive
        )
        beta_a_param = pyro.param(
            "beta_a", torch.tensor(1.0, device=self.device), constraint=constraints.positive
        )
        m_theta_param = pyro.param(
            "loc_ability", torch.zeros(self.num_subjects, device=self.device)
        )
        s_theta_param = pyro.param(
            "scale_ability",
            torch.ones(self.num_subjects, device=self.device),
            constraint=constraints.positive,
        )
        m_b_param = pyro.param("loc_diff", torch.zeros(self.num_items, device=self.device))
        s_b_param = pyro.param(
            "scale_diff",
            torch.ones(self.num_items, device=self.device),
            constraint=constraints.positive,
        )
        m_a_param = pyro.param("loc_slope", torch.zeros(self.num_items, device=self.device))
        s_a_param = pyro.param(
            "scale_slope",
            torch.ones(self.num_items, device=self.device),
            constraint=constraints.positive,
        )

        # sample statements
        pyro.sample("mu_b", dist.Normal(loc_mu_b_param, scale_mu_b_param))
        pyro.sample("u_b", dist.Gamma(alpha_b_param, beta_b_param))
        pyro.sample("mu_theta", dist.Normal(loc_mu_theta_param, scale_mu_theta_param))
        pyro.sample("u_theta", dist.Gamma(alpha_theta_param, beta_theta_param))
        pyro.sample("mu_a", dist.Normal(loc_mu_a_param, scale_mu_a_param))
        pyro.sample("u_a", dist.Gamma(alpha_a_param, beta_a_param))

        with pyro.plate("thetas", self.num_subjects, device=self.device):
            pyro.sample("theta", dist.Normal(m_theta_param, s_theta_param))
        with pyro.plate("bs", self.num_items, device=self.device):
            pyro.sample("b", dist.Normal(m_b_param, s_b_param))
            pyro.sample("a", dist.LogNormal(m_a_param, s_a_param))

    def get_model(self):
        if self.priors == "vague":
            return self.model_vague
        else:
            return self.model_hierarchical

    def get_guide(self):
        if self.priors == "vague":
            return self.guide_vague
        else:
            return self.guide_hierarchical

    def export(self):
        return {
            "ability": pyro.param("loc_ability").data.tolist(),
            "diff": pyro.param("loc_diff").data.tolist(),
            "disc": pyro.param("loc_slope").data.tolist(),
        }

    def fit_MCMC(self, models, items, responses, num_epochs):
        """Fit the IRT model with MCMC"""
        sites = ["theta", "b"]
        nuts_kernel = NUTS(self.model_vague, adapt_step_size=True)
        hmc_posterior = MCMC(nuts_kernel, num_samples=1000, warmup_steps=100).run(
            models, items, responses
        )
        theta_sum = self.summary(hmc_posterior, ["theta"]).items()
        b_sum = self.summary(hmc_posterior, ["b"]).items()
        print(theta_sum)
        print(b_sum)

    def predict(self, subjects, items, params_from_file=None):
        """predict p(correct | params) for a specified list of model, item pairs"""
        if params_from_file is not None:
            model_params = params_from_file
        else:
            model_params = self.export()
        abilities = np.array([model_params["ability"][i] for i in subjects])
        diffs = np.array([model_params["diff"][i] for i in items])
        discs = np.array([model_params["disc"][i] for i in items])
        return 1 / (1 + np.exp(-discs * (abilities - diffs)))

    def summary(self, traces, sites):
        """Aggregate marginals for MCM"""
        marginal = (
            EmpiricalMarginal(traces, sites)._get_samples_and_weights()[0].detach().cpu().numpy()
        )
        print(marginal)
        site_stats = {}
        for i in range(marginal.shape[1]):
            site_name = sites[i]
            marginal_site = pd.DataFrame(marginal[:, i]).transpose()
            describe = partial(pd.Series.describe, percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
            site_stats[site_name] = marginal_site.apply(describe, axis=1)[
                ["mean", "std", "5%", "25%", "50%", "75%", "95%"]
            ]
        return site_stats
