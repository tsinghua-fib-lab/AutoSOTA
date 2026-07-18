#!/usr/bin/env python3

from typing import Callable, Optional
import torch
from torch import Tensor
from botorch.acquisition import ExpectedImprovement, LogExpectedImprovement, UpperConfidenceBound
from botorch.generation.gen import gen_candidates_torch
from botorch.acquisition import PosteriorMean
from pandora_automl.acquisition.lcb import LowerConfidenceBound
from botorch.acquisition.multi_step_lookahead import warmstart_multistep
from pandora_automl.acquisition.gittins import GittinsIndex
from pandora_automl.acquisition.stable_gittins import StableGittinsIndex
from pandora_automl.acquisition.log_ei_puc import LogExpectedImprovementWithCost
from botorch.sampling.pathwise import draw_matheron_paths
from botorch.utils.sampling import optimize_posterior_samples
from botorch.optim import optimize_acqf
from scipy.stats import norm
from copy import copy
from pandora_automl.utils import fit_gp_model
import numpy as np
import math
import time

class BayesianOptimizer:
    DEFAULT_COST = torch.tensor(1.0)  # Default cost if not provided

    def __init__(self,
                 dim: int, 
                 maximize: bool, 
                 initial_points: Tensor, 
                 objective: Optional[Callable] = None, 
                 cost: Optional[Callable] = None, 
                 objective_cost: Optional[Callable] = None, 
                 kernel: Optional[torch.nn.Module] = None,
                 noisy_observation: bool = False,
                 noise_level: Optional[float] = 0.1,
                 output_standardize: bool = False,
                ):
        self.validate_functions(objective, objective_cost)
        self.initialize_attributes(objective, cost, objective_cost, dim, maximize, initial_points, kernel, noisy_observation, noise_level, output_standardize)


    def validate_functions(self, objective, objective_cost):
        # Make sure that the objective function and the cost function are passed in the correct form
        if objective_cost is None and objective is None:
            raise ValueError("At least one of 'objective' or 'objective_cost' must be provided.")
        if objective is not None and objective_cost is not None:
            raise ValueError("Only one of 'objective' or 'objective_cost' can be provided.")
        self.unknown_cost = callable(objective_cost)


    def initialize_attributes(self, objective, cost, objective_cost, dim, maximize, initial_points, kernel, noisy_observation, noise_level, output_standardize):
        self.objective = objective
        self.cost = cost if cost is not None else self.DEFAULT_COST
        self.objective_cost = objective_cost
        self.dim = dim
        self.maximize = maximize
        # need to be initialized after self.dim
        self.bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])
        self.best_f = None
        self.best_x = None 
        self.best_history = []
        self.cumulative_cost = 0.0
        self.cost_history = [0.0]
        self.acq_history = [np.nan]
        self.stopping_history = {}
        self.runtime_history = []
        self.lmbda_history = []
        # GP model parameters
        self.kernel = kernel
        self.noisy_observation = noisy_observation
        self.noise_level = noise_level
        self.output_standardize = output_standardize
        self.suggested_x_full_tree = None
        self.model = None
        self.old_model = None
        self.num_samples = 64
        self.granularity = 10001
        self.mask = torch.ones(self.granularity, dtype=torch.bool)
        self.iteration = 0
        self.initialize_points(initial_points)
    

    def initialize_points(self, initial_points):
        self.x = initial_points.detach() if hasattr(initial_points, 'detach') else initial_points
        if callable(self.objective):
            self.y = self.objective(initial_points).detach()
            if callable(self.cost):
                self.c = self.cost(initial_points).view(-1).detach()
            else:
                self.c = self.DEFAULT_COST
        if callable(self.objective_cost):
            y, c = self.objective_cost(initial_points)
            self.y = y.detach()
            self.c = c.detach()
        if self.noisy_observation:
            noise = torch.randn_like(self.y) * self.noise_level
            self.y = (self.y + noise).detach()

        if self.dim == 1:
            for point in initial_points:
                self.mask[int(point.detach()*(self.granularity - 1))] = False
        self.update_best()


    def update_best(self):
        self.best_f = self.y.max().item() if self.maximize else self.y.min().item()
        best_idx = self.y.argmax() if self.maximize else self.y.argmin()
        self.best_x = self.x[best_idx].detach()
        self.best_history.append(self.best_f)
    
    def iterate(self, acquisition_function_class, **acqf_kwargs):

        is_ts = False
        is_pes = False
        gaussian_likelihood = False
    
        # Determine if we need cost information for this acquisition function
        if acquisition_function_class in (LogExpectedImprovementWithCost, GittinsIndex, StableGittinsIndex):
            use_cost = True
        else:
            # For cost-unaware methods (UCB, TS, LogEI, etc.), use single-output model
            use_cost = False

        self.old_model = self.model
        # Store old single-outcome model if it exists
        if hasattr(self, 'single_outcome_model'):
            self.old_single_outcome_model = self.single_outcome_model
        
        # Always fit a multi-output model when self.unknown_cost=True (needed for stopping criteria)
        # Fit a single-output model when self.unknown_cost=False
        if self.unknown_cost:
            # Fit multi-output model (needed for cost-aware stopping criteria like StablePBGI, LogEIPC)
            model = fit_gp_model(
                X=self.x.detach(), 
                objective_X=self.y.detach(), 
                cost_X=self.c.detach(), 
                unknown_cost=True,
                kernel=self.kernel,
                gaussian_likelihood=gaussian_likelihood,
                output_standardize=self.output_standardize,
            )
            self.model = model
            if (self.old_model is None):
                self.old_model = model
            
            # Also fit a single-output model for cost-unaware acquisition functions and stopping criteria
            self.single_outcome_model = fit_gp_model(
                X=self.x.detach(), 
                objective_X=self.y.detach(), 
                cost_X=None, 
                unknown_cost=False,
                kernel=self.kernel,
                gaussian_likelihood=gaussian_likelihood,
                output_standardize=self.output_standardize,
            )
            if not hasattr(self, 'old_single_outcome_model') or self.old_single_outcome_model is None:
                self.old_single_outcome_model = self.single_outcome_model
        else:
            # When unknown_cost=False, fit single-output model
            model = fit_gp_model(
                X=self.x.detach(), 
                objective_X=self.y.detach(), 
                cost_X=None, 
                unknown_cost=False,
                kernel=self.kernel,
                gaussian_likelihood=gaussian_likelihood,
                output_standardize=self.output_standardize,
            )
            self.model = model
            if (self.old_model is None):
                self.old_model = model
            # When unknown_cost=False, the main model is already single-output
            self.single_outcome_model = model
            if not hasattr(self, 'old_single_outcome_model') or self.old_single_outcome_model is None:
                if self.old_model is not None:
                    self.old_single_outcome_model = self.old_model
                else:
                    self.old_single_outcome_model = self.single_outcome_model

        # Use appropriate model for the acquisition function
        if use_cost:
            acqf_model = self.model  # Use multi-output model for cost-aware methods
        else:
            acqf_model = self.single_outcome_model  # Use single-output model for cost-unaware methods
        
        acqf_args = {'model': acqf_model}
        
        if acquisition_function_class == "ThompsonSampling":
        
            # Draw sample path(s)
            paths = draw_matheron_paths(acqf_model, sample_shape=torch.Size([1]))
            
            # Optimize
            optimal_input, optimal_output = optimize_posterior_samples(paths=paths, bounds=self.bounds, maximize=self.maximize)

            is_ts = True
            new_point = optimal_input.detach()
            self.current_acq = optimal_output.item()
        
        if acquisition_function_class in (GittinsIndex, StableGittinsIndex):
            acqf_args['maximize'] = self.maximize
            
            if acqf_kwargs.get('step_divide') == True:
                if self.need_lmbda_update:
                    self.current_lmbda = self.current_lmbda / acqf_kwargs.get('alpha')
                    self.need_lmbda_update = False
                acqf_args['lmbda'] = self.current_lmbda
                self.lmbda_history.append(self.current_lmbda)

            else: 
                acqf_args['lmbda'] = acqf_kwargs['lmbda']
                self.lmbda_history.append(acqf_kwargs['lmbda'])

            acqf_args['cost'] = self.cost
            acqf_args['unknown_cost'] = self.unknown_cost

        
        elif acquisition_function_class == UpperConfidenceBound:
            if acqf_kwargs.get('heuristic') == True:
                print("Using heuristic for beta")
                acqf_args['beta'] = 2*np.log(self.dim*((self.cumulative_cost+1)**2)*(math.pi**2)/(6*0.1))/5
            else:
                acqf_args['beta'] = acqf_kwargs['beta']
            acqf_args['maximize'] = self.maximize
        
        
        elif acquisition_function_class == LogExpectedImprovement:
            acqf_args['best_f'] = self.best_f
            acqf_args['maximize'] = self.maximize

        
        elif acquisition_function_class == LogExpectedImprovementWithCost:
            acqf_args['best_f'] = self.best_f
            acqf_args['maximize'] = self.maximize
            acqf_args['cost'] = self.cost
            acqf_args['unknown_cost'] = self.unknown_cost
            if acqf_kwargs.get('cost_cooling') == True:
                cost_exponent = (self.budget - self.cumulative_cost) / self.budget
                cost_exponent = max(cost_exponent, 0)  # Ensure cost_exponent is non-negative
                acqf_args['cost_exponent'] = cost_exponent

        else:
            acqf_args.update(**acqf_kwargs)


        if is_ts == False and is_pes == False:
            acq_function = acquisition_function_class(**acqf_args)

            # use grid search if dimension is 1, otherwise use optimization
            if self.dim == 1:
                print("Using grid search for 1D optimization")
                candidates = torch.linspace(0, 1, self.granularity).unsqueeze(1).unsqueeze(1)
                candidates_acq_vals = acq_function.forward(candidates[self.mask])
                candidates =  candidates.detach()
                # change to reflect minimization objective
                if (self.maximize):
                    best_idx = torch.argmax(candidates_acq_vals.view(-1), dim=0)
                else:
                    best_idx = torch.argmin(candidates_acq_vals.view(-1), dim=0)
                best_point = candidates[best_idx]
                best_acq_val = candidates_acq_vals[best_idx].item()

            else:
                print("Using optimization for multi-dimensional optimization")
                candidate, candidate_acq_val = optimize_acqf(
                    acq_function=acq_function,
                    bounds=torch.stack([torch.zeros(self.dim), torch.ones(self.dim)]),
                    q=1,
                    num_restarts=10*self.dim,
                    raw_samples=1024*self.dim,
                    gen_candidates=gen_candidates_torch
                )
                best_point = candidate.detach()
                best_acq_val = candidate_acq_val.item() 

            new_point = best_point
            self.current_acq = best_acq_val


        if self.unknown_cost:
            new_value, new_cost = self.objective_cost(new_point.detach())
            new_value = new_value.detach()
            new_cost = new_cost.detach()
        else: 
            new_value = self.objective(new_point.detach()).detach()

        if self.noisy_observation:
            noise = torch.randn_like(new_value) * self.noise_level
            new_value += noise

        self.x = torch.cat((self.x.detach(), new_point.detach())).detach()
        self.y = torch.cat((self.y.detach(), new_value.detach())).detach()
        
        # Record statistics about different stopping rules
        # Only compute PRB if include_prb is True (defaults to False if not set)
        if getattr(self, 'include_prb', False):
            self.log_time(self.update_stopping_criteria, "PRB", skip_prb=False)
        else:
            # Initialize PRB key with NaN if not computing it
            key = 'PRB_0.1'
            self.if_not_exist_create_key(key)
            if len(self.stopping_history[key]) == 0:
                self.stopping_history[key].append(np.nan)
            else:
                self.stopping_history[key].append(self.stopping_history[key][-1])
        self.log_time(self.update_stopping_criteria, "StablePBGI", lmbda=0.001)
        self.log_time(self.update_stopping_criteria, "StablePBGI", lmbda=0.0001)
        self.log_time(self.update_stopping_criteria, "StablePBGI", lmbda=0.00001)
        self.log_time(self.update_stopping_criteria, "LogEIPC")
        self.log_time(self.update_stopping_criteria, "UCB-LCB")
        self.log_time(self.update_stopping_criteria, "Expected_Min_Regret_Gap")


        self.update_best()
        self.update_cost(new_point)
        print("New point:", new_point.detach())
        if (self.dim == 1):
            self.mask[int(new_point.detach()*(self.granularity - 1))] = False
        self.iteration += 1

        self.acq_history.append(self.current_acq)

        # Check if lmbda needs to be updated in the next iteration
        if acquisition_function_class in (GittinsIndex, StableGittinsIndex) and (acqf_kwargs.get('step_EIpu') == True or acqf_kwargs.get('step_divide') == True):
            if (self.maximize and self.current_acq < self.best_f) or (not self.maximize and -self.current_acq > self.best_f):
                self.need_lmbda_update = True

    def if_not_exist_create_key(self, key):
        if key not in self.stopping_history:
            self.stopping_history[key] = [np.nan]  # initialize if missing

    def log_time(self, func, *args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f"{func.__name__}({args[0] if args else ''}) took {elapsed:.4f} s")

        # --- record the timing in `stopping_history` ---------------------------
        if func.__name__ == "update_stopping_criteria":
            crit = args[0]                                    # e.g. "PRB", "StablePBGI"
            if crit == "StablePBGI":
                lmbda_val = kwargs.get('lmbda', 0.01)
                # Format lmbda consistently: use decimal notation to match access pattern
                crit_key = f"StablePBGI({lmbda_val:.10f}".rstrip('0').rstrip('.') + ')'
            elif crit == "PRB":                               # keep in sync with ε = 0.1
                crit_key = "PRB_0.1"
            else:
                crit_key = crit

            time_key = f"{crit_key}_time"
            self.if_not_exist_create_key(time_key) 
            self.stopping_history[time_key].append(elapsed)
        # -----------------------------------------------------------------------
        return result


    def update_stopping_criteria(self, stopping_criteria, lmbda=0.01, skip_prb=True):
        '''
        This function implements the following stopping rules: 
                                        'PBGI',
                                        'LogEIPC',
                                        'regret upper bound',
                                        'exp min regret gap',
                                        'PRB'
        '''
        # Currently only works for dim=1
        # Initialization for probabilistic regret bound (PRB) stopping rule
        epsilon = 0.1
        candidates = torch.linspace(0, 1, self.granularity).unsqueeze(1)

        if (stopping_criteria == "PRB"):
            # Probabilistic regret bound
            key = f'PRB_{epsilon}'
            self.if_not_exist_create_key(key)
            paths = draw_matheron_paths(self.model, sample_shape=torch.Size([self.num_samples]))
            bounds = torch.stack([torch.zeros(self.dim), torch.ones(self.dim)])
            
            # When to not skip PRB calculation
            # 1. If the iteration is less than 50 and the iteration is a multiple of 5
            # 2. If the iteration is a multiple of 10
            if (self.dim > 1 and skip_prb==True):
                skip_PRB = not ((self.iteration < 50 and self.iteration % 5 == 0) or self.iteration % 10 == 0)
                if skip_PRB:
                    print("Skipping PRB calculation")
                    self.if_not_exist_create_key(key)
                    self.stopping_history[key].append(self.stopping_history[key][-1])
                    return
        
            maximize_factor = 1 if self.maximize else -1
            if (self.dim == 1):
                regrets = (maximize_factor*paths(candidates)).max(dim=1).values - maximize_factor*paths(self.best_x.detach().unsqueeze(0)).squeeze(-1)
            else:
                # 1. build a QMC sampler that will internally draw your fantasy paths
                _, optimum_values = optimize_posterior_samples(paths=paths, 
                                                               bounds=bounds, 
                                                               raw_samples=200*self.dim, 
                                                               num_restarts=10*self.dim,
                                                               maximize=self.maximize)
                # print("optimum_values:", optimum_values)
                regrets = maximize_factor*optimum_values.squeeze(-1) - maximize_factor*paths(self.best_x.detach().unsqueeze(0)).squeeze(-1)
                # print("regrets:", regrets)
            
            prb_estimate = (regrets <= epsilon).float().mean().item()
            self.stopping_history[key].append(prb_estimate)
            
                # print("Probabilistic regret bound")
                # print(f'Epsilon: {epsilon}, regrets: {prb_estimate}, num_samples: {self.num_samples}')
            self.num_samples = min(math.ceil(self.num_samples * 1.5), 1000)
 
        elif (stopping_criteria == "StablePBGI"):
            # 3. Stable PBGI
            # Format lmbda consistently: use decimal notation to match access pattern
            # Remove trailing zeros and use fixed format for consistency
            key = f'StablePBGI({lmbda:.10f}'.rstrip('0').rstrip('.') + ')'
            self.if_not_exist_create_key(key) 
            StablePBGI = StableGittinsIndex(model=self.model, maximize=self.maximize, lmbda=lmbda, cost=self.cost, unknown_cost=self.unknown_cost)
            maximize_factor = 1 if self.maximize else -1
            if (self.dim == 1): 
                StablePBGI_acq = StablePBGI.forward(candidates.unsqueeze(1))
                new_config_acq = maximize_factor*torch.max(maximize_factor*StablePBGI_acq[self.mask]) 
                
            else:
                NegStablePBGI = lambda x: -StablePBGI(x)
                SignedStablePBGI = StablePBGI if self.maximize else NegStablePBGI
                # Optimize the acquisition function
                candidates, StablePBGI_acq = optimize_acqf(
                    acq_function=SignedStablePBGI,
                    bounds=self.bounds,
                    q=1,
                    num_restarts=10*self.dim,
                    raw_samples=1024*self.dim,
                    gen_candidates=gen_candidates_torch
                )
    
                new_config_acq = maximize_factor*StablePBGI_acq 

            self.stopping_history[key].append(new_config_acq.item())
            # print("StablePBGI")
            # print(f'Lambda: {lmbda}, acquisition: {new_config_acq.item()}') 

        elif (stopping_criteria == "LogEIPC"):
            key = 'LogEIPC'
            self.if_not_exist_create_key(key)
            LogEIPC = LogExpectedImprovementWithCost(model=self.model, best_f=self.best_f, maximize=self.maximize, cost=self.cost, unknown_cost=self.unknown_cost)
            # maximization or minimization objective has no effect on LogEIPC
            if (self.dim == 1):
                LogEIPC_acq = LogEIPC.forward(candidates.unsqueeze(1)) 
                new_config_acq = torch.max(LogEIPC_acq[self.mask]).detach()
            else:
                candidates, LogEIPC_acq = optimize_acqf(
                        acq_function=LogEIPC,
                        bounds=self.bounds,
                        q=1,
                        num_restarts=10*self.dim,
                        raw_samples=1024*self.dim,
                        gen_candidates=gen_candidates_torch
                    )
                new_config_acq = LogEIPC_acq # torch.max(LogEIPC_acq)
        
            self.stopping_history[key].append(new_config_acq.item())

        elif (stopping_criteria == "UCB-LCB"):  
            
            key = f'UCB-LCB'
            self.if_not_exist_create_key(key) 

            UCB = UpperConfidenceBound(model=self.single_outcome_model, maximize=self.maximize, beta=2 * np.log(self.dim * ((self.iteration + 1) ** 2) * (math.pi ** 2) / (6 * 0.1)) / 5)
            LCB = LowerConfidenceBound(model=self.single_outcome_model, maximize=self.maximize, beta=2 * np.log(self.dim * ((self.iteration + 1) ** 2) * (math.pi ** 2) / (6 * 0.1)) / 5)
            if (self.maximize):
                optimistic_CB = UCB; pessimistic_CB = LCB; maximize_factor = 1
            else:
                optimistic_CB = LCB; pessimistic_CB = UCB; maximize_factor = -1
            # print(f"beta: {beta}")
            if (self.dim == 1):
                optimistic_acq = optimistic_CB.forward(candidates.unsqueeze(1))
                pessimistic_acq = pessimistic_CB.forward(self.x.detach().unsqueeze(1))
                kappa = torch.max(maximize_factor*optimistic_acq) - torch.max(maximize_factor*pessimistic_acq)
            else:
                candidates, optimistic_acq = optimize_acqf(
                        acq_function=optimistic_CB,
                        bounds=self.bounds,
                        q=1,
                        num_restarts=10*self.dim,
                        raw_samples=1024*self.dim,
                        gen_candidates=gen_candidates_torch
                    ) 

                # print(candidates.shape)
                pessimistic_acq = pessimistic_CB.forward(self.x.detach().unsqueeze(1))
                # print(f"UCB {UCB_acq} and LCB {torch.max(LCB_acq)}")
                
                kappa = maximize_factor*optimistic_acq - torch.max(maximize_factor*pessimistic_acq)
                # print(f"kappa: {kappa}")
                # kappa = torch.max(UCB_acq) - torch.max(LCB_acq)
            self.stopping_history[key].append(kappa.item())
        
        elif (stopping_criteria == "Expected_Min_Regret_Gap"): 
            key = f'Expected-Min-Regret-Gap'
            self.if_not_exist_create_key(key) 
            
            UCB = UpperConfidenceBound(model=self.single_outcome_model, maximize=self.maximize, beta=2 * np.log(self.dim * ((self.iteration + 1) ** 2) * (math.pi ** 2) / (6 * 0.1)) / 5)
            LCB = LowerConfidenceBound(model=self.single_outcome_model, maximize=self.maximize, beta=2 * np.log(self.dim * ((self.iteration + 1) ** 2) * (math.pi ** 2) / (6 * 0.1)) / 5)
            if (self.maximize):
                optimistic_CB = UCB; pessimistic_CB = LCB; maximize_factor = 1
            else:
                optimistic_CB = LCB; pessimistic_CB = UCB; maximize_factor = -1
            # print(f"beta: {beta}")
            if (self.dim == 1):
                optimistic_acq = optimistic_CB.forward(candidates.unsqueeze(1))
                pessimistic_acq = pessimistic_CB.forward(self.x.detach().unsqueeze(1))
                kappa = torch.max(maximize_factor*optimistic_acq) - torch.max(maximize_factor*pessimistic_acq)
            else:
                candidates, optimistic_acq = optimize_acqf(
                        acq_function=optimistic_CB,
                        bounds=self.bounds,
                        q=1,
                        num_restarts=10*self.dim,
                        raw_samples=1024*self.dim,
                        gen_candidates=gen_candidates_torch
                    ) 

                # print(candidates.shape)
                pessimistic_acq = pessimistic_CB.forward(self.x.detach().unsqueeze(1))
                # print(f"UCB {UCB_acq} and LCB {torch.max(LCB_acq)}")
                
                kappa = maximize_factor*optimistic_acq - torch.max(maximize_factor*pessimistic_acq)
    
            
            # 7.1. Get the posterior mean for old and new GPs at the new and old best points.
            # new_config_x and old_config_x should be the configurations corresponding to the current
            # and previous best indices, respectively.
            x_pair = torch.stack([self.x[-1].detach(), self.x[-2].detach()]).detach()

            # 7.2. Get posterior mean and covariance from the new model.
            new_posterior = self.single_outcome_model.posterior(x_pair)
            new_mean = new_posterior.mean         # Shape: [2]
            new_covar = new_posterior.mvn.covariance_matrix     # Shape: [2, 2]

            # 7.3. Get posterior mean and covariance from the old model.
            old_posterior = self.old_single_outcome_model.posterior(x_pair)
            old_mean = old_posterior.mean           # Shape: [2]
            old_covar = old_posterior.mvn.covariance_matrix       # Shape: [2, 2]

            # 7.4. Compute delta_mu (the absolute change in best posterior mean)
            # Here, we assume that new_config_x corresponds to the current best (new point)
            # and old_config_x corresponds to the previous best.
            delta_mu = abs(old_mean[1].item() - new_mean[0].item())
            
            # 7.6. Compute KL divergence between old and new posteriors at the new point.
            old_var = old_covar[0, 0].clamp(min=1e-12)
            new_var = new_covar[0, 0].clamp(min=1e-12)
            old_mu_val = old_mean[0]
            new_mu_val = new_mean[0]
            kl = 0.5 * (torch.log(new_var / old_var) +
                        (old_var + (old_mu_val - new_mu_val).pow(2)) / new_var - 1).item()

            # 7.7. Compute ei_diff, the expected-improvement gap difference.
            # If new_config_x and old_config_x are (approximately) equal, we set ei_diff to zero.
            if not torch.allclose(x_pair[0], x_pair[1], atol=1e-6):
                # We use the new model's posterior for these two points.
                # new_mean and new_covar already contain the predictions.
                # Compute the difference in means:
                g = (new_mean[0] - new_mean[1]).item()
                # Compute the effective variance difference
                diff_var = (new_covar[0, 0] - 2 * new_covar[0, 1] + new_covar[1, 1]).item()
                if diff_var < 0:
                    beta_val = 0.0
                    pdf_val = np.sqrt(1.0 / (2 * np.pi))
                    cdf_val = 1.0
                else:
                    beta_val = np.sqrt(diff_var)
                    u = g / beta_val if beta_val > 0 else 0.0
                    pdf_val = norm.pdf(u)
                    cdf_val = norm.cdf(u)
                ei_diff = beta_val * pdf_val + g * cdf_val
            else:
                ei_diff = 0.0

            # 7.8. Final expression for ΔR̃_t (the expected minimal regret gap).
            exp_min_regret_gap = delta_mu + ei_diff + kappa.item() * np.sqrt(0.5 * kl)
            self.stopping_history[key].append(exp_min_regret_gap.item())
            
    def update_cost(self, new_point):
        if callable(self.cost):
            # If self.cost is a function, call it and update cumulative cost
            new_cost = self.cost(new_point).view(-1).detach()
            self.c = torch.cat((self.c.detach(), new_cost)).detach()
            self.cumulative_cost += new_cost.item()
        elif callable(self.objective_cost):
            new_value, new_cost = self.objective_cost(new_point)
            new_cost = new_cost.detach()
            self.c = torch.cat((self.c.detach(), new_cost)).detach()
            self.cumulative_cost += new_cost.sum().item()
        else:
            # If self.cost is not a function, just increment cumulative cost by self.cost
            self.cumulative_cost += self.cost.item()

        self.cost_history.append(self.cumulative_cost)


    def print_iteration_info(self, iteration):
        print(f"Iteration {iteration}, New point: {self.x[-1].squeeze().detach().numpy()}, New value: {self.y[-1].detach().numpy()}")
        print("Best observed value:", self.best_f)
        print("Current acquisition value:", self.current_acq)
        print("Cumulative cost:", self.cumulative_cost)
        if hasattr(self, 'need_lmbda_update'):
            print("Gittins lmbda:", self.lmbda_history[-1])
        print("Running time:", self.runtime)
        print()

    def run(self, num_iterations, acquisition_function_class, **acqf_kwargs):
        self.budget = num_iterations
        if acquisition_function_class in (GittinsIndex, StableGittinsIndex):
            self.lmbda_history = []
            if acqf_kwargs.get('step_divide') == True:
                self.current_lmbda = acqf_kwargs['init_lmbda']
                self.need_lmbda_update = False
                                
        for i in range(num_iterations):
            start = time.process_time()
            self.iterate(acquisition_function_class, **acqf_kwargs)
            end = time.process_time()
            runtime = end - start
            self.runtime = runtime
            self.runtime_history.append(runtime)
            self.print_iteration_info(i)

    def get_best_value(self):
        return self.best_f


    def get_best_history(self):
        return self.best_history


    def get_cumulative_cost(self):
        return self.cumulative_cost


    def get_cost_history(self):
        return self.cost_history


    def get_regret_history(self, global_optimum):
        """
        Compute the regret history.

        Parameters:
        - global_optimum (float): The global optimum value of the objective function.

        Returns:
        - list: The regret history.
        """
        return [global_optimum - f if self.maximize else f - global_optimum for f in self.best_history]

    def get_lmbda_history(self):
        return self.lmbda_history

    def get_acq_history(self):
        return self.acq_history

    def get_runtime_history(self):
        return self.runtime_history
    
    def get_stopping_history(self):
        return self.stopping_history