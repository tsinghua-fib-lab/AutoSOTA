"""
Temporal Score Rescaling (TSR) Scheduler Classes
Supplementary material for ICLR 2026 submission.

This module implements TSR schedulers based on HuggingFace Diffusers.
TSR_DDIMScheduler and TSR_FlowEulerScheduler extend standard schedulers with
temporal score rescaling for improved sampling performance.
"""

import torch
from typing import List, Optional, Tuple, Union
from diffusers.schedulers import DDIMScheduler, FlowMatchEulerDiscreteScheduler
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteSchedulerOutput

class TSR_DDIMScheduler(DDIMScheduler):
    def __init__(self, *args, psr_sigma=1.0, k=1.0, orig_scheduler,
                 use_adaptive_k=False, k_power=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.psr_sigma = psr_sigma
        self.k = k
        self.use_adaptive_k = use_adaptive_k
        self.k_power = k_power
        self.copy_set(orig_scheduler)
    
    def copy_set(self, ddim_scheduler):
        # self.config = ddim_scheduler.config
        self.betas = ddim_scheduler.betas
        self.alphas = ddim_scheduler.alphas
        self.alphas_cumprod = ddim_scheduler.alphas_cumprod
        self.final_alpha_cumprod = ddim_scheduler.final_alpha_cumprod
        self.init_noise_sigma = ddim_scheduler.init_noise_sigma
        self.num_inference_steps = ddim_scheduler.num_inference_steps
        self.timesteps = ddim_scheduler.timesteps


    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
    ) -> Union[DDIMSchedulerOutput, Tuple]:

        psr_ratio = self.get_psr_ratio(timestep)
        alpha_prod_t = self.alphas_cumprod[timestep]
        beta_prod_t = 1 - alpha_prod_t

        # Convert model_output to epsilon prediction
        if self.config.prediction_type == "epsilon":
            pred_epsilon = model_output
        elif self.config.prediction_type == "v_prediction":
            pred_epsilon = (alpha_prod_t ** 0.5) * model_output + (beta_prod_t ** 0.5) * sample
        elif self.config.prediction_type == "sample":
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * model_output) / (beta_prod_t ** (0.5) + 1e-12)
        else:
            raise ValueError(f"Unsupported prediction_type: {self.config.prediction_type}")

        # Apply TSR rescaling to epsilon
        pred_epsilon = pred_epsilon * psr_ratio

        # Convert epsilon back to the original prediction type
        if self.config.prediction_type == "epsilon":
            model_output = pred_epsilon
        elif self.config.prediction_type == "v_prediction":
            model_output = (pred_epsilon - (beta_prod_t ** 0.5) * sample) / (alpha_prod_t ** (0.5) + 1e-12)
        elif self.config.prediction_type == "sample":
            model_output = (sample - beta_prod_t ** (0.5) * pred_epsilon) / (alpha_prod_t ** (0.5) + 1e-12)

        return super().step(
            model_output=model_output,
            timestep=timestep,
            sample=sample,
            eta=eta,
            use_clipped_model_output=use_clipped_model_output,
            generator=generator,
            variance_noise=variance_noise,
            return_dict=return_dict,
        )
    
    def get_psr_ratio(self, timestep, k=None):
        if k is None:
            k = self.k
        if self.use_adaptive_k:
            alpha_prod = self.alphas_cumprod[timestep]
            k = 1.0 + (k - 1.0) * (float(alpha_prod) ** self.k_power)
        alpha_prod = self.alphas_cumprod[timestep]
        psr_ratio = (alpha_prod * self.psr_sigma**2 + 1 - alpha_prod) / \
                        (alpha_prod * self.psr_sigma**2 / k + 1 - alpha_prod)
        return psr_ratio


class TSR_FlowEulerScheduler(FlowMatchEulerDiscreteScheduler):
    def __init__(self, *args, k=1.0, psr_sigma=0.1, orig_scheduler, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k
        self.psr_sigma = psr_sigma
        self.copy_set(orig_scheduler)
        self.config.prediction_type = "flow"
    
    def copy_set(self, orig_scheduler):
        self.sigmas = orig_scheduler.sigmas
        # self.num_inference_steps = orig_scheduler.num_inference_steps
        self.timesteps = orig_scheduler.timesteps

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[FlowMatchEulerDiscreteSchedulerOutput, Tuple]:


        # Scale the score
        t = (timestep / self.config.num_train_timesteps)
        ratio = self.get_psr_ratio(timestep)
        if not t == 1.0:
            model_output = (ratio * ((1-t) * model_output + sample) - sample) / (1 - t)
            

        return super().step(
            model_output=model_output,
            timestep=timestep,
            sample=sample,
            s_churn=s_churn,
            s_tmin=s_tmin,
            s_tmax=s_tmax,
            s_noise=s_noise,
            generator=generator,
            return_dict=return_dict,
        )
    
    def get_psr_ratio(self, timestep, k=None):
        k = self.k if k is None else k
        t = (timestep / self.config.num_train_timesteps)
        if t == 0.0:
            ratio = self.k
        else:
            snr_t = (1 - t)**2 / t**2
            ratio = (snr_t * self.psr_sigma**2 + 1) / (snr_t * self.psr_sigma**2 / k + 1)
        return ratio
