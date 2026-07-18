import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
import torch
from typing import Optional, Union, List
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers import DDIMPipeline, AutoencoderKL


class CondDDIMPipeline(DDIMPipeline):
    def __init__(self, net, scheduler, vae: Optional[AutoencoderKL] = None):
        super().__init__(net, scheduler)
        self.vae = vae
        self.register_modules(vae=vae)
        self.net = net

    def get_sigma_t(self, t, device, dtype):
        alpha_bar_t = self.scheduler.alphas_cumprod[t.long()].to(device=device, dtype=dtype)
        return torch.sqrt(torch.clamp(1 - alpha_bar_t, min=1e-12))

    def get_sigma_step(self, timesteps, t_idx, device, dtype):
        t = timesteps[t_idx]
        sigma_t = self.get_sigma_t(t, device=device, dtype=dtype)

        if t_idx < len(timesteps) - 1:
            t_next = timesteps[t_idx + 1]
            sigma_next = self.get_sigma_t(t_next, device=device, dtype=dtype)
            return torch.clamp(sigma_t - sigma_next, min=0.0)

        return torch.zeros((), device=device, dtype=dtype)

    def calculate_ode_increment(self, neg_div, vel, combined_vel, sigma_t, d_sigma):
        # For the ODE case we cannot use the Itô estimator. Instead, we use
        # the continuity-equation update evaluated along the composed path:
        # d log q = [-div v_q + <v* - v_q, score_q>] d sigma
        score = vel / sigma_t
        cross_term = ((combined_vel - vel) * score).sum((1, 2, 3))
        return (neg_div + cross_term) * d_sigma
    
    # Initial log-density at t=T (pure noise)
    # log p(x) = -0.5 * (d * log(2pi) + ||x||^2)
    def get_initial_ll(self, x_T):
        """
        Calculates the log-density of the initial noise under a standard Gaussian prior.
        x_T: The initial noise tensor (batch_size, C, H, W)
        """
        batch_size = x_T.shape[0]
        
        # 1. Calculate the total number of dimensions (d)
        # For a 3x64x64 image, d = 12288
        dims = x_T.shape[1:]
        d = torch.prod(torch.tensor(dims)).to(x_T.device)
        
        # 2. Calculate the squared norm of the noise ||x||^2 
        # Flatten the image to (batch_size, d) then sum the squares
        norm_sq = torch.norm(x_T.view(batch_size, -1), dim=1)**2
        
        # 3. Compute log p(x) = -0.5 * (d * log(2pi) + ||x||^2)
        log_2pi = torch.log(torch.tensor(2 * torch.pi)).to(x_T.device)
        initial_ll = -0.5 * (d * log_2pi + norm_sq)
        
        return initial_ll

    def update_composite_ll_state(self, expr, ll_state):
        if expr is None:
            return None

        expr_key = str(expr)

        if hasattr(expr, "condition") and expr.condition is not None:
            return ll_state[expr_key]

        if hasattr(expr, "left") and hasattr(expr, "right"):
            ll_left = self.update_composite_ll_state(expr.left, ll_state)
            ll_right = self.update_composite_ll_state(expr.right, ll_state)

            class_name = expr.__class__.__name__
            if "Or" in class_name:
                ll_state[expr_key] = torch.logsumexp(torch.stack([ll_left, ll_right]), dim=0)
            elif "And" in class_name:
                # SuperDiff AND enforces equal-density trajectories rather than a
                # mixture density. For nested expressions we propagate the
                # common likelihood level via the symmetric mean.
                ll_state[expr_key] = 0.5 * (ll_left + ll_right)
            return ll_state[expr_key]

        return ll_state.get(expr_key)

    

    """
    A PyTorch Lightning module that implements the DDIM pipeline for image data. Taken from the original DDIM implementation."""
    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        use_clipped_model_output: Optional[bool] = None,
        return_dict: bool = True,
        query: Optional[torch.Tensor] = None,
        guidance_dict: Optional[dict] = None,
        null_token: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        noise_percentage: Optional[float] = None,
        **kwargs,
    ):
        """
        Args:
            noise_percentage: Optional float between 0 and 1. If provided, noise will be added to the input image
                            at the timestep corresponding to this percentage of total inference steps.
                            For example, 0.2 means add noise at 20% of the total timesteps.
                            If None, starts from pure random noise (default behavior).
        """
        if isinstance(self.net.config.sample_size, int):
            image_shape = (
                batch_size,
                self.net.config.in_channels,
                self.net.config.sample_size,
                self.net.config.sample_size,
            )
        elif self.net.config.in_channels ==0:
            image_shape = (batch_size, *self.net.config.sample_size)
        else:
            image_shape = (batch_size, self.net.config.in_channels, *self.net.config.sample_size)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if image is None:
            image = randn_tensor(image_shape, generator=generator, device=null_token.device)
            image = image * self.scheduler.init_noise_sigma
        
        self.scheduler.set_timesteps(num_inference_steps)
        
        # for debugging
        if noise_percentage is not None:
            noise_step_idx = int(len(self.scheduler.timesteps) * noise_percentage)
            timesteps = torch.ones(image.size(0), device=null_token.device, dtype=torch.long) * self.scheduler.timesteps[noise_step_idx]
            noise = torch.randn_like(image)
            noisy_image = self.scheduler.add_noise(image, noise, timesteps)
            image = noisy_image
            self.scheduler.timesteps = self.scheduler.timesteps[noise_step_idx + 1:]
        
        image = image.to(device=null_token.device)
    
        # Initialize ALL possible conditions in the query tree
        initial_ll = self.get_initial_ll(image)

        def get_strings_query(expr, collector_list=None):
            if collector_list is None:
                collector_list = []
            collector_list.append(str(expr))

            if hasattr(expr, 'expression') and expr.expression is not None:
                get_strings_query(expr.expression, collector_list)
                
            if hasattr(expr, 'left') and hasattr(expr, 'right'):
                get_strings_query(expr.left, collector_list)
                get_strings_query(expr.right, collector_list)
                
            return collector_list

        all_query_strings = get_strings_query(query)

        ll_state = {k: initial_ll.clone() for k in all_query_strings}

        # Extract guidance annealing alpha from guidance dict (Idea-05)
        anneal_alpha = 0.0
        if guidance_dict is not None and '_anneal_alpha' in guidance_dict:
            anneal_alpha = float(guidance_dict['_anneal_alpha'])

        for t in self.progress_bar(self.scheduler.timesteps):
            t = t.to(device=image.device)
            xt_current = image.clone()
            # Time-dependent guidance annealing (Idea-05)
            if anneal_alpha > 0 and guidance_dict is not None:
                T = self.scheduler.config.num_train_timesteps
                t_normalized = t.float() / T
                anneal_scale = 1.0 + anneal_alpha * (1.0 - t_normalized)
                scaled_guidance = dict(guidance_dict)
                scaled_guidance['_anneal_alpha'] = anneal_alpha
                for k in ['atom', 'not']:
                    if k in scaled_guidance and isinstance(scaled_guidance[k], (int, float)):
                        scaled_guidance[k] = scaled_guidance[k] * anneal_scale.item()
                for group in ['logdiff', 'constant']:
                    if group in scaled_guidance and isinstance(scaled_guidance[group], dict):
                        scaled_guidance[group] = {sk: sv * anneal_scale.item() for sk, sv in scaled_guidance[group].items()}
                current_guidance = scaled_guidance
            else:
                current_guidance = guidance_dict

            model_output, score_cache = self.net(
                xt_current, 
                t, 
                query, 
                guidance_dict=current_guidance, 
                null_token=null_token, 
                ll_state=ll_state, 
                scheduler=self.scheduler
            )
            if score_cache is not None:
                t_idx = (self.scheduler.timesteps.to(t.device) == t).nonzero().item()
                d_sigma = self.get_sigma_step(
                    self.scheduler.timesteps,
                    t_idx,
                    device=image.device,
                    dtype=xt_current.dtype,
                )
                combined_vel = score_cache.get(str(query), {}).get("v", -model_output)
                step_results = self.scheduler.step(
                    model_output, t, xt_current, eta=eta,
                    use_clipped_model_output=use_clipped_model_output, generator=generator
                )
                image = step_results.prev_sample
                for key, value_dict in score_cache.items():
                    if not isinstance(value_dict, dict):
                        continue
                    if "div" not in value_dict or "v" not in value_dict:
                        continue

                    sigma_t = self.get_sigma_t(t, device=xt_current.device, dtype=xt_current.dtype)
                    dll = self.calculate_ode_increment(
                        neg_div=value_dict["div"],
                        vel=value_dict["v"],
                        combined_vel=combined_vel,
                        sigma_t=sigma_t,
                        d_sigma=d_sigma,
                    )

                    if key not in ll_state:
                        ll_state[key] = self.get_initial_ll(xt_current)

                    ll_state[key] += dll

                self.update_composite_ll_state(query, ll_state)
            else:
                step_results = self.scheduler.step(
                    model_output, t, xt_current, eta=eta, 
                    use_clipped_model_output=use_clipped_model_output, generator=generator
                )
                image = step_results.prev_sample

        if self.vae is not None:
            image = self.vae.decode(image/self.vae.config.scaling_factor)[0]
            image = (image.clamp(-1, 1) + 1) / 2 
        if not return_dict:
            return (image,)
        return ImagePipelineOutput(images=image)
    
