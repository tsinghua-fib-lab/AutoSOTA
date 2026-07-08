import torch
from tqdm import tqdm
from .base import Algo
from utils.scheduler import Scheduler
import numpy as np
import time
import pickle
import os
# -----------------------------------------------------------------------------------------------
# Paper: Diffusion Posterior Sampling for General Noisy Inverse Problems
# Official implementation: https://github.com/DPS2022/diffusion-posterior-sampling
# -----------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt


class DPS(Algo):
    
    '''
    DPS algorithm implemented in EDM framework.
    '''
    
    def __init__(self, 
                 net,
                 forward_op,
                 diffusion_scheduler_config,
                 guidance_scale,
                 sde=True):
        super(DPS, self).__init__(net, forward_op)
        self.scale = guidance_scale
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.scheduler = Scheduler(**diffusion_scheduler_config)
        self.sde = sde
        self.REINF_K = 128
        
    def inference(self, observation, num_samples=1, **kwargs):
        self.gen_idx=-1
        if 'gen_idx' in kwargs:
            self.gen_idx = kwargs['gen_idx']
        device = self.forward_op.device
        if num_samples > 1:
            observation = observation.repeat(num_samples, 1, 1, 1)
        x_initial = torch.randn(num_samples, self.net.img_channels, self.net.img_resolution, self.net.img_resolution, device=device) * self.scheduler.sigma_max
        x_next = x_initial

        pbar = range(self.scheduler.num_steps)
        s1=time.time()
        
        all_min_phi = []
        all_mean_return_phi = []
        all_max_phi = []

        for i in pbar:
            s2=time.time()
            num_reinf_samples = 64  # round(self.REINF_K*self.scheduler.num_steps/(self.scheduler.num_steps-i))
            x_cur = x_next.detach()

            sigma, factor, scaling_factor = self.scheduler.sigma_steps[i], self.scheduler.factor_steps[i], self.scheduler.scaling_factor[i]
            
            denoised = self.net(x_cur / self.scheduler.scaling_steps[i], torch.as_tensor(sigma).to(x_cur.device))

            grad, min_phi, mean_return_phi, max_phi = self.batch_reinforce_grad(
                num_reinf_samples,
                64,
                i,
                x_cur,
                observation
            )
            print(f"Min Phi: {min_phi}, Mean Return Phi: {mean_return_phi}, Max Phi: {max_phi}")
            all_min_phi.append(min_phi)
            all_mean_return_phi.append(mean_return_phi)
            all_max_phi.append(max_phi)

            # alpha_bar = 1 / (1 + sigma^2)
            sigma_t = torch.as_tensor(sigma, device=x_cur.device, dtype=x_cur.dtype)
            alpha_bar = 1.0 / (1.0 + sigma_t**2)
            a_t = torch.sqrt(alpha_bar)
        
            b2_over_a = (1.0 - alpha_bar) / a_t
            
            denoised = denoised + (self.scale * b2_over_a) * grad
            

            score = (denoised - x_cur / self.scheduler.scaling_steps[i]) / sigma ** 2 / self.scheduler.scaling_steps[i]
            #pbar.set_description(f'Iteration {i + 1}/{self.scheduler.num_steps}. Data fitting loss: {best_loss**0.5}')
            
            if self.sde:
                epsilon = torch.randn_like(x_cur)
                x_next = x_cur * scaling_factor + factor * score + np.sqrt(factor) * epsilon
            else:
                x_next = x_cur * scaling_factor + factor * score * 0.5 

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,5))
        plt.plot(all_min_phi, label='Min Phi')
        plt.plot(all_mean_return_phi, label='Mean Phi')
        plt.plot(all_max_phi, label='Max Phi')
        plt.xlabel('Step')
        plt.ylabel('Phi Values')
        plt.yscale('symlog')
        plt.title(f'Phi Values Across Steps')
        plt.legend()
        plt.savefig(f'./step_samples/z_{self.REINF_K}_{self.scheduler.num_steps}_{self.gen_idx}-reinf_phi_values_across_steps.png')
        plt.close()

        return x_next
    
    
    
    
    #-----------------helpers
    def posterior_score(self, x_t: torch.Tensor, x0: torch.Tensor, alpha_bar_t: torch.Tensor):
        return (alpha_bar_t.sqrt() * x0 - x_t) / (1.0 - alpha_bar_t)
    
    
    def marginal_score_from_eps(self, x_t: torch.Tensor, eps_pred: torch.Tensor, alpha_bar_t: torch.Tensor):
        return -eps_pred / (1.0 - alpha_bar_t).sqrt()


    def conditional_score_from_samples(self, x_t: torch.Tensor, x0_samples: torch.Tensor, eps_pred: torch.Tensor, alpha_bar_t: torch.Tensor):
        K, B = x0_samples.shape[:2]
        x_t_exp   = x_t.unsqueeze(0).expand(K, -1, -1, -1, -1)
        alpha_exp = alpha_bar_t.unsqueeze(0).expand(K, -1, -1, -1, -1)
    
        g_fwd   = self.posterior_score(x_t_exp, x0_samples, alpha_exp)
        s_xt    = self.marginal_score_from_eps(x_t, eps_pred, alpha_bar_t)
        s_xt_exp = s_xt.unsqueeze(0).expand_as(g_fwd)
        return g_fwd - s_xt_exp
    
    
    
    #-----------------sampling
    @torch.no_grad()
    def unguided_posterior_sample(self, x_initial: torch.Tensor, step_idx: int):
        device = self.forward_op.device
        x_next = x_initial

        pbar = range(step_idx, self.scheduler.num_steps)
        
        for i in pbar:
            x_cur = x_next.detach()

            sigma, factor, scaling_factor = self.scheduler.sigma_steps[i], self.scheduler.factor_steps[i], self.scheduler.scaling_factor[i]
            denoised = self.net(x_cur / self.scheduler.scaling_steps[i], torch.as_tensor(sigma).to(x_cur.device))
            score = (denoised - x_cur / self.scheduler.scaling_steps[i]) / sigma ** 2 / self.scheduler.scaling_steps[i]
            
            if self.sde:
                epsilon = torch.randn_like(x_cur)
                x_next = x_cur * scaling_factor + factor * score + np.sqrt(factor) * epsilon
            else:
                x_next = x_cur * scaling_factor + factor * score * 0.5 
        return x_next
    
    @torch.no_grad()
    def posterior_samples(self, x_initial: torch.Tensor, step_idx: int, K: int):
        return [self.unguided_posterior_sample(x_initial, step_idx).detach() for _ in range(K)]
    
    
    
    def batch_reinforce_grad(
        self,
        num_samples,
        batch_size,
        step_idx,
        xt: torch.Tensor,
        observation: torch.Tensor,
        scale: float = 1.0,
        baseline: str = "mean",
        normalize: bool = True,
    ):
        device = xt.device
        B = xt.size(0)
        
        sigma, factor, scaling_factor = self.scheduler.sigma_steps[step_idx], self.scheduler.factor_steps[step_idx], self.scheduler.scaling_factor[step_idx]

        with torch.no_grad():
            x0_pred = self.net(xt / self.scheduler.scaling_steps[step_idx], torch.as_tensor(sigma).to(xt.device))
            eps_pred = xt - x0_pred

        alpha_bar_t = torch.as_tensor(1.0 / (1.0 + self.scheduler.sigma_steps[step_idx]**2))[None,None,None,...].cuda().repeat(B, 1, 1, 1)

        sum_phi = torch.zeros(B, device=device)
        sum_score = torch.zeros_like(xt)
        sum_phi_score = torch.zeros_like(xt)
        total_K = 0

        batch_idx = 0
        count = 0
        
        all_phi_vals = []

        while count < num_samples:
            cur_batch = min(batch_size, num_samples - count)

            cache_path = f'./step_samples/{self.REINF_K}_{self.scheduler.num_steps}_{self.gen_idx}-reinf_{step_idx}_{batch_idx}'
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    cur_samples = pickle.load(f)
            else:
                cur_samples = self.posterior_samples(xt.detach(), step_idx, cur_batch)
                os.makedirs('./step_samples/', exist_ok=True)
                with open(cache_path, 'wb') as f:
                    pickle.dump([s.cpu() for s in cur_samples], f) # move to cpu for ram

                # cur_samples has shape torch.Size([1, 1, 64, 64])
                # Plot it as an image grid

                for i, my_image in enumerate(cur_samples):
                    plt.imshow(my_image.reshape(64, 64).cpu(), cmap='gray')
                    os.makedirs(f'./step_samples/figs_step_{step_idx}/', exist_ok=True)
                    plt.savefig(f'./step_samples/figs_step_{step_idx}/{self.REINF_K}_{self.scheduler.num_steps}_{self.gen_idx}-reinf_{step_idx}_{batch_idx}_{i}.png')
                    plt.close()


            x0_batch = torch.stack(cur_samples, dim=0).to(device)
            Kb = x0_batch.size(0)

#             Hx = self.H.H(x0_batch.view(-1, *x0_batch.shape[2:]))
#             Hx = Hx.view(Kb, B, -1).contiguous()
#             y0_flat = y_0.view(B, -1).to(Hx.device)
#             residual = (y0_flat.unsqueeze(0) - Hx)
#             var = sigma_likelihood ** 2
#             phi_vals = (-0.5 / var) * (residual ** 2).sum(dim=2)
            phi_vals = -self.forward_op.loss(x0_batch, observation)

            phi_vals_list = phi_vals.cpu().numpy().tolist()

            all_phi_vals.extend(phi_vals_list)
            cond_scores = self.conditional_score_from_samples(
                xt, x0_batch, eps_pred, alpha_bar_t
            ).contiguous()

            sum_phi += phi_vals.sum(dim=0)
            sum_score += cond_scores.sum(dim=0)
            #print(phi_vals.shape,cond_scores.shape)
            sum_phi_score += (phi_vals[:, None, None, None, None] * cond_scores).sum(dim=0)
            total_K += Kb

            # free ram
            del cur_samples, x0_batch, phi_vals, cond_scores

            count += cur_batch
            batch_idx += 1
            print(f"-----{step_idx} -- Finished batch {batch_idx}: {count} / {num_samples} samples")

            mean_phi = (sum_phi / total_K).view(B, 1, 1, 1)
            grad_ckpt = (sum_phi_score / total_K) - mean_phi * (sum_score / total_K)

        all_phi_vals.sort()
        plt.figure(figsize=(10,5))
        plt.plot(all_phi_vals)
        plt.xlabel('Rank')
        plt.ylabel('Phi Value')
        plt.title(f'Phi Values for Step {step_idx}')
        plt.savefig(f'./step_samples/{self.REINF_K}_{self.scheduler.num_steps}_{self.gen_idx}-reinf_{step_idx}_phi_values.png')
        plt.close()
        
        min_phi, mean_return_phi, max_phi = all_phi_vals[0], np.mean(all_phi_vals), all_phi_vals[-1]

        if baseline == "mean":
            mean_phi = (sum_phi / total_K).view(B, 1, 1, 1)
            grad = (sum_phi_score / total_K) - mean_phi * (sum_score / total_K)
        else:
            grad = (sum_phi_score / total_K)

        plt.imshow(grad[0,0].cpu(), cmap='gray')
        plt.savefig(f'./step_samples/{self.REINF_K}_{self.scheduler.num_steps}_{self.gen_idx}-reinf_{step_idx}_grad.png')
        plt.close()

        return grad.contiguous(), min_phi, mean_return_phi, max_phi
