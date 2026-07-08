# Copyright (c) 2025, Weijian Luo. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Mean Flows for One-step Generative Modeling"."""

# NOTE: This file is a copy of the file of the EDM project.
#       It has been modified to fit the needs of the project.
#       The original file can be found at:
#       https://github.com/NVlabs/edm

import torch
from torch_utils import persistence
import torch.nn.functional as F

@persistence.persistent_class
class MeanFlowLoss:
    def __init__(self, P_mean=-0.4, P_std=1.0, sigma_data=0.5, 
                 noise_dist='logit_normal', detach_tgt=True,
                 data_proportion=0.75, num_classes=None,
                 class_dropout_prob=0.1, norm_p=1.0, norm_eps=1.0,
                 guidance_eq='cfg', omega=1.0, kappa=0.5, t_start = 0.0, t_end = 1.0):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.detach_tgt = detach_tgt
        self.data_proportion = data_proportion
        self.num_classes = num_classes
        self.class_dropout_prob = class_dropout_prob
        self.norm_p = norm_p
        self.norm_eps = norm_eps
        self.guidance_eq = guidance_eq
        self.omega = omega
        self.kappa = kappa
        self.noise_dist = noise_dist
        self.t_start = t_start
        self.t_end = t_end

    def _logit_normal_dist(self, shape, device):
        rnd_normal = torch.randn(shape, device=device)
        return torch.sigmoid(rnd_normal * self.P_std + self.P_mean)

    def _uniform_dist(self, shape, device):
        return torch.rand(shape, device=device)

    def noise_distribution(self, shape, device):
        if self.noise_dist == 'logit_normal':
            return self._logit_normal_dist(shape, device)
        elif self.noise_dist == 'uniform':
            return self._uniform_dist(shape, device)
        else:
            raise ValueError(f"Unknown noise distribution: {self.noise_dist}")

    def _apply_guidance(self, v, v_uncond, v_cond, t):
        omega = torch.where((t >= self.t_start) & (t <= self.t_end), 
                            self.omega, 1.0)
        
        if self.guidance_eq == 'cfg' and self.kappa == 0:
            return v_uncond + omega * (v - v_uncond)
        elif self.guidance_eq == 'cfg' and self.kappa > 0:
            kappa = torch.where((t >= self.t_start) & (t <= self.t_end), 
                                self.kappa, 0.0)
            return omega * v + (1 - omega - kappa) * v_uncond + kappa * v_cond
        else:
            return v

    def _cond_drop(self, labels, v, v_g):
        rand_mask = torch.rand(labels.shape[0], device=labels.device) < self.class_dropout_prob
        labels_drop = labels.clone()
        labels_drop[rand_mask] = self.num_classes
        v_g = torch.where(rand_mask[:, None, None, None], v, v_g)
        return labels_drop, v_g

    def __call__(self, net, images, labels=None, augment_pipe=None):
        x = images
        device = x.device
        batch_size = x.shape[0]
        shape = (batch_size, 1, 1, 1)

        t = self.noise_distribution(shape, device) # Sample t and r from noise distribution
        r = self.noise_distribution(shape, device)
        t, r = torch.max(t, r), torch.min(t, r)

        zero_mask = torch.arange(batch_size, device=device) < int(batch_size * self.data_proportion)
        zero_mask = zero_mask.view(shape)
        r = torch.where(zero_mask, t, r)  # Ensure t >= r and apply data proportion

        # Apply augmentations if needed
        if augment_pipe is not None:
            y, augment_labels = augment_pipe(x)
        else:
            y, augment_labels = x, None

        n = torch.randn_like(y) # Create noise and corrupted image
        z_t = (1 - t) * y + t * n
        v = n - y  # True velocity
        
        # Prepare labels for guidance
        labels_in = labels

        if labels is not None and self.class_dropout_prob > 0:
            with torch.no_grad():
                # Unconditional labels
                labels_null = torch.full_like(labels, self.num_classes)
                
                # Get conditional and unconditional velocities
                v_cond = net.module(z_t, t, class_labels=labels, h=torch.zeros_like(t), augment_labels=augment_labels)
                v_uncond = net.module(z_t, t, class_labels=labels_null, h=torch.zeros_like(t), augment_labels=augment_labels)
                
                # Apply guidance and conditional dropout
                v_g = self._apply_guidance(v, v_uncond, v_cond, t)
                labels_in, v_g = self._cond_drop(labels, v, v_g)
        else:
            v_g = v

        # Compute model output and time derivative
        def u_wrapper(z, t, r):
            return net.module(z, t, class_labels=labels_in, h=t-r, augment_labels=augment_labels)
        
        primals = (z_t, t, r)
        tangents = (v_g, torch.ones_like(t), torch.zeros_like(t))
        u, du_dt = torch.func.jvp(u_wrapper, primals, tangents)

        u_tgt = v_g - torch.clamp(t - r, min=0.0, max=1.0) * du_dt # Compute target velocity
        
        if self.detach_tgt:
            u_tgt = u_tgt.detach()

        unweighted_loss = (u - u_tgt).pow(2).sum(dim=[1, 2, 3]) # Adaptive loss weighting
        with torch.no_grad():
            adaptive_weight = 1 / (unweighted_loss + self.norm_eps).pow(self.norm_p)
        
        loss = unweighted_loss * adaptive_weight
        return loss.sum()
