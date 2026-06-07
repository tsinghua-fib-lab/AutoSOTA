"""Abstract SDE classes, Reverse SDE, and VE/VP SDEs."""
import abc
import torch
import os
import numpy as np
from models import utils as mutils
import ppo_utils


class RectifiedFlow():
    def __init__(self, init_type='gaussian', noise_scale=1.0, reflow_flag=False, 
                 reflow_t_schedule='uniform', reflow_loss='l2', use_ode_sampler='rk45', 
                 sigma_var=0.0, ode_tol=1e-5, sample_N=None, noise_sampler=None):
      if sample_N is not None:
        self.sample_N = sample_N
        print('Number of sampling steps:', self.sample_N)
      self.init_type = init_type
      
      self.noise_scale = noise_scale
      self.use_ode_sampler = use_ode_sampler
      self.ode_tol = ode_tol
      self.sigma_t = lambda t: (1. - t) * sigma_var
      self.noise_sampler = noise_sampler
      self.current_class = None
      self.preset_noise = None  # For directly setting noise
      self.deterministic = False  # Whether to use deterministic sampling (for evaluation)
      import lpips
      self.lpips_model = lpips.LPIPS(net='vgg')
      self.lpips_model = self.lpips_model.cuda()
      for p in self.lpips_model.parameters():
          p.requires_grad = False
      print('Init. Distribution Variance:', self.noise_scale)
      print('SDE Sampler Variance:', sigma_var)
      print('ODE Tolerence:', self.ode_tol)
            
      self.reflow_flag = reflow_flag
      if self.reflow_flag:
          self.reflow_t_schedule = reflow_t_schedule
          self.reflow_loss = reflow_loss
          if 'lpips' in reflow_loss:
              import lpips
              self.lpips_model = lpips.LPIPS(net='vgg')
              self.lpips_model = self.lpips_model.cuda()
              for p in self.lpips_model.parameters():
                  p.requires_grad = False

    @property
    def T(self):
      return 1.

    @torch.no_grad()
    def ode(self, init_input, model, reverse=False):
      ### run ODE solver for reflow. init_input can be \pi_0 or \pi_1
      from models.utils import from_flattened_numpy, to_flattened_numpy, get_score_fn
      from scipy import integrate
      rtol=1e-5
      atol=1e-5
      method='RK45'
      eps=1e-3

      # Initial sample
      x = init_input.detach().clone()

      model_fn = mutils.get_model_fn(model, train=False)
      shape = init_input.shape
      device = init_input.device

      def ode_func(t, x):
        x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
        vec_t = torch.ones(shape[0], device=x.device) * t
        drift = model_fn(x, vec_t*999)
        return to_flattened_numpy(drift)

      # Black-box ODE solver for the probability flow ODE
      if reverse:
        solution = integrate.solve_ivp(ode_func, (self.T, eps), to_flattened_numpy(x),
                                                     rtol=rtol, atol=atol, method=method)
      else:
        solution = integrate.solve_ivp(ode_func, (eps, self.T), to_flattened_numpy(x),
                                     rtol=rtol, atol=atol, method=method)
      x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)
      nfe = solution.nfev
      #print('NFE:', nfe) 

      return x

    @torch.no_grad()
    def euler_ode(self, init_input, model, reverse=False, N=100):
      ### run ODE solver for reflow. init_input can be \pi_0 or \pi_1
      eps=1e-3
      dt = 1./N

      # Initial sample
      x = init_input.detach().clone()

      model_fn = mutils.get_model_fn(model, train=False)
      shape = init_input.shape
      device = init_input.device
      
      for i in range(N):  
        num_t = i / N * (self.T - eps) + eps      
        t = torch.ones(shape[0], device=device) * num_t
        pred = model_fn(x, t*999)
        
        x = x.detach().clone() + pred * dt         

      return x

    def get_z0(self, batch, train=False, class_labels=None):
        """Get initial noise for sampling
        
        Args:
            batch: Input batch tensor with shape [n, c, h, w]
            train: Whether in training mode
            
        Returns:
            Noise tensor with shape [n, c, h, w]
        """
        n, c, h, w = batch.shape
        
        # If preset noise is available, use it (for evaluation with specific noise)
        if self.preset_noise is not None:
            noise = self.preset_noise
            self.preset_noise = None  # Clear after use
            return noise
        
        if self.init_type == 'gaussian':
            print('Using Gaussian initialization')
            ### standard gaussian
            cur_shape = (n, c, h, w)
            # tmp = torch.randn(cur_shape, device=batch.device)
            return torch.randn(cur_shape, device=batch.device) * self.noise_scale

        elif self.init_type == 'noise_sampler':
            ### Use noise sampler to generate noise
            if self.noise_sampler is None:
                raise ValueError("Noise sampler not set for noise_sampler mode")
            
            # Generate class labels for the batch
            if class_labels is None:
                if train:
                    # Training mode: randomly assign classes to each sample
                    # This creates diversity in training
                    class_labels = torch.randint(0, 10, (n,), 
                                                device=batch.device, dtype=torch.long)
                    exit()
                else:
                    # Evaluation mode
                    if self.current_class is not None:
                        # Use specified class (for class-specific evaluation)
                        class_labels = torch.full((n,), self.current_class, 
                                                device=batch.device, dtype=torch.long)
                        # print(class_labels)
                        # exit()
                    else:
                        # Random classes for evaluation (if no specific class set)
                        class_labels = torch.randint(0, 10, (n,), 
                                                    device=batch.device, dtype=torch.long)
            # Get distribution from noise sampler
            # class_labels = class_labels.to("cuda")
            with torch.no_grad():  # Ensure no gradients flow through frozen noise sampler
                noise_sample = ppo_utils.sample_noise(self.noise_sampler, class_labels, batch.device, deterministic=self.deterministic)
                # noise_sample = dist.rsample()  # [n, 3072] for RGB images
            
            # Reshape to image format
            # For CIFAR-10 RGB: 3072 = 3 * 32 * 32
            if c == 3 and h == 32 and w == 32:
                noise = noise_sample.view(n, c, h, w)  # [n, 3, 32, 32]
            elif c == 1 and h == 32 and w == 32:
                # For grayscale, noise_dim should be 1024 = 1 * 32 * 32
                noise = noise_sample[:, :1024].view(n, c, h, w)  # [n, 1, 32, 32]
            else:
                # For other resolutions, compute appropriate noise dimension
                expected_dim = c * h * w
                if noise_sample.shape[1] != expected_dim:
                    raise ValueError(f"Noise dimension mismatch: got {noise_sample.shape[1]}, expected {expected_dim}")
                noise = noise_sample.view(n, c, h, w)
            
            return noise * self.noise_scale
            
        else:
            raise NotImplementedError(f"Initialization type {self.init_type} not implemented")
    
    def set_target_class(self, target_class: int):
        """Set the target class for noise generation
        
        Args:
            target_class: Target class index (0-9 for CIFAR-10)
        """
        self.current_class = target_class
    
    def set_noise(self, noise: torch.Tensor):
        """Directly set the noise to be used in the next get_z0 call
        
        Args:
            noise: Pre-generated noise tensor with shape [n, c, h, w]
        """
        self.preset_noise = noise