import numpy as np
import torch


class BiasPotential:
    """
    Bias Potential for Well-Tempered Metadynamics / Ajoint Schrödinger Bridge Sampler (WT-ASBS).
    
    Supports:
    - 1D Collective Variables (CV), specifically geared towards Magnetization/Composition.
    - Gaussian Kernels (Standard Metadynamics)
    - Delta/Kronecker Kernels (Discrete/Histogram Biasing)
    """
    def __init__(self, 
                 cv_min, 
                 cv_max, 
                 grid_size, 
                 sigma, 
                 initial_height, 
                 bias_factor, 
                 T, 
                 kernel_type='gaussian',
                 device='cpu',
                 energy_scaling=1.0):
        """
        Args:
            cv_min (float): Minimum value of the CV (e.g., -1 for magnetization).
            cv_max (float): Maximum value of the CV (e.g., +1 for magnetization).
            grid_size (int): Number of bins in the grid.
            sigma (float): Width of Gaussian kernel (ignored for 'delta').
            initial_height (float): Initial height (W) of the added bias.
            bias_factor (float): Bias factor (gamma). gamma > 1. 
            T (float): Temperature (kB*T units).
            kernel_type (str): 'gaussian' or 'delta'.
            device (str): torch device.
            energy_scaling (float): Factor to scale delta_T (e.g. system size N).
                                  Essential for extensive barriers with intensive CVs.
        """
        self.cv_min = cv_min
        self.cv_max = cv_max
        self.grid_size = grid_size
        self.sigma = sigma
        self.initial_height = initial_height
        self.gamma = bias_factor
        self.T = T
        self.kernel_type = kernel_type.lower()
        self.device = device
        self.energy_scaling = energy_scaling
        
        # Initialize grid
        # using linspace for bucket centers
        self.grid_vals = torch.linspace(cv_min, cv_max, grid_size, device=device)
        self.bias_grid = torch.zeros(grid_size, device=device)
        
        # Precompute kernel constant for delta_T
        # Delta_T = (gamma - 1) * T
        # If energy_scaling > 1, we scale Delta_T to allow bias to reach extensive heights
        if self.gamma > 1.0:
            self.delta_T = (self.gamma - 1) * self.T * self.energy_scaling
        else:
            # If gamma=1, standard MD (no well-temperedness)
            self.delta_T = 1e9

    def _get_gaussian_kernel(self, center):
        """Returns a Gaussian kernel centered at `center` evaluated on the grid."""
        # exp( - (x - center)^2 / (2*sigma^2) )
        return torch.exp( - (self.grid_vals - center)**2 / (2 * self.sigma**2) )

    def _get_delta_kernel(self, center):
        """Returns a Delta kernel (1.0 at nearest bin, 0 elsewhere)."""
        # Find nearest bin index
        idx = torch.argmin(torch.abs(self.grid_vals - center))
        kernel = torch.zeros_like(self.grid_vals)
        kernel[idx] = 1.0
        return kernel

    def update(self, cv_batch):
        """
        Update the bias potential based on a batch of visited CV values.
        
        Args:
           cv_batch (Tensor): [B] tensor of visited CV values.
        """
        # Ensure input is on correct device
        cv_batch = cv_batch.to(self.device)
        
        for cv_val in cv_batch:
            # 1. Evaluate current bias at this CV location
            # Interpolation is ideal, but for discrete grid/dense grid, nearest lookup is often sufficient/faster
            # Let's use linear interpolation for 'gaussian' mode to be smooth, 
            # and nearest bin for 'delta' mode.
            
            if self.kernel_type == 'delta':
                idx = torch.argmin(torch.abs(self.grid_vals - cv_val))
                current_v = self.bias_grid[idx]
            else:
                # Simple finding of nearest for evaluating V(s) for the update rule 
                # (Standard WT uses V(s_t) exactly at the point)
                # For smooth grids, we can interpolate. For now, nearest grid point approximation 
                # is standard in many lightweight implementations.
                idx = torch.argmin(torch.abs(self.grid_vals - cv_val))
                current_v = self.bias_grid[idx]

            # 2. Calculate height scaling (Well-Tempered)
            # W_eff = W * exp( - V(s_t) / (kB * Delta_T) )
            scale_factor = torch.exp( - current_v / self.delta_T )
            height = self.initial_height * scale_factor
            
            # 3. Add kernel
            if self.kernel_type == 'delta':
                # Add directly to the bin
                self.bias_grid[idx] += height
                
            elif self.kernel_type == 'gaussian':
                # Add gaussian to the whole grid
                kernel = self._get_gaussian_kernel(cv_val)
                self.bias_grid += height * kernel

    def normalize(self):
        """
        Shift the bias potential so that the minimum value is 0.
        Often used to keep values bounded during training.
        """
        min_val = self.bias_grid.min()
        self.bias_grid -= min_val

    def evaluate(self, cv_batch):
        """
        Return the bias energy V(s) for a batch of CV values.
        
        Args:
            cv_batch (Tensor): [B] Values of CV.
            
        Returns:
            start_v (Tensor): [B] Bias values.
        """
        cv_batch = cv_batch.to(self.device)
        
        # For each sample, find value from grid
        # We can implement efficient batched interpolation
        
        # Find indices
        # Assuming uniform grid:
        # result = bias_grid[ (val - min) / step ] 
        step = (self.cv_max - self.cv_min) / (self.grid_size - 1)
        
        # Clamping to ensure we stay in bounds
        indices_float = (cv_batch - self.cv_min) / step
        indices_long = torch.round(indices_float).long()
        indices_long = torch.clamp(indices_long, 0, self.grid_size - 1)
        
        return self.bias_grid[indices_long]


    def get_bias_grid_np(self):
        """Return grid for plotting (numpy)"""
        return self.grid_vals.detach().cpu().numpy(), self.bias_grid.detach().cpu().numpy()

    def state_dict(self):
        """Return state dictionary for saving. Clones tensors so saved state is a snapshot."""
        return {
            'bias_grid': self.bias_grid.detach().clone(),
            'grid_vals': self.grid_vals.detach().clone(),
            'params': {
                'cv_min': self.cv_min,
                'cv_max': self.cv_max,
                'grid_size': self.grid_size,
                'sigma': self.sigma,
                'initial_height': self.initial_height,
                'bias_factor': self.gamma,
                'T': self.T,
                'kernel_type': self.kernel_type
            }
        }

    def load_state_dict(self, state_dict):
        """Load state from dictionary."""
        self.bias_grid = state_dict['bias_grid'].to(self.device)
        self.grid_vals = state_dict['grid_vals'].to(self.device)
        # We assume params are consistent or handled by init, 
        # but we could optionally overwrite them if needed. 
        # For now, just loading the grid is the most critical part.


class BiasPotentialMultiDim:
    """
    Bias Potential for Multi-Dimensional CVs.
    """
    def __init__(self, 
                 cv_min, 
                 cv_max, 
                 grid_size, 
                 sigma, 
                 initial_height, 
                 bias_factor, 
                 T, 
                 kernel_type='gaussian',
                 device='cpu',
                 energy_scaling=1.0, sigma_schedule='fixed',
                 sigma_start=None, sigma_end=None):
        # Normalize inputs to lists
        if isinstance(cv_min, (int, float)): cv_min = [float(cv_min)]
        if isinstance(cv_max, (int, float)): cv_max = [float(cv_max)]

        self.ndim = len(cv_min)
        assert len(cv_max) == self.ndim, "cv_min/max length mismatch"

        if isinstance(grid_size, int) or isinstance(grid_size, str):
             # Handle str if passed from args like "100" or "100,100"
             if isinstance(grid_size, str):
                 if ',' in grid_size:
                     grid_size = [int(g) for g in grid_size.split(',')]
                 else:
                     grid_size = int(grid_size)

        if isinstance(grid_size, int): grid_size = [grid_size] * self.ndim
        if isinstance(grid_size, list) and len(grid_size) == 1 and self.ndim > 1:
            grid_size = grid_size * self.ndim

        if isinstance(sigma, (int, float)) or isinstance(sigma, str):
             if isinstance(sigma, str):
                 if ',' in sigma:
                     sigma = [float(s) for s in sigma.split(',')]
                 else:
                     sigma = float(sigma)

        if isinstance(sigma, (int, float)): sigma = [float(sigma)] * self.ndim
        if isinstance(sigma, list) and len(sigma) == 1 and self.ndim > 1:
            sigma = sigma * self.ndim

        self.cv_min = torch.tensor(cv_min, device=device)
        self.cv_max = torch.tensor(cv_max, device=device)
        self.grid_size = torch.tensor(grid_size, device=device)
        self.grid_shape = [int(g) for g in grid_size]
        self.sigma = torch.tensor(sigma, device=device)
        self.initial_sigma = self.sigma.clone()

        self.initial_height = initial_height
        self.gamma = bias_factor
        self.T = T
        self.kernel_type = kernel_type.lower()
        self.device = device
        self.energy_scaling = energy_scaling

        # Adaptive sigma scheduling
        self.sigma_schedule = sigma_schedule  # 'fixed' or 'adaptive'
        if sigma_schedule == 'adaptive':
            if sigma_start is not None:
                if isinstance(sigma_start, str):
                    if ',' in sigma_start:
                        sigma_start_vals = [float(s) for s in sigma_start.split(',')]
                    else:
                        sigma_start_vals = [float(sigma_start)] * self.ndim
                elif isinstance(sigma_start, (int, float)):
                    sigma_start_vals = [float(sigma_start)] * self.ndim
                else:
                    sigma_start_vals = list(sigma_start)
            else:
                sigma_start_vals = [float(s) * 2.0 for s in self.initial_sigma.cpu().tolist()]

            if sigma_end is not None:
                if isinstance(sigma_end, str):
                    if ',' in sigma_end:
                        sigma_end_vals = [float(s) for s in sigma_end.split(',')]
                    else:
                        sigma_end_vals = [float(sigma_end)] * self.ndim
                elif isinstance(sigma_end, (int, float)):
                    sigma_end_vals = [float(sigma_end)] * self.ndim
                else:
                    sigma_end_vals = list(sigma_end)
            else:
                sigma_end_vals = [float(s) * 0.4 for s in self.initial_sigma.cpu().tolist()]

            self.sigma_start = torch.tensor(sigma_start_vals, device=device)
            self.sigma_end = torch.tensor(sigma_end_vals, device=device)
            self.sigma = self.sigma_start.clone()
            print(f"Adaptive sigma: start={self.sigma_start.tolist()}, end={self.sigma_end.tolist()}")

        self.update_count = 0
        self.total_cells = int(torch.prod(torch.tensor(self.grid_shape)).item())

        # Initialize N-D grid
        coords = []
        for i in range(self.ndim):
            line = torch.linspace(float(self.cv_min[i]), float(self.cv_max[i]), int(self.grid_size[i]), device=device)
            coords.append(line)

        self.grid_coords = torch.meshgrid(*coords, indexing='ij')
        # Stack to (G1, ..., GN, N)
        self.grid_vals = torch.stack(self.grid_coords, dim=-1)
        self.bias_grid = torch.zeros(tuple(self.grid_shape), device=device)

        if self.gamma > 1.0:
            self.delta_T = (self.gamma - 1) * self.T * self.energy_scaling
        else:
            self.delta_T = 1e9

    def _get_gaussian_kernel(self, center):
        # Center: [N]
        # grid_vals: [G1, ..., GN, N]
        diff_sq = (self.grid_vals - center) ** 2
        exponent = - torch.sum(diff_sq / (2 * self.sigma**2), dim=-1)
        return torch.exp(exponent)

    def _get_indices(self, cv_val):
        indices = []
        for i in range(self.ndim):
            val = cv_val[i]
            vmin = self.cv_min[i]
            vmax = self.cv_max[i]
            gsize = self.grid_size[i]
            
            step = (vmax - vmin) / (gsize - 1)
            idx_float = (val - vmin) / step
            idx_long = torch.round(idx_float).long()
            idx_long = torch.clamp(idx_long, 0, int(gsize) - 1)
            indices.append(idx_long)
        return tuple(indices)

    def update(self, cv_batch):
        """Update bias with [B, N] CV batch."""
        cv_batch = cv_batch.to(self.device).float()
        assert cv_batch.shape[1] == self.ndim

        for cv_val in cv_batch:
            indices = self._get_indices(cv_val)
            current_v = self.bias_grid[indices]

            scale_factor = torch.exp( - current_v / self.delta_T )
            height = self.initial_height * scale_factor

            if self.kernel_type == 'delta':
                self.bias_grid[indices] += height
            elif self.kernel_type == 'gaussian':
                kernel = self._get_gaussian_kernel(cv_val)
                self.bias_grid += height * kernel

        # Update sigma based on grid coverage (adaptive scheduling)
        self.update_count += 1
        if self.sigma_schedule == 'adaptive':
            # Compute coverage: fraction of grid cells with bias > threshold
            threshold = self.initial_height * 0.01  # Cells with meaningful bias
            coverage = (self.bias_grid > threshold).float().mean().item()
            # Linearly interpolate between start and end based on coverage
            t = min(coverage / 0.5, 1.0)  # Fully narrow by 50% coverage
            self.sigma = self.sigma_start + (self.sigma_end - self.sigma_start) * t

    def evaluate(self, cv_batch):
        """Evaluate bias for [B, N] CV batch."""
        cv_batch = cv_batch.to(self.device).float()
        assert cv_batch.shape[1] == self.ndim
        
        indices_list = []
        for i in range(self.ndim):
            val = cv_batch[:, i]
            vmin = self.cv_min[i]
            vmax = self.cv_max[i]
            gsize = self.grid_size[i]
            
            step = (vmax - vmin) / (gsize - 1)
            idx_float = (val - vmin) / step
            idx_long = torch.round(idx_float).long()
            idx_long = torch.clamp(idx_long, 0, int(gsize) - 1)
            indices_list.append(idx_long)
            
        return self.bias_grid[tuple(indices_list)]

    def get_bias_grid_np(self):
        """Return grid for plotting (numpy)."""
        coords_np = [c.detach().cpu().numpy() for c in self.grid_coords]
        bias_np = self.bias_grid.detach().cpu().numpy()
        return coords_np, bias_np

    def state_dict(self):
        """Return state dictionary for saving. Clones tensors so saved state is a snapshot."""
        state = {
            'bias_grid': self.bias_grid.detach().clone(),
            'params': {
                'cv_min': self.cv_min.cpu().tolist(),
                'cv_max': self.cv_max.cpu().tolist(),
                'grid_size': self.grid_size.cpu().tolist(),
                'sigma': self.sigma.cpu().tolist(),
                'initial_height': self.initial_height,
                'bias_factor': self.gamma,
                'T': self.T, 'kernel_type': self.kernel_type,
                'update_count': self.update_count,
                'sigma_schedule': self.sigma_schedule,
            }
        }
        if self.sigma_schedule == 'adaptive':
            state['params']['sigma_start'] = self.sigma_start.cpu().tolist()
            state['params']['sigma_end'] = self.sigma_end.cpu().tolist()
        return state
        
    def load_state_dict(self, state_dict):
        self.bias_grid = state_dict['bias_grid'].to(self.device)

