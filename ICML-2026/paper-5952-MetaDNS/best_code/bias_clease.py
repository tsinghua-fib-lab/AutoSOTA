import numpy as np
import torch


class BiasPotential:
    """
    Base class for Bias Potentials used in classical Metadynamics benchmarking.
    Mimics CLEASE structure but using PyTorch for compatibility with MDNS ecosystem.
    """
    def __init__(self, 
                 cv_min, 
                 cv_max, 
                 device='cpu'):
        """
        Args:
            cv_min (float): Minimum value of the Collective Variable (CV).
            cv_max (float): Maximum value of the CV.
            device (str): Device to store tensors on.
        """
        self.cv_min = cv_min
        self.cv_max = cv_max
        self.device = device

    def evaluate(self, cv_batch):
        """
        Return the bias energy V(s) for a batch of CV values.
        
        Args:
            cv_batch (Tensor): [B] tensor of CV values.
            
        Returns:
            bias_energy (Tensor): [B] tensor of bias energies.
        """
        raise NotImplementedError

    def update(self, cv_batch, height):
        """
        Update the bias potential based on visited states.
        
        Args:
            cv_batch (Tensor): [B] visited CV values.
            height (float): update height (dE) to add.
        """
        raise NotImplementedError

    def is_flat(self, flat_limit=0.8):
        """
        Check if the visited histogram is considered flat.
        
        Args:
            flat_limit (float): Threshold ratio (min_visits / mean_visits).
            
        Returns:
            bool: True if flat, False otherwise.
        """
        raise NotImplementedError

    def reset_visits(self):
        """Reset the visit histogram (used between modulation steps)."""
        raise NotImplementedError


class BinnedBiasPotential(BiasPotential):
    """
    Binned (Histogram-based) Bias Potential.
    Adapted from CLEASE's BinnedBiasPotential logic.
    """
    def __init__(self, 
                 cv_min=-1.0, 
                 cv_max=1.0, 
                 nbins=100, 
                 device='cpu'):
        super().__init__(cv_min, cv_max, device)
        self.nbins = nbins
        
        # Grid setup
        self.grid_vals = torch.linspace(cv_min, cv_max, nbins, device=device)
        # We assume uniform bins covering [cv_min, cv_max]
        # For simplicity in indexing: bin_idx = (val - min) / step
        self.step = (cv_max - cv_min) / (nbins - 1)
        
        # Potentials and Visits
        self.bias_grid = torch.zeros(nbins, device=device)
        self.visit_grid = torch.zeros(nbins, device=device)

    def _get_indices(self, cv_batch):
        """Convert CV values to bin indices with clamping."""
        # Indicies float
        indices_float = (cv_batch - self.cv_min) / self.step
        indices = torch.round(indices_float).long()
        # Clamp to ensure valid indices
        return torch.clamp(indices, 0, self.nbins - 1)

    def evaluate(self, cv_batch):
        cv_batch = cv_batch.to(self.device)
        indices = self._get_indices(cv_batch)
        return self.bias_grid[indices]

    def update(self, cv_batch, height):
        """
        Add `height` to the bins corresponding to cv_batch.
        Also increments visit count.
        """
        cv_batch = cv_batch.to(self.device)
        indices = self._get_indices(cv_batch)
        
        # Vectorized addition using scatter_add_ is cleanest, but simple loop or bincount works too.
        # Since we just want to add 'height' to specific indices:
        # We can use index_put_ with accumulate=True
        
        # Create a tensor of heights
        updates = torch.full_like(cv_batch, height)
        
        # Accumulate bias
        self.bias_grid.index_put_((indices,), updates, accumulate=True)
        
        # Accumulate visits (add 1)
        ones = torch.ones_like(cv_batch)
        self.visit_grid.index_put_((indices,), ones, accumulate=True)

    def is_flat(self, flat_limit=0.8):
        """
        Check flatness criterion: min(visits) > flat_limit * mean(visits).
        Ignores bins with 0 visits if they are "unreachable" or far outliers, 
        but CLEASE typically checks the range of interest.
        Here we check all bins between first and last visited bins to avoid edge artifacts.
        """
        visits = self.visit_grid.detach()
        
        # Find range of relevant bins (or use all)
        # CLEASE checks effective range. Let's filter for bins that have been visited at least once 
        # OR usually we check the whole range if we expect full coverage.
        # Let's strictly follow the definition given in user script/CLEASE:
        # "The histogram of visits is considered flat, when the minimum value is larger than flat_limit*np.mean(hist)"
        
        # However, checking strictly 0-visit bins might prevent ever finishing if boundaries are hard to reach.
        # Standard approach: Check only bins that have meaningful probability?
        # For Ising, we expect full -1 to 1 coverage eventually. 
        # But if we start, most bins are 0.
        
        # If mean is 0, we are definitely not flat (not started)
        mean_visits = torch.mean(visits)
        if mean_visits == 0:
            return False
            
        min_visits = torch.min(visits)
        
        return (min_visits > flat_limit * mean_visits).item()

    def reset_visits(self):
        """Reset visit counts to zero."""
        self.visit_grid.zero_()

    def get_bias_grid_np(self):
        """Return grid and bias values as numpy arrays for plotting."""
        return self.grid_vals.detach().cpu().numpy(), self.bias_grid.detach().cpu().numpy()

    def state_dict(self):
        return {
            'bias_grid': self.bias_grid,
            'visit_grid': self.visit_grid,
            'params': {
                'cv_min': self.cv_min, 
                'cv_max': self.cv_max, 
                'nbins': self.nbins
            }
        }

    def load_state_dict(self, state_dict):
        self.bias_grid = state_dict['bias_grid'].to(self.device)
        self.visit_grid = state_dict['visit_grid'].to(self.device)
