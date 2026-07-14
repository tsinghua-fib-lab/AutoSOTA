import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.func import functional_call, vmap, grad

from networks.network_interface import FisherInterface, Network
from networks.layers import BP_layer
from networks.activation_function import ReLU, Softplus, Linear


class oEWC_network(Network, FisherInterface):
    """
    Online Elastic Weight Consolidation (oEWC) network.
    
    Based on Schwarz et al. (2018) "Progress & Compress: A scalable framework 
    for continual learning" (Section 4: Online EWC).
    
    Key differences from standard EWC:
    1. Cumulative Fisher with decay: F*_i = γ * F*_{i-1} + F_i
    2. Single reference point: only stores the most recent θ* (re-centering)
    3. Fisher normalization: normalizes per-task Fisher to treat tasks equally
    
    The loss for task i is:
        L_task(θ) + (λ/2) * ||θ - θ*_{i-1}||²_{γF*_{i-1}}
    """
    
    def __init__(self, config, name="oEWC_network"):
        super().__init__(BP_layer, ReLU, Linear, config, name)
        
        # oEWC hyperparameters
        self.importance = config.importance_ewc  # λ: regularization strength
        self.gamma = getattr(config, 'gamma_oewc', 0.95)  # γ: Fisher decay factor
        self.normalize_fisher = getattr(config, 'normalize_fisher', True)
        
        # Storage for online EWC
        self._fisher = {}  # Cumulative Fisher: F*
        self._theta_star = {}  # Most recent optimal parameters: θ*
        self._first_task = True

    def backward(self, y):
        """Compute loss and backpropagate with oEWC regularization."""
        loss = self.loss_fn(self.y_hat, y)
        if not self._first_task:
            loss += self.oewc_loss()
        loss.backward()

    def oewc_loss(self):
        """
        Compute online EWC regularization loss.
        
        Loss = (λ/2) * Σ_n F*_n * (θ_n - θ*_n)²
        
        Where F* is the cumulative (decayed) Fisher and θ* is the most 
        recent optimal parameters.
        """
        loss = 0.0
        for n, p in self.named_parameters():
            if n in self._theta_star and n in self._fisher:
                loss += torch.sum(self._fisher[n] * (p - self._theta_star[n]) ** 2)
        return self.importance * loss

    def _normalize_fisher(self, fisher):
        """
        Normalize Fisher matrix to have unit norm.
        
        From Schwarz et al.: "We counteract this issue by normalising the 
        Fisher information matrices F_i for each task. This allows the 
        algorithm to compute the updates based on the relative importance 
        of weights in a network, i.e. treating each task equally."
        """
        # Compute total norm across all parameters
        total_norm = sum(torch.norm(f).item() ** 2 for f in fisher.values()) ** 0.5
        
        if total_norm > 0:
            for n in fisher:
                fisher[n] = fisher[n] / (total_norm + 1e-8)
        
        return fisher

    def complete_task(self, dataloader):
        """
        Update cumulative Fisher and reference parameters after task completion.
        
        Online EWC update rule (Eq. 8-9 from Schwarz et al.):
            F*_i = γ * F*_{i-1} + F_i
            θ*_i = argmin_θ L_i(θ) + (γ/2)||θ - θ*_{i-1}||²_{F*_{i-1}}
        
        The re-centering at θ*_i (latest MAP) is key to online EWC.
        """
        # Compute Fisher for current task
        current_fisher = self._calculate_fisher(dataloader)
        
        # Normalize Fisher if enabled (treats all tasks equally)
        if self.normalize_fisher:
            current_fisher = self._normalize_fisher(current_fisher)
        
        # Store current parameters as the new reference point (re-centering)
        self._theta_star = {
            n: p.data.clone() for n, p in self.named_parameters() if p.requires_grad
        }
        
        if self._first_task:
            # First task: just store the Fisher
            self._fisher = current_fisher
            self._first_task = False
        else:
            # Online update: F*_new = γ * F*_old + F_current
            for n in self._fisher:
                self._fisher[n] = self.gamma * self._fisher[n] + current_fisher[n]

    def get_regularization_strength(self):
        """Return current regularization statistics for monitoring."""
        if not self._fisher:
            return {}
        
        stats = {
            'fisher_norm': sum(torch.norm(f).item() ** 2 for f in self._fisher.values()) ** 0.5,
            'fisher_mean': torch.cat([f.flatten() for f in self._fisher.values()]).mean().item(),
            'fisher_max': torch.cat([f.flatten() for f in self._fisher.values()]).max().item(),
        }
        return stats