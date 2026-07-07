import torch
import torch.nn as nn

class Net(nn.Module):
    """Simple Feed-Forward Network."""
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x): return self.net(x)

class ALDNet(nn.Module):
    """Network with shared backbone and specific head."""
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.head = nn.Linear(hidden_dim, output_dim)
    def forward(self, x): return self.head(self.shared(x))

class PLCPNet(nn.Module):
    """Network for Partition Learning Conformal Prediction."""
    def __init__(self, input_dim, n_groups=20, hidden_dim=256):
        super().__init__()
        self.h_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_groups)
        )
        self.q = nn.Parameter(torch.randn(1, n_groups)) 
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x): return self.softmax(self.h_net(x)), self.q

class ConditionalQuantileNet(nn.Module):
    """Takes (X, tau) as input to predict quantile."""
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x, tau):
        cat_input = torch.cat([x, tau], dim=1)
        return self.net(cat_input)

class ELUPlusOne(nn.Module):
    """ELU activation shifted by +1 to ensure strictly positive outputs with better gradient flow than Softplus."""
    def __init__(self, alpha=1.0):
        super().__init__()
        self.elu = nn.ELU(alpha=alpha)
    def forward(self, x):
        return self.elu(x) + 1.0

class MonotonicThreeHeadNet(nn.Module):
    """
    Backbone for CPCP (Colorful Pinball).
    Outputs main quantile and positive gaps for lower/upper bounds to ensure monotonicity.
    """
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.head_main = nn.Linear(hidden_dim, 1)
        self.head_lo_gap = nn.Linear(hidden_dim, 1)
        self.head_hi_gap = nn.Linear(hidden_dim, 1)
        self.elu_plus_one = ELUPlusOne(alpha=1.0)

    def forward(self, x):
        h = self.shared(x)
        q_main = self.head_main(h)
        delta_lo = self.elu_plus_one(self.head_lo_gap(h))
        delta_hi = self.elu_plus_one(self.head_hi_gap(h))
        q_lo = q_main - delta_lo
        q_hi = q_main + delta_hi
        return torch.cat([q_lo, q_main, q_hi], dim=1)

class MultivariateGaussianNet(nn.Module):
    """Outputs Mean vector and Cholesky factor of Covariance matrix."""
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.output_dim = output_dim
        self.n_cholesky = (output_dim * (output_dim + 1)) // 2
        
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.mu_head = nn.Linear(hidden_dim, output_dim)
        self.chol_head = nn.Linear(hidden_dim, self.n_cholesky)

    def forward(self, x):
        h = self.shared(x)
        mu = self.mu_head(h)
        chol_flat = self.chol_head(h)
        
        batch_size = x.shape[0]
        L = torch.zeros(batch_size, self.output_dim, self.output_dim).to(x.device)
        tril_indices = torch.tril_indices(row=self.output_dim, col=self.output_dim, offset=0)
        L[:, tril_indices[0], tril_indices[1]] = chol_flat
        
        # Ensure positive diagonal
        diag_indices = torch.arange(self.output_dim)
        L[:, diag_indices, diag_indices] = nn.Softplus()(L[:, diag_indices, diag_indices]) + 1e-6
        return mu, L