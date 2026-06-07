import torch
import numpy as np

class LinearPolarizer(torch.nn.Module):
    def __init__(self, init_value=0.0,opt = None):
        super().__init__()
        self.phi = torch.nn.Parameter(torch.tensor(init_value, dtype=torch.float32))
        if opt:
            self.optimizer = torch.optim.Adam([self.phi], lr=opt.lp_lr)
        else:
            self.optimizer = torch.optim.Adam([self.phi], lr=0.01)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.98)
    def forward(self, stokes):
        sin, cos = torch.sin, torch.cos
        phi = (self.phi).to(stokes.device)  
        one = torch.tensor(1., dtype=stokes.dtype, device=stokes.device)
        zero = torch.tensor(0., dtype=stokes.dtype, device=stokes.device)
        H, W = stokes.shape[-3], stokes.shape[-2]
        
        cos2p = torch.cos(2*phi)
        sin2p = torch.sin(2*phi)
        M_phi = torch.stack([
            torch.stack([one, cos2p, sin2p, zero]),
            torch.stack([cos2p, cos2p**2, sin2p*cos2p, zero]),
            torch.stack([sin2p, sin2p*cos2p, sin2p**2, zero]),
            torch.stack([zero, zero, zero, zero])
        ])  # shape (3,3)
        
        stokes_out = 0.5 * torch.einsum('...ij,...j->...i', M_phi[...,:3,:3], stokes)
        return stokes_out
    def set_phi(self, value):
        if isinstance(value, (int, float)):
            self.phi.data.fill_(value)
        elif isinstance(value, torch.Tensor):
            self.phi.data.copy_(value)
        else:
            raise ValueError("Value must be a scalar or a tensor.")

    def get_phi(self):
        return self.phi.item()
    def get_phi_tensor(self):
        return self.phi