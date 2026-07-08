import math
import torch.fft as fft
import torch

from .base import Algo

import wandb

# -----------------------------------------------------------------------------------------------
# Paper: Ensemble Kalman methods for inverse problems.
# This implementation is based on the implemenation in https://github.com/devzhk/enkg-pytorch
# -----------------------------------------------------------------------------------------------

class GaussianRF2d(object):
    def __init__(self, s1, s2, 
                 L1=2*math.pi, L2=2*math.pi, 
                 alpha=2.0, tau=3.0, sigma=None, mean=None, 
                 boundary="periodic", device=None, dtype=torch.float32):

        self.s1 = s1
        self.s2 = s2

        self.mean = mean

        self.device = device
        self.dtype = dtype

        if sigma is None:
            self.sigma = tau**(0.5*(2*alpha - 2.0))
        else:
            self.sigma = sigma

        const1 = (4*(math.pi**2))/(L1**2)
        const2 = (4*(math.pi**2))/(L2**2)
        norm_const = math.sqrt(2.0/(L1*L2))

        freq_list1 = torch.cat((torch.arange(start=0, end=s1//2, step=1),\
                                torch.arange(start=-s1//2, end=0, step=1)), 0)
        k1 = freq_list1.view(-1,1).repeat(1, s2).type(dtype).to(device)

        freq_list2 = torch.cat((torch.arange(start=0, end=s2//2, step=1),\
                                torch.arange(start=-s2//2, end=0, step=1)), 0)

        k2 = freq_list2.view(1,-1).repeat(s1, 1).type(dtype).to(device)

        self.sqrt_eig = s1*s2*self.sigma*norm_const*((const1*k1**2 + const2*k2**2 + tau**2)**(-alpha/2.0))
        self.sqrt_eig[0,0] = 0.0
        self.sqrt_eig[torch.logical_and(k1 + k2 <= 0.0, torch.logical_or(k1 + k2 != 0.0, k1 <= 0.0))] = 0.0

    @torch.no_grad()
    def sample(self, N, xi=None):
        if xi is None:
            xi  = torch.randn(N, self.s1, self.s2, 2, dtype=self.dtype, device=self.device)
        
        xi[...,0] = self.sqrt_eig*xi [...,0]
        xi[...,1] = self.sqrt_eig*xi [...,1]
        
        u = fft.ifft2(torch.view_as_complex(xi), s=(self.s1, self.s2)).imag

        if self.mean is not None:
            u += self.mean
        
        return u
    

class EKI(Algo):
    def __init__(self, net, forward_op, 
                 guidance_scale, num_updates, 
                 num_samples=1024, 
                 resolution=128, 
                 L=2 * math.pi, 
                 device=torch.device('cuda')
                 ):
        super().__init__(net, forward_op)
        self.guidance_scale = guidance_scale
        self.num_updates = num_updates
        self.num_samples = num_samples

        self.grf = GaussianRF2d(s1=resolution, s2=resolution, L1=L, L2=L, 
                                alpha=4.0, tau=3.0, device=device)
    
    @torch.no_grad()
    def inference(self, observation, num_samples=1, verbose=False):
        x_initial = self.grf.sample(self.num_samples)
        x_next = x_initial.reshape(self.num_samples, self.net.img_channels, self.net.img_resolution, self.net.img_resolution)
        for i in range(self.num_updates):
            ys = self.forward_op.forward(x_next)
            xs_diff = x_next - x_next.mean(dim=0, keepdim=True)
            ys_diff = ys - ys.mean(dim=0, keepdim=True)
            ys_err = ys - observation

            coef = (
                torch.matmul(
                    ys_err.reshape(ys_err.shape[0], -1),
                    ys_diff.reshape(ys_diff.shape[0], -1).T,
                )
                / self.num_samples
            )
            
            dxs = coef @ xs_diff.reshape(self.num_samples, -1)
            lr = self.guidance_scale / torch.linalg.matrix_norm(coef)

            x_next = x_next - lr * dxs.reshape(x_next.shape)
            if wandb.run is not None:
                abs_err = torch.abs(ys_err)
                avg_err = torch.mean(abs_err)
                max_err = torch.max(abs_err)
                std = torch.std(x_next, dim=0, keepdim=True)
                avg_std = torch.mean(std)
                wandb.log(
                    {
                        "EKI/abs error": avg_err.item(),
                        'EKI/max error': max_err.item(),
                        "EKI/std": avg_std.item(),
                    }
                )

        return x_next
