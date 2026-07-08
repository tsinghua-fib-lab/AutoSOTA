import numpy as np
import torch
from torch.autograd.functional import jacobian

def sqdist_euclid(x,y):
    """
    squared euclidean distance
    """
    return np.linalg.norm(y-x)**2

def grad_sqdist_euclid(x,y, wrt_first = False):
    """
    derivative of squared euclidean distance
    """
    deriv = 2*(y - x)

    return -deriv if wrt_first else deriv

def mixhess_sqdist_euclid(x,y, wrt_first = False):
    """
    mixed second derivatives of squared euclidean distance 
    """
    return -2*np.identity(len(y))

def sqdist_euclid_torch(x,y):
    """
    squared euclidean distance
    """
    return torch.norm(y-x)**2

def grad_sqdist_euclid_torch(x,y, wrt_first = False):
    """
    derivative of squared euclidean distance
    """
    deriv = 2*(y - x)
    
    return -deriv if wrt_first else deriv

def mixhess_sqdist_euclid_torch(x,y, wrt_first = False):
    """
    mixed second derivatives of squared euclidean distance 
    """
    n = y.numel()
    return -2*torch.eye(n, dtype=y.dtype, device=y.device)
                
def sqdist_pbeuclid(z1, z2, decoder, returntensor = True, args = None):
    """
    Squared distance based on pullback euclidean metric.
    Assumes decoder returns reconstruction directly.
    """
    if not (torch.is_tensor(z1) and torch.is_tensor(z2)):
        z1 = torch.tensor(z1, requires_grad=True, dtype = torch.float)
        z2 = torch.tensor(z2, requires_grad=True, dtype = torch.float)       

    # Ensure batch dimension
    if z1.dim() == 1:
        z1 = z1.unsqueeze(0)  # (1, latent_dim)
    if z2.dim() == 1:
        z2 = z2.unsqueeze(0)

    dec1 = decoder(z1)
    dec2 = decoder(z2)

    dist = torch.sum((dec2-dec1)**2)

    if returntensor is False:
        dist = dist.cpu().detach().numpy()
    return dist

def mix_hess_sqdist_pbeuclid(z1, z2, decoder, wrt_first=True, returntensor=True, args=None):
    """
    Computes mixed second derivatives of squared distance function sqdist_pbeuclid using backpropagation.
    Avoids second differentiation of decoder.
    """
    if not (torch.is_tensor(z1) and torch.is_tensor(z2)):
        z1 = torch.tensor(z1, requires_grad=True, dtype=torch.float)
        z2 = torch.tensor(z2, requires_grad=True, dtype=torch.float)

    # Ensure batch dimension
    if z1.dim() == 1:
        z1 = z1.unsqueeze(0)  # (1, latent_dim)
    if z2.dim() == 1:
        z2 = z2.unsqueeze(0)
    
    def f(z): return decoder(z).view(-1)
  
    deriv_z1 = jacobian(f, z1, create_graph=False).squeeze(1)
    deriv_z2 = jacobian(f, z2, create_graph=False).squeeze(1)

    if wrt_first:
        mixed_hess = -2 * deriv_z1.T @ deriv_z2
    else:
        mixed_hess = -2 * deriv_z2.T @ deriv_z1
    if returntensor is False:
        mixed_hess = mixed_hess.cpu().detach().numpy()
    return mixed_hess

def sqdist_KL(z1, z2, decoder, returntensor = True, args = None, factor=1000):
    """
    Squared distance based on KL divergence of gaussian decoder distributions.
    Assumes decoder returns mean and logvariance of diagonal Gaussian.
    """
    if not (torch.is_tensor(z1) and torch.is_tensor(z2)):
        z1 = torch.tensor(z1, requires_grad=True, dtype =torch.float)
        z2 = torch.tensor(z2, requires_grad=True, dtype = torch.float)       

    # ensure batch dimension
    if z1.dim() == 1:
        z1 = z1.unsqueeze(0)  # (1, latent_dim)
    if z2.dim() == 1:
        z2 = z2.unsqueeze(0)

    mu1, logvar1 = decoder(z1)
    mu2, logvar2 = decoder(z2)

    var1 = logvar1.exp()
    var2 = logvar2.exp()

    # KL divergence between two diagonal Gaussians
    kl = 0.5 * torch.sum(
        logvar2 - logvar1 +
        (var1 + (mu1 - mu2).pow(2)) / var2 -
        1, dim=1
    )
    if returntensor is False:
        kl = kl.cpu().detach().numpy()
    return kl.squeeze()/factor

def sqdist_KL_torchdistr(z1, z2, decoder, returntensor = True, args = None, factor=1000):
    """
    Squared distance based on KL divergence of decoder distributions using torch distributions.
    Assumes decoder returns a torch distribution object.
    """
    if not (torch.is_tensor(z1) and torch.is_tensor(z2)):
        z1 = torch.tensor(z1, requires_grad=True, dtype =torch.float)
        z2 = torch.tensor(z2, requires_grad=True, dtype = torch.float)       

    # ensure batch dimension
    if z1.dim() == 1:
        z1 = z1.unsqueeze(0)  # (1, latent_dim)
    if z2.dim() == 1:
        z2 = z2.unsqueeze(0)

    dist1 = decoder(z1)
    dist2 = decoder(z2)

    kl = torch.distributions.kl_divergence(dist1, dist2).sum(dim=1)
    if returntensor is False:
        kl = kl.cpu().detach().numpy()
    return kl.squeeze()/factor        

def grad_sqdist(z1, z2, decoder, sqdist, wrt_first = False, returntensor=True, args=None):
    """
    General gradient of squared distance function sqdist using backpropagation.
    """
    if not (torch.is_tensor(z1) and torch.is_tensor(z2)):
        z1 = torch.tensor(z1, requires_grad=True, dtype = torch.float)
        z2 = torch.tensor(z2, requires_grad=True, dtype = torch.float)
    
    if wrt_first:
        deriv = torch.autograd.grad(outputs = sqdist(z1, z2 , decoder, returntensor=True, args=args), 
                                    inputs=z1 , 
                                    create_graph=False,retain_graph=False)[0]

    else: 
        deriv = torch.autograd.grad(outputs = sqdist(z1, z2 , decoder, returntensor=True, args=args), 
                                    inputs=z2 ,
                                    create_graph=False, retain_graph=False)[0]
    if returntensor is False:
        deriv = deriv.cpu().detach().numpy()
    return deriv