import torch

from .metrics import sqdist_euclid_torch, grad_sqdist_euclid_torch, mixhess_sqdist_euclid_torch

        
def KExponentialTorch(ConstrObj, ConstrGrad, x0, v, K, sqdist = sqdist_euclid_torch, gradsqdist=grad_sqdist_euclid_torch, 
                      mixhesssqdist = mixhess_sqdist_euclid_torch, constrfact=1, firstStep='step'):
    """
    Compute K steps of the exponential map on a constrained manifold given bei the approximate zero level set
    of ConstrObj.
    
    This function uses a numerical scheme that reproduces discrete variational 
    geodesics minimizing the path energy starting from point x0 with initial (discrete) velocity v.
    'LBFGS' from torch.optim is used for the inner optimization problem in each step.
    The gradient is implemented manually using mixhesssqdist to avoid second order autograd in case of pullback metrics.

    Parameters
    ----------
    ConstrObj : callable
        Constraint function that implicitly defines the manifold.
        Signature: ConstrObj(coord) -> torch.Tensor (n-d, m)
        See GeodesicSolverTorch documentation for details.
        
    ConstrGrad : callable
        Gradient (Jacobian) of the constraint function.
        Signature: ConstrGrad(coord) -> torch.Tensor of shape (n-d, m, n)
        See GeodesicSolverNumpy documentation for details.
        
    x0 : array-like, shape (n,)
        Initial point on the manifold. Should satisfy ConstrObj(x0) ≈ 0.
        
    v : array-like, shape (n,)
        Initial velocity vector in the tangent space at x0.
        The total geodesic length is approximately ||v||.
        
    K : int
        Number of discrete steps to compute along the geodesic.
        
    sqdist : callable, optional (default: sqdist_euclid)
        Squared distance function defining the metric.
        Signature: sqdist(x, y) -> torch.Tensor (scalar)
        See GeodesicSolverTorch documentation for details.
        
    gradsqdist : callable, optional (default: grad_sqdist_euclid)
        Gradient of the squared distance function.
        Signature: gradsqdist(x, y, wrt_first=False) -> torch.Tensor of shape (n,)

        
    mixhesssqdist : callable, optional (default: mixhess_sqdist_euclid)
        Mixed Hessian (second derivatives) of the squared distance function.
        Signature: mixhesssqdist(x, y, wrt_first=False) -> torch.Tensor of shape (n, n)
        
        Returns ∂²sqdist/∂x∂y, the mixed partial derivatives.
        - If wrt_first=True: first derivative wrt first argument
        - If wrt_first=False: first derivative wrt second argument
        
        For Euclidean metric:
            mixhess_sqdist_euclid(x, y, wrt_first=False) = 2*I (identity matrix)
            
    constrfact : float, optional (default: 1)
        Weight factor for the constraint penalty term.
        Depends on the accuracy of the constraint

    firstStep : {'step', 'linear'}, optional (default: 'step')
        Method for computing the first step:
        - 'step': Use exponential step with midpoint x0 + v/(2K)
        - 'linear': Use simple linear step x0 + v/K
            
    Returns
    -------
    result : ndarray, shape (K+1, n)
        Discrete exponential paths with K+1 points
    """
    v = v/K
    x = torch.zeros((K+1, x0.numel()), device=x0.device, dtype=x0.dtype)
    x[0,:] = x0
    if firstStep == 'step':
        x[1,:] = ExponentialStep(ConstrObj, ConstrGrad, x[0,:], x0 + v/2, sqdist, gradsqdist, 
                                 mixhesssqdist, constrfact=constrfact)
    if firstStep == 'linear':
        x[1,:] = x0 + v
    for i in range(1,K):
        x[i+1,:] = ExponentialStep(ConstrObj,ConstrGrad, x[i-1,:], x[i,:], sqdist, gradsqdist, 
                                   mixhesssqdist, constrfact=constrfact)

    return x.view( -1, x0.numel())

def ExponentialStep(ConstrObj, ConstrGrad, x0, x1, sqdist = sqdist_euclid_torch, gradsqdist=grad_sqdist_euclid_torch, 
                    mixhesssqdist = mixhess_sqdist_euclid_torch, constrfact=1):
    
    n = x0.numel()
    d = n - ConstrObj(x0.view(1, n)).shape[0]

    lam = torch.zeros(n - d, device=x0.device, dtype=x0.dtype)
    x_init = torch.cat([x1,lam])
    x_it  = x_init.clone().detach().requires_grad_(True)

    optimizer = torch.optim.LBFGS([x_it],lr=1, max_iter=1000, tolerance_grad=1e-5)

    def objective():
        optimizer.zero_grad()
        loss = F(x_it, ConstrObj, ConstrGrad, x0, x1, sqdist, gradsqdist, mixhesssqdist, constrfact)

        manual_grad = DF(x_it, ConstrObj, ConstrGrad, x0, x1, sqdist, gradsqdist, mixhesssqdist, constrfact)

        #manually implemented derivative that avoids second order autograd
        x_it.grad = manual_grad
        return loss
        
    optimizer.step(objective)
    x_it = x_it.detach()
    loss = F(x_it, ConstrObj, ConstrGrad, x0, x1, sqdist, gradsqdist, mixhesssqdist, constrfact)

    print('Loss: ', loss.item())
    return x_it[:n]


def F(x_it, ConstrObj, ConstrGrad, x0, x1, sqdist=sqdist_euclid_torch, gradsqdist=grad_sqdist_euclid_torch, 
      mixhesssqdist = mixhess_sqdist_euclid_torch, constrfact=1):
    
    n = x0.numel()
    x2 = x_it[:n]
    lam = x_it[n:]
    l = lam.numel()
    x1  = x1.clone().detach().requires_grad_(True)
    dphi_x1 = ConstrGrad(x1.view(-1,n))

    geod = 2*(gradsqdist(x0, x1, wrt_first = False) + gradsqdist(x1, x2, wrt_first = True))
    for i in range(l):
        geod -= lam[i]*dphi_x1[i,0,:]
    geod = 0.5*torch.norm(geod)**2 
    constr = 0.5*torch.norm( ConstrObj(x2.view(-1,n)) )**2
    return geod + constrfact*constr

def DF(x_it, ConstrObj, ConstrGrad, x0, x1, sqdist=sqdist_euclid_torch, gradsqdist=grad_sqdist_euclid_torch, 
       mixhesssqdist=mixhess_sqdist_euclid_torch, constrfact=1):
    """
    derivative of F, manually implemented to avoid second order autograd
    """
    n = x0.numel()
    x2 = x_it[:n]
    lam = x_it[n:]
    l = lam.numel()
   
    dphi_x2 = ConstrGrad(x2.view(-1,n)).detach()
    x1  = x1.clone().detach().requires_grad_(True)
    dphi_x1 = ConstrGrad(x1.view(-1,n)).detach()
    inner = 2*(gradsqdist(x0, x1, wrt_first = False) + gradsqdist(x1,x2, wrt_first = True))

    inner -= lam @ dphi_x1[:,0,:]
    mixedhess = mixhesssqdist(x1,x2, wrt_first = True).detach()
    dF_x = 2*inner @ (mixedhess)
    dF_x += constrfact*ConstrObj(x2.view(-1,n))[:,0] @ (dphi_x2[:,0,:])
    dF_x = dF_x.view(-1)   
    dF_lam =  -inner @ dphi_x1[:,0,:].T
    deriv = torch.cat( [dF_x,dF_lam])
    return deriv.detach()