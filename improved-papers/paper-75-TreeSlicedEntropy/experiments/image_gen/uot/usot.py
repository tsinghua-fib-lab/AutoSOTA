import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def minkowski_ip(x, y, keepdim=True):
    if len(x.shape)==1:
        x = x.reshape(1,-1)
    if len(y.shape)==1:
        y = y.reshape(1,-1)
    
    if x.shape[0] != y.shape[0]:
        return -x[...,0][None]*y[...,0][:,None] + torch.sum(x[...,1:][None]*y[...,1:][:,None], axis=-1)
    else:
        return (-x[...,0]*y[...,0])[:,None] + torch.sum(x[...,1:]*y[...,1:], axis=-1, keepdim=True)
    
def minkowski_ip2(x, y):
    """
        Return a n x m matrix where n and m are the number of batchs of x and y.
    """
    return -x[:,0][None]*y[:,0][:,None] + torch.sum(x[:,1:][None]*y[:,1:][:,None], axis=-1)


def lorentz_to_poincare(y, r=1):
    return r*y[...,1:]/(r+y[...,0][:,None])

# def poincare_to_lorentz(x):
#     norm_x = np.linalg.norm(x, axis=-1)[:,None]
#     return np.concatenate([1+norm_x**2, 2*x], axis=-1)/(1-norm_x**2)

#def poincare_to_lorentz(x):
    #norm_x = torch.linalg.norm(x, axis=-1)[:,None]
    #return torch.cat([1+norm_x**2, 2*x], axis=-1)/(1-norm_x**2)

def poincare_to_lorentz(x):
    norm_x = torch.norm(x, dim=-1, keepdim=True)
    return torch.cat([1+norm_x**2, 2*x], dim=-1)/(1-norm_x**2)


def sum_mobius(z, y, r=1):
    ip = torch.sum(z*y, axis=-1)
    y_norm2 = torch.sum(y**2, axis=-1)
    z_norm2 = torch.sum(z**2, axis=-1)
    num = (1+2*r*ip+r*y_norm2)[:,None]*z + (1-r*z_norm2)[:,None]*y
    denom = 1+2*r*ip+r**2*z_norm2*y_norm2
    return num/denom[:,None]

def prod_mobius(r, x):
    norm_x = torch.sum(x**2, axis=-1)**(1/2)
    return torch.tanh(r[:,None]*torch.arctanh(norm_x)) * x/norm_x

def dist_poincare(x, y, r=1):
    num = torch.linalg.norm(x-y, axis=-1)**2
    denom = (1-r*torch.linalg.norm(y, axis=-1)**2) * (1-r*torch.linalg.norm(x, axis=-1)**2)
    frac = num/denom
    return torch.arccosh(1+2*r*frac)/np.sqrt(r)

def dist_poincare2(x, y, r=1):
    num = torch.linalg.norm(x[:,None]-y[None], axis=-1)**2
    denom = (1-r*torch.linalg.norm(y, axis=-1)**2)[None] * (1-r*torch.linalg.norm(x, axis=-1)**2)[:,None]
    frac = num/denom
    return torch.arccosh(1+2*r*frac)/np.sqrt(r)


def projection(x, x0, v):
    ip_x0_x = minkowski_ip(x0, x)
    ip_v_x = minkowski_ip(v, x)
        
    if v.shape[0] != x.shape[0]:
        num = -(ip_x0_x[:,None]*x0) + ip_v_x[:,:,None]*v[None]
        denom = torch.sqrt((ip_x0_x)**2 - ip_v_x**2)[:,:,None]
    else:
        num = -ip_x0_x*x0 + ip_v_x*v
        denom = torch.sqrt((ip_x0_x)**2 - ip_v_x**2)
    proj = num/denom
    return proj


def parallelTransport(v, x0, x1):
    """
        Transport v in T_x0 H to u in T_x1 H by following the geodesics by parallel transport
    """
    n, d = v.shape
    if len(x0.shape)==1:
        x0 = x0.reshape(-1, d)
    if len(x1.shape)==1:
        x1 = x1.reshape(-1, d)
        
    u = v + minkowski_ip(x1, v)*(x0+x1)/(1-minkowski_ip(x1, x0))
    return u

def expMap(u, x):
    """
        Project u in T_x H to the surface
    """
    
    if len(x.shape)==1:
        x = x.reshape(1, -1)
    
    norm_u = minkowski_ip(u,u)**(1/2)
    y = torch.cosh(norm_u)*x + torch.sinh(norm_u) * u/norm_u
    return y

def proj_along_horosphere(x, p):
    """
        Projection along the horosphere characterized by the ideal point $p in S^{d-1}$
        and $x in mathbb{B}^d$ (the Poincaré ball)
    """
    norm_x = torch.norm(x, dim=-1, keepdim=True)**2
    dist_px = torch.norm(p-x, dim=-1, keepdim=True)**2
    lambd = (1-norm_x-dist_px)/(1-norm_x+dist_px)
    return lambd*p


def lambd(x):
    norm_x = torch.norm(x, dim=-1, keepdim=True)
    return 2/(1-norm_x**2)


def exp_poincare(v, x):
    lx = lambd(x)
    norm_v = torch.norm(v, dim=-1, keepdim=True)
    
    ch = torch.cosh(torch.clamp(lx*norm_v, min=-20, max=20))
    th = torch.tanh(lx*norm_v)
    normalized_v = v/torch.clamp(norm_v, min=1e-6)
    ip_xv = torch.sum(x*normalized_v, dim=-1, keepdim=True)

    num1 = lx * (1 + ip_xv * th) * x
    num2 = th * normalized_v
    denom = 1/ch + (lx-1) + lx*ip_xv*th
    
    return (num1+num2)/denom

def proj_horosphere_lorentz(x, x0, v):
    ip = minkowski_ip(x, x0+v)[0]
    u = (1+ip)/(1-ip)
    ch = (1+u**2)/(1-u**2)
    sh = 2*u/(1-u**2)
        
    if v.shape[0] != x.shape[0]:
        return ch[:,:,None] * x0[None] + sh[:,:,None] * v
    else:
        return ch * x0 + sh * v

def busemann_lorentz(v, z, x0):
    ip = minkowski_ip(v+x0, z)
    return torch.log(-ip)

def busemann_lorentz2(v, z, x0):
    ip = minkowski_ip2(v+x0, z)
    return torch.log(-ip)

def busemann_poincare(p, z):
    return torch.log(torch.norm(p-z, dim=-1)**2/(1-torch.norm(z,dim=-1)**2))

def busemann_poincare2(p, z):
    return torch.log(torch.norm(p[None]-z[:,None], dim=-1)**2/(1-torch.norm(z,dim=-1,keepdim=True)**2))

def generate_measure(n_sample, n_dim, slice=True):
    """
    Generate a batch of probability measures in R^d sampled over
    the unit square
    :param n_batch: Number of batches
    :param n_sample: Number of sampling points in R^d
    :param n_dim: Dimension of the feature space
    :return: A (Nbatch, Nsample, Ndim) torch.Tensor
    """
    m = torch.distributions.exponential.Exponential(1.0)
    if slice:
        a = m.sample(torch.Size([n_sample, n_dim]))
        a = a / a.sum(dim=1)[:,None]
    else:
        a = m.sample(torch.Size([n_sample]))
        a = a / a.sum()
    m = torch.distributions.uniform.Uniform(0.0, 1.0)
    x = m.sample(torch.Size([n_sample, n_dim]))
    return a, x



def sampleWrappedNormal(mu, Sigma, n):
    device = mu.device
    
    d = len(mu)
    normal = torch.distributions.MultivariateNormal(torch.zeros((d-1,), device=device),Sigma)
    x0 = torch.zeros((1,d), device=device)
    x0[0,0] = 1
    
    ## Sample in T_x0 H
    v_ = normal.sample((n,))
    v = torch.nn.functional.pad(v_, (1,0))
    
    ## Transport to T_\mu H and project on H
    u = parallelTransport(v, x0, mu)    
    y = expMap(u, mu)
    
    return y 

def cost_function(p):
    """Outputs the cost C(x-y) function used to compare the inverse cdf
    WARNING: The cost C must be a convex function.

    Args:
        p (int): Exponent of the cost C(x) = |x|^p

    Returns:
        function: torch function which takes torch.Tensor as input and output.
    """
    if p == 1:
        return torch.abs
    if p == 2:
        return torch.square
    else:
        def cost(x):
            return torch.pow(torch.abs(x), p)
        return cost



def emd1D(u_values, v_values, u_weights=None, v_weights=None,p=2, require_sort=True):
    """computes sliced-wise the p-th norm between the inverse cdf of two measures

    Args:
        u_values (torch.Tensor of size [Proj, N]): support of first measures
        v_values (torch.Tensor of size [Proj, M]): support of second measures
        u_weights (torch.Tensor of size [Proj, N]): weights of first measures. Defaults to None for uniform weights.
        v_weights (torch.Tensor of size [Proj, M]): weights of second measures. Defaults to None for uniform weights.
        p (int, optional): Exponent of cost C(x)=|x|^p. Defaults to 2.
        require_sort (bool, optional): Ask whether support must be sorted or not. Defaults to True if inputs are already sorted.

    Returns:
        loss (torch.Tensor of size [Proj]): univariate OT loss between the [Proj] pairs of measures.
    """
    proj = u_values.shape[0]
    n = u_values.shape[-1]
    m = v_values.shape[-1]

    device = u_values.device
    dtype = u_values.dtype

    # Init weights or broadcast if necessary
    if u_weights is None:
        u_weights = torch.full((proj,n), 1/n, dtype=dtype, device=device)
    elif u_weights.dim() == 1:
        u_weights = u_weights.repeat(u_values.shape[0], 1)
    assert (u_weights.dim()) == 2 and (u_values.size()== u_weights.size())


    if v_weights is None:
        v_weights = torch.full((proj,m), 1/m, dtype=dtype, device=device)
    elif v_weights.dim() == 1:
        v_weights = v_weights.repeat(v_values.shape[0], 1)
    assert (v_weights.dim()) == 2 and (v_values.size()== v_weights.size())

    # Sort w.r.t. support if not already done
    if require_sort:
        u_values, u_sorter = torch.sort(u_values, -1)
        v_values, v_sorter = torch.sort(v_values, -1)

        u_weights = torch.gather(u_weights, -1, u_sorter)
        v_weights = torch.gather(v_weights, -1, v_sorter)
        # u_weights = u_weights[..., u_sorter]
        # v_weights = v_weights[..., v_sorter]
   
    u_cdf = torch.clamp(torch.cumsum(u_weights, -1), max=1.)
    v_cdf = torch.clamp(torch.cumsum(v_weights, -1), max=1.)

    cdf_axis, _ = torch.sort(torch.cat((u_cdf, v_cdf), -1), -1)
    
    u_index = torch.searchsorted(u_cdf, cdf_axis)
    v_index = torch.searchsorted(v_cdf, cdf_axis)

    u_icdf = torch.gather(u_values, -1, u_index.clip(0, n-1))
    v_icdf = torch.gather(v_values, -1, v_index.clip(0, m-1))

    cdf_axis = torch.nn.functional.pad(cdf_axis, (1, 0))
    delta = cdf_axis[..., 1:] - cdf_axis[..., :-1]

    cost = cost_function(p)
    loss = torch.sum(delta * cost(u_icdf - v_icdf), axis=-1)
    return loss



def emd1D_dual(u_values, v_values, u_weights=None, v_weights=None, p=1, require_sort=True):
    
    proj = u_values.shape[0]
    n = u_values.shape[-1]
    m = v_values.shape[-1]

    device = u_values.device
    dtype = u_values.dtype

    # Init weights or broadcast if necessary
    if u_weights is None:
        u_weights = torch.full((proj,n), 1/n, dtype=dtype, device=device)
    elif u_weights.dim() == 1:
        u_weights = u_weights.repeat(u_values.shape[0], 1)
    assert (u_weights.dim()) == 2 and (u_values.size()== u_weights.size())


    if v_weights is None:
        v_weights = torch.full((proj,m), 1/m, dtype=dtype, device=device)
    elif v_weights.dim() == 1:
        v_weights = v_weights.repeat(v_values.shape[0], 1)
    assert (v_weights.dim()) == 2 and (v_values.size()== v_weights.size())

    # Sort w.r.t. support if not already done
    if require_sort:
        u_values, u_sorter = torch.sort(u_values, -1)
        v_values, v_sorter = torch.sort(v_values, -1)

        u_weights = torch.gather(u_weights, 1, u_sorter)
        v_weights = torch.gather(v_weights, 1, v_sorter)

    # eps trick to have strictly increasing cdf and avoid zero mass issues
    eps=1e-12
    u_cdf = torch.cumsum(u_weights + eps, -1) - eps
    v_cdf = torch.cumsum(v_weights + eps, -1) - eps

    cdf_axis, _ = torch.sort(torch.cat((u_cdf, v_cdf), -1), -1)
    
    u_index = torch.searchsorted(u_cdf, cdf_axis).clip(0, n-1)
    v_index = torch.searchsorted(v_cdf, cdf_axis).clip(0, m-1)

    u_icdf = torch.gather(u_values, -1, u_index)
    v_icdf = torch.gather(v_values, -1, v_index)

    cost = cost_function(p)
    diff_dist = cost(u_icdf - v_icdf)
    cdf_axis = torch.nn.functional.pad(cdf_axis, (1, 0))
    loss = torch.sum((cdf_axis[..., 1:] - cdf_axis[..., :-1]) * diff_dist, axis=-1)
    
    mask_u = u_index[...,1:]-u_index[...,:-1]
    mask_u = torch.nn.functional.pad(mask_u, (1, 0))
    mask_v = v_index[...,1:]-v_index[...,:-1]
    mask_v = torch.nn.functional.pad(mask_v, (1, 0))

    c1 = torch.where((mask_u[...,:-1]+mask_u[...,1:])>1,-1,0)
    c1 = torch.cumsum(c1*diff_dist[...,:-1],dim=-1)
    c1 = torch.nn.functional.pad(c1, (1, 0))

    c2 = torch.where((mask_v[...,:-1]+mask_v[...,1:])>1,-1,0)
    c2 = torch.cumsum(c2*diff_dist[...,:-1],dim=-1)
    c2 = torch.nn.functional.pad(c2, (1, 0))
    
    masked_u_dist = mask_u*diff_dist
    masked_v_dist = mask_v*diff_dist

    T = torch.cumsum(masked_u_dist-masked_v_dist,dim=-1) + c1  - c2
    tmp = mask_u.clone() # avoid in-place problem
    tmp[...,0]=1
    f = torch.masked_select(T, tmp.bool()).reshape_as(u_values)
    f[...,0]=0
    tmp = mask_v.clone() # avoid in-place problem
    tmp[...,0]=1
    g = -torch.masked_select(T, tmp.bool()).reshape_as(v_values) # TODO: Apparently buggy line (v_values/mask format unstable)
    return f, g, loss


def emd1D_dual_backprop(u_values, v_values, u_weights=None, v_weights=None, p=1, require_sort=True):
    n = u_values.shape[-1]
    m = v_values.shape[-1]
    if u_weights is None:
        mu_1 = torch.full((u_values.shape[0], u_values.shape[1]), 1/u_values.shape[1], dtype=u_values.dtype, device=u_values.device)
    elif u_weights.dim() == 1:
        mu_1 = u_weights.repeat(u_values.shape[0], 1).clone().detach()
    else:
        mu_1 = u_weights.clone().detach()
    assert u_values.dim() == mu_1.dim()

    if v_weights is None:
        mu_2 = torch.full((v_values.shape[0], v_values.shape[1]), 1/v_values.shape[1], dtype=v_values.dtype, device=v_values.device)
    elif v_weights.dim() == 1:
        mu_2 = v_weights.repeat(v_values.shape[0], 1).clone().detach()
    else:
        mu_2 = v_weights.clone().detach()
    assert v_values.dim() == mu_2.dim()
    
    mu_1.requires_grad_(True)
    mu_2.requires_grad_(True)
    value = emd1D(u_values, v_values, u_weights=mu_1, v_weights=mu_2, p=p, require_sort=require_sort).sum()
    value.backward()

    return mu_1.grad, mu_2.grad, value # value can not be backward anymore


def logsumexp(f, a):
    # stabilized
    assert f.dim() == a.dim()
    if f.dim() > 1:
        xm = torch.amax(f + torch.log(a),dim=1).reshape(-1,1)
        return xm + torch.log(torch.sum(torch.exp(f + torch.log(a) - xm),dim=1)).reshape(-1,1)
    else:
        xm = torch.amax(f + torch.log(a))
        return xm + torch.log(torch.sum(torch.exp(f + torch.log(a) - xm)))


def rescale_potentials(f, g, a, b, rho1, rho2):
    tau = (rho1 * rho2) / (rho1 + rho2)
    transl = tau * (logsumexp(-f / rho1, a) - logsumexp(-g / rho2, b))
    return transl


def kullback_leibler(a, b):
    return (a * (a/b +1e-12).log()).sum(dim=-1) - a.sum(dim=-1) + b.sum(dim=-1)


def sample_projections(num_features, num_projections, dummy_data, type_proj="linear", seed_proj=None):
    if seed_proj is not None:
        torch.manual_seed(seed_proj)
        
    if type_proj == "linear" or type_proj == "poincare_horo":
        projections = torch.normal(mean=torch.zeros([num_features, num_projections]), std=torch.ones([num_features, num_projections])).type(dummy_data.dtype).to(dummy_data.device)
        projections = F.normalize(projections, p=2, dim=0)
    elif type_proj == "lorentz_geod" or type_proj == "lorentz_horo":
        vs = np.random.normal(size=(num_projections, num_features-1))
        vs = F.normalize(torch.from_numpy(vs), p=2, dim=-1).type(dummy_data.dtype).to(dummy_data.device)
        projections = F.pad(vs, (1,0))
        
    return projections


def project_support(x, y, projections, type_proj="linear"):
    if type_proj == "linear":
        x_proj = (x @ projections).T
        y_proj = (y @ projections).T
        
    elif type_proj == "lorentz_geod":
        n_proj, d = projections.shape

        x0 = torch.zeros((1,d), device=x.device)
        x0[0,0] = 1
        
        ip_x0_x = minkowski_ip(x0, x)
        ip_v_x = minkowski_ip(projections, x)

        ip_x0_y = minkowski_ip(x0, y)
        ip_v_y = minkowski_ip(projections, y)

        x_proj = torch.arctanh(-ip_v_x/ip_x0_x).reshape(-1, n_proj).T
        y_proj = torch.arctanh(-ip_v_y/ip_x0_y).reshape(-1, n_proj).T
        
    elif type_proj == "lorentz_horo":
        n_proj, d = projections.shape

        x0 = torch.zeros((1,d), device=x.device)
        x0[0,0] = 1
        
        x_proj = busemann_lorentz2(projections, x, x0).reshape(-1, n_proj).T
        y_proj = busemann_lorentz2(projections, y, x0).reshape(-1, n_proj).T
        
    elif type_proj == "poincare_horo":
        d, n_proj = projections.shape

        x_proj = busemann_poincare2(projections.T, x).reshape(-1, n_proj).T
        y_proj = busemann_poincare2(projections.T, y).reshape(-1, n_proj).T
        
    return x_proj, y_proj


def sort_support(x_proj):
    x_sorted, x_sorter = torch.sort(x_proj, -1)
    x_rev_sort = torch.argsort(x_sorter, dim=-1)
    return x_sorted, x_sorter, x_rev_sort


def sample_project_sort_data(x, y, num_projections, type_proj="linear", seed_proj=None, projections=None):
    num_features = x.shape[1] # data dim

    # Random projection directions, shape (num_features, num_projections)
    if projections is None:
        projections = sample_projections(num_features, num_projections, dummy_data=x, seed_proj=seed_proj, type_proj=type_proj)

    # 2 ---- Project samples along directions and sort
    x_proj, y_proj = project_support(x, y, projections, type_proj)
    x_sorted, x_sorter, x_rev_sort = sort_support(x_proj)
    y_sorted, y_sorter, y_rev_sort = sort_support(y_proj)
    return x_sorted, x_sorter, x_rev_sort, y_sorted, y_sorter, y_rev_sort, projections

    

def sliced_unbalanced_ot(a, b, x, y, p, num_projections, rho1, rho2=None, niter=10, mode='backprop',
                         seed_proj=None, type_proj="linear", projections=None):
    """
        Compute SUOT

        Parameters
        ----------
        a: tensor, shape (n_samples_a,), weights in the source domain
        b: tensor, shape (n_samples_b,), weights in the target domain
        x: tensor, shape (n_samples_a, d), samples in the source domain
        y: tensor, shape (n_samples_b, d), samples in the target domain
        p: float, power
        num_projections: int, number of projections
        rho1: float, first marginal relaxation term
        rho2: float, second marginal relaxation term (default = rho1)
        niter: int, number of Frank-Wolfe algorithm
        mode: "backprop" or "icdf", how to compute the potentials
        seed_proj
        type_proj: "linear", "lorentz_geod", "lorentz_horo" or "poincare_horo": Euclidean or hyperbolic projection
        projections: shape (d, num_projections), by default None and sample projections
    """
    if rho2 is None:
        rho2 = rho1
    assert mode in ['backprop', 'icdf']

    # 1 ---- draw some random directions
    x_sorted, x_sorter, x_rev_sort, y_sorted, y_sorter, y_rev_sort, projections = sample_project_sort_data(x, y, num_projections, type_proj, seed_proj, projections)
    a = a[..., x_sorter]
    b = b[..., y_sorter]

    # 3 ----- Prepare and start FW

    # Initialize potentials
    f = torch.zeros_like(a)
    g = torch.zeros_like(b)

    for k in range(niter):
        # Output FW descent direction
        transl = rescale_potentials(f, g, a, b, rho1, rho2)

        # translate potentials
        f = f + transl
        g = g - transl
        # update measures
        A = a * torch.exp(-f / rho1)
        B = b * torch.exp(-g / rho2)
        # solve for new potentials
        if mode == 'icdf':
            fd, gd, loss = emd1D_dual(x_sorted, y_sorted, u_weights=A, v_weights=B, p=p, require_sort=False)
        if mode == 'backprop':
            fd, gd, loss = emd1D_dual_backprop(x_sorted, y_sorted, u_weights=A, v_weights=B, p=p, require_sort=False)
        # default step for FW
        t = 2. / (2. + k)

        f = f + t * (fd - f)
        g = g + t * (gd - g)

    # 4 ----- We are done. Get me out of here !
    # Last iter before output
    transl = rescale_potentials(f, g, a, b, rho1, rho2)
    f, g = f + transl, g - transl
    A, B = a * torch.exp(-f / rho1), b * torch.exp(-g / rho2)
    loss = torch.mean(emd1D(x_sorted, y_sorted, u_weights=A, v_weights=B, p=p, require_sort=False))
    loss = loss + rho1 * torch.mean(kullback_leibler(A, a)) + rho2 * torch.mean(kullback_leibler(B, b))

    # Reverse sort potentials and measures w.r.t order not sample (not sorted)
    f, g = torch.gather(f, 1, x_rev_sort), torch.gather(g, 1, y_rev_sort)
    A, B = torch.gather(A, 1, x_rev_sort), torch.gather(B, 1, y_rev_sort)
    
    return loss, f, g, A, B, projections



def unbalanced_sliced_ot(a, b, x, y, p, num_projections, rho1, rho2=None, niter=10,
                         mode='backprop', stochastic_proj=False, seed_proj=None, type_proj="linear", projections=None):
    """
        Compute USOT

        Parameters
        ----------
        a: tensor, shape (n_samples_a,), weights in the source domain
        b: tensor, shape (n_samples_b,), weights in the target domain
        x: tensor, shape (n_samples_a, d), samples in the source domain
        y: tensor, shape (n_samples_b, d), samples in the target domain
        p: float, power
        num_projections: int, number of projections
        rho1: float, first marginal relaxation term
        rho2: float, second marginal relaxation term (default =rho1)
        niter: int, number of Frank-Wolfe algorithm
        mode: "backprop" or "icdf", how to compute the potentials
        seed_proj
        type_proj: "linear", "lorentz_geod", "lorentz_horo" or "poincare_horo": Euclidean or hyperbolic projection
        projections: shape (d, num_projections), by default None and sample projections
    """
    if rho2 is None:
        rho2 = rho1
        
    assert mode in ['backprop', 'icdf']

    # 1 ---- draw some random directions
    if not stochastic_proj:
        x_sorted, x_sorter, x_rev_sort, y_sorted, y_sorter, y_rev_sort, projections = sample_project_sort_data(x, y, num_projections, type_proj, seed_proj, projections)

    # 3 ----- Prepare and start FW

    # Initialize potentials - WARNING: They correspond to non-sorted samples
    f = torch.zeros(x.shape[0], dtype=a.dtype, device=a.device)
    g = torch.zeros(y.shape[0], dtype=a.dtype, device=a.device)

    for k in range(niter):
        # Output FW descent direction
        # translate potentials
        transl = rescale_potentials(f, g, a, b, rho1, rho2)
        f = f + transl
        g = g - transl

        # If stochastic version then sample new directions and re-sort data
        if stochastic_proj:
            x_sorted, x_sorter, x_rev_sort, y_sorted, y_sorter, y_rev_sort, projections = sample_project_sort_data(x, y, num_projections, type_proj, seed_proj)

        # update measures
        A = (a * torch.exp(-f / rho1))[..., x_sorter]
        B = (b * torch.exp(-g / rho2))[..., y_sorter]
        
        # solve for new potentials
        if mode == 'icdf':
            fd, gd, loss = emd1D_dual(x_sorted, y_sorted, u_weights=A, v_weights=B, p=p, require_sort=False)
        if mode == 'backprop':
            fd, gd, loss = emd1D_dual_backprop(x_sorted, y_sorted, u_weights=A, v_weights=B, p=p, require_sort=False)
        # default step for FW
        t = 2. / (2. + k)
        f = f + t * (torch.mean(torch.gather(fd, 1, x_rev_sort), dim=0) - f)
        g = g + t * (torch.mean(torch.gather(gd, 1, y_rev_sort), dim=0) - g)

    # 4 ----- We are done. Get me out of here !
    # Last iter before output
    transl = rescale_potentials(f, g, a, b, rho1, rho2)
    f, g = f + transl, g - transl
    A, B = (a * torch.exp(-f / rho1))[..., x_sorter], (b * torch.exp(-g / rho2))[..., y_sorter]
    loss = torch.mean(emd1D(x_sorted, y_sorted, u_weights=A, v_weights=B, p=p, require_sort=False))
    A, B = a * torch.exp(-f / rho1), b * torch.exp(-g / rho2)
    loss = loss + rho1 * kullback_leibler(A, a) + rho2 * kullback_leibler(B, b)
    
    return loss, f, g, A, B, projections
    

def sliced_ot(a, b, x, y, p, num_projections, niter=10, mode='backprop', stochastic_proj=False, seed_proj=None, type_proj="linear"):
    """
        Compute SOT

        Parameters
        ----------
        a: tensor, shape (n_samples_a,), weights in the source domain
        b: tensor, shape (n_samples_b,), weights in the target domain
        x: tensor, shape (n_samples_a, d), samples in the source domain
        y: tensor, shape (n_samples_b, d), samples in the target domain
        p: float, power
        num_projections: int, number of projections
        niter: int, number of Frank-Wolfe algorithm
        mode: "backprop" or "icdf", how to compute the potentials
        seed_proj
        type_proj: "linear", "lorentz_geod", "lorentz_horo" or "poincare_horo": Euclidean or hyperbolic projection
    """
    assert mode in ['backprop', 'icdf']

    # 1 ---- draw some random directions
    if not stochastic_proj:
        x_sorted, x_sorter, x_rev_sort, y_sorted, y_sorter, y_rev_sort, projections = sample_project_sort_data(x, y, num_projections, type_proj, seed_proj)

    # 3 ----- Prepare and start FW

    # Initialize potentials - WARNING: They correspond to non-sorted samples
    f = torch.zeros(x.shape[0], dtype=a.dtype, device=a.device)
    g = torch.zeros(y.shape[0], dtype=a.dtype, device=a.device)

    # Output FW descent direction

    # If stochastic version then sample new directions and re-sort data
    if stochastic_proj:
        x_sorted, x_sorter, x_rev_sort, y_sorted, y_sorter, y_rev_sort, projections = sample_project_sort_data(x, y, num_projections, type_proj, seed_proj)

    # update measures
    A = a[..., x_sorter]
    B = b[..., y_sorter]
    
    # solve for new potentials
    if mode == 'icdf':
        fd, gd, loss = emd1D_dual(x_sorted, y_sorted, u_weights=A, v_weights=B, p=p, require_sort=False)
    if mode == 'backprop':
        fd, gd, loss = emd1D_dual_backprop(x_sorted, y_sorted, u_weights=A, v_weights=B, p=p, require_sort=False)
    # default step for FW
    f = torch.mean(torch.gather(fd, 1, x_rev_sort), dim=0)
    g = torch.mean(torch.gather(gd, 1, y_rev_sort), dim=0)

    # 4 ----- We are done. Get me out of here !
    loss = torch.mean(emd1D(x_sorted, y_sorted, u_weights=A, v_weights=B, p=p, require_sort=False))
    
    return loss, f, g, projections


def __main__():

    a = torch.ones(100)/100
    b = torch.ones(100)/100

    Xs = torch.randn((100, 2))
    Xt = torch.randn((100, 2))

    usot, _, _, a_USOT, b_USOT, _ = unbalanced_sliced_ot(a, b, Xs, Xt, p=2, num_projections=500, rho1=1, rho2=1, niter=10)
    suot, _, _, a_SUOT, b_SUOT, _ = sliced_unbalanced_ot(a, b, Xs, Xt, p=2, num_projections=500, rho1=1, rho2=1, niter=10)

    print(usot)
    print(suot)

if __name__ == "__main__":
    __main__()