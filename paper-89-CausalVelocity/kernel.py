import jax
import jax.numpy as jnp

def jnp_euclidean_cdist_(x,z):
    """
    https://github.com/jax-ml/jax/discussions/11841
    (x_ij): n_1 x d
    (z_ik): n_2 x d 
    out: K(x_ij, z_ik) (n x n_1 x n_2)
    """
    return (((x[:, None, :] - z[None, :, :])**2).sum(-1))

def jnp_euclidean_cdist(x,z):
    """
    Handle input shapes for jnp_euclidean_cdist_. 
    If x or z are 1d, add a trailing dimension.
    """
    if x.ndim == 1:
        x = x[:, None]
    if z.ndim == 1:
        z = z[:, None]
    out = jnp_euclidean_cdist_(x,z)
    return out

@jax.jit
def gauss_kernel_gram(x, z, lengthscale, scale):
    """
    Computes the Gaussian kernel function k(x, x') = exp(-||x - x'||^2 / (2 * lengthscale^2))
    (x_ij): n x n_1 x d
    (z_ik): n x n_2 x d 
    out: K(x_ij, z_ik) (n x n_1 x n_2)

    lengthscale: scalar, or optionally a d-dim array
    """
    k_xx = jnp.exp(-0.5*jnp_euclidean_cdist(x/lengthscale, z/lengthscale))
    return k_xx*scale

@jax.jit
def exp_kernel_gram(x, z, lengthscale, scale):
    """
    Computes the exponential kernel function k(x, x') = exp(-||x - x'|| / (2 * lengthscale))

    (x_ij): n x n x d 
    (z_ik): n x n x d 
    out: K(x_ij, z_ik) (n x n x n)
    lengthscale: scalar, or optionally a d-dim array
    """
    k_xx = jnp.exp(-0.5*jnp_euclidean_cdist(x/lengthscale, z/lengthscale)**0.5)
    return k_xx*scale

def median_heuristic(x, box = True):
    """
    Returns median heuristic lengthscale, which is the median of pairwise 2-norms of the data.
    """
    # Median heurstic for inputs
    if box:
        lengthscale = jnp.array([])
        for _ in range(x.shape[-1]):
            dist = jnp_euclidean_cdist(x[:,_],x[:,_])**0.5
            lower_tri = (jnp.tril(dist, k=-1)).flatten()
            lower_tri = jnp.where(lower_tri == 0, jnp.nan, lower_tri)
            lengthscale = jnp.concatenate([lengthscale, jnp.nanmedian(lower_tri, keepdims = True)])
    else:
        dist = jnp_euclidean_cdist(x,x)**0.5
        lower_tri = (jnp.tril(dist, k=-1)).flatten()
        lower_tri = jnp.where(lower_tri == 0, jnp.nan, lower_tri)
        lengthscale = jnp.nanmedian(lower_tri, keepdims = True)
    return lengthscale

def silverman(x):
    """
    Returns Silverman's rule of thumb for Kernel bandwidth with unit variance
    """
    n, d = x.shape
    if d == 1:
        IQR = jnp.percentile(x, 75) - jnp.percentile(x, 25)
        A = min(jnp.std(x), IQR/1.34)
        return 1.06 * A * (n)**(-1/5.)
        # return 1.06 * A * n**(-1/10.)
    else: 
        lengthscale = jnp.array([])
        for i in range(d):
            IQR = jnp.percentile(x[:,i], 75) - jnp.percentile(x[:,i], 25)
            A = min(jnp.std(x[:,i]), IQR/1.34)
            lengthscale = jnp.concatenate([
                lengthscale, 
                jnp.array(
                    [1.06 * A * ((n) * (d + 2) / 4.)**(-1. / (d + 4))]
                    )])
            # lengthscale = jnp.concatenate([
            #     lengthscale, 
            #     jnp.array(
            #         [1.06 * A * (n * (d + 2) / 4.)**(-1. / (d + 9))]
            #         )])
        return lengthscale

def silverman_log(x):
    """
    Returns Silverman's rule of thumb for Kernel bandwidth with unit variance replacing n with log(n)
    """
    n, d = x.shape
    if d == 1:
        IQR = jnp.percentile(x, 75) - jnp.percentile(x, 25)
        A = min(jnp.std(x), IQR/1.34)
        return 1.06 * A * jnp.log(n)**(-1/5.)
        # return 1.06 * A * n**(-1/10.)
    else: 
        lengthscale = jnp.array([])
        for i in range(d):
            IQR = jnp.percentile(x[:,i], 75) - jnp.percentile(x[:,i], 25)
            A = min(jnp.std(x[:,i]), IQR/1.34)
            lengthscale = jnp.concatenate([
                lengthscale, 
                jnp.array(
                    [1.06 * A * (jnp.log(n) * (d + 2) / 4.)**(-1. / (d + 4))]
                    )])
            # lengthscale = jnp.concatenate([
            #     lengthscale, 
            #     jnp.array(
            #         [1.06 * A * (n * (d + 2) / 4.)**(-1. / (d + 9))]
            #         )])
        return lengthscale
    
def silverman_sqrt(x):
    """
    Returns Silverman's rule of thumb for Kernel bandwidth with unit variance replacing n with n**(1/2)
    """
    n, d = x.shape
    if d == 1:
        IQR = jnp.percentile(x, 75) - jnp.percentile(x, 25)
        A = min(jnp.std(x), IQR/1.34)
        return 1.06 * A * n**(-1/10.)
    else: 
        lengthscale = jnp.array([])
        for i in range(d):
            IQR = jnp.percentile(x[:,i], 75) - jnp.percentile(x[:,i], 25)
            A = min(jnp.std(x[:,i]), IQR/1.34)
            lengthscale = jnp.concatenate([
                lengthscale, 
                jnp.array(
                    [1.06 * A * ((n ** (1/2)) * (d + 2) / 4.)**(-1. / (d + 4))]
                    )])
            # lengthscale = jnp.concatenate([
            #     lengthscale, 
            #     jnp.array(
            #         [1.06 * A * (n * (d + 2) / 4.)**(-1. / (d + 9))]
            #         )])
        return lengthscale

def eb_mx(x):
    """
    Returns Wibisino's Kernel bandwidth 
    """
    n, d = x.shape
    lengthscale = (d**3 * jnp.log(n)**(d/2) / jnp.log(n))**(2/(d+4))
    if d == 1:
        return lengthscale
    else:
        return jnp.array([lengthscale]*d)


