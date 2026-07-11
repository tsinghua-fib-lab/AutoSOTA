import jax.numpy as jnp
import jax
from jaxtyping import Array
from functools import partial
from jax import vmap

def _rescale(x: Array, scale: Array) -> Array:
    return x / scale

def _l2_norm_squared(x: Array) -> Array:
    return jnp.sum(jnp.square(x))

class gaussian_kernel:
    def __init__(self, sigma: float):
        self.sigma = sigma

    def __call__(self, x: Array, y: Array) -> Array:
        return jnp.exp(- 0.5 * _l2_norm_squared(_rescale(x - y, self.sigma)))

    def make_distance_matrix(self, X: Array, Y: Array) -> Array:
        return vmap(vmap(type(self).__call__, (None, None, 0)), (None, 0, None))(
            self, X, Y
        )
    
    def mean_embedding(self, X: Array, mu: Array, Sigma: Array) -> Array:
        """
        The implementation of the kernel mean embedding of the RBF kernel with Gaussian distribution
        A fully vectorized implementation.

        Args:
            mu: Gaussian mean, (D, )
            Sigma: Gaussian covariance, (D, D)
            X: (M, D)

        Returns:
            kernel mean embedding: (M, )
        """
        kme_RBF_Gaussian_func_ = partial(kme_RBF_Gaussian_func, mu, Sigma, self.sigma)
        if X.ndim == 1:
            # Handle inputs of shape (D,)
            return kme_RBF_Gaussian_func_(X)
        if X.ndim == 2:
            # Handle inputs of shape (B, D)
            kme_RBF_Gaussian_vmap_func = jax.vmap(kme_RBF_Gaussian_func_)
            return kme_RBF_Gaussian_vmap_func(X)
        if X.ndim == 3:
            # Handle inputs of shape (M, B, D)
            kme_RBF_Gaussian_vmap_func = jax.vmap(jax.vmap(kme_RBF_Gaussian_func_))
            return kme_RBF_Gaussian_vmap_func(X)
    
    def mean_mean_embedding(self, mu1, mu2, Sigma1, Sigma2) -> float:
        return kme_double_RBF_Gaussian(mu1, mu2, Sigma1, Sigma2, self.sigma)

@jax.jit
def kme_RBF_Gaussian_func(mu, Sigma, l, y):
    """
    The implementation of the kernel mean embedding of the RBF kernel with Gaussian distribution.
    Not vectorized.

    Args:
        mu: Gaussian mean, (D, )
        Sigma: Gaussian covariance, (D, D)
        y: (D, )
        l: float

    Returns:
        kernel mean embedding: scalar
    """
    D = mu.shape[0]
    l_ = l ** 2
    Lambda = jnp.eye(D) * l_
    Lambda_inv = jnp.eye(D) / l_
    part1 = jnp.linalg.det(jnp.eye(D) + Sigma @ Lambda_inv)
    part2 = jnp.exp(-0.5 * (mu - y).T @ jnp.linalg.inv(Lambda + Sigma) @ (mu - y))
    return part1 ** (-0.5) * part2


@jax.jit
def kme_double_RBF_Gaussian(mu_1, mu_2, Sigma_1, Sigma_2, l):
    """
    Computes the double integral a gaussian kernel with lengthscale l, with two different Gaussians.
    
    Args:
        mu_1, mu_2: (D,) 
        Sigma_1, Sigma_2: (D, D)
        l : scalar

    Returns:
        A scalar: the value of the integral.
    """
    D = mu_1.shape[0]
    l_ = l ** 2
    Lambda = jnp.eye(D) * l_
    sum_ = Sigma_1 + Sigma_2 + Lambda
    part_1 = jnp.sqrt(jnp.linalg.det(Lambda) / jnp.linalg.det(sum_))
    sum_inv = jnp.linalg.inv(sum_)
    # Compute exponent: - (1/2) * mu^T * (Σ1 + Σ2 + Lambda)⁻¹ * Γ⁻¹ * mu
    exp_term = -0.5 * ((mu_1 - mu_2).T @ sum_inv @ (mu_1 - mu_2))
    exp_value = jnp.exp(exp_term)
    result = part_1 * exp_value
    return result

def compute_mmd2(x, y, bandwidth=1.0):
    """Compute unbiased squared MMD between two sets of samples x, y."""
    from utils.kernel import gaussian_kernel
    kernel = gaussian_kernel(bandwidth)
    Kxx = kernel.make_distance_matrix(x, x)
    Kyy = kernel.make_distance_matrix(y, y)
    Kxy = kernel.make_distance_matrix(x, y)

    n = x.shape[0]
    m = y.shape[0]

    sum_Kxx = jnp.sum(Kxx) / (n * n)
    sum_Kyy = jnp.sum(Kyy) / (m * m)
    sum_Kxy = jnp.sum(Kxy) / (n * m)

    mmd2 = sum_Kxx + sum_Kyy - 2 * sum_Kxy
    return mmd2

class GradientKernel:
    def __init__(self, S_PQ, k):
        self.S_PQ = S_PQ
        self.k = jax.jit(k)
        self.dkx = jax.jit(jax.jacrev(self.k, argnums=0))
        self.dky = jax.jit(jax.jacrev(self.k, argnums=1))
        self.d2k = jax.jit(jax.jacfwd(self.dky, argnums=0))

        self.K = lambda X : jax.vmap(lambda x: jax.vmap(lambda y: self.k(x, y))(X))(X)
        self.dK1 = lambda X : jax.vmap(lambda x: jax.vmap(lambda y: self.dkx(x, y))(X))(X)
        self.d2K = lambda X : jax.vmap(lambda x: jax.vmap(lambda y: jnp.trace(self.d2k(x, y)))(X))(X)

        #for extensible sampling
        self.Kx = lambda X,x : jax.vmap(lambda y: self.k(x, y))(X)
        self.dK1x = lambda X,x : jax.vmap(lambda y: self.dkx(x, y))(X)
        self.dK2x = lambda X,x : jax.vmap(lambda y : self.dky(x, y))(X)
        self.d2Kx = lambda X,x : jax.vmap(lambda y: jnp.trace(self.d2k(x, y)))(X)

    def gram_matrix(self,X): #Gram_matrix
        K = self.K(X)
        dK = self.dK1(X)
        d2K = self.d2K(X)
        S_PQ = self.S_PQ(X)
        S_dK = jnp.einsum('ijk, ijk -> ij', dK, (S_PQ[None, :, :]))
        k_pq = d2K + S_dK + S_dK.T + K * jnp.dot(S_PQ, S_PQ.T)
        return k_pq
    



class KernelGradientDiscrepancy:
    def __init__(self, k_pq):
        self.K_pq = k_pq.gram_matrix
        self.k_pq = k_pq

    def evaluate(self, X):
        n = len(X)
        K_pq = self.K_pq(X)
        sum = 1/n * jnp.sqrt(jnp.sum(K_pq))
        return sum
    
    def square_kgd(self, X):
        n = len(X)
        K_pq = self.K_pq(X)
        sum = jnp.mean(K_pq)
        return sum
    
    def kde_KGD(self,X,num_samples=100):
        n,d = X.shape
        bandwidth = 1/jnp.sqrt(n)  
        key = jax.random.PRNGKey(0)
        key, key_idx, key_noise = jax.random.split(key, 3)
        indices = jax.random.choice(key_idx, n, shape=(num_samples,), replace=True)
        samples = jax.random.normal(key_noise, (num_samples, d)) * bandwidth + X[indices]
        samples = jax.device_put(samples, X.device)
        return self.evaluate(samples)


def k_imq(x, y, c, b, scale=1.):
    assert b > 0
    return (c**2 + (x-y).dot(x-y)/scale**2)**(-b)


def k_lin (x,y,c):
    return jnp.dot(x,y) + c**2

def a_(x,s,c):
    return (c**2 + jnp.sum((x)**2))**(s/2)

def recommended_kernel(x,y,L,alpha,beta,c):
    a_s_x = a_(x,alpha - beta,c)
    a_s_y = a_(y,alpha - beta,c)
    k_lin_xy = k_lin(x,y,c)/(k_lin(x,x,c)*k_lin(y,y,c))**0.5
    return a_s_x*(L(x,y) + k_lin_xy)*a_s_y



class Distribution:
    def __init__(self, kernel):
        """
        A class that supports Mixture of Gaussians distributions.
        """
        self.kernel = kernel
        self.means = jnp.load('data/mog_means.npy') 
        self.covariances = jnp.load('data/mog_covs.npy')
        self.k, self.d = self.means.shape
        self.weights = jnp.ones(self.k) / self.k

    def mean_embedding(self, Y):
        # Vectorized computation using vmap
        kme_values = jax.vmap(self.kernel.mean_embedding, in_axes=(None, 0, 0))(Y, self.means, self.covariances)
        kme = jnp.tensordot(self.weights, kme_values, axes=1)
        return kme
    
    def mean_mean_embedding(self):
        if self.k == 1:
            double_kme = self.kernel.mean_mean_embedding(self.means[0], self.covariances[0])
            return double_kme
        else:
            double_kme = 0
            for i in range(self.k):
                for j in range(self.k):
                    double_kme += self.weights[i] * self.weights[j] * self.kernel.mean_mean_embedding(self.means[i], self.means[j], self.covariances[i], self.covariances[j])
            return double_kme
    
    def sample(self, sample_size, rng_key):
        """
        Sample i.i.d from the mixture of Gaussians.

        Parameters:
        - sample_size: int, the number of samples to draw.
        - rng_key: JAX PRNGKey for reproducibility.

        Returns:
        - samples: (sample_size, d) array of samples.
        """
        rng_key, _ = jax.random.split(rng_key)
        component_indices = jax.random.choice(rng_key, self.k, shape=(sample_size,), p=self.weights)

        means = self.means[component_indices, :]
        covs = self.covariances[component_indices, :, :]

        def sample_gaussian(mean, cov, key):
            return jax.random.multivariate_normal(key, mean, cov)

        subkeys = jax.random.split(rng_key, sample_size)
        samples = jax.vmap(sample_gaussian)(means, covs, subkeys)
        return samples

    def pdf(self, Y):
        """
        Compute the probability density function of the mixture of Gaussians.

        Parameters:
        - Y: (n, d) array of points to evaluate the PDF at.

        Returns:
        - pdf: (n,) array of PDF values.
        """
        pdf = jnp.zeros(len(Y))
        for i in range(self.k):
            pdf += self.weights[i] * jax.scipy.stats.multivariate_normal.pdf(Y, self.means[i], self.covariances[i])
        return pdf