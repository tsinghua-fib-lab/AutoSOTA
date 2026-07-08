"""Gaussian kernel functions"""

import numpy as np
from jax import numpy as jnp
from jax import vmap, lax
import scipy.spatial.distance as distance
from scipy import stats
from sklearn.kernel_approximation import RBFSampler


def k(x, y, l):
    """Gaussian kernel with numpy"""

    return np.exp(-(1 / (2 * l**2)) * distance.cdist(x, y, "sqeuclidean"))


def sqeuclidean_distance(x, y):
    return jnp.sum((x - y) ** 2)


def rbf_kernel(x, y, l):
    return jnp.exp(-(1 / (2 * l**2)) * sqeuclidean_distance(x, y))


def k_jax(x, y, l):
    """Gaussian kernel compatible with JAX library"""

    x = x.astype("float64")
    y = y.astype("float64")
    mapx1 = vmap(lambda x, y: rbf_kernel(x, y, l), in_axes=(0, None), out_axes=0)
    mapx2 = vmap(lambda x, y: mapx1(x, y), in_axes=(None, 0), out_axes=1)

    # kernel
    K = mapx2(x, y)

    return K

def k_fourier(x,l, seed):
    g = 1 / (2 * l**2)
    rbf_feature = RBFSampler(gamma=g, random_state=seed)
    X_features = rbf_feature.fit_transform(x)
    K = np.dot(X_features, X_features.T)
    return K

def k_comp(x, y):
    """Composition of Gaussian kernels with different lengthscale parameters"""

    l_range = np.array(
        [1.0, 10.0, 20.0, 40.0, 80.0, 100.0, 130.0, 200.0, 400.0, 800.0, 1000.0]
    )
    n = len(x)
    m = len(y)
    k_gaus = np.zeros((n, m))
    for l in l_range:
        k_gaus += k_jax(x, y, l)

    return k_gaus

def mat_decomp_jax(K):
    """Function for matrix decomposition"""
    rank = jnp.linalg.matrix_rank(K)
    # Check if the matrix is singular
    is_singular = rank < min(K.shape)
    if is_singular:
        # print('warning, Gram matrix K is singular')
        d, v = jnp.linalg.eigh(K) #L == U*diag(d)*U'. the scipy function forces real eigs
        d = jnp.where(d < 0, 0, d) # get rid of small eigs
        L = v @ jnp.diag(jnp.sqrt(d))
    else:
        L = lax.linalg.cholesky(K)

    return L