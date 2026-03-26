"""
Hilbert Schmidt Information Criterion with a Gaussian kernel, based on the
following references
[1]: https://link.springer.com/chapter/10.1007/11564089_7
[2]: https://www.researchgate.net/publication/301818817_Kernel-based_Tests_for_Joint_Independence

"""
import numpy
import jax.numpy as jnp


def centering(M):
    """
    Calculate the centering matrix
    """
    n = M.shape[0]
    unit = numpy.ones([n, n])
    identity = numpy.eye(n)
    H = identity - unit/n

    return numpy.matmul(M, H)

def jnp_centering(M):
    """
    Calculate the centering matrix
    """
    n = M.shape[0]
    unit = numpy.ones([n, n])
    identity = numpy.eye(n)
    H = identity - unit/n

    return jnp.matmul(M, H)

def gaussian_grammat(x, sigma=None):
    """
    Calculate the Gram matrix of x using a Gaussian kernel.
    If the bandwidth sigma is None, it is estimated using the median heuristic:
    ||x_i - x_j||**2 = 2 sigma**2
    """
    try:
        x.shape[1]
    except IndexError:
        x = x.reshape(x.shape[0], 1)

    xxT = numpy.matmul(x, x.T)
    xnorm = numpy.diag(xxT) - xxT + (numpy.diag(xxT) - xxT).T
    if sigma is None:
        mdist = numpy.median(xnorm[xnorm!= 0])
        sigma = numpy.sqrt(mdist*0.5)


   # --- If bandwidth is 0, add machine epsilon to it
    if sigma==0:
        eps = 7./3 - 4./3 - 1
        sigma += eps

    KX = - 0.5 * xnorm / sigma / sigma
    numpy.exp(KX, KX)
    return KX

def jnp_gaussian_grammat(x, sigma=None):
    """
    Calculate the Gram matrix of x using a Gaussian kernel.
    If the bandwidth sigma is None, it is estimated using the median heuristic:
    ||x_i - x_j||**2 = 2 sigma**2
    """
    try:
        x.shape[1]
    except IndexError:
        x = x.reshape(x.shape[0], 1)

    xxT = jnp.matmul(x, x.T)
    xnorm = jnp.diag(xxT) - xxT + (jnp.diag(xxT) - xxT).T
    if sigma is None:
        mdist = jnp.median(xnorm[xnorm!= 0])
        sigma = jnp.sqrt(mdist*0.5)


   # --- If bandwidth is 0, add machine epsilon to it
    if sigma==0:
        eps = 7./3 - 4./3 - 1
        sigma += eps

    KX = - 0.5 * xnorm / sigma / sigma
    KX = jnp.exp(KX)
    return KX

def dHSIC_calc(K_list):
    """
    Calculate the HSIC estimator in the general case d > 2, as in
    [2] Definition 2.6
    """
    if not isinstance(K_list, list):
        K_list = list(K_list)

    n_k = len(K_list)

    length = K_list[0].shape[0]
    term1 = 1.0
    term2 = 1.0
    term3 = 2.0/length

    for j in range(0, n_k):
        K_j = K_list[j]
        term1 = numpy.multiply(term1, K_j)
        term2 = 1.0/length/length*term2*numpy.sum(K_j)
        term3 = 1.0/length*term3*K_j.sum(axis=0)

    term1 = numpy.sum(term1)
    term3 = numpy.sum(term3)
    dHSIC = (1.0/length)**2*term1+term2-term3
    return dHSIC

def HSIC(x, y):
    """
    Calculate the HSIC estimator for d=2, as in [1] eq (9)
    """
    n = x.shape[0]
    return numpy.trace(numpy.matmul(centering(gaussian_grammat(x)),centering(gaussian_grammat(y))))/n/n

def jnp_HSIC(x, y):
    """
    Calculate the HSIC estimator for d=2, as in [1] eq (9)
    """
    n = x.shape[0]
    return jnp.trace(jnp.matmul(jnp_centering(jnp_gaussian_grammat(x)),jnp_centering(jnp_gaussian_grammat(y))))/n/n


def dHSIC(*argv):
    assert len(argv) > 1, "dHSIC requires at least two arguments"

    if len(argv) == 2:
        x, y = argv
        return HSIC(x, y)

    K_list = [gaussian_grammat(_arg) for _arg in argv]

    return dHSIC_calc(K_list)
