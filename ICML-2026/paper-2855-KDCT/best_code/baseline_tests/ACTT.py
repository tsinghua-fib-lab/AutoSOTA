import numpy as np
from goodpoints.ctt import actt
import jax.numpy as jnp
from jax import vmap, jit
from functools import partial
import sys
import os
sys.path.append(os.path.abspath('..'))
import time
from dataloader import load_data
from sklearn.decomposition import PCA

def PCA_transform(X, n_components=4):
    pc = PCA(n_components)
    pc.fit(X)
    X = pc.transform(X)
    return X

class HiddenPrints:
    """
    Hide prints and warnings.
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        self._original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        sys.stderr.close()
        sys.stderr = self._original_stderr

def jax_distances(X, Y, l, max_samples=None, matrix=False):
    np.random.seed(seed=1102)

    if l == "l1":

        def dist(x, y):
            z = x - y
            return jnp.sum(jnp.abs(z))

    elif l == "l2":

        def dist(x, y):
            z = x - y
            return jnp.sqrt(jnp.sum(jnp.square(z)))

    else:
        raise ValueError("Value of 'l' must be either 'l1' or 'l2'.")
    vmapped_dist = vmap(dist, in_axes=(0, None))
    pairwise_dist = vmap(vmapped_dist, in_axes=(None, 0))
    output = pairwise_dist(X[:max_samples], Y[:max_samples])
    if matrix:
        return output
    else:
        return output[jnp.triu_indices(output.shape[0])]

@partial(jit, static_argnums=(2, 3, 4))
def compute_bandwidths(X, Y, l, number_bandwidths, only_median=False):
    np.random.seed(seed=1102)

    Z = jnp.concatenate((X, Y))
    distances = jax_distances(Z, Z, l, matrix=False)
    median = jnp.median(distances)
    if only_median:
        return median
    distances = distances + (distances == 0) * median
    dd = jnp.sort(distances)
    lambda_min = dd[(jnp.floor(len(dd) * 0.05).astype(int))] / 2
    lambda_max = dd[(jnp.floor(len(dd) * 0.95).astype(int))] * 2
    bandwidths = jnp.linspace(lambda_min, lambda_max, number_bandwidths)
    return bandwidths

def ACTT(X, Y, seed):
    np.random.seed(seed=1102)
    lam = compute_bandwidths(X, Y, "l1", 10, only_median=False)
    weights = np.array([1/len(lam),] * len(lam)).astype('double')
    X = np.asarray(X).astype('double')
    Y = np.asarray(Y).astype('double')
    lam = np.asarray(lam).astype('double')
    with HiddenPrints():
        output = actt(X, Y, g=4, lam=lam, weights=weights, null_seed=seed, statistic_seed=seed).rejects
    return output

def TST_ACTT(name, N1, rs, check, n_test, alpha=0.05):
    assert alpha == 0.05
    np.random.seed(rs)
    X_train, Y_train = load_data(name, N1, rs, check)

    H_ACTT = np.zeros(n_test)
    N_test_all = 10 * N1
    X_test_all, Y_test_all = load_data(name, N_test_all, rs + 283, check)

    X = np.concatenate((X_train,X_test_all))
    X = PCA_transform(X, n_components=4)
    X_train = X[:len(X_train)]
    X_test_all = X[len(X_train):]
    Y = np.concatenate((Y_train,Y_test_all))
    Y = PCA_transform(Y, n_components=4)
    Y_train = Y[:len(Y_train)]
    Y_test_all = Y[len(Y_train):]

    test_time = 0
    # test by MMDAgg
    for k in range(n_test):
        ind_test = np.random.choice(N_test_all, N1, replace=False)
        X_test = X_test_all[ind_test]
        Y_test = Y_test_all[ind_test]

        S_x = np.concatenate((X_train, X_test), axis=0)
        S_y = np.concatenate((Y_train, Y_test), axis=0)

        start_time = time.time()
        h_ACTT = ACTT(S_x, S_y, seed=rs)
        test_time += time.time() - start_time
        H_ACTT[k] = h_ACTT

    return H_ACTT, 0, test_time