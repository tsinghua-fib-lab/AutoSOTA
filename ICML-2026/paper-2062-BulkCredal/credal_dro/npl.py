"""NPL posterior"""

from typing import Optional
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import dirichlet
import scipy.spatial.distance as distance
import itertools
from tqdm import tqdm
import jax
from jax import numpy as jnp
from jax import vmap, value_and_grad, jit, config
from jax.example_libraries import optimizers
from .gaussian_kernel import k, k_jax, k_comp
from .models import *
from .constants import upper_triangular_size, NPL_ETA


def sample_npl(
    data: np.ndarray,
    inference: str,
    likelihood: str,
    num_posterior_samples: int,
    seed: int,
    lengthscale: float = -1.0,
    dim: int = 1,
    generator: Optional[np.random.Generator] = None,
    p: int = 1,
    kernel_name: str = "k_jax",
    eta: float = NPL_ETA,
) -> np.ndarray:
    """Wrapper function for sampling from the NPL posterior with either WLL or MMD loss function

    Args:
        data: Observations sampled from DGP
        inference: Either 'npl_wll' or 'npl_mmd'
        likelihood: Form of likelihood, e.g. 'exponential'
        num_posterior_samples: Number of times to sample from posterior
        generator: numpy random generator
        p: numbers of unknown parameters

    Returns:
        Array of size `num_posterior_samples`
    """
    # NPL posterior sample for theta
    m = data.shape[0]
    if likelihood == "exponential":
        model = ExponentialModel(m)
        p = 1
        dim = 1
    elif likelihood == "normal":
        model = univariate_GaussianModel(m)
        p = 2
        dim = 1
    elif likelihood == "normal_known_var":
        model = univariate_GaussianModel_known_variance(m)
        p = 1
    elif likelihood == "multivariate_normal_known_cov":
        d = data.shape[1]
        model = multivariate_GaussianModel(m, d, known_cov=True)
        p = d
    elif likelihood == "multivariate_normal":
        d = data.shape[1]
        model = multivariate_GaussianModel(m, d, known_cov=False)
        p = d + upper_triangular_size(d)
    else:
        raise NotImplementedError(
            f"Posterior '{likelihood}' is not implemented for '{inference}' inference."
        )
    npl_toy = Npl(
        data.reshape((data.shape[0], dim)),
        num_posterior_samples,
        p,
        m,
        model,
        seed,
        l=lengthscale,
        loss_fn=inference,
        kernel_name=kernel_name,
        eta=eta,
    )
    npl_toy.draw_samples(random_state=generator)
    theta_sample = npl_toy.sample
    return theta_sample


class Npl:
    """This class contains functions to perform NPL inference (for alpha = 0 in the DP prior) for the Exponential distribution model."""

    def __init__(self, X, B, p, m, model, seed, l=-1, loss_fn="npl_wlb", eta: float = NPL_ETA, kernel_name: str = "k_jax"):
        """
        Args:
            X: Data set
            B: number of bootstrap iterations
            p: number of unknown parameters
            m: number of points sampled from the model at each approximation of the MMD (compatible with value of m within model class)
            l: lengthscale of gaussian kernel; set l = -1 to use median heuristic
            model: model class from models.py
            loss_fn : string set to 'wll' or 'mmd' to specify either the negative log-lkh or mmd-based loss function
        """
        self.B = B
        self.X = X
        self.p = p
        self.loss_fn = loss_fn
        self.n, self.d = self.X.shape
        self.m = m
        self.l = l
        if self.l == -1:  # median heuristic
            self.l = np.sqrt(
                (1 / 2) * np.median(distance.cdist(self.X, self.X, "sqeuclidean"))
            )
        self.kernel_name = kernel_name
        self.kxx = k(
            self.X, self.X, self.l
        )  # pre calculate kernel matrix of data k(x,x)
        self.model = model
        self.seed = seed
        self.sample = None
        self.eta = jnp.float64(eta) # NOTE this cast seems to fix a bug, but does create a warning - dunno why?!

    def draw_single_mmd_sample(self, weights, key):
        """Draws a single sample from the nonparametric posterior specified via
        data X and Dirichlet weights"""
        # FIXME pass eta as a parameter via the experiment setup
        # return self.minimise_MMD(self.X, weights, key)
        return self.minimise_MMD(self.X, weights, key, eta=self.eta)

    def draw_samples(self, n_jobs: int = -1, random_state=None):
        """Draws B samples in parallel from the nonparametric posterior"""

        weights = dirichlet.rvs(np.ones(self.n), size=self.B, random_state=random_state)
        samples = np.zeros((self.B, self.p))

        if self.loss_fn == "npl_wlb":
            # FIXME n_jobs > 1
            temp = Parallel(
                n_jobs=n_jobs,
                backend="multiprocessing",
                max_nbytes=None,
                batch_size="auto",
            )(delayed(self.WLL)(self.X, weights[i, :]) for i in range(self.B))

            for i in range(self.B):
                samples[i, :] = temp[i]
                self.sample = np.array(samples)
        elif self.loss_fn == "npl_mmd":
            key = jax.random.PRNGKey(self.seed)
            key, *subkeys = jax.random.split(key, num=self.B+1) # generate B random keys

            mmd_samples = vmap(self.draw_single_mmd_sample, in_axes=0)(
                weights, jnp.array(subkeys)
            )
            self.sample = np.array(mmd_samples)
        else:
            raise NotImplementedError("NPL loss function not recognised: " + self.loss_fn)

    def WLL(self, data, weights):
        """Get weighted negative log likelihood minimizer, for Exponential distribution model"""

        theta = np.zeros(self.d)
        for i in range(self.n):
            theta += weights[i] * data[i, :]
        return 1 / theta

    def MMD_approx(self, kxy, kyy):
        """Approximation of the squared MMD given Gram matrices kxy and kyy"""

        # first sum
        diag_elements = jnp.diag_indices_from(kyy)
        kyy = kyy.at[diag_elements].set(jnp.repeat(0, self.m))
        sum1 = jnp.sum(kyy)

        # second sum
        sum2 = jnp.sum(kxy)

        # third sum
        diag_elements = jnp.diag_indices_from(self.kxx)
        kxx = self.kxx.at[diag_elements].set(jnp.repeat(0, self.n))
        sum3 = jnp.sum(kxx)

        return (
            (1 / (self.m * (self.m - 1))) * sum1
            - (2 / (self.n * self.m)) * sum2
            + (1 / (self.n * (self.n - 1))) * sum3
        )

    def minimise_MMD(self, data, weights, key, Nstep=1000, eta=0.1, batch_size=10):
        """Function to minimise the MMD using adam optimisation in JAX"""

        key, key1, key2 = jax.random.split(key, num=2 + 1)
        params = self.model.init_params(data)
        config.update("jax_enable_x64", True)
        num_batches = self.n // batch_size

        # objective function to feed the optimizer
        def obj_fun(theta, x, n, key):
            y = self.model.sample(
                theta, key
            )  # Returnes self.m random samples from the model with parameter theta

            if self.kernel_name == "k_comp":
                # Compute kernel Gram matrices
                kyy = k_comp(y, y) #, self.l
                kxy = k_comp(y, x) #, self.l
            elif self.kernel_name == "k_jax":
                kyy = k_jax(y, y, self.l)
                kxy = k_jax(y, x, self.l)
            else:
                raise ValueError(f"Kernel name {self.kernel_name} not recognised")

            # first sum
            diag_elements = jnp.diag_indices_from(kyy)
            kyy = kyy.at[diag_elements].set(jnp.repeat(0, self.m))
            sum1 = jnp.sum(kyy)

            # second sum
            sum2 = jnp.sum(kxy)

            # Return first two terms of squared MMD; note that the third term does not depend on theta!
            return (1 / (self.m * (self.m - 1))) * sum1 - (2 / (n * self.m)) * sum2

        opt_init, opt_update, get_params = optimizers.adam(step_size=eta)
        opt_state = opt_init(params)
        itercount = itertools.count()

        # Define gradient function
        grad_fn = vmap(
            jit(value_and_grad(obj_fun, argnums=0)), in_axes=(None, 0, None, None)
        )

        # Function to evaluate gradient and loss value at each step
        def step(step, opt_state, batches, key):
            key, subkey = jax.random.split(key)
            values, grads = grad_fn(get_params(opt_state), batches, batch_size, subkey)
            opt_state = opt_update(step, np.mean(grads, axis=0), opt_state)
            value = np.mean(values, axis=0)
            return value, opt_state

        smallest_loss = 1000000
        best_theta = get_params(opt_state)
        key1, *rng_inputs1 = jax.random.split(key1, num=Nstep + 1)
        key2, *rng_inputs2 = jax.random.split(key2, num=Nstep + 1)
        for i in range(Nstep):
            batches = []
            _, *rng_inputs = jax.random.split(rng_inputs2[i], num=num_batches + 1)
            for j in range(num_batches):
              inds = jax.random.choice(rng_inputs[j], a=self.n, shape=(batch_size,), p=weights) #default is with replacement
              batch_x = jnp.take(a=data, indices=inds, axis=0)
              batches.append(batch_x)

            batches = jnp.array(batches)
            # Update loss and gradient
            value, opt_state = step(next(itercount), opt_state, batches, rng_inputs1[i])
            # print(get_params(opt_state))
            # Update smallest loss and best theta value if loss has decreased
            pred = value < smallest_loss  # Prediction that loss (value) has decreased

            def true_func(args):
                value, smallest_loss, best_theta, opt_state = (
                    args[0],
                    args[1],
                    args[2],
                    args[3],
                )
                smallest_loss = value
                best_theta = get_params(opt_state)
                return smallest_loss, best_theta

            def false_func(args):
                value, smallest_loss, best_theta, opt_state = (
                    args[0],
                    args[1],
                    args[2],
                    args[3],
                )
                smallest_loss = jnp.array(smallest_loss, dtype="float64")
                return smallest_loss, best_theta

            # Updates smallest loss and best theta if prediction (pred) is correct
            smallest_loss, best_theta = jax.lax.cond(
                pred,
                true_func,
                false_func,
                [value, smallest_loss, best_theta, opt_state],
            )
            
            
       
        best_theta = self.model.parametrise(best_theta)
        return best_theta 
