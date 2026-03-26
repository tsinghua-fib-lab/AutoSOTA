# This method builds upon the kernel-synth method introduced by the authors of Chronos.
# We adapted the synthetic data generation process to accommodate multivariate cases using the Linear Coregionalization Model.
# This approach utilizes kernel-synth to generate latent variates. The original license is as follows:
# 
# "Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: Apache-2.0"
# 
# The kernel-synth method can be found at: https://github.com/amazon-science/chronos-forecasting

import argparse
import functools
from pathlib import Path
from typing import Optional

import numpy as np
from gluonts.dataset.arrow import ArrowWriter
from joblib import Parallel, delayed
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    ExpSineSquared,
    Kernel,
    RationalQuadratic,
    WhiteKernel,
)
from tqdm.auto import tqdm

LENGTH = 1024
KERNEL_BANK = [
    ExpSineSquared(periodicity=24 / LENGTH),  # H
    ExpSineSquared(periodicity=48 / LENGTH),  # 0.5H
    ExpSineSquared(periodicity=96 / LENGTH),  # 0.25H
    ExpSineSquared(periodicity=24 * 7 / LENGTH),  # H
    ExpSineSquared(periodicity=48 * 7 / LENGTH),  # 0.5H
    ExpSineSquared(periodicity=96 * 7 / LENGTH),  # 0.25H
    ExpSineSquared(periodicity=7 / LENGTH),  # D
    ExpSineSquared(periodicity=14 / LENGTH),  # 0.5D
    ExpSineSquared(periodicity=30 / LENGTH),  # D
    ExpSineSquared(periodicity=60 / LENGTH),  # 0.5D
    ExpSineSquared(periodicity=365 / LENGTH),  # D
    ExpSineSquared(periodicity=365 * 2 / LENGTH),  # 0.5D
    ExpSineSquared(periodicity=4 / LENGTH),  # W
    ExpSineSquared(periodicity=26 / LENGTH),  # W
    ExpSineSquared(periodicity=52 / LENGTH),  # W
    ExpSineSquared(periodicity=4 / LENGTH),  # M
    ExpSineSquared(periodicity=6 / LENGTH),  # M
    ExpSineSquared(periodicity=12 / LENGTH),  # M
    ExpSineSquared(periodicity=4 / LENGTH),  # Q
    ExpSineSquared(periodicity=4 * 10 / LENGTH),  # Q
    ExpSineSquared(periodicity=10 / LENGTH),  # Y
    DotProduct(sigma_0=0.0),
    DotProduct(sigma_0=1.0),
    DotProduct(sigma_0=10.0),
    RBF(length_scale=0.1),
    RBF(length_scale=1.0),
    RBF(length_scale=10.0),
    RationalQuadratic(alpha=0.1),
    RationalQuadratic(alpha=1.0),
    RationalQuadratic(alpha=10.0),
    WhiteKernel(noise_level=0.1),
    WhiteKernel(noise_level=1.0),
    ConstantKernel(),
]


def random_binary_map(a: Kernel, b: Kernel):
    """
    Applies a random binary operator (+ or *) with equal probability
    on kernels ``a`` and ``b``.

    Parameters
    ----------
    a
        A GP kernel.
    b
        A GP kernel.

    Returns
    -------
        The composite kernel `a + b` or `a * b`.
    """
    binary_maps = [lambda x, y: x + y, lambda x, y: x * y]
    return np.random.choice(binary_maps)(a, b)


def sample_from_gp_prior(
    kernel: Kernel, X: np.ndarray, random_seed: Optional[int] = None
):
    """
    Draw a sample from a GP prior.

    Parameters
    ----------
    kernel
        The GP covaraince kernel.
    X
        The input "time" points.
    random_seed, optional
        The random seed for sampling, by default None.

    Returns
    -------
        A time series sampled from the GP prior.
    """
    if X.ndim == 1:
        X = X[:, None]

    assert X.ndim == 2
    gpr = GaussianProcessRegressor(kernel=kernel)
    ts = gpr.sample_y(X, n_samples=1, random_state=random_seed)

    return ts


def sample_from_gp_prior_efficient(
    kernel: Kernel,
    X: np.ndarray,
    random_seed: Optional[int] = None,
    method: str = "eigh",
):
    """
    Draw a sample from a GP prior. An efficient version that allows specification
    of the sampling method. The default sampling method used in GaussianProcessRegressor
    is based on SVD which is significantly slower that alternatives such as `eigh` and
    `cholesky`.

    Parameters
    ----------
    kernel
        The GP covaraince kernel.
    X
        The input "time" points.
    random_seed, optional
        The random seed for sampling, by default None.
    method, optional
        The sampling method for multivariate_normal, by default `eigh`.

    Returns
    -------
        A time series sampled from the GP prior.
    """
    if X.ndim == 1:
        X = X[:, None]

    assert X.ndim == 2

    cov = kernel(X)
    ts = np.random.default_rng(seed=random_seed).multivariate_normal(
        mean=np.zeros(X.shape[0]), cov=cov, method=method
    )

    return ts


def generate_time_series(dirichlet_min, dirichlet_max, weibull_shape, weibul_scale, max_kernels: int = 5, num_channels: int = 10):
    """Generate a synthetic time series from LMC_Synth.
    """
    while True:
        X = np.linspace(0, 1, LENGTH)

        #sample from weibull the number of latent functions
        latent_num = np.rint(np.random.weibull(weibull_shape, 1)*weibul_scale + 1)
        latent_num = np.clip(latent_num, max(2, num_channels//20), num_channels)
        latent_num = int(latent_num[0])
        
        #now we will have latent num of latent functions. Each latent function will have a random number of kernels
        kernel_numbers = np.random.randint(1, max_kernels + 1, size = latent_num)
        #now that  we have the number of kernels for each latent function
        #we will sample the kernels for each latent function
        latent_kernels = [functools.reduce(random_binary_map, 
                                           np.random.choice(KERNEL_BANK, 
                                                            num_kernels, replace=True)) 
                                                            for num_kernels in kernel_numbers]
        
        try:
            #now we have the kernels for each latent function
            #we will now sample the latent functions
            latent_functions = np.array([sample_from_gp_prior_efficient(kernel=kernel, X=X) for kernel in latent_kernels])
            #sample dirichlet parameter between dirichlet_min and dirichlet_max uniformly
            dirichlet = np.random.uniform(dirichlet_min, dirichlet_max)
            #now sample the weights for each latent function
            weights = np.random.dirichlet(dirichlet*np.ones(latent_num), size=num_channels)
            #now we will combine the latent functions with the weights
            ts = np.dot(weights, latent_functions)
        except np.linalg.LinAlgError as err:
            print("Error caught:", err)
            continue

        return {"start": np.datetime64("2000-01-01 00:00", "s"), "target": ts}



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", "--num-series", type=int, default=15000)
    parser.add_argument("-J", "--max-kernels", type=int, default=5)
    parser.add_argument("-C", "--num-channels", type=int, default=160)
    parser.add_argument("-O", "--output", type=str, default="LMC_synth_MTS.arrow")
    parser.add_argument("-D", "--directory", type=str, default="./")
    parser.add_argument("-L", "--length", type=int, default=1024)
    parser.add_argument("-M", "--dirichlet_min", type=float)
    parser.add_argument("-X", "--dirichlet_max", type=float)
    parser.add_argument("-W", "--weibull_shape", type=float)
    parser.add_argument("-Z", "--weibul_scale", type=float)

    args = parser.parse_args()
    path = args.directory + args.output

    LENGTH = args.length

    generated_dataset = Parallel(n_jobs=-1)(
        delayed(generate_time_series)(max_kernels=args.max_kernels, num_channels=args.num_channels, dirichlet_min = args.dirichlet_min, dirichlet_max = args.dirichlet_max, weibull_shape = args.weibull_shape, weibul_scale = args.weibul_scale)
        for _ in tqdm(range(args.num_series))
    )

    ArrowWriter(compression="lz4").write_to_file(
        generated_dataset,
        path=path,
    )
