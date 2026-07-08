import torch

# Adapted from BayesFlow (bayesflow.org)
# Copyright (c) 2022 The BayesFlow Developers

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

MMD_BANDWIDTH_LIST = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]


def gaussian_kernel_matrix(x, y, sigmas=None):
    """
    Computes a Gaussian radial basis function (RBF) kernel matrix between the samples of x and y.

    A sum of multiple Gaussian kernels is computed, each with a different width sigma.

    Parameters
    ----------
    x : torch.Tensor of shape (num_draws_x, num_features)
        Samples from the source distribution.
    y : torch.Tensor of shape (num_draws_y, num_features)
        Samples from the target distribution.
    sigmas : list of float, optional
        List of bandwidths. If None, a default range will be used.

    Returns
    -------
    kernel : torch.Tensor of shape (num_draws_x, num_draws_y)
        The computed kernel matrix.
    """
    if sigmas is None:
        sigmas = MMD_BANDWIDTH_LIST
    # Convert sigmas to a tensor on the same device as x
    sigmas = torch.tensor(sigmas, device=x.device, dtype=x.dtype)

    # Compute beta for each sigma.
    beta = 1.0 / (2.0 * sigmas)  # shape: (K,)

    # Compute the pairwise squared Euclidean distances between x and y.
    # x has shape (N, d) and y has shape (M, d).
    # The result dist will have shape (N, M).
    dist = (x.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(2)

    # Reshape beta to allow broadcasting: (K, 1, 1)
    beta = beta.view(-1, 1, 1)
    # Compute each kernel matrix: shape (K, N, M)
    kernel_vals = torch.exp(- beta * dist)
    # Sum over the different bandwidths (K dimension) to get final kernel matrix of shape (N, M)
    kernel = kernel_vals.sum(dim=0)
    return kernel


def batched_kernel_mean(a, b, kernel_fn, batch_size=None):
    """
    Computes the mean value of kernel_fn applied to pairs of batches from a and b.

    Parameters
    ----------
    a : torch.Tensor of shape (N, d)
    b : torch.Tensor of shape (M, d)
    kernel_fn : callable
        Function that computes a kernel matrix for two tensors.
    batch_size : int
        Maximum block size for batching.

    Returns
    -------
    mean_val : float
        The mean of the computed kernel values.
    """
    if batch_size is None:
        batch_size_a = a.shape[0]
        batch_size_b = b.shape[0]
    elif isinstance(batch_size, int):
        batch_size_a = batch_size_b = batch_size
    else:
        raise NotImplementedError(f"{batch_size=}")
    del batch_size
    total_sum, total_count = 0.0, 0
    for i in range(0, a.shape[0], batch_size_a):
        for j in range(0, b.shape[0], batch_size_b):
            block = kernel_fn(a[i:i + batch_size_a], b[j:j + batch_size_b])
            total_sum += block.sum()
            total_count += block.numel()
    return total_sum / total_count


def batched_mmd_kernel(x, y, kernel_fn, batch_size=None):
    """
    Computes the (biased) squared Maximum Mean Discrepancy (MMD) between x and y in a batched manner,
    using the provided kernel function.

    Parameters
    ----------
    x : torch.Tensor of shape (N, num_features)
    y : torch.Tensor of shape (M, num_features)
    kernel_fn : callable
        Function that computes the kernel matrix between two tensors.
    batch_size : int
        Maximum block size for batching.

    Returns
    -------
    loss : torch.Tensor
        The (biased) squared MMD value.
    """
    mean_xx = batched_kernel_mean(x, x, kernel_fn, batch_size)
    mean_yy = batched_kernel_mean(y, y, kernel_fn, batch_size)
    mean_xy = batched_kernel_mean(x, y, kernel_fn, batch_size)
    return mean_xx + mean_yy - 2 * mean_xy


def maximum_mean_discrepancy(source_samples, target_samples, kernel="gaussian", batch_size=None):
    """Computes the MMD given a particular choice of kernel.

    For details, consult Gretton et al. (2012):
    https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf

    Parameters
    ----------
    source_samples : tf.Tensor of shape (N, num_features)
        An array of `N` random draws from the "source" distribution.
    target_samples : tf.Tensor of shape  (M, num_features)
        An array of `M` random draws from the "target" distribution.
    kernel         : str in ('gaussian',), optional, default: 'gaussian'
        The kernel to use for computing the distance between pairs of random draws.
    batch_size     : int, optional, default: None
        The maximum block size for batching.

    Returns
    -------
    loss_value : tf.Tensor
        A scalar Maximum Mean Discrepancy, shape (,)
    """

    # Determine kernel, fall back to Gaussian if unknown string passed
    if kernel == "gaussian":
        kernel_fun = gaussian_kernel_matrix
    else:
        raise ValueError("kernel must be 'gaussian'")

    # Compute and return MMD value
    return batched_mmd_kernel(source_samples, target_samples, kernel_fn=kernel_fun, batch_size=batch_size)
