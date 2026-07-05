# Copyright 2021 Juan L Gamella

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:

# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
"""

import numpy as np
from .decomposable_score import DecomposableScore

# --------------------------------------------------------------------
# l0-penalized Gaussian log-likelihood score for a sample from a single
# (observational) environment


class GaussObsL0Pen(DecomposableScore):
    """
    Implements a cached l0-penalized gaussian likelihood score.

    """

    def __init__(self, data, lmbda=None, method='scatter', cache=True, debug=0):
        """Creates a new instance of the class.

        Parameters
        ----------
        data : numpy.ndarray
            the nxp matrix containing the observations of each
            variable (each column corresponds to a variable).
        lmbda : float or NoneType, optional
            the regularization parameter. If None, defaults to the BIC
            score, i.e. lmbda = 1/2 * log(n), where n is the number of
            observations.
        method : {'scatter', 'raw'}, optional
            the method used to compute the likelihood. If 'scatter',
            the empirical covariance matrix (i.e. scatter matrix) is
            used. If 'raw', the likelihood is computed from the raw
            data. In both cases an intercept is fitted.
        cache : bool, optional
           if computations of the local score should be cached for
           future calls. Defaults to True.
        debug : int, optional
            if larger than 0, debug are traces printed. Higher values
            correspond to increased verbosity.

        """
        if type(data) != np.ndarray:
            raise TypeError("data should be numpy.ndarray, not %s." % type(data))

        super().__init__(data, cache=cache, debug=debug)

        self.n, self.p = data.shape
        self.lmbda = 0.5 * np.log(self.n) if lmbda is None else lmbda
        self.method = method

        # Precompute scatter matrices if necessary
        if method == 'scatter':
            self._scatter = np.cov(data, rowvar=False, ddof=0)
        elif method == 'raw':
            self._centered = data - np.mean(data, axis=0)
        else:
            raise ValueError('Unrecognized method "%s"' % method)

    def full_score(self, A):
        """
        Given a DAG adjacency A, return the l0-penalized log-likelihood of
        a sample from a single environment, by finding the maximum
        likelihood estimates of the corresponding connectivity matrix
        (weights) and noise term variances.

        Parameters
        ----------
        A : np.array
            The adjacency matrix of a DAG, where A[i,j] != 0 => i -> j.

        Returns
        -------
        score : float
            the penalized log-likelihood score.

        """
        # Compute MLE
        B, omegas = self._mle_full(A)
        # Compute log-likelihood (without log(2π) term)
        K = np.diag(1 / omegas)
        I_B = np.eye(self.p) - B.T
        log_term = self.n * np.log(omegas.prod())
        if self.method == 'scatter':
            # likelihood = 0.5 * self.n * (np.log(det_K) - np.trace(K @ I_B @ self._scatter @ I_B.T))
            likelihood = log_term + self.n * np.trace(K @ I_B @ self._scatter @ I_B.T)
        else:
            # Center the data, exclude the intercept column
            inv_cov = I_B.T @ K @ I_B
            cov_term = 0
            for i, x in enumerate(self._centered):
                cov_term += x @ inv_cov @ x
            likelihood = log_term + cov_term
        #   Note: the number of parameters is the number of edges + the p marginal variances
        l0_term = self.lmbda * (np.sum(A != 0) + 1 * self.p)
        score = -0.5 * likelihood - l0_term
        return score

    # Note: self.local_score(...), with cache logic, already defined
    # in parent class DecomposableScore.

    def _compute_local_score(self, x, pa):
        """
        Given a node and its parents, return the local l0-penalized
        log-likelihood of a sample from a single environment, by finding
        the maximum likelihood estimates of the weights and noise term
        variances.

        Parameters
        ----------
        x : int
            a node.
        pa : set of ints
            the node's parents.

        Returns
        -------
        score : float
            the penalized log-likelihood score.

        """
        pa = list(pa)
        # Compute MLE
        b, sigma = self._mle_local(x, pa)
        # Compute log-likelihood (without log(2π) term)
        likelihood = -0.5 * self.n * (1 + np.log(sigma))
        #  Note: the number of parameters is the number of parents (one
        #  weight for each) + the marginal variance of x
        l0_term = self.lmbda * (len(pa) + 1)
        score = likelihood - l0_term
        return score

    # --------------------------------------------------------------------
    #  Functions for the maximum likelihood estimation of the
    #  weights/variances

    def _mle_full(self, A):
        """
        Finds the maximum likelihood estimate for the whole graph,
        specified by the adjacency A.

        Parameters
        ----------
        A : np.array
            The adjacency matrix of a DAG, where A[i,j] != 0 => i -> j.

        Returns
        -------
        B : np.array
            the connectivity (weights) matrix, which respects the
            adjacency in A.
        omegas : np.array
            the estimated noise-term variances of the observed
            variables.

        """
        B = np.zeros(A.shape)
        omegas = np.zeros(self.p)
        for j in range(self.p):
            parents = np.where(A[:, j] != 0)[0]
            B[:, j], omegas[j] = self._mle_local(j, parents)
        return B, omegas

    def _mle_local(self, j, parents):
        """Finds the maximum likelihood estimate of the local model
        between a node and its parents.

        Parameters
        ----------
        x : int
            a node.
        pa : set of ints
            the node's parents.

        Returns
        -------
        b : np.array
            an array of size p, with the estimated weights from the
            parents to the node, and zeros for non-parent variables.
        sigma : float
            the estimate noise-term variance of variable x

        """
        parents = list(parents)
        b = np.zeros(self.p)
        # Compute the regression coefficients from a least squares
        # regression on the raw data
        if self.method == 'raw':
            Y = self._centered[:, j]
            if len(parents) > 0:
                X = np.atleast_2d(self._centered[:, parents])
                # Perform regression
                coef = np.linalg.lstsq(X, Y, rcond=None)[0]
                b[parents] = coef
                sigma = np.var(Y - X @ coef)
            else:
                sigma = np.var(Y, ddof=0)
        # Or compute the regression coefficients from the
        # empirical covariance (scatter) matrix
        # i.e. b = Σ_{j,pa(j)} @ Σ_{pa(j), pa(j)}^-1
        elif self.method == 'scatter':
            sigma = self._scatter[j, j]
            if len(parents) > 0:
                cov_parents = self._scatter[parents, :][:, parents]
                cov_j = self._scatter[j, parents]
                # Use solve instead of inverting the matrix
                coef = np.linalg.solve(cov_parents, cov_j)
                sigma = sigma - cov_j @ coef
                b[parents] = coef
        # For very low sample-sizes sigma can become lower than or
        # zero -> set to machine epsilon to avoid log(0) in the
        # likelihood computation
        if sigma <= 0:
            sigma = abs(np.finfo(float).eps)
        return b, sigma
