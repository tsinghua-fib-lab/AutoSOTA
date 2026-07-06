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

# modified by Federico Baldo for I-PERI

"""
"""
from typing import List

import numpy as np

from pyciphod.causal_discovery.federated.regret_based.ges.scores.decomposable_score import DecomposableScore    
from pyciphod.causal_discovery.federated.regret_based.iperi.client import Client

# --------------------------------------------------------------------
# l0-penalized Gaussian log-likelihood score for a sample from a single
# (observational) environment

# Modified by Federico Baldo to include the I-PERI Score

class IPeriScore(DecomposableScore):
    """
    Implements a cached l0-penalized gaussian likelihood score.
    """
    def __init__(self, data: np.ndarray, clients: List[Client], cache=False, debug=0, undirected: bool = False):
        """Creates a new instance of the class.

        Parameters
        ----------
        data : numpy.ndarray
            the np matrix containing the observations of each
            variable (each column corresponds to a variable).
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
        # in this context data are not available, but rather just the number of variables
        _, self.p = data.shape
        self.clients = clients  
        self.undirected = undirected

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
        # code modified by Federico Baldo 
        score = 0
        for x in range(self.p):
            parents = np.nonzero(A[:, x])[0].tolist()
            parents.sort()
            score += self._compute_local_score(x, parents)
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
        # code written by Osman Min, modified by Federico Baldo
        pa = list(pa)
        pa.sort()
	
		#since our implementation of GES is meant to maximize the score, 
		#we have to re-cast min max(regret) as max min(-regret) , 
		#negation of regret part should be handled within the env[cc].score() 
		#here we only take the min
        regrets = [client.score(pa, x, undirected=self.undirected) for client in self.clients]
        score = min(regrets) #if self.undirected else max(regrets)
        return score
    