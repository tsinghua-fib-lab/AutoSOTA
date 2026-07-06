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

"""Module containing the DecomposableScore class, inherited by all
classes which implement a locally decomposable score for directed
acyclic graphs. By default, the class also caches the results of
computing local scores.

NOTE: It is not mandatory to inherit this class when developing custom
scores to use with the GES implementation in ges.py. The only
requirement is that the class defines:
  1. the local_score function (see below),
  2. an attribute "p" for the total number of variables.

"""

import copy

# --------------------------------------------------------------------
# l0-penalized Gaussian log-likelihood score for a sample from a single
# (observational) environment


class DecomposableScore():

    def __init__(self, data, cache=True, debug=0):
        self._data = copy.deepcopy(data)
        self._cache = {} if cache else None
        self._debug = debug
        self.p = None

    def local_score(self, x, pa):
        """
        Return the local score of a given node and a set of
        parents. If self.cache=True, will use previously computed
        score if possible.

        Parameters
        ----------
        x : int
            a node
        pa : set of ints
            the node's parents

        Returns
        -------
        score : float
            the corresponding score

        """
        if self._cache is None:
            return self._compute_local_score(x, pa)
        else:
            key = (x, tuple(sorted(pa)))
            try:
                score = self._cache[key]
                print("score%s: using cached value %0.2f" %
                      (key, score)) if self._debug >= 2 else None
            except KeyError:
                score = self._compute_local_score(x, pa)
                self._cache[key] = score
                print("score%s = %0.2f" % (key, score)) if self._debug >= 2 else None
            return score

    def _compute_local_score(self, x, pa):
        """
        Compute the local score of a given node and a set of
        parents.

        Parameters
        ----------
        x : int
            a node
        pa : set of ints
            the node's parents

        Returns
        -------
        score : float
            the corresponding score

        """
        return 0
