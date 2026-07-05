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

import unittest
import numpy as np
import sempler

from ges.scores.gauss_obs_l0_pen import GaussObsL0Pen

# ---------------------------------------------------------------------
# Tests for the l0-penalized scores


class ScoreTests(unittest.TestCase):
    np.random.seed(12)
    true_A = np.array([[0, 0, 1, 0, 0],
                       [0, 0, 1, 0, 0],
                       [0, 0, 0, 1, 1],
                       [0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0]])
    factorization = [(4, (2, 3)), (3, (2,)), (2, (0, 1)), (0, ()), (1, ())]
    true_B = true_A * np.random.uniform(1, 2, size=true_A.shape)
    scm = sempler.LGANM(true_B, (0, 0), (0.3, 0.4))
    p = len(true_A)
    n = 10000
    obs_data = scm.sample(n=n)
    obs_score = GaussObsL0Pen(obs_data)
    obs_score_raw = GaussObsL0Pen(obs_data, method='raw')

    # ------------------------------------------------------
    # White-box tests:
    #   testing the inner workings of the ges.scores module, e.g. the
    #   intermediate functions used to compute the likelihoods

    def test_mle_obs(self):
        # Check that the parameters are correctly estimated when
        # passing a subgraph to GaussObsL0Pen._mle_full
        for score in [self.obs_score, self.obs_score_raw]:
            print("Testing %s" % score)
            local_B = np.zeros_like(self.true_B)
            local_omegas = np.zeros(self.p)
            for (x, pa) in self.factorization:
                local_B[:, x], local_omegas[x] = score._mle_local(x, pa)
            full_B, full_omegas = score._mle_full(self.true_A)
            print("Locally estimated", local_B, local_omegas)
            print("Fully estimated", full_B, full_omegas)
            print("Truth", self.true_B, self.scm.variances)
            # Make sure zeros are respected
            self.assertTrue((local_B[self.true_A == 0] == 0).all())
            self.assertTrue((full_B[self.true_A == 0] == 0).all())
            # Make sure estimation of weights is similar
            self.assertTrue((local_B == full_B).all())
            # Make sure estimation of noise variances is similar
            self.assertTrue((local_omegas == full_omegas).all())
            # Compare with true model
            self.assertTrue(np.allclose(self.true_B, local_B, atol=5e-2))
            self.assertTrue(np.allclose(self.true_B, full_B, atol=5e-2))
            self.assertTrue(np.allclose(self.scm.variances, local_omegas, atol=1e-1))
            self.assertTrue(np.allclose(self.scm.variances, full_omegas, atol=1e-1))

    # ------------------------------------------------------
    # Black-box tests:
    #   Testing the behaviour of the "API" functions, i.e. the
    #   functions to compute the full/local
    #   observational/interventional BIC scores from a given DAG
    #   structure and the data

    def test_parameters_obs(self):
        # Fails if data is not ndarray
        try:
            GaussObsL0Pen([self.obs_data])
            self.fail()
        except TypeError:
            pass
        except Exception:
            self.fail()

    def test_full_score_obs(self):
        # Verify that the true adjacency yields a higher score than
        # the empty graph Compute score of true adjacency
        for score_fun in [self.obs_score, self.obs_score_raw]:
            print("Testing %s" % score_fun)
            true_score = score_fun.full_score(self.true_A)
            self.assertIsInstance(true_score, float)
            # Compute score of unconnected graph
            score = score_fun.full_score(np.zeros((self.p, self.p)))
            self.assertIsInstance(score, float)
            self.assertGreater(true_score, score)

    def test_score_decomposability_obs(self):
        # As a black-box test, make sure the score functions
        # preserve decomposability
        for score_fun in [self.obs_score, self.obs_score_raw]:
            print("Decomposability of observational score")
            print("Testing %s" % score_fun)
            full_score = score_fun.full_score(self.true_A)
            acc = 0
            for (j, pa) in self.factorization:
                local_score = score_fun.local_score(j, pa)
                print("  ", j, pa, local_score)
                acc += local_score
            print("Full vs. acc:", full_score, acc)
            self.assertAlmostEqual(full_score, acc, places=2)
