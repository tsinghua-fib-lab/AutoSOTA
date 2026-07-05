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
Test vs. the output of the R implementation of GES.
"""

import unittest
import numpy as np
import sempler
import sempler.generators
import time

import ges
import ges.scores.gauss_obs_l0_pen
import ges.utils

# For CDT
from cdt.causality.graph import GES
import networkx as nx
import pandas as pd

# ---------------------------------------------------------------------

NUM_GRAPHS = 500


class OverallGESTests(unittest.TestCase):
    true_A = np.array([[0, 0, 1, 0, 0],
                       [0, 0, 1, 0, 0],
                       [0, 0, 0, 1, 1],
                       [0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0]])
    factorization = [(4, (2, 3)), (3, (2,)), (2, (0, 1)), (0, ()), (1, ())]
    true_B = true_A * np.random.uniform(1, 2, size=true_A.shape)
    scm = sempler.LGANM(true_B, (0, 0), (0.3, 0.4))
    p = len(true_A)
    n = 100000
    interventions = [{0: (0, 1.0)},
                     {1: (0, 1.1)},
                     {2: (0, 1.2)},
                     {3: (0, 1.3)},
                     {4: (0, 1.4)}]
    obs_data = scm.sample(n=n)
    int_data = [obs_data]
    # Sample interventional distributions and construct true interventional
    # variances for later reference in tests
    interventional_variances = np.tile(scm.variances, (len(interventions) + 1, 1))
    for i, intervention in enumerate(interventions):
        int_data.append(scm.sample(n=n, shift_interventions=intervention))
        for (target, params) in intervention.items():
            interventional_variances[i + 1, target] += params[1]

    # ------------------------------------------------------
    # Tests

    def test_vs_cdt_1(self):
        # Test that behaviour matches that of the implementation in
        # the R package pcalg, using NUM_GRAPHS randomly generated
        # Erdos-Renyi graphs. The call is made through the ges.fit_bic
        # function
        np.random.seed(15)
        G = NUM_GRAPHS  # number of graphs
        p = 15  # number of variables
        n = 1500  # size of the observational sample
        for i in range(G):
            print("  Checking SCM %d" % (i))
            start = time.time()
            A = sempler.generators.dag_avg_deg(p, 3, 1, 1)
            W = A * np.random.uniform(1, 2, A.shape)
            obs_sample = sempler.LGANM(W, (1, 10), (0.5, 1)).sample(n=n)
            # Estimate the equivalence class using the pcalg
            # implementation of GES (package cdt)
            data = pd.DataFrame(obs_sample)
            output = GES(verbose=True).predict(data)
            estimate_cdt = nx.to_numpy_array(output)
            end = time.time()
            print("    GES-CDT done (%0.2f seconds)" % (end - start))
            start = time.time()
            # Estimate using this implementation
            # Test debugging output for the first 2 SCMs
            estimate, _ = ges.fit_bic(obs_sample, iterate=True, debug=4 if i < 2 else 2)
            end = time.time()
            print("    GES-own done (%0.2f seconds)" % (end - start))
            self.assertTrue((estimate == estimate_cdt).all())
        print("\nCompared with PCALG implementation on %d DAGs" % (i + 1))

    def test_vs_cdt_2(self):
        # Test that behaviour matches that of the implementation in
        # the R package pcalg, using NUM_GRAPHS randomly generated
        # Erdos-Renyi graphs. The call is made through the ges.fit
        # function; for half of the cases, manually specify the
        # completion algorithm.
        np.random.seed(16)
        G = NUM_GRAPHS  # number of graphs
        p = 15  # number of variables
        n = 1500  # size of the observational sample
        for i in range(G):
            print("  Checking SCM %d" % (i))
            start = time.time()
            A = sempler.generators.dag_avg_deg(p, 3, 1, 1)
            W = A * np.random.uniform(1, 2, A.shape)
            obs_sample = sempler.LGANM(W, (1, 10), (0.5, 1)).sample(n=n)
            # Estimate the equivalence class using the pcalg
            # implementation of GES (package cdt)
            data = pd.DataFrame(obs_sample)
            score_class = ges.scores.gauss_obs_l0_pen.GaussObsL0Pen(obs_sample)
            completion_algorithm = None if i % 2 == 0 else ges.utils.pdag_to_cpdag
            output = GES(verbose=True).predict(data)
            estimate_cdt = nx.to_numpy_array(output)
            end = time.time()
            print("    GES-CDT done (%0.2f seconds)" % (end - start))
            start = time.time()
            # Estimate using this implementation
            # Test debugging output for the first 2 SCMs
            estimate, _ = ges.fit(
                score_class, completion_algorithm=completion_algorithm, iterate=True, debug=4 if i < 2 else 2)
            end = time.time()
            print("    GES-own done (%0.2f seconds)" % (end - start))
            self.assertTrue((estimate == estimate_cdt).all())
        print("\nCompared with PCALG implementation on %d DAGs" % (i + 1))

    def test_vs_cdt_2_raw(self):
        # Test that behaviour matches that of the implementation in
        # the R package pcalg, using NUM_GRAPHS randomly generated
        # Erdos-Renyi graphs. The call is made through the ges.fit
        # function
        np.random.seed(17)
        G = NUM_GRAPHS  # number of graphs
        p = 15  # number of variables
        n = 1500  # size of the observational sample
        for i in range(G):
            print("  Checking SCM %d" % (i))
            start = time.time()
            A = sempler.generators.dag_avg_deg(p, 3, 1, 1)
            W = A * np.random.uniform(1, 2, A.shape)
            obs_sample = sempler.LGANM(W, (1, 10), (0.5, 1)).sample(n=n)
            # Estimate the equivalence class using the pcalg
            # implementation of GES (package cdt)
            data = pd.DataFrame(obs_sample)
            score_class = ges.scores.gauss_obs_l0_pen.GaussObsL0Pen(obs_sample, method='raw')
            output = GES(verbose=True).predict(data)
            estimate_cdt = nx.to_numpy_array(output)
            end = time.time()
            print("    GES-CDT done (%0.2f seconds)" % (end - start))
            start = time.time()
            # Estimate using this implementation
            # Test debugging output for the first 2 SCMs
            estimate, _ = ges.fit(score_class, iterate=True, debug=4 if i < 2 else 2)
            end = time.time()
            print("    GES-own done (%0.2f seconds)" % (end - start))
            self.assertTrue((estimate == estimate_cdt).all())
        print("\nCompared with PCALG implementation on %d DAGs" % (i + 1))
